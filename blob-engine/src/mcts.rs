//! Arena-allocated MCTS over sampled deals (gen-2.md §5.4).
//!
//! Nodes are stored contiguously in `MctsArena::nodes`. Children are indices
//! into that vec, kept inline with `SmallVec<[u32; 14]>` (worst-case fan-out
//! is 14 bids; up to 13 plays).
//!
//! **Every visit carries a value for every seat.** A leaf is scored for all
//! seats at once and turned into per-round utilities `u_s` (`scoring.rs`):
//! - at the end of the round, from the exact round scores;
//! - otherwise from the value net's expected ŝ on the sampled deal.
//!
//! Backup adds `u_s` to seat `s`'s sum at every node on the path, and UCB
//! reads the acting seat's mean, so a node's mean for any seat is over all
//! of its visits. Gen 1 credited a network leaf only to the seat about to
//! move there, and the deciding seat rarely heard about its own options
//! (gen-2.md §2.3).
//!
//! **Leaves** cost one policy call (priors for the seat to move, from its own
//! view) and one value call (every seat, on the sampled deal), each batched
//! across the sampled deals.
//!
//! **Budgets** are per phase (`MctsConfig::bid_budget`, `play_budget`): a
//! bid's value depends mostly on the hidden cards, so bids default to more
//! sampled deals and fewer simulations each.
//!
//! Action encoding on each child is phase-stable (not re-indexed across
//! depth):
//! - Bidding: `action` = bid value in `0..=13`.
//! - Playing: `action` = card index in `0..=51` (absolute — hand-card
//!   positions shift after every play).

use rand::Rng;
use smallvec::SmallVec;

use crate::belief::{determinize, void_suits, DEFAULT_DETERMINIZE_ATTEMPTS};
use crate::bidding::{apply_bid, legal_bids};
use crate::encoder::hand_card_indices;
use crate::evaluator::{PolicyEvaluator, ValueEvaluator, NUM_BIDS};
use crate::playing::{apply_play, legal_plays};
use crate::scoring::{terminal_utilities, utilities, DEFAULT_LAMBDA};
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// Default `c_puct` exploration constant. 0.2 since Phase 4 (gen 1 used
/// 1.5): the warm start's priors are sharp, and at 1.5 the visit counts
/// barely depart from them; 0.1 and 0.2 scored the same (gen-2.md §6).
pub const DEFAULT_C_PUCT: f32 = 0.2;

/// Whether 1-card bids are searched (`MctsConfig::search_one_card_bids`).
/// Off: search lost to P alone there on both warm starts, because the
/// sampled deals ignore the bids already made (gen-2.md §6 Phase 4).
pub const DEFAULT_SEARCH_ONE_CARD_BIDS: bool = false;

/// Initial node capacity reserved per search. 10k nodes × ~80 B ≈ 800 KB.
pub const DEFAULT_ARENA_CAPACITY: usize = 10_000;

/// Default leaves per batched network call in the lockstep driver: 5, one
/// per sampled deal at the play budget. Gen 1 measured this as the
/// per-game-wall optimum on the 7950X / 1.63M-param transformer: per-call
/// ONNX cost rises super-linearly past it because the CPU is already
/// saturated by 32 concurrent batched forwards (gen-2.md §3.2). While
/// `target_batch` is at most the number of sampled deals, every tree has at
/// most one descent in flight, so virtual loss stays dormant. Raising it
/// is parked behind a model-size revisit (d_model ≥ 256).
pub const DEFAULT_TARGET_BATCH: usize = 5;

/// Virtual-loss weight. Each in-flight leaf along a path subtracts this
/// from `value_sums[acting]` during UCB1 selection, so concurrent descents
/// in the same tree pick different leaves. Utilities lie in [−1, 1] at
/// λ = 1, so 1.0 treats an in-flight leaf as a loss.
pub const VIRTUAL_LOSS_WEIGHT: f32 = 1.0;

/// Single MCTS tree node.
///
/// - `visit_count`: simulations that passed through this node. Used as
///   `N_parent`/`N_child` in the UCB1 exploration term and as the
///   denominator of every seat's mean.
/// - `value_sums`: per-seat sums of backed-up utilities; every visit adds
///   one value for every seat.
/// - `prior`: policy probability for the edge leading *into* this node
///   (populated at expansion).
/// - `action`: phase-stable edge label (see module docs). Root stores `0`.
/// - `children`: arena indices of child nodes; empty until expansion.
/// - `in_flight`: number of currently-pending descents that hold this node
///   on their path. Bumped along the path when a leaf is queued for a
///   batched network call and decremented right before that leaf's
///   expand/backup runs. UCB1 reads it as a temporary pessimistic visit so
///   concurrent descents inside the same tree pick different leaves.
///   Single-thread mutation only (one worker owns the arena), and `u16`
///   covers any plausible `target_batch`.
#[derive(Debug, Clone)]
pub struct MctsNode {
    pub visit_count: u32,
    pub value_sums: [f32; MAX_PLAYERS],
    pub prior: f32,
    pub action: u8,
    pub children: SmallVec<[u32; 14]>,
    pub in_flight: u16,
}

impl MctsNode {
    /// New unexpanded node. `prior` and `action` are edge-labelled at
    /// expansion; the root passes `prior = 1.0`, `action = 0`.
    #[inline]
    pub fn new(prior: f32, action: u8) -> Self {
        Self {
            visit_count: 0,
            value_sums: [0.0; MAX_PLAYERS],
            prior,
            action,
            children: SmallVec::new(),
            in_flight: 0,
        }
    }

    /// Mean utility of `seat` over this node's visits; `None` before the
    /// first visit.
    #[inline]
    pub fn q(&self, seat: u8) -> Option<f32> {
        (self.visit_count > 0).then(|| self.value_sums[seat as usize] / self.visit_count as f32)
    }

    /// True once the node has at least one child (i.e. has been expanded).
    #[inline]
    pub fn is_expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

/// Arena-backed MCTS tree. Node 0 is always the root.
#[derive(Debug, Clone)]
pub struct MctsArena {
    pub nodes: Vec<MctsNode>,
    /// Seat whose move is being searched (the root's acting player).
    pub root_player: u8,
}

impl MctsArena {
    /// Create an arena pre-allocated for `DEFAULT_ARENA_CAPACITY` nodes with
    /// an empty root.
    pub fn new(root_player: u8) -> Self {
        Self::with_capacity(root_player, DEFAULT_ARENA_CAPACITY)
    }

    pub fn with_capacity(root_player: u8, capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(MctsNode::new(1.0, 0));
        Self { nodes, root_player }
    }

    #[inline]
    pub fn root(&self) -> &MctsNode {
        &self.nodes[0]
    }

    #[inline]
    pub fn node(&self, idx: u32) -> &MctsNode {
        &self.nodes[idx as usize]
    }

    #[inline]
    pub fn node_mut(&mut self, idx: u32) -> &mut MctsNode {
        &mut self.nodes[idx as usize]
    }

    /// Push a child node and return its arena index. Caller is responsible
    /// for appending that index into the parent's `children` vec — splitting
    /// the borrow this way avoids aliasing when allocating many children in
    /// a loop.
    pub fn alloc(&mut self, prior: f32, action: u8) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MctsNode::new(prior, action));
        idx
    }
}

/// UCB1 score for `child` under `parent` from the acting seat's viewpoint,
/// including virtual-loss decoration from in-flight descents.
///
/// Without in-flight leaves (`child.in_flight == 0`) this is the standard
/// AlphaZero score
///
/// `score = Q(acting) + c_puct * P * sqrt(N_parent) / (1 + N_child)`
///
/// where `Q(acting) = value_sums[acting] / visit_count`. An unvisited child
/// returns `f32::INFINITY`, so every option is tried once before priors
/// and values rank them.
///
/// With `in_flight > 0`, each pending leaf counts as a visit worth
/// `−VIRTUAL_LOSS_WEIGHT` until its real value lands, in both the mean and
/// the exploration term.
#[inline]
pub fn ucb1_score(parent: &MctsNode, child: &MctsNode, acting: u8, c_puct: f32) -> f32 {
    let in_flight = child.in_flight as u32;
    let n = child.visit_count + in_flight;
    if n == 0 {
        return f32::INFINITY;
    }
    let vloss = VIRTUAL_LOSS_WEIGHT * in_flight as f32;
    let q = (child.value_sums[acting as usize] - vloss) / n as f32;
    let n_parent = parent.visit_count.max(1) as f32;
    q + c_puct * child.prior * n_parent.sqrt() / (1.0 + n as f32)
}

/// Pick the child of `parent_idx` with the highest UCB1 score. Ties go to
/// the first child (stable). Panics if the node has no children — callers
/// must check `is_expanded()` first.
pub fn select_best_child(arena: &MctsArena, parent_idx: u32, acting: u8, c_puct: f32) -> u32 {
    let parent = arena.node(parent_idx);
    debug_assert!(parent.is_expanded(), "select on unexpanded node");

    let mut best_idx = parent.children[0];
    let mut best_score = ucb1_score(parent, arena.node(best_idx), acting, c_puct);
    for &child_idx in &parent.children[1..] {
        let score = ucb1_score(parent, arena.node(child_idx), acting, c_puct);
        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
    }
    best_idx
}

/// Walk from the root, picking the UCB1-best child at each step, until
/// reaching an unexpanded node. Returns `(leaf_idx, path)` where `path`
/// contains every node index from root to leaf inclusive.
///
/// `acting_at` maps each node to the seat that acts at it. Callers derive
/// this by replaying actions on a scratch `BlobState`.
pub fn select_leaf<F>(
    arena: &MctsArena,
    c_puct: f32,
    mut acting_at: F,
) -> (u32, Vec<u32>)
where
    F: FnMut(u32) -> u8,
{
    let mut path = Vec::with_capacity(16);
    let mut idx: u32 = 0;
    path.push(idx);
    loop {
        let node = arena.node(idx);
        if !node.is_expanded() {
            return (idx, path);
        }
        let acting = acting_at(idx);
        idx = select_best_child(arena, idx, acting, c_puct);
        path.push(idx);
    }
}

/// Apply a phase-stable `action` label to `state`, dispatching to
/// `apply_bid` or `apply_play` based on the current phase. No-op in
/// terminal phases (`Scoring`, `Complete`) — expansion never produces
/// children from those phases, so the search horizon is the current round
/// (gen-2.md §5.2).
#[inline]
pub fn apply_action(state: &mut BlobState, action: u8) {
    match state.phase() {
        GamePhase::Bidding => apply_bid(state, action),
        GamePhase::Playing => apply_play(state, action),
        GamePhase::Scoring | GamePhase::Complete => {}
    }
}

/// True for phases where no more decisions exist in this round.
#[inline]
pub fn is_terminal(state: &BlobState) -> bool {
    matches!(state.phase(), GamePhase::Scoring | GamePhase::Complete)
}

/// Return `Some(action)` when exactly one legal move exists at `state`
/// (i.e. the position is forced), else `None`.
///
/// Used by the leaf-descent fast-path to skip the network calls entirely on
/// forced nodes — the prior carries no information when there is only
/// one child to put it on, so a placeholder child with `prior = 1.0` is
/// equivalent in expectation and saves the inference. Forced moves are
/// common in trick-taking games (root-forced rate ~37% in gen-1
/// measurements), so the saved evaluations compound across the search.
///
/// Always `None` in terminal phases; the descent loop checks
/// [`is_terminal`] before consulting this helper anyway.
#[inline]
pub fn forced_action(state: &BlobState) -> Option<u8> {
    let mask = match state.phase() {
        GamePhase::Bidding => legal_bids(state) as u64,
        GamePhase::Playing => legal_plays(state),
        GamePhase::Scoring | GamePhase::Complete => return None,
    };
    (mask.count_ones() == 1).then(|| mask.trailing_zeros() as u8)
}

/// Expand `node_idx` by creating one child per legal action.
///
/// `policy` is the policy evaluator's output for `state` (bidding: length
/// `NUM_BIDS` over bid values; playing: length `hand_card_indices.len()`
/// over hand positions — see [`crate::evaluator`]).
///
/// Children are labelled with phase-stable actions:
/// - Bidding: `action = bid`, `prior = policy[bid]`.
/// - Playing: `action = card_idx`, `prior = policy[pos]` where `pos` is
///   the card's position in `hand_card_indices`.
///
/// No-op if the node is already expanded or the state is terminal.
pub fn expand(arena: &mut MctsArena, node_idx: u32, state: &BlobState, policy: &[f32]) {
    crate::profiling::time(&crate::profiling::EXPAND, || {
        if arena.node(node_idx).is_expanded() || is_terminal(state) {
            return;
        }
        match state.phase() {
            GamePhase::Bidding => {
                let mask = legal_bids(state);
                let mut new_children: SmallVec<[u32; 14]> = SmallVec::new();
                for b in 0..NUM_BIDS as u8 {
                    if (mask >> b) & 1 == 1 {
                        let prior = policy.get(b as usize).copied().unwrap_or(0.0);
                        new_children.push(arena.alloc(prior, b));
                    }
                }
                arena.node_mut(node_idx).children = new_children;
            }
            GamePhase::Playing => {
                let hand = hand_card_indices(state, state.current_player);
                let legal = legal_plays(state);
                let mut new_children: SmallVec<[u32; 14]> = SmallVec::new();
                for (pos, &card_idx) in hand.iter().enumerate() {
                    if (legal >> card_idx) & 1 == 1 {
                        let prior = policy.get(pos).copied().unwrap_or(0.0);
                        new_children.push(arena.alloc(prior, card_idx));
                    }
                }
                arena.node_mut(node_idx).children = new_children;
            }
            GamePhase::Scoring | GamePhase::Complete => {}
        }
    })
}

/// Back up one simulation along `path` (root → leaf inclusive): count the
/// visit and add `utilities[s]` to every active seat's sum at every node.
/// Slots `>= num_players` are left untouched.
pub fn backup(arena: &mut MctsArena, path: &[u32], utilities: &[f32; MAX_PLAYERS], num_players: u8) {
    crate::profiling::time(&crate::profiling::BACKPROP, || {
        let n = (num_players as usize).min(MAX_PLAYERS);
        for &idx in path {
            let node = arena.node_mut(idx);
            node.visit_count += 1;
            for (sum, u) in node.value_sums[..n].iter_mut().zip(utilities) {
                *sum += u;
            }
        }
    })
}

/// Walk from the root of `arena` along UCB1-best children, replaying
/// actions on a clone of `root_state`, until reaching either an
/// unexpanded multi-legal node or a terminal state. Returns
/// `(leaf_idx, path, leaf_state)` where `path` includes both endpoints
/// (root and leaf inclusive).
///
/// **Forced-move fast-path:** when descent lands on an unexpanded node
/// whose state has exactly one legal action, the placeholder child is
/// allocated inline with `prior = 1.0` and descent continues — no network
/// call is queued for the forced node. The returned leaf is therefore
/// either terminal or multi-legal-unexpanded. Takes `&mut MctsArena`
/// because the fast-path allocates placeholder nodes during descent.
pub fn select_leaf_state(
    arena: &mut MctsArena,
    root_state: &BlobState,
    c_puct: f32,
) -> (u32, Vec<u32>, BlobState) {
    let mut state = *root_state;
    let mut path: Vec<u32> = Vec::with_capacity(16);
    let mut idx: u32 = 0;
    path.push(idx);
    loop {
        if is_terminal(&state) {
            return (idx, path, state);
        }
        if arena.node(idx).is_expanded() {
            let acting = state.current_player;
            let child_idx = select_best_child(arena, idx, acting, c_puct);
            let action = arena.node(child_idx).action;
            apply_action(&mut state, action);
            idx = child_idx;
            path.push(idx);
            continue;
        }
        if let Some(action) = forced_action(&state) {
            let child_idx = arena.alloc(1.0, action);
            arena.node_mut(idx).children.push(child_idx);
            apply_action(&mut state, action);
            idx = child_idx;
            path.push(idx);
            continue;
        }
        return (idx, path, state);
    }
}

/// Run `num_simulations` simulations on one tree, one leaf at a time.
///
/// Each simulation descends with [`select_leaf_state`]. A terminal leaf
/// backs up the exact utilities; any other leaf is expanded with the
/// policy's priors and backs up the utilities of the value net's ŝ. The
/// first simulation expands the root.
///
/// Reads `cfg.c_puct` and `cfg.lambda`. Search proper runs
/// [`run_lockstep_search`] over many trees; this is its serial reference.
pub fn run_search<P, V>(
    arena: &mut MctsArena,
    root_state: &BlobState,
    policy: &P,
    value: &V,
    num_simulations: u32,
    cfg: &MctsConfig,
) where
    P: PolicyEvaluator + ?Sized,
    V: ValueEvaluator + ?Sized,
{
    for _ in 0..num_simulations {
        let (leaf_idx, path, state) = select_leaf_state(arena, root_state, cfg.c_puct);
        let u = if is_terminal(&state) {
            terminal_utilities(&state, cfg.lambda)
        } else {
            expand(arena, leaf_idx, &state, &policy.policy(&state));
            utilities(&value.values(&state), state.num_players, cfg.lambda)
        };
        backup(arena, &path, &u, state.num_players);
    }
}

/// Lockstep search across several trees (one per sampled deal), batching
/// their leaves into shared network calls of up to `cfg.target_batch`
/// states.
///
/// Driver loop:
///
/// 1. Round-robin pick the not-yet-exhausted tree with the fewest
///    simulations so far, ties to the lowest index, so with
///    `target_batch >= trees` the first batch fills as
///    `[tree 0, tree 1, …, tree 0, tree 1, …]`.
/// 2. Walk root → leaf with [`select_leaf_state`]; UCB1 reads
///    `MctsNode::in_flight`, so descents already in this batch steer
///    later ones away from their paths.
/// 3. Terminal leaves back up their exact utilities immediately (no
///    network call, no `in_flight` decoration).
/// 4. Other leaves bump `in_flight` along the path and join the batch.
/// 5. **Cold-start duplicate guard.** If a fresh descent lands on a leaf
///    whose `in_flight > 0` (only possible while a tree's root is still
///    unexpanded — virtual loss can't redirect *through* an unexpanded
///    node), that tree sits out the rest of this batch.
/// 6. When the batch is full (or every eligible tree is exhausted or
///    sitting out), call the policy and value evaluators once each on the
///    whole batch, then, leaf by leaf in queue order: undo `in_flight`,
///    expand with the priors, back up the value net's utilities.
///
/// **Special cases:**
/// - `target_batch = 1`: one descent in flight at a time, so each tree's
///   node sequence matches [`run_search`] bit-for-bit.
/// - `target_batch = trees`: at most one descent per tree per batch, so
///   virtual loss never engages and node sequences again match
///   [`run_search`]. Both are pinned by `lockstep_search_matches_serial_per_det`.
/// - `target_batch > trees`: virtual loss steers concurrent descents inside
///   one tree apart, so visit counts no longer match serial search.
///
/// Post-condition: every node in every arena has `in_flight == 0`;
/// `debug_assert!`ed at the end so a path-bookkeeping bug fails loudly.
pub fn run_lockstep_search<P, V>(
    arenas: &mut [MctsArena],
    root_states: &[BlobState],
    policy: &P,
    value: &V,
    num_simulations: u32,
    cfg: &MctsConfig,
) where
    P: PolicyEvaluator + ?Sized,
    V: ValueEvaluator + ?Sized,
{
    debug_assert_eq!(arenas.len(), root_states.len());
    let num_dets = arenas.len();
    if num_dets == 0 || num_simulations == 0 {
        return;
    }
    let target_batch = cfg.target_batch.max(1);

    struct Pending {
        det: usize,
        leaf_idx: u32,
        path: Vec<u32>,
        leaf_state: BlobState,
    }

    let mut pending: Vec<Pending> = Vec::with_capacity(target_batch);
    let mut sims_done = vec![0u32; num_dets];
    let mut blocked = vec![false; num_dets];

    loop {
        if sims_done.iter().all(|&n| n >= num_simulations) {
            break;
        }
        pending.clear();
        for b in blocked.iter_mut() {
            *b = false;
        }

        // Fill one batch.
        while pending.len() < target_batch {
            let next_det = (0..num_dets)
                .filter(|&d| sims_done[d] < num_simulations && !blocked[d])
                .min_by_key(|&d| sims_done[d]);
            let Some(det) = next_det else {
                break;
            };

            let (leaf_idx, path, leaf_state) =
                select_leaf_state(&mut arenas[det], &root_states[det], cfg.c_puct);

            if is_terminal(&leaf_state) {
                let u = terminal_utilities(&leaf_state, cfg.lambda);
                backup(&mut arenas[det], &path, &u, leaf_state.num_players);
                sims_done[det] += 1;
                continue;
            }

            // Cold-start duplicate: an unexpanded root is reachable only
            // through itself, so a second descent before its expansion
            // lands on the same leaf. Virtual loss only redirects between
            // expanded siblings, so skip this tree until the batch flushes.
            if arenas[det].node(leaf_idx).in_flight > 0 {
                blocked[det] = true;
                continue;
            }

            for &n in &path {
                arenas[det].node_mut(n).in_flight += 1;
            }
            pending.push(Pending { det, leaf_idx, path, leaf_state });
            sims_done[det] += 1;
        }

        if pending.is_empty() {
            // Every tree is exhausted, or every descent this round ended
            // at a terminal leaf (already backed up).
            continue;
        }

        let states: Vec<&BlobState> = pending.iter().map(|p| &p.leaf_state).collect();
        let priors = policy.policy_batch(&states);
        let values = value.values_batch(&states);
        debug_assert_eq!((priors.len(), values.len()), (pending.len(), pending.len()));

        for ((p, prior), v) in pending.drain(..).zip(priors).zip(values) {
            // Undo the virtual visit before the real one lands.
            for &n in &p.path {
                arenas[p.det].node_mut(n).in_flight -= 1;
            }
            let n = p.leaf_state.num_players;
            expand(&mut arenas[p.det], p.leaf_idx, &p.leaf_state, &prior);
            backup(&mut arenas[p.det], &p.path, &utilities(&v, n, cfg.lambda), n);
        }
    }

    debug_assert!(
        arenas.iter().all(|a| a.nodes.iter().all(|n| n.in_flight == 0)),
        "lockstep search left non-zero in_flight on some node",
    );
}

/// Index of the most-visited action. Ties go to the higher prior, then to
/// the lower index (gen-2.md §5.4: with near-flat visits, the gen-1
/// `max_by_key` handed ties to the highest bid). `priors` shorter than
/// `visits` reads as 0. Returns 0 for an empty slice.
fn most_visited<V: PartialOrd + Copy>(visits: &[V], priors: &[f32]) -> usize {
    let prior = |i: usize| priors.get(i).copied().unwrap_or(0.0);
    let mut best = 0;
    for i in 1..visits.len() {
        if visits[i] > visits[best] || (visits[i] == visits[best] && prior(i) > prior(best)) {
            best = i;
        }
    }
    best
}

/// Action probabilities over the root's children, sharpened/flattened by
/// temperature `tau`. Returns `(action, probability)` pairs in the order
/// children were allocated (phase-stable action labels).
///
/// - `tau == 1.0`: directly proportional to visit counts.
/// - `tau → 0`: approaches argmax on visit count (deterministic; ties go to
///   the higher prior); the implementation treats `tau < 1e-3` as argmax to
///   avoid `f32::powf` overflow.
/// - `tau > 1.0`: flatter distribution (more exploration).
///
/// Returns an empty vec if the root is unexpanded.
pub fn root_action_probs(arena: &MctsArena, tau: f32) -> Vec<(u8, f32)> {
    let root = arena.root();
    if root.children.is_empty() {
        return Vec::new();
    }

    let visits: Vec<(u8, u32)> = root
        .children
        .iter()
        .map(|&c| {
            let n = arena.node(c);
            (n.action, n.visit_count)
        })
        .collect();

    // Argmax regime: any near-zero tau, or all-zero visits (nothing to
    // distribute proportionally without NaNs).
    let total_visits: u32 = visits.iter().map(|(_, n)| *n).sum();
    if tau < 1e-3 || total_visits == 0 {
        let mut out: Vec<(u8, f32)> = visits.iter().map(|(a, _)| (*a, 0.0)).collect();
        let counts: Vec<u32> = visits.iter().map(|(_, n)| *n).collect();
        let priors: Vec<f32> = root.children.iter().map(|&c| arena.node(c).prior).collect();
        out[most_visited(&counts, &priors)].1 = 1.0;
        return out;
    }

    let inv_tau = 1.0 / tau;
    let weights: Vec<f32> = visits
        .iter()
        .map(|(_, n)| (*n as f32).powf(inv_tau))
        .collect();
    let z: f32 = weights.iter().sum();
    if z == 0.0 {
        return visits.iter().map(|(a, _)| (*a, 0.0)).collect();
    }
    visits
        .iter()
        .zip(weights.iter())
        .map(|((a, _), w)| (*a, w / z))
        .collect()
}

fn default_target_batch() -> usize {
    DEFAULT_TARGET_BATCH
}

fn default_lambda() -> f32 {
    DEFAULT_LAMBDA
}

fn default_root_dirichlet_alpha() -> f32 {
    0.0
}

fn default_root_dirichlet_epsilon() -> f32 {
    0.0
}

fn default_search_one_card_bids() -> bool {
    DEFAULT_SEARCH_ONE_CARD_BIDS
}

/// Sample a single `Gamma(alpha, 1)` variate via Marsaglia–Tsang for
/// `alpha >= 1`, with the standard boost trick
/// (`G(alpha) ≡ G(alpha+1) · U^(1/alpha)`) for `alpha < 1`. Used by
/// `sample_dirichlet` for root-prior noise.
#[inline]
fn sample_gamma<R: Rng + ?Sized>(rng: &mut R, alpha: f32) -> f32 {
    if alpha < 1.0 {
        let u: f32 = rng.gen_range(1e-9_f32..1.0);
        return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal via Box–Muller (one sample per call; the
        // accept/reject loop already discards most variates).
        let u1: f32 = rng.gen_range(1e-9_f32..1.0);
        let u2: f32 = rng.gen_range(0.0_f32..1.0);
        let n = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        let v_root = 1.0 + c * n;
        if v_root <= 0.0 {
            continue;
        }
        let v = v_root * v_root * v_root;
        let u: f32 = rng.gen_range(0.0_f32..1.0);
        let n2 = n * n;
        if u < 1.0 - 0.0331 * n2 * n2 {
            return d * v;
        }
        if u.ln() < 0.5 * n2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample a Dirichlet(α, …, α) vector of length `n`. Returns a uniform
/// `1/n` vector as a degenerate fallback if all Gamma samples underflow
/// to zero (vanishingly rare; included so noise mixing can never produce
/// NaNs).
fn sample_dirichlet<R: Rng + ?Sized>(rng: &mut R, alpha: f32, n: usize) -> Vec<f32> {
    let mut samples: Vec<f32> = (0..n).map(|_| sample_gamma(rng, alpha)).collect();
    let s: f32 = samples.iter().sum();
    if s > 0.0 {
        for v in samples.iter_mut() {
            *v /= s;
        }
    } else {
        let uniform = 1.0 / n.max(1) as f32;
        for v in samples.iter_mut() {
            *v = uniform;
        }
    }
    samples
}

/// Mix Dirichlet noise into the priors of `node_idx`'s children in place:
/// `P'(a) = (1 − ε) · P(a) + ε · η(a)` with `η ~ Dir(α, …, α)`. Intended
/// for the root only.
///
/// No-op when the node is unexpanded or has zero children.
pub fn apply_root_dirichlet_noise<R: Rng + ?Sized>(
    arena: &mut MctsArena,
    node_idx: u32,
    alpha: f32,
    epsilon: f32,
    rng: &mut R,
) {
    let n = arena.node(node_idx).children.len();
    if n == 0 || epsilon <= 0.0 || alpha <= 0.0 {
        return;
    }
    let noise = sample_dirichlet(rng, alpha, n);
    let child_ids: SmallVec<[u32; 14]> = arena.node(node_idx).children.clone();
    for (i, child_idx) in child_ids.iter().enumerate() {
        let child = arena.node_mut(*child_idx);
        child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
    }
}

/// Per-decision temperature schedule. When set, the effective τ used by
/// `mcts_search` to convert root visit counts into the
/// **action-sampling** distribution depends on the decision index the
/// caller passes (e.g. the decision's number within its round, forced
/// moves included).
///
/// Note: the τ-schedule applies **only** to `MctsResult.policy_sampling`.
/// `MctsResult.policy_target` is held at τ=1 (proportional
/// to visit counts) so the policy head trains against the full
/// search-visit distribution regardless of late sampling sharpness.
/// Canonical AlphaZero: τ=1 for the target, τ→0 for sampling after the
/// opening.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TemperatureSchedule {
    /// τ = `early` for `decision_index < switch_at`, otherwise `late`.
    /// AlphaZero-style hard step.
    HardStep {
        early: f32,
        late: f32,
        switch_at: usize,
    },
}

impl TemperatureSchedule {
    /// Resolve the effective τ for a given decision index.
    pub fn temperature_at(&self, decision_index: usize) -> f32 {
        match *self {
            TemperatureSchedule::HardStep {
                early,
                late,
                switch_at,
            } => {
                if decision_index < switch_at {
                    early
                } else {
                    late
                }
            }
        }
    }
}

/// How much search one decision gets: `determinizations` sampled deals,
/// one tree each, with `sims_per_determinization` simulations per tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBudget {
    pub determinizations: u32,
    pub sims_per_determinization: u32,
}

impl SearchBudget {
    pub const fn new(determinizations: u32, sims_per_determinization: u32) -> Self {
        Self { determinizations, sims_per_determinization }
    }
}

impl std::fmt::Display for SearchBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.determinizations, self.sims_per_determinization)
    }
}

/// Default bid budget, 20 × 25: a bid's value depends mostly on the hidden
/// cards, so more sampled deals and fewer simulations (gen-2.md §5.4).
pub const DEFAULT_BID_BUDGET: SearchBudget = SearchBudget::new(20, 25);

/// Default play budget, 5 × 100 as in gen 1, until measurements say
/// otherwise (gen-2.md §5.4).
pub const DEFAULT_PLAY_BUDGET: SearchBudget = SearchBudget::new(5, 100);

/// Search-time configuration threaded through `mcts_search`.
///
/// Unknown keys are an error, so a typo or a stale config can't half-load
/// (gen-2.md §4).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MctsConfig {
    pub c_puct: f32,
    /// λ of the per-round utility (`scoring.rs`): the weight of the other
    /// seats' mean ŝ. 1 = my points minus the table's (default); 0 = my
    /// points only. Optional in config files.
    #[serde(default = "default_lambda")]
    pub lambda: f32,
    /// Search for bids.
    pub bid_budget: SearchBudget,
    /// Search for card plays.
    pub play_budget: SearchBudget,
    /// Constant temperature, used when `temperature_schedule` is `None`.
    /// Call sites read it through `MctsConfig::temperature_at`.
    pub temperature: f32,
    /// Optional per-decision schedule. When `Some`, overrides
    /// `temperature` and `mcts_search` resolves τ from the schedule using
    /// the `decision_index` argument. Optional in config files.
    #[serde(default)]
    pub temperature_schedule: Option<TemperatureSchedule>,
    pub arena_capacity: usize,
    /// Target leaves per batched network call (see
    /// [`DEFAULT_TARGET_BATCH`] and [`run_lockstep_search`]). `1`
    /// degenerates to fully serial search. Optional in config files.
    #[serde(default = "default_target_batch")]
    pub target_batch: usize,
    /// Dirichlet concentration α of the root-prior noise. When `<= 0` the
    /// heuristic `α = 10 / num_legal` is used per call — DeepMind's scaling
    /// rule, more robust across Blob's variable branching (2 plays … 14
    /// bids). Noise is off whenever `root_dirichlet_epsilon <= 0`.
    /// Optional in config files.
    #[serde(default = "default_root_dirichlet_alpha")]
    pub root_dirichlet_alpha: f32,
    /// Mixing weight ε for root Dirichlet noise:
    /// `P'(a) = (1 − ε) · P(a) + ε · η(a)`. AlphaZero uses 0.25; any value
    /// `<= 0` disables noise. Each tree draws its own η. Optional in config
    /// files.
    #[serde(default = "default_root_dirichlet_epsilon")]
    pub root_dirichlet_epsilon: f32,
    /// Search the bid of a 1-card round. Off (the default), the bid is P's
    /// policy, with no tree: the bid is the round's only decision (every
    /// play is forced), and on the Phase-4 warm starts search made it worse
    /// than P alone (gen-2.md §6 Phase 4). Optional in config files.
    #[serde(default = "default_search_one_card_bids")]
    pub search_one_card_bids: bool,
}

impl MctsConfig {
    /// Effective τ at a given decision index. Falls back to the constant
    /// `temperature` field when no schedule is configured.
    pub fn temperature_at(&self, decision_index: usize) -> f32 {
        match self.temperature_schedule {
            Some(s) => s.temperature_at(decision_index),
            None => self.temperature,
        }
    }

    /// The budget of a decision in `phase` (bidding or playing).
    pub fn budget(&self, phase: GamePhase) -> SearchBudget {
        match phase {
            GamePhase::Bidding => self.bid_budget,
            _ => self.play_budget,
        }
    }
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: DEFAULT_C_PUCT,
            lambda: DEFAULT_LAMBDA,
            bid_budget: DEFAULT_BID_BUDGET,
            play_budget: DEFAULT_PLAY_BUDGET,
            temperature: 1.0,
            temperature_schedule: None,
            arena_capacity: DEFAULT_ARENA_CAPACITY,
            target_batch: DEFAULT_TARGET_BATCH,
            root_dirichlet_alpha: default_root_dirichlet_alpha(),
            root_dirichlet_epsilon: default_root_dirichlet_epsilon(),
            search_one_card_bids: DEFAULT_SEARCH_ONE_CARD_BIDS,
        }
    }
}

/// Aggregated result of an `mcts_search` call.
///
/// The policy vectors and `root_prior` / `action_values` are dense, indexed
/// by the phase's canonical action space: bids 0..14 in `Bidding`,
/// hand-card positions (per `EncodedState::hand_card_indices`) in
/// `Playing`.
///
/// The training target and the action-sampling distribution are
/// deliberately decoupled:
///
/// - **`policy_target`** is always computed at τ = 1.0 from aggregated
///   root visit counts (`v_i / Σ v`). This is the training label, and
///   `visit_entropy` / `top1_visit_share` are computed from it. Keeping the
///   target at τ=1 preserves entropy in the policy's training signal even
///   when the sampler is sharp.
/// - **`policy_sampling`** is computed at
///   `cfg.temperature_at(decision_index)`. Used only for action sampling in
///   self-play. At τ→0 this collapses to one-hot on the most-visited
///   action. These two were once a single fused field, which meant the
///   τ-schedule collapsed both — a gen-1 run regressed strength because of
///   this fusion.
#[derive(Debug, Clone)]
pub struct MctsResult {
    pub policy_target: Vec<f32>,
    pub policy_sampling: Vec<f32>,
    /// Root prior per action, averaged over the sampled deals (after root
    /// noise, when on). Breaks visit ties for greedy play.
    pub root_prior: Vec<f32>,
    /// The deciding seat's mean utility per action, over every visit to it
    /// in every tree; 0 for unvisited actions.
    pub action_values: Vec<f32>,
    pub visit_entropy: f32,
    pub top1_visit_share: f32,
    pub total_visits: u32,
    /// The deciding seat's mean utility at the roots, averaged over the
    /// sampled deals.
    pub value_estimate: f32,
}

/// Shannon entropy of a probability vector (base e). Zero probabilities
/// contribute zero (`0·ln 0 = 0` by convention).
#[inline]
fn entropy(p: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &v in p {
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    h
}

/// Normalized signal quality: `1 - H(policy) / ln(num_legal)`. Zero when
/// the policy is uniform over legal actions, one when the policy is a
/// delta function. Measures decisiveness, not correctness: read it only
/// next to the bench (gen-2.md §5.7).
pub fn signal_ratio(result: &MctsResult, num_legal: usize) -> f32 {
    if num_legal <= 1 {
        return 1.0;
    }
    let h_max = (num_legal as f32).ln();
    if h_max <= 0.0 {
        return 0.0;
    }
    (1.0 - result.visit_entropy / h_max).clamp(0.0, 1.0)
}

/// Turn an action label into its dense-policy index for the given phase.
///
/// - Bidding: bid value is its own index.
/// - Playing: card index → hand-card-position via `hand_card_indices`.
///   Returns `None` if the card is not in the perspective hand (should
///   not happen for a legal child).
fn action_to_policy_index(
    phase: GamePhase,
    action: u8,
    hand_card_indices: &[u8],
) -> Option<usize> {
    match phase {
        GamePhase::Bidding => Some(action as usize),
        GamePhase::Playing => hand_card_indices.iter().position(|&c| c == action),
        _ => None,
    }
}

/// Full search over sampled deals, with diagnostics.
///
/// The phase's budget (`cfg.budget`) sets how many deals are sampled for
/// the hidden hands, consistent with known voids, and how many simulations
/// each deal's tree gets. The trees run in lockstep ([`run_lockstep_search`])
/// and their root visit counts are summed into one dense policy.
/// Temperature, entropy and top-1 share come from the sum, not per tree.
///
/// A forced move returns at once, with no tree and no network call; so does
/// a 1-card bid with `search_one_card_bids` off, from one P call.
pub fn mcts_search<P, V, R>(
    state: &BlobState,
    policy: &P,
    value: &V,
    cfg: &MctsConfig,
    rng: &mut R,
    decision_index: usize,
) -> MctsResult
where
    P: PolicyEvaluator + ?Sized,
    V: ValueEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    crate::profiling::time(&crate::profiling::MCTS_SEARCH, || {
        let phase = state.phase();
        if matches!(phase, GamePhase::Scoring | GamePhase::Complete) {
            return MctsResult {
                policy_target: Vec::new(),
                policy_sampling: Vec::new(),
                root_prior: Vec::new(),
                action_values: Vec::new(),
                visit_entropy: 0.0,
                top1_visit_share: 0.0,
                total_visits: 0,
                value_estimate: 0.0,
            };
        }

        let perspective = state.current_player;

        // Canonical action space + forced-move detection.
        let (policy_len, hand_card_indices, num_legal) = match phase {
            GamePhase::Bidding => {
                (NUM_BIDS, SmallVec::<[u8; 13]>::new(), legal_bids(state).count_ones() as usize)
            }
            GamePhase::Playing => {
                let hand = hand_card_indices(state, perspective);
                (hand.len(), hand, legal_plays(state).count_ones() as usize)
            }
            _ => unreachable!(),
        };

        // Forced move: skip search entirely. Both target and sampling
        // distributions are one-hot on the only legal action — there is no
        // τ-dependent decision to make.
        if let Some(action) = forced_action(state) {
            let mut policy = vec![0.0f32; policy_len];
            if let Some(idx) = action_to_policy_index(phase, action, &hand_card_indices) {
                policy[idx] = 1.0;
            }
            return MctsResult {
                policy_target: policy.clone(),
                policy_sampling: policy.clone(),
                root_prior: policy,
                action_values: vec![0.0; policy_len],
                visit_entropy: 0.0,
                top1_visit_share: 1.0,
                total_visits: 0,
                value_estimate: 0.0,
            };
        }

        // A 1-card bid from P alone: the prior is both policies (sampling at
        // the configured τ), and there are no visits or values.
        if !cfg.search_one_card_bids && phase == GamePhase::Bidding && state.cards_dealt == 1 {
            let prior = policy.policy(state);
            let tau = cfg.temperature_at(decision_index);
            return MctsResult {
                policy_sampling: prior_at_temperature(&prior, tau),
                visit_entropy: entropy(&prior),
                top1_visit_share: prior.iter().cloned().fold(0.0f32, f32::max),
                policy_target: prior.clone(),
                root_prior: prior,
                action_values: vec![0.0; policy_len],
                total_visits: 0,
                value_estimate: 0.0,
            };
        }

        let budget = cfg.budget(phase);
        let num_dets = budget.determinizations.max(1) as usize;
        let sims_per = budget.sims_per_determinization.max(1);
        let voids = void_suits(state);

        // One sampled deal and one arena per tree, driven in lockstep.
        let mut det_states: Vec<BlobState> = Vec::with_capacity(num_dets);
        let mut arenas: Vec<MctsArena> = Vec::with_capacity(num_dets);
        for _ in 0..num_dets {
            det_states.push(determinize(state, perspective, &voids, rng, DEFAULT_DETERMINIZE_ATTEMPTS));
            arenas.push(MctsArena::with_capacity(perspective, cfg.arena_capacity));
        }

        // Root Dirichlet noise: expand every root with one batched call
        // first, so the priors can be decorated with `(1−ε)·P + ε·Dir(α)`
        // before any selection runs. That counts as each tree's first
        // simulation (the one that would have expanded the root anyway), so
        // the lockstep run gets one fewer.
        let noise_on = cfg.root_dirichlet_epsilon > 0.0;
        let effective_sims = if noise_on {
            let alpha = if cfg.root_dirichlet_alpha > 0.0 {
                cfg.root_dirichlet_alpha
            } else {
                (10.0 / num_legal.max(1) as f32).max(1e-3)
            };
            let roots: Vec<&BlobState> = det_states.iter().collect();
            let priors = policy.policy_batch(&roots);
            let values = value.values_batch(&roots);
            for (det, (prior, v)) in priors.into_iter().zip(values).enumerate() {
                let n = det_states[det].num_players;
                expand(&mut arenas[det], 0, &det_states[det], &prior);
                apply_root_dirichlet_noise(&mut arenas[det], 0, alpha, cfg.root_dirichlet_epsilon, rng);
                backup(&mut arenas[det], &[0], &utilities(&v, n, cfg.lambda), n);
            }
            sims_per - 1
        } else {
            sims_per
        };

        run_lockstep_search(&mut arenas, &det_states, policy, value, effective_sims, cfg);

        let mut agg_visits = vec![0u64; policy_len];
        let mut value_sums = vec![0.0f32; policy_len];
        let mut root_prior = vec![0.0f32; policy_len];
        let mut value_sum = 0.0f32;
        for arena in &arenas {
            let root = arena.root();
            for &c in root.children.iter() {
                let child = arena.node(c);
                if let Some(idx) = action_to_policy_index(phase, child.action, &hand_card_indices) {
                    agg_visits[idx] += child.visit_count as u64;
                    value_sums[idx] += child.value_sums[perspective as usize];
                    root_prior[idx] += child.prior / num_dets as f32;
                }
            }
            value_sum += root.q(perspective).unwrap_or(0.0);
        }
        let action_values: Vec<f32> = value_sums
            .iter()
            .zip(&agg_visits)
            .map(|(&s, &n)| if n > 0 { s / n as f32 } else { 0.0 })
            .collect();

        // Two policy vectors over the summed visits: `policy_target` at
        // τ=1 for the training label, `policy_sampling` at the configured τ
        // for action selection (identical when that τ is 1).
        let tau_sampling = cfg.temperature_at(decision_index);
        let policy_target = visits_to_policy(&agg_visits, &root_prior, 1.0);
        let policy_sampling = if (tau_sampling - 1.0).abs() < 1e-6 {
            policy_target.clone()
        } else {
            visits_to_policy(&agg_visits, &root_prior, tau_sampling)
        };

        // Diagnostics read from the τ=1 target, the canonical "what does
        // search prefer" signal; from the τ-applied vector they collapse to
        // ~0 entropy under a late schedule.
        let visit_entropy = entropy(&policy_target);
        let top1_visit_share = policy_target.iter().cloned().fold(0.0f32, f32::max);

        MctsResult {
            policy_target,
            policy_sampling,
            root_prior,
            action_values,
            visit_entropy,
            top1_visit_share,
            total_visits: agg_visits.iter().sum::<u64>() as u32,
            value_estimate: value_sum / num_dets as f32,
        }
    })
}

/// `prior` at temperature `tau`: `p^(1/τ)` normalized; `tau < 1e-3` is
/// one-hot on the most likely action.
fn prior_at_temperature(prior: &[f32], tau: f32) -> Vec<f32> {
    if (tau - 1.0).abs() < 1e-6 {
        return prior.to_vec();
    }
    let mut out = vec![0.0f32; prior.len()];
    if tau < 1e-3 {
        let best = (0..prior.len()).fold(0, |b, i| if prior[i] > prior[b] { i } else { b });
        out[best] = 1.0;
        return out;
    }
    let weights: Vec<f32> = prior.iter().map(|&p| p.powf(1.0 / tau)).collect();
    let z: f32 = weights.iter().sum();
    if z > 0.0 {
        for (o, w) in out.iter_mut().zip(&weights) {
            *o = w / z;
        }
    }
    out
}

/// Map aggregated root visit counts to a dense probability vector at
/// temperature `tau`. `tau < 1e-3` collapses to one-hot on the most-visited
/// action, ties going to the higher `priors` entry; `tau == 1.0` is
/// proportional to visits. Empty / all-zero inputs return an all-zero
/// vector (caller treats as no-op).
fn visits_to_policy(agg_visits: &[u64], priors: &[f32], tau: f32) -> Vec<f32> {
    let mut policy = vec![0.0f32; agg_visits.len()];
    let sum_visits: u64 = agg_visits.iter().sum();
    if sum_visits == 0 {
        return policy;
    }
    if tau < 1e-3 {
        policy[most_visited(agg_visits, priors)] = 1.0;
        return policy;
    }
    let inv_tau = 1.0 / tau;
    let weights: Vec<f32> = agg_visits
        .iter()
        .map(|&v| (v as f32).powf(inv_tau))
        .collect();
    let z: f32 = weights.iter().sum();
    if z > 0.0 {
        for (i, w) in weights.iter().enumerate() {
            policy[i] = w / z;
        }
    }
    policy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{apply_bid as bid_apply, legal_bids as bid_legal};
    use crate::dealing::{deal, new_round, RoundParams};
    use crate::encoder::encode;
    use crate::evaluator::DummyEvaluator;
    use crate::game::new_game;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const D: DummyEvaluator = DummyEvaluator;

    fn cfg_with(bid: (u32, u32), play: (u32, u32)) -> MctsConfig {
        MctsConfig {
            bid_budget: SearchBudget::new(bid.0, bid.1),
            play_budget: SearchBudget::new(play.0, play.1),
            ..MctsConfig::default()
        }
    }

    /// An unknown key, from a typo or a stale config, fails to parse
    /// instead of being ignored — including gen 1's removed keys.
    #[test]
    fn mcts_config_rejects_unknown_keys() {
        let top = "c_puct = 1.5\ntemperature = 1.0\narena_capacity = 4096\n";
        let tables = "[bid_budget]\ndeterminizations = 20\nsims_per_determinization = 25\n\
                      [play_budget]\ndeterminizations = 5\nsims_per_determinization = 100\n";
        let cfg: MctsConfig = toml::from_str(&format!("{top}{tables}")).expect("valid config");
        assert_eq!(cfg.target_batch, DEFAULT_TARGET_BATCH);
        assert_eq!(cfg.lambda, DEFAULT_LAMBDA);
        assert_eq!((cfg.bid_budget, cfg.play_budget), (DEFAULT_BID_BUDGET, DEFAULT_PLAY_BUDGET));
        for stale in ["min_sims_floor = 60\n", "num_determinizations = 5\n", "epochs = 1\n"] {
            let err = toml::from_str::<MctsConfig>(&format!("{top}{stale}{tables}")).unwrap_err();
            assert!(err.to_string().contains("unknown field"), "{err}");
        }
        let typo = format!("{top}{tables}sims = 1\n");
        assert!(toml::from_str::<MctsConfig>(&typo).is_err(), "unknown key in a budget");

        let sched = "kind = \"hard_step\"\nearly = 1.0\nlate = 0.1\nswitch_at = 15\n";
        assert!(toml::from_str::<TemperatureSchedule>(sched).is_ok());
        assert!(toml::from_str::<TemperatureSchedule>(&format!("{sched}typo = 1\n")).is_err());
    }

    #[test]
    fn arena_root_is_node_zero() {
        let a = MctsArena::new(3);
        assert_eq!(a.nodes.len(), 1);
        assert_eq!(a.root_player, 3);
        assert!(!a.root().is_expanded());
        assert_eq!(a.root().prior, 1.0);
    }

    #[test]
    fn alloc_returns_increasing_indices() {
        let mut a = MctsArena::new(0);
        let c0 = a.alloc(0.5, 7);
        let c1 = a.alloc(0.5, 8);
        assert_eq!(c0, 1);
        assert_eq!(c1, 2);
        assert_eq!(a.node(c0).action, 7);
        assert_eq!(a.node(c1).action, 8);
    }

    #[test]
    fn q_is_each_seats_mean_over_all_visits() {
        let mut node = MctsNode::new(0.1, 0);
        assert_eq!(node.q(0), None, "no value before the first visit");
        node.visit_count = 4;
        node.value_sums[2] = 2.0;
        node.value_sums[4] = -1.2;
        assert!((node.q(2).unwrap() - 0.5).abs() < 1e-6);
        assert!((node.q(4).unwrap() + 0.3).abs() < 1e-6);
        assert_eq!(node.q(0), Some(0.0));
    }

    #[test]
    fn unvisited_child_has_infinite_ucb1() {
        let mut parent = MctsNode::new(1.0, 0);
        parent.visit_count = 10;
        let child = MctsNode::new(0.01, 0);
        let score = ucb1_score(&parent, &child, 0, DEFAULT_C_PUCT);
        assert!(score.is_infinite() && score > 0.0);
    }

    #[test]
    fn ucb1_matches_hand_computed_value() {
        let mut parent = MctsNode::new(1.0, 0);
        parent.visit_count = 16;
        let mut child = MctsNode::new(0.25, 0);
        child.visit_count = 4;
        child.value_sums[1] = 2.0; // Q = 0.5 for seat 1.
        child.value_sums[0] = -2.0;
        let c_puct = 1.5;
        let explore = c_puct * 0.25 * (16f32).sqrt() / (1.0 + 4.0);
        assert!((ucb1_score(&parent, &child, 1, c_puct) - (0.5 + explore)).abs() < 1e-6);
        assert!((ucb1_score(&parent, &child, 0, c_puct) - (-0.5 + explore)).abs() < 1e-6);
        // One leaf in flight: a fifth visit worth −1.
        child.in_flight = 1;
        let with_vl = (2.0 - VIRTUAL_LOSS_WEIGHT) / 5.0 + c_puct * 0.25 * 4.0 / 6.0;
        assert!((ucb1_score(&parent, &child, 1, c_puct) - with_vl).abs() < 1e-6);
    }

    #[test]
    fn select_best_child_prefers_unvisited_then_higher_score() {
        let mut arena = MctsArena::new(0);
        // Two visited children with known Q, one unvisited.
        let a = arena.alloc(0.3, 1);
        let b = arena.alloc(0.3, 2);
        let c = arena.alloc(0.3, 3);
        arena.node_mut(a).visit_count = 4;
        arena.node_mut(a).value_sums[0] = 0.4;
        arena.node_mut(b).visit_count = 4;
        arena.node_mut(b).value_sums[0] = 3.2; // much higher Q
        // c left unvisited
        arena.node_mut(0).visit_count = 8;
        arena.node_mut(0).children.extend_from_slice(&[a, b, c]);

        // Unvisited `c` wins on infinite UCB1.
        assert_eq!(select_best_child(&arena, 0, 0, DEFAULT_C_PUCT), c);

        // Give c a visit; b (higher Q) should now win over a.
        arena.node_mut(c).visit_count = 1;
        assert_eq!(select_best_child(&arena, 0, 0, DEFAULT_C_PUCT), b);
    }

    #[test]
    fn select_leaf_walks_to_unexpanded_node() {
        let mut arena = MctsArena::new(0);
        // root -> a -> a1 (leaf)
        let a = arena.alloc(1.0, 1);
        let a1 = arena.alloc(1.0, 2);
        arena.node_mut(0).children.push(a);
        arena.node_mut(a).children.push(a1);
        arena.node_mut(0).visit_count = 2;
        arena.node_mut(a).visit_count = 1;
        // a1 unvisited → leaf.

        let (leaf, path) = select_leaf(&arena, DEFAULT_C_PUCT, |_| 0);
        assert_eq!(leaf, a1);
        assert_eq!(path, vec![0, a, a1]);
    }

    fn playing_state(seed: u64) -> BlobState {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        while s.game_phase == GamePhase::Bidding as u8 {
            let mask = bid_legal(&s);
            let b = mask.trailing_zeros() as u8;
            bid_apply(&mut s, b);
        }
        s
    }

    #[test]
    fn expand_bidding_creates_one_child_per_legal_bid() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        assert_eq!(s.phase(), GamePhase::Bidding);

        let mut arena = MctsArena::new(s.current_player);
        let policy = D.policy(&s);
        expand(&mut arena, 0, &s, &policy);

        let mask = legal_bids(&s);
        let expected: Vec<u8> = (0..NUM_BIDS as u8).filter(|b| (mask >> b) & 1 == 1).collect();
        let actions: Vec<u8> = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).action)
            .collect();
        assert_eq!(actions, expected);
        // Priors match the dummy's uniform-over-legal policy.
        for &c in arena.root().children.iter() {
            let child = arena.node(c);
            assert!((child.prior - policy[child.action as usize]).abs() < 1e-6);
        }
    }

    #[test]
    fn expand_playing_uses_card_index_actions() {
        let s = playing_state(11);
        let mut arena = MctsArena::new(s.current_player);
        expand(&mut arena, 0, &s, &D.policy(&s));

        let enc = encode(&s, s.current_player);
        let legal = legal_plays(&s);
        let expected_card_indices: Vec<u8> = enc
            .hand_card_indices
            .iter()
            .filter(|&&ci| (legal >> ci) & 1 == 1)
            .copied()
            .collect();
        let got: Vec<u8> = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).action)
            .collect();
        assert_eq!(got, expected_card_indices);
    }

    #[test]
    fn expand_is_noop_when_already_expanded_or_terminal() {
        let s = playing_state(3);
        let mut arena = MctsArena::new(s.current_player);
        let policy = D.policy(&s);
        expand(&mut arena, 0, &s, &policy);
        let n = arena.nodes.len();
        expand(&mut arena, 0, &s, &policy);
        assert_eq!(arena.nodes.len(), n);

        // Terminal state: forge a Scoring phase.
        let mut term = s;
        term.game_phase = GamePhase::Scoring as u8;
        let mut a2 = MctsArena::new(0);
        expand(&mut a2, 0, &term, &[]);
        assert!(!a2.root().is_expanded());
    }

    #[test]
    fn backup_credits_every_active_seat_on_the_path() {
        let mut arena = MctsArena::new(0);
        // Path root → A → B; C is off the path.
        let a = arena.alloc(0.5, 1);
        let b = arena.alloc(0.5, 2);
        let c = arena.alloc(0.5, 3);
        arena.node_mut(0).children.extend_from_slice(&[a, c]);
        arena.node_mut(a).children.push(b);

        let mut u = [0.0f32; MAX_PLAYERS];
        for (i, v) in u.iter_mut().enumerate() {
            *v = i as f32 * 0.25 - 0.5; // distinct per seat, nonzero beyond 4
        }
        let path = [0, a, b];
        backup(&mut arena, &path, &u, 4);
        backup(&mut arena, &path, &u, 4);

        for &idx in &path {
            let n = arena.node(idx);
            assert_eq!(n.visit_count, 2);
            for (s, x) in u.iter().enumerate().take(4) {
                assert!((n.value_sums[s] - 2.0 * x).abs() < 1e-6, "seat {s}");
            }
            assert!(n.value_sums[4..].iter().all(|&v| v == 0.0), "inactive seats untouched");
        }
        assert_eq!(arena.node(c).visit_count, 0);
    }

    #[test]
    fn run_search_visits_all_legal_actions_and_sums_correctly() {
        let s = playing_state(42);
        let mut arena = MctsArena::new(s.current_player);
        let sims = 100u32;
        run_search(&mut arena, &s, &D, &D, sims, &MctsConfig::default());

        // Every legal child of root should be visited at least once.
        for &c in arena.root().children.iter() {
            assert!(
                arena.node(c).visit_count > 0,
                "child action {} unvisited",
                arena.node(c).action
            );
        }

        // Root visit count equals number of simulations.
        assert_eq!(arena.root().visit_count, sims);
        let sum_child_visits: u32 = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).visit_count)
            .sum();
        // One visit lands on the root itself on the first sim (pre-expansion),
        // the remaining `sims - 1` descend into exactly one root child.
        assert_eq!(sum_child_visits, sims - 1);
    }

    #[test]
    fn root_action_probs_match_visits_at_tau_one() {
        let s = playing_state(5);
        let mut arena = MctsArena::new(s.current_player);
        run_search(&mut arena, &s, &D, &D, 80, &MctsConfig::default());

        let probs = root_action_probs(&arena, 1.0);
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");

        let total_visits: u32 = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).visit_count)
            .sum();
        for (&c, (a, p)) in arena.root().children.iter().zip(probs.iter()) {
            let n = arena.node(c);
            assert_eq!(n.action, *a);
            let expected = n.visit_count as f32 / total_visits as f32;
            assert!((p - expected).abs() < 1e-5);
        }
    }

    /// Each phase searches with its own budget: every tree's root gets
    /// `sims` visits, all but the first of which land on a child.
    #[test]
    fn budgets_are_per_phase() {
        let cfg = cfg_with((3, 10), (2, 12));
        assert_eq!(cfg.budget(GamePhase::Bidding), SearchBudget::new(3, 10));
        assert_eq!(cfg.budget(GamePhase::Playing), SearchBudget::new(2, 12));
        let d = MctsConfig::default();
        assert_eq!((d.bid_budget.to_string(), d.play_budget.to_string()), ("20x25".into(), "5x100".into()));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let mut bidding = new_game(4, 5).unwrap();
        deal(&mut bidding, &mut rng);
        assert_eq!(mcts_search(&bidding, &D, &D, &cfg, &mut rng, 0).total_visits, 3 * 9);
        let playing = playing_state(4);
        assert_eq!(mcts_search(&playing, &D, &D, &cfg, &mut rng, 0).total_visits, 2 * 11);
    }

    #[test]
    fn mcts_search_forced_move_shortcut() {
        // A synthetic 0-card bidding state: only bid 0 is legal for a
        // non-dealer.
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 0;
        s.dealer = 2;
        s.current_player = 0;
        s.game_phase = GamePhase::Bidding as u8;
        assert_eq!(legal_bids(&s), 1);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let r = mcts_search(&s, &D, &D, &MctsConfig::default(), &mut rng, 0);
        assert_eq!(r.policy_target.len(), NUM_BIDS);
        assert!((r.policy_target[0] - 1.0).abs() < 1e-6);
        assert_eq!(r.total_visits, 0);
        assert!((r.top1_visit_share - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mcts_search_bidding_produces_normalized_policy() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let mut s = new_game(4, 5).unwrap();
        crate::dealing::deal(&mut s, &mut rng);
        assert_eq!(s.phase(), GamePhase::Bidding);
        let num_legal = legal_bids(&s).count_ones() as usize;

        let r = mcts_search(&s, &D, &D, &cfg_with((2, 40), (1, 1)), &mut rng, 0);
        assert_eq!(r.policy_target.len(), NUM_BIDS);
        let sum: f32 = r.policy_target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");

        // Every legal bid has nonzero probability, illegal bids are zero.
        let mask = legal_bids(&s);
        let mut legal_nonzero = 0usize;
        for b in 0..NUM_BIDS {
            if (mask >> b) & 1 == 1 {
                assert!(r.policy_target[b] > 0.0, "legal bid {b} has zero policy");
                legal_nonzero += 1;
            } else {
                assert_eq!(r.policy_target[b], 0.0, "illegal bid {b} has nonzero policy");
            }
        }
        assert_eq!(legal_nonzero, num_legal);
        assert!(r.total_visits > 0);
    }

    #[test]
    fn mcts_search_playing_indexes_policy_by_hand_position() {
        let s = playing_state(77);
        let perspective = s.current_player;
        let enc = encode(&s, perspective);
        let legal = legal_plays(&s);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(77);
        let r = mcts_search(&s, &D, &D, &cfg_with((1, 1), (2, 30)), &mut rng, 0);
        assert_eq!(r.policy_target.len(), enc.hand_card_indices.len());
        let sum: f32 = r.policy_target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");

        for (pos, &ci) in enc.hand_card_indices.iter().enumerate() {
            let legal_card = (legal >> ci) & 1 == 1;
            if legal_card {
                assert!(r.policy_target[pos] > 0.0, "legal pos {pos} has zero policy");
            } else {
                assert_eq!(r.policy_target[pos], 0.0, "illegal pos {pos} has nonzero policy");
            }
        }
    }

    /// ŝ = 1 for one fixed seat, 0 for the others.
    struct SeatWins(u8);

    impl ValueEvaluator for SeatWins {
        fn values(&self, _: &BlobState) -> [f32; MAX_PLAYERS] {
            let mut v = [0.0; MAX_PLAYERS];
            v[self.0 as usize] = 1.0;
            v
        }
    }

    /// Exit criterion (gen-2.md §6 Phase 3): every explored root option
    /// carries a value for the deciding seat, on every visit. V says the
    /// decider makes 1 and everyone else 0, so the decider's utility is
    /// exactly 1 at every leaf and each other seat's −1/(n−1). The round's
    /// end is out of reach, so every backup comes from V. Gen 1 credited a
    /// leaf only to the seat to move there, so most bid options never got
    /// the decider's value (gen-2.md §2.3: 70%).
    #[test]
    fn every_explored_root_option_carries_the_deciders_value() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0);
        let params = RoundParams { num_players: 5, cards_dealt: 7, trump: 1, dealer: 4 };
        let s = new_round(params, &mut rng).unwrap();
        let me = s.current_player;
        let r = mcts_search(&s, &D, &SeatWins(me), &MctsConfig::default(), &mut rng, 0);

        let mask = legal_bids(&s);
        let mut explored = 0;
        for b in 0..NUM_BIDS {
            if (mask >> b) & 1 == 0 {
                continue;
            }
            assert!(r.policy_target[b] > 0.0, "bid {b} never explored");
            assert!((r.action_values[b] - 1.0).abs() < 1e-5, "bid {b}: {}", r.action_values[b]);
            explored += 1;
        }
        assert_eq!(explored, 8);
        assert!((r.value_estimate - 1.0).abs() < 1e-5);

        // The other seats hear about every visit too.
        let mut arena = MctsArena::new(me);
        run_search(&mut arena, &s, &D, &SeatWins(me), 40, &MctsConfig::default());
        for &c in &arena.root().children {
            let child = arena.node(c);
            for seat in 0..5u8 {
                let want = if seat == me { 1.0 } else { -0.25 };
                assert!((child.q(seat).unwrap() - want).abs() < 1e-5, "seat {seat}");
            }
        }
    }

    /// Exit criterion (gen-2.md §6 Phase 3): for the last bidder in a fully
    /// known 1-card round, each bid's search value equals its exact `u_s`.
    /// After the last bid every play is forced, so each bid's line runs to
    /// the end of the round without a network call.
    #[test]
    fn last_bidder_in_known_one_card_round_values_each_bid_exactly() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1C);
        let mut checked = 0;
        for game in 0..40u64 {
            let n = 3 + (game % 4) as u8;
            let params = RoundParams { num_players: n, cards_dealt: 1, trump: (game % 5) as u8, dealer: 0 };
            let mut s = new_round(params, &mut rng).unwrap();
            // Two bids of 1 leave the dealer a free choice of 0 or 1.
            for k in 0..n - 1 {
                bid_apply(&mut s, (k < 2) as u8);
            }
            assert_eq!(s.current_player, s.dealer);
            assert_eq!(legal_bids(&s), 0b11);

            for lambda in [1.0, 0.0] {
                let cfg = MctsConfig { lambda, ..MctsConfig::default() };
                let mut arena = MctsArena::new(s.dealer);
                run_search(&mut arena, &s, &D, &D, 10, &cfg);
                for &c in &arena.root().children {
                    let child = arena.node(c);
                    let mut end = s;
                    bid_apply(&mut end, child.action);
                    while let Some(card) = forced_action(&end) {
                        apply_play(&mut end, card);
                    }
                    assert_eq!(end.phase(), GamePhase::Scoring);
                    let exact = terminal_utilities(&end, lambda);
                    for seat in 0..n {
                        let q = child.q(seat).expect("every bid is visited");
                        assert!((q - exact[seat as usize]).abs() < 1e-6, "bid {}, seat {seat}", child.action);
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 40 * 2 * 2);
    }

    /// Counts the states each network sees.
    #[derive(Default)]
    struct Counting {
        policy_states: AtomicUsize,
        value_states: AtomicUsize,
    }

    impl PolicyEvaluator for Counting {
        fn policy(&self, state: &BlobState) -> Vec<f32> {
            self.policy_states.fetch_add(1, Ordering::Relaxed);
            D.policy(state)
        }
    }

    impl ValueEvaluator for Counting {
        fn values(&self, state: &BlobState) -> [f32; MAX_PLAYERS] {
            assert!(!is_terminal(state));
            let total: u32 = state.hands[..state.num_players as usize].iter().map(|h| h.count_ones()).sum();
            assert_eq!(total, state.num_players as u32 * state.cards_dealt as u32
                - (state.tricks_completed as u32 * state.num_players as u32 + state.trick_cards_played as u32),
                "V sees a full deal");
            self.value_states.fetch_add(1, Ordering::Relaxed);
            [0.0; MAX_PLAYERS]
        }
    }

    /// With `search_one_card_bids` off, a 1-card bid is P's policy from
    /// one P call and no tree; larger rounds' bids are still searched.
    #[test]
    fn one_card_bids_can_come_from_p_alone() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let off = MctsConfig { search_one_card_bids: false, ..cfg_with((4, 20), (1, 1)) };
        let one = new_round(RoundParams { num_players: 4, cards_dealt: 1, trump: 0, dealer: 0 }, &mut rng).unwrap();
        let c = Counting::default();
        let r = mcts_search(&one, &c, &c, &off, &mut rng, 0);
        assert_eq!((c.policy_states.load(Ordering::Relaxed), c.value_states.load(Ordering::Relaxed)), (1, 0));
        assert_eq!(r.policy_target, D.policy(&one));
        assert_eq!((r.total_visits, r.policy_target.clone()), (0, r.root_prior.clone()));
        let greedy = MctsConfig { temperature: 0.0, ..off };
        let sharp = mcts_search(&one, &D, &D, &greedy, &mut rng, 0).policy_sampling;
        assert_eq!(sharp.iter().filter(|&&p| p == 1.0).count(), 1, "τ = 0 is one-hot: {sharp:?}");

        let two = new_round(RoundParams { num_players: 4, cards_dealt: 2, trump: 0, dealer: 0 }, &mut rng).unwrap();
        let c = Counting::default();
        let r = mcts_search(&two, &c, &c, &off, &mut rng, 0);
        assert!(r.total_visits > 0 && c.value_states.load(Ordering::Relaxed) > 0, "a 2-card bid is searched");
        let on = MctsConfig { search_one_card_bids: true, ..off };
        assert!(mcts_search(&one, &D, &D, &on, &mut rng, 0).total_visits > 0);
    }

    /// Each leaf costs one P call and one V call; terminal and forced
    /// nodes cost none.
    #[test]
    fn each_leaf_costs_one_policy_and_one_value_call() {
        let s = playing_state(8);
        let c = Counting::default();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(8);
        let r = mcts_search(&s, &c, &c, &cfg_with((1, 1), (3, 50)), &mut rng, 0);
        let (p, v) = (c.policy_states.load(Ordering::Relaxed), c.value_states.load(Ordering::Relaxed));
        assert_eq!(p, v);
        assert!(p > 0 && p <= 3 * 50, "{p} leaves");
        assert!(r.total_visits > 0);
    }

    #[test]
    fn temperature_schedule_hard_step_resolves_correctly() {
        let sched = TemperatureSchedule::HardStep {
            early: 1.0,
            late: 0.1,
            switch_at: 15,
        };
        assert_eq!(sched.temperature_at(0), 1.0);
        assert_eq!(sched.temperature_at(14), 1.0);
        assert_eq!(sched.temperature_at(15), 0.1);
        assert_eq!(sched.temperature_at(100), 0.1);

        // MctsConfig falls back to constant temperature when schedule is None.
        let cfg = MctsConfig {
            temperature: 0.5,
            temperature_schedule: None,
            ..MctsConfig::default()
        };
        assert_eq!(cfg.temperature_at(0), 0.5);
        assert_eq!(cfg.temperature_at(50), 0.5);

        let cfg = MctsConfig {
            temperature: 1.0,
            temperature_schedule: Some(sched),
            ..MctsConfig::default()
        };
        assert_eq!(cfg.temperature_at(0), 1.0);
        assert_eq!(cfg.temperature_at(15), 0.1);
    }

    /// A late τ→0 schedule collapses the **sampling** policy to one-hot on
    /// the most-visited action; early τ=1 spreads sampling mass across all
    /// visited children. Same state, same seed, two `decision_index`
    /// values — verifies the schedule wires through `mcts_search`. The
    /// target stays τ=1 either way.
    #[test]
    fn mcts_search_honors_temperature_schedule() {
        let s = playing_state(123);
        let cfg = MctsConfig {
            temperature: 1.0,
            temperature_schedule: Some(TemperatureSchedule::HardStep {
                early: 1.0,
                late: 0.0,
                switch_at: 15,
            }),
            ..cfg_with((1, 1), (2, 60))
        };

        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(7);
        let r_early = mcts_search(&s, &D, &D, &cfg, &mut rng_a, 0);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(7);
        let r_late = mcts_search(&s, &D, &D, &cfg, &mut rng_b, 50);

        let nonzero_early = r_early
            .policy_sampling
            .iter()
            .filter(|&&p| p > 0.0)
            .count();
        assert!(
            nonzero_early >= 2,
            "early τ=1 sampling should spread mass; nonzero={nonzero_early}"
        );
        assert_eq!(
            r_early.policy_target, r_early.policy_sampling,
            "at τ=1 target and sampling should be identical"
        );

        // Late: τ→0 → sampling is one-hot.
        let max_late = r_late
            .policy_sampling
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        assert!(
            (max_late - 1.0).abs() < 1e-6,
            "late sampling argmax mass={max_late}"
        );
        let nonzero_late_sampling = r_late
            .policy_sampling
            .iter()
            .filter(|&&p| p > 0.0)
            .count();
        assert_eq!(
            nonzero_late_sampling, 1,
            "late τ→0 sampling must be one-hot"
        );

        // The late target is *not* collapsed by the schedule.
        let nonzero_late_target =
            r_late.policy_target.iter().filter(|&&p| p > 0.0).count();
        assert!(
            nonzero_late_target >= 2,
            "late τ→0 target should stay at τ=1 (≥ 2 nonzero); got {nonzero_late_target}"
        );
        let target_sum: f32 = r_late.policy_target.iter().sum();
        assert!((target_sum - 1.0).abs() < 1e-5, "target sum={target_sum}");
    }

    #[test]
    fn signal_ratio_zero_for_uniform_policy() {
        let r = MctsResult {
            policy_target: vec![0.25, 0.25, 0.25, 0.25],
            policy_sampling: vec![0.25, 0.25, 0.25, 0.25],
            root_prior: vec![0.25, 0.25, 0.25, 0.25],
            action_values: vec![0.0; 4],
            visit_entropy: (4f32).ln(),
            top1_visit_share: 0.25,
            total_visits: 40,
            value_estimate: 0.0,
        };
        let sr = signal_ratio(&r, 4);
        assert!(sr.abs() < 1e-6, "sr={sr}");
    }

    /// Distinct, state-dependent values per seat, so a mixed-up backup
    /// shows in the parity tests below.
    struct HandValue;

    impl ValueEvaluator for HandValue {
        fn values(&self, state: &BlobState) -> [f32; MAX_PLAYERS] {
            let mut v = [0.0; MAX_PLAYERS];
            for (s, x) in v.iter_mut().enumerate().take(state.num_players as usize) {
                *x = (state.hands[s] % 97) as f32 / 97.0;
            }
            v
        }
    }

    fn assert_same_trees(a: &[MctsArena], b: &[MctsArena]) {
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.nodes.len(), b.nodes.len(), "det {i}: node count differs");
            for (j, (na, nb)) in a.nodes.iter().zip(&b.nodes).enumerate() {
                assert_eq!(na.visit_count, nb.visit_count, "det {i} node {j} visit_count");
                assert_eq!(na.action, nb.action, "det {i} node {j} action");
                assert_eq!(na.children.as_slice(), nb.children.as_slice(), "det {i} node {j} children");
                for seat in 0..MAX_PLAYERS {
                    assert!(
                        (na.value_sums[seat] - nb.value_sums[seat]).abs() < 1e-5,
                        "det {i} node {j} value_sums[{seat}]: {} vs {}",
                        na.value_sums[seat],
                        nb.value_sums[seat],
                    );
                }
            }
        }
    }

    /// Lockstep batching across trees reproduces each tree of the serial
    /// driver bit-for-bit while virtual loss never engages:
    /// `target_batch` equal to the number of trees, or 1.
    #[test]
    fn lockstep_search_matches_serial_per_det() {
        let states = [playing_state(101), playing_state(202), playing_state(303)];
        let sims = 80u32;
        // A batch of one leaf per tree matches serial search while no
        // descent ends at the round's end: such a leaf is backed up at once,
        // and its tree may then add a second leaf to the same batch. At
        // c_puct 1.5 none does within 80 simulations here. A batch of 1
        // matches at any c_puct.
        for (target_batch, c_puct) in [(states.len(), 1.5), (1, DEFAULT_C_PUCT)] {
            let cfg = MctsConfig { target_batch, c_puct, ..MctsConfig::default() };
            let mut serial: Vec<MctsArena> =
                states.iter().map(|s| MctsArena::new(s.current_player)).collect();
            for (arena, state) in serial.iter_mut().zip(states.iter()) {
                run_search(arena, state, &D, &HandValue, sims, &cfg);
            }
            let mut lockstep: Vec<MctsArena> =
                states.iter().map(|s| MctsArena::new(s.current_player)).collect();
            run_lockstep_search(&mut lockstep, &states, &D, &HandValue, sims, &cfg);
            assert_same_trees(&serial, &lockstep);
        }
    }

    /// Every virtual visit is undone by the matching expand/backup step,
    /// also when several descents share one tree (`target_batch` above the
    /// number of trees).
    #[test]
    fn lockstep_search_clears_in_flight_at_target_batch_above_num_dets() {
        let states = [playing_state(11), playing_state(22), playing_state(33)];
        let mut arenas: Vec<MctsArena> =
            states.iter().map(|s| MctsArena::new(s.current_player)).collect();
        let cfg = MctsConfig { target_batch: 8, ..MctsConfig::default() };
        run_lockstep_search(&mut arenas, &states, &D, &HandValue, 60, &cfg);
        for (i, arena) in arenas.iter().enumerate() {
            for (j, n) in arena.nodes.iter().enumerate() {
                assert_eq!(n.in_flight, 0, "det {i} node {j} left with in_flight={}", n.in_flight);
            }
        }
    }

    /// Whatever `target_batch`, each tree's root ends with exactly
    /// `num_simulations` visits — the summed policy relies on it.
    #[test]
    fn lockstep_search_root_visit_count_matches_sim_budget() {
        let states = [playing_state(7), playing_state(13)];
        let sims = 50u32;
        for target_batch in [1usize, 2, 5, 8] {
            let cfg = MctsConfig { target_batch, ..MctsConfig::default() };
            let mut arenas: Vec<MctsArena> =
                states.iter().map(|s| MctsArena::new(s.current_player)).collect();
            run_lockstep_search(&mut arenas, &states, &D, &D, sims, &cfg);
            for (i, arena) in arenas.iter().enumerate() {
                assert_eq!(arena.root().visit_count, sims, "target_batch={target_batch} det {i}");
            }
        }
    }

    #[test]
    fn root_action_probs_argmax_at_tau_zero() {
        let s = playing_state(9);
        let mut arena = MctsArena::new(s.current_player);
        run_search(&mut arena, &s, &D, &D, 60, &MctsConfig::default());

        let probs = root_action_probs(&arena, 0.0);
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Exactly one non-zero entry at 1.0.
        let ones = probs.iter().filter(|(_, p)| *p == 1.0).count();
        assert_eq!(ones, 1);
    }

    #[test]
    fn greedy_ties_go_to_the_higher_prior_then_the_lower_index() {
        // Visits tie at 7 between indices 1, 2 and 3; index 2 has the
        // highest prior. Gen 1's `max_by_key` picked the last (3).
        let visits = [3u64, 7, 7, 7, 0];
        let priors = [0.1, 0.2, 0.4, 0.2, 0.1];
        assert_eq!(most_visited(&visits, &priors), 2);
        assert_eq!(visits_to_policy(&visits, &priors, 0.0), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        // Equal priors too: the lower index wins.
        assert_eq!(most_visited(&visits, &[0.2; 5]), 1);
        // A visit lead beats any prior.
        assert_eq!(most_visited(&[5u64, 4], &[0.0, 1.0]), 0);
    }

    #[test]
    fn root_action_probs_argmax_breaks_visit_ties_by_prior() {
        let mut arena = MctsArena::new(0);
        for (action, prior, visits) in [(0u8, 0.2f32, 5u32), (1, 0.5, 5), (2, 0.3, 5)] {
            let c = arena.alloc(prior, action);
            arena.node_mut(c).visit_count = visits;
            arena.node_mut(0).children.push(c);
        }
        let probs = root_action_probs(&arena, 0.0);
        assert_eq!(probs, vec![(0, 0.0), (1, 1.0), (2, 0.0)]);
    }

    #[test]
    fn mcts_search_reports_root_priors() {
        // DummyEvaluator priors are uniform over legal bids, so the averaged
        // root prior is too.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        let mask = bid_legal(&s);
        let r = mcts_search(&s, &D, &D, &cfg_with((2, 30), (1, 1)), &mut rng, 0);
        let n = mask.count_ones() as f32;
        for (b, &p) in r.root_prior.iter().enumerate() {
            let expected = if (mask >> b) & 1 == 1 { 1.0 / n } else { 0.0 };
            assert!((p - expected).abs() < 1e-6, "bid {b}: {p}");
        }
    }

    /// A Dirichlet(α, …, α) sample of length `n` must be non-negative and
    /// sum to ~1.
    #[test]
    fn sample_dirichlet_is_a_probability_vector() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD18E_C1E7);
        for &alpha in &[0.1f32, 0.3, 1.0, 3.0, 10.0] {
            for &n in &[2usize, 5, 14, 26] {
                let v = sample_dirichlet(&mut rng, alpha, n);
                assert_eq!(v.len(), n);
                let s: f32 = v.iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "α={alpha} n={n} sum={s}");
                for x in &v {
                    assert!(x.is_finite() && *x >= 0.0, "α={alpha} n={n} got {x}");
                }
            }
        }
    }

    /// After `apply_root_dirichlet_noise`, root child priors must
    /// (a) differ from the raw evaluator output by the expected mixing
    /// weight (`|P' − (1 − ε)·P| = ε · η`), and (b) still sum to 1.
    #[test]
    fn apply_root_dirichlet_noise_mixes_and_renormalizes() {
        let s = playing_state(11);
        let mut arena = MctsArena::new(s.current_player);
        expand(&mut arena, 0, &s, &D.policy(&s));

        let raw_priors: Vec<f32> = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).prior)
            .collect();
        let raw_sum: f32 = raw_priors.iter().sum();
        assert!((raw_sum - 1.0).abs() < 1e-4, "raw priors sum {raw_sum} ≠ 1");

        let epsilon = 0.25f32;
        let alpha = 0.3f32;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026_05_12);
        apply_root_dirichlet_noise(&mut arena, 0, alpha, epsilon, &mut rng);

        let mixed: Vec<f32> = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).prior)
            .collect();
        let mixed_sum: f32 = mixed.iter().sum();
        // (1−ε)·P sums to (1−ε) and ε·η sums to ε, so the mixed
        // distribution must still sum to 1 (within fp noise).
        assert!(
            (mixed_sum - 1.0).abs() < 1e-4,
            "mixed prior sum {mixed_sum} ≠ 1"
        );
        // Recover the noise vector and check it lies on the simplex.
        let mut noise: Vec<f32> = mixed
            .iter()
            .zip(raw_priors.iter())
            .map(|(m, p)| (m - (1.0 - epsilon) * p) / epsilon)
            .collect();
        let noise_sum: f32 = noise.iter().sum();
        assert!(
            (noise_sum - 1.0).abs() < 1e-3,
            "recovered noise sum {noise_sum} ≠ 1"
        );
        for n in noise.iter_mut() {
            assert!(*n > -1e-4 && *n < 1.0 + 1e-4, "noise out of [0,1]: {n}");
        }
    }

    /// Disabled (`epsilon == 0`) noise leaves priors untouched — protects
    /// the noise-free regime and the parity tests.
    #[test]
    fn apply_root_dirichlet_noise_is_noop_when_epsilon_zero() {
        let s = playing_state(17);
        let mut arena = MctsArena::new(s.current_player);
        expand(&mut arena, 0, &s, &D.policy(&s));
        let before: Vec<f32> = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).prior)
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        apply_root_dirichlet_noise(&mut arena, 0, 0.3, 0.0, &mut rng);
        let after: Vec<f32> = arena
            .root()
            .children
            .iter()
            .map(|&c| arena.node(c).prior)
            .collect();
        assert_eq!(before, after);
    }

    /// With `temperature = 0.1`, `policy_target` (held at τ=1) must have
    /// **higher entropy** than `policy_sampling` (computed at τ=0.1)
    /// for the same visit counts.
    #[test]
    fn mcts_search_policy_target_has_higher_entropy_than_sampling_at_low_tau() {
        let s = playing_state(41);
        let cfg = MctsConfig { temperature: 0.1, ..cfg_with((1, 1), (2, 60)) };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026_05_12);
        let r = mcts_search(&s, &D, &D, &cfg, &mut rng, 0);

        let nonzero_target = r.policy_target.iter().filter(|&&p| p > 0.0).count();
        assert!(
            nonzero_target >= 2,
            "test position must be multi-legal; got nonzero_target={nonzero_target}",
        );

        let h_target = entropy(&r.policy_target);
        let h_sampling = entropy(&r.policy_sampling);
        assert!(
            h_target > h_sampling,
            "policy_target entropy {h_target} should exceed policy_sampling entropy {h_sampling} at τ=0.1",
        );

        let s_t: f32 = r.policy_target.iter().sum();
        let s_s: f32 = r.policy_sampling.iter().sum();
        assert!((s_t - 1.0).abs() < 1e-4, "target sum {s_t}");
        assert!((s_s - 1.0).abs() < 1e-4, "sampling sum {s_s}");
    }

    /// At τ=1 the target and sampling distributions are equal
    /// bit-for-bit (the fast-path branch in `mcts_search`).
    #[test]
    fn mcts_search_policy_target_equals_sampling_at_tau_one() {
        let s = playing_state(83);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        let r = mcts_search(&s, &D, &D, &cfg_with((1, 1), (2, 30)), &mut rng, 0);
        assert_eq!(
            r.policy_target, r.policy_sampling,
            "τ=1 should yield identical target and sampling vectors",
        );
    }

    /// Terminal leaves back up the exact utilities for every seat; with
    /// V = 0 everywhere else, the root's per-seat sums are exactly the
    /// terminal utilities collected, so they sum to zero at λ = 1.
    #[test]
    fn terminal_leaves_back_up_exact_utilities() {
        // Last trick of a 4p/5c round: every remaining play is forced.
        let mut s = playing_state(31);
        while s.tricks_completed < 4 {
            let c = legal_plays(&s).trailing_zeros() as u8;
            apply_play(&mut s, c);
        }
        let mut end = s;
        while let Some(c) = forced_action(&end) {
            apply_play(&mut end, c);
        }
        let exact = terminal_utilities(&end, 1.0);
        let mut arena = MctsArena::new(s.current_player);
        run_search(&mut arena, &s, &D, &D, 5, &MctsConfig::default());
        let root = arena.root();
        assert_eq!(root.visit_count, 5);
        for seat in 0..4u8 {
            assert!((root.q(seat).unwrap() - exact[seat as usize]).abs() < 1e-6, "seat {seat}");
        }
        assert!(root.value_sums.iter().sum::<f32>().abs() < 1e-5);
    }

    /// When the descent reaches an unexpanded node whose state has
    /// exactly one legal action, the fast-path allocates a placeholder
    /// child inline with `prior = 1.0`, applies that action and keeps
    /// descending, rather than returning the unexpanded node as a leaf.
    ///
    /// Scenario: a 3P 0-card bidding state with `current_player = 0`,
    /// `dealer = 2`. Only bid 0 is legal for the two non-dealers, so the
    /// descent chains through both placeholders and stops at the dealer
    /// (no legal bid: neither forced nor expandable).
    #[test]
    fn select_leaf_state_takes_forced_fast_path() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 0;
        s.dealer = 2;
        s.current_player = 0;
        s.game_phase = GamePhase::Bidding as u8;
        assert_eq!(legal_bids(&s), 1);
        assert_eq!(forced_action(&s), Some(0));

        let mut arena = MctsArena::new(s.current_player);
        let (leaf_idx, path, _leaf_state) =
            select_leaf_state(&mut arena, &s, DEFAULT_C_PUCT);

        let root = arena.root();
        assert_eq!(root.children.len(), 1, "root should have one forced child");
        let forced_child = arena.node(root.children[0]);
        assert_eq!(forced_child.action, 0, "forced action should be bid 0");
        assert!(
            (forced_child.prior - 1.0).abs() < 1e-6,
            "forced placeholder prior should be 1.0, got {}",
            forced_child.prior,
        );
        assert!(path.len() >= 2, "descent didn't advance past root; path={:?}", path);
        assert_eq!(path[0], 0, "path should start at root");
        assert_ne!(leaf_idx, 0, "leaf should not be the root after fast-path");
    }

    /// Placeholders on a forced chain are visited by every simulation that
    /// passes through them, like any other node.
    #[test]
    fn forced_fast_path_credits_placeholders_along_path() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 0;
        s.dealer = 2;
        s.current_player = 0;
        s.game_phase = GamePhase::Bidding as u8;

        let mut arena = MctsArena::new(s.current_player);
        let sims = 20u32;
        run_search(&mut arena, &s, &D, &D, sims, &MctsConfig::default());

        assert_eq!(arena.root().visit_count, sims);
        let forced = arena.node(arena.root().children[0]);
        assert!(
            forced.visit_count >= sims - 1,
            "forced child undervisited: {} sims, child visit_count = {}",
            sims,
            forced.visit_count,
        );
    }

    #[test]
    fn mcts_search_with_root_noise_preserves_budget_and_policy() {
        let s = playing_state(23);
        let cfg = MctsConfig {
            root_dirichlet_alpha: 0.3,
            root_dirichlet_epsilon: 0.25,
            ..cfg_with((1, 1), (3, 25))
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026_05_12_01);
        let result = mcts_search(&s, &D, &D, &cfg, &mut rng, 0);
        let sum: f32 = result.policy_target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy_target sum {sum} ≠ 1");
        // The noise pre-step is each tree's first simulation.
        assert_eq!(result.total_visits, 3 * 24);
    }
}
