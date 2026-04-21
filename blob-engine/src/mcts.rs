//! Session 4.1 — Arena-allocated MCTS tree and UCB1 selection.
//!
//! Nodes are stored contiguously in `MctsArena::nodes`. Children are indices
//! into that vec, kept inline with `SmallVec<[u32; 14]>` (worst-case fan-out
//! is 14 bids; up to 13 plays).
//!
//! Per-player value storage is load-bearing for multiplayer: the network
//! evaluates a leaf from `state.current_player`'s perspective only, so a
//! single scalar would be diluted non-uniformly across subtrees. Each node
//! tracks `value_sums[seat]` and `value_counts[seat]` and UCB1 reads the
//! **acting** player's slot.
//!
//! Session 4.2 adds expansion, evaluation, backpropagation, and the
//! full search loop on top of the selection primitives above.
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
use crate::encoder::encode;
use crate::evaluator::{Evaluator, NUM_BIDS};
use crate::playing::{apply_play, legal_plays};
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// Default `c_puct` exploration constant. Tuned later in Section 6.
pub const DEFAULT_C_PUCT: f32 = 1.5;

/// Initial node capacity reserved per search. 10k nodes × ~80 B ≈ 800 KB.
pub const DEFAULT_ARENA_CAPACITY: usize = 10_000;

/// Single MCTS tree node.
///
/// - `visit_count`: total simulations that passed through this node. Used as
///   `N_parent`/`N_child` in the UCB1 exploration term.
/// - `value_sums` / `value_counts`: per-seat accumulators. Leaf value `v`
///   evaluated from seat `p` is added to `value_sums[p]` with
///   `value_counts[p] += 1` on every node along the path (Session 4.2).
///   `Q(acting) = value_sums[acting] / value_counts[acting]` with `Q = 0`
///   when that player has not yet been evaluated in this subtree.
/// - `prior`: network policy probability for the edge leading *into* this
///   node (populated at expansion).
/// - `action`: phase-stable edge label (see module docs). Root stores `0`.
/// - `children`: arena indices of child nodes; empty until expansion.
#[derive(Debug, Clone)]
pub struct MctsNode {
    pub visit_count: u32,
    pub value_sums: [f32; MAX_PLAYERS],
    pub value_counts: [u32; MAX_PLAYERS],
    pub prior: f32,
    pub action: u8,
    pub children: SmallVec<[u32; 14]>,
}

impl MctsNode {
    /// New unexpanded node. `prior` and `action` are edge-labelled at
    /// expansion; the root passes `prior = 1.0`, `action = 0`.
    #[inline]
    pub fn new(prior: f32, action: u8) -> Self {
        Self {
            visit_count: 0,
            value_sums: [0.0; MAX_PLAYERS],
            value_counts: [0; MAX_PLAYERS],
            prior,
            action,
            children: SmallVec::new(),
        }
    }

    /// Mean value from `seat`'s perspective, or `0.0` if that seat has not
    /// been evaluated in this subtree. See UCB1 docs.
    #[inline]
    pub fn q(&self, seat: u8) -> f32 {
        let n = self.value_counts[seat as usize];
        if n == 0 {
            0.0
        } else {
            self.value_sums[seat as usize] / n as f32
        }
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

/// UCB1 score for `child` under `parent` from the acting player's viewpoint.
///
/// `score = Q(acting) + c_puct * P * sqrt(N_parent) / (1 + N_child)`
///
/// An unvisited child (`visit_count == 0`) returns `f32::INFINITY` so it is
/// picked before any visited sibling, regardless of prior. This matches the
/// AlphaZero convention and avoids starving low-prior moves entirely.
#[inline]
pub fn ucb1_score(parent: &MctsNode, child: &MctsNode, acting: u8, c_puct: f32) -> f32 {
    if child.visit_count == 0 {
        return f32::INFINITY;
    }
    let q = child.q(acting);
    let n_parent = parent.visit_count.max(1) as f32;
    let explore = c_puct * child.prior * n_parent.sqrt() / (1.0 + child.visit_count as f32);
    q + explore
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
/// `acting_at` maps each depth (distance from root) to the seat that acts
/// at that node. Session 4.2 derives this by replaying actions on a scratch
/// `BlobState`; for now callers own that stepping.
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
/// children from those phases.
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

/// Expand `node_idx` by creating one child per legal action.
///
/// `policy` is the evaluator output for `state` (bidding: length
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
            let enc = encode(state, state.current_player);
            let legal = legal_plays(state);
            let mut new_children: SmallVec<[u32; 14]> = SmallVec::new();
            for (pos, &card_idx) in enc.hand_card_indices.iter().enumerate() {
                if (legal >> card_idx) & 1 == 1 {
                    let prior = policy.get(pos).copied().unwrap_or(0.0);
                    new_children.push(arena.alloc(prior, card_idx));
                }
            }
            arena.node_mut(node_idx).children = new_children;
        }
        GamePhase::Scoring | GamePhase::Complete => {}
    }
}

/// Backpropagate a leaf value `v` evaluated from seat `leaf_seat`'s
/// perspective along every node in `path` (root → leaf inclusive).
///
/// - `visit_count += 1` at every node (used by UCB1's exploration term).
/// - `value_sums[leaf_seat] += v`, `value_counts[leaf_seat] += 1` only in
///   that seat's slot. UCB1's `Q` averages per-seat, so other seats stay
///   undiluted. See module docs.
pub fn backprop(arena: &mut MctsArena, path: &[u32], leaf_seat: u8, v: f32) {
    for &idx in path {
        let node = arena.node_mut(idx);
        node.visit_count += 1;
        node.value_sums[leaf_seat as usize] += v;
        node.value_counts[leaf_seat as usize] += 1;
    }
}

/// Run `num_simulations` MCTS iterations against `root_state`.
///
/// Each iteration: walk from the root picking UCB1-best children (using
/// the acting seat stored in a replayed `BlobState`), evaluate the leaf
/// with `eval`, expand it, and backpropagate the value. Terminal leaves
/// skip expansion and backprop the evaluator's value as-is.
///
/// The initial root expansion is performed inside the first simulation
/// (path is just `[0]` when the tree is empty).
pub fn run_search<E: Evaluator + ?Sized>(
    arena: &mut MctsArena,
    root_state: &BlobState,
    eval: &E,
    num_simulations: u32,
    c_puct: f32,
) {
    for _ in 0..num_simulations {
        let mut state = *root_state;
        let mut path: Vec<u32> = Vec::with_capacity(16);
        let mut idx: u32 = 0;
        path.push(idx);

        // Walk from root, descending while expanded. The acting seat at
        // each step is `state.current_player` (the seat that will act
        // *from* this node), which UCB1 needs to read its Q slot from.
        while arena.node(idx).is_expanded() && !is_terminal(&state) {
            let acting = state.current_player;
            let child_idx = select_best_child(arena, idx, acting, c_puct);
            let action = arena.node(child_idx).action;
            apply_action(&mut state, action);
            idx = child_idx;
            path.push(idx);
        }

        // Evaluate the leaf (terminal phases short-circuit to v = 0).
        let (policy, value) = if is_terminal(&state) {
            (Vec::new(), 0.0)
        } else {
            eval.evaluate(&state)
        };
        let leaf_seat = state.current_player;

        if !is_terminal(&state) {
            expand(arena, idx, &state, &policy);
        }
        backprop(arena, &path, leaf_seat, value);
    }
}

/// Action probabilities over the root's children, sharpened/flattened by
/// temperature `tau`. Returns `(action, probability)` pairs in the order
/// children were allocated (phase-stable action labels).
///
/// - `tau == 1.0`: directly proportional to visit counts.
/// - `tau → 0`: approaches argmax on visit count (deterministic); the
///   implementation treats `tau < 1e-3` as argmax to avoid `f32::powf`
///   overflow.
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
        // Pick the first child with the max visit count. Ties break on
        // allocation order, matching `select_best_child`.
        let (mut best_i, mut best_n) = (0usize, 0u32);
        for (i, (_, n)) in visits.iter().enumerate() {
            if *n > best_n {
                best_n = *n;
                best_i = i;
            }
        }
        out[best_i].1 = 1.0;
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

/// Search-time configuration threaded through `mcts_search`.
///
/// Defaults match the plan's target budget (`5 × 100 = 500 sims`) with
/// a hard floor of 60 total simulations for any non-forced decision.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MctsConfig {
    pub c_puct: f32,
    /// Default determinizations per decision; `adaptive_budget` may
    /// raise this to satisfy per-branching-factor floors.
    pub num_determinizations: u32,
    /// Default simulations per determinization; `adaptive_budget` may
    /// raise this too.
    pub sims_per_determinization: u32,
    /// Hard absolute floor on `num_determinizations * sims_per_det`.
    /// Python post-mortem showed that starving MCTS produces no
    /// learning signal — this floor blocks that failure mode.
    pub min_sims_floor: u32,
    pub temperature: f32,
    pub arena_capacity: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: DEFAULT_C_PUCT,
            num_determinizations: 5,
            sims_per_determinization: 100,
            min_sims_floor: 60,
            temperature: 1.0,
            arena_capacity: DEFAULT_ARENA_CAPACITY,
        }
    }
}

/// Aggregated result of an `mcts_search` call.
///
/// `policy` is a dense vector indexed by the phase's canonical action
/// space: bids 0..14 in `Bidding`, hand-card positions (per
/// `EncodedState::hand_card_indices`) in `Playing`. `visit_entropy`,
/// `top1_visit_share`, and `value_estimate` are diagnostic signals used
/// by the training loop to calibrate the adaptive budget table.
#[derive(Debug, Clone)]
pub struct MctsResult {
    pub policy: Vec<f32>,
    pub visit_entropy: f32,
    pub top1_visit_share: f32,
    pub total_visits: u32,
    pub value_estimate: f32,
}

/// Pick per-decision `(num_determinizations, sims_per_determinization)`.
///
/// Branching-factor floors come from the development plan:
/// - `num_legal == 1`: `(1, 0)` — caller short-circuits, no search.
/// - `num_legal ≤ 3`: at least `3 × 20` = 60 sims.
/// - `num_legal ≤ 7`: at least `3 × 50` = 150 sims.
/// - `num_legal > 7`: at least `5 × 80` = 400 sims.
///
/// The config's values are used as the *baseline* — floors raise them
/// but never lower them. The absolute `min_sims_floor` is enforced on
/// the product `dets × sims` as a safety net.
pub fn adaptive_budget(num_legal: usize, cfg: &MctsConfig) -> (u32, u32) {
    if num_legal <= 1 {
        return (1, 0);
    }
    let (floor_dets, floor_sims) = if num_legal <= 3 {
        (3u32, 20u32)
    } else if num_legal <= 7 {
        (3, 50)
    } else {
        (5, 80)
    };
    let dets = cfg.num_determinizations.max(floor_dets);
    let mut sims = cfg.sims_per_determinization.max(floor_sims);

    // Absolute floor on the product: if dets×sims < min_sims_floor,
    // bump sims just enough. `dets` is already ≥ floor_dets, so this
    // only grows from here.
    let total = dets.saturating_mul(sims);
    if total < cfg.min_sims_floor {
        let needed = cfg.min_sims_floor.div_ceil(dets);
        sims = sims.max(needed);
    }
    (dets, sims)
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
/// delta function. Development plan target: `> 0.3`.
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

/// Full multi-determinization MCTS search with diagnostics.
///
/// Orchestrates Session 4.3: for each of `adaptive_budget`-selected
/// determinizations, sample opponent hands consistent with void
/// beliefs, run a fresh tree (Sessions 4.1 / 4.2), and aggregate root
/// visit counts into a single dense policy. Temperature, entropy, and
/// top-1 share come from the aggregate, not per-tree.
pub fn mcts_search<E, R>(
    state: &BlobState,
    eval: &E,
    cfg: &MctsConfig,
    rng: &mut R,
) -> MctsResult
where
    E: Evaluator + ?Sized,
    R: Rng + ?Sized,
{
    let phase = state.phase();
    if matches!(phase, GamePhase::Scoring | GamePhase::Complete) {
        return MctsResult {
            policy: Vec::new(),
            visit_entropy: 0.0,
            top1_visit_share: 0.0,
            total_visits: 0,
            value_estimate: 0.0,
        };
    }

    let perspective = state.current_player;

    // Canonical action space + forced-move detection.
    let (policy_len, hand_card_indices, num_legal, forced_action) = match phase {
        GamePhase::Bidding => {
            let mask = legal_bids(state);
            let n = mask.count_ones() as usize;
            let forced = if n == 1 {
                Some(mask.trailing_zeros() as u8)
            } else {
                None
            };
            (NUM_BIDS, SmallVec::<[u8; 13]>::new(), n, forced)
        }
        GamePhase::Playing => {
            let enc = encode(state, perspective);
            let legal = legal_plays(state);
            let n = legal.count_ones() as usize;
            let forced = if n == 1 {
                Some(legal.trailing_zeros() as u8)
            } else {
                None
            };
            (enc.hand_card_indices.len(), enc.hand_card_indices, n, forced)
        }
        _ => unreachable!(),
    };

    // Forced move: skip MCTS entirely. Signal ratio is 1 by convention.
    if num_legal == 1 {
        let mut policy = vec![0.0f32; policy_len];
        if let Some(action) = forced_action {
            if let Some(idx) = action_to_policy_index(phase, action, &hand_card_indices) {
                policy[idx] = 1.0;
            }
        }
        return MctsResult {
            policy,
            visit_entropy: 0.0,
            top1_visit_share: 1.0,
            total_visits: 0,
            value_estimate: 0.0,
        };
    }

    let (num_dets, sims_per) = adaptive_budget(num_legal, cfg);
    let voids = void_suits(state);

    let mut agg_visits = vec![0u64; policy_len];
    let mut total_visits: u32 = 0;
    let mut value_sum = 0.0f32;
    let mut value_n = 0u32;

    for _ in 0..num_dets {
        let det_state = determinize(state, perspective, &voids, rng, DEFAULT_DETERMINIZE_ATTEMPTS);
        let mut arena = MctsArena::with_capacity(perspective, cfg.arena_capacity);
        run_search(&mut arena, &det_state, eval, sims_per, cfg.c_puct);

        let root = arena.root();
        for &c in root.children.iter() {
            let child = arena.node(c);
            if let Some(idx) = action_to_policy_index(phase, child.action, &hand_card_indices) {
                agg_visits[idx] += child.visit_count as u64;
                total_visits = total_visits.saturating_add(child.visit_count);
            }
        }
        // Per-determinization root Q from the perspective seat. The
        // root is always acted on by `perspective`, so that slot is
        // the one UCB1 read during selection.
        if root.value_counts[perspective as usize] > 0 {
            value_sum += root.q(perspective);
            value_n += 1;
        }
    }

    // Temperature-applied policy over aggregated visits.
    let mut policy = vec![0.0f32; policy_len];
    let sum_visits: u64 = agg_visits.iter().sum();
    if sum_visits > 0 {
        if cfg.temperature < 1e-3 {
            let (best_i, _) = agg_visits
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| **v)
                .unwrap();
            policy[best_i] = 1.0;
        } else {
            let inv_tau = 1.0 / cfg.temperature;
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
        }
    }

    let visit_entropy = entropy(&policy);
    let top1_visit_share = policy.iter().cloned().fold(0.0f32, f32::max);
    let value_estimate = if value_n > 0 {
        value_sum / value_n as f32
    } else {
        0.0
    };

    MctsResult {
        policy,
        visit_entropy,
        top1_visit_share,
        total_visits,
        value_estimate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{apply_bid as bid_apply, legal_bids as bid_legal};
    use crate::dealing::deal;
    use crate::evaluator::DummyEvaluator;
    use crate::game::new_game;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

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
    fn q_defaults_to_zero_without_visits() {
        let node = MctsNode::new(0.2, 4);
        for seat in 0..MAX_PLAYERS as u8 {
            assert_eq!(node.q(seat), 0.0);
        }
    }

    #[test]
    fn q_averages_only_acting_players_slot() {
        let mut node = MctsNode::new(0.1, 0);
        node.visit_count = 5;
        // Seat 2 got three evaluations summing to 1.5 → mean 0.5.
        node.value_sums[2] = 1.5;
        node.value_counts[2] = 3;
        // Seat 4 got two evaluations summing to -0.6 → mean -0.3.
        node.value_sums[4] = -0.6;
        node.value_counts[4] = 2;

        assert!((node.q(2) - 0.5).abs() < 1e-6);
        assert!((node.q(4) + 0.3).abs() < 1e-6);
        assert_eq!(node.q(0), 0.0);
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
        child.value_sums[1] = 2.0;
        child.value_counts[1] = 4; // Q = 0.5 for seat 1.
        let c_puct = 1.5;
        let expected = 0.5 + c_puct * 0.25 * (16f32).sqrt() / (1.0 + 4.0);
        let got = ucb1_score(&parent, &child, 1, c_puct);
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn select_best_child_prefers_unvisited_then_higher_score() {
        let mut arena = MctsArena::new(0);
        // Two visited children with known Q, one unvisited.
        let a = arena.alloc(0.3, 1);
        let b = arena.alloc(0.3, 2);
        let c = arena.alloc(0.3, 3);
        {
            let na = arena.node_mut(a);
            na.visit_count = 4;
            na.value_sums[0] = 0.4;
            na.value_counts[0] = 4;
        }
        {
            let nb = arena.node_mut(b);
            nb.visit_count = 4;
            nb.value_sums[0] = 3.2;
            nb.value_counts[0] = 4; // much higher Q
        }
        // c left unvisited
        arena.node_mut(0).visit_count = 8;
        arena.node_mut(0).children.extend_from_slice(&[a, b, c]);

        // Unvisited `c` wins on infinite UCB1.
        let pick = select_best_child(&arena, 0, 0, DEFAULT_C_PUCT);
        assert_eq!(pick, c);

        // Give c a visit; b (higher Q) should now win over a.
        {
            let nc = arena.node_mut(c);
            nc.visit_count = 1;
            nc.value_sums[0] = 0.0;
            nc.value_counts[0] = 1;
        }
        let pick = select_best_child(&arena, 0, 0, DEFAULT_C_PUCT);
        assert_eq!(pick, b);
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
        let (policy, _) = DummyEvaluator.evaluate(&s);
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
        let (policy, _) = DummyEvaluator.evaluate(&s);
        expand(&mut arena, 0, &s, &policy);

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
        let (policy, _) = DummyEvaluator.evaluate(&s);
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
    fn backprop_updates_path_and_only_leaf_seat_slot() {
        let mut arena = MctsArena::new(0);
        let c1 = arena.alloc(0.5, 1);
        let c2 = arena.alloc(0.5, 2);
        arena.node_mut(0).children.extend_from_slice(&[c1, c2]);

        backprop(&mut arena, &[0, c1], 3, 0.75);
        assert_eq!(arena.node(0).visit_count, 1);
        assert_eq!(arena.node(c1).visit_count, 1);
        assert_eq!(arena.node(c2).visit_count, 0);
        assert!((arena.node(0).value_sums[3] - 0.75).abs() < 1e-6);
        assert_eq!(arena.node(0).value_counts[3], 1);
        for seat in 0..MAX_PLAYERS {
            if seat != 3 {
                assert_eq!(arena.node(0).value_counts[seat], 0);
                assert_eq!(arena.node(0).value_sums[seat], 0.0);
            }
        }
    }

    #[test]
    fn run_search_visits_all_legal_actions_and_sums_correctly() {
        let s = playing_state(42);
        let mut arena = MctsArena::new(s.current_player);
        let sims = 100u32;
        run_search(&mut arena, &s, &DummyEvaluator, sims, DEFAULT_C_PUCT);

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
        run_search(&mut arena, &s, &DummyEvaluator, 80, DEFAULT_C_PUCT);

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

    #[test]
    fn adaptive_budget_respects_branching_floors() {
        let cfg = MctsConfig {
            num_determinizations: 1,
            sims_per_determinization: 1,
            min_sims_floor: 60,
            ..MctsConfig::default()
        };
        // Forced move → (1, 0).
        assert_eq!(adaptive_budget(1, &cfg), (1, 0));
        // Small branching → at least 3×20.
        let (d, s) = adaptive_budget(2, &cfg);
        assert!(d >= 3 && s >= 20 && d * s >= 60);
        // Mid branching → at least 3×50.
        let (d, s) = adaptive_budget(5, &cfg);
        assert!(d >= 3 && s >= 50);
        // Large branching → at least 5×80.
        let (d, s) = adaptive_budget(10, &cfg);
        assert!(d >= 5 && s >= 80);
        // Config values take over when higher than floor.
        let cfg2 = MctsConfig {
            num_determinizations: 8,
            sims_per_determinization: 200,
            ..MctsConfig::default()
        };
        assert_eq!(adaptive_budget(10, &cfg2), (8, 200));
    }

    #[test]
    fn mcts_search_forced_move_shortcut() {
        // Build a state where only one bid is legal. Easiest: 0-card round
        // not representable; instead craft a small synthetic bidding state
        // where cards_dealt=0 forces bid=0 (mask = 0b1).
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 0;
        s.dealer = 2;
        s.current_player = 0;
        s.game_phase = GamePhase::Bidding as u8;
        // legal_bids returns bits 0..=0 with dealer forbidden check; since
        // current_player != dealer, mask = 1.
        assert_eq!(legal_bids(&s), 1);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let cfg = MctsConfig::default();
        let r = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng);
        assert_eq!(r.policy.len(), NUM_BIDS);
        assert!((r.policy[0] - 1.0).abs() < 1e-6);
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

        let cfg = MctsConfig {
            num_determinizations: 2,
            sims_per_determinization: 40,
            min_sims_floor: 60,
            ..MctsConfig::default()
        };
        let r = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng);
        assert_eq!(r.policy.len(), NUM_BIDS);
        let sum: f32 = r.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");

        // Every legal bid has nonzero probability, illegal bids are zero.
        let mask = legal_bids(&s);
        let mut legal_nonzero = 0usize;
        for b in 0..NUM_BIDS {
            if (mask >> b) & 1 == 1 {
                assert!(r.policy[b] > 0.0, "legal bid {b} has zero policy");
                legal_nonzero += 1;
            } else {
                assert_eq!(r.policy[b], 0.0, "illegal bid {b} has nonzero policy");
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
        let cfg = MctsConfig {
            num_determinizations: 2,
            sims_per_determinization: 30,
            min_sims_floor: 60,
            ..MctsConfig::default()
        };
        let r = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng);
        assert_eq!(r.policy.len(), enc.hand_card_indices.len());
        let sum: f32 = r.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");

        for (pos, &ci) in enc.hand_card_indices.iter().enumerate() {
            let legal_card = (legal >> ci) & 1 == 1;
            if legal_card {
                assert!(r.policy[pos] > 0.0, "legal pos {pos} has zero policy");
            } else {
                assert_eq!(r.policy[pos], 0.0, "illegal pos {pos} has nonzero policy");
            }
        }
    }

    #[test]
    fn signal_ratio_zero_for_uniform_policy() {
        let r = MctsResult {
            policy: vec![0.25, 0.25, 0.25, 0.25],
            visit_entropy: (4f32).ln(),
            top1_visit_share: 0.25,
            total_visits: 40,
            value_estimate: 0.0,
        };
        let sr = signal_ratio(&r, 4);
        assert!(sr.abs() < 1e-6, "sr={sr}");
    }

    #[test]
    fn root_action_probs_argmax_at_tau_zero() {
        let s = playing_state(9);
        let mut arena = MctsArena::new(s.current_player);
        run_search(&mut arena, &s, &DummyEvaluator, 60, DEFAULT_C_PUCT);

        let probs = root_action_probs(&arena, 0.0);
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Exactly one non-zero entry at 1.0.
        let ones = probs.iter().filter(|(_, p)| *p == 1.0).count();
        assert_eq!(ones, 1);
    }
}
