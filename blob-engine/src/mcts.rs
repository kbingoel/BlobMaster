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

/// Default leaves-per-`evaluate_batch` target for the lockstep driver.
/// Set to `num_determinizations` (5) — the 2026-04-27 sweep
/// (`{5, 8, 12, 16}` at T=32, see [self-play-profile.md]) showed this is
/// the per-game-wall optimum on the 7950X / 1.63M-param transformer; per-
/// call ONNX cost rises super-linearly past `num_dets` because the CPU
/// is already saturated by 32 concurrent batched forwards. At
/// `target_batch == num_determinizations` the round-robin keeps every
/// path's `in_flight` ≤ 1, so the virtual-loss term in UCB1 is dormant
/// and the search behaves as Stage-1 cross-determinization batching.
/// Raising past `num_dets` is parked behind a model-size revisit
/// (d_model ≥ 256 would tip the GEMM regime to genuinely batch-bound).
pub const DEFAULT_TARGET_BATCH: usize = 5;

/// Virtual-loss weight (Session 7.4c stage 2). Each in-flight leaf along
/// a path subtracts this from `value_sums[acting]` during UCB1 selection,
/// so concurrent descents in the same det's tree pick different leaves.
/// 1.0 is the standard AlphaZero choice (assume in-flight = loss).
pub const VIRTUAL_LOSS_WEIGHT: f32 = 1.0;

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
/// - `in_flight`: number of currently-pending descents that hold this
///   node on their path (Session 7.4c stage 2). Bumped along the path
///   when a leaf is queued for a batched `evaluate` call and decremented
///   right before that leaf's `expand`/`backprop` runs. UCB1 reads it as
///   a temporary pessimistic visit so concurrent descents inside the
///   same tree pick different leaves. Single-thread mutation only — one
///   rayon worker owns the arena, no atomic needed. `u16` is enough to
///   cover any plausible `target_batch` (default 8, plan limit 16).
#[derive(Debug, Clone)]
pub struct MctsNode {
    pub visit_count: u32,
    pub value_sums: [f32; MAX_PLAYERS],
    pub value_counts: [u32; MAX_PLAYERS],
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
            value_counts: [0; MAX_PLAYERS],
            prior,
            action,
            children: SmallVec::new(),
            in_flight: 0,
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

/// UCB1 score for `child` under `parent` from the acting player's viewpoint,
/// including virtual-loss decoration from in-flight descents.
///
/// Without any in-flight leaves (`child.in_flight == 0`) this is the standard
/// AlphaZero score:
///
/// `score = Q(acting) + c_puct * P * sqrt(N_parent) / (1 + N_child)`
///
/// where `Q(acting) = value_sums[acting] / value_counts[acting]` and an
/// unvisited child (`visit_count == 0`) returns `f32::INFINITY` so it is
/// picked before any visited sibling.
///
/// When `child.in_flight > 0` (Session 7.4c stage 2), virtual loss kicks in:
/// the child's effective denominator becomes `value_counts[acting] +
/// in_flight`, the numerator subtracts `VIRTUAL_LOSS_WEIGHT * in_flight`
/// (treat each pending leaf as a loss until the real value lands), and the
/// exploration term divides by `1 + visit_count + in_flight` so paths that
/// already have queued descents look more thoroughly explored. This
/// degenerates exactly to the no-VL formula at `in_flight == 0`, so callers
/// that never decorate paths (`run_search`, the parity tests) keep their
/// existing behavior bit-for-bit.
#[inline]
pub fn ucb1_score(parent: &MctsNode, child: &MctsNode, acting: u8, c_puct: f32) -> f32 {
    let in_flight = child.in_flight as u32;
    if child.visit_count == 0 && in_flight == 0 {
        return f32::INFINITY;
    }
    let value_n = child.value_counts[acting as usize] + in_flight;
    let q_eff = if value_n == 0 {
        // Visited via path-throughs but never as the acting seat's leaf,
        // and not in-flight: fall back to Q = 0 (existing `q()` semantics).
        0.0
    } else {
        let vloss = VIRTUAL_LOSS_WEIGHT * in_flight as f32;
        (child.value_sums[acting as usize] - vloss) / value_n as f32
    };
    let n_parent = parent.visit_count.max(1) as f32;
    let n_child_eff = child.visit_count + in_flight;
    let explore = c_puct * child.prior * n_parent.sqrt() / (1.0 + n_child_eff as f32);
    q_eff + explore
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
    })
}

/// Backpropagate a leaf value `v` evaluated from seat `leaf_seat`'s
/// perspective along every node in `path` (root → leaf inclusive).
///
/// - `visit_count += 1` at every node (used by UCB1's exploration term).
/// - `value_sums[leaf_seat] += v`, `value_counts[leaf_seat] += 1` only in
///   that seat's slot. UCB1's `Q` averages per-seat, so other seats stay
///   undiluted. See module docs.
pub fn backprop(arena: &mut MctsArena, path: &[u32], leaf_seat: u8, v: f32) {
    crate::profiling::time(&crate::profiling::BACKPROP, || {
        for &idx in path {
            let node = arena.node_mut(idx);
            node.visit_count += 1;
            node.value_sums[leaf_seat as usize] += v;
            node.value_counts[leaf_seat as usize] += 1;
        }
    })
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

/// Walk from the root of `arena` along UCB1-best children, replaying
/// actions on a clone of `root_state`, until reaching either an unexpanded
/// node or a terminal state. Returns `(leaf_idx, path, leaf_state)` where
/// `path` includes both endpoints (root and leaf inclusive). Mirrors the
/// per-sim descent inside `run_search` exactly so callers can round-trip
/// through batched evaluation without changing semantics.
pub fn select_leaf_state(
    arena: &MctsArena,
    root_state: &BlobState,
    c_puct: f32,
) -> (u32, Vec<u32>, BlobState) {
    let mut state = *root_state;
    let mut path: Vec<u32> = Vec::with_capacity(16);
    let mut idx: u32 = 0;
    path.push(idx);
    while arena.node(idx).is_expanded() && !is_terminal(&state) {
        let acting = state.current_player;
        let child_idx = select_best_child(arena, idx, acting, c_puct);
        let action = arena.node(child_idx).action;
        apply_action(&mut state, action);
        idx = child_idx;
        path.push(idx);
    }
    (idx, path, state)
}

/// Generalized lockstep search across multiple determinization trees with
/// a configurable per-`evaluate_batch` target leaf count.
///
/// Stage 1 (Session 7.4c) batched leaves *across* dets only: one descent per
/// det per step, batch size capped at `num_dets`. Stage 2 (this function)
/// raises the cap to `target_batch`, queueing additional descents *within*
/// the same det's tree behind a virtual-loss decoration so concurrent
/// descents pick different leaves.
///
/// Driver loop:
///
/// 1. Round-robin pick the not-yet-exhausted det with the fewest
///    sims-so-far. Tie break on lowest det index, so with
///    `target_batch >= num_dets` the first batch fills as
///    `[det 0, det 1, …, det num_dets-1, det 0, det 1, …]` — the same
///    order Stage 1 used.
/// 2. Walk root → unexpanded leaf using the standard UCB1 scorer, which
///    already reads `MctsNode::in_flight` so descents already in this
///    batch bias subsequent descents away from their paths.
/// 3. Terminal leaves backprop `v=0` immediately (no eval, no
///    `in_flight` decoration), matching `run_search`.
/// 4. Non-terminal leaves: increment `in_flight` along the path and push
///    onto the pending batch.
/// 5. **Cold-start duplicate guard.** If a fresh descent lands on a leaf
///    whose `in_flight > 0` (only possible while a det's root is still
///    unexpanded — virtual loss can't redirect *through* an unexpanded
///    node), mark that det blocked for this batch and try the next. The
///    blocked det is unblocked once eval runs and its root is expanded.
/// 6. When `pending.len() == target_batch` (or every eligible det is
///    exhausted/blocked), call `evaluate_batch`, decrement `in_flight`
///    along each pending path, expand, and backprop in queue order.
///
/// **Special cases for callers / tests:**
/// - `target_batch = 1`: only one descent in flight at a time, so
///   virtual loss never engages and the per-det node sequence matches
///   `run_search` bit-for-bit on the same inputs (pinned by
///   `target_batch_one_matches_serial_per_det`).
/// - `target_batch = num_dets`: at most one descent per det per outer
///   iteration, so virtual loss again never engages and behavior matches
///   the pre-stage-2 cross-det driver bit-for-bit (pinned by
///   `lockstep_search_matches_serial_per_det`).
/// - `target_batch > num_dets`: virtual loss biases concurrent descents
///   inside the same det's tree away from each other. Visit-count
///   distributions on identical inputs will *not* match serial MCTS — the
///   policy target is the visit distribution either way and softmax
///   absorbs small biases (plan §7.4c stage 2).
///
/// Post-condition: every node in every arena has `in_flight == 0` (every
/// path that incremented it ran the matching decrement before backprop).
/// `debug_assert!`ed at the end so a future bug in path bookkeeping fails
/// loudly in tests.
pub fn run_lockstep_search<E: Evaluator + ?Sized>(
    arenas: &mut [MctsArena],
    root_states: &[BlobState],
    eval: &E,
    num_simulations: u32,
    c_puct: f32,
    target_batch: usize,
) {
    debug_assert_eq!(arenas.len(), root_states.len());
    let num_dets = arenas.len();
    if num_dets == 0 || num_simulations == 0 {
        return;
    }
    let target_batch = target_batch.max(1);

    struct Pending {
        det: usize,
        leaf_idx: u32,
        path: Vec<u32>,
        leaf_state: BlobState,
        leaf_seat: u8,
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
        loop {
            if pending.len() >= target_batch {
                break;
            }
            // Round-robin: pick the lowest-sims_done det that's neither
            // exhausted nor blocked-this-batch. `min_by_key` ties on the
            // first match, which gives det-index-ascending order at any
            // tied sims count — preserving the legacy Stage 1 fill order.
            let next_det = (0..num_dets)
                .filter(|&d| sims_done[d] < num_simulations && !blocked[d])
                .min_by_key(|&d| sims_done[d]);
            let Some(det) = next_det else {
                break;
            };

            let (leaf_idx, path, leaf_state) =
                select_leaf_state(&arenas[det], &root_states[det], c_puct);
            let leaf_seat = leaf_state.current_player;

            if is_terminal(&leaf_state) {
                // Terminal leaves never need eval or expand; backprop the
                // canonical v=0 immediately, matching `run_search`. No
                // `in_flight` decoration since nothing is queued.
                backprop(&mut arenas[det], &path, leaf_seat, 0.0);
                sims_done[det] += 1;
                continue;
            }

            // Cold-start duplicate: an unexpanded root is reachable
            // through itself (the descent loop bails at the first
            // unexpanded node), so a second descent into the same det
            // before its root expansion lands on the same leaf. UCB1's
            // virtual-loss bias only redirects *between expanded
            // siblings*, so the only fix is to skip this det until the
            // pending batch flushes and its root is expanded. (Once the
            // first batch returns, every root is expanded and the
            // descent loop can use VL bias to diverge.)
            if arenas[det].node(leaf_idx).in_flight > 0 {
                blocked[det] = true;
                continue;
            }

            for &n in &path {
                arenas[det].node_mut(n).in_flight += 1;
            }
            pending.push(Pending {
                det,
                leaf_idx,
                path,
                leaf_state,
                leaf_seat,
            });
            sims_done[det] += 1;
        }

        if pending.is_empty() {
            // Either every det is exhausted (outer `break` next iter) or
            // every fill attempt this round resolved to a terminal leaf
            // (sims_done already advanced by those). Either way, no eval
            // call needed.
            continue;
        }

        let states_ref: Vec<&BlobState> = pending.iter().map(|p| &p.leaf_state).collect();
        let results = eval.evaluate_batch(&states_ref);
        debug_assert_eq!(results.len(), pending.len());

        for (p, (policy, value)) in pending.drain(..).zip(results.into_iter()) {
            // Decrement in_flight *before* expand+backprop so the real
            // visit replaces the virtual one cleanly. Order matters: the
            // arenas being mutated are owned single-threaded so there is
            // no race, but a future reader stepping through expansion
            // order should see consistent counters.
            for &n in &p.path {
                arenas[p.det].node_mut(n).in_flight -= 1;
            }
            expand(&mut arenas[p.det], p.leaf_idx, &p.leaf_state, &policy);
            backprop(&mut arenas[p.det], &p.path, p.leaf_seat, value);
        }
    }

    // Sanity: every virtual visit must have a matching decrement.
    debug_assert!(
        arenas
            .iter()
            .all(|a| a.nodes.iter().all(|n| n.in_flight == 0)),
        "lockstep search left non-zero in_flight on some node",
    );
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

fn default_target_batch() -> usize {
    DEFAULT_TARGET_BATCH
}

fn default_root_dirichlet_alpha() -> f32 {
    0.0
}

fn default_root_dirichlet_epsilon() -> f32 {
    0.0
}

/// Sample a single `Gamma(alpha, 1)` variate via Marsaglia–Tsang for
/// `alpha >= 1`, with the standard boost trick
/// (`G(alpha) ≡ G(alpha+1) · U^(1/alpha)`) for `alpha < 1`. Used by
/// `sample_dirichlet` for root-prior noise (Step 1 of fix-mcts-plan.md).
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
/// for the root only (Step 1 of fix-mcts-plan.md).
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

/// Per-decision temperature schedule (Session 7.4d). When set, the
/// effective τ used by `mcts_search` to convert root visit counts into
/// `MctsResult.policy` depends on the global decision index within the
/// game (one increment per `mcts_search` call, covering both bid and
/// play decisions of every seat).
///
/// Note: `MctsResult.policy` is fused — the same vector serves both the
/// training target and the action-sampling distribution in self-play.
/// At τ → 0 the late-game training target collapses to one-hot on the
/// argmax-visit action, which is intentional: late-game positions are
/// where MCTS visit counts are highest-signal and we want the policy
/// head to commit. If you need a τ=1 training target with τ→0 sampling
/// (canonical AlphaZero), split `MctsResult` into separate `policy_target`
/// and `policy_sampling` fields — out of scope for 7.4d.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemperatureSchedule {
    /// τ = `early` for `decision_index < switch_at`, otherwise `late`.
    /// Dev-plan §7.4d default proposal: `early = 1.0`, `late = 0.1`,
    /// `switch_at = 15`. AlphaZero-style hard step.
    HardStep {
        early: f32,
        late: f32,
        switch_at: usize,
    },
}

impl TemperatureSchedule {
    /// Resolve the effective τ for a given global decision index.
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
    /// Constant temperature, used when `temperature_schedule` is `None`.
    /// Pre-7.4d call sites continue to read this directly via
    /// `MctsConfig::temperature_at`.
    pub temperature: f32,
    /// Optional per-decision schedule (Session 7.4d). When `Some`,
    /// overrides `temperature` and `mcts_search` resolves τ from the
    /// schedule using the `decision_index` argument.
    /// `#[serde(default)]` so existing TOMLs without this field keep
    /// working.
    #[serde(default)]
    pub temperature_schedule: Option<TemperatureSchedule>,
    pub arena_capacity: usize,
    /// Session 7.4c stage 2: target leaves per batched `evaluate` call.
    /// The lockstep driver fills each batch round-robin from all dets,
    /// using virtual loss to redirect concurrent descents inside the
    /// same det's tree away from each other. `1` degenerates to fully
    /// serial MCTS (one leaf per eval, no VL bias possible — pinned by
    /// `target_batch_one_matches_serial_per_det`). `num_determinizations`
    /// reproduces stage 1 (cross-det only). Plan suggests 5..=16; 8 is
    /// the default. `#[serde(default)]` so existing TOMLs without this
    /// field keep working.
    #[serde(default = "default_target_batch")]
    pub target_batch: usize,
    /// Dirichlet concentration α used to inject exploration noise into
    /// root priors before search (fix-mcts-plan.md Step 1 / C1). When
    /// `<= 0` the heuristic `α = 10 / num_legal` is used per call —
    /// DeepMind's scaling rule, more robust across Blob's variable
    /// branching (5 plays … 14 bids). A fixed value (e.g. 0.3) is also
    /// supported. Mixing weight is controlled by
    /// `root_dirichlet_epsilon`; noise is fully disabled whenever
    /// `epsilon <= 0`, so leaving this at the default with epsilon=0
    /// keeps the pre-7.5 behavior.
    #[serde(default = "default_root_dirichlet_alpha")]
    pub root_dirichlet_alpha: f32,
    /// Mixing weight ε for root Dirichlet noise:
    /// `P'(a) = (1 − ε) · P(a) + ε · η(a)`. AlphaZero default is 0.25;
    /// any value `<= 0` disables noise injection entirely. Per-det
    /// (each determinization tree gets its own η sample).
    #[serde(default = "default_root_dirichlet_epsilon")]
    pub root_dirichlet_epsilon: f32,
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
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: DEFAULT_C_PUCT,
            num_determinizations: 5,
            sims_per_determinization: 100,
            min_sims_floor: 60,
            temperature: 1.0,
            temperature_schedule: None,
            arena_capacity: DEFAULT_ARENA_CAPACITY,
            target_batch: DEFAULT_TARGET_BATCH,
            root_dirichlet_alpha: default_root_dirichlet_alpha(),
            root_dirichlet_epsilon: default_root_dirichlet_epsilon(),
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
/// Phase-A baseline (Session 7.3c): flat `5 × 100 = 500` sims for every
/// non-forced decision, matching the 7.2 run that reached 0.77 eval win
/// rate at iter 10. The 7.3a bucketed schedule starved low-branching
/// decisions (60 / 90 sims at `nl ∈ {2, 3}`) — see `7.3b-analysis.md`
/// §5 and §7.1. Forced moves (`num_legal ≤ 1`) still short-circuit with
/// `(1, 0)` so `mcts_search` can skip the tree.
///
/// `min_sims_floor` remains as a safety net in case a future change
/// lowers these numbers.
pub fn adaptive_budget(num_legal: usize, cfg: &MctsConfig) -> (u32, u32) {
    if num_legal <= 1 {
        return (1, 0);
    }
    let (dets, mut sims) = (5u32, 100u32);

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
    decision_index: usize,
) -> MctsResult
where
    E: Evaluator + ?Sized,
    R: Rng + ?Sized,
{
    crate::profiling::time(&crate::profiling::MCTS_SEARCH, || {
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

        // Session 7.4c stage-2: target_batch lockstep with virtual loss.
        //
        // Allocate every det's state and arena up front, then drive in
        // lockstep — each batch fills up to `cfg.target_batch` leaves
        // round-robin across dets, using virtual loss to redirect
        // multiple descents inside the same det's tree away from each
        // other. Terminal leaves are backpropagated immediately with v=0
        // (no expand needed, no in_flight decoration). At
        // `target_batch == cfg.num_determinizations` this degenerates to
        // stage 1 (one descent per det per batch, no VL engagement); at
        // `target_batch == 1` it degenerates to fully serial MCTS.
        let mut det_states: Vec<BlobState> = Vec::with_capacity(num_dets as usize);
        let mut arenas: Vec<MctsArena> = Vec::with_capacity(num_dets as usize);
        for _ in 0..num_dets {
            let det_state =
                determinize(state, perspective, &voids, rng, DEFAULT_DETERMINIZE_ATTEMPTS);
            det_states.push(det_state);
            arenas.push(MctsArena::with_capacity(perspective, cfg.arena_capacity));
        }

        // Root Dirichlet noise (fix-mcts-plan.md Step 1 / C1): when
        // enabled, pre-expand each det's root via a single batched eval
        // call so we can decorate the network priors with
        // `(1−ε)·P + ε·Dir(α)` before any UCB1 selection runs. The
        // pre-step accounts for one simulation per det (matches what
        // the first lockstep descent would have done anyway), so we
        // pass `sims_per - 1` to `run_lockstep_search` to keep the
        // total sim budget unchanged. Disabled when `epsilon <= 0` or
        // `sims_per == 0` (forced-move branch is already short-circuited
        // above, so this safeguards adaptive_budget edge cases).
        let noise_on = cfg.root_dirichlet_epsilon > 0.0 && sims_per > 0;
        let effective_sims = if noise_on {
            // Resolve α: heuristic (10 / num_legal) when config sets a
            // non-positive sentinel, fixed value otherwise.
            let alpha = if cfg.root_dirichlet_alpha > 0.0 {
                cfg.root_dirichlet_alpha
            } else {
                (10.0 / num_legal.max(1) as f32).max(1e-3)
            };
            let states_ref: Vec<&BlobState> = det_states.iter().collect();
            let results = eval.evaluate_batch(&states_ref);
            debug_assert_eq!(results.len(), det_states.len());
            for (det_idx, (policy, value)) in results.into_iter().enumerate() {
                expand(&mut arenas[det_idx], 0, &det_states[det_idx], &policy);
                apply_root_dirichlet_noise(
                    &mut arenas[det_idx],
                    0,
                    alpha,
                    cfg.root_dirichlet_epsilon,
                    rng,
                );
                // Root visit is acted on by `perspective`; mirrors what
                // `run_search`/`run_lockstep_search` would record on the
                // first descent that hits the unexpanded root.
                backprop(&mut arenas[det_idx], &[0], perspective, value);
            }
            sims_per - 1
        } else {
            sims_per
        };

        run_lockstep_search(
            &mut arenas,
            &det_states,
            eval,
            effective_sims,
            cfg.c_puct,
            cfg.target_batch,
        );

        let mut agg_visits = vec![0u64; policy_len];
        let mut total_visits: u32 = 0;
        let mut value_sum = 0.0f32;
        let mut value_n = 0u32;

        for arena in &arenas {
            let root = arena.root();
            for &c in root.children.iter() {
                let child = arena.node(c);
                if let Some(idx) =
                    action_to_policy_index(phase, child.action, &hand_card_indices)
                {
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
        let tau = cfg.temperature_at(decision_index);
        let mut policy = vec![0.0f32; policy_len];
        let sum_visits: u64 = agg_visits.iter().sum();
        if sum_visits > 0 {
            if tau < 1e-3 {
                let (best_i, _) = agg_visits
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, v)| **v)
                    .unwrap();
                policy[best_i] = 1.0;
            } else {
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
    })
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
    fn adaptive_budget_is_flat_5x100() {
        let cfg = MctsConfig {
            num_determinizations: 1,
            sims_per_determinization: 1,
            min_sims_floor: 60,
            ..MctsConfig::default()
        };
        // Forced move → (1, 0).
        assert_eq!(adaptive_budget(1, &cfg), (1, 0));
        // Every non-forced branching factor → flat (5, 100) per Phase-A.
        for n in 2..=25 {
            assert_eq!(adaptive_budget(n, &cfg), (5, 100), "nl={n}");
        }
        // A larger `min_sims_floor` can still raise sims above 100.
        let cfg2 = MctsConfig {
            min_sims_floor: 750,
            ..cfg
        };
        let (dets, sims) = adaptive_budget(4, &cfg2);
        assert_eq!(dets, 5);
        assert!(dets * sims >= 750, "floor not enforced: {dets}×{sims}");
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
        let r = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng, 0);
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
        let r = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng, 0);
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
        let r = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng, 0);
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

    /// Late-game τ→0 with a hard-step schedule must collapse the policy to
    /// one-hot on the argmax-visit action; early-game τ=1 must spread mass
    /// across all visited children. Same state, same seed, two different
    /// `decision_index` arguments — verifies the schedule actually wires
    /// through `mcts_search`.
    #[test]
    fn mcts_search_honors_temperature_schedule() {
        let s = playing_state(123);
        let cfg = MctsConfig {
            num_determinizations: 2,
            sims_per_determinization: 60,
            min_sims_floor: 60,
            temperature: 1.0,
            temperature_schedule: Some(TemperatureSchedule::HardStep {
                early: 1.0,
                late: 0.0,
                switch_at: 15,
            }),
            ..MctsConfig::default()
        };

        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(7);
        let r_early = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng_a, 0);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(7);
        let r_late = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng_b, 50);

        // Early: τ=1 → at least two non-zero entries (some spread).
        let nonzero_early = r_early.policy.iter().filter(|&&p| p > 0.0).count();
        assert!(
            nonzero_early >= 2,
            "early τ=1 should spread mass; nonzero={nonzero_early}"
        );

        // Late: τ→0 → exactly one entry == 1.0, rest == 0.0.
        let max_late = r_late.policy.iter().cloned().fold(0.0f32, f32::max);
        assert!((max_late - 1.0).abs() < 1e-6, "late argmax mass={max_late}");
        let nonzero_late = r_late.policy.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero_late, 1, "late τ→0 must be one-hot");
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

    /// Session 7.4c stage-1 parity: lockstep batching across multiple
    /// independent dets must reproduce the per-det visit-count distribution
    /// and per-seat value accumulators of the serial driver bit-for-bit.
    ///
    /// We run `run_search` on each `(arena, root_state)` independently and
    /// compare against `run_lockstep_search` on the same inputs. The
    /// `DummyEvaluator` makes `evaluate` and `evaluate_batch` produce
    /// identical output (the trait's default `evaluate_batch` just loops
    /// `evaluate`), so any divergence would indicate a logic bug in the
    /// lockstep driver, not numerical noise from a real ONNX session.
    #[test]
    fn lockstep_search_matches_serial_per_det() {
        let states = [
            playing_state(101),
            playing_state(202),
            playing_state(303),
        ];
        let sims = 80u32;

        // Serial baseline.
        let mut serial_arenas: Vec<MctsArena> = states
            .iter()
            .map(|s| MctsArena::new(s.current_player))
            .collect();
        for (arena, state) in serial_arenas.iter_mut().zip(states.iter()) {
            run_search(arena, state, &DummyEvaluator, sims, DEFAULT_C_PUCT);
        }

        // Lockstep driver. target_batch = num_dets keeps the driver in
        // pure cross-det mode (Stage 1) — at most one descent per det per
        // outer iteration, so virtual loss never engages and per-det
        // node sequences match `run_search` bit-for-bit.
        let mut lockstep_arenas: Vec<MctsArena> = states
            .iter()
            .map(|s| MctsArena::new(s.current_player))
            .collect();
        let states_vec: Vec<BlobState> = states.to_vec();
        run_lockstep_search(
            &mut lockstep_arenas,
            &states_vec,
            &DummyEvaluator,
            sims,
            DEFAULT_C_PUCT,
            states.len(),
        );

        // Each det's arena must match: same node count, same per-node
        // visit counts, value sums, and child structure.
        for (i, (a, b)) in serial_arenas.iter().zip(lockstep_arenas.iter()).enumerate() {
            assert_eq!(
                a.nodes.len(),
                b.nodes.len(),
                "det {i}: node count differs (serial={}, lockstep={})",
                a.nodes.len(),
                b.nodes.len()
            );
            for (j, (na, nb)) in a.nodes.iter().zip(b.nodes.iter()).enumerate() {
                assert_eq!(na.visit_count, nb.visit_count, "det {i} node {j} visit_count");
                assert_eq!(na.action, nb.action, "det {i} node {j} action");
                assert_eq!(
                    na.children.as_slice(),
                    nb.children.as_slice(),
                    "det {i} node {j} children",
                );
                for seat in 0..MAX_PLAYERS {
                    assert_eq!(
                        na.value_counts[seat], nb.value_counts[seat],
                        "det {i} node {j} value_counts[{seat}]"
                    );
                    assert!(
                        (na.value_sums[seat] - nb.value_sums[seat]).abs() < 1e-6,
                        "det {i} node {j} value_sums[{seat}]: serial={}, lockstep={}",
                        na.value_sums[seat],
                        nb.value_sums[seat],
                    );
                }
            }
        }
    }

    /// Session 7.4c stage-2 parity: at `target_batch = 1` the lockstep
    /// driver only ever has one descent in flight, so virtual loss never
    /// engages and per-det node sequences match `run_search` bit-for-bit
    /// — the same parity guarantee Stage 1 provided at
    /// `target_batch = num_dets` (pinned by
    /// `lockstep_search_matches_serial_per_det`), now extended to the
    /// degenerate batch=1 setting that downstream callers can use to
    /// disable VL entirely without touching the search code.
    #[test]
    fn target_batch_one_matches_serial_per_det() {
        let states = [
            playing_state(101),
            playing_state(202),
            playing_state(303),
        ];
        let sims = 80u32;

        let mut serial_arenas: Vec<MctsArena> = states
            .iter()
            .map(|s| MctsArena::new(s.current_player))
            .collect();
        for (arena, state) in serial_arenas.iter_mut().zip(states.iter()) {
            run_search(arena, state, &DummyEvaluator, sims, DEFAULT_C_PUCT);
        }

        let mut tb1_arenas: Vec<MctsArena> = states
            .iter()
            .map(|s| MctsArena::new(s.current_player))
            .collect();
        let states_vec: Vec<BlobState> = states.to_vec();
        run_lockstep_search(
            &mut tb1_arenas,
            &states_vec,
            &DummyEvaluator,
            sims,
            DEFAULT_C_PUCT,
            1,
        );

        for (i, (a, b)) in serial_arenas.iter().zip(tb1_arenas.iter()).enumerate() {
            assert_eq!(
                a.nodes.len(),
                b.nodes.len(),
                "det {i}: node count differs",
            );
            for (j, (na, nb)) in a.nodes.iter().zip(b.nodes.iter()).enumerate() {
                assert_eq!(na.visit_count, nb.visit_count, "det {i} node {j} visit_count");
                assert_eq!(na.action, nb.action, "det {i} node {j} action");
                assert_eq!(
                    na.children.as_slice(),
                    nb.children.as_slice(),
                    "det {i} node {j} children",
                );
                for seat in 0..MAX_PLAYERS {
                    assert_eq!(
                        na.value_counts[seat], nb.value_counts[seat],
                        "det {i} node {j} value_counts[{seat}]"
                    );
                    assert!(
                        (na.value_sums[seat] - nb.value_sums[seat]).abs() < 1e-6,
                        "det {i} node {j} value_sums[{seat}]"
                    );
                }
            }
        }
    }

    /// Stage 2 sanity: every virtual visit decoration must be undone by
    /// the matching expand/backprop step. A non-zero `in_flight` left
    /// over after search would corrupt subsequent UCB1 reads (the next
    /// search reuses the same arena layout via `MctsArena::with_capacity`
    /// only if the tree is freshly allocated, but the invariant is still
    /// load-bearing for any future caller that recycles arenas — and
    /// it's the simplest tripwire if a path-bookkeeping bug slips into
    /// the driver).
    #[test]
    fn lockstep_search_clears_in_flight_at_target_batch_above_num_dets() {
        let states = [
            playing_state(11),
            playing_state(22),
            playing_state(33),
        ];
        let sims = 60u32;
        let mut arenas: Vec<MctsArena> = states
            .iter()
            .map(|s| MctsArena::new(s.current_player))
            .collect();
        let states_vec: Vec<BlobState> = states.to_vec();
        // target_batch > num_dets exercises the intra-det virtual-loss
        // path; if the decrement step misses any path entry the assert
        // below trips.
        run_lockstep_search(
            &mut arenas,
            &states_vec,
            &DummyEvaluator,
            sims,
            DEFAULT_C_PUCT,
            8,
        );
        for (i, arena) in arenas.iter().enumerate() {
            for (j, n) in arena.nodes.iter().enumerate() {
                assert_eq!(
                    n.in_flight, 0,
                    "det {i} node {j} left with in_flight={}",
                    n.in_flight,
                );
            }
        }
    }

    /// Stage 2 budget invariant: regardless of `target_batch`, each det's
    /// root must end the search with exactly `num_simulations` total
    /// visits (visit_count grows by 1 per descent, terminal or not).
    /// This is the contract the policy aggregator in `mcts_search`
    /// relies on — if a det undercounts visits, its share of the
    /// aggregated visit distribution is wrong.
    #[test]
    fn lockstep_search_root_visit_count_matches_sim_budget() {
        let states = [playing_state(7), playing_state(13)];
        let sims = 50u32;
        let states_vec: Vec<BlobState> = states.to_vec();

        for &target_batch in &[1usize, 2, 5, 8] {
            let mut arenas: Vec<MctsArena> = states
                .iter()
                .map(|s| MctsArena::new(s.current_player))
                .collect();
            run_lockstep_search(
                &mut arenas,
                &states_vec,
                &DummyEvaluator,
                sims,
                DEFAULT_C_PUCT,
                target_batch,
            );
            for (i, arena) in arenas.iter().enumerate() {
                assert_eq!(
                    arena.root().visit_count,
                    sims,
                    "target_batch={target_batch} det {i}: root visits != sims",
                );
            }
        }
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

    /// fix-mcts-plan.md Step 1: a Dirichlet(α, …, α) sample of length `n`
    /// must be non-negative and sum to ~1.
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

    /// fix-mcts-plan.md Step 1 unit-test contract: after
    /// `apply_root_dirichlet_noise`, root child priors must
    /// (a) differ from the raw evaluator output by the expected mixing
    /// weight (`|P' − (1 − ε)·P| = ε · η`), and (b) still sum to 1.
    #[test]
    fn apply_root_dirichlet_noise_mixes_and_renormalizes() {
        let s = playing_state(11);
        let mut arena = MctsArena::new(s.current_player);
        let (policy, _) = DummyEvaluator.evaluate(&s);
        expand(&mut arena, 0, &s, &policy);

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
    /// the pre-7.5 regime and the parity tests.
    #[test]
    fn apply_root_dirichlet_noise_is_noop_when_epsilon_zero() {
        let s = playing_state(17);
        let mut arena = MctsArena::new(s.current_player);
        let (policy, _) = DummyEvaluator.evaluate(&s);
        expand(&mut arena, 0, &s, &policy);
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

    /// `mcts_search` with `root_dirichlet_epsilon > 0` must:
    /// - keep the total simulation budget intact (root visits == sims),
    /// - leave the returned policy as a valid distribution.
    /// Combined with `apply_root_dirichlet_noise_mixes_and_renormalizes`
    /// this exercises the C1 integration path end-to-end against the
    /// dummy evaluator.
    #[test]
    fn mcts_search_with_root_noise_preserves_budget_and_policy() {
        let s = playing_state(23);
        let cfg = MctsConfig {
            num_determinizations: 3,
            sims_per_determinization: 25,
            min_sims_floor: 60,
            root_dirichlet_alpha: 0.3,
            root_dirichlet_epsilon: 0.25,
            ..MctsConfig::default()
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026_05_12_01);
        let result = mcts_search(&s, &DummyEvaluator, &cfg, &mut rng, 0);
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy sum {sum} ≠ 1");
        // adaptive_budget may raise dets×sims to satisfy min_sims_floor,
        // so just assert the aggregate is at least the configured floor.
        assert!(
            result.total_visits >= cfg.min_sims_floor,
            "total_visits {} below floor {}",
            result.total_visits,
            cfg.min_sims_floor
        );
    }
}
