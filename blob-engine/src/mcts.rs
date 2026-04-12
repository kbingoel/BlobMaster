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

use smallvec::SmallVec;

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
pub fn run_search<E: Evaluator>(
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
