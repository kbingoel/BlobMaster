# Determinization Optimization Plan

**Status**: Ready to execute
**Expected Impact**: 136 days → 1-20 days training time
**Primary Goal**: Reduce avg_attempts_per_success from 1,737 → <50

## Problem Summary

**Determinization sampling is the bottleneck, not inference.**

Latest profiling runs (see [profiling-readme.md](profiling-readme.md)) show:
- **Determinization**: ~97.7% of MCTS time (~4,795s total)
- **Success rate**: ~0.06% (~1,737 attempts per successful sample)
- **Per-sample cost**: ~58ms average
- **MCTS node expansion**: Minor by comparison (~111s total, ~0.29ms per call)
- **Batched evaluator**: Healthy throughput (~2.5s total, ~159µs per item, avg batch size 29.3/30)

**Inference is not limiting** - GPU batching path shows high efficiency. The bottleneck is purely in determinization sampling logic.

**Root Cause**: Hard enforcement of `must_have_suits` on current hands encodes "had suit at time T" as "must have suit now". Once a player follows suit (e.g., plays ♥), they're marked `must_have_suits={♥}` forever, even after exhausting that suit. This creates stale constraints that exponentially crush the feasible sample space.

**Plan**: Fix must-have semantics (soft prior), add feasibility precheck, order by constraint tightness with early propagation, then validate via profiling artifacts.

## Validation Targets

- ✅ `avg_attempts_per_success`: 1,737 → **<50**
- ✅ `avg_sample_ms`: 58ms → **<5-10ms**
- ✅ Determinization wall-time: 97.7% → **<60%**

## Implementation Sessions

### **Session 1 (4h): Remove Hard Must-Have Constraint** ⚡ CRITICAL

**Goal**: Fix correctness bug - make `must_have_suits` a soft prior, not hard validation

**Changes**:

1. **[belief_tracker.py:311-314](ml/mcts/belief_tracker.py#L311-L314)** - Remove hard enforcement:
   ```python
   # REMOVE these lines from is_consistent_hand():
   # hand_suits = set(card.suit for card in hand)
   # if not constraints.must_have_suits.issubset(hand_suits):
   #     return False
   ```

2. **[determinization.py:186-215](ml/mcts/determinization.py#L186-L215)** - Add explicit bias for must-have suits:
   ```python
   def _sample_with_probabilities(self, available_cards, num_cards, belief, player_pos):
       """Sample cards with probability weighting, biased toward must-have suits."""

       constraints = belief.player_constraints[player_pos]
       probabilities = []

       for card in available_cards:
           prob = belief.get_card_probability(player_pos, card)

           # Boost probability for must-have suits (soft prior)
           if card.suit in constraints.must_have_suits:
               prob *= 2.5  # Weight boost (configurable: 1.5-3.0)

           probabilities.append(prob)

       # Normalize and sample
       # ... existing logic ...
   ```

   **Note**: Current implementation uses uniform probabilities over feasible cards. This change explicitly biases toward must-have suits without hard-enforcing them.

**Alternative (Conservative)**: Compute "effective must-have now"
- Only enforce suit if player has ANY feasible cards of that suit left
- Avoids need to estimate original suit holdings

**Tests**:
- Update `test_determinization.py`: Remove expectations that must-have is strictly enforced
- Add test: player exhausts suit → sampling succeeds without that suit
- Verify: `test_belief_tracker.py` constraint tests still pass

**Expected Impact**: 100-500x improvement in success rate

**Validation**: Run 100-game benchmark, measure attempts/success

---

### **Session 2 (4h): Feasibility Precheck**

**Goal**: Short-circuit impossible constraint combinations before sampling

**Implementation**:

**[determinization.py:145](ml/mcts/determinization.py#L145)** - Add before `_attempt_sample` loop:
```python
from ml.game.constants import SUITS

def _check_global_feasibility(
    self, belief: BeliefState, unseen_pool: List[Card]
) -> bool:
    """Check if must-have constraints are satisfiable (player-aware)."""

    # Group by suit (use SUITS from constants.py)
    for suit in SUITS:
        # Find all players requiring this suit
        players_needing = [
            p for p, c in belief.player_constraints.items()
            if suit in c.must_have_suits
        ]

        if not players_needing:
            continue

        # Collect constraints for players needing this suit
        needing_constraints = [
            belief.player_constraints[p] for p in players_needing
        ]

        # Count feasible cards of this suit (union across needing players)
        # Only count cards that at least ONE needing player can have
        all_feasible_suit_cards = {
            c for c in unseen_pool
            if c.suit == suit and any(pc.can_have_card(c) for pc in needing_constraints)
        }

        # Per-player check: each player must have at least one feasible card
        for player_pos in players_needing:
            constraints = belief.player_constraints[player_pos]
            player_feasible = [
                c for c in unseen_pool
                if c.suit == suit and constraints.can_have_card(c)
            ]

            # Player needs suit but has no feasible cards
            if not player_feasible:
                return False

        # Global check: more players need suit than feasible cards exist
        if len(players_needing) > len(all_feasible_suit_cards):
            return False

    return True
```

**Usage**: Call at start of `sample_determinization` before retry loop

**Tests**:
- Impossible scenario: 3 players need hearts, only 2 hearts unseen → immediate failure
- Verify short-circuit happens (no sampling attempts)

**Expected Impact**: 2-5x speedup by avoiding doomed attempts

---

### **Session 3 (4h): Tightness Ordering + Constraint Propagation**

**Goal**: Sample constrained players first and detect conflicts early

**Part A: Dynamic Tightness Ordering**

**[determinization.py:155](ml/mcts/determinization.py#L155)** - Replace fixed sort with dynamic recomputation:
```python
def _compute_tightness(self, player_pos, unseen_pool, belief):
    """Compute ordering key: tighter constraints = sample first."""
    constraints = belief.player_constraints[player_pos]

    # Count feasible cards for this player NOW (pool shrinks each iteration)
    feasible = [
        c for c in unseen_pool if constraints.can_have_card(c)
    ]

    headroom = len(feasible) - constraints.cards_in_hand
    num_must_have = len(constraints.must_have_suits)

    # Sort ascending headroom (tight first), descending must-have count
    return (headroom, -num_must_have)

# In _attempt_sample - refactor to recompute each iteration:
sampled_hands = {}
remaining_players = list(belief.player_constraints.keys())

while remaining_players:
    # Recompute ordering after each assignment (pool changes)
    ordered = sorted(
        remaining_players,
        key=lambda p: self._compute_tightness(p, unseen_pool, belief)
    )

    player_pos = ordered[0]  # Sample tightest first
    # ... sample for this player ...
    remaining_players.remove(player_pos)
```

**Part B: In-Loop Propagation with Retry**

**[determinization.py:178](ml/mcts/determinization.py#L178)** - Call after tentative assignment, retry before giving up:
```python
# Try to sample for this player (with limited retries)
max_player_retries = 3
for retry in range(max_player_retries):
    sampled_cards = # ... sampling logic ...

    # Tentative assignment
    temp_hands = {**sampled_hands, player_pos: sampled_cards}

    # Check if partial sample is still feasible
    if self._propagate_constraints(belief, temp_hands):
        # Success - commit and continue
        sampled_hands[player_pos] = sampled_cards
        for card in sampled_cards:
            unseen_pool.remove(card)
        break
else:
    # Exhausted retries for this player
    return None  # Fall back to full restart
```

**Note**: Try different hands for the current player before giving up entirely. This reduces full restarts.

**Tests**:
- Scenario: Player 1 has loose constraints, Player 2 tight
  - Fixed order fails frequently
  - Tightness order succeeds more often
- Verify propagation catches conflicts before full sample

**Expected Impact**: 10-30x speedup combined

---

### **Session 4 (4h): Profiling & Validation**

**Goal**: Measure actual improvements and decide if further optimization needed

**Actions**:

1. **Run full benchmark** (1000 games, 32 workers):
   ```bash
   python ml/profile_selfplay.py --instrument --workers 32 --games 1000
   ```

2. **Analyze results**:
   - Check `profile_*_aggregate.json`:
     - `determinization.avg_attempts_per_success` → target <50
     - `determinization.total_time` vs `batch_evaluator.total_time`
     - `determinization.avg_sample_ms` → target <10ms

3. **Compare to baseline**:
   - Baseline: 36.7 games/min (Medium MCTS)
   - Target after Sessions 1-3: **>100 games/min**

4. **Decision point**:
   - If targets met → DONE, start training
   - If determinization still >60% of time → continue to Session 5+

**Deliverable**: Performance report with go/no-go for additional sessions

---

### **Session 5 (4h): Single-Level Backtracking** [IF NEEDED]

**Goal**: Avoid full restart when late player fails

**Implementation**:

**[determinization.py:145-184](ml/mcts/determinization.py#L145-L184)** - Refactor `_attempt_sample`:
```python
from collections import defaultdict

def _attempt_sample_with_backtrack(self, ...):
    """Sample with single-level backtracking and retry caps."""

    max_backtracks_per_player = 3
    max_total_attempts = 10  # Cap total attempts to avoid loops
    player_attempts = defaultdict(int)  # Track retries per player
    total_attempts = 0

    sampled_hands = {}
    player_order = [...]  # From tightness ordering

    i = 0  # Current player index
    while i < len(player_order) and total_attempts < max_total_attempts:
        player_pos = player_order[i]
        total_attempts += 1

        # Try to sample for this player (randomize for diversity)
        if use_probabilities:
            sampled_cards = self._sample_with_probabilities(...)
        else:
            # Add randomization to escape local traps
            sampled_cards = random.sample(available_cards, cards_needed)

        # Validate sample
        success = _validate_player_sample(sampled_cards, ...)

        if success:
            sampled_hands[player_pos] = sampled_cards
            # Remove from pool
            for card in sampled_cards:
                unseen_pool.remove(card)
            i += 1  # Move to next player
            continue

        # Failed - can we backtrack?
        if i > 0 and player_attempts[player_order[i-1]] < max_backtracks_per_player:
            # Backtrack to previous player
            prev_player = player_order[i-1]
            # Restore their cards to pool
            unseen_pool.extend(sampled_hands[prev_player])
            del sampled_hands[prev_player]
            player_attempts[prev_player] += 1
            i -= 1  # Retry previous player
        else:
            # Can't backtrack - full restart
            return None

    return sampled_hands if i == len(player_order) else None
```

**Note**: Import `defaultdict` from `collections`. Randomize alternative hands to escape local traps. Cap total attempts to avoid infinite loops.

**Tests**:
- Scenario where Player 3 fails but Player 2 re-sampling succeeds
- Verify fewer full restarts

**Expected Impact**: 50-100x additional speedup

---

### **Session 6 (4h): Bipartite Matching for Must-Have** [IF NEEDED]

**Goal**: Pre-assign must-have suit cards optimally before random sampling

**Implementation**:

**[determinization.py:154](ml/mcts/determinization.py#L154)** - Add before main loop:
```python
from ml.game.constants import SUITS

def _assign_must_have_suits(self, belief, unseen_pool):
    """Use bipartite matching to pre-assign must-have cards."""

    assignments = {}  # player_pos -> [assigned cards]

    for suit in SUITS:  # Use SUITS from constants.py
        # Find players requiring this suit
        players = [
            p for p, c in belief.player_constraints.items()
            if suit in c.must_have_suits
        ]

        if not players:
            continue

        # Build bipartite graph: players ↔ feasible suit cards
        suit_cards = [c for c in unseen_pool if c.suit == suit]

        # Try greedy matching first (fast)
        matched = self._greedy_match_suit(players, suit_cards, belief)

        # If greedy fails for some players, use Kuhn's algorithm (DFS-based)
        if len(matched) < len(players):
            matched = self._kuhn_match_suit(players, suit_cards, belief)

        # Apply matched assignments
        for player_pos, card in matched.items():
            assignments.setdefault(player_pos, []).append(card)
            unseen_pool.remove(card)

    return assignments  # Merge with sampled_hands later

def _kuhn_match_suit(self, players, suit_cards, belief):
    """Kuhn's algorithm (augmenting path) for maximum bipartite matching."""

    match_player = {}  # player -> card
    match_card = {}    # card -> player

    def dfs(player, visited):
        """Find augmenting path from player."""
        constraints = belief.player_constraints[player]

        for card in suit_cards:
            if card in visited or not constraints.can_have_card(card):
                continue

            visited.add(card)

            # Card is unmatched or we can augment from matched player
            if card not in match_card or dfs(match_card[card], visited):
                match_player[player] = card
                match_card[card] = player
                return True

        return False

    # Try to match each player
    for player_pos in players:
        dfs(player_pos, set())

    return match_player
```

**Note**: No SciPy dependency. Greedy for common cases, Kuhn's algorithm for complex cases. Uses `SUITS` from constants.py.

**Tests**:
- 3 players need hearts, 3 hearts available → each gets one
- 2 players need hearts, 1 heart available → greedy assigns to first, Kuhn finds best matching
- Verify remaining cards sampled randomly

**Expected Impact**: 100-200x additional speedup (guarantees must-have satisfied)

---

### **Session 7 (4h): Belief Signature Caching** [IF NEEDED]

**Goal**: Reuse valid samples when constraints unchanged

**Implementation**:

**[determinization.py:90](ml/mcts/determinization.py#L90)** - Extend existing `Determinizer` class:
```python
from collections import deque

class Determinizer:  # Extend existing class, don't create new one
    def __init__(self):
        # ... existing init ...
        self.sample_cache = {}  # signature -> deque of valid hands
        self.cache_size = 8  # Per signature

    def _belief_signature(self, belief: BeliefState):
        """Compact hash of constraint state using full card identity."""

        # Use full card identity (rank, suit), not just value
        # Cards are hashable, so we can use them directly
        unseen_cards = tuple(sorted(belief.unseen_cards))

        # Or use string representation for clarity:
        # unseen_cards = tuple(sorted(str(card) for card in belief.unseen_cards))

        constraints_tuple = tuple(
            (
                pos,
                frozenset(c.cannot_have_suits),  # Suit bitmask
                frozenset(c.must_have_suits),
                c.cards_in_hand,
            )
            for pos, c in sorted(belief.player_constraints.items())
        )

        return hash((unseen_cards, constraints_tuple))

    def sample_determinization(self, belief, ...):
        sig = self._belief_signature(belief)

        # Try cache first
        if sig in self.sample_cache and self.sample_cache[sig]:
            cached = self.sample_cache[sig].popleft()
            # Optional: perturb 1-2 cards for diversity
            # (e.g., swap one card between two players)
            return cached

        # ... normal sampling ...

        # On success, cache result
        if sig not in self.sample_cache:
            self.sample_cache[sig] = deque(maxlen=self.cache_size)
        self.sample_cache[sig].append(sampled_hands)
```

**Note**: Use full card identity `(rank, suit)` not just `card.value`. Extend existing `Determinizer` class, don't create new class. Import `deque` from `collections`.

**Tests**:
- Same belief state called twice → cache hit
- Verify cache size bounds respected

**Expected Impact**: Variable (depends on cache hit rate), target >30% hits

---

## Ready-To-Proceed Checklist

Before starting implementation, ensure the following are understood:

- ✅ **Session 1**: Update [ml/mcts/belief_tracker.py:311-314](ml/mcts/belief_tracker.py#L311-L314) to remove hard must-have enforcement
- ✅ **Session 1**: Add sampling bias for must-have suits in [ml/mcts/determinization.py](ml/mcts/determinization.py)'s `_sample_with_probabilities` (multiply by 1.5-3.0)
- ✅ **Session 2**: Implement feasibility precheck with `SUITS` from [ml/game/constants.py](ml/game/constants.py) and feasible-card union across needing players
- ✅ **Session 3**: Add dynamic tightness ordering (recompute after each assignment) and in-loop `_propagate_constraints` call with per-player retry
- ✅ **Session 4**: Profile with instrumentation; gate further work on hitting <50 attempts/success and <10ms avg sample
- ✅ **Session 5+**: Only proceed if Session 4 profiling shows determinization still >60% of time

**Validation Artifacts**: Confirm profiling setup produces files matching [profiling-readme.md](profiling-readme.md) format:
- `profile_*_aggregate.json` with determinization metrics
- `profile_w32_g*_c*.prof` for cProfile analysis

---

## Execution Order

### **Phase 1: Core Fixes** (Sessions 1-4, ~16h)
1. Session 1: Remove hard must-have ← **HIGHEST PRIORITY**
2. Session 2: Feasibility precheck
3. Session 3: Tightness + propagation
4. Session 4: Profile & validate

**Expected Outcome**: 10-100x speedup, determinization <60% of time

### **Phase 2: Advanced Optimizations** (Sessions 5-7, ~12h, IF NEEDED)
5. Session 5: Backtracking
6. Session 6: Matching
7. Session 7: Caching

**Expected Outcome**: 100-500x total speedup, determinization <20% of time

## Test Strategy

**Regression Tests** (must pass after each session):
- `pytest ml/mcts/test_determinization.py` - 12 existing tests
- `pytest ml/mcts/test_belief_tracker.py` - Constraint logic
- `pytest ml/tests/test_imperfect_info_integration.py` - End-to-end

**New Tests** (add during implementation):
- Session 1: Must-have relaxation scenarios
- Session 2: Infeasible constraint detection
- Session 3: Tightness ordering effectiveness
- Session 5+: Backtracking, matching, caching

**Profiling Validation** (after each phase):
```bash
python ml/profile_selfplay.py --instrument --workers 32 --games 100
python benchmarks/performance/analyze_profile.py profile_*_aggregate.json
```

**Success Criteria**:
- All tests pass
- No correctness regressions (belief tracking still sound)
- Performance targets met

## Risk Mitigation

**Correctness Risks**:
- Session 1 changes core semantics → extensive testing required
- Validate belief tracking still deduces information correctly
- Compare game outcomes before/after optimization (statistical tests)

**Performance Risks**:
- Changes might not achieve expected speedup → profile after each session
- New code might introduce overhead → benchmark carefully
- Go/no-go decision after Session 4 before investing in advanced optimizations

**Rollback Plan**:
- Git branch for each session
- Keep baseline performance numbers
- Can cherry-pick successful optimizations if some fail

## Files Modified

| File | Sessions | Changes |
|------|----------|---------|
| [ml/mcts/belief_tracker.py](ml/mcts/belief_tracker.py) | 1 | Remove hard must-have enforcement |
| [ml/mcts/determinization.py](ml/mcts/determinization.py) | 2,3,5,6,7 | Precheck, ordering, propagation, backtracking, matching, cache |
| [ml/mcts/test_determinization.py](ml/mcts/test_determinization.py) | 1-7 | Update/add tests for each optimization |
| [ml/mcts/test_belief_tracker.py](ml/mcts/test_belief_tracker.py) | 1 | Verify constraint semantics |

## Success Metrics

**Immediate** (after Phase 1):
- ✅ Training time: 136 days → **5-20 days** (7-27x speedup)
- ✅ Games/min: 36.7 → **>100** (Medium MCTS)
- ✅ Attempts/success: 1,737 → **<100**

**Stretch** (after Phase 2):
- ✅ Training time: 136 days → **1-3 days** (45-136x speedup)
- ✅ Games/min: 36.7 → **>300** (Medium MCTS)
- ✅ Attempts/success: 1,737 → **<10**
- ✅ Determinization share: 97.7% → **<20%**
