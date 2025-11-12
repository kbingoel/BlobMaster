# Training Design: Rounds vs Full Games vs Mixed Rounds

## The Question
Should training generate:
- **Option A**: 10,000 independent **rounds** (each with fixed 5 cards)?
- **Option B**: 10,000 full **games** (each with 5,4,3,2,1,2,3,4,5 card sequence)?
- **Option C (TOMS Proposal)**: 10,000 independent **rounds** with **variable cards/players**?

## Current Implementation (Code Analysis)

```python
# ml/training/selfplay.py - SelfPlayWorker.generate_game()
def generate_game(self, num_players=4, cards_to_deal=5):
    """Generate a single self-play game."""
    game = BlobGame(num_players=num_players)
    result = game.play_round(cards_to_deal, get_bid, get_card)
    # ^^^ Plays ONE round with fixed cards_to_deal
    return examples

# ml/config.py
games_per_iteration: int = 10_000  # Actually rounds!
cards_to_deal: int = 5             # Fixed
```

**Current code generates:** 10,000 independent 5-card rounds

## AlphaZero Literature Review

### What AlphaGo/AlphaZero Did

**Go/Chess/Shogi:**
- Generated **complete games** (from start to finish)
- Each game varied in length naturally
- Training examples came from all positions in the game
- Games correlated (positions from same game related)

**Reasoning:**
- Games have natural start/end states
- Full game provides context for evaluation
- Final outcome meaningful across all positions
- Position evaluation depends on game progression

### How This Applies to Blob

**Full Blob Game Structure:**
- Round 1: 5 cards (high complexity, many tricks)
- Round 2: 4 cards (medium complexity)
- Round 3: 3 cards (medium complexity)
- Round 4: 2 cards (low complexity, simple)
- Round 5: 1 card (trivial, no decisions)
- Round 6: 2 cards (repeat)
- ... back to 5 cards

**Total: 9 rounds, variable complexity**

## Analysis

### Option A: Independent 5-Card Rounds (Current)

**Pros:**
- ✅ **Consistent complexity**: All training data at same difficulty
- ✅ **Faster generation**: 660 rounds/min vs ~73 games/min (9x speedup)
- ✅ **More positions per time**: More training examples per day
- ✅ **Simpler implementation**: Already working
- ✅ **5 cards optimal**: Most interesting (not too simple, not too complex)

**Cons:**
- ❌ **No score accumulation**: Can't learn multi-round strategy
- ❌ **Missing low-card scenarios**: Never trains on 1-2 card situations
- ❌ **Independent rounds**: Loses correlation of full game
- ❌ **Single task**: Only learns 5-card play, not general Blob

**Training data per iteration:**
- 10,000 rounds × 24 decisions = 240,000 examples
- All at 5-card complexity level

### Option B: Full 9-Round Games

**Pros:**
- ✅ **Complete experience**: Learns full game flow
- ✅ **Score accumulation**: Understands multi-round strategy
- ✅ **Variable complexity**: Trains on 1-5 card scenarios
- ✅ **Realistic play**: Matches actual Blob games
- ✅ **Transfer learning**: Can adapt to different card counts

**Cons:**
- ❌ **Slower generation**: ~73 games/min (9x slower)
- ❌ **Training time**: 48 days instead of 5.3 days
- ❌ **Imbalanced data**: Mostly low-card rounds (less interesting)
- ❌ **Diluted signal**: 1-card rounds trivial, waste compute

**Training data per iteration:**
- 10,000 games × 9 rounds × avg 16 decisions = ~1,440,000 examples
- But heavily skewed toward simple 1-3 card rounds

### Distribution Analysis (Full Games)

In a full game (5,4,3,2,1,2,3,4,5 cards):
```
5 cards: 2 rounds × 24 decisions = 48 decisions (3.3%)
4 cards: 2 rounds × 20 decisions = 40 decisions (2.8%)
3 cards: 2 rounds × 16 decisions = 32 decisions (2.2%)
2 cards: 2 rounds × 12 decisions = 24 decisions (1.7%)
1 card:  1 round  × 8 decisions  = 8 decisions  (0.6%)
────────────────────────────────────────────────
Total: 9 rounds, 152 decisions per game

Distribution:
- 5-card (most complex):  32% of decisions
- 4-card:                 26% of decisions
- 3-card:                 21% of decisions
- 2-card:                 16% of decisions
- 1-card (trivial):       5% of decisions
```

**Problem:** Only 32% of training data comes from the most interesting complexity level!

## Recommendation

### **Option A: Independent 5-Card Rounds** ✅

**Why:**

1. **Quality over quantity**: Focus training on most strategically interesting complexity
   - 5 cards: Rich bidding decisions, complex trick-taking
   - 1-2 cards: Trivial, mostly deterministic
   - Full games: 68% of decisions are "easier" than 5-card

2. **Sample efficiency**:
   - More meaningful training examples per day
   - 660 rounds/min of high-quality data vs 73 games/min of mixed quality

3. **AlphaZero precedent**:
   - AlphaGo trained on **positions** (not always full games)
   - Used expert games + self-play positions
   - Position quality mattered more than full-game context

4. **Practical training time**:
   - 5.3 days (5-card rounds) vs 48 days (full games)
   - 9x faster iteration cycle for experimentation

5. **Transfer learning later**:
   - Model trained on 5-card rounds can **fine-tune** on other card counts
   - Start with strong 5-card play, then generalize
   - Optional: Add variety later (mix of 3,4,5-card rounds)

### Implementation Path

**Phase 4 (Current):** Train on 5-card rounds
- 500 iterations × 10,000 rounds = 5M rounds
- 5.3 days training time
- Focus: Master complex bidding + trick-taking

**Phase 4.5 (Optional Enhancement):**
- Add variety: 70% 5-card, 20% 4-card, 10% 3-card
- Skip 1-2 card rounds (too trivial)
- Train model on variable complexity

**Phase 5 (Future):**
- Fine-tune on full games if needed
- Test transfer learning to complete matches
- Evaluate multi-round strategic planning

### Option C: Mixed Rounds with Variable Cards/Players (TOMS Proposal)

**Pros:**
- ✅ **Comprehensive coverage**: Trains on all round complexities (1-7+ cards)
- ✅ **Equal scoring consideration**: Model learns to nail 1-2 card "easy" rounds AND complex 5-7 card rounds
- ✅ **Transformer flexibility**: Variable player counts (3-8) and card counts teach generalization
- ✅ **Realistic distribution**: Can weight toward most common real-world scenarios
- ✅ **Transfer learning**: Single model handles all game variants

**Cons:**
- ❌ **Slower than fixed 5-card**: More variety means longer training per iteration
- ❌ **Complex implementation**: Need weighted sampling, distribution tracking
- ❌ **Imbalanced difficulty**: Some rounds trivial (1-2 cards), some hard (7+ cards)
- ❌ **Potential inefficiency**: Time spent on trivial rounds could be used on complex ones

**TOMS Proposed Distribution:**
```
5 players, 7 cards:  50%  (primary target - most strategically rich)
4 players, 6 cards:  15%
4 players, 7 cards:  10%
6 players, 6 cards:  10%
6 players, 7 cards:  10%
Other configs:       5%   (3-8 players, 5-9 cards)
```

**Training data per iteration:**
- 10,000 rounds × varied complexity = 180,000-280,000 examples
- 50% at 7 cards (high complexity)
- 35% at 6 cards (medium complexity)
- 10% at 5 cards (still covered)
- 5% at other counts (broad coverage)

## Critical Insight: All-or-Nothing Scoring Changes Everything

### Why This Matters

In Blob's scoring system:
```
Score = (tricks_won == bid) ? (10 + bid) : 0
```

**This means:**
- 1-card round where you bid 0 and win 0: **10 points**
- 7-card round where you bid 3 and win 3: **13 points**
- **Difference: Only 3 points!**

**Key Implications:**
1. **1-2 card rounds are NOT trivial from a scoring perspective** - they're worth nearly as much as complex rounds
2. **Model MUST perform well on ALL round sizes** - missing easy rounds loses games
3. **Strategic complexity ≠ scoring importance** - the easiest rounds matter just as much

### Revised Analysis

**Option A (Fixed 5-card) Problems:**
- ❌ **Never trains on 1-2-3 card scenarios**: Model will fail at "easy" rounds in real games
- ❌ **Missing 20% of game complexity**: Real Blob games include 1-5 card rounds equally
- ❌ **Catastrophic failure mode**: Model might struggle with simple situations it never saw

**Option B (Full Games) Problems:**
- ❌ **Too slow**: 48 days training time unacceptable
- ❌ **Sequential correlation**: Rounds within same game not independent
- ❌ **Fixed sequence**: Always 5,4,3,2,1,2,3,4,5 - no flexibility

**Option C (TOMS Mixed Rounds) Strengths:**
- ✅ **Covers all round sizes**: Model sees 1-7+ card scenarios
- ✅ **Independent rounds**: No correlation issues like full games
- ✅ **Weighted toward complexity**: 50% at 7 cards keeps training challenging
- ✅ **Variable players**: Transformer learns to handle 3-8 players
- ✅ **Scoring alignment**: Training distribution matches scoring importance

## Recommendation: Modified TOMS Approach

### **Use Mixed Rounds with Revised Distribution** ✅

**Proposed Distribution (CURRICULUM-ALIGNED):**
```python
# Card count distribution (BALANCED across complexity levels)
card_distribution = {
    1: 0.05,  # 5%  - Trivial (deterministic, must nail)
    2: 0.10,  # 10% - Simple (low attention complexity)
    3: 0.15,  # 15% - Low (basic trick-taking)
    4: 0.15,  # 15% - Medium-low (suit tracking)
    5: 0.20,  # 20% - Medium (balanced complexity) ← PEAK
    6: 0.20,  # 20% - Medium-high (card counting)
    7: 0.15,  # 15% - High (full strategic depth)
}

# Player count distribution (UNCHANGED - well-balanced)
player_distribution = {
    5: 0.50,  # 50% - standard size
    4: 0.25,  # 25% - smaller group
    6: 0.20,  # 20% - larger group
    3: 0.03,  # 3%  - minimum
    7: 0.02,  # 2%  - large group
}
```

**Why This Distribution?**

1. **Transformer learning dynamics**:
   - **Peak at 5-6 cards (40%)**: Medium complexity where most learning happens
   - **Uniform coverage**: 30% at low (1-3), 40% at medium (4-6), 30% at high (7)
   - **Attention pattern transfer**: Simpler rounds seen frequently enough for model to learn distinct patterns
   - **Avoids over-emphasizing extremes**: Neither trivial (1-2) nor ultra-complex (7) dominates training

2. **Curriculum learning alignment**:
   - More balanced than inverse curriculum (40% at hardest cases)
   - Model builds foundation on simple cases (30% at 1-3 cards)
   - Peak learning zone at medium complexity (5-6 cards)
   - Sufficient hard cases for strategic depth (35% at 6-7 cards)

3. **Equal scoring respect**:
   - Model sees ALL round sizes frequently (unlike Option A)
   - Enough exposure to 1-3 card rounds to avoid catastrophic failure
   - But not wasting 68% of training on trivial rounds (unlike Option B full games)

4. **Training efficiency**:
   - Average ~4.7 cards per round vs 5 cards (Option A) or 3.7 cards (Option B full games)
   - Balanced speed/coverage trade-off
   - Estimated: ~600 rounds/min (vs 660 for fixed 5-card)
   - Training time: ~5.8 days for 500 iterations (vs 5.3 days for Option A)

## CRITICAL: Reward Structure Problem ⚠️

### The Goal Misalignment Issue

**Current implementation trains on independent rounds:**
```python
# ml/training/selfplay.py
def generate_game(self, num_players=4, cards_to_deal=5):
    game = BlobGame(num_players=num_players)
    result = game.play_round(cards_to_deal, get_bid, get_card)
    # Reward: Did we hit our bid THIS ROUND?
    reward = (tricks_won == bid) ? (10 + bid) : 0
```

**Problem: Model learns to maximize individual round scores, NOT total game scores**

### Missing Strategic Context:

In a real Blob game:
- **Round 7 of 9, I'm ahead +30 points**: Bid conservatively (protect lead)
- **Round 7 of 9, I'm behind -30 points**: Bid aggressively (take risks)
- **Round 1 of 9, scores 0-0-0-0**: Play standard strategy

**Current approach: Model NEVER sees score context across rounds**

### Three Solutions

#### **Solution A: Add Game Context to State Encoding** ✅ RECOMMENDED

Keep independent rounds (fast) but encode game context:

```python
# Enhanced StateEncoder in ml/network/encode.py
class StateEncoder:
    def encode_state(self, game_state):
        base_encoding = [
            my_hand,          # Cards I have
            trick_history,    # Cards played
            bids,             # All bids
            # ADD THESE: ↓↓↓
            current_scores,   # [my: 25, p2: 20, p3: 15, p4: 18]
            rounds_completed, # 5 (currently on round 6)
            cards_previous,   # [5, 4, 3, 2, 1] (game trajectory)
        ]
        return transformer_encode(base_encoding)
```

**How it works:**
- State includes: "I'm at +20 points, round 6 of 9"
- Transformer attention learns: "When ahead, adjust bidding strategy"
- Model learns context-dependent play WITHOUT needing full game sequences
- Reward signal: Model sees correlation between "current score" and "optimal action"

**Benefits:**
- ✅ Still trains on independent rounds (5.8 days training)
- ✅ Model learns multi-round strategy via context features
- ✅ No correlation issues (rounds still independent)
- ✅ Approximates full game optimization efficiently

**Implementation:**
```python
# ml/config.py - Add to StateEncoder config
include_game_context: bool = True  # NEW!
max_rounds_per_game: int = 9       # For trajectory encoding

# Training generates rounds with varied "game context":
# - 30% early game (rounds 1-3 of 9, scores near 0)
# - 40% mid game (rounds 4-6 of 9, varied scores)
# - 30% late game (rounds 7-9 of 9, divergent scores)
```

#### **Solution B: Full Games (Original Option B)**

Train on complete games with 5,4,3,2,1,2,3,4,5 sequence:

**Benefits:**
- ✅ Perfect reward alignment (total game score is the target)
- ✅ Natural multi-round context
- ✅ Model learns cross-round dependencies automatically

**Drawbacks:**
- ❌ 48 days training time (vs 5.8 days)
- ❌ Correlated examples (rounds within same game)
- ❌ Inefficient (33% of training on 1-2 card trivial rounds)

#### **Solution C: Hybrid Curriculum** 🎯 BEST OF BOTH WORLDS

Two-phase training combining speed + context:

```
PHASE 1: Independent Rounds (400 iterations, ~4.6 days)
  - Train on independent rounds with game context encoding
  - Learn: Basic bidding, trick-taking, card play
  - Fast convergence: 600 rounds/min

PHASE 2: Full Games (100 iterations, ~10 days)
  - Train on full game sequences (5,4,3,2,1,2,3,4,5)
  - Learn: Multi-round optimization, score management
  - Reward: Total game score (true objective)

Total: ~14.6 days (vs 48 days for full games only, 5.8 days for independent only)
```

**Benefits:**
- ✅ Phase 1 builds strong foundation quickly (4.6 days)
- ✅ Phase 2 fine-tunes multi-round strategy (10 days)
- ✅ Balanced efficiency (14.6 days)
- ✅ Progressive difficulty (curriculum learning)
- ✅ Perfect for TOMS integration (natural phase boundary)

**Timeline:**
```
Iterations 1-400:   Independent rounds + context encoding
  - 4.6 days @ 600 rounds/min
  - Learns basic strategy fast

Iterations 401-500: Full games (5,4,3,2,1,2,3,4,5)
  - 10 days @ ~73 games/min
  - Learns multi-round optimization
  - Can use Heavy MCTS (higher quality) since shorter phase
```

### Recommendation: Solution A + C Combined

**OPTIMAL APPROACH:**

1. **Implement Solution A** (game context encoding) - REQUIRED for any independent round training
2. **Start with Phase 1** (400 iterations, independent rounds + context) - Fast learning
3. **Evaluate after Phase 1**: If model shows context-dependent play, may be sufficient!
4. **Optionally add Phase 2** (100 iterations, full games) - Fine-tuning if needed

**Implementation Priority:**
```
IMMEDIATE (before training):
  ✅ Add game context to StateEncoder (current_scores, rounds_completed, trajectory)
  ✅ Generate rounds with varied game context (early/mid/late game scenarios)
  ✅ Test encoding with 5-10 iteration pilot run

SHORT TERM (after 400 iterations):
  ✅ Evaluate: Does model show context-dependent bidding?
  ✅ If yes: Continue with independent rounds (fast)
  ✅ If no: Add Phase 2 full games (100 iterations)

LONG TERM (Phase 5):
  ✅ Final evaluation on full game tournaments
  ✅ Measure: Does model optimize for total score or individual rounds?
```

### Why NOT Use TOMS Distribution As-Is

**TOMS Problem:**
```
5 players, 7 cards:  50%  (good!)
4 players, 6 cards:  15%  (good!)
4 players, 7 cards:  10%  (good!)
6 players, 6 cards:  10%  (good!)
6 players, 7 cards:  10%  (good!)
Other configs:       5%   (TOO LITTLE!)
```

**Issues:**
- ❌ **Missing 1-3 card rounds almost entirely**: Only 5% "other" means model rarely sees trivial rounds
- ❌ **Too narrow player focus**: 100% at 4-6 players, ignores 3-player and 7-8 player games
- ❌ **No 4-5 card coverage**: Focuses on 6-7 cards only, skips medium complexity

**Improved Distribution:**
- ✅ **10% at 1-3 cards**: Ensures model can handle "easy" rounds (critical for scoring!)
- ✅ **15% at 4-5 cards**: Medium complexity coverage
- ✅ **75% at 6-7 cards**: Primary strategic focus
- ✅ **5% at 3-player, 2% at 7-player**: Broad player count handling

### Implementation Path

**Phase 4 (Current - REVISED):** Train on mixed rounds with revised distribution
- 500 iterations × 10,000 rounds = 5M rounds
- ~6.3 days training time (vs 5.3 days for Option A)
- Distribution: 75% at 6-7 cards, 15% at 4-5 cards, 10% at 1-3 cards
- Player counts: 50% at 5p, 25% at 4p, 20% at 6p, 5% at 3/7p

**Phase 4.5 (Optional - TOMS Integration):**
- Add MCTS curriculum: 1→2→3→4→5 determinizations over 500 iterations
- Dynamic games per iteration: 5K→10K→15K
- Estimated time savings: ~60-80 days (vs 136 days at Medium MCTS throughout)

**Phase 5 (Future):**
- Evaluate model on full games (not just rounds)
- Test transfer learning to 7-8 player games
- Fine-tune on specific weaknesses identified in evaluation

### Configuration Update

```python
# ml/config.py (UPDATED RECOMMENDATION)
rounds_per_iteration: int = 10_000  # Renamed for clarity

# Card distribution (emphasize complexity but cover all ranges)
card_distribution: Dict[int, float] = {
    7: 0.40,  # Primary focus
    6: 0.25,  # High complexity
    5: 0.15,  # Medium-high
    4: 0.10,  # Medium
    3: 0.05,  # Low (but critical!)
    2: 0.03,  # Trivial (but critical!)
    1: 0.02,  # Trivial (but critical!)
}

# Player distribution (emphasize 4-6 players)
player_distribution: Dict[int, float] = {
    5: 0.50,  # Standard
    4: 0.25,  # Small group
    6: 0.20,  # Large group
    3: 0.03,  # Minimum
    7: 0.02,  # Very large
}
```

## Comparison: All Four Approaches

| Criterion | Option A<br>(Fixed 5-card) | Option B<br>(Full Games) | **Option C**<br>**(Mixed Rounds)** ✅ | **Option C+**<br>**(+ Game Context)** 🎯 |
|-----------|------------------------|----------------------|-------------------------------|----------------------------------|
| **Training Time** | 5.3 days | 48 days | **5.8 days** | **5.8 days** (Phase 1 only)<br>**14.6 days** (w/ Phase 2) |
| **Covers 1-3 cards** | ❌ Never | ✅ 33% (too much) | ✅ 30% (balanced) | ✅ 30% (balanced) |
| **Covers 4-6 cards** | ✅ 100% (too much) | ✅ 56% | ✅ 55% (peak at 5-6) | ✅ 55% (peak at 5-6) |
| **Covers 7 cards** | ❌ Never | ✅ 11% | ✅ 15% (sufficient) | ✅ 15% (sufficient) |
| **Player count variety** | ❌ Fixed | ❌ Fixed | ✅ Variable (3-7) | ✅ Variable (3-7) |
| **Curriculum aligned** | ❌ No | ❌ No | ✅ Yes (peak at medium) | ✅ Yes (peak at medium) |
| **Transformer transfer** | ❌ Limited | ⚠️ Diluted | ✅ Balanced coverage | ✅ Balanced coverage |
| **Game context** | ❌ None | ✅ Natural | ❌ None | ✅ **Encoded** |
| **Reward alignment** | ❌ Round-only | ✅ Total score | ❌ Round-only | ✅ **Context-dependent** |
| **Independent rounds** | ✅ Yes | ❌ No (correlated) | ✅ Yes | ✅ Yes |
| **Strategic depth** | ⚠️ High but narrow | ⚠️ Diluted | ✅ Balanced | ✅ Balanced + context |
| **AlphaZero precedent** | ✅ Position-based | ❌ Full games | ✅ Position-based | ✅ Position-based + context |

**Clear Winner: Option C+ (Mixed Rounds + Game Context + Optional Phase 2)**

## Improvements to TOMS.md

### Recommendation 1: Revise Game Distribution

**Change:**
```diff
- 5 players, 7 cards:  50%  (primary target)
- 4 players, 6 cards:  15%
- 4 players, 7 cards:  10%
- 6 players, 6 cards:  10%
- 6 players, 7 cards:  10%
- Other configs:       5%   (3-8 players, 5-9 cards)
+ # Separate card and player distributions for better coverage
+
+ Card distribution:
+   7 cards: 40%
+   6 cards: 25%
+   5 cards: 15%
+   4 cards: 10%
+   3 cards: 5%
+   2 cards: 3%
+   1 card:  2%
+
+ Player distribution:
+   5 players: 50%
+   4 players: 25%
+   6 players: 20%
+   3 players: 3%
+   7 players: 2%
```

**Rationale:**
- Independent card/player sampling gives better coverage
- Ensures 10% at 1-3 cards (critical for "easy" round competence)
- Maintains 65% at 6-7 cards (strategic focus)
- Adds 3-player and 7-player coverage (currently missing)

### Recommendation 2: Add Low-Card Round Validation

**Add to Phase 4:**
```python
# ml/evaluation/low_card_validator.py
def validate_low_card_performance(model, encoder, masker):
    """Ensure model can handle 1-3 card rounds (easy but critical)."""
    results = {}
    for num_cards in [1, 2, 3]:
        # Generate 100 rounds with num_cards
        # Measure accuracy vs optimal play
        # Flag if accuracy < 95% (should be near-perfect!)
        results[num_cards] = accuracy
    return results
```

**Rationale:**
- All-or-nothing scoring means missing "easy" rounds is catastrophic
- Model MUST achieve >95% accuracy on 1-3 card rounds
- Separate validation ensures we don't overlook this

### Recommendation 3: Adjust MCTS Curriculum Timing

**Change:**
```diff
- Iterations 1-50:    1 det × 15 sims
- Iterations 51-150:  2 det × 25 sims
- Iterations 151-300: 3 det × 35 sims
- Iterations 301-450: 4 det × 45 sims
- Iterations 451-500: 5 det × 50 sims
+ Iterations 1-100:   1 det × 15 sims  (longer exploration)
+ Iterations 101-250: 2 det × 25 sims  (longer light phase)
+ Iterations 251-400: 3 det × 35 sims  (longer medium phase)
+ Iterations 401-475: 4 det × 45 sims  (shorter high phase)
+ Iterations 476-500: 5 det × 50 sims  (shorter max phase)
```

**Rationale:**
- Longer exploration phase (100 vs 50) builds better foundation
- Longer light/medium phases where most learning happens
- Shorter high-quality phases for fine-tuning
- Better matches typical RL learning curves (fast early, slow late)

### Recommendation 4: Add Mixed-Round Benchmarking

**Add to Phase 4:**
```bash
# benchmarks/performance/benchmark_mixed_rounds.py
# Test actual throughput with revised distribution
# Measure: rounds/min with 40% 7-card, 25% 6-card, etc.
# Compare to fixed 5-card baseline
```

**Rationale:**
- Need to validate 6.3 day estimate with actual benchmarks
- Distribution may affect GPU batching efficiency
- Important to know before committing to 500 iteration run

## Conclusion

**Use Mixed Rounds (Option C+) with Game Context Encoding + Hybrid Curriculum** 🎯

### Final Recommendation: Three-Tier Approach

**TIER 1: Curriculum-Aligned Card Distribution** ✅
```python
card_distribution = {
    1-3 cards: 30%,  # Simple (learn basics, avoid catastrophic failure)
    4-6 cards: 55%,  # Medium (peak learning zone)
    7 cards:   15%,  # Complex (strategic depth)
}
# Peak at 5-6 cards (medium complexity) for optimal transformer learning
# Balanced coverage across all complexity levels
```

**TIER 2: Game Context Encoding** ⚠️ CRITICAL
```python
# Add to StateEncoder BEFORE training starts:
state_features = [
    my_hand, trick_history, bids,  # Existing
    current_scores,                 # NEW: [25, 20, 15, 18]
    rounds_completed,               # NEW: 5 (on round 6 of 9)
    cards_previous_rounds,          # NEW: [5, 4, 3, 2, 1]
]
# Enables model to learn: "I'm ahead +20, play conservative"
# Solves reward alignment problem WITHOUT slow full games
```

**TIER 3: Hybrid Curriculum (Optional but Recommended)** 🎯
```
Phase 1 (400 iterations):  Independent rounds + context
  - 4.6 days @ 600 rounds/min
  - Learn basic strategy FAST

Phase 2 (100 iterations):  Full games (5,4,3,2,1,2,3,4,5)
  - 10 days @ 73 games/min
  - Fine-tune multi-round optimization
  - Validate: Does model maximize total score?

Total: 14.6 days (best of both worlds)
```

### Why This Approach Wins:

1. **Scoring System Demands It**:
   - All-or-nothing scoring means ALL round sizes equally important
   - 30% at 1-3 cards ensures competence on "easy" rounds
   - 55% at 4-6 cards is peak transformer learning zone
   - 15% at 7 cards provides strategic depth

2. **Transformer Learning Dynamics**:
   - Balanced distribution enables better attention pattern transfer
   - Peak at medium complexity (5-6 cards) matches curriculum learning
   - Avoids over-emphasizing extremes (neither 1-card nor 7-card dominates)
   - Model sees full range regularly for robust generalization

3. **Reward Alignment** (CRITICAL):
   - Game context encoding solves goal misalignment
   - Model learns: "I'm ahead +20, play differently"
   - No need for slow full games in Phase 1
   - Optional Phase 2 validates multi-round optimization

4. **Training Efficiency**:
   - Phase 1: 4.6 days (faster than any prior option!)
   - Phase 1+2: 14.6 days (vs 48 days for full games only)
   - Hybrid curriculum: 70% faster than Option B with better learning

5. **TOMS Integration Path**:
   - Phase 1 = TOMS iterations 1-400 (with revised distribution)
   - Phase 2 = TOMS iterations 401-500 (full games validation)
   - MCTS curriculum still applicable (1→5 determinizations)
   - Natural phase boundary for curriculum changes

### Immediate Actions (Priority Order):

**CRITICAL - Before Training Starts:**
1. ✅ **Add game context to StateEncoder** - Solves reward alignment
   - `current_scores: List[int]` (all player scores)
   - `rounds_completed: int` (0-8 for 9-round game)
   - `cards_trajectory: List[int]` (e.g., [5, 4, 3, 2, 1])

2. ✅ **Update `ml/config.py`** with revised card/player distributions
   - Curriculum-aligned: 30% low, 55% medium, 15% high
   - Player distribution: 50% 5p, 25% 4p, 20% 6p, 5% 3/7p

3. ✅ **Implement weighted sampling** in `ml/training/selfplay.py`
   - Sample cards from distribution (not fixed 5-card)
   - Sample players from distribution (not fixed 4-player)
   - Generate varied game contexts (30% early, 40% mid, 30% late game)

**SHORT TERM - Testing & Validation:**
4. ✅ **Run 5-10 iteration pilot** to validate encoding
   - Verify state encoding includes game context
   - Check: Do rounds vary in complexity (1-7 cards)?
   - Check: Do rounds vary in game context (scores, round number)?

5. ✅ **Benchmark mixed rounds** to validate 5.8 day estimate
   - Measure: rounds/min with new distribution
   - Compare: vs fixed 5-card (660 rounds/min baseline)

**MEDIUM TERM - Phase 1 Training:**
6. ✅ **Start Phase 1 training** (400 iterations, independent rounds)
   - Expected: 4.6 days @ 600 rounds/min
   - Monitor: Does model show context-dependent play?

7. ✅ **Evaluate after Phase 1**:
   - Play 100 full games (not just rounds)
   - Measure: Total score optimization vs round-only optimization
   - Decide: Is Phase 2 (full games) needed?

**LONG TERM - Phase 2 (if needed):**
8. ✅ **Implement full game training** (100 iterations)
   - Use full game sequences (5,4,3,2,1,2,3,4,5)
   - Reward: Total game score (true objective)
   - Can use Heavy MCTS (higher quality) since shorter phase

### TOMS.md Updates:

1. ✅ **Revise Phase 4 game distribution** (separate card/player sampling)
   - Card: 30% low, 55% medium, 15% high
   - Player: 50% 5p, 25% 4p, 20% 6p, 5% 3/7p

2. ✅ **Add game context encoding** (CRITICAL - do this first!)
   - Update StateEncoder before training starts
   - Generate rounds with varied contexts

3. ✅ **Split into two-phase curriculum**:
   - Phase 4a: Iterations 1-400 (independent rounds + context)
   - Phase 4b: Iterations 401-500 (full games validation)

4. ✅ **Add low-card round validation** to evaluation
   - Ensure >95% accuracy on 1-3 card rounds

5. ✅ **Adjust MCTS curriculum timing** (longer exploration)
   - Iterations 1-100: 1 det (fast exploration)
   - Iterations 101-250: 2 det (light refinement)
   - Iterations 251-400: 3 det (medium quality)
   - Iterations 401-500: 4-5 det (full games, high quality)

6. ✅ **Update time estimates**:
   - Phase 1: 4.6 days (independent rounds)
   - Phase 2: 10 days (full games, optional)
   - Total: 14.6 days (if both phases)

### Result:

**Strong, comprehensive model in 4.6 days (Phase 1 only) or 14.6 days (with Phase 2 validation):**
- ✅ Handles ALL round sizes (1-7 cards) with balanced coverage
- ✅ Variable player counts (3-7 players)
- ✅ Context-dependent strategy (learns from game scores)
- ✅ Reward-aligned (optimizes for total game score via context encoding)
- ✅ Transformer-friendly (curriculum-aligned distribution)
- ✅ Ready for TOMS MCTS curriculum
- ✅ 70% faster than full games (48 days → 14.6 days)

**Training can start as soon as game context encoding is implemented!**
