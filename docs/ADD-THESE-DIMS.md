# Additional State Encoder Dimensions - Phase 1A

**Status:** Ready for implementation before 500-iteration training run
**Target:** Dimensions 191-201 (11 reserved slots in Section 11: Positional Features)
**Expected Impact:** 1-1.5 day training time reduction (25-30% faster convergence to ELO 1400)

---

## Features to Add (11 dimensions)

### **Dimension 191: Trump Count**
**Priority:** CRITICAL ⭐⭐⭐⭐⭐

Count of trump cards in player's hand, normalized by 13.0. Handles no-trump rounds with 0.0.

**Why:** Requires scanning 52 hand dimensions + cross-referencing trump suit (dims 178-181). Computationally expensive for transformer to derive. Trump count is the single most important bidding factor.

**Expected impact:** +10-15% bidding accuracy, +15% convergence speed

---

### **Dimensions 192-195: Suit Distribution**
**Priority:** CRITICAL ⭐⭐⭐⭐⭐

Cards per suit (♠♥♣♦), each normalized by 13.0. Four dimensions encoding [spades_count, hearts_count, clubs_count, diamonds_count].

**Why:** Long suits (5+ cards), short suits (1-2 cards), and voids (0 cards) are fundamental to trick-taking strategy. Requires scanning 13 non-contiguous dimensions per suit (52 total lookups).

**Expected impact:** +8-12% bidding accuracy, +5-7% play accuracy, +20% convergence speed

---

### **Dimensions 196-199: High Card Indicators (Ace/King)**
**Priority:** HIGH ⭐⭐⭐⭐

Ace and King ownership per suit (♠♥♣♦). Encoded as: 0.0 = neither, 0.5 = King only, 1.0 = Ace (King presence implied if 0.5).

**Why:** Aces win ~90% of tricks, Kings win ~60%. Explicitly highlights trick-winning potential rather than forcing network to learn rank values implicitly. Speeds up early training.

**Expected impact:** +5-8% bidding accuracy, +2-4% play accuracy, +10% convergence speed

---

### **Dimension 200: Trump Strength (Average Rank)**
**Priority:** HIGH ⭐⭐⭐⭐

Average rank of trump cards held, normalized by 12.0 (Ace = highest rank). Returns 0.0 if no trump cards or no-trump round.

**Why:** Trump COUNT (dim 191) tells quantity, trump STRENGTH tells quality. Five low trump cards (2-3-4-5-6) vs five high trump cards (A-K-Q-J-10) require completely different bidding strategies. COUNT + STRENGTH together provide complete trump assessment.

**Expected impact:** +3-5% bidding accuracy, especially in marginal bidding decisions

---

### **Dimension 201: Dealer Forbidden Bid**
**Priority:** MEDIUM-HIGH ⭐⭐⭐⭐

The forbidden bid value for dealer, normalized by cards dealt. Returns -1.0 if not dealer or not in bidding phase. Calculated as: forbidden_bid = cards_dealt - sum(other_players_bids).

**Why:** Dealer constraint is critical rule enforcement. Requires checking position (dim 182) + summing bids (dims 156-163) + arithmetic. Explicit encoding reduces early training violations and helps network learn dealer-specific strategy faster.

**Expected impact:** +2-3% bidding accuracy for dealer position (~12.5% of decisions)

---

## Implementation Priority

**Must implement (9 dims):**
- Trump count (1)
- Suit distribution (4)
- High card indicators (4)

**Strongly recommended (2 dims):**
- Trump strength (1)
- Dealer forbidden bid (1)

**Total: 11 dimensions** - Perfectly fills reserved space 191-201

---

## Testing Requirements

1. **Unit tests:** Verify each dimension produces expected values for known game states
2. **Range validation:** Ensure all values fall within expected normalized ranges
3. **Edge cases:** Test no-trump rounds, dealer position, void suits, no aces/kings
4. **Fast config test:** Run 5-iteration training to validate encoding doesn't break pipeline
5. **Golden test generation:** Create test cases for future TypeScript/ONNX port validation

---

## Expected Training Improvements

**Baseline (current 256-dim encoding):**
- Time to ELO 1400: ~5 days with curriculum
- Bidding accuracy @ iter 100: ~65%
- Play accuracy @ iter 100: ~72%
- Final ELO @ iter 500: 1400-1600

**With Phase 1A features (11 dims added):**
- Time to ELO 1400: ~3.5-4 days ⚡ (1-1.5 day savings)
- Bidding accuracy @ iter 100: ~78-85% (+13-20%)
- Play accuracy @ iter 100: ~77-81% (+5-9%)
- Final ELO @ iter 500: 1600-1850 (+200-450)

**Conservative estimate:** Save 24-36 hours of GPU training time

---

## Risk Assessment

**Low risk because:**
- ✅ Fits in reserved space (no breaking architectural changes)
- ✅ All features are deterministic (easy to test, no randomness)
- ✅ Features are aggregations, not predictions (counts/averages of existing data)
- ✅ No additional trainable parameters added to network

**Maintenance considerations:**
- ⚠️ Must port identical logic to TypeScript for production inference (Phase 6)
- ⚠️ ONNX export validation must verify all 11 new dimensions
- ⚠️ Unit tests increase from ~460 to ~480 total

---

## Post-Training Validation

After 100 iterations, conduct ablation study:
1. Train baseline model WITHOUT Phase 1A features (same config otherwise)
2. Compare convergence curves, bidding/play accuracy, and ELO trajectory
3. Measure actual time-to-ELO-1400 for both models
4. Document which features provided most value

If ELO < 1600 at iteration 500, consider Phase 1B additions:
- Void suits explicit (4 dims, binary flags)
- Current trick winnability (1 dim, "can I win this trick?")
- Opponent trick pressure (8 dims, per-player on-pace metrics)

---

## Related Documentation

- **EXTRA-DIM.md** - Full catalog of 17 potential features across 5 tiers
- **STATE_ENCODER_SPEC.md** - Complete 256-dimension encoding specification
- **ml/network/encode.py** - StateEncoder implementation (modify `_encode_positional_features`)
- **CLAUDE.md** - Project overview and Phase 1 training context

---

## Decision: Implement These 11 Dimensions?

**Recommendation: YES** ✅

**Rationale:** 1-1.5 day training time savings justifies 1 day of implementation effort. Features target computationally expensive aggregations (counting across non-contiguous dimensions) that transformers struggle to learn efficiently in early training.

**Next steps:**
1. Modify `ml/network/encode.py` → `_encode_positional_features()` method
2. Update `STATE_ENCODER_SPEC.md` with new dimension mappings
3. Write unit tests in `ml/network/test_network.py`
4. Run fast config validation (5 iterations)
