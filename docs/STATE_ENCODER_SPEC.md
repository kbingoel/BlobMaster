# State Encoder Specification (256 Dimensions)

**Version:** 1.0
**Last Updated:** 2025-11-15
**Python Reference:** [ml/network/encode.py](../ml/network/encode.py)

---

## Overview

The BlobNet neural network accepts a **256-dimensional float vector** encoding the current game state from a single player's perspective. This document provides the complete specification of all 256 dimensions to enable accurate cross-platform implementation (TypeScript/ONNX) that matches the Python training code exactly.

**Critical Implementation Notes:**
- All dimensions must match the Python implementation **exactly** (within 1e-6 tolerance)
- Use golden tests to verify cross-platform compatibility
- Most values normalized to [0, 1] range; exceptions noted below
- Supports 3-8 player games with padding for unused player slots

---

## Card Index Mapping (0-51 Scheme)

All card-related sections use consistent indexing:

**Formula:** `card_index = suit_index × 13 + rank_index`

**Suit Order:**
- 0: Spades (♠)
- 1: Hearts (♥)
- 2: Clubs (♣)
- 3: Diamonds (♦)

**Rank Order:**
- 0-12: [2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A]

**Index Ranges:**
```
0-12:   Spades   (♠2 through ♠A)
13-25:  Hearts   (♥2 through ♥A)
26-38:  Clubs    (♣2 through ♣A)
39-51:  Diamonds (♦2 through ♦A)
```

**Example:** Ace of Diamonds = suit 3 × 13 + rank 12 = 39 + 12 = **51**

---

## Complete Dimension Breakdown

### Section 1: My Hand (Dimensions 0-51)

**Size:** 52 dimensions
**Data Type:** Binary (0.0 or 1.0)
**Encoding:** One-hot style binary vector

| Dimension Range | Feature | Value |
|---|---|---|
| 0-51 | All 52 cards | 1.0 if card is in player's hand, 0.0 otherwise |

**Implementation:**
```python
for card in player.hand:
    card_idx = card_to_index(card)
    state[card_idx] = 1.0
```

**Notes:**
- Sparse encoding: typically only 1-13 dimensions are 1.0 (hand size varies)
- Remaining dimensions are 0.0 (cards not in hand)

---

### Section 2: Cards Played This Trick (Dimensions 52-103)

**Size:** 52 dimensions
**Data Type:** Ordinal/Sequential (0.0 or 1.0-8.0)
**Encoding:** Play order indicator

| Dimension Range | Feature | Value |
|---|---|---|
| 52-103 | All 52 cards | 0 = not played this trick<br/>1-8 = position played in trick (1st, 2nd, ..., 8th) |

**Implementation:**
```python
if current_trick:
    for play_position, (player, card) in enumerate(current_trick.cards_played, start=1):
        card_idx = card_to_index(card)
        state[52 + card_idx] = float(play_position)
```

**Notes:**
- Encodes **which player in turn order** played each card
- Max value = number of players (3-8)
- Most dimensions remain 0.0 (only 1-8 cards played per trick)

---

### Section 3: All Cards Played This Round (Dimensions 104-155)

**Size:** 52 dimensions
**Data Type:** Binary (0.0 or 1.0)
**Encoding:** Cumulative history

| Dimension Range | Feature | Value |
|---|---|---|
| 104-155 | All 52 cards | 1.0 if card has been played in ANY trick this round, 0.0 otherwise |

**Implementation:**
```python
for card in cards_played_this_round:
    card_idx = card_to_index(card)
    state[104 + card_idx] = 1.0
```

**Notes:**
- Accumulates over all completed tricks in current round
- Helps AI track which cards are still "live" in opponents' hands
- Complements Section 1 (my hand) to deduce opponent holdings

---

### Section 4: Player Bids (Dimensions 156-163)

**Size:** 8 dimensions (supports max 8 players)
**Data Type:** Normalized float
**Encoding:** Per-player bid values

| Dimension | Feature | Normalization |
|---|---|---|
| 156-163 | Bids for players 0-7 | `bid / cards_dealt` → [0, 1]<br/>**-1.0** if player hasn't bid yet<br/>**0.0** for unused player slots (if <8 players) |

**Implementation:**
```python
for i in range(8):
    if i < num_players:
        if players[i].bid is not None:
            state[156 + i] = players[i].bid / cards_dealt
        else:
            state[156 + i] = -1.0  # Not yet bid
    else:
        state[156 + i] = 0.0  # Padding for unused players
```

**Notes:**
- **-1.0 is special value** indicating "no bid yet" (not in [0,1] range)
- Normalized by cards dealt (e.g., bid of 3 with 5 cards → 0.6)
- Padding with 0.0 for games with fewer than 8 players

---

### Section 5: Player Tricks Won (Dimensions 164-171)

**Size:** 8 dimensions (supports max 8 players)
**Data Type:** Normalized float
**Encoding:** Per-player trick counts

| Dimension | Feature | Normalization |
|---|---|---|
| 164-171 | Tricks won by players 0-7 | `tricks_won / cards_dealt` → [0, 1]<br/>**0.0** for unused player slots |

**Implementation:**
```python
for i in range(8):
    if i < num_players:
        state[164 + i] = players[i].tricks_won / cards_dealt if cards_dealt > 0 else 0.0
    else:
        state[164 + i] = 0.0  # Padding
```

**Notes:**
- Normalized by cards dealt (e.g., 2 tricks with 5 cards → 0.4)
- Always in [0, 1] range (no special values)
- Critical for AI to assess if players are on pace with their bids

---

### Section 6: My Bid (Dimension 172)

**Size:** 1 dimension
**Data Type:** Normalized float
**Encoding:** Current player's bid

| Dimension | Feature | Normalization |
|---|---|---|
| 172 | My bid | `bid / cards_dealt` → [0, 1]<br/>**-1.0** if not yet bid |

**Implementation:**
```python
if player.bid is not None:
    state[172] = player.bid / cards_dealt
else:
    state[172] = -1.0
```

**Notes:**
- Duplicates information from Section 4 for convenience
- Allows AI to quickly access its own bid without searching player array

---

### Section 7: My Tricks Won (Dimension 173)

**Size:** 1 dimension
**Data Type:** Normalized float
**Encoding:** Current player's trick count

| Dimension | Feature | Normalization |
|---|---|---|
| 173 | My tricks won | `tricks_won / cards_dealt` → [0, 1] |

**Implementation:**
```python
state[173] = player.tricks_won / cards_dealt if cards_dealt > 0 else 0.0
```

**Notes:**
- Duplicates information from Section 5 for convenience
- Critical for bid tracking and strategy

---

### Section 8: Round Metadata (Dimensions 174-181)

**Size:** 8 dimensions
**Data Type:** Mixed (normalized floats + one-hot)
**Encoding:** Core game state information

| Dimension | Feature | Normalization | Range |
|---|---|---|---|
| 174 | Cards dealt | `cards_dealt / 13.0` | [0, 1] |
| 175 | Current trick number | `completed_tricks / cards_dealt` | [0, 1] |
| 176 | My position relative to dealer | `(my_pos - dealer_pos) % num_players / num_players` | [0, 1) |
| 177 | Number of players | `num_players / 8.0` | [0.375, 1.0] |
| 178-181 | Trump suit (one-hot, 4-dim) | See below | Binary |

**Trump Suit Encoding (Dimensions 178-181):**
- **178:** 1.0 if trump is Spades (♠), else 0.0
- **179:** 1.0 if trump is Hearts (♥), else 0.0
- **180:** 1.0 if trump is Clubs (♣), else 0.0
- **181:** 1.0 if trump is Diamonds (♦), else 0.0
- **All zeros:** No-trump round

**Implementation:**
```python
state[174] = cards_dealt / 13.0
state[175] = len(tricks_history) / cards_dealt if cards_dealt > 0 else 0.0
state[176] = ((player.position - dealer_position) % num_players) / num_players
state[177] = num_players / 8.0

# Trump suit one-hot
if trump_suit is not None:
    trump_idx = SUITS.index(trump_suit)
    state[178 + trump_idx] = 1.0
# else: all zeros (already initialized)
```

**Notes:**
- Position relative to dealer: 0.0 = first after dealer, approaching 1.0 = dealer
- Trump encoding: exactly one dimension = 1.0, or all zeros for no-trump
- Max cards dealt = 13 (one full suit per player)

---

### Section 9: Bidding Constraint (Dimension 182)

**Size:** 1 dimension
**Data Type:** Binary
**Encoding:** Dealer constraint indicator

| Dimension | Feature | Value |
|---|---|---|
| 182 | Is dealer constraint active? | 1.0 if current player is dealer AND in bidding phase<br/>0.0 otherwise |

**Implementation:**
```python
state[182] = 1.0 if (player.position == dealer_position and game_phase == 'bidding') else 0.0
```

**Notes:**
- Signals AI that forbidden bid rule applies to this decision
- Forbidden bid = `cards_dealt - sum(other_bids)` (makes total equal cards dealt)
- Only relevant during bidding phase

---

### Section 10: Game Phase (Dimensions 183-185)

**Size:** 3 dimensions
**Data Type:** One-hot binary
**Encoding:** Current game phase

| Dimension | Feature | Value |
|---|---|---|
| 183 | Bidding phase | 1.0 if `game_phase == 'bidding'`, else 0.0 |
| 184 | Playing phase | 1.0 if `game_phase == 'playing'`, else 0.0 |
| 185 | Scoring/Complete | 1.0 if `game_phase in ['scoring', 'complete']`, else 0.0 |

**Implementation:**
```python
state[183] = 1.0 if game_phase == 'bidding' else 0.0
state[184] = 1.0 if game_phase == 'playing' else 0.0
state[185] = 1.0 if game_phase in ['scoring', 'complete'] else 0.0
```

**Notes:**
- Exactly one dimension = 1.0 at any time
- Helps AI context-switch between bidding and card-playing strategy

---

### Section 11: Positional Features (Dimensions 186-201)

**Size:** 16 dimensions
**Data Type:** Normalized floats (with 11 reserved dimensions)
**Encoding:** Tactical position information

| Dimension | Feature | Normalization | Range |
|---|---|---|---|
| 186 | Cards in hand | `cards_in_hand / 13.0` | [0, 1] |
| 187 | Cards played by me | `cards_played / 13.0` | [0, 1] |
| 188 | Round progress | `completed_tricks / total_tricks` | [0, 1] |
| 189 | Am I on pace? | `(tricks_won - bid) / cards_dealt`<br/>**0.0** if no bid yet | [-1, 1] |
| 190 | Play order in trick | `cards_in_current_trick / num_players` | [0, 1] |
| 191-201 | **Reserved** | Placeholder for future features | **0.0** |

**Implementation:**
```python
state[186] = len(player.hand) / 13.0
state[187] = len(player.cards_played) / 13.0
state[188] = len(tricks_history) / cards_dealt if cards_dealt > 0 else 0.0

# On-pace metric: positive = ahead of bid, negative = behind
if player.bid is not None and cards_dealt > 0:
    state[189] = (player.tricks_won - player.bid) / cards_dealt
else:
    state[189] = 0.0

# Play position in current trick
if current_trick:
    state[190] = len(current_trick.cards_played) / num_players
else:
    state[190] = 0.0

# Reserved dimensions (191-201)
state[191:202] = 0.0
```

**Notes:**
- **Dimension 189** can be negative (behind bid) or positive (ahead of bid)
- Reserved dimensions (191-201) allow future extensibility without breaking model compatibility
- Total 11 reserved dimensions in this section

---

### Section 12: Game Context Features (Dimensions 202-255)

**Size:** 54 dimensions
**Data Type:** Mixed (mostly normalized floats)
**Encoding:** Multi-round game context (OPTIONAL)

**IMPORTANT:** This section is **only populated** if `game_context` parameter is provided to `encode()`. Otherwise, all 54 dimensions are **0.0**. For Phase 1 training (independent rounds), this section remains zero.

---

#### Subsection 12a: Cumulative Scores (Dimensions 202-209)

**Size:** 8 dimensions
**Encoding:** Running total scores for all players

| Dimension | Feature | Normalization |
|---|---|---|
| 202-209 | Total scores for players 0-7 | `score / (rounds_completed × (10 + start_cards))`<br/>**0.0** if `rounds_completed == 0` or unused player |

**Implementation:**
```python
if game_context and rounds_completed > 0:
    max_per_round = 10 + start_cards
    for i in range(8):
        if i < num_players:
            state[202 + i] = players[i].total_score / (rounds_completed * max_per_round)
        else:
            state[202 + i] = 0.0
else:
    state[202:210] = 0.0
```

**Notes:**
- Normalization assumes max score per round ≈ 10 (base) + starting cards
- Not clamped at 1.0 (allows signal when player significantly ahead/behind)
- Zero if no rounds completed (division by zero protection)

---

#### Subsection 12b: Round Position (Dimensions 210-213)

**Size:** 4 dimensions
**Encoding:** Game progression metrics

| Dimension | Feature | Normalization |
|---|---|---|
| 210 | Rounds completed | `rounds_completed / total_rounds` | [0, 1] |
| 211 | Rounds remaining | `rounds_remaining / total_rounds` | [0, 1] |
| 212 | One-card rounds left | `count_ones_remaining / num_players` | [0, 1] |
| 213 | Game phase code | 0.0 = descending (C→1)<br/>0.5 = one-card plateau<br/>1.0 = ascending (1→C) |

**Implementation:**
```python
if game_context:
    state[210] = rounds_completed / total_rounds
    state[211] = rounds_remaining / total_rounds
    state[212] = count_ones_remaining / num_players  # Num 1-card rounds left

    # Phase encoding
    if in_descending_phase:
        state[213] = 0.0
    elif in_ones_plateau:
        state[213] = 0.5
    elif in_ascending_phase:
        state[213] = 1.0
else:
    state[210:214] = 0.0
```

**Notes:**
- Game structure: C → C-1 → ... → 1 → 1 → ... → 1 → 2 → ... → C
- "Ones plateau" = P consecutive rounds with 1 card each (where P = num_players)
- Helps AI adjust strategy for early vs late game

---

#### Subsection 12c: Previous Round History (Dimensions 214-228)

**Size:** 15 dimensions
**Encoding:** Recent card count history

| Dimension | Feature | Normalization |
|---|---|---|
| 214-228 | Card counts for last 15 rounds | `cards_dealt / start_cards` → [0, 1]<br/>**0.0** if round not yet played (padding) |

**Implementation:**
```python
if game_context:
    recent_rounds = previous_card_counts[-15:]  # Last 15 rounds only
    for i in range(15):
        if i < len(recent_rounds):
            state[214 + i] = recent_rounds[i] / start_cards
        else:
            state[214 + i] = 0.0  # Padding if <15 rounds played
else:
    state[214:229] = 0.0
```

**Notes:**
- Stores only MOST RECENT 15 rounds (sliding window)
- Left-padded with zeros if fewer than 15 rounds completed
- Helps AI recognize patterns (e.g., "we just played three 1-card rounds")

---

#### Subsection 12d: Game Configuration (Dimensions 229-230)

**Size:** 2 dimensions
**Encoding:** Fixed game parameters

| Dimension | Feature | Normalization |
|---|---|---|
| 229 | Number of players | `num_players / 8.0` | [0.375, 1.0] |
| 230 | Starting card count | `start_cards / 8.0` | [0.125, 1.0] |

**Implementation:**
```python
if game_context:
    state[229] = num_players / 8.0
    state[230] = start_cards / 8.0  # Max starting cards typically 7-8
else:
    state[229:231] = 0.0
```

**Notes:**
- Duplicates some info from Section 8 but in multi-game context
- Allows AI to adapt strategy based on game variant

---

#### Subsection 12e: Reserved Future Use (Dimensions 231-255)

**Size:** 25 dimensions
**Encoding:** Placeholder for future features

| Dimension | Feature | Value |
|---|---|---|
| 231-255 | Reserved | **All zeros** |

**Implementation:**
```python
state[231:256] = 0.0  # Reserved for future use
```

**Notes:**
- Provides 25 dimensions for future model extensions
- Maintains fixed 256-dim input size (no retraining needed for new features)
- Possible future features: opponent modeling, meta-game stats, time pressure, etc.

---

## Dimension Accounting Summary

| Section | Dimensions | Count | Description |
|---------|-----------|-------|-------------|
| 1 | 0-51 | 52 | My Hand (binary) |
| 2 | 52-103 | 52 | Cards Played This Trick (ordinal) |
| 3 | 104-155 | 52 | All Cards Played This Round (binary) |
| 4 | 156-163 | 8 | Player Bids (normalized, -1 = no bid) |
| 5 | 164-171 | 8 | Player Tricks Won (normalized) |
| 6 | 172 | 1 | My Bid (normalized, -1 = no bid) |
| 7 | 173 | 1 | My Tricks Won (normalized) |
| 8 | 174-181 | 8 | Round Metadata (cards dealt, trick#, position, players, trump) |
| 9 | 182 | 1 | Bidding Constraint (dealer forbidden bid) |
| 10 | 183-185 | 3 | Game Phase (one-hot: bidding/playing/scoring) |
| 11 | 186-201 | 16 | Positional Features (5 used + 11 reserved) |
| 12a | 202-209 | 8 | Cumulative Scores (multi-round context) |
| 12b | 210-213 | 4 | Round Position (multi-round context) |
| 12c | 214-228 | 15 | Previous Round History (multi-round context) |
| 12d | 229-230 | 2 | Game Configuration (multi-round context) |
| 12e | 231-255 | 25 | Reserved for Future Use |
| **TOTAL** | **0-255** | **256** | **Complete state vector** |

**Breakdown:**
- **Base features** (always populated): Sections 1-11 = **202 dimensions**
- **Game context** (optional, Phase 2+): Section 12 = **54 dimensions**
  - Used: 29 dimensions (12a-12d)
  - Reserved: 25 dimensions (12e)

---

## Implementation Guidelines

### TypeScript Port Requirements

When implementing the state encoder in TypeScript (for backend/frontend):

1. **Exact Match Required:** Output must match Python within **1e-6 tolerance**
2. **Use Golden Tests:** Generate test cases from Python, validate TypeScript output
3. **Preserve Special Values:** -1.0 for "no bid yet" (not 0.0 or null)
4. **Consistent Indexing:** Use same card indexing formula (suit × 13 + rank)
5. **Normalization Formulas:** Copy exact normalization from Python (no "close enough")

### Testing Strategy

**Golden Test Generation (Python):**
```python
# Generate 100 diverse game states
for seed in range(100):
    game = create_test_game(seed)
    player = game.players[0]
    state = encoder.encode(game, player)

    # Save to JSON for TypeScript validation
    save_golden_test({
        'seed': seed,
        'state_vector': state.tolist(),
        'game_snapshot': serialize_game(game)
    })
```

**TypeScript Validation:**
```typescript
// Load golden test cases
for (const testCase of goldenTests) {
    const game = deserializeGame(testCase.game_snapshot);
    const player = game.players[0];
    const tsState = encoder.encode(game, player);

    // Compare with Python output
    for (let i = 0; i < 256; i++) {
        const diff = Math.abs(tsState[i] - testCase.state_vector[i]);
        expect(diff).toBeLessThan(1e-6);  // Strict tolerance
    }
}
```

### Common Pitfalls

1. **Forgetting -1.0 for unbid players:** Use 0.0 instead of -1.0 → breaks bidding logic
2. **Wrong card indexing:** Off-by-one errors in suit/rank mapping
3. **Integer division:** Use float division for normalization (e.g., `3/5` not `3//5`)
4. **Skipping game context:** Forgetting to zero out dimensions 202-255 when `game_context` is null
5. **Hardcoding player count:** Not padding unused player slots with zeros

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-15 | Initial comprehensive specification based on ml/network/encode.py |

---

## References

- **Python Implementation:** [ml/network/encode.py](../ml/network/encode.py)
- **Neural Network Model:** [ml/network/model.py](../ml/network/model.py)
- **Game Engine:** [ml/game/blob.py](../ml/game/blob.py)
- **Training Config:** [ml/config.py](../ml/config.py)
