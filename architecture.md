# Structured Entity Transformer — Architecture Specification

**Date**: 2026-03-15
**Supersedes**: Legacy BlobNet (6-layer Transformer on 256-dim flat vector, ~4.9M params)
**New architecture**: Structured Entity Transformer (~1.6M params)

---

Given your deep understanding of the game, critically propose improvements to this plan, as we will use it to derive a outline document that guides us in working out these ideas into code. Question the choice of embeddings, input token dimensions, etc. You can agree with the choices if you think they are genuinely good.
You can always refer to the rules of the game as outlined in prepare-migration.md line 102-130.

## 1. Game Dynamics and Information Flow

### 1.1 Progressive Information Revelation

Blob's defining characteristic is how information transforms from hidden to known over the course of a round. This progression — not any single snapshot — is the primary basis for expert play.

Consider a concrete 5-player, 7-card round with hearts as trump:

**Before trick 1** — Maximum uncertainty. You know 7 cards (your hand). 28 cards are distributed across 4 opponents, but you know nothing about who holds what. Your decisions rely almost entirely on hand-strength heuristics: trump count, high cards, suit length.

**After tricks 1-2** — Critical signals emerge. You've observed 10 cards played (8 by opponents). Each play reveals information on two levels:
- **What was played**: Card counting narrows the remaining possibilities
- **How it was played**: A card *led* was chosen freely (revealing strength in that suit); a card played to *follow suit* was constrained (less informative); a card played *off-suit* reveals a permanent void

A single off-suit play (Player 3 plays a diamond when spades were led) creates a constraint that persists for the entire round: Player 3 has zero spades. This eliminates roughly 25% of possible hand distributions for that opponent in one observation.

**After tricks 3-4** — Card counting becomes powerful. 20 cards observed + 3-4 in hand = 23-24 known cards. Combined with suit void constraints accumulated over 4 tricks, the distribution of opponent hands narrows considerably. However, 17 cards remain unseen — distributed across opponent hands and, critically, the 17 undealt cards that will never appear in this round. An unplayed card could be in any opponent's hand or simply absent from the game entirely. This ambiguity places a hard ceiling on deductive certainty even with strong constraint propagation.

**After tricks 5-6** — Void constraints and card counting reduce the uncertainty significantly, but the undealt cards continue to impose fundamental limits. When Player 3 hasn't played a spade, you know they're either void or holding spades — but you cannot determine which without further evidence. The game becomes more tactical as possibilities narrow, but rarely reaches the point of fully known opponent hands.

### 1.2 The Information Gradient

This progression creates an **information gradient**: the value of different reasoning capabilities shifts as the game progresses.

| Game Stage | Dominant Reasoning | Key Information |
|---|---|---|
| Bidding | Hand-strength estimation | My cards, trump suit, position, others' bids |
| Tricks 1-2 | Probabilistic hand ranging | Which cards played, who led, who followed |
| Tricks 3-4 | Constraint propagation | Suit voids, card counting, bid progress |
| Tricks 5-7 | Constrained probabilistic reasoning | Narrowed hand distributions, persistent uncertainty from undealt cards |

The optimal network must handle all four stages with a single architecture. This means it must process the *entire history of play* — not just the current trick — because the accumulated evidence from earlier tricks is what enables the constraint propagation and deduction in later tricks.

---

## 2. Strategic Patterns in Sequential Play

Five strategic patterns depend on sequential information that is only available through the ordered history of card plays. These patterns represent the core skills that separate expert from novice Blob players.

### 2.1 Card Counting with Play Attribution

Card counting — tracking which specific cards remain unplayed — is the foundation of trick-taking strategy. But raw card counting ("the Ace of Spades has been played") is far less valuable than *attributed* card counting ("Player 2 played the Ace of Spades as a lead in trick 1").

Attribution reveals:
- **Who has strength in which suits**: Player 2 leading the Ace of Spades suggests spade strength and aggressive trick-winning intent
- **Who is exhausting which suits**: If Player 2 played 3 spades in 4 tricks, they likely have 0-1 spades remaining
- **Remaining card distribution per opponent**: Combining card counts with attribution narrows each opponent's hand to specific possibilities

Example: You hold the Queen of Spades. Is it safe to lead? With unattributed counting, you know the King and Ace are both gone — your Queen is high. With attributed counting, you know Player 4 played both the King and Ace in tricks 1 and 2 — they're out of high spades. But Player 3 played two low spades — they might still hold the Jack. This distinction affects whether you lead the Queen now or wait.

The full play history with per-card player attribution enables the network to learn these patterns directly from the data. Each played card carries its identity, its player, its trick context, and whether it was a lead or follow — all the raw material needed for attributed card counting.

### 2.2 Suit Void Detection and Exploitation

When a player fails to follow the led suit, they reveal a permanent constraint: they hold zero cards of that suit for the remainder of the round. This is the single most informative event in trick-taking games.

A suit void has cascading strategic implications:
- **For the voided player**: They can now trump that suit (if they have trump) or dump unwanted cards when that suit is led
- **For opponents**: Leading that suit "through" the voided player becomes either dangerous (they might trump) or safe (they'll dump low)
- **For hand deduction**: Eliminating an entire suit from a player's possibilities dramatically constrains their remaining hand

Critically, **when** a void is revealed matters as much as **that** it exists:
- **Trick 1 void**: The player was dealt zero cards of that suit. Strong signal — their entire hand is concentrated in 3 suits
- **Trick 5 void**: The player played through all their cards in that suit over 5 tricks. Weaker signal — they may have had 4-5 cards of that suit initially, and their remaining hand composition depends on what they played

The trick number of the void-revealing play, the card played instead (trump vs. low off-suit), and the player's bid all combine to form a rich inference about their remaining hand and strategy. This is inherently sequential information — the moment of revelation within the ordered play history is critical context.

### 2.3 Contextual Hand Strength Evaluation

A card's strength is not intrinsic — it depends entirely on what has been played before. The King of Hearts means different things at different points in the game:

- **Trick 1**: The Ace is probably out there. The King is strong but vulnerable. Lead it only if you want to force the Ace out.
- **After the Ace of Hearts was played in trick 2**: Your King is now the highest heart. It's a guaranteed trick-winner when hearts are led.
- **After 3 of 4 opponents showed heart voids**: Only one opponent can beat your King with the Ace (if they still have it). Your King is very likely safe.
- **No-trump round vs. hearts-as-trump**: In a no-trump round, the King of Hearts wins heart tricks. In a hearts-as-trump round, the King of Hearts also wins against non-trump leads (by trumping).

These evaluations require cross-referencing your hand cards against the full history of played cards. "Is my card now the highest remaining in its suit?" is answered by attending to all previously played cards of the same suit and checking whether higher-ranked cards are among them.

This is a natural attention pattern: a hand card token queries all played card tokens of the same suit, checking which ranks have been removed. The result tells the hand card whether it's now dominant, vulnerable, or irrelevant. This computation is structurally native to attention — it's exactly what query-key matching was designed for.

### 2.4 Bid-Progress Tracking and Opponent Modeling

Each player's bid creates a contract: win exactly that many tricks. As the game progresses, each player's bid fulfillment status drives their strategy:

- **On pace** (tricks_won matches expected rate): Continue playing normally
- **Ahead of bid** (won too many tricks too early): Switch to defensive play — duck tricks, dump high cards, avoid winning
- **Behind bid** (need more tricks than remain): Bid is already impossible — play becomes unpredictable (nothing to lose)
- **Bid exactly met with tricks remaining**: Pure avoidance — must lose every remaining trick

The *trajectory* of bid fulfillment — not just the current count — reveals opponent strategy:

Example: Player 2 bid 3. After trick 4 of 7, they've won 3 tricks.
- If they won tricks 1, 2, 3 (frontloaded): They led aggressively early, likely have weaker cards now. They'll successfully dodge remaining tricks.
- If they won tricks 2, 3, 4 (recent cluster): They may still hold strong cards and are at risk of winning unwanted tricks. Pressure them by leading their strong suits.

The ordered play history reveals this trajectory. Player state tokens carry the bid and tricks-won count (the "what"), while the played card tokens carry the trick-by-trick attribution (the "how" and "when").

### 2.5 Multi-Trick Planning

Expert Blob players execute plans spanning multiple tricks:

**Void creation**: "I have 2 hearts and 3 trumps. I'll play my hearts in tricks 1-2 to create a heart void, then trump opponents' hearts from trick 3 onward." This plan requires recognizing that short-term losses (playing hearts that might lose) enable long-term gains (trumping later).

**Trump exhaustion**: "I have 4 trumps including the Ace and King. I'll lead trumps for 2-3 tricks to drain opponents' trump holdings. Once they're out of trump, my off-suit Queens are safe to lead."

**Duck-and-grab**: "I bid 2 out of 7. I'll play low for the first 5 tricks (ducking), then grab the last 2 with my Ace and King when I know exactly what opponents hold."

These plans require the network to maintain coherence across decisions. A snapshot encoding treats each decision independently — the network doesn't know whether it's executing a void creation plan or randomly playing hearts. The sequential history of its own previous plays provides the context for continuing or abandoning a multi-trick plan.

---

## 3. Sequential Information and MCTS Efficiency

### 3.1 Prior Quality and Search Convergence

In AlphaZero-style MCTS, the neural network serves two roles:
1. **Policy prior**: Initial probability distribution over actions, used to guide exploration
2. **Value estimate**: Expected game outcome from a given state, used to evaluate leaf nodes

The policy prior determines how efficiently MCTS explores the game tree. With a budget of 100 simulations per determinization and ~10 legal actions per move:

**With a weak prior** (near-uniform distribution):
- Each action receives ~10 visits initially
- UCB1 exploration dominates: the exploration bonus `c_puct × P × √N_parent / (1 + N_child)` is similar across all actions because priors P are similar
- After 100 sims, visit counts remain noisy — the top action might have 15 visits vs. an average of 10
- The resulting visit distribution (the MCTS policy target) is nearly uniform
- Training on near-uniform targets teaches the network to output... near-uniform priors
- **Vicious cycle**: weak prior → unfocused search → uniform targets → weak prior

**With a strong prior** (concentrated on good actions):
- MCTS allocates 50-60 visits to the best 2-3 actions, 5-10 to the rest
- The best actions are evaluated deeply — their value estimates converge to ground truth
- Exploration budget is spent confirming the prior rather than wasting visits on clearly bad actions
- After 100 sims, the top action has 40+ visits vs. an average of 10
- The resulting visit distribution has strong signal — clear peaks on good moves
- Training on peaked targets teaches the network to refine its prior further
- **Virtuous cycle**: strong prior → focused search → peaked targets → stronger prior

The critical threshold is whether the prior is good enough to make MCTS converge within the simulation budget. For Blob's 5×100 configuration (500 total evaluations per move), the prior needs to rank the best action in the top 3 out of ~10 legal actions for MCTS to reliably identify it.

### 3.2 Value Accuracy and Leaf Evaluation

The value head's accuracy directly determines MCTS search quality. When MCTS reaches a leaf node and queries the neural network for a value estimate:

- **Accurate value** → the backed-up values in the tree reflect true game prospects → MCTS correctly identifies winning lines
- **Inaccurate value** → backed-up values are noisy → MCTS may pursue losing lines and ignore winning ones → more simulations needed to overcome noise

A network that understands the sequential game dynamics produces better value estimates because it can reason about:
- "Player 3 has shown void in my strong suit — my high cards are unsafe" → lower value
- "I've exhausted opponents' trumps by leading trump twice — my off-suit Aces are now guaranteed" → higher value
- "Player 2's bid is already busted — they'll play unpredictably, adding variance" → value closer to 0

These inferences, drawn from the full play history, improve value accuracy beyond what a snapshot encoding can achieve.

### 3.3 The Feedback Loop

The training loop creates a compounding relationship between network quality and training data quality:

```
Better network (from structured input)
  → Better policy priors and value estimates
    → MCTS converges faster with fewer simulations
      → Visit distributions have stronger signal
        → Training targets are more informative
          → Network improves faster
            → Cycle accelerates
```

This feedback loop means that an architectural improvement to the network doesn't just improve play quality linearly — it accelerates the entire learning process. A network that produces 20% better priors might produce 50% faster ELO growth because the MCTS training signal is exponentially more useful.

The structured entity approach, by giving the network direct access to the sequential play history, crosses the critical threshold where priors are good enough for MCTS to converge within budget — breaking the vicious cycle that stalled the legacy implementation.

---

## 4. Architecture: Structured Entity Transformer

### 4.1 Design Principles

The architecture represents the game state as a set of typed entities rather than a flat feature vector. Each entity — a card in hand, a card played, a player's state, the global context — becomes a token with domain-specific features. The Transformer processes these tokens through self-attention, allowing each entity to gather information from all other entities.

This design is motivated by three observations:

1. **Attention over entities is meaningful**: A hand card attending to played cards of the same suit computes "is my card still the highest?" — a natural and strategically valuable query. This is in contrast to the legacy architecture where attention over a single 256-dim token is mathematically degenerate (self-attention with sequence length 1 reduces to a linear projection).

2. **Parameter sharing across cards**: The same embedding and attention weights process all cards regardless of suit/rank. The model learns "what makes a card strong" once and applies it to all 52 cards, rather than learning separate weights for each position in a flat vector. This is dramatically more data-efficient.

3. **Variable structure is natural**: A 4-player game with 3 cards in hand produces ~20 tokens. An 8-player game with 13 cards in hand produces ~60 tokens. The Transformer handles both with padding masks, without wasting capacity on fixed-size padding in the input encoding.

### 4.2 Token Specification

Five token types represent the complete game state. Each token is constructed from learned embeddings plus scalar features, projected to the model dimension.

#### Hand Card Tokens (1-13 per state, private information)

Each card currently in the player's hand becomes one token.

| Feature | Dimensions | Encoding |
|---|---|---|
| Rank embedding | 16 | Learned embedding, `rank_index` ∈ [0, 12] |
| Suit embedding | 8 | Learned embedding, `suit_index` ∈ [0, 3] |
| `is_trump` | 1 | 1.0 if card's suit matches trump, 0.0 otherwise (0.0 for no-trump) |
| `suit_count_in_hand` | 1 | Number of cards of this suit in hand / 13.0 |
| `is_highest_in_suit` | 1 | 1.0 if no higher card of this suit remains unplayed (in deck or other hands) |
| `is_lowest_in_suit` | 1 | 1.0 if no lower card of this suit is in hand |
| `cards_above_remaining` | 1 | Count of higher cards of this suit that haven't been played / 12.0 |
| `cards_below_remaining` | 1 | Count of lower cards of this suit that haven't been played / 12.0 |

**Input dimension**: 16 + 8 + 6 = **30** → `Linear(30 → d_model)`

**Notes**:
- `is_highest_in_suit` requires cross-referencing with played cards — this is a derived feature computed at encoding time, not learned. It provides a strong bootstrapping signal: the network immediately knows which cards dominate their suit without needing to learn this through attention alone.
- `cards_above_remaining` counts unplayed cards (across all hidden hands + undealt cards) that outrank this card in the same suit. This is computed from the played-cards record.

#### Played Card Tokens (0-48 per state, public information, chronologically ordered)

Each card played in the current round becomes one token, ordered by play time.

| Feature | Dimensions | Encoding |
|---|---|---|
| Rank embedding | 16 | Shared with hand card rank embedding |
| Suit embedding | 8 | Shared with hand card suit embedding |
| Player embedding | 16 | Learned embedding, `player_index` ∈ [0, 7] |
| `trick_number` | 1 | Which trick this card was played in / `cards_dealt` → [0, 1] |
| `position_in_trick` | 1 | Play order within the trick / `num_players` → [0, 1] |
| `was_lead` | 1 | 1.0 if this was the first card played in the trick |
| `followed_suit` | 1 | 1.0 if card suit matches the led suit, 0.0 if off-suit |
| `is_trump_play` | 1 | 1.0 if this card is a trump card |
| `trick_complete` | 1 | 1.0 if the trick this card belongs to is complete, 0.0 if in progress |
| `won_trick` | 1 | 1.0 if this card won the trick, 0.0 otherwise (meaningful only when `trick_complete = 1.0`) |
| `is_current_trick` | 1 | 1.0 if this card belongs to the trick currently being played |

**Input dimension**: 16 + 8 + 16 + 8 = **48** → `Linear(48 → d_model)`

**Notes**:
- Rank and suit embeddings are **shared** with hand card tokens. This is critical — the same learned representation of "Ace of Spades" is used whether the card is in your hand or on the table. The model learns card identity once.
- Player embeddings encode relative position (relative to the current player, not absolute seat number). Player 0 is always "me," Player 1 is "left of me," etc. This ensures the model learns position-relative strategy.
- `followed_suit` is the key feature for suit void detection. When `followed_suit = 0.0` and `was_lead = 0.0`, the player couldn't follow suit — revealing a void.
- `trick_complete` and `won_trick` are split into separate features to avoid overloading a single feature with status and outcome semantics. The previous design used -1.0 as a sentinel for "in progress," which conflates "unknown" with "negative outcome." Splitting provides cleaner gradients — the network learns completion status and win/loss independently.

#### Player State Tokens (3-8 per state, public information)

Each player in the game becomes one token.

| Feature | Dimensions | Encoding |
|---|---|---|
| Player embedding | 16 | Shared with played card player embedding |
| `bid` | 1 | `bid / cards_dealt` → [0, 1], **-1.0** if not yet bid |
| `tricks_won` | 1 | `tricks_won / cards_dealt` → [0, 1] |
| `tricks_needed` | 1 | `max(0, bid - tricks_won) / cards_dealt`, -1.0 if no bid |
| `bid_status` | 1 | 1.0 if bid met exactly, -1.0 if bid busted (exceeded), 0.0 otherwise |
| `is_dealer` | 1 | 1.0 if this player is the current dealer |
| `is_me` | 1 | 1.0 if this token represents the current player |
| `relative_position` | 1 | Position relative to me / `num_players` → [0, 1) |
| `cumulative_score` | 1 | Total game score (multi-round) / `(rounds_completed × (10 + start_cards))`, 0.0 if single-round or no rounds completed |
| `cards_in_hand` | 1 | Number of cards remaining in hand / `cards_dealt` → [0, 1] |
| `void_spades` | 1 | 1.0 if this player has shown void in spades (played off-suit when spades were led), 0.0 otherwise |
| `void_hearts` | 1 | 1.0 if this player has shown void in hearts |
| `void_clubs` | 1 | 1.0 if this player has shown void in clubs |
| `void_diamonds` | 1 | 1.0 if this player has shown void in diamonds |

**Input dimension**: 16 + 13 = **29** → `Linear(29 → d_model)`

**Notes**:
- `tricks_needed` provides a direct urgency signal. A value > `tricks_remaining / cards_dealt` means the player's bid is in jeopardy or already impossible.
- `bid_status` gives the network an immediate signal about whether a player is still "live" (actively trying to meet their bid) or "done" (already met or busted). This is the most important feature for opponent modeling during the playing phase.
- **Suit void flags** are precomputed at encoding time by scanning played card records for off-suit plays (cards where `was_lead = 0` and `followed_suit = 0`). Void detection is the single most informative event in trick-taking games (section 2.2), and providing it as a direct feature saves the network from expensive multi-hop attention. The network can still learn nuanced near-void reasoning through attention over played cards — the binary flags bootstrap the critical hard constraint.

#### Context Token (1 per state, public information)

A single token capturing the global game state.

| Feature | Dimensions | Encoding |
|---|---|---|
| Trump suit one-hot | 5 | 4 suits + no-trump (exactly one is 1.0) |
| `cards_dealt` | 1 | `cards_dealt / 13.0` → [0, 1] |
| `current_trick` | 1 | `completed_tricks / cards_dealt` → [0, 1] |
| `tricks_remaining` | 1 | `(cards_dealt - completed_tricks) / cards_dealt` → [0, 1] |
| `num_players` | 1 | `num_players / 8.0` → [0.375, 1.0] |
| `round_number` | 1 | `round_number / total_rounds` → [0, 1] (0.0 if single-round) |
| `game_phase` | 2 | One-hot: bidding / playing |
| `bidding_constraint_active` | 1 | 1.0 if I am the dealer and it's bidding phase |

**Input dimension**: **13** → `Linear(13 → d_model)`

#### CLS Token (1 per state, learned)

A single learned parameter vector of dimension `d_model`, prepended to the token sequence. After processing by the Transformer, the CLS token's representation serves as the global readout for the policy and value heads.

**No input features** — the CLS token is a learned `nn.Parameter(d_model)` that is the same for every input. It learns to aggregate information from all other tokens through attention.

### 4.3 Embedding Strategy

#### Type Embeddings

Each of the 5 token types (hand, played, player, context, CLS) receives a learned type embedding of dimension `d_model`, added to the token representation after input projection:

```
token_repr = input_projection(features) + type_embedding[token_type]
```

Type embeddings allow the model to distinguish "this is a card in my hand" from "this is a card on the table" even when both share rank/suit embeddings. They are the primary mechanism for entity-type awareness.

#### Chronological Embeddings (Played Cards Only)

Played card tokens receive an additional learned chronological embedding based on their position in the play sequence (0 to 51):

```
played_card_repr += chronological_embedding[play_order_index]
```

- Maximum 52 positions (one per card in the deck)
- Learned embedding table: `nn.Embedding(52, d_model)`
- Applied **only** to played card tokens — hand cards, player states, context, and CLS tokens do not receive chronological embeddings because they are unordered sets

This gives the Transformer a sense of temporal ordering among played cards. "This card was played early" vs "this card was played recently" is encoded in the embedding, complementing the explicit `trick_number` feature.

#### Shared Embeddings

Three embedding tables are shared across token types:
- **Rank embedding** (`nn.Embedding(13, 16)`): Shared between hand cards and played cards
- **Suit embedding** (`nn.Embedding(4, 8)`): Shared between hand cards and played cards
- **Player embedding** (`nn.Embedding(8, 16)`): Shared between played cards and player states

Sharing ensures that the model learns a single coherent representation of each rank, suit, and player position, regardless of which token type uses it.

### 4.4 Transformer Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `d_model` | 128 | Each attention head gets 16 dims — sufficient to encode suit (4 values), rank (13 values), and player (8 values) comparisons. The structured entity input provides strong inductive bias, so width beyond 128 yields diminishing returns. |
| `num_layers` | 8 | Supports 4-5 step reasoning chains (e.g., "Player 3 showed void in hearts" → "they might have trumps" → "my heart lead is risky" → "lead clubs instead"). 6 layers is the minimum for multi-hop inference over game history; 8 provides headroom for deeper strategic reasoning. Worth benchmarking against 6 layers at equivalent training compute — if 6 matches, the freed ~400K parameters could be reallocated to wider `d_model`. |
| `num_heads` | 8 | 16 dims per head. Eight heads accommodate the distinct attention patterns needed: suit matching, rank comparison, player grouping, temporal proximity, trump tracking, bid comparison, lead/follow distinction, and global aggregation. |
| `ffn_dim` | 512 | 4× expansion ratio (standard). The FFN layers learn nonlinear feature combinations within each token — e.g., "high rank + trump suit + opponent showed void = very strong card." |
| `dropout` | 0.1 | Applied in attention weights and FFN. Provides regularization against overfitting on the 500K replay buffer. |
| `activation` | GeLU | Modern default for Transformers (BERT, GPT-2+, ViT). Provides smoother gradients near zero compared to ReLU, marginally improving training dynamics at small model scales. Supported in tch-rs via `Tensor::gelu`. |
| `normalization` | Pre-norm | LayerNorm applied before attention and FFN (not after). Pre-norm provides more stable gradients in deeper networks (8 layers) and is the modern default. |

#### Attention Masking

Standard padding mask: tokens beyond the actual sequence length (padded to max length in a batch) are masked to prevent attention. No causal mask is used — all tokens attend to all other tokens bidirectionally, because the game state is a complete observation (not an autoregressive generation task).

#### Initialization

- **Linear layers**: Xavier uniform (`nn.init.xavier_uniform_`)
- **Biases**: Zeros
- **Embeddings** (rank, suit, player, type, chronological): Normal distribution, `mean=0, std=0.02`
- **CLS token**: Normal distribution, `mean=0, std=0.02`
- **Residual scaling**: The second linear in each FFN block is scaled by `1/√num_layers` to prevent signal explosion in the 8-layer network

### 4.5 Output Heads

The policy and value heads use different readout strategies. The policy head uses **phase-specific readout** — separate mechanisms for bidding and playing — while the value head reads from the CLS token.

#### Playing Policy Head (Entity-Based Readout)

During the playing phase, each hand card token produces a scalar "play this card" score through a shared MLP. Softmax over these scores yields play probabilities.

```
Each hand card token (128 dims)
  → Linear(128 → 32)
  → GeLU
  → Linear(32 → 1)
  → scalar score

Stack scores for all hand cards → Softmax → play probabilities
```

This is architecturally native to the entity structure: after 8 layers of attention, each hand card's representation already encodes its relationship to played cards (is my card still highest?), player states (are opponents competing for tricks?), and the game context (trump suit, trick count). Asking "should this card be played?" directly from that representation is the natural readout.

**Advantages over a unified 52-dim output**:
- **No wasted capacity**: Only cards in hand produce scores — no parameters dedicated to always-masked positions
- **Parameter efficient**: ~4K params vs ~23K for a CLS → 52-dim head
- **Generalization**: The scoring MLP is shared across all cards and hand sizes. The same weights evaluate any card in any position

**MCTS integration**: During MCTS, the playing head returns a probability distribution over hand card indices (0 to `hand_size - 1`). The MCTS engine maps these back to card indices (`suit_index × 13 + rank_index`) using the known hand composition. Visit count distributions for training targets use the same hand-card-index space.

#### Bidding Policy Head (CLS-Based Readout)

During the bidding phase, the CLS token feeds a dedicated bidding head.

```
CLS_repr (128)
  → Linear(128 → 64)
  → GeLU
  → Dropout(0.1)
  → Linear(64 → 14)
  → apply bid mask (illegal bids → -inf)
  → Softmax
  → bid probabilities (14)
```

**Bid action space**: Actions 0-13 represent bid values 0 through 13. Illegal bids are masked:
- Bids > `cards_dealt` are always masked
- If the dealer constraint is active (total bids cannot equal `cards_dealt`), the forbidden bid value is additionally masked

**Why separate heads**: Bidding and playing are fundamentally different decision types — "choose a number" vs. "choose a card." A unified 52-dim head forces output positions 0-13 to serve dual roles (bid values AND card indices ♠2-♠A), coupled only by the game_phase signal. This coupling means playing-phase gradients corrupt bidding weights at positions 0-13, and vice versa. Separate heads eliminate this gradient interference entirely.

**On bidding-playing alignment**: Separating the heads prevents gradient interference but does not inherently prevent "optimistic bidding" — a bidding head that is too aggressive early in training. The mechanism that keeps bidding honest is the **value head**: during MCTS for bidding, each candidate bid leads to simulated play. If the value head accurately predicts "I bid 5 with a weak hand → score 0," MCTS will down-weight aggressive bids regardless of the bidding policy's prior. The training loop self-corrects: optimistic bids → failing games → low value targets → corrected value estimates → realistic MCTS bid targets next iteration. This is why value head accuracy (well-defined target, proper loss weighting) is critical — it is the mechanism that aligns bidding ambition with playing capability.

#### Value Head (CLS-Based Readout)

```
CLS_repr (128)
  → Linear(128 → 64)
  → GeLU
  → Dropout(0.1)
  → Linear(64 → 1)
  → Tanh
  → value estimate ∈ [-1, 1]
```

The value represents the expected game outcome for the current player. Tanh bounds the output to [-1, 1], where 1 represents the best possible outcome and -1 the worst.

**Value target definition**: The training target for the value head is the **z-scored cumulative score** at game end:

```
target = clip((my_score - mean_score) / max(std_score, ε), -1, 1)
```

where `my_score` is the current player's final game score, `mean_score` and `std_score` are computed across all players in the game, and `ε` prevents division by zero (e.g., when all players score 0).

This target is preferred over alternatives:
- **Win/loss (+1/-1)**: Too coarse — a player who lost by 1 point after 17 rounds gets the same signal as one who scored 0. Wastes information in a multi-round game.
- **Raw score**: Unbounded and varies with game length. Poor normalization makes training unstable.
- **Rank-based**: Insensitive to margin of victory. A 2-point win and a 50-point win produce the same target.

Z-scored score captures both relative standing and magnitude of advantage, providing richer gradients for MCTS value estimation. When all players score identically (std = 0), the target is 0.0 for all players, correctly reflecting no differentiation.

### 4.6 Loss Function and Training Configuration

- **Policy loss**: Cross-entropy with masking: `-sum(target × log(pred + 1e-8))`, averaged over batch. During bidding, the target is the MCTS visit distribution over bid actions (14-dim). During playing, the target is the MCTS visit distribution mapped to hand card indices.
- **Value loss**: MSE between predicted and actual z-scored game outcome (see value target definition in 4.5)
- **Combined**: `policy_loss + c_value × value_loss`, where `c_value = 2.0`. Policy loss (cross-entropy over ~5-10 legal actions, typical range 0.5-2.5) naturally dominates value loss (MSE over [-1, 1], typical range 0.01-0.25) at equal weighting, starving the value head of learning signal. A coefficient of 2.0 approximately balances gradient magnitudes. This is tunable — monitor both loss curves and adjust if one plateaus while the other continues improving.
- **Optimizer**: AdamW (`lr=3e-4`, `β₁=0.9`, `β₂=0.999`, `weight_decay=1e-4`)
- **Learning rate schedule**: Linear warmup over the first 1,000 training batches from 0 to `3e-4`, followed by cosine annealing to `1e-5` over the remaining training budget. Warmup prevents early instability when embeddings are near-random. Cosine annealing provides smooth convergence without hard learning rate drops.
- **Gradient clipping**: `max_norm=1.0`

### 4.7 Parameter Budget

| Component | Parameters | Calculation |
|---|---|---|
| **Embeddings** | | |
| Rank embedding | 208 | 13 × 16 |
| Suit embedding | 32 | 4 × 8 |
| Player embedding | 128 | 8 × 16 |
| Type embedding | 640 | 5 × 128 |
| Chronological embedding | 6,656 | 52 × 128 |
| CLS token | 128 | 1 × 128 |
| **Input projections** | | |
| Hand card projection | 3,968 | (30 + 1) × 128 |
| Played card projection | 6,272 | (48 + 1) × 128 |
| Player state projection | 3,840 | (29 + 1) × 128 |
| Context projection | 1,792 | (13 + 1) × 128 |
| **Transformer (×8 layers)** | | |
| Self-attention (Q, K, V, O) | 66,048 | 4 × (128 × 128 + 128) per layer |
| FFN (up + down) | 131,584 | (128 × 512 + 512) + (512 × 128 + 128) per layer |
| LayerNorm (×2) | 512 | 2 × (128 + 128) per layer |
| Per-layer total | 198,144 | |
| 8 layers total | **1,585,152** | |
| **Output heads** | | |
| Playing head (entity-based) | 4,161 | (128 × 32 + 32) + (32 × 1 + 1) |
| Bidding head | 9,166 | (128 × 64 + 64) + (64 × 14 + 14) |
| Value head | 8,321 | (128 × 64 + 64) + (64 × 1 + 1) |
| | | |
| **Total** | **~1.63M** | |

At 1.63M parameters, this model is **3× smaller** than the legacy 4.9M Transformer while being far more expressive — every parameter is doing useful work on structured entity relationships rather than computing degenerate self-attention over a single token. The entity-based playing head is more parameter-efficient than a unified 52-dim head (4K vs 23K), while the separate bidding head adds only 9K parameters for clean gradient separation.

### 4.8 Inference Performance

**Token counts by game configuration:**

| Configuration | Hand | Played | Player | Context | CLS | Total |
|---|---|---|---|---|---|---|
| 5p, 7 cards, bidding phase | 7 | 0 | 5 | 1 | 1 | 14 |
| 5p, 7 cards, trick 1 (3 played) | 7 | 3 | 5 | 1 | 1 | 17 |
| 5p, 7 cards, trick 4 (mid-game) | 4 | 18 | 5 | 1 | 1 | 29 |
| 5p, 7 cards, trick 7 (late game) | 1 | 33 | 5 | 1 | 1 | 41 |
| 8p, 6 cards, trick 6 (late game) | 1 | 45 | 8 | 1 | 1 | 56 |

**Estimated compute per forward pass (typical 35 tokens):**

| Operation | MACs per layer | Notes |
|---|---|---|
| Self-attention (Q, K, V projections) | 3 × 35 × 128² ≈ 1.7M | Three projections |
| Attention scores + weighted sum | 35² × 128 ≈ 160K | Quadratic in sequence length |
| Output projection | 35 × 128² ≈ 574K | |
| FFN (up + down) | 35 × 128 × 512 × 2 ≈ 4.6M | Two linear layers |
| **Per layer total** | **~7M** | |
| **8 layers** | **~56M** | |
| **Heads + embeddings** | **~0.5M** | |
| **Grand total** | **~57M MACs** | |

**Estimated inference latency (ONNX Runtime, CPU):**

| Token count | Latency | Context |
|---|---|---|
| 14 (bidding) | ~0.05ms | Minimal sequence |
| 29 (mid-game) | ~0.12ms | Typical MCTS evaluation point |
| 41 (late game) | ~0.18ms | Near-complete play history |
| 56 (8p late game) | ~0.25ms | Worst case |

**MCTS budget implications (5 determinizations × 100 simulations):**

At ~0.15ms average inference, 500 evaluations per move take **~75ms**. With 40 decisions per game (5 bids + 35 plays), one full game takes ~3 seconds of neural network time. At 32 rayon threads on the 7950X, the self-play throughput is approximately **640 games per minute** — comfortably supporting the 2,000-10,000 games per training iteration target.

---

## 5. Encoding the Game State as Entity Tokens

### 5.1 From BlobState to Tokens

The Rust game engine maintains a compact `BlobState` struct (~200 bytes, stack-allocated). The entity encoder transforms this into a variable-length sequence of tokens at each MCTS leaf evaluation.

```
BlobState (200 bytes)
  ├── hands[current_player] → Hand Card tokens
  ├── trick_play_order + played_this_trick → Current Trick Played Card tokens
  ├── trick_history (accumulated) → Past Trick Played Card tokens
  ├── bids, tricks_won, num_players → Player State tokens
  └── trump_suit, cards_dealt, game_phase → Context token
```

**Additional game state for encoding**: The entity encoder requires information not present in the minimal `BlobState` struct from the migration plan. Specifically, it needs the **full trick history** — not just `played_this_round` (a bitmask of which cards are out), but an ordered log of which player played which card in which trick. This requires extending the game engine to maintain a trick history log:

```rust
struct TrickRecord {
    cards: [(u8, u8); 8],   // (player_index, card_index) pairs, in play order
    num_played: u8,          // Cards played in this trick
    winner: u8,              // Player who won (255 if incomplete)
    suit_led: u8,            // Suit of the first card played
}

struct BlobState {
    // ... existing fields ...
    trick_history: [TrickRecord; 13],  // One per trick, max 13 tricks per round
    tricks_completed: u8,
}
```

This adds ~210 bytes to BlobState (13 tricks × ~16 bytes each), bringing total size to ~410 bytes. State copies remain fast at ~50-100ns (well within the performance budget).

### 5.2 Replay Buffer Format

The replay buffer stores raw `BlobState` structs rather than pre-encoded tokens. Entity tokens are generated on-the-fly during training batch construction.

```rust
struct TrainingExample {
    state: BlobState,       // ~410 bytes — compact game state
    policy: [f32; 52],      // MCTS visit count distribution
    value: f32,             // Final game outcome for this player
}
```

**Storage**: 500K examples × ~620 bytes = **~310 MB** (vs. ~500 MB for the legacy flat encoding approach with 256 × f32 per state).

**Advantages of store-raw, encode-on-demand**:
- Encoding changes don't invalidate the replay buffer
- Smaller storage footprint (compact game state vs. padded token sequences)
- Encoding is cheap (feature extraction, no neural network call) and parallelizable during batch loading
- The same buffer supports architecture experiments without regenerating self-play data

### 5.3 Batching and Padding

Within a training batch, token sequences have variable length. Standard Transformer batching applies:

1. Determine the maximum sequence length in the batch
2. Pad all sequences to this length with zero-vectors
3. Construct a padding mask: `true` for real tokens, `false` for padding
4. Apply the mask in self-attention (padding tokens get `-inf` attention scores)

The maximum possible sequence length is **57** (8 players, 6 cards each: 1 hand + 47 played + 8 players + 1 context). For the dominant 5-player configuration, typical sequences are 20-40 tokens. Padding overhead is modest.

### 5.4 Bidding Phase vs. Playing Phase

During the bidding phase:
- **Hand card tokens**: Full hand (all dealt cards), present
- **Played card tokens**: None (no tricks played yet)
- **Player state tokens**: Present, with partial bid information (`-1.0` for players who haven't bid yet)
- **Context token**: `game_phase = bidding`

The token sequence is short (hand cards + players + context + CLS ≈ 14-22 tokens), and attention operates over a small set of entities. The model learns bidding strategy primarily from hand card features (strength, suit distribution, trump count) and player state features (others' bids, position).

During the playing phase:
- **Hand card tokens**: Shrinking as cards are played
- **Played card tokens**: Growing as tricks complete
- **Player state tokens**: Updated with tricks won, bid status
- **Context token**: `game_phase = playing`

The token sequence grows with each trick. Attention over played card tokens enables the model to reason about the full game history — the core capability that motivates this architecture.

### 5.5 MCTS Integration

The neural network evaluates game states at MCTS leaf nodes. In imperfect-information MCTS with determinization:

1. MCTS samples a consistent assignment of opponent hands (determinization)
2. MCTS simulates forward from the current state using the assumed hands
3. At each leaf node, the entity encoder converts the current `BlobState` to tokens
4. The Transformer processes the tokens and outputs (policy, value)
5. MCTS uses the policy as child priors and the value for backpropagation

The network always evaluates from the perspective of the **current player** at the leaf node. During MCTS simulation, this player may differ from the root player (we simulate opponents' turns too). The encoding adjusts automatically:
- Hand card tokens reflect the current player's hand (which is known in the determinized world)
- Player embeddings are relative to the current player (Player 0 = "me" for whoever is deciding)
- Played card tokens include all cards played up to the current point in the simulation (including hypothetical plays from the determinized rollout)

---

## 6. Learned Attention Patterns

The following attention patterns are expected to emerge during training. These are not hard-coded — they arise naturally from the entity structure and the training signal.

### 6.1 Suit-Matching Attention (Hand → Played)

A hand card token queries played card tokens of the same suit to determine its relative strength. The shared suit embedding provides the key for matching; the rank embedding provides the comparison value.

**Strategic function**: "Is my Queen of Spades the highest remaining spade?" → attend to all played spade tokens, check if King and Ace are among them.

### 6.2 Void-Detection Attention (Played → Played, grouped by player)

Played card tokens from the same player are grouped by attention, with the `followed_suit` feature providing the critical signal. When a played card token has `followed_suit = 0.0` and `was_lead = 0.0`, it indicates the player couldn't follow suit.

**Strategic function**: "Does Player 3 have any hearts left?" → attend to Player 3's plays where hearts were led, check if any have `followed_suit = 0.0`.

### 6.3 Bid-Progress Attention (Player → Player)

Player state tokens attend to each other to compare bid fulfillment. The `tricks_needed` and `bid_status` features enable direct comparison.

**Strategic function**: "Who is competing with me for tricks?" → attend to players whose `tricks_needed > 0` and who have enough cards remaining to fulfill their bids.

### 6.4 Card-Player Interaction (Hand → Player)

Hand card tokens attend to player state tokens to evaluate play decisions in the context of opponent strategies.

**Strategic function**: "Should I lead my high card now?" → attend to opponents' bid status. If opponents have met their bids and are trying to lose, a high card lead is safe. If opponents still need tricks, they might play higher.

### 6.5 Temporal Attention (Played → Played, ordered)

The chronological embeddings enable attention patterns that respect temporal ordering. Recent plays are more strategically relevant than early plays (they reveal current hand composition more directly).

**Strategic function**: "What was the most recent play by Player 2?" → chronological embeddings create a recency gradient in attention weights, allowing the model to weight recent information more heavily.

### 6.6 Global Aggregation (CLS → All)

The CLS token attends to all entity tokens to build a global game representation for the policy and value heads. Different attention heads in the CLS token specialize in different aspects: one might focus on hand strength, another on opponent voids, another on bid progress.

**Strategic function**: "What's the overall game situation?" → CLS aggregates across all entity types, weighting each according to the current game phase and strategic demands.

---

## 7. Summary

| Property | Value |
|---|---|
| Architecture | Structured Entity Transformer |
| Parameters | ~1.63M |
| d_model | 128 |
| Layers | 8 |
| Attention heads | 8 (16 dims/head) |
| FFN dimension | 512 |
| Activation | GeLU |
| Token types | 5 (hand card, played card, player state, context, CLS) |
| Typical token count | 20-40 |
| Max token count | 57 |
| Policy output (playing) | Entity-based per-card scoring via shared MLP |
| Policy output (bidding) | 14-dim CLS-based head |
| Value output | Scalar ∈ [-1, 1] (z-scored cumulative score) |
| Value loss weight | 2.0× |
| Learning rate | 3e-4 with linear warmup + cosine annealing |
| Inference (CPU, ONNX) | ~0.15ms average |
| Training framework | tch-rs (libtorch) for GPU training |
| Inference framework | ort (ONNX Runtime) for CPU inference during MCTS |
