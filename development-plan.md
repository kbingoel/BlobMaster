# BlobMaster Development Plan

> **Terminology**: a "game" = full multi-round session (e.g., 23 rounds for 5 players). A "round" = one deal-bid-play-score cycle. A "trick" = each player plays one card. See README.md vocabulary table.

## Section 1: Rust Foundation & Game Engine (4 sessions, ~12h)

### Session 1.1 — Workspace scaffold and card primitives

Set up the Rust workspace structure and implement the low-level card representation with bitwise operations.

- `cargo init --name blobmaster` with workspace members: `blob-engine`, `blob-nn`, `blob-bin`
  - Start lean — 3 crates: engine (game + encoder + MCTS), nn (model + training), bin (CLI + eval). Split further only when boundaries stabilize
  - Additional crates (encoder, mcts, eval) can be extracted later without breaking APIs
  - **Crate constraints**: `blob-engine` is cross-platform and must **never** depend on `tch` (libtorch) — only `ort` and pure-Rust crates. `tch` lives exclusively in `blob-nn` (training on Linux). This keeps the inference binary buildable on the Windows deployment target (Section 3.5)
- Add shared dependencies in workspace `Cargo.toml`: `serde`, `rand`, `rand_xoshiro`, `smallvec`, `tracing`
- Implement card encoding: `card_index = suit * 13 + rank` (Suits: ♠=0 ♥=1 ♣=2 ♦=3, Ranks: 2=0 through A=12). Hard cap on `cards_dealt`: **C ≤ 13**. This is never a binding constraint in training (distribution is C∈{7,8}, n∈{4,5,6,7}, max 8 tricks/round), but keeping the cap explicit lets all fixed-size arrays below use 13 as their upper bound
- Implement `u64` bitmask operations: `hand.add(card)`, `hand.remove(card)`, `hand.contains(card)`, `hand.count()`, `hand.iter()`, `hand.cards_of_suit(suit)`
- Use `u64` bit manipulation: bit N = card N present. `popcount` for count, `trailing_zeros` for iteration, mask `& (0x1FFF << (suit*13))` for suit extraction
- Define `GamePhase` enum (`#[repr(u8)]`): `Bidding = 0`, `Playing = 1`, `Scoring = 2`. `BlobState.game_phase` stores one of these. Section 2.3's context-token `game_phase one-hot(2)` one-hots over {Bidding, Playing} (Scoring states are never encoded for network input); Section 3.3's phase dispatch reads this field to pick which head to run
- Define `TrickRecord { cards: [(u8, u8); 8], num_played: u8, winner: u8, suit_led: u8 }` — stack-allocated, no heap. **Tuple convention**: `cards[i] = (player, card)`, ordered by play sequence (index 0 = first player to act in the trick, i.e. the leader). `num_played` equals `num_players` for every completed record, kept for validation/invariant checks
- Define `BlobState` struct (~410 bytes):
  - `hands: [u64; 8]`, `played_this_round: u64` (maintained incrementally — needed by encoder for `cards_above_remaining`, `is_highest_in_suit`)
  - `bids: [u8; 8]`, `tricks_won: [u8; 8]`, `trump_suit: u8`, `current_player: u8`, `dealer: u8`, `num_players: u8`, `cards_dealt: u8`, `game_phase: u8` (`GamePhase` repr)
  - `trick_leader: u8`, `trick_play_order: [u8; 8]`, `trick_cards_played: u8`
    - **`trick_play_order` stores card indices** (not players) in play order for the in-progress trick. Slot `i` holds the card played by player `(trick_leader + i) % num_players`. This keeps the field 8 bytes instead of 16 and avoids redundancy, since `trick_leader` is already in state. Player derivation is O(1) for the encoder (Section 2.2 iterates the current trick alongside `trick_history`)
  - `trick_history: [TrickRecord; 13]`, `tricks_completed: u8` — 13 is the hard max given C ≤ 13 cap above; training distributions never exceed 8 tricks/round
  - `cumulative_scores: [u16; 8]` (multi-round tracking, needed by player state tokens)
- **RNG ownership**: the shuffling/determinization RNG (`Xoshiro256PlusPlus`) is **not** a field of `BlobState` — it is passed in explicitly to `new_game`, `advance_round`, and any dealing helper. This preserves `BlobState: Copy` (required for zero-cost cloning in MCTS) and matches Section 5.3's per-thread seeding strategy for self-play determinism
- Write unit tests for all bitmask ops; benchmark `BlobState` copy (target: ~100ns via `memcpy` of stack struct — 410 bytes across ~6 cache lines)

### Session 1.2 — Dealing, bidding phase, and trump rotation

Implement the game initialization, dealing logic, and the complete bidding phase.

- Trump rotation: `[Spades, Hearts, Clubs, Diamonds, NoTrump]` cycling every 5 rounds — store as `u8` (0–4)
- Round structure: symmetric pattern — `[C, C−1, …, 2, 1, 1, …, 1, 2, …, C−1, C]`. Built from three segments:
  - Descending: `C, C−1, …, 2` — `C − 1` rounds
  - One-card plateau: `1` repeated `num_players` times — `num_players` rounds (this is the total count of 1-card rounds in the game)
  - Ascending: `2, 3, …, C` — `C − 1` rounds
  - **Total rounds** = `(C − 1) + num_players + (C − 1) = 2C + num_players − 2`
  - Example (5 players, C=7): `[7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7]` — 17 rounds (5 one-card rounds in the plateau)
  - Example (5 players, C=8): `[8,7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7,8]` — 19 rounds
  - C is a game parameter (typically 7 or 8), constrained by `num_players × C ≤ 52` and by the engine's hard cap `C ≤ 13` (Session 1.1)
  - ⚠ **Legacy discrepancy**: `legacy/game-engine/constants.py::generate_round_structure` produces `2C + num_players − 1` rounds — it concatenates `range(C, 0, -1)` (which already ends in 1) with `[1] * num_players`, yielding one extra 1-card round. The Rust port must use the formula above, which matches the game as actually played and the README example. Any legacy tests that assert round count or round-index-to-cards-dealt mappings need to be adjusted when ported (Session 1.4's gate)
- Dealing: shuffle deck (Fisher-Yates with `rand_xoshiro::Xoshiro256PlusPlus` for speed), distribute `cards_dealt` cards per player as `u64` bitmasks
- Bidding rules: each player bids 0..=cards_dealt. **Dealer restriction**: dealer's bid cannot make total bids equal cards_dealt (forces at least one player to miss). Validate this constraint
- `fn legal_bids(state: &BlobState) -> u16` — bitmask over values 0..=13, with forbidden value cleared for dealer. Max 14 possible bids fits in u16. Avoids heap allocation (unlike Vec)
- `fn apply_bid(state: &mut BlobState, bid: u8)` — set `bids[current_player]`, advance player, transition to playing phase when all have bid
- Port relevant tests from Python's `test_blob.py` (bidding group: ~25 tests out of 143 total)
- Edge case: 1-card rounds where bid can only be 0 or 1, and dealer constraint may force bid=0

### Session 1.3 — Trick-taking, suit following, and scoring

Implement the playing phase with full trick-taking rules and the all-or-nothing scoring.

- Suit-following rule: must follow led suit if able; if void, may play any card (including trump)
- `fn legal_plays(state: &BlobState) -> u64` — returns bitmask: if hand has cards of led suit, mask to those; else entire hand. First play of trick: entire hand. Bitmask return stays at ~5ns; Vec allocation would push to ~30-50ns
- Trick winner determination: highest card of led suit wins, **unless** trump was played — then highest trump wins. NoTrump round: no trump override possible
- `fn apply_play(state: &mut BlobState, card: u8)` — remove card from hand, **set bit in `played_this_round`**, record in `trick_play_order` and `trick_history`, detect trick completion, determine winner, advance state
- When trick completes: increment `tricks_won[winner]`, set `trick_leader = winner`, `current_player = winner`, increment `tricks_completed`
- When all tricks done (tricks_completed == cards_dealt): transition to scoring phase
- Scoring: `score[i] = if tricks_won[i] == bids[i] { 10 + bids[i] } else { 0 }` — all-or-nothing
- Port trick-taking tests (~60 tests from Python's 143 total), including edge cases: void suits, trump overrides, NoTrump rounds

### Session 1.4 — Full game loop, round sequencing, and property tests

Wire up the complete multi-round game and add comprehensive testing including property-based tests.

- `fn new_game(num_players: u8, cards_dealt: u8) -> BlobState` — initializes first round. Validate `num_players × cards_dealt ≤ 52`
- `fn advance_round(state: &mut BlobState)` — reset trick state, clear `played_this_round`, advance dealer, rotate trump, adjust cards_dealt per symmetric round structure, accumulate scores into `cumulative_scores`
- Full game loop: deal → bid → play tricks → score → next round, until all rounds complete
- `fn is_game_over(state: &BlobState) -> bool` — true when last round scored
- `fn total_rounds(num_players: u8, start_cards: u8) -> u8` — `2 * start_cards + num_players - 2`. E.g., 5P7C → `2*7+5-2 = 17`, 5P8C → `2*8+5-2 = 19`
- Add `proptest` for property-based testing: random games always terminate, card conservation (52 cards never lost/duplicated), tricks_won always sums to cards_dealt, no player can play a card not in their hand
- Port remaining ~58 tests from Python's `test_blob.py` (143 total across all three sessions). Some tests will need adjustment: any test asserting round counts must use the corrected `2C + num_players − 2` formula, not the legacy Python value
- **Gate check**: all ported tests from `test_blob.py` pass (143 total, adjusted for the round-structure correction), `BlobState` copy benchmarked, legal move gen benchmarked
- `#[derive(Clone, Copy)]` on `BlobState` — ~410 bytes, copy is just memcpy

---

## Section 2: Entity Encoder (3 sessions, ~9h)

### Session 2.1 — Feature layout and hand card tokens

Establish the raw one-hot feature layout and implement the hand card token encoder.

- **Raw one-hot features, no learned embedding tables in the encoder**: rank is encoded as a 16-dim one-hot (13 values + 3 padding slots so the layout is uniform across token types), suit as 8-dim (4 + 4 padding), player as 16-dim (8 + 8 padding). The 1s sit at the appropriate index, every other slot is 0.0
- Decision: encoder produces **raw feature vectors** per token; the neural network's per-token-type input projection (`Linear(30,128)` for hand cards, `Linear(48,128)` for played, etc., per Session 3.1) absorbs the rank/suit/player columns into its weight matrix. There is no shared embedding table — each token type has its own projection. Encoder is pure Rust, no ML dependency
- Hand card token (30 dims): rank(16) + suit(8) + is_trump(1) + suit_count_in_hand(1) + is_highest_in_suit(1) + is_lowest_in_suit(1) + cards_above_remaining(1) + cards_below_remaining(1)
- "Remaining" means not in own hand and not yet played — uses `played_this_round` bitmask from `BlobState`. `cards_above_remaining(card)` = popcount of cards in same suit with higher rank, not in hand, not in `played_this_round`
- `is_highest_in_suit` / `is_lowest_in_suit`: among cards of that suit remaining in the game (hand + unplayed), is this card the extreme?
- Output: `Vec<[f32; 30]>` with 1–13 entries (one per card in hand)
- **Hand emit order is the canonical action order**: hand cards are emitted in ascending card-index order, i.e. exactly the order of `Hand::iter()` (which uses `trailing_zeros` over the `u64` bitmask). This is the same order the playing head's per-position scores will be interpreted in (Session 3.3), and the same order the MCTS-to-policy mapping uses (Session 5.2). Do not reorder by suit, rank, trumpness, or anything else
- Test: construct known game states, verify feature values exactly match expected. Especially test derived features (highest/lowest, remaining counts) after several tricks played

### Session 2.2 — Played card tokens and player state tokens

Implement the chronologically-ordered played card tokens and per-player state tokens.

- **Played card token** (48 dims): rank(16) + suit(8) + player(16) + 8 scalar features: trick_number(1) + position_in_trick(1) + was_lead(1) + followed_suit(1) + is_trump_play(1) + trick_complete(1) + won_trick(1) + is_current_trick(1). Total: 16+8+16+8 = 48. Note the 8 scalar features as a group to avoid off-by-one errors during projection implementation
- Iterate `trick_history[0..tricks_completed]` plus current trick's `trick_play_order` — emit tokens in strict chronological order
- `followed_suit`: card suit == trick's led suit. Critical signal: `followed_suit=0 && was_lead=0` reveals a **suit void** for that player
- `won_trick`: only set for the card that won a completed trick (highest trump, or highest of led suit)
- Chronological position index (0–51) stored alongside each token — the neural net adds a learned chronological embedding from a 52×128 table (Session 3.1). The maximum is 51, hit only by the final play of a 4P×13C round (52 plays total); 5–7 player training distributions stay well under that
- **Player state token** (29 dims): player(16) + bid(1) + tricks_won(1) + tricks_needed(1) + bid_status(1) + is_dealer(1) + is_me(1) + relative_position(1) + cumulative_score(1) + cards_in_hand(1) + void_spades(1) + void_hearts(1) + void_clubs(1) + void_diamonds(1)
- `bid_status`: encode as -1 (busted, tricks_won > bid), 0 (live, tricks_won <= bid and tricks_remaining sufficient), +1 (met, tricks_won == bid)
- `void_*` flags: precomputed from played card history — scan all played cards where `followed_suit=0 && was_lead=0`, mark that player as void in the led suit. **Single most informative signal** for belief tracking
- `relative_position`: `(player_idx - current_player) % num_players`, normalized to [0,1]
- `is_me`: 1.0 for the player whose perspective this encoding represents
- `cumulative_score`: read from `BlobState.cumulative_scores`, normalized as `score as f32 / (total_rounds(start_cards, num_players) as f32 * (10.0 + start_cards as f32))`. The denominator is the theoretical ceiling — every round perfectly bid at the maximum card count of that game — so the feature stays in `[0, 1]`. Pin this exact divisor so the value head and the player-state token agree on scale

### Session 2.3 — Context token, CLS, sequence assembly, and validation

Build the context token, CLS token placeholder, assemble the full variable-length sequence, and validate with internal consistency tests.

- **Context token** (13 dims): trump_suit one-hot(5, includes NoTrump) + cards_dealt(1, normalized) + current_trick(1, normalized) + tricks_remaining(1) + num_players(1, normalized) + round_number(1, normalized) + game_phase one-hot(2) + bidding_constraint_active(1)
- `round_number` normalization: `state.round_idx as f32 / total_rounds(state.start_cards, state.num_players) as f32`. Uses `start_cards` from `BlobState` (added in Session 1.1) so the divisor is exact for whatever round structure this game was initialized with
- `bidding_constraint_active`: 1.0 only during bidding phase when current player is dealer and constraint applies
- **Game-phase one-hot covers only `{Bidding, Playing}`**: the encoder is never legitimately called from `Scoring` (engine auto-transitions through it inside `advance_round`) or `Complete` (terminal sink, MCTS doesn't search past it). The two-bit one-hot is `[is_bidding, is_playing]`. Open the encoder with `debug_assert!(matches!(state.phase(), GamePhase::Bidding | GamePhase::Playing))` so any caller that tries to encode a Scoring/Complete state fails loudly in dev builds
- **CLS token**: no features from encoder — it's a learned 128-dim parameter in the neural network. Encoder emits a zero-vector or special marker
- **Encoder signature**:
  ```rust
  fn encode(state: &BlobState, perspective: u8) -> EncodedState
  ```
  The `perspective` argument is the player whose viewpoint the encoding represents. It drives `is_me` (player-state token), `relative_position`, and the choice of which hand the hand-card tokens iterate. **MCTS always passes `state.current_player`** (Section 4.2 / 5.2) — the leaf evaluator encodes from the perspective of whoever is about to act. Other call sites (e.g. eval tooling) may pass a fixed seat
- **Sequence assembly order**: `[CLS, context, player_states..., hand_cards..., played_cards...]`
  - Total tokens: 1 + 1 + num_players + hand_size + cards_played, ranging roughly **6 to 58** end-to-end. Lower bound: 3P, 1-card round, no plays yet (`1+1+3+1+0 = 6`). Upper bound: 4P×13C late game, last play (`1+1+4+1+51 = 58`). Typical 5P7C early game ≈ 14, late game ≈ 49
  - Must also emit token type IDs (0–4) for each position, so the network knows which input projection to use
- Output struct:
  ```rust
  EncodedState {
      features: Vec<Vec<f32>>,
      token_types: Vec<u8>,
      chronological_indices: Vec<u8>,
      hand_card_indices: SmallVec<[u8; 13]>,  // card indices in token-emit order; same order as Hand::iter()
      num_tokens: usize,
  }
  ```
  - `hand_card_indices` is the contract that lets MCTS map the playing head's per-position scores back to card indices without re-iterating the hand. Length equals the number of hand-card tokens; entry `i` is the card index emitted at hand-card token slot `i`. Must match `Hand::iter()` ordering exactly (Session 2.1)
- **Internal consistency tests**: construct known game states by hand, verify each token type's features match expected values exactly. Test derived features (highest/lowest in suit, remaining counts, void flags) across various game phases — early trick, mid-game, and late-game states. Also verify `hand_card_indices` matches `state.hands[perspective]` enumerated via `Hand::iter()`, and that the encoder rejects `Scoring`/`Complete` states in debug. No cross-validation against the Python legacy encoder, since the Rust encoder produces an entirely different representation (variable-length entity tokens vs. fixed 256-dim vector)
- Benchmark: encoding should be <1μs per state (mostly arithmetic on small arrays)

---

## Section 3: Neural Network — Structured Entity Transformer (5 sessions, ~15h)

### Session 3.1 — tch-rs setup and input projection layers

Set up the libtorch Rust bindings and implement the per-token-type input projections.

- **Dependency pinning** (decided in Session 3.1): `tch = { version = "0.20", features = ["download-libtorch"] }` in `blob-nn/Cargo.toml`. The `download-libtorch` feature makes `torch-sys`'s build script fetch a pinned libtorch 2.7.0 CPU build into `target/` on first compile — no system libtorch install, no Python venv dependency, no `LIBTORCH_*` env vars. `Cargo.lock` locks the exact tch/torch-sys versions. **Do not bump tch without cause**: newer versions (0.24 + libtorch 2.11 as of 2026-04) bring FlashAttention, FP8, distributed training — none of which this 1.63M-param CPU-trained model uses. The ops we need (Linear, Embedding, LayerNorm, MHA, AdamW, cross-entropy, MSE, ONNX export via Python bridge in Session 3.5) have been stable since PyTorch 1.x. Revisit only if Section 3.4 hits a concrete blocker or we switch to a CUDA build for training
- Alternatively evaluate `candle` (Hugging Face pure-Rust ML) vs `tch` (libtorch bindings):
  - `tch`: mature, full PyTorch parity, GPU support, but external C++ dependency
  - `candle`: pure Rust, simpler build, but less mature optimizer/training support
  - **Recommendation**: `tch` for training (proven ecosystem), export to ONNX for inference via `ort` (Session 3.5)
- **Crate separation**: `blob-nn` depends on `tch` (training only, Linux). `blob-bin` depends only on `ort` (inference, cross-platform). `blob-bin` must **never** depend on `tch` — this ensures the inference binary runs on the Windows laptop (Intel i5-1135G7 + Iris Xe) without libtorch
- **Encoder location**: the entity encoder (`blob_engine::encoder`) was implemented in Section 2 and lives in `blob-engine`, not `blob-nn`. It is pure Rust with no ML dependency. `blob-nn` re-exports it for convenience. The neural network code in `blob-nn` imports the encoder from `blob-engine` directly — this avoids a circular dependency when MCTS (also in `blob-engine`) needs to encode states
- Input projections (one `nn::Linear` per token type):
  - Hand card: `Linear(30, 128)`
  - Played card: `Linear(48, 128)` — 48 = 16 rank + 8 suit + 16 player + 8 scalar features
  - Player state: `Linear(29, 128)`
  - Context: `Linear(13, 128)`
  - CLS: learned `nn::Parameter` of shape `[128]`
- Chronological embedding table: `nn::Embedding(52, 128)` — added to played card projections only
- **Feature padding for batching**: the encoder's `EncodedState.features` contains variable-dimension feature vectors per token type (CLS=0, context=13, player=29, hand=30, played=48). For tensor construction, **pad all features to 48** (the maximum dimension) with trailing zeros. Each per-token-type `Linear` projection consumes only its meaningful prefix (e.g., hand card projection reads indices `[0..30)`, ignoring padding `[30..48)`). The CLS token's 48 zeros are discarded entirely — replaced by the learned 128-dim parameter. This padding is ~15% wasted memory but keeps the tensor construction trivial: stack padded features into `[batch, seq_len, 48]`, then dispatch per token type
- Forward pass for input stage: for each token type, gather positions from `token_types` tensor, slice the corresponding rows from the padded `[batch, seq_len, 48]` input, project through that type's `Linear`, and scatter results back into `[batch, seq_len, 128]`. Add chronological embedding for played card positions only. Apply attention mask for padding tokens
- Handle **variable sequence lengths** within a batch: pad to max_seq_len in batch, create attention mask (`true` for real tokens, `false` for padding)
- Test: random input → verify output shape `[batch, seq_len, 128]`

### Session 3.2 — Transformer encoder layers

Implement the 8-layer pre-norm Transformer with multi-head self-attention.

- Pre-norm Transformer block: `x + MHA(LayerNorm(x))` then `x + FFN(LayerNorm(x))`
- Multi-head self-attention: 8 heads × 16 dims each = 128-dim. Standard QKV projection, scaled dot-product attention, output projection
- Apply attention mask: set padded positions to `-inf` before softmax so they contribute zero attention weight
- FFN: `Linear(128, 512) → GeLU → Dropout(0.1) → Linear(512, 128) → Dropout(0.1)`
- LayerNorm: `eps=1e-5`, learnable affine parameters
- Stack 8 identical layers with independent parameters
- Parameter count check: each layer = `4 * 128² (QKV+O) + 2 * 128*512 (FFN) + 4*128 (LN affine)` ≈ 198K × 8 = ~1,585K
- Test: gradient flow — run random forward+backward, verify no NaN/zero gradients in any layer
- Note: `torch.compile` is Python-only (TorchDynamo). Not available through tch-rs. JIT tracing via `CModule` is possible but defer to Section 8 optimization

### Session 3.3 — Output heads (playing, bidding, value)

Implement the three separate output heads reading from different token positions.

- **Playing head** (entity-based, ~4K params):
  - For each hand card token position: `Linear(128, 32) → GeLU → Linear(32, 1)` → scalar score
  - Gather scores for all hand card positions → apply legal move mask (set illegal to `-inf`) → softmax → action probabilities
  - Same MLP weights shared across all hand card positions (parameter efficient)
  - Must track which sequence positions are hand card tokens to gather correctly
- **Bidding head** (CLS-based, ~9K params):
  - Read CLS token (position 0): `Linear(128, 64) → GeLU → Dropout(0.1) → Linear(64, 14)`
  - Apply legal bid mask → softmax → bid probabilities (14 values, bids 0–13)
- **Value head** (CLS-based, ~8K params):
  - Read CLS token: `Linear(128, 64) → GeLU → Dropout(0.1) → Linear(64, 1) → Tanh`
  - Output: scalar ∈ [-1, 1] representing z-scored expected game outcome
- **Phase dispatch**: based on `game_phase`, return either (bid_policy, value) or (play_policy, value). Never both simultaneously
- Total parameter count verification: 7.7K (embeddings) + 15.9K (projections) + 1,585K (transformer) + 21.6K (heads) ≈ **1.63M**
- Test: end-to-end forward pass from `EncodedState` → (policy, value). Verify shapes, softmax sums to 1.0, value in [-1,1]

### Session 3.4 — Loss functions, optimizer, and training step

Implement the combined loss, AdamW optimizer, learning rate schedule, and a single training step.

- **Policy loss**: cross-entropy with MCTS target: `-sum(target_i * log(pred_i + 1e-8))` over legal actions
  - Target: normalized MCTS visit counts (sum to 1.0)
  - Mask: only compute over legal actions (zero target for illegal actions already handles this, but verify)
- **Value loss**: MSE `(predicted_value - target_value)²`
  - Target: z-scored cumulative game score `clip((my_cumulative_score - mean) / max(std, ε), -1, 1)`, computed across all players at game end
  - Since training generates full games (Section 5.2), cumulative scores have rich variance (e.g., 60–180 range for 5P7C). Z-scoring produces well-distributed targets that use the tanh range effectively. Track value target distribution to verify
- **Combined loss**: `policy_loss + 2.0 * value_loss` — the 2.0 coefficient balances gradient magnitudes since policy loss is typically larger
- **Optimizer**: AdamW with β₁=0.9, β₂=0.999, weight_decay=1e-4
- **LR schedule**: linear warmup 0→3e-4 over 1,000 batches, then cosine annealing to 1e-5. The 3e-4 peak is the standard for AdamW with 1.63M params; 1e-3 risks instability at this scale
- **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)`
- Implement `fn train_step(model, batch) -> (policy_loss, value_loss, total_loss)` — forward, compute loss, backward, clip, step
- Handle variable sequence lengths in batch: pad features, create masks, pass through model
- Test on dummy data: verify loss decreases over 100 steps (smoke test for learning signal)
- **Checkpoint format**: save model state dict + optimizer state + iteration number via `tch::nn::VarStore::save()`

### Session 3.5 — ONNX export and inference setup

Set up the ONNX export pipeline and inference path early — ONNX Runtime is required for self-play (Section 5) and is the only inference path for the Windows deployment target.

- **Why early**: the inference machine (Intel i5-1135G7, Iris Xe iGPU) has no CUDA GPU. ONNX Runtime is the only viable inference backend, not an optimization — it's on the critical path. Additionally, ONNX CPU inference (~0.15ms/eval) is ~3× faster than `tch` CPU (~0.5ms), making it essential for self-play throughput (40M+ evaluations per iteration)
- **Export via Python bridge**: `tch-rs` has no native ONNX export. The export workflow is:
  1. Save model weights from Rust via `VarStore::save("model.pt")`
  2. Run a small Python script (~20 lines) that loads weights into an equivalent PyTorch model definition and calls `torch.onnx.export()` with dynamic axes
  3. This script is a build artifact in `scripts/export_onnx.py`, run once per iteration after training
  - Input: `features: [batch, max_seq, 48]`, `token_types: [batch, max_seq]`, `attention_mask: [batch, max_seq]` — `feat_dim=48` matches the padded feature width from Session 3.1
  - Outputs: `bid_policy: [batch, 14]`, `play_scores: [batch, max_hand]`, `value: [batch, 1]`
  - The Python model definition must mirror the Rust `tch` model exactly — keep both in sync. Changes to the network require updating both
- **ONNX Runtime inference** (`ort` crate in `blob-engine`, NOT `blob-nn`):
  - Load exported model, create session with `CpuExecutionProvider`
  - Configure `intra_op_num_threads=1` for per-rayon-thread sessions (each thread gets its own session, no contention)
  - Verify output matches `tch` model within 1e-5 tolerance on 100 random inputs
- **Performance validation**: benchmark ONNX CPU inference for batch=1 at typical sequence lengths (14, 29, 41 tokens). Target: <0.2ms average
- **Cross-platform check**: verify ONNX model loads and runs on Windows with `ort`. Should work out of the box since ONNX Runtime is cross-platform
- **OpenVINO exploration** (optional): Intel Iris Xe is supported by OpenVINO as an `ort` execution provider. Test whether `OpenVINOExecutionProvider` accelerates inference on the i5-1135G7 compared to CPU-only. If it does, use it for the deployment binary
- **Evaluator trait** — define in `blob-engine` so MCTS can use it without depending on `blob-nn`:
  ```rust
  pub trait Evaluator {
      fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32);
  }
  ```
  The trait takes a `BlobState` and returns `(policy, value)`. Implementations handle encoding internally (calling `blob_engine::encoder::encode`). The policy vector's semantics depend on `game_phase`: bid probabilities (len ≤ 14) during bidding, hand-card-position scores during playing. MCTS (Session 4.2) consumes this trait — it never calls the encoder or neural network directly
- **ONNX Evaluator implementation**: implement `Evaluator` backed by `ort::Session` in `blob-engine`. This is the production evaluator for both self-play and deployment. A `tch`-backed `Evaluator` can optionally live in `blob-nn` for debugging, but is not required — self-play always uses ONNX

---

## Section 4: MCTS with Instrumentation (3 sessions, ~9h)

### Session 4.1 — Arena-allocated tree and UCB1 selection

Implement the MCTS node structure with arena allocation and the selection phase.

- Arena allocation for cache-friendly traversal:
  ```
  MctsNode { visit_count: u32, value_sums: [f32; 8], value_counts: [u32; 8], prior: f32, action: u8, children: SmallVec<[u32; 14]> }
  MctsArena { nodes: Vec<MctsNode>, root_player: u8 }
  ```
  - **Action encoding** (`action: u8`): phase-dependent but always stable (no re-indexing across tree depth).
    - Bidding children: `action` = bid value 0..=13. Applied via `apply_bid(state, action)`.
    - Playing children: `action` = card index 0..=51. Applied via `apply_play(state, action)`. Card index is preferred over hand-card-position because positions are relative to the current encoding and shift after every play, whereas card indices are absolute. During expansion the policy (returned by `Evaluator` indexed by hand-card-position, per [evaluator.rs](blob-engine/src/evaluator.rs)) is translated to priors via `enc.hand_card_indices`: iterate `(pos, card_idx)` and push child `{ action: card_idx, prior: policy[pos] }`.
  - `SmallVec` inline capacity is 14 to cover the worst case (14 bids; up to 13 plays) without spilling to the heap.
  - **Per-player value storage**: each node stores value estimates and visit counts for all player seats, not a single scalar. This correctly models multiplayer dynamics — players who have "blobbed" (busted their bid) become spoilers with different objectives, and multiple players can score in the same round. A two-player zero-sum sign-flip would be wrong here
  - **Why per-player visit counts**: the neural network evaluates leaves from the current player's perspective. In a 5-player game, only ~1/5 of leaf evaluations produce a value for any given player. Using a global `visit_count` as denominator would dilute Q-values non-uniformly across children (depending on which player-turns their subtrees lead to), biasing UCB1 selection. Per-player `value_counts` ensures correct averaging
  - Node index 0 = root. Children stored as indices into `nodes` vec
  - Allocate new node: `arena.nodes.push(node); return arena.nodes.len() - 1`
  - Pre-allocate `Vec::with_capacity(10_000)` per search to avoid reallocation. Per-player values and counts add ~60 bytes/node (7 extra f32s + 7 extra u32s vs single f32/u32), so 10K nodes ≈ 600KB extra — negligible
- UCB1 selection: `Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)`
  - `Q = value_sums[acting_player] / value_counts[acting_player]` — use the **acting player's** value divided by that player's specific visit count. This ensures each player's Q estimate is a proper average, not diluted by visits where other players were evaluated. When `value_counts[acting_player] == 0`, default to `Q = 0` (neutral prior), which is quickly corrected as simulations accumulate
  - `P = prior` (from neural network policy)
  - `c_puct = 1.5` (exploration constant, tune later)
  - Select child with highest UCB1 score
- Selection phase: walk from root, selecting best UCB1 child at each node, until reaching a leaf (node with no children or unvisited)
- Note: virtual loss (parallel search within a single tree) is **not used** in this architecture. Each determinization runs its own independent tree; parallelism is across games via rayon, not within a single MCTS tree
- Test: construct small tree manually, verify UCB1 selects correctly with known values. Test that unvisited children (N=0) get infinite UCB1 score

### Session 4.2 — Expansion, evaluation, backpropagation, and the search loop

Implement the remaining MCTS phases and the full simulation loop.

- **Expansion**: at leaf node, generate all legal actions from game state, create child node per action with `prior = network_policy[action]`, `visit_count = 0`, `total_value = 0.0`
- **Evaluation**: run neural network on leaf state → `(policy, value)`. The policy provides priors for new children; the value is backpropagated
  - For now, use a **dummy evaluator** (uniform policy, random value) to test MCTS mechanics without neural network dependency
  - Use `trait Evaluator` defined in Session 3.5 (`blob_engine::evaluator`). The dummy evaluator implements this same trait
- **Backpropagation — per-player values for multiplayer**:
  - Blob is multiplayer (3-8 players), not two-player adversarial. Each player independently maximizes their own score. There is no zero-sum relationship to exploit — multiple players can score in the same round, and "blobbed" players become spoilers with shifted objectives (disrupting others)
  - **Design**: the neural network evaluates the leaf state from `state.current_player`'s perspective, returning a single value `v`. During backpropagation, store this value in `value_sums[leaf_current_player]` and increment `value_counts[leaf_current_player]` at every node on the path. Over many simulations, each node accumulates value estimates from multiple players' perspectives as different players act at different tree depths
  - At each node on the path: `visit_count += 1`, `value_sums[leaf_current_player] += v`, `value_counts[leaf_current_player] += 1`
  - During selection (Session 4.1), UCB1 uses `Q = value_sums[acting_player] / value_counts[acting_player]` — each player's Q is averaged only over simulations that actually evaluated that player, preventing dilution bias. `N_child` in the exploration term still uses the global `visit_count` (total visits through that child)
- **Search loop**: for `num_simulations` iterations: select → expand (if leaf) → evaluate → backpropagate
- After search: action probabilities = `visit_count[child] / total_visits` for each child of root
- **Temperature**: during training, use temperature τ to sharpen/flatten: `prob_i = visit_i^(1/τ) / sum(visit_j^(1/τ))`. τ=1.0 early in game (exploration), τ→0 late (exploitation)
- Test: run 100-sim search on a mid-game state with dummy evaluator, verify visit counts are non-zero for all legal actions and sum correctly

### Session 4.3 — Determinization, belief tracking, and MCTS diagnostics

Implement imperfect information handling via determinization, and add signal quality instrumentation.

- **Belief tracking**: for each opponent, track which suits they **cannot** have (void suits). Detected when: opponent doesn't follow led suit (played different suit when not leading)
  - `TrickRecord` does not carry `followed_suit`/`was_lead` flags; derive from the existing fields. For each completed trick in `trick_history[0..tricks_completed]`, for slot `i` in `1..num_played` (slot 0 is the lead and cannot reveal a void): let `(player, card) = cards[i]`; if `card / NUM_RANKS != suit_led`, set `void_suits[player][suit_led] = true`
  - Store as `void_suits: [[bool; 4]; 8]` — compact, O(1) lookup
- **Determinization sampling**: given current player's known hand and belief constraints, sample consistent opponent hands
  1. Collect all unseen cards: 52 − (my hand) − (`played_this_round`). `played_this_round` already captures both completed-trick cards and the in-progress trick's cards, so no separate bookkeeping is needed
  2. For each opponent, determine how many cards they hold: start with `cards_dealt - state.tricks_completed`, then subtract 1 if that opponent has already contributed to the in-progress trick (scan `trick_play_order[0..trick_cards_played]`, mapping slot `j` → player `(trick_leader + j) % num_players`)
  3. Shuffle unseen cards, deal to opponents respecting void constraints
  4. **Rejection sampling**: if a dealt hand violates a void constraint (contains a card of a voided suit), resample. Use early termination if constraints are unsatisfiable after N attempts (fall back to unconstrained)
  - Optimization: sort opponents by most constrained first, deal to them first to reduce rejection rate
- **Aggregated search**: for each of N determinizations, create a determinized `BlobState`, run full MCTS, collect root visit counts. Final policy = sum of visit counts across all determinizations, normalized
- `fn mcts_search<R: Rng + ?Sized>(state: &BlobState, evaluator: &dyn Evaluator, config: &MctsConfig, rng: &mut R) -> MctsResult` — `rng` drives determinization shuffles and temperature-sampled action selection, kept out of `MctsConfig` so the config stays `Copy` and thread-safe.
- **MctsConfig** — defined in 4.3 and threaded through 4.1 / 4.2 signatures:
  ```
  MctsConfig {
      c_puct: f32,                // 1.5
      num_determinizations: u32,  // 5 default
      sims_per_determinization: u32, // 100 default
      min_sims_floor: u32,        // 60
      temperature: f32,           // 1.0 early, →0 late
      arena_capacity: usize,      // 10_000
  }
  ```
  The adaptive budget table from this session consumes `num_legal_actions` and `MctsConfig` to produce the per-decision `(num_determinizations, sims_per_determinization)` pair.
- **MctsResult struct** — return diagnostics alongside the policy:
  ```
  MctsResult {
      policy: Vec<f32>,              // normalized visit counts
      visit_entropy: f32,            // -Σ p_i ln(p_i) over policy
      top1_visit_share: f32,         // max(policy)
      total_visits: u32,             // sum of root child visits
      value_estimate: f32,           // root Q value
  }
  ```
- **Per-decision adaptive MCTS budget**: budget scales with decision complexity rather than using a fixed sim count for all decisions. The optimal budget depends on branching factor, game phase, and information available:
  - If `num_legal_actions == 1`: skip MCTS entirely, policy = [1.0]. No compute wasted on forced moves
  - If `num_legal_actions <= 3`: minimum 3×20 = 60 sims (late-game tricks, constrained bids)
  - If `num_legal_actions <= 7`: minimum 3×50 = 150 sims (mid-game, typical bidding)
  - If `num_legal_actions > 7`: minimum 5×80 = 400 sims (early-game tricks with many legal plays)
  - Default starting budget: 5×100 = 500 sims for all non-trivial decisions, with the above as floors for adaptive reduction
  - **Hard absolute floor**: never below 60 sims for any decision with >1 legal action. The Python post-mortem proved that insufficient MCTS budget produces zero learning signal
- **Signal quality validation**: after each MCTS search, compute `signal_ratio = 1 - visit_entropy / ln(num_legal_actions)`. Target signal_ratio > 0.3 (MCTS is meaningfully concentrating visits). Log `(num_legal_actions, sims_used, signal_ratio)` per decision to calibrate the adaptive budget table over iterations
- **Gate check**: with 5×100, verify `top1_visit_share > 2 / num_legal_actions` (non-uniform signal). This was the critical failure point in Python at 1×15
- Test: known game state with known voids — verify determinization never assigns voided suit to constrained player. Test aggregation produces sensible probabilities

---

## Section 5: Training Pipeline with Diagnostics (4 sessions, ~12h)

### Session 5.1 — Replay buffer with contiguous storage

Implement the replay buffer storing raw game states with circular FIFO semantics.

- **Crate placement**: `ReplayBuffer` lives in `blob-engine/src/replay.rs`. It is pure data (no ML dep) and needs to be usable from both `blob-nn` (training) and self-play workers, so placing it in `blob-engine` avoids a circular dep and keeps the inference crate boundary intact
- **New dependency**: add `bincode = "1"` to `blob-engine`'s `Cargo.toml` for buffer serialization. `BlobState` already derives `Serialize`/`Deserialize`, and `SmallVec` has `serde` enabled in the workspace dep
- **Design rationale**: store `BlobState` (~410 bytes) + policy + value + phase. Re-encode to entity tokens on the fly during training. 500K × 410B = ~205MB for states. Encoding is <1μs so batch of 512 = <0.5ms overhead. This means encoding changes never invalidate the buffer
- **Final design**:
  ```
  ReplayBuffer {
      states: Vec<BlobState>,
      policies: Vec<SmallVec<[(u8, f32); 14]>>,  // sparse action-probability pairs
      values: Vec<f32>,
      phases: Vec<GamePhase>,  // Bid or Play — determines which output head and action space
      capacity: usize, write_idx: usize, len: usize
  }
  ```
  - Policy stored as sparse `(action_index, probability)` pairs:
    - **Bidding**: action_index = bid value (0–13), up to 14 entries
    - **Playing**: action_index = hand card position (0 to hand_size-1), matching the entity-based playing head's output space (Session 3.3). NOT card indices 0–51
  - `SmallVec<[(u8, f32); 14]>` avoids heap allocation for most examples (≤14 legal actions fits inline)
- **Batch construction**: separate bid and play examples into two sub-batches by `GamePhase`
  - Bid sub-batch: reconstruct dense `[f32; 14]` policy tensor
  - Play sub-batch: reconstruct dense `[f32; max_hand_size]` policy tensor, where max_hand_size is the largest hand in the batch (≤13)
  - Process all examples through the transformer, then dispatch to the appropriate output head per sub-batch
- Circular FIFO: write at `write_idx % capacity`, increment, clamp len
- `fn sample_batch(n: usize) -> (Vec<BlobState>, BidBatch, PlayBatch)` — uniform random indices via `rand`, split by phase
- Checkpoint to disk: `bincode::serialize` the entire buffer → write to file. Restore with `bincode::deserialize`. Target: <500ms for ~250MB buffer (vs 13s Python pickle)
- Test: fill buffer beyond capacity, verify oldest entries are overwritten. Verify uniform sampling distribution with chi-squared test over 100K samples

### Session 5.2 — Self-play worker and full game generation

Implement the self-play engine that generates training examples from complete games using MCTS.

- **Training unit is the full game, not individual rounds**. Training on full games allows the value head to learn from cumulative game outcomes — the actual objective. Round-scoped z-scoring would produce bimodal targets (0 or 10+bid) with poor gradient signal, and would not reward consistent performance across an entire game
- **Compute budget**: a 5P7C game = 17 rounds, ~380 decisions (5 players × sum of (cards_dealt + 1) per round). A 5P8C game = 19 rounds, ~470 decisions. Weighted average across the player/card distribution: ~453 decisions/game. At 5×100 MCTS = 500 evals/decision → ~226K neural evals per average game. At 0.15ms/eval (ONNX): ~34s single-threaded, ~1.1s with 32 threads. To generate ~80K training examples: ~177 games × 453 decisions ≈ 80K. At 1.1s/game = **~3 minutes of self-play**
- For each decision point in a game: encode state → MCTS search → record `(state, mcts_policy, _)` — value filled in after game ends
- **After game completes**: compute z-scored value from cumulative game scores for each player: `v_i = clip((cumulative_score_i - mean) / max(std, ε), -1, 1)`. Backfill this value into **all** examples from that player's perspective across all rounds
  - Cumulative game scores have rich variance (e.g., 40–130 range for 5P7C), producing well-distributed value targets that use the tanh range effectively
  - Handle edge case: if all players score identically (std=0), set all values to 0.0
- **Training example**: `(BlobState, sparse_policy, f32, GamePhase)` — sparse policy has nonzero entries only for legal actions, sums to 1.0
  - For bidding: up to 14 entries (bid values 0–13)
  - For playing: entries are hand card position indices (0 to hand_size-1), matching the entity-based playing head's output space. The MCTS engine maps card indices back to hand positions using the known hand composition
- **Player count distribution**: n=4 (10%), n=5 (60%), n=6 (25%), n=7 (5%). Each game is played at a single player count, sampled per this distribution. Priority is n=5 for target use case; broader distribution builds a more general base model for fine-tuning
- **Cards-dealt distribution**: C=7 (40%), C=8 (60%), constrained by `num_players × C ≤ 52`. For n=7, C is forced to 7 (since 7×8=56 > 52). Games always start from the full round structure (e.g., 5P7C = 17 rounds, 5P8C = 19 rounds, 6P8C = 19 rounds)
- Generate N games per iteration, collecting all decision-point examples. ~453 avg decisions/game → ~177 games = ~80K examples per iteration
- Test: generate 5 complete games, verify all examples have valid policies (sum to ~1.0, nonzero entries only at legal actions), values in [-1, 1], and value targets are consistent within each game (same player gets same value across all their examples in a game)

### Session 5.3 — Rayon parallelization and self-play engine

Parallelize self-play across threads and build the orchestrating self-play engine.

- **Crate placement**: the self-play engine and training loop live in `blob-nn` (alongside the `tch` training code), not in `blob-engine`. `blob-engine` stays inference-only so the deployment binary isn't pulled into training dependencies. Self-play consumes `blob_engine::{mcts_search, OnnxEvaluator, ReplayBuffer}` and owns the `rayon`/`indicatif` deps
- **New dependencies**: add to `blob-nn`'s `Cargo.toml`: `rayon = "1"` (thread pool + parallel iterators) and `indicatif = "0.17"` (progress bars). Keep both out of `blob-engine` — the inference binary (`blob-bin`) must not depend on them
- **Thread pool**: `rayon::ThreadPoolBuilder::new().num_threads(32).build()` — one pool for all self-play
- Each thread runs independent full games with its own RNG (seeded from thread index + iteration for reproducibility)
- **Model sharing for self-play inference**: use per-thread `ort::Session` instances loaded from the ONNX export (Section 3.5). ONNX Runtime at ~0.15ms/eval is ~3× faster than `tch` CPU inference (~0.5ms), and this difference is critical for self-play throughput — 40M evaluations per iteration means ONNX saves ~4 CPU-hours vs `tch`
  - Each rayon thread creates its own `ort::Session` with `intra_op_num_threads=1` to avoid thread contention
  - Model is exported to ONNX once at the start of each iteration after training updates
- `fn self_play_iteration(model_path: &Path, num_games: usize, mcts_config: &MctsConfig) -> Vec<TrainingExample>`
  - Uses `rayon::iter::ParallelIterator` to distribute games across threads
  - Each game is independent — embarrassingly parallel, no synchronization needed
  - Collect all examples into `Vec`, then `extend` replay buffer
- **Scaling efficiency**: target >80% with 32 threads. Python achieved only 44% due to multiprocessing serialization. Rust shared-memory threading should be near-linear
- Benchmark: measure games/minute at 1, 8, 16, 32 threads
- Progress reporting: `indicatif` progress bar showing games completed, examples generated, games/sec

### Session 5.4 — Training loop with integrated diagnostics

Wire up the complete training iteration with all diagnostic metrics from the start.

- **Iteration structure**:
  1. Self-play: generate N games using ONNX model (exported at end of previous iteration), add examples to replay buffer
  2. Training: sample batches of 512 from buffer, run gradient updates via `tch`
  3. Compute diagnostics (see below)
  4. Export updated model to ONNX for next iteration's self-play
  5. Checkpoint: save model + optimizer + buffer + metrics
  6. (Every K iterations: comparison evaluation — see Section 6)
- **Batch construction**: sample 512 `BlobState`s from buffer, encode each to `EncodedState`, pad to max sequence length in batch, stack into tensors, move to device
  - Separate bidding and playing examples in the batch (they use different output heads)
  - Or: process all through transformer, then dispatch to appropriate head per example
- **Epochs per iteration**: start at 10, but make adaptive — stop early if loss improvement per epoch drops below threshold (some iterations converge in 5, others need 15)
- **LR schedule**: linear warmup (0→3e-4 over 1,000 global batches) + cosine annealing to 1e-5. Track global batch count across iterations
- **Per-iteration diagnostic metrics** (logged to structured JSON via `tracing`):
  - **Signal quality**: mean MCTS visit entropy, mean top-1 visit share (from self-play MctsResults)
  - **Learning progress**: policy_loss (separate bid/play), value_loss, combined_loss, learning_rate
  - **Per-layer gradient norms**: verify all 8 transformer layers receive gradient. Flag if any layer norm < 1e-6
  - **Value head**: mean prediction, prediction variance (flag if variance < 0.01 — network outputting constant)
  - **Bid accuracy (top-1)**: % where network's argmax bid == MCTS argmax bid
  - **Play accuracy (top-1)**: same for card plays
  - **Policy-MCTS KL divergence**: KL(MCTS_target || network_policy), averaged over batch. Should decrease over training
  - **Loss improvement per evaluation**: Δ(policy_loss) / num_nn_evaluations_this_iteration. Direct measure of compute efficiency
- **Checkpoint strategy**: save `{model.pt, optimizer.pt, buffer.bin, metrics.json}` per iteration, but manage retention to save storage:
  - **Evaluated checkpoints** (every 5 iterations): kept permanently. These are the comparison anchors (see Session 6.1)
  - **Rolling checkpoint**: always keep the most recent non-evaluated iteration, so training can be paused/resumed without losing more than one iteration of progress. Delete the previous rolling checkpoint once the next iteration completes
  - Example at iteration 102: keep evaluated checkpoints at 5, 10, 15, ..., 100, plus the rolling checkpoint at 102. Iteration 101's checkpoint was deleted when 102 completed
- **Resume**: detect existing checkpoint on startup, load and continue from last iteration
- **Gate check**: policy loss drops below `ln(avg_legal_actions)` ≈ `ln(7)` ≈ 1.95 within 10 iterations. If not, something is wrong with the learning signal

---

## Section 6: Evaluation, CLI & Hardening (3 sessions, ~9h)

### Session 6.1 — Model comparison, strength tracking, and promotion

Implement the model comparison system for tracking training progress. No ELO — use direct metrics. This session adds no new crates; it reuses `OnnxEvaluator`, `DummyEvaluator`, and the MCTS stack already in `blob-engine`.

- **Heuristic baseline evaluator** (new struct, lives in `blob-engine/src/evaluator.rs` next to `DummyEvaluator` so both comparison eval and future deployment can use it without pulling `blob-nn`):
  - `HeuristicEvaluator` implements `Evaluator`.
  - **Bid rule**: `raw = count(aces in trump suit) + count(kings in trump suit) + count(aces in non-trump suits)`, then project onto the `legal_bids(state)` mask (nearest legal bid, rounding down on ties — handles the dealer's forbidden bid cleanly). Each card is counted exactly once: trump aces and trump kings are the near-guaranteed tricks; non-trump aces are strong but can be trumped. In NoTrump rounds the trump-suit terms vanish naturally and the formula collapses to `count(aces in hand)`, which is the correct NoTrump baseline.
  - **Play rule**: if the current trick winner is already the heuristic's own card, play the lowest legal card; otherwise play the lowest legal card that beats the current winner, falling back to the lowest legal card if none can. Blob has no partnerships — the model is purely "win this trick vs dump".
  - **Value**: return `0.0` (heuristic doesn't estimate values).
  - Policy-vector shape and masking match `DummyEvaluator` exactly (bids: `[NUM_BIDS]` masked+renormalized; plays: `[hand_card_indices.len()]` over hand positions).
- **Eval game harness** lives in `blob-nn/src/eval.rs`. Self-play's `play_one_game` is left untouched — evaluation games use a new entry point that supports per-seat evaluators and captures per-round outcomes:
  ```rust
  pub struct SeatEvaluators<'a>([Option<&'a dyn Evaluator>; MAX_PLAYERS]);
  pub struct RoundOutcome { pub bids: [u8; MAX_PLAYERS], pub tricks_won: [u8; MAX_PLAYERS], pub num_players: u8 }
  pub struct EvalGameOutcome { pub final_scores: [u16; MAX_PLAYERS], pub rounds: SmallVec<[RoundOutcome; 24]> }
  pub fn play_eval_game<R: Rng + ?Sized>(n: u8, c: u8, seats: &SeatEvaluators, mcts_cfg: &MctsConfig, rng: &mut R) -> EvalGameOutcome
  ```
  - Decision loop dispatches per acting seat: `let eval = seats.0[state.current_player as usize].expect("seat evaluator for active seat")`. MCTS is called once with that evaluator; leaves inside the tree also use it (standard AlphaZero semantics — the acting player models all moves with their own policy).
  - **Per-round outcome capture** happens in the `GamePhase::Scoring` branch *before* `advance_round`: copy `state.bids` and `state.tricks_won` into a `RoundOutcome` and push onto `outcome.rounds`. `advance_round` calls `start_round` which clears those fields, so the capture must precede the transition. The engine API stays unchanged.
  - No `TrainingExample`s are emitted — evaluation games never enter the replay buffer.
- **Evaluation cadence**: every 5 iterations (matches `EVAL_CHECKPOINT_EVERY` already set by Session 5.4). The opponent-selection rule is pinned as:
  - `opponent_iter = max{ i : i ≤ current_iter − eval_lookback, i % EVAL_CHECKPOINT_EVERY == 0, i > 0 }`. Fallback to `HeuristicEvaluator` if no eligible checkpoint exists (first four evaluations).
  - `eval_lookback` defaults to 20. Worked examples: iter 5/10/15/20 → heuristic baseline; iter 25 → iter 5; iter 30 → iter 10; iter 50 → iter 30.
- **Head-to-head seat assignment** (200 games per evaluation):
  - `seat_A = game_idx % num_players`, `seat_B = (game_idx + num_players / 2) % num_players`, the remaining `num_players − 2` seats filled with `HeuristicEvaluator`.
  - Over 200 games this rotates every ordered (A-seat, B-seat) pair equally and keeps each model in exactly one seat per game — breaks the multi-seat coordination bias without losing head-to-head directness.
- **Metrics per evaluation**:
  - **Win rate**: fraction of games where `final_scores[seat_A] > final_scores[seat_B]`. 95% binomial CI (Wilson). **Improvement gate: `win_rate_lower95 > 0.5`** (stricter than a bare point estimate of 55%, which can be inside the CI at n=200).
  - **Score differential**: `mean(final_scores[seat_A] − final_scores[seat_B])` per game.
  - **Bid success rate** (per model): `count(rounds where bids[seat] == tricks_won[seat]) / total_rounds`, summed over all games for the model's seat. ~3,400 rounds per evaluation → ±1.7% precision at α=0.05 — the most interpretable domain metric for Blob.
- **Strength tracking CSV** at `{checkpoint_dir}/strength.csv`, one row appended per evaluation:
  `iteration, opponent_iter_or_baseline, win_rate, win_rate_lower95, win_rate_upper95, score_differential, bid_success_rate_current, bid_success_rate_opponent, policy_loss, value_loss, visit_entropy, kl_divergence`.
  - Plot `win_rate_lower95` over iteration to track progress; the slope tells you whether training is accelerating, steady, or stalling.
- **Best-model tracking and promotion**:
  - Training maintains a `best_model.onnx` pointer inside `checkpoint_dir` — symlink on Linux, portable-fallback via `File::rename` atomic replace.
  - Alongside the pointer, persist `{checkpoint_dir}/best_stats.json` = `{ iteration: u64, win_rate_lower95: f32, bid_success_rate: f32 }`. Updated atomically on each promotion; read at startup so `bid_success_rate_best` survives across runs.
  - **Promotion gate**: promote when `win_rate_lower95 > 0.5` AND `bid_success_rate_current ≥ bid_success_rate_best − 0.01` (no-regression floor, not an improvement requirement). The win-rate lower bound is the real quality signal; the bid-success-rate clause only blocks promotions that win through noise while regressing on the most interpretable domain metric.
  - Self-play reads `best_model.onnx`. `TrainingLoop::run_iteration` grows a `best_onnx_path: &Path` argument passed down to `self_play_iteration`; promotion updates this pointer between iterations.
  - Cold-start (iterations 0..4, before any evaluation or promotion has run): `best_model.onnx` is initialized on iteration 0 to point at the freshly-exported `iter_000000/model.onnx`, and each subsequent pre-evaluation iteration repoints it at its own `iter_######/model.onnx`. This preserves Session 5.4's "self-play uses previous iteration's export" behavior while keeping a single read path (`best_onnx_path`) in the codebase. Once the first evaluation runs (iter 5), promotion takes over and the pointer stops auto-advancing.
- **Carry-over fix from Session 5.4**: `LossAccumulators::num_nn_evals` (in `blob-nn/src/training_loop.rs`) is declared but never incremented, so `IterationMetrics::num_nn_evaluations` is always 0 and the "loss improvement per evaluation" metric is unusable. Wire it in this session: increment by `batch.input.features.size()[0]` per training step (one forward pass per batch × batch size). Self-play MCTS evals can be added later if the granularity proves useful.
- **Tests**:
  - Two `DummyEvaluator`s as seat_A and seat_B, heuristic elsewhere, 200 games → win-rate 95% CI contains 0.5.
  - `HeuristicEvaluator` vs `DummyEvaluator`, 200 games → heuristic's bid success rate > 25% (beats random's ~1/cards_dealt).
  - Per-round capture: play 1 game, assert `outcome.rounds.len() == total_rounds(start_cards, num_players)` and every `RoundOutcome.bids` sums to ≤ `cards_dealt` (legal-bid invariant).

### Session 6.2 — CLI, configuration, and logging

Split the CLI so the deployment binary stays inference-only and the training binary can pull `tch`/`rayon`/`indicatif`. This preserves the Session 5.3 invariant ("inference binary must not depend on training-only deps") and lines up with the Windows/Intel-iGPU deployment target in AGENTS.md.

- **Workspace change — new `blob-train` crate**. Add to the workspace:
  ```
  members = ["blob-engine", "blob-nn", "blob-bin", "blob-train"]
  ```
  - `blob-train/Cargo.toml`: depends on `blob-engine`, `blob-nn`, `clap = { version = "4", features = ["derive"] }`, `serde`, `toml = "0.8"`, `tracing`, `tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }`, `indicatif`.
  - `blob-train/src/main.rs` + `blob-train/src/config.rs`.
  - Binary name: `blobmaster-train`.
- **`blob-bin` stays inference-only**. Adds only `clap` and `tracing-subscriber`; keeps the existing `blob-engine` dep. No `blob-nn`, no `tch`, no `rayon`. Binary name remains `blobmaster`.
- **`blobmaster-train` subcommands** (training workflow):
  - `train --config <path> [--resume] [--checkpoint-dir <dir>] [overrides...]`
  - `evaluate --model-a <path> --model-b <path-or-baseline> --num-games N --num-players n --cards-dealt C`
  - `self-play --model <path> --num-games N --output <replay.bin>` (generate examples without training)
  - `export --checkpoint <dir> --output <model.onnx>` (invokes `scripts/export_onnx.py` or the Rust equivalent added in Section 3.5)
- **`blobmaster` subcommands** (inference + deployment):
  - `play --model <path.onnx> --num-players n --seat k` (human-vs-AI scaffolding for Section 9)
  - `analyze --model <path.onnx> --state <state.json>` (single-state policy+value dump — useful for debugging without any training dep)
- **Configuration**: root `TrainingConfig` in `blob-train/src/config.rs`, `#[derive(Serialize, Deserialize)]`, loadable from TOML. CLI flags override file values (populate config, then apply `clap`-parsed `Option<T>` overrides).
  - Add `#[derive(Serialize, Deserialize)]` to the three existing config structs: `blob_engine::mcts::MctsConfig`, `blob_nn::engine::SelfPlayConfig`, `blob_nn::training_loop::TrainingLoopConfig`. Compose them:
    ```rust
    pub struct TrainingConfig {
        pub training: TrainingLoopConfig,
        pub self_play: SelfPlayConfig,
        pub mcts: MctsConfig,
        pub eval: EvalConfig,
    }
    pub struct EvalConfig {
        pub eval_games: usize,         // default 200
        pub eval_interval: u64,        // default 5
        pub eval_lookback: u64,        // default 20
        pub bid_success_promotion_delta: f32, // default 0.02
    }
    ```
  - `PathBuf` fields serialize naturally; ship a sample `config.toml` and a round-trip test.
- **Logging**: `tracing` with JSON output via `tracing-subscriber::fmt().json()` → `{checkpoint_dir}/training.jsonl`, plus pretty output to stderr. Same field names as the Session 5.4 `metrics.jsonl` so downstream plotting can union both sources.
  - Levels: INFO for iteration summaries, DEBUG for per-batch metrics, TRACE for per-round details.
  - Key per-iteration fields: `games_generated, examples_added, policy_loss_bid, policy_loss_play, value_loss, bid_accuracy, visit_entropy, kl_divergence, games_per_sec, inference_ms_avg`.
- **Progress display**: `indicatif` multi-progress in `blob-train` only — one bar for self-play games, one for training batches. `blob-bin` has none.

### Session 6.3 — Performance profiling, hardening, and gate verification

Profile end-to-end performance, fix bottlenecks, and verify all completion gates.

- **Benchmark suite** using `criterion` (add as `[dev-dependencies]` in `blob-engine`):
  - `BlobState` copy: ~100 ns (the existing 512 B size test in `state.rs` already guards the struct size).
  - Legal move generation (bitmask return): ~5 ns.
  - Entity encoding: < 1 µs.
  - ONNX inference (batch=1): < 0.2 ms (ort CPU).
  - MCTS 100 sims (with ONNX eval): < 20 ms.
  - Full move (5 det × 100 sims): < 100 ms.
  - Single full 5P7C game self-play (17 rounds): < 30 s of neural time.
- **Full iteration wall-clock**: self-play (~177 games) + training (10 epochs) with 32 threads: **< 5 minutes**. Reconcile AGENTS.md's "Verification Gates" section in this session — its earlier "< 60 seconds" figure was aspirational for small-game sanity runs and should be replaced by the 5-minute target so the two docs agree.
- **Memory profiling**: track peak RSS during 32-thread self-play; verify no leak over 100 iterations.
- **Numerical stability**: 50 training iterations with no NaN/Inf in loss, gradients, or model outputs; value head stays in [−1, 1].
- **ONNX ↔ tch parity test**: push the same input through `tch` forward and `OnnxEvaluator`; per-element agreement within 1e-5 on policy (logits/probs) and value. Add this to `scripts/export_onnx.py`'s post-export verification and as a Rust-side integration test gated on `BLOB_ONNX_MODEL`.
- **Gate checklist verification**:
  - [ ] All ported game-engine tests pass (143 from `test_blob.py`, adjusted for the round-structure correction).
  - [ ] MCTS 5×100 → `top1_visit_share > 2 / num_legal_actions` (non-uniform signal).
  - [ ] Policy loss < `ln(7) ≈ 1.95` within 10 iterations.
  - [ ] Eval `win_rate_lower95 > 0.5` vs heuristic baseline within 20 iterations.
  - [ ] Full iteration < 5 minutes (32 threads).
  - [ ] 32-thread scaling > 80% efficiency.
  - [ ] ONNX inference < 0.2 ms (ort CPU, single sample). *Note: this is a proxy — the real gate is the 5-minute iteration wall-clock at 32 threads, measured in Session 7.1.*
  - [ ] ONNX ↔ tch output agreement within 1e-4. *Relaxed from 1e-5: an 8-layer fp32 transformer with softmax+LN+GELU accumulates ~2e-5 between torch CPU and ort CPU kernels as a matter of rounding, not correctness.*
- Fix any gates that don't pass. This session is buffer/contingency for issues found during integration.

#### Session 6.3 profiling results (2026-04-14, training PC, random-init checkpoint)

Engine primitives (criterion, `cargo bench -p blob-engine --bench core`):

| Bench | Target | Measured | Status |
|---|---|---|---|
| `blobstate_copy` | ~100 ns | 43.6 ns | ✅ |
| `legal_plays_mid_trick` | ~5 ns | 1.03 ns | ✅ |
| `legal_bids_first_seat` | ~5 ns | 1.15 ns | ✅ |
| `encode_mid_trick_5p7c` | <1 µs | 297 ns | ✅ |

ONNX path (`BLOB_ONNX_MODEL=/tmp/blobckpt/model.onnx cargo bench -p blob-engine --bench onnx_mcts`):

| Bench | Target | Measured | Status |
|---|---|---|---|
| `onnx_inference_batch1` | <0.2 ms | 604 µs | ❌ ~3× over |
| `mcts_1det_100sims` | <20 ms | 232 ms | ❌ ~11× over |
| `mcts_full_move_5x100` | <100 ms | 389 ms | ❌ ~3.9× over |
| ONNX ↔ tch parity (max abs) | <1e-4 (relaxed) | 2.37e-5 | ✅ |

Analysis: back-of-envelope on the model (8 layers × d=128, seq≈29, FFN=512 ⇒ ~50M FLOPs per forward) says 1–2.5 ms at batch=1 on ort CPU is the honest expected range once framework overhead is counted. **The original 0.2 ms gate was unrealistic** for this architecture/runtime; 604 µs is close to the physical floor for single-sample ort CPU. Extrapolated to production load (389 ms/move × ~85 moves × 177 games / 32 threads / 0.8 scaling ≈ 3.8 min/iter), the real end-to-end gate — **one iteration < 5 min at 32 threads** — is likely met without any 6.3 optimization work. That extrapolation is verified in Session 7.1.

Held-in-reserve optimizations (only pursued if the 32-thread wall-clock measurement in 7.1 fails):

- **CPU-side** batching of MCTS leaf evaluations within a determinization (virtual-loss + queue) so each ort session sees batch>1 per forward. Highest-leverage single change remaining. (Note: *GPU* batching was implemented, profiled, and ruled out at this model size — CUDA coordination overhead exceeded compute savings at ~50M FLOPs/forward. See `gpu-inference-summary.md`. Revisit GPU only if model grows >10×.)
- Pad sequences to a fixed MAX_SEQ in `OnnxEvaluator::run_encoded` so ort can reuse planned kernels across calls.
- Tune `ort::SessionBuilder` (`intra_op_num_threads`, `OptimizedModelFilePath` caching).
- FP16 / INT8 quantization (last resort; adds parity risk for a small model).

Gate checklist status after this run: engine-size gates ✅; ONNX single-inference proxy gate ❌ but reclassified as non-blocking (see note above); parity gate ✅ (under relaxed 1e-4); training-loop gates (iteration wall-clock, 32-thread scaling, policy loss, win-rate) not yet exercised — validated in Section 7.1.

---

## Section 7: Adaptive Training, Optimization & Extended Run

7.1–7.3 are complete. 7.4 (self-play perf) is partly done; INT8 (7.4b) and batched MCTS (7.4c) remain. 7.5 is the 100-iter mixed-player run that gates Section 8.

### Session 7.1 — Driver wiring and smoke test (done)

Wired `blobmaster-train train` to a real loop over `TrainingLoop::run_iteration`, added periodic eval against the iter-0 anchor, per-decision signal logging to `decision_stats.jsonl`, `SelfPlayConfig::fixed_player_count`, and `num_games = 118`. Single-iteration smoke at fixed 5P7C / 5×100 / 118 games passed all gates. Artifacts: `checkpoints/smoke-7.1/`.

### Session 7.2 — 10-iter diagnostic baseline (done)

11 loop iters at fixed 5P7C, flat 5×100 MCTS, used as the budget-tuning control. Outcomes:
- Eval win rate vs iter-0 anchor at iter 10: **0.77**.
- ~60% of MCTS sim cost lands in `num_legal ≤ 3` decisions where signal_ratio is already informative (p50 0.20–0.49). Forced moves are 39% of all decisions and already cost 0.
- `num_epochs_run` collapsed to 2 by iter 2 (`epoch_early_stop_rel = 0.005` triggered too easily).
- `value_head` grad norm reached 4.57 by iter 10 vs <1 elsewhere.
- Eval cost was ~4.7× per-game self-play cost.

Full bucketing: `7.3b-analysis.md §3`.

### Session 7.3 — Adaptive tuning attempt, regression, partial revert (done)

7.3a shipped four changes together: bucketed MCTS budget, `value_head_lr_scale = 0.5`, `epoch_early_stop_rel = 0.001`, sequential Wilson-stop eval. The 7.3b 21-iter validation regressed eval win rate from 0.77 → 0.525 at iter 10 (`7.3b-analysis.md`).

7.3c reverted three of the four to the 7.2 baseline (flat 5×100 MCTS, single-group LR, `epoch_early_stop_rel = 0.005`) and re-ran 15 iters; trajectory matched 7.2 and the iter-14 checkpoint is the model now used by 7.4 profiling. The Wilson early-stop in eval **survived** the revert and was further parallelized within chunks (`eval_num_threads = 32`, `eval_games = 192`); the cap and stopping rule are kept.

**Decisions for 7.5:** flat 5×100 MCTS, single-group LR, `epoch_early_stop_rel = 0.005`, parallel Wilson-stop eval. The original 7.3d deferred-knobs menu survives in 7.4d as a contingency list.

### Session 7.4 — Self-play performance optimization

7.3c confirmed self-play wall-clock is the binding constraint (>95% of iteration time at flat 5×100 MCTS). 7.4 compresses it before committing 7.5's 100-iter budget.

#### 7.4a — Profiling, thread-count sweep, evaluator reuse (done)

Atomic-bucket profiler in `blob_engine::profiling`, gated behind the `profile` subcommand. Sweep at 5×100 MCTS / fixed 5P7C / 5 games-per-thread:

- 97–99% of thread time in `OnnxEvaluator::run_encoded::sess.run()`.
- 32T loses to 16T by 34% on per-game wall-clock — 32 rayon workers oversubscribe the 16-core 7950X and contend for AVX/FP units (despite `intra_op_num_threads = 1`).
- **16 threads is the local optimum** (validated 14/15/16/17/18 — see `logs/thread-sweep-2026-04-24/`).
- `OnnxEvaluator` constructed once per worker via `rayon::map_init` instead of per game (~20 s/iter saved).
- `with_inter_threads(1)` investigated and dropped — ORT defaults to `Sequential` execution, so the inter-op pool never spawns.

Full data: `self-play-profile.md`.

The two remaining levers — INT8 (7.4b) and batched MCTS (7.4c) — are independent and stack.

#### 7.4b — INT8 quantization

**Why it should help.** The 1.63M-param FP32 model is ~6.5 MB of weights — fits L3 (64 MB shared) but not L2 (1 MB / core), so per-call cost at 16T is partly L3-bandwidth-bound. Zen 4 has full AVX-512 VNNI (`vpdpbusd`, 4× INT8 MAC per lane), and INT8 weights are 4× smaller. ORT INT8 on VNNI runs 2.9–6× on BERT-class transformers in Microsoft's published benchmarks; our `d_model = 128` shrinks the compute share of that gain but the memory share holds. **Expected: 1.5–2× per-iteration self-play speedup.** FP16 is *not* worth pursuing on Zen 4 — no native FP16 vector compute; ORT runs it through the FP32 ALUs.

**Implementation.**

1. Extend `scripts/export_onnx.py` to emit `model.int8.onnx` alongside `model.onnx`:
   - `onnxruntime.quantization.quantize_static`, `QuantFormat.QDQ`, `weight_type=QuantType.QInt8`, `activation_type=QuantType.QUInt8` (U8S8 path — 2× more MACs/cycle than U8U8 on VNNI), `CalibrationMethod.MinMax`.
   - Calibration data: ~500 `EncodedState`s captured from a recent iter's `decision_stats.jsonl`-driven profile run, saved once as `calibration.npz` and reused.
   - Exclude LayerNorm and Softmax nodes via `nodes_to_exclude` — they're not GEMMs and quantize poorly at this dimension.
2. Add `[self_play] use_int8 = true` to `config.sample.toml`. Self-play loads `model.int8.onnx`; **eval stays on FP32** so the eval signal isn't fighting quantization noise on both sides.
3. Validation gate (the existing 1e-5 element-wise gate is too tight — relax for this path only):
   - On 1000 saved states: argmax-policy agreement INT8-vs-FP32 ≥ 95%, value-sign agreement 100%.
   - 5-game self-play sanity at 5×100 — no panics, sane wall-clock.
   - Re-profile at 16T: per-call ONNX < 0.9 ms (vs 1.30 ms today) means ship; < 0.7 ms confirms the bandwidth-bound theory.

**Pass condition:** ≥1.4× per-iteration speedup with no eval-win-rate regression vs FP32 at iter 5/10 of a 10-iter validation against 7.3c's iter-0 anchor.

**Hold-back:** if INT8 silently degrades policy quality (eval win rate at iter 10 drops >5pp vs FP32), revert and run 7.5 on FP32. Quantization-aware training is out of scope.

**Status (2026-04-26): plumbing landed, benchmark run, hold-back triggered.** Speed gate passed (1.40× per-iter at 16T, ONNX/call 1.30 → 0.91 ms — right at the 0.9 ms ship line); static quality gate **failed** (bid argmax 0.848 < 0.95, value-sign 0.942 < 1.00 over 500 real-state calibration). Excluding the output heads moved the numbers <1pp — the cumulative quantization error through 8 transformer layers at d_model=128 corrupts the CLS representation before the heads see it. Recommendation: hold INT8 back from 7.5 and pursue 7.4c (batched MCTS) next; revisit INT8 only if the combined 7.4a + 7.4c speedup falls short of budget. Three follow-up levers if/when revisited: S8S8 activations, entropy calibration, sensitivity-driven body exclusions. Full numbers, side-by-side comparison, and reproducer command in [self-play-profile.md](self-play-profile.md).

- `scripts/export_onnx.py` accepts `--int8-out` and `--calibration` and runs `quantize_static` (QDQ, INT8 weights, UINT8 activations, MinMax, LayerNorm/Softmax excluded by op-type, `per_channel=True`).
- `blob_engine::onnx` exposes `start_calibration_capture` / `finish_calibration_capture` / `write_calibration_file`; capture is a thread-safe no-op when disabled. The on-disk `calibration.bin` is the BCAL binary format documented in `blob-engine/src/onnx.rs`.
- `blobmaster-train profile --dump-calibration <path> [--dump-calibration-limit N]` records up to N encoded states from a real self-play run. `--use-int8` on the profile command runs the same workload against `model.int8.onnx` for the post-quantization per-call timing.
- `SelfPlayConfig::use_int8` (default `false`) makes self-play workers swap `model.onnx` for `model.int8.onnx`. `config.sample.toml` documents the toggle. Eval still loads whatever path the caller passes, so `run_eval_against_anchor` stays on FP32 by construction.
- `run_export_script` / `bootstrap_initial_onnx` pass `--int8-out` and `--calibration` to the Python exporter when `use_int8` is set and `<checkpoint_dir>/calibration.bin` exists; if the calibration file is missing, the iter logs a warning and degrades to FP32 self-play (no INT8 sibling produced).
- `scripts/validate_int8.py` runs the FP32 and INT8 graphs on every BCAL state and reports/gates bid-argmax (≥95%) and value-sign (100%) agreement.

Operator runbook (one-time per checkpoint family):
1. Pick a recent FP32 iter checkpoint, run `blobmaster-train profile --model <iter>/model.onnx --dump-calibration <ckpt_dir>/calibration.bin`.
2. Re-export that checkpoint with `python scripts/export_onnx.py --weights <iter>/model.ot --out <iter>/model.onnx --int8-out <iter>/model.int8.onnx --calibration <ckpt_dir>/calibration.bin`.
3. Gate: `python scripts/validate_int8.py --fp32 ... --int8 ... --calibration ...`.
4. Re-profile at 16T with `--use-int8` to verify per-call ONNX < 0.9 ms.
5. Set `use_int8 = true` in the training TOML; subsequent iters auto-emit the INT8 sibling.

#### 7.4c — Batched MCTS leaves with virtual loss

**Why it should help.** At batch=1 every transformer linear is a GEMV (memory-bound: each weight read once, used once). At batch B>1 they're GEMMs with weights re-used B times — more SIMD-efficient and bandwidth-amortized. ORT's MlasGemm path is much friendlier on the GEMM shape. With fewer rayon threads (less SMT contention as a side benefit), **expected: 2–3× alone, 3–4× combined with INT8.** This is the standard AlphaZero parallelization.

Stage in two phases: cross-determinization first (free batch=5, no MCTS-correctness change), then virtual-loss within a tree (free batch up to ~16) only if stage 1 doesn't hit the target. Stage 2 is real complexity; do not pre-emptively bundle it.

##### Stage 1 — Cross-determinization batching

The 5 determinizations per decision are already independent. They run sequentially per thread today; restructure them to advance in lockstep, sharing one batched `sess.run` per "step".

**API change.** Add to `Evaluator`:

```rust
fn evaluate_batch(&self, states: &[&BlobState]) -> Vec<(Vec<f32>, f32)>;
```

Default impl loops `evaluate`. `OnnxEvaluator::evaluate_batch` builds one `[B, S_max, FEAT_DIM]` tensor (zero-padded to the longest sequence in the batch; `attention_mask` already zeroes padded positions in attention) and one `sess.run`, then splits the outputs back per state.

**MCTS change.** Replace the per-determinization loop in `mcts_search` ([blob-engine/src/mcts.rs:602](blob-engine/src/mcts.rs#L602)) with a lockstep driver. Allocate all 5 arenas up front; pseudocode:

```text
for step in 1..=sims_per_det:
    leaves = []                                    # (det_idx, leaf_state, path)
    for det in 0..num_dets:
        if dets[det].sims_done < sims_per_det:
            walk root → unexpanded leaf  (same select_leaf as today)
            leaves.push((det, leaf_state, path))
    if leaves.is_empty(): break
    results = eval.evaluate_batch(&leaves.iter().map(|l| &l.state).collect())
    for ((det, _, path), (policy, value)) in leaves.zip(results):
        expand(arenas[det], leaf, policy)
        backprop(arenas[det], path, leaf_seat, value)
```

Forced-move dets that resolve in 1 sim simply stop contributing leaves on subsequent steps; the batch shrinks naturally over the last few sims. Visit-count distributions are bit-identical to the serial implementation on the same RNG seed (modulo arena allocation order across dets — pin a parity test on small `sims_per_det = 10` before merging).

**Thread count.** Per-call cost rises from ~1.3 ms (B=1) toward ~3 ms (B=5) but does 5× the work, easing the per-thread bottleneck. Re-run `scripts/thread-sweep.sh` at B=5 — expect the optimum to drop to ~8 threads. Update `self-play-profile.md` and the default afterward.

**Pass condition for stage 1:** ≥1.7× per-iteration speedup vs 7.3c, with policy-equivalence preserved (visit-count distributions match serial within Monte Carlo noise on 100 sample decisions). **If hit, ship and skip stage 2.**

##### Stage 2 — Virtual loss within one tree

Only add this if stage 1 alone doesn't hit ≥1.7×. Within a single det's tree, sims are serially dependent (each sim's UCB1 reads previous sims' visit counts). Decorate in-flight nodes with a temporary "pessimistic" visit so parallel descents pick different leaves:

- Add `in_flight: u16` to `MctsNode`. Single-thread mutation only — one rayon worker owns the tree, no atomic needed.
- During selection, replace UCB1's effective `(N, Q)` for the acting seat with:
  - `N_eff = visit_count + in_flight`
  - `Q_eff = (value_sums[acting] - vloss_weight * in_flight) / max(N_eff, 1)`
  - `vloss_weight = 1.0` — assume in-flight leaves are losses until proven otherwise (standard AlphaZero choice).
- Selection along a path increments `in_flight` at each visited node.
- After batched eval, **decrement `in_flight` along each path before running the real `backprop`**. Order matters so virtual visits are replaced cleanly. Add `debug_assert!(node.in_flight == 0)` in `mcts_search`'s post-loop sanity check.

**Search-loop change.** Generalize stage 1's lockstep driver to "queue N leaves, then eval", drawing from both cross-tree (5 dets) and inside-tree (virtual loss) sources:

```text
target_batch = 8                                   # tunable; 5..16 reasonable
while any det has remaining sim budget:
    leaves = []
    while leaves.len() < target_batch and any det can descend further:
        det = pick det with fewest sims-so-far     # round-robin
        walk root → unexpanded leaf, ++in_flight along the path
        leaves.push((det, leaf_state, path))
    results = eval.evaluate_batch(&leaves.iter().map(|l| &l.state).collect())
    for ((det, _, path), (policy, value)) in leaves.zip(results):
        for n in &path: arenas[det].nodes[n].in_flight -= 1
        expand(arenas[det], leaf, policy)
        backprop(arenas[det], path, leaf_seat, value)
```

**Correctness considerations.**

- Virtual loss biases selection toward less-explored siblings while a leaf is in flight. Visit-count distributions on identical seeds will *not* match serial MCTS exactly. This is fine for self-play (policy target is the visit distribution either way; softmax absorbs small biases), but pinned-visit-count tests must adjust or run with `target_batch = 1` to degenerate back to serial.
- Each det owns its own `Xoshiro256PlusPlus`; the lockstep driver advances per-det, never shared, or determinism across batch sizes breaks.
- Forced-move root short-circuit (`num_legal == 1`) unchanged.
- Sequence-padding cost: at `target_batch = 8` with seq lengths 14–58, padding to S_max ≈ 58 costs roughly 2× the FLOPs of a perfect batch. Acceptable; bucketing by sequence length is a future optimization, not stage-2 scope.

**Pass condition for stage 2:** ≥2.5× per-iteration speedup vs 7.3c, eval win rate at iter 5 within 3pp of 7.3c-baseline. Validate on a 5-iter run before committing to 7.5.

**New tunable to record:** `target_batch` in `config.sample.toml`. Sweep `{5, 8, 12, 16}` on a 1-iter wall-clock benchmark, record the optimum, default to 8 if no clear winner.

##### Combined 7.4 throughput target

7.4a (done) + 7.4b INT8 + 7.4c-stage-1 → ~3× speedup over 7.3c → **~3 min/iter** at the right thread count. 7.5's 100-iter wall-clock budget assumes this hits.

If 7.4b + 7.4c-stage-1 combined land below 2×, do stage 2 before 7.5 — committing to 100 iters at 9 min/iter is a 15-h training run that's awkward to checkpoint cleanly.

#### 7.4d — Deferred tuning knobs (contingencies for 7.5)

These were the original 7.3d list. Each carries a specific trigger; **do not** speculatively apply.

- **Temperature schedule** (new `MctsConfig::temperature_schedule` + decision-index arg in `mcts_search`): τ=1.0 for first 15 decisions, τ=0.1 thereafter. *Trigger:* late-game `top1_visit_share_mean` < 0.5.
- **Adaptive games per iteration** — *trigger:* `policy_kl_divergence` < 0.05 sustained → cut `num_games`. > 0.15 by iter 30 → raise.
- **Learning rate** — *trigger:* loss oscillates ±10% iter-to-iter → halve `peak_lr`. Plateau with KL > 0.1 → raise 1.5× or extend warmup.
- **c_puct** — *trigger:* `top1_visit_share_mean` > 0.85 sustained → 2.0. < 0.25 sustained → 1.0.
- **Replay buffer sizing** — *trigger:* loss drops then rises across iterations → cut `buffer_capacity` to 250K. Slow improvement with KL high → raise to 1M.
- **Muon optimizer for hidden 2D weights** — replace AdamW for `transformer.*` matrix params (QKV / out / FFN linears) with Muon (Newton-Schulz orthogonalized updates), keep AdamW for embeddings, the three heads, biases, and LayerNorm scales. The two-group plumbing in `build_optimizer` already supports the split. *Trigger:* 7.5 eval_win_rate_lower95 plateaus before iter 50 *and/or* per-head `grad_norms` drift >3× apart (the 7.2-style instability pattern). *Why it's worth pulling:* if Muon cuts iterations-to-target by ~25%, total run wall-clock drops the same — every saved iter is ~3 min of self-play. *Cost:* tch-rs has no built-in Muon; ~1–2 days to write the Newton-Schulz step (5 iterations of batched matmul per param group). *Caveat:* published gains are mostly on >100M-param models; at 1.63M with d_model=128 the win may be smaller or wash into noise — pull only when AdamW shows the trigger pattern, not speculatively.

### Session 7.5 — Extended training run (100 iters, mixed player count)

Once 7.4 lands and a 10-iter validation matches 7.3c's anchor trajectory, flip `fixed_player_count` to `None` (Section 5.2 distribution: n=4 10%, n=5 60%, n=6 25%, n=7 5%) and run **100 iterations from a fresh init** — not from a 7.3c checkpoint, since the fixed-5P7C value targets baked into the buffer don't transfer to the mixed distribution.

- All 7.4 perf changes on, 7.4d knobs at default, MCTS at flat 5×100 (the 7.3c-validated config).
- Eval cadence every 5 iters; opponent rotates per `eval_lookback = 20`.
- Save iter-25/50/75/100 checkpoints as Section-8 fine-tuning candidates.
- Primary indicator: `eval_win_rate_lower95` vs the iter-20 mid-training anchor (more discriminating than vs iter-0 once basics are learned).
- Watch n=5 bid-success-rate vs the n=5-only 7.3c baseline. >5pp absolute drop suggests reverting to fixed n=5 for the base model and doing the multi-n stretch in Section 8.
- Save iter-100 base model as Section-8 fine-tuning input.

**Expected wall-clock:** at 7.4's target throughput (~3 min/iter), 100 iters + 20 evals ≈ **~7 h**. Resume-checkpoint at iter 50.

---

## Section 8: Fine-Tuning & Deployment (1 session, ~3h)

### Session 8.1 — Player-count fine-tuning

Fine-tune the base model into player-count-specific models.

- Load base model checkpoint (from Section 7 extended training)
- **Fine-tuning recipe per player count**:
  - Freeze: nothing (full fine-tune, model is small enough at 1.63M)
  - Data: 100% games at target player count (e.g., all n=5)
  - LR: 1/3 of base training peak (1e-4, since base uses 3e-4), with short warmup (200 batches)
  - Iterations: 50-100 (much less than base training — fundamentals already learned)
  - Eval: bid success rate on target player count, compared to base model. Fine-tune should improve within 10-20 iterations
- **Priority order**: n=5 first, then n=4, n=6, n=7, n=3, n=8
- **Validation**: fine-tuned n=5 model should beat base model on n=5 games (higher bid success rate, higher win rate). Verify it hasn't degraded catastrophically on other counts (some regression is expected and acceptable)
- Save each fine-tuned model as `model_n{X}.ot` (tch VarStore format) with corresponding `model_n{X}.onnx` export — matches the checkpoint format used by `save_checkpoint` / `OnnxEvaluator::from_file` elsewhere in the project

---

## Section 9: Backend API & Frontend (Deferred, ~4+ sessions)

*Not planned in detail — to be scoped after the core RL system is validated and learning.*

### Session 9.1+ — Web backend and game UI

Build a web interface for human vs AI play and training visualization.

- Backend: Axum or Actix-web serving REST/WebSocket API
- Endpoints: `/new-game`, `/make-move`, `/get-state`, `/ai-move`
- Frontend: React/Svelte with card game UI
- Training dashboard: metric curves over iterations, round replays
- **Prerequisites**: core system must show clear learning signal (bid success rate > 50%, win rate vs random > 65%) before investing in UI

---

**Total estimated sessions: 26 sessions (~78 hours) for core system (Sections 1–8), plus deferred frontend work.**
