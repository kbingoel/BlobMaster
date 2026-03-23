# BlobMaster Development Plan

> **Terminology**: a "game" = full multi-round session (e.g., 23 rounds for 5 players). A "round" = one deal-bid-play-score cycle. A "trick" = each player plays one card. See README.md vocabulary table.

## Section 1: Rust Foundation & Game Engine (4 sessions, ~12h)

### Session 1.1 — Workspace scaffold and card primitives

Set up the Rust workspace structure and implement the low-level card representation with bitwise operations.

- `cargo init --name blobmaster` with workspace members: `blob-engine`, `blob-nn`, `blob-bin`
  - Start lean — 3 crates: engine (game + encoder + MCTS), nn (model + training), bin (CLI + eval). Split further only when boundaries stabilize
  - Additional crates (encoder, mcts, eval) can be extracted later without breaking APIs
- Add shared dependencies in workspace `Cargo.toml`: `serde`, `rand`, `rand_xoshiro`, `smallvec`, `tracing`
- Implement card encoding: `card_index = suit * 13 + rank` (Suits: ♠=0 ♥=1 ♣=2 ♦=3, Ranks: 2=0 through A=12)
- Implement `u64` bitmask operations: `hand.add(card)`, `hand.remove(card)`, `hand.contains(card)`, `hand.count()`, `hand.iter()`, `hand.cards_of_suit(suit)`
- Use `u64` bit manipulation: bit N = card N present. `popcount` for count, `trailing_zeros` for iteration, mask `& (0x1FFF << (suit*13))` for suit extraction
- Define `TrickRecord { cards: [(u8, u8); 8], num_played: u8, winner: u8, suit_led: u8 }` — stack-allocated, no heap
- Define `BlobState` struct (~410 bytes):
  - `hands: [u64; 8]`, `played_this_round: u64` (maintained incrementally — needed by encoder for `cards_above_remaining`, `is_highest_in_suit`)
  - `bids: [u8; 8]`, `tricks_won: [u8; 8]`, `trump_suit: u8`, `current_player: u8`, `dealer: u8`, `num_players: u8`, `cards_dealt: u8`, `game_phase: u8`
  - `trick_leader: u8`, `trick_play_order: [u8; 8]`, `trick_cards_played: u8`
  - `trick_history: [TrickRecord; 13]`, `tricks_completed: u8`
  - `cumulative_scores: [u16; 8]` (multi-round tracking, needed by player state tokens)
- Write unit tests for all bitmask ops; benchmark `BlobState` copy (target: ~100ns via `memcpy` of stack struct — 410 bytes across ~6 cache lines)

### Session 1.2 — Dealing, bidding phase, and trump rotation

Implement the game initialization, dealing logic, and the complete bidding phase.

- Trump rotation: `[Spades, Hearts, Clubs, Diamonds, NoTrump]` cycling every 5 rounds — store as `u8` (0–4)
- Round structure: symmetric pattern — descends from C to 1, stays at 1 for N rounds, ascends back to C. `C = 52 / num_players` (integer division). Number of "stay at 1" rounds = `num_players`
- Dealing: shuffle deck (Fisher-Yates with `rand_xoshiro::Xoshiro256PlusPlus` for speed), distribute `cards_dealt` cards per player as `u64` bitmasks
- Bidding rules: each player bids 0..=cards_dealt. **Dealer restriction**: dealer's bid cannot make total bids equal cards_dealt (forces at least one player to miss). Validate this constraint
- `fn legal_bids(state: &BlobState) -> u16` — bitmask over values 0..=13, with forbidden value cleared for dealer. Max 14 possible bids fits in u16. Avoids heap allocation (unlike Vec)
- `fn apply_bid(state: &mut BlobState, bid: u8)` — set `bids[current_player]`, advance player, transition to playing phase when all have bid
- Port relevant tests from Python's `test_blob.py` (bidding group: ~25 tests)
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
- Port trick-taking tests (~60 tests from Python), including edge cases: void suits, trump overrides, NoTrump rounds

### Session 1.4 — Full game loop, round sequencing, and property tests

Wire up the complete multi-round game and add comprehensive testing including property-based tests.

- `fn new_game(num_players: u8) -> BlobState` — initializes first round (cards_dealt = 52 / num_players)
- `fn advance_round(state: &mut BlobState)` — reset trick state, clear `played_this_round`, advance dealer, rotate trump, adjust cards_dealt per symmetric round structure, accumulate scores into `cumulative_scores`
- Full game loop: deal → bid → play tricks → score → next round, until all rounds complete
- `fn is_game_over(state: &BlobState) -> bool` — true when last round scored
- `fn total_rounds(num_players: u8) -> u8` — compute total rounds for a given player count
- Add `proptest` for property-based testing: random games always terminate, card conservation (52 cards never lost/duplicated), tricks_won always sums to cards_dealt, no player can play a card not in their hand
- Port remaining ~50 tests from Python's `test_blob.py`
- **Gate check**: all 135 ported tests pass, `BlobState` copy benchmarked, legal move gen benchmarked
- `#[derive(Clone, Copy)]` on `BlobState` — ~410 bytes, copy is just memcpy

---

## Section 2: Entity Encoder (3 sessions, ~9h)

### Session 2.1 — Shared embeddings and hand card tokens

Implement the shared embedding tables and the hand card token encoder.

- Shared embedding tables (learned parameters, but for encoding we produce raw features — embeddings are in the neural network): rank one-hot (16-dim, 13 values + 3 padding), suit one-hot (8-dim, 4 values + 4 padding), player one-hot (16-dim, 8 values + 8 padding)
- Decision: encoder produces **raw feature vectors** per token; the neural network's input projection layers convert to d_model=128. Encoder is pure Rust, no ML dependency
- Hand card token (30 dims): rank(16) + suit(8) + is_trump(1) + suit_count_in_hand(1) + is_highest_in_suit(1) + is_lowest_in_suit(1) + cards_above_remaining(1) + cards_below_remaining(1)
- "Remaining" means not in own hand and not yet played — uses `played_this_round` bitmask from `BlobState`. `cards_above_remaining(card)` = popcount of cards in same suit with higher rank, not in hand, not in `played_this_round`
- `is_highest_in_suit` / `is_lowest_in_suit`: among cards of that suit remaining in the game (hand + unplayed), is this card the extreme?
- Output: `Vec<[f32; 30]>` with 1–13 entries (one per card in hand)
- Test: construct known game states, verify feature values exactly match expected. Especially test derived features (highest/lowest, remaining counts) after several tricks played

### Session 2.2 — Played card tokens and player state tokens

Implement the chronologically-ordered played card tokens and per-player state tokens.

- **Played card token** (48 dims): rank(16) + suit(8) + player(16) + 8 scalar features: trick_number(1) + position_in_trick(1) + was_lead(1) + followed_suit(1) + is_trump_play(1) + trick_complete(1) + won_trick(1) + is_current_trick(1). Total: 16+8+16+8 = 48. Note the 8 scalar features as a group to avoid off-by-one errors during projection implementation
- Iterate `trick_history[0..tricks_completed]` plus current trick's `trick_play_order` — emit tokens in strict chronological order
- `followed_suit`: card suit == trick's led suit. Critical signal: `followed_suit=0 && was_lead=0` reveals a **suit void** for that player
- `won_trick`: only set for the card that won a completed trick (highest trump, or highest of led suit)
- Chronological position index (0–47) stored alongside each token — the neural net adds a learned chronological embedding from a 52×128 table
- **Player state token** (29 dims): player(16) + bid(1) + tricks_won(1) + tricks_needed(1) + bid_status(1) + is_dealer(1) + is_me(1) + relative_position(1) + cumulative_score(1) + cards_in_hand(1) + void_spades(1) + void_hearts(1) + void_clubs(1) + void_diamonds(1)
- `bid_status`: encode as -1 (busted, tricks_won > bid), 0 (live, tricks_won <= bid and tricks_remaining sufficient), +1 (met, tricks_won == bid)
- `void_*` flags: precomputed from played card history — scan all played cards where `followed_suit=0 && was_lead=0`, mark that player as void in the led suit. **Single most informative signal** for belief tracking
- `relative_position`: `(player_idx - current_player) % num_players`, normalized to [0,1]
- `is_me`: 1.0 for the player whose perspective this encoding represents
- `cumulative_score`: read from `BlobState.cumulative_scores`, normalized by max possible score

### Session 2.3 — Context token, CLS, sequence assembly, and validation

Build the context token, CLS token placeholder, assemble the full variable-length sequence, and validate with internal consistency tests.

- **Context token** (13 dims): trump_suit one-hot(5, includes NoTrump) + cards_dealt(1, normalized) + current_trick(1, normalized) + tricks_remaining(1) + num_players(1, normalized) + round_number(1, normalized) + game_phase one-hot(2) + bidding_constraint_active(1)
- `bidding_constraint_active`: 1.0 only during bidding phase when current player is dealer and constraint applies
- **CLS token**: no features from encoder — it's a learned 128-dim parameter in the neural network. Encoder emits a zero-vector or special marker
- **Sequence assembly order**: `[CLS, context, player_states..., hand_cards..., played_cards...]`
  - Total tokens: 1 + 1 + num_players + hand_size + cards_played = 14–56
  - Must also emit token type IDs (0–4) for each position, so the network knows which input projection to use
- Output struct: `EncodedState { features: Vec<Vec<f32>>, token_types: Vec<u8>, chronological_indices: Vec<u8>, num_tokens: usize }`
- **Internal consistency tests**: construct known game states by hand, verify each token type's features match expected values exactly. Test derived features (highest/lowest in suit, remaining counts, void flags) across various game phases — early trick, mid-game, and late-game states. No cross-validation against the Python legacy encoder, since the Rust encoder produces an entirely different representation (variable-length entity tokens vs. fixed 256-dim vector)
- Benchmark: encoding should be <1μs per state (mostly arithmetic on small arrays)

---

## Section 3: Neural Network — Structured Entity Transformer (5 sessions, ~15h)

### Session 3.1 — tch-rs setup and input projection layers

Set up the libtorch Rust bindings and implement the per-token-type input projections.

- Add `tch = "0.17"` (or latest) to `blob-nn/Cargo.toml`. Requires libtorch installed — document setup for Linux (training only)
- Alternatively evaluate `candle` (Hugging Face pure-Rust ML) vs `tch` (libtorch bindings):
  - `tch`: mature, full PyTorch parity, GPU support, but external C++ dependency
  - `candle`: pure Rust, simpler build, but less mature optimizer/training support
  - **Recommendation**: `tch` for training (proven ecosystem), export to ONNX for inference via `ort` (Session 3.5)
- **Crate separation**: `blob-nn` depends on `tch` (training only, Linux). `blob-bin` depends only on `ort` (inference, cross-platform). `blob-bin` must **never** depend on `tch` — this ensures the inference binary runs on the Windows laptop (Intel i5-1135G7 + Iris Xe) without libtorch
- Input projections (one `nn::Linear` per token type):
  - Hand card: `Linear(30, 128)`
  - Played card: `Linear(48, 128)` — 48 = 16 rank + 8 suit + 16 player + 8 scalar features
  - Player state: `Linear(29, 128)`
  - Context: `Linear(13, 128)`
  - CLS: learned `nn::Parameter` of shape `[128]`
- Chronological embedding table: `nn::Embedding(52, 128)` — added to played card projections only
- Forward pass for input stage: iterate tokens, dispatch to appropriate projection by token_type, add chronological embedding for played cards, collect into `[batch, seq_len, 128]` tensor
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
- Parameter count check: each layer = `4 * 128² (QKV+O) + 2 * 128*512 (FFN) + 4*128 (LN affine)` ≈ 198K × 8 = ~1,585K. Verify matches architecture.md
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
- **LR schedule**: linear warmup 0→3e-4 over 1,000 batches, then cosine annealing to 1e-5 (matching architecture.md §4.6). The 3e-4 peak is the standard for AdamW with 1.63M params; 1e-3 risks instability at this scale
- **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)`
- Implement `fn train_step(model, batch) -> (policy_loss, value_loss, total_loss)` — forward, compute loss, backward, clip, step
- Handle variable sequence lengths in batch: pad features, create masks, pass through model
- Test on dummy data: verify loss decreases over 100 steps (smoke test for learning signal)
- **Checkpoint format**: save model state dict + optimizer state + iteration number via `tch::nn::VarStore::save()`

### Session 3.5 — ONNX export and inference setup

Set up the ONNX export pipeline and inference path early — ONNX Runtime is required for self-play (Section 5) and is the only inference path for the Windows deployment target.

- **Why early**: the inference machine (Intel i5-1135G7, Iris Xe iGPU) has no CUDA GPU. ONNX Runtime is the only viable inference backend, not an optimization — it's on the critical path. Additionally, ONNX CPU inference (~0.15ms/eval) is ~3× faster than `tch` CPU (~0.5ms), making it essential for self-play throughput (40M+ evaluations per iteration)
- **Export**: trace the `tch` model with dummy input → save as ONNX. Handle dynamic sequence length via dynamic axes
  - Input: `features: [batch, max_seq, feat_dim]`, `token_types: [batch, max_seq]`, `attention_mask: [batch, max_seq]`
  - Outputs: `bid_policy: [batch, 14]`, `play_scores: [batch, max_hand]`, `value: [batch, 1]`
- **ONNX Runtime inference** (`ort` crate in `blob-engine` or shared crate, NOT `blob-nn`):
  - Load exported model, create session with `CpuExecutionProvider`
  - Configure `intra_op_num_threads=1` for per-rayon-thread sessions (each thread gets its own session, no contention)
  - Verify output matches `tch` model within 1e-5 tolerance on 100 random inputs
- **Performance validation**: benchmark ONNX CPU inference for batch=1 at typical sequence lengths (14, 29, 41 tokens). Target: <0.2ms average
- **Cross-platform check**: verify ONNX model loads and runs on Windows with `ort`. Should work out of the box since ONNX Runtime is cross-platform
- **OpenVINO exploration** (optional): Intel Iris Xe is supported by OpenVINO as an `ort` execution provider. Test whether `OpenVINOExecutionProvider` accelerates inference on the i5-1135G7 compared to CPU-only. If it does, use it for the deployment binary
- **Evaluator trait implementation**: implement `trait Evaluator` (from Session 4.2) backed by `ort::Session`, so MCTS can use ONNX inference directly

---

## Section 4: MCTS with Instrumentation (3 sessions, ~9h)

### Session 4.1 — Arena-allocated tree and UCB1 selection

Implement the MCTS node structure with arena allocation and the selection phase.

- Arena allocation for cache-friendly traversal:
  ```
  MctsNode { visit_count: u32, value_sums: [f32; 8], prior: f32, action: u8, children: SmallVec<[u32; 8]> }
  MctsArena { nodes: Vec<MctsNode>, root_player: u8 }
  ```
  - **Per-player value storage**: each node stores value estimates for all player seats, not a single scalar. This correctly models multiplayer dynamics — players who have "blobbed" (busted their bid) become spoilers with different objectives, and multiple players can score in the same round. A two-player zero-sum sign-flip would be wrong here
  - Node index 0 = root. Children stored as indices into `nodes` vec
  - Allocate new node: `arena.nodes.push(node); return arena.nodes.len() - 1`
  - Pre-allocate `Vec::with_capacity(10_000)` per search to avoid reallocation. Per-player values add ~28 bytes/node (7 extra f32s vs single f32), so 10K nodes ≈ 280KB extra — negligible
- UCB1 selection: `Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)`
  - `Q = value_sums[acting_player] / visit_count` — use the **acting player's** value at each node during selection. This ensures each player's moves are selected to maximize their own expected outcome
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
  - Define trait: `trait Evaluator { fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32); }`
- **Backpropagation — per-player values for multiplayer**:
  - Blob is multiplayer (3-8 players), not two-player adversarial. Each player independently maximizes their own score. There is no zero-sum relationship to exploit — multiple players can score in the same round, and "blobbed" players become spoilers with shifted objectives (disrupting others)
  - **Design**: the neural network evaluates the leaf state from `state.current_player`'s perspective, returning a single value `v`. During backpropagation, store this value in `value_sums[current_player]` at every node on the path. Over many simulations, each node accumulates value estimates from multiple players' perspectives as different players act at different tree depths
  - At each node on the path: `visit_count += 1`, `value_sums[leaf_current_player] += v`
  - During selection (Session 4.1), UCB1 uses `Q = value_sums[acting_player] / visit_count` — each player's moves are evaluated from their own perspective. Nodes where the acting player has never been evaluated default to `Q = 0` (neutral prior), which is quickly corrected as simulations accumulate
- **Search loop**: for `num_simulations` iterations: select → expand (if leaf) → evaluate → backpropagate
- After search: action probabilities = `visit_count[child] / total_visits` for each child of root
- **Temperature**: during training, use temperature τ to sharpen/flatten: `prob_i = visit_i^(1/τ) / sum(visit_j^(1/τ))`. τ=1.0 early in game (exploration), τ→0 late (exploitation)
- Test: run 100-sim search on a mid-game state with dummy evaluator, verify visit counts are non-zero for all legal actions and sum correctly

### Session 4.3 — Determinization, belief tracking, and MCTS diagnostics

Implement imperfect information handling via determinization, and add signal quality instrumentation.

- **Belief tracking**: for each opponent, track which suits they **cannot** have (void suits). Detected when: opponent doesn't follow led suit (played different suit when not leading)
  - Scan `trick_history`: for each play where `followed_suit == false && was_lead == false`, mark that player as void in the led suit
  - Store as `void_suits: [[bool; 4]; 8]` — compact, O(1) lookup
- **Determinization sampling**: given current player's known hand and belief constraints, sample consistent opponent hands
  1. Collect all unseen cards: 52 - (my hand) - (all played cards via `played_this_round`)
  2. For each opponent, determine how many cards they hold: `cards_dealt - tricks_completed` (minus cards played this trick)
  3. Shuffle unseen cards, deal to opponents respecting void constraints
  4. **Rejection sampling**: if a dealt hand violates a void constraint (contains a card of a voided suit), resample. Use early termination if constraints are unsatisfiable after N attempts (fall back to unconstrained)
  - Optimization: sort opponents by most constrained first, deal to them first to reduce rejection rate
- **Aggregated search**: for each of N determinizations, create a determinized `BlobState`, run full MCTS, collect root visit counts. Final policy = sum of visit counts across all determinizations, normalized
- `fn mcts_search(state: &BlobState, evaluator: &dyn Evaluator, config: &MctsConfig) -> MctsResult`
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
    - **Playing**: action_index = hand card position (0 to hand_size-1), matching the entity-based playing head's output space (architecture.md §4.5). NOT card indices 0–51
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
- **Compute budget**: a 5P7C game = 17 rounds × ~40 decisions = ~680 decisions. At 5×100 MCTS = 500 evals/decision → ~340K neural evals per game. At 0.15ms/eval (ONNX): ~51s single-threaded, ~1.6s with 32 threads. To generate ~80K training examples: ~118 games × 680 decisions ≈ 80K. At 1.6s/game = **~3 minutes of self-play** — comparable to the previous round-based estimate
- For each decision point in a game: encode state → MCTS search → record `(state, mcts_policy, _)` — value filled in after game ends
- **After game completes**: compute z-scored value from cumulative game scores for each player: `v_i = clip((cumulative_score_i - mean) / max(std, ε), -1, 1)`. Backfill this value into **all** examples from that player's perspective across all rounds
  - Cumulative game scores have rich variance (e.g., 60–180 range for 5P7C), producing well-distributed value targets that use the tanh range effectively
  - Handle edge case: if all players score identically (std=0), set all values to 0.0
- **Training example**: `(BlobState, sparse_policy, f32, GamePhase)` — sparse policy has nonzero entries only for legal actions, sums to 1.0
  - For bidding: up to 14 entries (bid values 0–13)
  - For playing: entries are hand card position indices (0 to hand_size-1), matching the entity-based playing head's output space. The MCTS engine maps card indices back to hand positions using the known hand composition
- **Player count distribution**: n=4 (10%), n=5 (60%), n=6 (25%), n=7 (5%). Each game is played at a single player count, sampled per this distribution. Priority is n=5 for target use case; broader distribution builds a more general base model for fine-tuning
- **Cards-dealt configuration**: for each player count, use the standard maximum `C = 52 / num_players`. Games always start from the full round structure (e.g., 5P10C = 23 rounds, 4P13C = 27 rounds)
- Generate N games per iteration, collecting all decision-point examples. ~680 decisions per 5P7C game → ~118 games = ~80K examples per iteration
- Test: generate 5 complete games, verify all examples have valid policies (sum to ~1.0, nonzero entries only at legal actions), values in [-1, 1], and value targets are consistent within each game (same player gets same value across all their examples in a game)

### Session 5.3 — Rayon parallelization and self-play engine

Parallelize self-play across threads and build the orchestrating self-play engine.

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
- **LR schedule**: linear warmup (0→3e-4 over 1,000 global batches) + cosine annealing to 1e-5 (matching architecture.md). Track global batch count across iterations
- **Per-iteration diagnostic metrics** (logged to structured JSON via `tracing`):
  - **Signal quality**: mean MCTS visit entropy, mean top-1 visit share (from self-play MctsResults)
  - **Learning progress**: policy_loss (separate bid/play), value_loss, combined_loss, learning_rate
  - **Per-layer gradient norms**: verify all 8 transformer layers receive gradient. Flag if any layer norm < 1e-6
  - **Value head**: mean prediction, prediction variance (flag if variance < 0.01 — network outputting constant)
  - **Bid accuracy (top-1)**: % where network's argmax bid == MCTS argmax bid
  - **Play accuracy (top-1)**: same for card plays
  - **Policy-MCTS KL divergence**: KL(MCTS_target || network_policy), averaged over batch. Should decrease over training
  - **Loss improvement per evaluation**: Δ(policy_loss) / num_nn_evaluations_this_iteration. Direct measure of compute efficiency
- **Checkpoint**: `{model.pt, optimizer.pt, buffer.bin, metrics.json}` per iteration. Keep last 5 checkpoints, delete older
- **Resume**: detect existing checkpoint on startup, load and continue from last iteration
- **Gate check**: policy loss drops below `ln(avg_legal_actions)` ≈ `ln(7)` ≈ 1.95 within 10 iterations. If not, something is wrong with the learning signal

---

## Section 6: Evaluation, CLI & Hardening (3 sessions, ~9h)

### Session 6.1 — Model comparison and strength tracking

Implement the model comparison system for tracking training progress. No ELO — use direct metrics.

- **Checkpoint comparison**: every K=10 iterations, play current model vs checkpoint from 20 iterations ago
  - Each model controls exactly **one seat** per game. Remaining seats filled by a fixed baseline (random or heuristic). This avoids the multi-seat coordination bias of putting one model in multiple seats
  - Play 50 full games with fair seat rotation. Track:
    - **Win rate** (higher cumulative game score more often): primary comparison metric
    - **Score differential**: mean(model_A_cumulative - model_B_cumulative) per game. More granular than binary win
    - **Bid success rate**: % of rounds across all games where model hits bid exactly. Domain-specific strength measure
  - Win rate with 95% confidence interval (binomial). Need >55% to declare improvement
- **Strength tracking over time** (replaces ELO):
  - CSV log per iteration: `iteration, win_rate_vs_prev, bid_success_rate, score_differential, policy_loss, value_loss, visit_entropy, kl_divergence`
  - Plot curves: the slope of win_rate_vs_prev tells you whether training is accelerating, steady, or stalling
  - **Bid success rate** is the single most interpretable domain metric — a strong Blob player bids accurately
- **Heuristic baseline** (valuable intermediate benchmark): simple rule-based player — bid = count of (aces + kings in trump suit + aces in non-trump), play highest legal card if winning else lowest. Win rate vs heuristic measures absolute progress beyond random
- **Promotion logic**: if current model's bid success rate exceeds best model's by >2% over 50 games, promote as new best. Self-play uses best model for training stability
- Test: run comparison with two random models — verify ~50% win rate within statistical noise

### Session 6.2 — CLI, configuration, and logging

Build the command-line interface for running training, evaluation, and analysis.

- **CLI** with `clap` derive macro: subcommands `train`, `evaluate`, `self-play`, `export`
  - `train`: `--iterations`, `--games-per-iter`, `--batch-size`, `--epochs`, `--resume`, `--checkpoint-dir`
  - `evaluate`: `--model-a`, `--model-b`, `--num-games`, `--num-players`
  - `self-play`: `--model`, `--num-games`, `--output` (generate examples without training)
  - `export`: `--model`, `--output` (export to ONNX)
- **Configuration**: `TrainingConfig` struct with serde, loadable from TOML file or CLI args. CLI args override file values
  - MCTS params: `determinizations`, `simulations_per_det`, `c_puct`, `temperature`, `temp_threshold`
  - Training params: `lr_peak`, `lr_min`, `weight_decay`, `batch_size`, `max_epochs_per_iter`, `buffer_capacity`
  - Self-play params: `num_games`, `num_threads`, `player_distribution: [f32; 4]` (for n=4,5,6,7)
  - Eval params: `eval_rounds`, `eval_interval`, `promotion_threshold`
- **Logging**: `tracing` with structured fields. Log levels: INFO for iteration summaries, DEBUG for per-batch metrics, TRACE for per-round details
  - Output to both terminal (pretty) and file (JSON lines) via `tracing-subscriber` layers
  - Key metrics per iteration: games_generated, examples_added, policy_loss_bid, policy_loss_play, value_loss, bid_accuracy, visit_entropy, kl_divergence, games_per_sec, inference_ms_avg
- **Progress display**: `indicatif` multi-progress bars during training — one for self-play rounds, one for training batches

### Session 6.3 — Performance profiling, hardening, and gate verification

Profile end-to-end performance, fix bottlenecks, and verify all completion gates.

- **Benchmark suite** using `criterion`:
  - `BlobState` copy: target ~100ns (~410 bytes — verify actual size with `std::mem::size_of::<BlobState>()`)
  - Legal move generation (bitmask return): target ~5ns
  - Entity encoding: target <1μs
  - ONNX inference (single, batch=1): target <0.2ms (ort CPU)
  - MCTS 100 sims (with ONNX eval): target <20ms
  - Full move (5 det × 100 sims): target <100ms
  - Single full game self-play (5P7C, 17 rounds): target <30s neural time
- **Full iteration benchmark**: self-play (~118 games) + training (10 epochs) — target <5 minutes total with 32 threads
- **Memory profiling**: track peak RSS during 32-thread self-play. Verify no memory leaks over 100 iterations
- **Numerical stability**: run 50 training iterations, verify no NaN/Inf in loss, gradients, or model outputs. Check value head stays in [-1, 1] range
- **Gate checklist verification**:
  - [ ] All 135 game engine tests pass
  - [ ] MCTS 5×100 → top1_visit_share > 2/num_legal_actions (non-uniform signal)
  - [ ] Policy loss < ln(7) ≈ 1.95 within 10 iterations
  - [ ] Win rate vs random > 55% within 20 iterations
  - [ ] Full iteration < 5 minutes (32 threads)
  - [ ] 32-thread scaling > 80% efficiency
  - [ ] ONNX inference < 0.2ms (ort CPU, single sample)
  - [ ] ONNX ↔ tch output agreement within 1e-5
- Fix any gates that don't pass. This session is buffer/contingency for issues found during integration

---

## Section 7: Adaptive Training & Analysis (2 sessions, ~6h)

### Session 7.1 — Run initial training and diagnose with metrics

Execute a real training run and use the diagnostic metrics to identify issues and opportunities.

- Launch 20-iteration training with default config: 5×100 MCTS, ~118 games/iter (~80K examples), 512 batch, 10 epochs, n=5 only (simplify first run)
- **Read the diagnostics after every few iterations** — this is not a blind 100-iteration run:
  - After iter 1–3: is visit entropy dropping? Is policy loss moving? If not, something is fundamentally broken
  - After iter 5–10: is bid accuracy above random (>1/cards_dealt)? Is KL divergence dropping?
  - After iter 10–20: is win rate vs iter-1 checkpoint positive? Is bid success rate improving?
- **Expected learning milestones**:
  - Iter 1–5: loss drops sharply from random (~ln(7)) as network learns basic card ranking
  - Iter 5–15: bid accuracy improves (learns hand-strength-to-bid correlation)
  - Iter 15–20: win rate vs early checkpoints becomes clearly positive
- **Diagnostic-driven debugging**:
  - If visit entropy stays high: MCTS budget insufficient (but we start at 5×100, so this would be surprising). Check evaluator is connected correctly
  - If policy loss plateaus at random: check MCTS outputs — print 10 sample policies, verify they're non-uniform
  - If value loss doesn't decrease: check target distribution — print histogram of z-scored values. If bimodal extremes, consider alternative normalization
  - If per-layer gradient norms vary by >100×: attention or skip connection issue
  - If KL divergence stays high but loss drops: network is learning something, but not what MCTS teaches — possible head wiring issue

### Session 7.2 — Adaptive tuning based on metrics

Use iteration 1–20 diagnostics to tune hyperparameters, then run extended training.

- **Adaptive MCTS budget** using per-decision signal quality data (collected in Session 4.3):
  - After each iteration, analyze the logged `(num_legal_actions, sims_used, signal_ratio)` triples. Fit a lookup table or simple function that predicts the minimum sims needed to achieve `signal_ratio > 0.3` for a given branching factor
  - If most decisions achieve signal_ratio > 0.5 at 5×100: reduce budget for those decision classes to free compute for harder decisions. E.g., late-game tricks with 2-3 legal plays may only need 3×20
  - If decisions with >7 legal actions consistently show signal_ratio < 0.2: increase their budget to 5×150 or 7×100
  - The total compute per game should stay roughly constant — savings from easy decisions fund harder ones
  - **Hard floor**: never below 60 sims for any decision with >1 legal action. Skip MCTS entirely when only 1 legal action
- **Adaptive epochs**: if loss improvement per epoch drops below 1% after epoch 5, stop that iteration early. If still improving at epoch 10, allow up to 15
- **Adaptive games per iteration**: if policy-MCTS KL divergence drops below 0.1 quickly (network matches MCTS after few examples), reduce games/iter. If KL stays high, generate more
- **Learning rate**: if loss oscillates between iterations, reduce peak LR. If loss plateaus while KL is high, increase peak LR or adjust warmup
- **c_puct tuning**: if top1_visit_share > 0.8 consistently (MCTS too greedy), increase c_puct toward 2.5. If < 0.2 (too diffuse), decrease toward 1.0
- **Temperature schedule**: τ=1.0 for first 15 decisions per round (exploration), τ=0.1 for remaining (exploit). Adjust threshold based on average decisions per round at different card counts. Applied per-round within a game, not per-game
- **Player count**: introduce n=4,6,7 mix after n=5-only training shows clear learning (bid accuracy > 40%). Verify loss doesn't regress on n=5 games after mixing
- **Replay buffer sizing**: if loss drops fast then rises (overfitting to stale data), reduce capacity. If training is slow to improve, increase to allow more diverse examples
- **Extended run**: with tuned parameters, run 100+ iterations. Track all metrics. Win rate vs checkpoint-20 is the primary progress indicator
- Save final base model checkpoint for fine-tuning

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
- Save each fine-tuned model as `model_n{X}.pt` and corresponding ONNX export

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

**Total estimated sessions: 25 sessions (~75 hours) for core system (Sections 1–8), plus deferred frontend work.**
