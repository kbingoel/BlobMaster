# BlobMaster

AlphaZero-style AI for the card game "Blob" (trick-taking with bidding, 3-8 players).

## Vocabulary

| Term | Meaning |
|---|---|
| **Game** | A complete session from first deal to final scores. For 5 players, 7 cards: 17 rounds, ~380 decisions. For 5 players, 8 cards: 19 rounds, ~470 decisions. |
| **Round** | One deal-bid-play-score cycle at a fixed card count (e.g., "the 7-card round"). Each round has one bidding phase followed by tricks. |
| **Trick** | One cycle where each player plays a card; the highest card (respecting suit/trump rules) wins. A round of N cards has N tricks. |
| **Bid** | A player's declaration of exactly how many tricks they expect to win this round. |
| **Play** | A single card played into a trick by one player. |
| **Decision point** | Any moment the network must choose: either a bid or a play. Decisions per round = `num_players × (cards_dealt + 1)`. |

## Game Rules Reference

### Card System
- **Deck**: Standard 52 cards (4 suits x 13 ranks)
- **Suits**: Spades(♠), Hearts(♥), Clubs(♣), Diamonds(♦)
- **Ranks**: 2,3,4,5,6,7,8,9,10,J,Q,K,A (values 2-14)
- **Card index**: `suit_index * 13 + rank_index` (0-51): ♠2=0, ♠A=12, ♥2=13, ..., ♦A=51

### Round Structure
- **Players**: 3-8
- **Cards per round (C)**: a game parameter, typically 7 or 8. Constrained by `num_players × C ≤ 52`. Descends from C to 1, stays at 1 for N rounds (N = num_players), ascends back to C. Total rounds = `2C + num_players - 2`. Example (5 players, C=7): `[7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7]` — 17 rounds. Example (5 players, C=8): `[8,7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7,8]` — 19 rounds
- **Trump rotation**: ♠→♥→♣→♦→None→♠→... (cycles every 5 rounds)

### Bidding Phase
- Players bid sequentially (left of dealer first)
- Bid = exact number of tricks you expect to win (0 to cards_dealt)
- **Dealer constraint**: Dealer cannot bid such that total_bids == cards_dealt (forces at least one player to miss)

### Playing Phase (Trick-Taking)
- Must follow led suit if able; if void, may play any card (including trump)
- Trick winner: highest trump if any trump played, otherwise highest card of led suit
- Trick winner leads the next trick

### Scoring
- `score = (tricks_won == bid) ? (10 + bid) : 0`
- All-or-nothing: exact bid required for points. Cumulative across all rounds.

## Status: Rust Rewrite — Architecture Finalized

The original Python/PyTorch implementation (Phases 1-4) has been **concluded and archived**. It produced a correct, well-tested game engine and training pipeline, but Python's per-operation overhead made MCTS too slow to generate useful learning signal. The model never learned. Full post-mortem in [conclusion.md](conclusion.md).

The Rust rewrite is in progress. The neural network architecture has been fully specified and supersedes the legacy design.

## What's Here

```
README.md              ← You are here
development-plan.md    ← Complete Rust development plan (architecture + implementation)
conclusion.md          ← Why Python failed, what to do differently
prepare-migration.md   ← Rust rewrite plan (game engine, MCTS, training pipeline)
legacy/                ← Archived Python reference code
```

## Neural Network: Structured Entity Transformer (~1.63M params)

The legacy BlobNet was a 6-layer Transformer over a flat 256-dim state vector (~4.9M params). Self-attention over a single token is mathematically degenerate — it reduces to a linear projection. The new architecture fixes this.

**Core idea**: represent the game state as a set of typed entity tokens, each encoding domain-specific features. The Transformer then learns meaningful attention patterns across entities.

### Token Types

| Token | Count | What it encodes |
|---|---|---|
| Hand card | 1–8 | Rank, suit, trump status, relative strength, suit counts |
| Played card | 0–48 | Rank, suit, player, trick number, lead/follow, win status |
| Player state | 3–8 | Bid, tricks won, bid status, suit voids, position |
| Context | 1 | Trump suit, cards dealt, trick progress, game phase |
| CLS | 1 | Learned aggregation token, feeds value and bid heads |

**Shared embeddings**: rank (13×16) and suit (4×8) are shared between hand and played card tokens; player (8×16) is shared between played cards and player states. The model learns card identity once.

**Chronological embeddings**: played card tokens receive an additional learned positional embedding (52×128) encoding their order in the play sequence.

### Transformer Config

`d_model=128`, `num_layers=8`, `num_heads=8`, `ffn_dim=512`, Pre-norm, GeLU, dropout=0.1

### Output Heads

- **Playing head**: each hand card token → shared MLP → scalar score → softmax over legal cards. Entity-native, 4K params vs 23K for a CLS→52 head.
- **Bidding head**: CLS → Linear(128→64) → Linear(64→14) → masked softmax. Separate from playing head to eliminate gradient interference at bid-value positions.
- **Value head**: CLS → Linear(128→64) → Linear(64→1) → tanh. Target is z-scored final score: `clip((my_score − mean) / std, −1, 1)`.

### Why This Architecture

Blob's key reasoning patterns are inherently sequential and relational:
- **Suit voids**: when a player fails to follow suit, they reveal a permanent constraint. Precomputed as binary flags on player tokens.
- **Card counting with attribution**: knowing *who* played *what* in *which trick* enables hand ranging. This is a natural attention query: a hand card attends to played cards of the same suit to determine its current dominance.
- **Bid-progress tracking**: `bid_status` (met / busted / live) drives opponent modeling; `tricks_needed` provides urgency. Both are precomputed features.
- **Multi-trick planning**: the full ordered play history gives the network context for continuing or abandoning multi-trick plans.

Precomputing key derived features (void flags, `is_highest_in_suit`, `cards_above_remaining`) reduces the burden on attention while leaving the network free to learn nuanced patterns through self-attention.

### Inference Performance

At a typical 35-token sequence: ~57M MACs per forward pass, ~0.15ms on CPU (ONNX Runtime). With 5 determinizations × 100 MCTS simulations per move, one 7-card round (~40 decisions) takes ~3s of neural network time. At 32 rayon threads on the 7950X: **~640 rounds/minute** for 7-card rounds, supporting ~3 minutes of self-play per iteration (~177 games, ~80K training examples).

## Porting Order

1. **Game Engine** — Blob rules, bitwise card representation (`u64` bitmasks, ~50ns state copy), trick history log (`TrickRecord` struct)
2. **Entity Encoder** — `BlobState` → variable-length token sequence
3. **Structured Entity Transformer** — as specified in [development-plan.md](development-plan.md) Section 3
4. **MCTS** — Arena-allocated tree search with belief tracking and determinization
5. **Training Pipeline** — Self-play (`rayon`), contiguous replay buffer, training loop
6. **Evaluation + CLI** — Model comparison, strength tracking, `clap` CLI
7. **ONNX + Fine-Tuning** — ONNX export for fast inference, player-count-specific fine-tuned models

**Key fixes over Python version:**
- MCTS starts at 5×100 sims/move — enough signal to actually learn
- Strong policy prior breaks the vicious cycle: weak prior → uniform MCTS targets → weaker prior
- Replay buffer stores raw `BlobState` (~410 bytes), re-encodes on the fly — not 500K Python dicts across a 650MB heap
- Diagnostic-driven adaptive training replaces fixed schedules — every iteration is monitored and tuned
- Full iteration in ~3–5 min (32 threads) vs ~5 min (Python, single-threaded bottleneck)

## Training Strategy

### Player Distribution & Fine-Tuning

Base model trains on mixed configurations:
- **Player count**: n=4 (10%), n=5 (60%), n=6 (25%), n=7 (5%)
- **Cards dealt**: C=7 (40%), C=8 (60%), constrained by `n × C ≤ 52` (forces C=7 for n=7)

The Structured Entity Transformer handles variable player counts and hand sizes natively via its token design — more players simply means more player state and played card tokens.

After base model training, **player-specific fine-tuned models** are produced for each count (n=3 through n=8), prioritizing n=5. Fine-tuning is cheap: the transformer layers already encode card fundamentals, suit reasoning, and void detection generically — fine-tuning adjusts output heads and final layers to count-specific dynamics.

### Training Optimization

Given limited hardware (single RTX 4060), every wasted evaluation is expensive. Training is driven by diagnostic metrics, not fixed schedules. Key metrics logged every iteration:

| Metric | What it tells you | Action if wrong |
|---|---|---|
| **MCTS visit entropy** | Is MCTS producing non-random targets? Low = good signal. | If high (≈ log(n_legal)), increase sim budget |
| **Top-1 visit share** | Fraction of visits on best action. Should be >0.3. | If ≈ 1/n_legal, sim budget insufficient |
| **Policy-MCTS KL divergence** | How much the network disagrees with MCTS. Should drop over time. | If stuck high, LR or architecture issue |
| **Policy loss** (bid/play separate) | Should drop below ln(avg_legal) ≈ 1.95 within ~10 iterations. | If flat, check MCTS signal quality |
| **Value prediction variance** | Is the value head producing diverse outputs? | If all ≈ 0, check z-scoring targets |
| **Bid accuracy (top-1)** | % network top bid matches MCTS top bid. Interpretable. | Plateau = capacity or signal issue |
| **Win rate vs checkpoint N-20** | Is training still improving? More telling than vs random. | If flat, training has stalled |
| **Bid success rate** | % of rounds where model hits bid exactly. Domain-specific strength. | Direct measure of play quality |
| **Per-layer gradient norms** | Are all transformer layers learning? | Vanishing = architecture problem |
| **Loss improvement per eval** | Δloss / num_nn_evaluations. Training efficiency. | If flat, wasting compute |

These metrics enable **adaptive training**: simulation budget adjusts based on visit entropy, epochs per iteration adjust based on loss improvement rate, and buffer size adjusts based on effective sample rate.

## Hardware

- **Training**: Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8GB, 128GB DDR5
- **Future inference**: Windows laptop, Intel i5 iGPU, ONNX Runtime
