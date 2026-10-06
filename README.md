# BlobMaster

An AI for the card game **Blob** (a trick-taking game with exact bidding, related to Oh Hell), for 3–8 players. It is built AlphaZero-style: tree search over sampled deals, guided by a transformer network trained by self-play. Written in Rust; training uses libtorch on the GPU and self-play uses ONNX Runtime on the CPU.

## Status

**Generation 1 is concluded; generation 2 is under way.** See **[gen-2.md](gen-2.md)**, the single source of truth.

- **Gen 0 (Python, 2025–2026-03):** correct engine; never learned. Archived in [legacy/](legacy/).
- **Gen 1 (Rust, 2026-03 → 2026-05):** fast engine and pipeline, 168-iteration final run. On 2026-10-02 it was measured against a fixed rule bot, and it loses (−10 points per game with full search). The causes are in the value targets and in the search's value backup, not in compute (gen-2.md §2). Its code is at git tags `gen-1-final` and `gen-1-compat`.
- **Gen 2:** per-round, per-seat values; a value network that sees the sampled deal; evaluation against a fixed yardstick; async actor–learner training. Done so far (2026-10-06): the yardsticks (`bench`, `play`, two rule bots), the encoder and determinization fixes, the removal of gen-1 support, the gen-2 engine (per-round utility, search that backs values up to every seat, policy and value evaluators, a guarded encoder layout), and the two networks with a supervised warm start from rule bot 2 (Phase 4). The warm start scores **+15.5 points per game** against the rule bot without search and **+20.0** with it, where gen 1 lost by 10–12; against four copies of rule bot 2, its teacher, search scores +4.7. Next: self-play RL (Phase 5).

## Vocabulary

| Term | Meaning |
|---|---|
| **Game** | A full session of rounds. 5 players starting at 7 cards: 17 rounds, ~380 decisions. |
| **Round** | One deal–bid–play–score cycle at a fixed number of cards. |
| **Trick** | Every player plays one card; the highest card wins (see rules). A round with C cards has C tricks. |
| **Bid** | How many tricks a player says they will win this round, exactly. |
| **Decision** | A bid or a card play. Per round: `num_players × (cards_dealt + 1)`. |

## Rules

- **Deck:** 52 cards. Suits ♠ ♥ ♣ ♦; ranks 2–A. Card index = `suit × 13 + rank`, with ♠=0 ♥=1 ♣=2 ♦=3 and 2=0 … A=12.
- **Rounds:** card count C goes down from the start value to 1, stays at 1 for one round per player, then goes back up. Total rounds = `2C + num_players − 2`; for example 5 players, C=7: `7 6 5 4 3 2 1 1 1 1 1 2 3 4 5 6 7`. Requires `num_players × C ≤ 52`.
- **Trump** rotates ♠ → ♥ → ♣ → ♦ → no trump, repeating every 5 rounds.
- **Bidding:** the player left of the dealer bids first; the dealer bids last. Each bid is 0..C. The dealer may not make the total of all bids equal C, so at least one player must miss.
- **Playing:** the player left of the dealer leads the first trick. Players must follow the led suit if they can; otherwise they may play any card. The highest trump wins; with no trump played, the highest card of the led suit wins. The winner leads the next trick.
- **Scoring:** `10 + bid` if a player wins exactly their bid, else 0. Scores add up over the game.

## Repository layout

| Path | What it is |
|---|---|
| `blob-engine/` | Game rules, entity encoder (policy and value modes), determinization, MCTS, ONNX inference, replay buffer, rule bots, strength benchmark. No libtorch dependency. |
| `blob-nn/` | The policy and value transformers (tch / libtorch), losses and optimizer, the learner. |
| `blob-train/` | `blobmaster-train` CLI: `pretrain` (supervised warm start from rule bot 2) and `export` (writes a model directory: policy and value nets as ONNX); `train` (self-play RL) comes in Phase 5. Run it through `scripts/blobmaster-train.sh`. |
| `blob-bin/` | `blobmaster` inference CLI: `bench` (strength vs the rule bots) and `play` (you vs bots in the terminal). |
| `scripts/` | ONNX export (Python) and plotting. |
| `logs/` | Measurement outputs (rule bot 2 rollout sweeps). |
| `legacy/` | Gen-0 Python reference code (read-only). |
| [gen-2.md](gen-2.md) | Findings, design and roadmap. |
| [AGENTS.md](AGENTS.md) | Working notes for Claude Code: runtime environment, commands, conventions. |

## Build and test

```bash
cargo build --release
cargo test -p blob-engine          # debug profile — some tests expect debug assertions
cargo test -p blob-bin
./target/release/blobmaster bench rulebot2 --deals 4000   # rule bot 2 vs 4 rule bots: +14.5 ± 0.3
./target/release/blobmaster play                          # you vs 4 rule bots
```

A model is a directory (`policy.onnx`, `value.onnx`, `meta.json`), which `bench` and `play --model <dir>` accept. Train the warm start with `scripts/blobmaster-train.sh pretrain --config blob-train/pretrain.sample.toml --output checkpoints/<run>` (~80 min on an RTX 4060; the model lands in `<run>/model`), or write a random-init one with `scripts/blobmaster-train.sh export --output <dir>`. Training and the export need the downloaded libtorch on the library path (the script sets it), and the export also the pinned Python venv; [AGENTS.md](AGENTS.md) has the setup.

## Hardware

- **Training:** Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8 GB, 128 GB DDR5.
- **Planned inference:** Windows laptop with an Intel iGPU, via ONNX Runtime.
