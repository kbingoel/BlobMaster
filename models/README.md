# Models for play

Models committed on purpose, for `blobmaster play`, the GUI and playtests on any machine. Every other weight file stays out of git (`.gitignore`, gen-2.md §9). Add a model here only when it replaces the one people play against, and keep each under ~10 MB.

A model is a directory: `policy.onnx` (P), `value.onnx` (V) and `meta.json` (gen-2.md §5.3). Both networks carry the encoder layout they were trained on (`blob_layout_id`), and `OnnxPolicy` / `OnnxValue` refuse any other layout. A model therefore works only with code at the same `encoder::LAYOUT_ID`.

## `gen2-2026-10-09/`: the strongest model so far

- **Source:** the final model of the night run `checkpoints/pi-2026-10-08b` (learner step 2931; policy iteration by rollouts, gen-2.md §6 Phase 5, day 4). It is a byte-for-byte copy. `meta.json`'s `checkpoint` names the training checkpoint, which exists only on the training machine.
- **Layout:** `layout-4`.
- **Strength, P alone, no search** (5 players, 7 cards, duplicate deals, 95% CI over deals):

  | Against four copies of | Deals | Points per game | Wins (fair share 0.20) |
  |---|---|---|---|
  | the rule bot | 128 | **+24.8 ± 1.7** | 0.54 |
  | rule bot 2 | 512 | **+11.6 ± 0.7** | 0.36 |
  | rule bot 2r (lookahead, 128 samples) | 256 | **+5.8 ± 0.9** | 0.25 |
  | itself, with P + V search for the focal seat | 64 | −0.0 ± 0.6 (search adds nothing) | 0.19 |

- **Play it as P alone** (`--bot network`): search makes it no stronger, and P alone is fast. Rounds of P alone ran at ~78k per hour per CPU thread (gen-2.md §6 Phase 5, day 3), about a millisecond per move. `blobmaster play`'s default with a model is search with c_puct 0.2, which fell behind P alone once P got strong (gen-2.md §6 Phase 5).

```bash
cargo build --release -p blob-bin
./target/release/blobmaster play --model models/gen2-2026-10-09 --bot network
./target/release/blobmaster bench models/gen2-2026-10-09 --mode network --opponent rulebot2 --deals 512
```

It has not played people yet: that is the first human playtest (gen-2.md §6 Phase 5).
