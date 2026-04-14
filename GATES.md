# BlobMaster Verification Gates — Session 6.3

Consolidated checklist tracking the Session 6.3 completion gates from
`development-plan.md`. Update the checkboxes as each gate is verified.

## How to verify

| Gate | Command / source |
| --- | --- |
| Ported game-engine tests (143 from `test_blob.py`, adjusted) | `cargo test -p blob-engine` (`test_blob_ports.rs`) |
| `BlobState` copy ~100 ns | `cargo bench -p blob-engine --bench core -- blobstate_copy` |
| Legal move generation ~5 ns | `cargo bench -p blob-engine --bench core -- legal_plays_mid_trick` |
| Entity encoding < 1 µs | `cargo bench -p blob-engine --bench core -- encode_mid_trick_5p7c` |
| ONNX inference (batch=1) < 0.2 ms | `BLOB_ONNX_MODEL=… cargo bench -p blob-engine --bench onnx_mcts -- onnx_inference_batch1` |
| MCTS 100 sims < 20 ms (with ONNX eval) | `BLOB_ONNX_MODEL=… cargo bench -p blob-engine --bench onnx_mcts -- mcts_1det_100sims` |
| Full move (5 det × 100 sims) < 100 ms | `BLOB_ONNX_MODEL=… cargo bench -p blob-engine --bench onnx_mcts -- mcts_full_move_5x100` |
| Single full 5P7C game self-play < 30 s neural time | self-play engine timings (`blob-nn::engine::self_play_iteration`) |
| Full iteration wall-clock < 5 min (32 threads) | wall-clock measurement around `blob_nn::training_loop::run_iteration` |
| Memory: no leak over 100 iterations | external `/usr/bin/time -v` or `rusage` around `blobmaster-train train` |
| 50 training iterations with no NaN/Inf; value ∈ [−1, 1] | `cargo test -p blob-nn --release -- --ignored numerical_stability` |
| ONNX ↔ tch output agreement within 1e-5 | `BLOB_ONNX_MODEL=… BLOB_TCH_CHECKPOINT=… cargo test -p blob-nn onnx_tch_value_parity` and `python scripts/export_onnx.py --weights … --out … --check` |
| MCTS 5×100 → `top1_visit_share > 2 / num_legal_actions` | `MctsResult::top1_visit_share` in self-play logs |
| Policy loss < `ln(7) ≈ 1.95` within 10 iterations | `training.jsonl` / `metrics.jsonl` |
| Eval `win_rate_lower95 > 0.5` vs heuristic baseline within 20 iterations | `blob_nn::eval` comparison runs |
| 32-thread scaling > 80% efficiency | thread-scaling bench in `blob-nn::engine` |

## Gate checklist

- [ ] All 143 ported game-engine tests pass (round-structure corrections included).
- [ ] MCTS 5×100 → `top1_visit_share > 2 / num_legal_actions` on the reference state.
- [ ] Policy loss < `ln(7) ≈ 1.95` within 10 iterations.
- [ ] Eval `win_rate_lower95 > 0.5` vs heuristic baseline within 20 iterations.
- [ ] Full iteration < 5 minutes (32 threads).
- [ ] 32-thread scaling > 80% efficiency.
- [ ] ONNX inference < 0.2 ms (ort CPU, single sample).
- [ ] ONNX ↔ tch output agreement within 1e-5 (`cargo test onnx_tch_value_parity` + `export_onnx.py --check`).
- [ ] 50 training iterations with no NaN/Inf; value head in [−1, 1] (`numerical_stability` test, ignored).
- [ ] Peak RSS during 32-thread self-play is stable over 100 iterations (no monotonic leak).

## Notes

- `AGENTS.md` previously carried a "<60 seconds" full-iteration target; Session
  6.3 reconciles that with the realistic "<5 minutes" budget. Both documents
  now agree.
- Benches under `blob-engine/benches/` gate on `BLOB_ONNX_MODEL` when they need
  a trained model, and print a skip line when the env var is unset.
- The Rust-side parity test additionally gates on `BLOB_TCH_CHECKPOINT`, which
  must point at a directory containing a `model.ot` saved by
  `blob_nn::train::save_checkpoint`.
