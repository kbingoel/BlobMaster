# Realistic Self-Play Acceleration Plan

**Project**: BlobMaster
**Goal**: Accelerate self-play from 6-12 games/min to 60-120 games/min (10-20x speedup)
**Timeline**: 3-4 weeks (NOT 3-4 days)
**Status**: Planning phase
**Created**: 2025-10-27

---

## Abstract

The original JAX transition plan was **fundamentally flawed** due to a critical oversight: you cannot achieve 20-50x speedup while keeping the game simulator in Python. JAX JIT compilation requires pure array operations - every Python callback forces host synchronization and destroys performance.

**This plan presents a realistic 3-phase approach:**

1. **Phase 0 (2-3 days)**: Validate Gumbel sampling benefits in current Python MCTS - de-risk algorithm change before infrastructure overhaul
2. **Phase 1 (1-2 weeks)**: Port core game logic to pure JAX arrays - prerequisite for any speedup
3. **Phase 2 (1-2 weeks)**: Integrate mctx library with vectorized determinization and optimize performance

**Conservative target: 10-20x speedup** (60-120 games/min) instead of unrealistic 20-50x. This still reduces training time from months to **~30 days**.

**Key insight**: Incremental validation at each phase gates risk. If Gumbel doesn't help in Python, or JAX port proves too difficult, we can stop before wasting weeks on a doomed approach.

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Critical Findings](#critical-findings)
3. [Recommended Approach](#recommended-approach)
4. [Phase 0: Python Gumbel Validation](#phase-0-python-gumbel-validation)
5. [Phase 1: JAX Environment Port](#phase-1-jax-environment-port)
6. [Phase 2: mctx Integration & Optimization](#phase-2-mctx-integration--optimization)
7. [Risk Mitigation](#risk-mitigation)
8. [Success Metrics](#success-metrics)
9. [Revised Timeline & Estimates](#revised-timeline--estimates)
10. [Dependencies & Setup](#dependencies--setup)

---

## Current State Assessment

### Benchmark Results
From `Benchmarks/results/20251027_161731_benchmark_report.md`:

| Configuration | Games/Min | Bottleneck |
|--------------|-----------|------------|
| 3 det × 30 sims | 6-12 | CPU-bound Python loops |
| 5 det × 50 sims | 3-6 | Worse scaling |
| Worker scaling | Plateaus at 8 workers | Diminishing returns |

**Root cause**: Python MCTS is inherently sequential, cannot leverage GPU parallelism.

### Project Status
- **Phase 4 complete**: 426+ tests passing, full training pipeline ready
- **Training infrastructure**: Solid (PyTorch, replay buffer, ELO tracking)
- **Game engine**: 135 tests, 97% coverage - BUT all Python classes
- **Network**: Transformer architecture, ~4.9M parameters, fast enough (not bottleneck)

### What Works vs What Doesn't
✅ **Keep**: PyTorch training loop, replay buffer, evaluation arena
❌ **Replace**: Python MCTS search, sequential determinization
⚠️ **Port Required**: Game state representation, legal move generation, state transitions

---

## Critical Findings

### Fatal Flaw in Original Plan
**Assumption**: "Use real game rules" while achieving 20-50x speedup
**Reality**: Original plan kept `BlobGame` Python class ([line 168](GUMBEL-TRANSITION-PLAN.md:168)) while calling it from JAX RecurrentFn
**Problem**: JAX JIT cannot cross Python boundaries - every `game.play_card()` call forces expensive host sync

### Why JAX Environment Port is Non-Negotiable

```
Current Python MCTS:
for sim in range(90):
    state.play_card(card)  ← Python method call
    value = network(state)  ← PyTorch on CPU
```

```
JAX MCTS (what we need):
states = jax.vmap(step_fn)(states, actions)  ← Pure JAX arrays
values = jax.vmap(network)(states)           ← JIT-compiled GPU batch
```

**The game state MUST be JAX arrays** for vectorization to work.

### Other Critical Gaps
1. **Windows GPU support**: JAX CUDA wheels are experimental on Windows - need WSL2 path
2. **Compilation overhead**: First JIT call is 10-100x slower - benchmarks must warm cache
3. **Shape constraints**: All JAX arrays need static shapes - variable player counts (3-8) require padding/masking strategy
4. **Determinization sampling**: Current belief tracker uses Python rejection loops - need pure JAX alternative

---

## Recommended Approach

### Conservative 3-Phase Plan with Risk Gates

Each phase has **clear success criteria** that gate the next phase. If any phase fails, we stop and reassess.

```
Phase 0: Python Gumbel (2-3 days)
  ├─ Validate algorithm works in familiar environment
  ├─ Success gate: 30-50% simulation reduction
  └─ Risk: Low (pure Python, easy to debug)

Phase 1: JAX Environment (1-2 weeks)
  ├─ Port game to pure JAX arrays
  ├─ Success gate: Can play full games, passes tests
  └─ Risk: High (major refactor, shape constraints)

Phase 2: mctx Integration (1-2 weeks)
  ├─ Combine JAX game + mctx library + vectorization
  ├─ Success gate: 10-20x speedup achieved
  └─ Risk: Medium (debugging JIT issues, perf tuning)
```

### Why This Order?

**Phase 0 first**: Validates Gumbel MuZero helps for Blob before infrastructure overhaul. Original plan assumed this without proof.

**Phase 1 prerequisite**: Cannot integrate mctx until game state is pure JAX. Original plan skipped this entirely.

**Phase 2 payoff**: Once Phases 0-1 complete, mctx integration is straightforward following DeepMind examples.

### Target Speedup: 10-20x (Conservative)

| Component | Current | After JAX | Speedup |
|-----------|---------|-----------|---------|
| Network inference | PyTorch CPU | JAX GPU JIT | 3-5x |
| MCTS tree search | Python loops | Vectorized JAX | 5-10x |
| Determinization | Sequential | vmap parallel | 2-3x |
| Gumbel pruning | N/A | Fewer sims (50 vs 90) | 1.5-2x |
| **Realistic total** | 6-12 games/min | **60-120 games/min** | **10-20x** |

**Training time**: ~30 days instead of months (still massive win)

---

## Phase 0: Python Gumbel Validation

**Duration**: 2-3 days
**Goal**: Prove Gumbel sampling reduces simulations needed for same quality play
**Risk**: Low (no infrastructure changes)

### Implementation Steps

1. **Install dependencies**
   - NumPy Gumbel sampling utilities
   - Keep existing PyTorch MCTS infrastructure

2. **Implement Gumbel action selection**
   - Add Gumbel noise to policy logits: `logits + Gumbel(0, 1)`
   - Use sequential halving to prune low-probability actions
   - Keep top-k actions (k=16) for tree expansion

3. **Compare simulation efficiency**
   - Baseline: Current MCTS with 90 simulations
   - Gumbel: Test with 50, 60, 70 simulations
   - Metric: Win rate vs baseline in 100-game tournaments

4. **Measure quality/speed tradeoff**
   - Plot: Simulations vs ELO strength
   - Find minimum simulations for 95% of baseline strength
   - Target: 50-60 simulations (30-40% reduction)

### Success Criteria
- [ ] Gumbel MCTS with 50-60 sims matches 90-sim baseline within 5% win rate
- [ ] Implementation passes all existing MCTS tests
- [ ] No regression in training pipeline integration

### Exit Decision
- **If successful**: Proceed to Phase 1 (JAX environment port)
- **If fails**: Investigate why (hyperparameter tuning?) or abort Gumbel approach, focus on pure JAX AlphaZero

---

## Phase 1: JAX Environment Port

**Duration**: 1-2 weeks
**Goal**: Pure JAX game state, legal moves, and transitions (no Python callbacks)
**Risk**: High (major refactor, most complex phase)

### Core Challenge: State Representation

Current Python state (object-oriented):
```
BlobGame:
  - players: List[Player]  ← Python objects
  - current_trick: Trick   ← Python object
  - history: List[Move]    ← Variable length
```

Required JAX state (pure arrays):
```
game_state: Dict[str, Array]
  - hands: (8 players, 52 cards) bool array
  - bids: (8,) int array (-1 = not bid yet)
  - tricks_won: (8,) int array
  - current_trick: (8, 4) card indices (padded)
  - phase: int (0=bidding, 1=playing, 2=done)
```

### Implementation Steps

1. **Define JAX game state schema**
   - All arrays static shape (max 8 players, pad smaller games)
   - Card representation: one-hot or indices (52 cards)
   - Mask arrays for variable player counts: `(8,) bool` where `[T,T,T,T,F,F,F,F]` = 4 players
   - Document shape invariants and padding rules

2. **Port legal move generation**
   - `get_legal_bids_jax(state) -> (14,) bool mask`
   - `get_legal_cards_jax(state, player_idx) -> (52,) bool mask`
   - Must handle "must follow suit" logic without Python conditionals
   - Use JAX `where` and masking instead of if/else

3. **Port state transition function**
   - `step_jax(state, player_idx, action) -> new_state`
   - Pure function: no side effects, no mutation
   - Handle bidding phase: update `bids` array
   - Handle playing phase: update `current_trick`, check trick completion, update `tricks_won`
   - Terminal state detection: return done flag

4. **Port scoring logic**
   - `compute_rewards_jax(state) -> (8,) float rewards`
   - All-or-nothing scoring: `(tricks_won == bid) ? (10 + bid) : 0`

5. **State encoder/decoder**
   - `encode_state_jax(state, player_idx) -> (256,) embedding` for network
   - Handle imperfect information: mask opponent cards
   - Compatible with existing StateEncoder output format

6. **Test suite migration**
   - Port critical tests from `ml/game/test_blob.py` to JAX versions
   - Verify JAX game matches Python game on 1000 random games
   - Check edge cases: dealer bidding constraint, trump suit logic, trick winner determination

### Success Criteria
- [ ] JAX game plays full 4-player, 5-card rounds correctly
- [ ] Legal move generation matches Python version (100% agreement on 1000 random states)
- [ ] State transitions deterministic and reproducible
- [ ] All game logic tests pass (port 135 Python tests to JAX equivalents)
- [ ] Performance: 1000 game simulations in <1 second (pure JAX, no MCTS yet)

### Exit Decision
- **If successful**: Proceed to Phase 2 (mctx integration)
- **If too difficult**: Fallback to hybrid approach (vectorize only network inference, keep Python MCTS) - still get 3-5x speedup

---

## Phase 2: mctx Integration & Optimization

**Duration**: 1-2 weeks
**Goal**: Combine JAX game + mctx library + vectorization for 10-20x speedup
**Risk**: Medium (JIT debugging, performance tuning)

### Implementation Steps

1. **Network conversion**
   - Port PyTorch BlobNet to Flax (JAX neural network library)
   - Weight converter: `pytorch_state_dict -> jax_params`
   - Validation: Compare outputs on 100 random states (tolerance <1e-5)
   - JIT-compile network inference: `@jax.jit def forward(params, state)`

2. **Implement mctx RecurrentFn**
   - Signature: `recurrent_fn(params, rng, action, embedding) -> RecurrentFnOutput`
   - Embedding = JAX game state arrays (from Phase 1)
   - Action application: call `step_jax(state, player_idx, action)`
   - Reward/discount: 0.0/1.0 for non-terminal, score/0.0 for terminal
   - Network evaluation: call JIT-compiled network on new state
   - Legal action masking: apply mask before returning logits

3. **Implement mctx RootFn**
   - Evaluate initial state with network
   - Apply legal action mask to policy logits
   - Return `RootFnOutput(prior_logits, value, embedding=state)`

4. **Vectorized determinization sampling**
   - Current: Sample opponent hands one at a time (sequential)
   - Target: `sample_determinizations_jax(state, key, num_samples) -> (B, state_shape)`
   - Use JAX PRNG splitting for reproducibility
   - Constraint satisfaction: opponent hands must be valid (cards sum to deck)
   - Avoid rejection sampling: use constrained random permutation
   - Batch MCTS search: `vmap(gumbel_muzero_policy)(determinizations)` - runs B searches in parallel

5. **Gumbel MuZero policy wrapper**
   - Call `mctx.gumbel_muzero_policy(params, rng, root, recurrent_fn, num_simulations=50)`
   - Hyperparameters:
     - `max_num_considered_actions`: Start with 16, ablate 8/32
     - `qtransform`: Use `mctx.qtransform_completed_by_mix_value` (DeepMind default)
     - `gumbel_scale`: 1.0 (standard)
   - Aggregate action weights across determinizations: average visit counts

6. **Self-play integration**
   - Modify `SelfPlayWorker` to use JAX MCTS
   - Weight synchronization: Convert PyTorch checkpoint -> JAX params each iteration
   - Maintain same training example format for replay buffer compatibility

7. **Performance optimization**
   - JIT compilation: Pre-compile all hot paths, cache compiled functions
   - Static shapes: Ensure no dynamic shape operations (batch size, player count must be static args)
   - Memory management: Profile GPU memory, tune batch sizes for RTX 4060 8GB
   - XLA flags: Enable `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda` for optimal compilation

### Success Criteria
- [ ] Network parity: PyTorch vs JAX outputs match within 1e-5 on 100 random states
- [ ] Legal actions only: 100% of MCTS outputs are valid moves
- [ ] Speedup achieved: >=10x games/min improvement (60+ games/min)
- [ ] Training integration: Generate 1000 games, feed to replay buffer, train 1 iteration successfully
- [ ] Correctness: JAX MCTS plays competently (beats random baseline)

### Exit Decision
- **If 10-20x achieved**: Deploy for full training run
- **If only 5-10x**: Still worthwhile - reassess whether to optimize further or proceed
- **If <5x**: Debug performance (likely JIT issues or Python callbacks remaining)

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **JAX game port too complex** | High | High | Start with simplified 4-player, 5-card game; defer variable player counts |
| **Shape constraint issues** | Medium | Medium | Use generous padding, static max sizes; accept memory overhead |
| **Windows JAX GPU fails** | Medium | High | **Set up WSL2 Ubuntu as primary environment** + CPU fallback config |
| **Performance doesn't scale** | Low | High | Profile with JAX profiler, check XLA HLO graphs, optimize hotspots |
| **mctx API complexity** | Medium | Low | Use `muzero_policy` (simpler) instead of `gumbel_muzero_policy` as fallback |
| **Weight conversion bugs** | Low | High | Extensive numerical testing, visual network output inspection |

### Process Risks

| Risk | Mitigation |
|------|------------|
| **Timeline slips** | Each phase has clear exit criteria - stop if blocked >3 days |
| **Scope creep** | Phase 1 focuses on 4-player only; defer 3-8 player support to Phase 3 |
| **Testing burden** | Reuse existing test cases, compare JAX vs Python outputs |
| **GPU memory overflow** | Start with batch_size=1, scale up; monitor with `nvidia-smi` |

### Fallback Plan B

If Phase 1 (JAX game port) proves too difficult:
1. **Hybrid approach**: Keep Python game, vectorize only network inference
2. **Expected speedup**: 3-5x (still worthwhile)
3. **Implementation**: Batch network calls across MCTS leaf nodes
4. **Timeline**: 3-5 days instead of 3-4 weeks

---

## Success Metrics

### Phase 0 Metrics
- **Simulation reduction**: 30-50% fewer sims for same quality (90 -> 50-60)
- **Win rate parity**: Gumbel MCTS >= 95% baseline win rate
- **Test compatibility**: All 148 MCTS tests pass

### Phase 1 Metrics
- **Correctness**: JAX game matches Python game on 1000 random playouts
- **Performance**: 1000 game steps/second (pure JAX, no MCTS)
- **Test coverage**: 100+ JAX game tests pass

### Phase 2 Metrics
- **Speedup**: 10-20x games/min (60-120 games/min target)
- **Network parity**: Max absolute diff <1e-5 between PyTorch/JAX
- **GPU utilization**: >70% during self-play (check `nvidia-smi`)
- **Memory footprint**: <6GB VRAM (stay within RTX 4060 limits)
- **Training integration**: 1 full iteration completes successfully

### Training Impact Metrics (Long-term)
- **Training time**: 5M games in ~30 days (vs months currently)
- **Model quality**: No regression in ELO progression
- **Stability**: No crashes or NaN losses over 50 iterations

---

## Revised Timeline & Estimates

### Realistic 3-4 Week Plan

| Phase | Duration | Workdays | Key Deliverable |
|-------|----------|----------|-----------------|
| **Phase 0: Python Gumbel** | 2-3 days | 2-3 | Gumbel MCTS in Python, validated |
| **Phase 1: JAX Game** | 1-2 weeks | 7-10 | Pure JAX game state + transitions |
| **Phase 2: mctx Integration** | 1-2 weeks | 7-10 | 10-20x speedup achieved |
| **Total** | **3-4 weeks** | **16-23 days** | Production-ready JAX self-play |

### Detailed Breakdown

**Week 1**:
- Days 1-2: Phase 0 (Python Gumbel validation)
- Days 3-5: Phase 1 start (JAX state schema, legal moves)

**Week 2**:
- Days 6-10: Phase 1 continue (state transitions, testing)

**Week 3**:
- Days 11-12: Phase 1 completion (test suite, validation)
- Days 13-15: Phase 2 start (network conversion, mctx integration)

**Week 4**:
- Days 16-18: Phase 2 continue (vectorized determinization, optimization)
- Days 19-20: Phase 2 completion (benchmarking, validation)
- Buffer: 3 days for unexpected issues

### Training Timeline Projection

**Assumptions**:
- 10,000 games/iteration
- 500 iterations
- 5,000,000 total games

| Scenario | Games/Min | Total Hours | Total Days |
|----------|-----------|-------------|------------|
| **Current (Python)** | 6-12 | 7,000-14,000 | 290-580 |
| **Conservative (10x)** | 60-120 | 700-1,400 | **29-58** |
| **Optimistic (20x)** | 120-240 | 350-700 | **15-29** |

**Realistic target: ~30 days** of continuous training (vs months currently)

---

## Dependencies & Setup

### Software Requirements

**Critical: Use WSL2 Ubuntu on Windows 11**

JAX GPU support on native Windows is experimental and unreliable. WSL2 provides Linux environment with native GPU passthrough.

```bash
# WSL2 Ubuntu setup (Windows host)
wsl --install Ubuntu-22.04
wsl --set-default Ubuntu-22.04

# Inside WSL2
sudo apt update
sudo apt install python3.10 python3-pip nvidia-cuda-toolkit
```

### JAX Installation (Pinned Versions)

```bash
# CUDA 12.x for RTX 4060
pip install "jax[cuda12]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# mctx library (verify compatible version)
pip install "dm-mctx==0.0.5"

# Flax for neural networks
pip install "flax==0.8.1"

# Utilities
pip install "chex==0.1.85"  # Shape checking
pip install "optax==0.1.8"  # Optimizers (for future use)
```

**Version pinning rationale**: JAX/mctx APIs change frequently; pinning prevents breakage.

### CPU Fallback Config

For testing or if GPU setup fails:

```bash
# Force CPU-only JAX
export JAX_PLATFORM_NAME=cpu
pip install jax[cpu]==0.4.23
```

**Performance**: 2-3x slower than GPU, but still faster than Python MCTS due to JIT.

### Verification

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")  # Should show GPU

# Test GPU
x = jax.numpy.ones(1000)
y = jax.numpy.dot(x, x)
print(f"GPU test: {y}")  # Should complete without error
```

### Development Environment

Recommended setup:
- **WSL2**: Primary development environment
- **VS Code Remote**: Edit files on Windows, run in WSL2
- **PyTorch**: Keep on Windows for training loop (can access GPU via native CUDA)
- **JAX**: Only in WSL2 for self-play workers

This hybrid setup leverages best of both: PyTorch stability on Windows, JAX performance on Linux.

---

## Next Steps

### Immediate Actions (This Week)

1. **Review and approve this plan** - ensure stakeholder alignment on realistic timeline
2. **Set up WSL2 Ubuntu** - install CUDA toolkit, verify GPU access
3. **Install JAX dependencies** - pinned versions, test GPU functionality
4. **Create feature branch** - `git checkout -b jax-acceleration` for all Phase 0-2 work

### Phase 0 Kickoff (Days 1-2)

1. Read Gumbel MuZero paper (DeepMind 2021) - understand algorithm
2. Implement Gumbel action selection in `ml/mcts/gumbel_selection.py`
3. Add `--use_gumbel` flag to existing MCTS classes
4. Run 100-game tournament: Gumbel vs baseline
5. **Gate decision**: If Gumbel helps, proceed to Phase 1; else investigate or pivot

### Monitoring & Communication

- **Daily**: Commit progress to feature branch with clear messages
- **Weekly**: Write brief status update (what worked, what blocked, next steps)
- **Milestones**: Tag commits at phase boundaries for rollback safety

---

## References

### Essential Reading

1. **Gumbel MuZero** (Danihelka et al., 2021)
   https://openreview.net/forum?id=bERaNdoegnO
   Key insight: Sequential halving + Gumbel noise = 50% fewer simulations

2. **mctx Documentation**
   https://github.com/deepmind/mctx
   API reference, RecurrentFn examples, Gumbel policy usage

3. **JAX Sharp Bits**
   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
   Pure functions, static shapes, JIT compilation rules

### Supporting Materials

- JAX GPU setup guide: https://github.com/google/jax#installation
- Flax tutorial: https://flax.readthedocs.io/en/latest/getting_started.html
- WSL2 CUDA setup: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

---

**END OF PLAN**

This plan is designed to be **maximally realistic** based on current evidence. Timeline assumes single developer working full-time; adjust proportionally for part-time work or team collaboration.

**Critical success factor**: Ruthless adherence to phase gates. If any phase fails acceptance criteria, STOP and reassess rather than continuing with broken foundation.
