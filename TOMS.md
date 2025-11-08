# TOMS: Training Optimization & Monitoring System

**BlobMaster AI Training Pipeline Enhancement Plan**

**Document Version:** 1.0
**Created:** 2025-11-06
**Status:** Implementation Ready

---

## Executive Summary

This document outlines the comprehensive enhancement plan for the BlobMaster training pipeline, implementing graceful shutdown, dynamic scheduling, aggressive MCTS curriculum, optimized game configurations, and a real-time web monitoring dashboard. The goal is to achieve a usable AI model as quickly as possible while maintaining the option for extended high-quality training.

**Key Objectives:**
- ✅ Enable safe pause/resume at any point (even mid-iteration)
- ✅ Reduce training time via aggressive MCTS curriculum
- ✅ Optimize for 4-6 players, 6-7 cards (50%+ focus on 5p/7c)
- ✅ Provide real-time monitoring and control via web dashboard
- ✅ Smart checkpoint management to balance safety and disk space

**Estimated Time Investment:** 41 hours across 16 sessions (3-4h each)

**Expected Impact:**
- **Training Time:** Reduced from 136 days to ~60-80 days for initial strong model
- **Safety:** Zero progress loss on interruption (game-level checkpoints)
- **Convergence:** Faster reaching competitive ELO via optimized game mix
- **Control:** Full visibility and control via web dashboard

---

## Table of Contents

1. [Phase Breakdown](#phase-breakdown)
2. [Session-by-Session Work Plan](#session-by-session-work-plan)
3. [Technical Specifications](#technical-specifications)
4. [Implementation Details](#implementation-details)
5. [Testing & Validation](#testing--validation)
6. [Timeline & Dependencies](#timeline--dependencies)

---

## Phase Breakdown

### Phase 1: Graceful Shutdown & Advanced Checkpointing
**Priority:** CRITICAL
**Duration:** 4 sessions (~11h)
**Goal:** Enable safe pause/resume at any point, with smart disk space management

**Key Features:**
- Signal handlers (SIGTERM, SIGINT) for clean shutdown
- Intra-iteration checkpointing (save after each game)
- Tiered checkpoint retention policy
- Full state preservation (model, optimizer, replay buffer, game progress)

**Deliverables:**
- Enhanced `ml/train.py` with signal handling
- Intra-iteration checkpoint system in `ml/training/pipeline.py`
- Checkpoint retention manager in `ml/training/checkpoint_manager.py`
- Resume validation tests

---

### Phase 2: Learning Rate Warmup & Cosine Annealing
**Priority:** HIGH (Quick Win)
**Duration:** 2 sessions (~4h)
**Goal:** Stabilize early training and enable smooth convergence

**Key Features:**
- Learning rate warmup (0 → 0.001 over first 10 iterations)
- Cosine annealing (0.001 → 0.00001 over 500 iterations)
- Smooth decay instead of abrupt StepLR drops
- Scheduler state preserved in checkpoints

**Deliverables:**
- New `WarmupCosineScheduler` class in `ml/training/schedulers.py`
- Updated `trainer.py` to use new scheduler
- Learning curve validation

---

### Phase 3: Dynamic MCTS Scheduling
**Priority:** HIGH (Biggest Impact)
**Duration:** 3 sessions (~8h)
**Goal:** Aggressive curriculum to reach strong play faster

**Key Features:**
- Iteration-based configuration scheduler
- Aggressive MCTS ramp: 1→2→3→4→5 determinizations
- Simulation count scaling: 15→25→35→45→50 sims/det
- JSON schedule files for easy experimentation

**Deliverables:**
- `ml/scheduling/config_scheduler.py` (scheduler infrastructure)
- `ml/scheduling/schedules/aggressive_mcts.json` (schedule definition)
- Integration with training pipeline
- Performance benchmarking

**MCTS Schedule:**
```
Iterations 1-50:    1 det × 15 sims = ~150 games/min  (ultra-fast exploration)
Iterations 51-150:  2 det × 25 sims = ~90 games/min   (light refinement)
Iterations 151-300: 3 det × 35 sims = ~50 games/min   (medium quality)
Iterations 301-450: 4 det × 45 sims = ~30 games/min   (high quality)
Iterations 451-500: 5 det × 50 sims = ~20 games/min   (maximum quality)
```

**Time Savings:**
- First 50 iterations: ~3.3 days (vs 13.6 days at baseline)
- First 150 iterations: ~13.5 days (vs 40.8 days at baseline)
- Total 500 iterations: ~63 days (vs 136 days at baseline)
- **Reduction: ~73 days saved (54% faster)**

---

### Phase 4: Game Configuration Optimization
**Priority:** MEDIUM
**Duration:** 3 sessions (~7h)
**Goal:** Optimize for 4-6 players, 6-7 cards, with 50%+ at 5p/7c

**Key Features:**
- Weighted random sampling for game configurations
- At least 50% of games at 5 players, 7 cards
- Dynamic games per iteration (5K → 10K → 15K)
- Configurable distribution via JSON

**Deliverables:**
- `GameDistributionConfig` class in `ml/config.py`
- Weighted sampling in `ml/training/selfplay.py`
- Distribution validation and logging
- Dynamic games/iteration scheduler

**Game Distribution:**
```
5 players, 7 cards:  50%  (primary target)
4 players, 6 cards:  15%
4 players, 7 cards:  10%
6 players, 6 cards:  10%
6 players, 7 cards:  10%
Other configs:       5%   (3-8 players, 5-9 cards)
```

**Games Per Iteration Schedule:**
```
Iterations 1-100:   5,000 games  (fast iteration cycles)
Iterations 101-300: 10,000 games (baseline quality)
Iterations 301-500: 15,000 games (high-quality fine-tuning)
```

---

### Phase 5: Web Training Dashboard
**Priority:** MEDIUM (Quality of Life)
**Duration:** 4 sessions (~11h)
**Goal:** Real-time monitoring and control via web interface

**Key Features:**
- Live training metrics (iteration, games played, ELO, speed)
- Start/pause/stop controls
- Current configuration display
- Loss curves and performance graphs
- WebSocket updates (real-time, no polling)

**Deliverables:**
- `ml/monitoring/training_server.py` (FastAPI backend)
- `ml/monitoring/dashboard.html` (web UI)
- `ml/monitoring/dashboard.js` (real-time updates)
- `ml/monitoring/api_routes.py` (REST API)

**Dashboard Features:**
- **Status Panel:** Current state (running/paused), iteration number, games completed
- **Metrics Panel:** Training speed (games/min), current ELO, loss values
- **Configuration Panel:** Current MCTS settings, learning rate, games/iteration
- **Controls:** Start, Pause, Stop, Resume buttons
- **Progress Panel:** Iteration progress bar, estimated time remaining
- **History Panel:** ELO progression chart, loss curves

**API Endpoints:**
```
GET  /api/status          - Get current training status
GET  /api/metrics         - Get latest metrics
GET  /api/config          - Get current configuration
POST /api/control/start   - Start training
POST /api/control/pause   - Pause training
POST /api/control/stop    - Stop training
GET  /api/history/elo     - Get ELO history
GET  /api/history/loss    - Get loss history
WS   /ws/updates          - WebSocket for real-time updates
```

---

## Session-by-Session Work Plan

### **Phase 1: Graceful Shutdown & Advanced Checkpointing**

#### Session 1.1: Signal Handlers + Checkpoint on Interrupt (2-3h)
**Goal:** Ensure training saves checkpoint when interrupted

**Tasks:**
1. Add signal handler registration in `ml/train.py`
   - Handle SIGTERM, SIGINT, SIGHUP
   - Set global flag `shutdown_requested = True`
   - Trigger checkpoint save before exit
2. Implement explicit checkpoint save in interrupt handler
   - Call `pipeline.save_checkpoint(reason="interrupt")`
   - Log checkpoint location
   - Clean up worker pool
3. Update `KeyboardInterrupt` handler to save checkpoint
4. Add `--checkpoint-on-interrupt` flag (default: True)
5. Test: Start training, hit Ctrl+C, verify checkpoint saved, resume

**Files Modified:**
- `ml/train.py` (signal handling, interrupt handler)
- `ml/training/pipeline.py` (add checkpoint reason logging)

**Success Criteria:**
- Ctrl+C saves checkpoint within 5 seconds
- Resume restores exact iteration state
- No worker pool hangs or zombie processes

---

#### Session 1.2: Intra-Iteration Checkpointing (3-4h)
**Goal:** Save progress after each game completes

**Tasks:**
1. Create `IntraIterationCheckpoint` class in `ml/training/checkpoint_manager.py`
   - Store: iteration number, games_completed, replay_buffer_state
   - Lightweight format (don't re-save full model every game)
   - Quick serialize/deserialize (<100ms per save)
2. Add checkpoint triggers in self-play loop
   - After each game completes: `checkpoint_manager.save_game_checkpoint()`
   - Check shutdown flag after each game: `if shutdown_requested: break`
3. Implement resume from intra-iteration checkpoint
   - Load last game number completed
   - Skip already-played games on resume
   - Continue from games_completed + 1
4. Add progress tracking: `logger.info(f"Game {i+1}/{games_per_iter} completed")`
5. Test: Interrupt during iteration, resume, verify continues from correct game

**Files Modified:**
- `ml/training/checkpoint_manager.py` (new file)
- `ml/training/pipeline.py` (integrate intra-iteration checkpoints)
- `ml/training/selfplay.py` (add checkpoint triggers)

**Success Criteria:**
- Can interrupt at any game within iteration
- Resume skips completed games
- <1% overhead from checkpoint saves
- No replay buffer corruption

---

#### Session 1.3: Smart Checkpoint Retention Policy (2-3h)
**Goal:** Manage disk space with tiered deletion

**Tasks:**
1. Implement `CheckpointRetentionPolicy` class
   - Rule 1: Keep all checkpoints from last 10 iterations
   - Rule 2: Keep every 2nd checkpoint from 11-30 iterations ago
   - Rule 3: Keep every 5th checkpoint from 31-50 iterations ago
   - Rule 4: Keep every 10th checkpoint for 51+ iterations ago
   - Always keep "best_model.pth"
2. Add automatic cleanup after each checkpoint save
   - Scan checkpoint directory
   - Apply retention rules
   - Delete old checkpoints
   - Log deletions: `logger.info(f"Deleted checkpoint {filename} (retention policy)")`
3. Add `--retain-all-checkpoints` flag to disable cleanup
4. Add disk space monitoring (warn if <10GB free)
5. Test: Run 50 iterations, verify correct checkpoints retained

**Retention Example (50 iterations completed):**
```
Keep: iterations 41-50 (last 10)          → 10 checkpoints
Keep: iterations 30, 32, 34, 36, 38, 40  → 6 checkpoints
Keep: iterations 25, 30, 35, 40, 45      → (some overlap, ~3 new)
Keep: iterations 10, 20, 30, 40, 50      → (some overlap, ~2 new)
Total: ~21 checkpoints (vs 50 without policy)
```

**Files Modified:**
- `ml/training/checkpoint_manager.py` (retention policy)
- `ml/config.py` (add retention config)
- `ml/train.py` (add --retain-all-checkpoints flag)

**Success Criteria:**
- Disk usage ~40% of naive approach
- All important checkpoints retained
- No accidental deletion of recent/best checkpoints

---

#### Session 1.4: Testing Interrupt/Resume Cycles (1-2h)
**Goal:** Validate all checkpoint scenarios

**Test Cases:**
1. **Interrupt at iteration boundary** (between iterations)
   - Start training, wait for iteration N to complete
   - Hit Ctrl+C immediately after
   - Resume: Should start iteration N+1
2. **Interrupt mid-iteration** (during self-play)
   - Interrupt after 3000/10000 games
   - Resume: Should continue from game 3001
3. **Interrupt during training** (NN update phase)
   - Interrupt during epoch 5/10
   - Resume: Should complete remaining epochs, then next iteration
4. **Interrupt during evaluation**
   - Interrupt during model tournament
   - Resume: Should re-run evaluation (don't save partial eval)
5. **Kill process** (simulate crash)
   - `kill -9 <pid>` during training
   - Resume: Should restore from last successful checkpoint
6. **Multiple interrupts**
   - Start → interrupt after 2h → resume → interrupt after 1h → resume
   - Verify no data corruption or state drift

**Files Modified:**
- `ml/tests/test_checkpoint_resume.py` (new test suite)

**Success Criteria:**
- All test cases pass
- No progress loss >1 game
- Replay buffer integrity maintained
- ELO tracking consistent across resumes

---

### **Phase 2: Learning Rate Warmup & Cosine Annealing**

#### Session 2.1: Warmup + Cosine Annealing Scheduler (2-3h)
**Goal:** Replace StepLR with better schedule

**Tasks:**
1. Create `WarmupCosineScheduler` class in `ml/training/schedulers.py`
   ```python
   class WarmupCosineScheduler:
       def __init__(self, optimizer, warmup_iterations, total_iterations,
                    lr_min, lr_max):
           # Warmup: linear 0 → lr_max over warmup_iterations
           # Cosine: lr_max → lr_min over (total_iterations - warmup_iterations)
   ```
2. Implement `get_lr()` method
   - Iteration 0-10: Linear warmup `lr = lr_max * (iter / warmup_iter)`
   - Iteration 10-500: Cosine annealing `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * (iter - warmup) / (total - warmup)))`
3. Update `ml/training/trainer.py` to use new scheduler
   - Replace StepLR initialization
   - Pass total_iterations from config
4. Add scheduler visualization script
   - Plot LR curve over 500 iterations
   - Save to `docs/training/lr_schedule.png`
5. Test: Run 20 iterations, verify LR follows expected curve

**Files Modified:**
- `ml/training/schedulers.py` (new file)
- `ml/training/trainer.py` (replace StepLR)
- `ml/config.py` (add warmup_iterations, lr_min params)
- `scripts/visualize_lr_schedule.py` (new visualization script)

**Success Criteria:**
- Smooth LR curve (no discontinuities)
- Warmup prevents early instability
- Scheduler state saves/loads correctly
- Final LR reaches lr_min (0.00001)

**Expected Learning Rate Curve:**
```
Iteration 0:    0.000000 (start of warmup)
Iteration 5:    0.000500 (mid-warmup)
Iteration 10:   0.001000 (end warmup, start cosine)
Iteration 100:  0.000950 (slight decay)
Iteration 250:  0.000500 (halfway)
Iteration 400:  0.000100 (rapid decay)
Iteration 500:  0.000010 (minimum)
```

---

#### Session 2.2: Validate Learning Curves (1h)
**Goal:** Verify improved training stability

**Tasks:**
1. Run baseline training (5 iterations, old StepLR)
   - Log loss curves
   - Check for spikes or instability
2. Run warmup training (5 iterations, new scheduler)
   - Compare loss curves
   - Verify smoother early iterations
3. Compare first 10 iterations: old vs new
   - Policy loss variance
   - Value loss convergence speed
4. Document findings in `docs/training/warmup_validation.md`
5. Update `ml/train.py` to use warmup scheduler by default

**Success Criteria:**
- Lower loss variance in first 10 iterations
- No training divergence (NaN losses)
- Comparable or better final loss values

---

### **Phase 3: Dynamic MCTS Scheduling**

#### Session 3.1: Configuration Scheduler Infrastructure (3-4h)
**Goal:** Build system for iteration-based config updates

**Tasks:**
1. Create `ConfigScheduler` class in `ml/scheduling/config_scheduler.py`
   ```python
   class ConfigScheduler:
       def __init__(self, schedule_file: str):
           # Load JSON schedule
           # Validate iteration ranges

       def get_config_for_iteration(self, iteration: int) -> dict:
           # Return config overrides for this iteration
   ```
2. Define JSON schedule format
   ```json
   {
     "schedule": [
       {
         "iterations": "1-50",
         "overrides": {
           "num_determinizations": 1,
           "simulations_per_determinization": 15
         }
       },
       {
         "iterations": "51-150",
         "overrides": {
           "num_determinizations": 2,
           "simulations_per_determinization": 25
         }
       }
     ]
   }
   ```
3. Implement schedule validation
   - Check for overlapping iteration ranges
   - Verify all parameters are valid config keys
   - Ensure complete coverage (no gaps)
4. Add schedule preview command
   - `python ml/train.py --preview-schedule schedules/aggressive_mcts.json`
   - Print table showing config at each iteration boundary
5. Test: Load schedule, query various iterations, verify correct overrides

**Files Modified:**
- `ml/scheduling/config_scheduler.py` (new file)
- `ml/scheduling/__init__.py` (new file)
- `ml/train.py` (add --schedule flag, preview command)
- `ml/config.py` (add `apply_overrides()` method)

**Success Criteria:**
- Schedule loads without errors
- Iteration queries return correct overrides
- Invalid schedules raise clear errors
- Preview shows human-readable table

---

#### Session 3.2: Aggressive MCTS Schedule Implementation (2-3h)
**Goal:** Define and integrate aggressive curriculum

**Tasks:**
1. Create `ml/scheduling/schedules/aggressive_mcts.json`
   ```json
   {
     "name": "Aggressive MCTS Curriculum",
     "description": "Fast ramp from ultra-light to heavy MCTS",
     "schedule": [
       {
         "iterations": "1-50",
         "overrides": {
           "num_determinizations": 1,
           "simulations_per_determinization": 15,
           "games_per_iteration": 5000
         }
       },
       {
         "iterations": "51-150",
         "overrides": {
           "num_determinizations": 2,
           "simulations_per_determinization": 25,
           "games_per_iteration": 8000
         }
       },
       {
         "iterations": "151-300",
         "overrides": {
           "num_determinizations": 3,
           "simulations_per_determinization": 35,
           "games_per_iteration": 10000
         }
       },
       {
         "iterations": "301-450",
         "overrides": {
           "num_determinizations": 4,
           "simulations_per_determinization": 45,
           "games_per_iteration": 12000
         }
       },
       {
         "iterations": "451-500",
         "overrides": {
           "num_determinizations": 5,
           "simulations_per_determinization": 50,
           "games_per_iteration": 15000
         }
       }
     ]
   }
   ```
2. Integrate scheduler into training pipeline
   - Load schedule at training start
   - Query config before each iteration
   - Apply overrides to self-play engine
   - Log config changes: `logger.info(f"Iteration {i}: MCTS updated to {det} det × {sims} sims")`
3. Update checkpoint format to include current schedule
   - Save schedule file path in checkpoint
   - Verify resume uses same schedule
4. Add schedule validation in resume
   - Warn if resuming with different schedule
   - Allow override with `--force-schedule`
5. Test: Run 10 iterations, verify config changes at iteration 6 (crossing 1-50 → 51-150 boundary)

**Files Modified:**
- `ml/scheduling/schedules/aggressive_mcts.json` (new schedule)
- `ml/training/pipeline.py` (integrate scheduler)
- `ml/train.py` (schedule loading and validation)
- `ml/training/trainer.py` (checkpoint schedule metadata)

**Success Criteria:**
- Config changes take effect at correct iterations
- No performance degradation from scheduler overhead
- Schedule state preserved across resumes
- Clear logging of config transitions

---

#### Session 3.3: Testing and Performance Benchmarking (1-2h)
**Goal:** Validate schedule works and measure speedup

**Tasks:**
1. Benchmark each MCTS configuration
   - Run 100 games at each setting (1×15, 2×25, 3×35, 4×45, 5×50)
   - Measure games/min
   - Compare to baseline (3×30 = 36.7 games/min)
2. Calculate weighted average speed
   - Iterations 1-50: 50 iter × 5000 games × (1/150 games/min) = 27.8h
   - Iterations 51-150: 100 iter × 8000 games × (1/90 games/min) = 148h
   - Iterations 151-300: 150 iter × 10000 games × (1/50 games/min) = 500h
   - Iterations 301-450: 150 iter × 12000 games × (1/30 games/min) = 1000h
   - Iterations 451-500: 50 iter × 15000 games × (1/20 games/min) = 625h
   - **Total: ~2301h (~96 days)** vs baseline 136 days @ 10K games/iter
3. Run end-to-end test
   - Train for 20 iterations with schedule
   - Verify config changes happen automatically
   - Check no errors or warnings
4. Document performance in `docs/training/mcts_schedule_benchmarks.md`
5. Update README.md with new expected training time

**Success Criteria:**
- All MCTS configs run without errors
- Performance matches expected games/min (±10%)
- Total training time reduced by >30%
- No quality degradation in early iterations

**Expected Performance Table:**
```
MCTS Config  | Games/Min | Iterations | Games/Iter | Total Time
-------------|-----------|------------|------------|------------
1 det × 15   |   ~150    |    1-50    |    5,000   |   ~28h
2 det × 25   |    ~90    |   51-150   |    8,000   |  ~148h
3 det × 35   |    ~50    |  151-300   |   10,000   |  ~500h
4 det × 45   |    ~30    |  301-450   |   12,000   | ~1000h
5 det × 50   |    ~20    |  451-500   |   15,000   |  ~625h
-------------+---------------------------------------------------------------
TOTAL:                                                 ~2301h (~96 days)
BASELINE: 3 det × 30, 10K games/iter for 500 iters  → ~3264h (~136 days)
SPEEDUP: ~40 days saved (29% faster)
```

---

### **Phase 4: Game Configuration Optimization**

#### Session 4.1: Weighted Sampling for Game Configurations (2-3h)
**Goal:** Focus training on 4-6 players, 6-7 cards (50%+ at 5p/7c)

**Tasks:**
1. Create `GameDistributionConfig` class in `ml/config.py`
   ```python
   @dataclass
   class GameDistributionConfig:
       distribution: Dict[Tuple[int, int], float]  # (players, cards) → weight

       def validate(self):
           # Sum of weights should equal 1.0
           # All configurations should be valid (3-8 players, 5-9 cards)
   ```
2. Define weighted distribution
   ```python
   OPTIMIZED_DISTRIBUTION = {
       (5, 7): 0.50,  # Primary target
       (4, 6): 0.15,
       (4, 7): 0.10,
       (6, 6): 0.10,
       (6, 7): 0.10,
       (3, 5): 0.01,  # Edge cases
       (3, 6): 0.01,
       (7, 7): 0.01,
       (7, 8): 0.01,
       (8, 8): 0.01,
   }
   ```
3. Implement weighted random sampling in `ml/training/selfplay.py`
   - Replace `random.randint(3, 8)` with weighted choice
   - Sample (num_players, num_cards) from distribution
   - Log distribution every 1000 games
4. Add distribution validation
   - Track actual game counts per configuration
   - Log distribution report after each iteration
   - Verify 50%+ at 5p/7c
5. Test: Generate 10,000 games, verify distribution matches target (±2%)

**Files Modified:**
- `ml/config.py` (add GameDistributionConfig)
- `ml/training/selfplay.py` (weighted sampling)
- `ml/training/pipeline.py` (distribution logging)

**Success Criteria:**
- Actual distribution matches target (±2%)
- No bias toward certain configurations (beyond specified weights)
- Performance comparable to uniform sampling
- Clear logging of game mix

**Expected Game Mix (10,000 games):**
```
5 players, 7 cards:  ~5000 games (50.0%)
4 players, 6 cards:  ~1500 games (15.0%)
4 players, 7 cards:  ~1000 games (10.0%)
6 players, 6 cards:  ~1000 games (10.0%)
6 players, 7 cards:  ~1000 games (10.0%)
Other configs:       ~500 games  (5.0%)
```

---

#### Session 4.2: Dynamic Games Per Iteration Schedule (1-2h)
**Goal:** Ramp from 5K → 15K games as training progresses

**Tasks:**
1. Add `games_per_iteration` to schedule overrides (already supported in Session 3.2)
2. Update `aggressive_mcts.json` to include games/iteration ramp
   - Iterations 1-50: 5,000 games
   - Iterations 51-150: 8,000 games
   - Iterations 151-300: 10,000 games
   - Iterations 301-450: 12,000 games
   - Iterations 451-500: 15,000 games
3. Update training pipeline to respect dynamic games_per_iteration
   - Read from config each iteration
   - Adjust self-play loop accordingly
4. Log games/iteration changes
5. Test: Run 20 iterations, verify game counts change at boundaries

**Files Modified:**
- `ml/scheduling/schedules/aggressive_mcts.json` (update with games/iter)
- `ml/training/pipeline.py` (dynamic games/iteration support)

**Success Criteria:**
- Games/iteration changes at correct boundaries
- Total game count per iteration matches config
- No off-by-one errors in game counting

---

#### Session 4.3: Validation of Game Distribution (1-2h)
**Goal:** Ensure convergence isn't harmed by weighted sampling

**Tasks:**
1. Run A/B test: uniform vs weighted distribution (10 iterations each)
   - Track loss curves
   - Track ELO progression
   - Compare training stability
2. Validate 50%+ constraint
   - Run 50 iterations
   - Count total games at 5p/7c
   - Verify ≥50%
3. Create distribution report script
   - `python scripts/analyze_game_distribution.py`
   - Read replay buffer or game logs
   - Generate histogram of game configurations
   - Save to `results/game_distribution_report.json`
4. Document findings in `docs/training/game_distribution_validation.md`

**Success Criteria:**
- Weighted sampling shows comparable or better ELO progression
- Distribution constraint (50%+ at 5p/7c) consistently met
- No evidence of overfitting to single configuration

---

### **Phase 5: Web Training Dashboard**

#### Session 5.1: Backend API (3-4h)
**Goal:** Expose training state via REST API

**Tasks:**
1. Create `ml/monitoring/training_server.py` using FastAPI
   ```python
   from fastapi import FastAPI, WebSocket
   from fastapi.staticfiles import StaticFiles
   import uvicorn

   app = FastAPI()

   @app.get("/api/status")
   async def get_status():
       # Return: state (running/paused/stopped), iteration, games_played, etc.

   @app.get("/api/metrics")
   async def get_metrics():
       # Return: loss, ELO, games/min, GPU memory, etc.

   @app.get("/api/config")
   async def get_config():
       # Return: current MCTS settings, LR, games/iteration

   @app.post("/api/control/start")
   async def start_training():
       # Set flag to start/resume training

   @app.post("/api/control/pause")
   async def pause_training():
       # Set flag to pause after current game
   ```
2. Create shared state manager `ml/monitoring/training_state.py`
   ```python
   class TrainingState:
       # Thread-safe state shared between training loop and API
       state: str = "stopped"  # running/paused/stopped
       iteration: int = 0
       games_played: int = 0
       current_config: dict = {}
       latest_metrics: dict = {}
   ```
3. Integrate state manager with training loop
   - Update state after each game
   - Check control flags (pause_requested, stop_requested)
   - Emit state changes for WebSocket broadcast
4. Add CORS middleware for local development
5. Test: Start server, query endpoints, verify data

**Files Modified:**
- `ml/monitoring/training_server.py` (new file)
- `ml/monitoring/training_state.py` (new file)
- `ml/monitoring/api_routes.py` (new file)
- `ml/train.py` (integrate state manager)
- `requirements.txt` (add fastapi, uvicorn)

**Success Criteria:**
- API server starts on localhost:8080
- All endpoints return valid JSON
- State updates reflect training progress
- Control endpoints set flags correctly

---

#### Session 5.2: Web UI (3-4h)
**Goal:** Build dashboard HTML/CSS/JS

**Tasks:**
1. Create `ml/monitoring/dashboard.html`
   - Status panel (state, iteration, games played)
   - Metrics panel (ELO, loss, speed)
   - Configuration panel (MCTS settings, LR)
   - Control panel (start/pause/stop buttons)
   - Progress panel (iteration progress bar, ETA)
   - History panel (ELO chart, loss chart)
2. Style with clean, modern CSS
   - Dark theme (easier on eyes for long monitoring)
   - Responsive grid layout
   - Color-coded status indicators (green=running, yellow=paused, red=stopped)
3. Use Chart.js for graphs
   - ELO progression over iterations
   - Loss curves (policy loss, value loss)
   - Games/min over time
4. Add auto-refresh (every 5 seconds via polling initially)
5. Test: Open in browser, verify displays correct data

**Files Modified:**
- `ml/monitoring/dashboard.html` (new file)
- `ml/monitoring/dashboard.css` (new file)
- `ml/monitoring/static/` (create directory for assets)

**Success Criteria:**
- Dashboard loads in browser (http://localhost:8080)
- All panels display data correctly
- Graphs render without errors
- UI is clean and readable

**Dashboard Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  BlobMaster Training Dashboard                          │
├─────────────┬──────────────────┬────────────────────────┤
│   Status    │     Metrics      │      Configuration     │
│             │                  │                        │
│  Running    │  ELO: 1420      │  MCTS: 3 det × 35 sims │
│  Iter: 175  │  Speed: 48 g/m  │  LR: 0.000542          │
│  Games: 7234│  Loss: 0.234    │  Games/Iter: 10000     │
├─────────────┴──────────────────┴────────────────────────┤
│  Controls                                               │
│  [Start] [Pause] [Stop]                                 │
├─────────────────────────────────────────────────────────┤
│  Progress: Iteration 175/500                            │
│  [████████████████░░░░░░░░░░] 35%                       │
│  ETA: 42 days, 6 hours                                  │
├─────────────────────────────────────────────────────────┤
│  ELO History                                            │
│  [Line chart showing ELO progression]                   │
├─────────────────────────────────────────────────────────┤
│  Loss History                                           │
│  [Line chart showing policy/value loss]                 │
└─────────────────────────────────────────────────────────┘
```

---

#### Session 5.3: Real-Time Updates via WebSocket (2-3h)
**Goal:** Replace polling with WebSocket for live updates

**Tasks:**
1. Add WebSocket endpoint in `training_server.py`
   ```python
   @app.websocket("/ws/updates")
   async def websocket_endpoint(websocket: WebSocket):
       await websocket.accept()
       while True:
           # Broadcast state changes
           await websocket.send_json({
               "type": "status_update",
               "data": training_state.to_dict()
           })
           await asyncio.sleep(1)  # Update every second
   ```
2. Create `ml/monitoring/dashboard.js`
   - Connect to WebSocket on page load
   - Listen for updates
   - Update DOM elements in real-time
   - Reconnect on disconnect (with backoff)
3. Implement event-driven updates
   - Training loop emits events (game_completed, iteration_completed, etc.)
   - State manager broadcasts to all WebSocket clients
4. Add connection status indicator
   - Green dot: connected
   - Red dot: disconnected
5. Test: Open dashboard, start training, verify updates appear instantly

**Files Modified:**
- `ml/monitoring/training_server.py` (WebSocket endpoint)
- `ml/monitoring/dashboard.js` (new file)
- `ml/monitoring/dashboard.html` (include dashboard.js)
- `ml/monitoring/training_state.py` (event emission)

**Success Criteria:**
- WebSocket connects successfully
- Updates appear within 1 second of training events
- No memory leaks or connection drops
- Multiple clients can connect simultaneously

---

#### Session 5.4: Testing, Polish, Documentation (1-2h)
**Goal:** Ensure dashboard is production-ready

**Tasks:**
1. End-to-end testing
   - Start training via dashboard
   - Pause mid-iteration
   - Resume training
   - Stop training
   - Verify all actions work correctly
2. Error handling
   - Handle API errors gracefully (show error message)
   - Handle WebSocket disconnects (retry logic)
   - Handle invalid control actions (e.g., pause when already paused)
3. Performance testing
   - Monitor CPU/memory usage of API server
   - Verify no slowdown to training loop
   - Test with multiple browser tabs open
4. Polish UI
   - Add tooltips to explain metrics
   - Add timestamps to last update
   - Add keyboard shortcuts (Space = pause/resume)
5. Write documentation
   - Create `docs/training/DASHBOARD_GUIDE.md`
   - Include screenshots
   - Document API endpoints
   - Add troubleshooting section
6. Update README.md with dashboard instructions

**Files Modified:**
- `ml/monitoring/dashboard.html` (polish, tooltips)
- `ml/monitoring/dashboard.css` (final styling)
- `docs/training/DASHBOARD_GUIDE.md` (new file)
- `README.md` (add dashboard section)

**Success Criteria:**
- All control actions work reliably
- Dashboard handles errors without crashing
- API server adds <1% overhead to training
- Documentation is clear and complete

---

## Technical Specifications

### Checkpoint Format

**Full Checkpoint (per iteration):**
```python
{
    'iteration': int,
    'network_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'scaler_state_dict': OrderedDict,  # if using mixed precision
    'training_stats': {
        'total_steps': int,
        'total_epochs': int,
        'best_elo': float,
    },
    'elo_history': List[float],
    'loss_history': List[Tuple[float, float]],  # (policy_loss, value_loss)
    'config': dict,  # Full config at this iteration
    'schedule_file': str,  # Path to schedule JSON (if using)
    'timestamp': str,  # ISO 8601 format
}
```

**Intra-Iteration Checkpoint (per game):**
```python
{
    'iteration': int,
    'games_completed': int,
    'games_total': int,
    'replay_buffer_size': int,
    'timestamp': str,
}
```

**Replay Buffer Checkpoint:**
```python
{
    'buffer': List[Tuple],  # (state, policy, value) tuples
    'position': int,
    'capacity': int,
    'size': int,
    'metadata': dict,
}
```

---

### Checkpoint Retention Rules

**Rule Set:**
1. **Recent checkpoints (last 10 iterations):** Keep ALL
   - Reason: Maximum safety for recent work
   - Example: Iterations 491-500 → keep all 10
2. **Medium-term (11-30 iterations ago):** Keep EVERY 2nd
   - Reason: Balance safety and disk space
   - Example: Iterations 461-490 → keep 461, 463, 465, ..., 489 (15 checkpoints)
3. **Long-term (31-50 iterations ago):** Keep EVERY 5th
   - Reason: Coarser granularity for older checkpoints
   - Example: Iterations 441-460 → keep 445, 450, 455, 460 (4 checkpoints)
4. **Ancient (51+ iterations ago):** Keep EVERY 10th
   - Reason: Minimal storage for historical reference
   - Example: Iterations 1-440 → keep 10, 20, 30, ..., 440 (44 checkpoints)
5. **Special checkpoints:** ALWAYS keep
   - `best_model.pth` (highest ELO model)
   - `checkpoint_latest.pth` (symlink to most recent)

**Total Checkpoints (at iteration 500):**
- Recent: 10 checkpoints
- Medium: 15 checkpoints
- Long: 4 checkpoints
- Ancient: 44 checkpoints
- **Total: 73 checkpoints** (vs 500 without retention)
- **Disk savings: ~85%**

---

### MCTS Schedule Details

**Aggressive Curriculum (500 iterations):**

| Phase | Iterations | Determinizations | Sims/Det | Games/Min | Games/Iter | Phase Duration |
|-------|------------|------------------|----------|-----------|------------|----------------|
| 1     | 1-50       | 1                | 15       | ~150      | 5,000      | ~28h           |
| 2     | 51-150     | 2                | 25       | ~90       | 8,000      | ~148h          |
| 3     | 151-300    | 3                | 35       | ~50       | 10,000     | ~500h          |
| 4     | 301-450    | 4                | 45       | ~30       | 12,000     | ~1000h         |
| 5     | 451-500    | 5                | 50       | ~20       | 15,000     | ~625h          |
| **TOTAL** | **500** |                  |          |           |            | **~2301h (~96 days)** |

**Rationale:**
- **Phase 1 (Ultra-light):** Model is weak, benefits from rapid iteration, not deep search
- **Phase 2 (Light):** Model improving, needs slightly better search quality
- **Phase 3 (Medium):** Model competent, medium search balances speed and quality
- **Phase 4 (High):** Model strong, deeper search for subtle improvements
- **Phase 5 (Maximum):** Fine-tuning phase, maximum search quality

**Comparison to Baseline:**
- Baseline: 3 det × 30 sims, 10K games/iter for 500 iterations = ~3264h (~136 days)
- Aggressive: Variable MCTS as above = ~2301h (~96 days)
- **Time saved: ~963h (~40 days, 29% reduction)**
- **Quality trade-off:** Early iterations have lighter search, but model is weak anyway (minimal quality loss)

---

### Learning Rate Schedule

**Warmup Phase (Iterations 0-10):**
```
LR(iter) = lr_max * (iter / warmup_iterations)
```
- Iteration 0: LR = 0.000000
- Iteration 5: LR = 0.000500
- Iteration 10: LR = 0.001000

**Cosine Annealing Phase (Iterations 10-500):**
```
LR(iter) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * (iter - warmup) / (total - warmup)))
```
- Iteration 10: LR = 0.001000 (max)
- Iteration 100: LR = 0.000955
- Iteration 250: LR = 0.000505
- Iteration 400: LR = 0.000095
- Iteration 500: LR = 0.000010 (min)

**Parameters:**
- `lr_min`: 0.00001 (1e-5)
- `lr_max`: 0.001 (1e-3)
- `warmup_iterations`: 10
- `total_iterations`: 500

---

### Game Distribution

**Target Weights:**
```python
GAME_DISTRIBUTION = {
    # Primary focus (50%)
    (5, 7): 0.50,

    # Secondary focus (40%)
    (4, 6): 0.15,
    (4, 7): 0.10,
    (6, 6): 0.10,
    (6, 7): 0.05,

    # Edge cases (10%)
    (3, 5): 0.02,
    (3, 6): 0.02,
    (3, 7): 0.01,
    (7, 7): 0.02,
    (7, 8): 0.01,
    (8, 8): 0.01,
    (8, 9): 0.01,
}
```

**Validation:**
- Sum of weights = 1.0
- All configs valid: 3 ≤ players ≤ 8, 5 ≤ cards ≤ 9
- At least 50% at (5 players, 7 cards)

**Implementation:**
```python
def sample_game_config(distribution: dict) -> Tuple[int, int]:
    configs = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(configs, weights=weights, k=1)[0]
```

---

### Dashboard API Specification

**Base URL:** `http://localhost:8080`

#### GET `/api/status`
Returns current training state.

**Response:**
```json
{
  "state": "running",  // "running" | "paused" | "stopped"
  "iteration": 175,
  "games_played": 7234,
  "games_total": 10000,
  "progress": 0.7234,
  "timestamp": "2025-11-06T14:32:10Z"
}
```

#### GET `/api/metrics`
Returns latest training metrics.

**Response:**
```json
{
  "elo": 1420.5,
  "policy_loss": 0.234,
  "value_loss": 0.156,
  "games_per_min": 48.3,
  "gpu_memory_used": 5120,  // MB
  "gpu_memory_total": 8192,  // MB
  "timestamp": "2025-11-06T14:32:10Z"
}
```

#### GET `/api/config`
Returns current training configuration.

**Response:**
```json
{
  "num_determinizations": 3,
  "simulations_per_determinization": 35,
  "learning_rate": 0.000542,
  "games_per_iteration": 10000,
  "num_workers": 32,
  "batch_size": 512,
  "current_schedule_phase": "3 (Medium MCTS)"
}
```

#### POST `/api/control/start`
Starts or resumes training.

**Request:** None

**Response:**
```json
{
  "success": true,
  "message": "Training started",
  "state": "running"
}
```

#### POST `/api/control/pause`
Pauses training after current game completes.

**Request:** None

**Response:**
```json
{
  "success": true,
  "message": "Training will pause after current game",
  "state": "pausing"
}
```

#### POST `/api/control/stop`
Stops training and saves checkpoint.

**Request:** None

**Response:**
```json
{
  "success": true,
  "message": "Training stopped, checkpoint saved",
  "state": "stopped"
}
```

#### GET `/api/history/elo`
Returns ELO history over iterations.

**Response:**
```json
{
  "iterations": [0, 5, 10, 15, ..., 175],
  "elo": [800, 850, 920, 1050, ..., 1420],
  "timestamp": "2025-11-06T14:32:10Z"
}
```

#### GET `/api/history/loss`
Returns loss history over training steps.

**Response:**
```json
{
  "steps": [0, 100, 200, ..., 175000],
  "policy_loss": [2.5, 2.1, 1.8, ..., 0.234],
  "value_loss": [1.2, 0.9, 0.7, ..., 0.156],
  "timestamp": "2025-11-06T14:32:10Z"
}
```

#### WebSocket `/ws/updates`
Real-time updates via WebSocket.

**Message Format:**
```json
{
  "type": "status_update" | "metrics_update" | "config_update",
  "data": {
    // Same format as corresponding GET endpoint
  },
  "timestamp": "2025-11-06T14:32:10Z"
}
```

**Event Types:**
- `status_update`: Training state changed (running → paused, etc.)
- `metrics_update`: New metrics available (after each game)
- `config_update`: Configuration changed (schedule transition)
- `error`: Error occurred during training

---

## Implementation Details

### File Structure

```
BlobMaster/
├── TOMS.md                                     # This document
├── ml/
│   ├── train.py                                # ✏️ Modified: signal handlers, state manager
│   ├── config.py                               # ✏️ Modified: game distribution, retention policy
│   ├── training/
│   │   ├── trainer.py                          # ✏️ Modified: warmup scheduler, checkpoint retention
│   │   ├── pipeline.py                         # ✏️ Modified: config scheduler integration
│   │   ├── selfplay.py                         # ✏️ Modified: weighted game sampling
│   │   ├── schedulers.py                       # ✨ New: WarmupCosineScheduler
│   │   └── checkpoint_manager.py               # ✨ New: CheckpointRetentionPolicy, IntraIterationCheckpoint
│   ├── scheduling/
│   │   ├── __init__.py                         # ✨ New
│   │   ├── config_scheduler.py                 # ✨ New: ConfigScheduler class
│   │   └── schedules/
│   │       ├── aggressive_mcts.json            # ✨ New: Aggressive MCTS curriculum
│   │       ├── conservative_mcts.json          # ✨ New: Conservative alternative
│   │       └── fast_test.json                  # ✨ New: Quick testing schedule
│   └── monitoring/
│       ├── __init__.py                         # ✨ New
│       ├── training_server.py                  # ✨ New: FastAPI backend
│       ├── training_state.py                   # ✨ New: Shared state manager
│       ├── api_routes.py                       # ✨ New: REST API routes
│       ├── dashboard.html                      # ✨ New: Web UI
│       ├── dashboard.css                       # ✨ New: Styling
│       ├── dashboard.js                        # ✨ New: Real-time updates
│       └── static/                             # ✨ New: Static assets
├── docs/
│   ├── training/
│   │   ├── DASHBOARD_GUIDE.md                  # ✨ New: Dashboard documentation
│   │   ├── warmup_validation.md                # ✨ New: LR schedule validation
│   │   ├── mcts_schedule_benchmarks.md         # ✨ New: Performance benchmarks
│   │   └── game_distribution_validation.md     # ✨ New: Game mix validation
├── scripts/
│   ├── visualize_lr_schedule.py                # ✨ New: Plot LR curve
│   └── analyze_game_distribution.py            # ✨ New: Game mix histogram
└── ml/tests/
    └── test_checkpoint_resume.py               # ✨ New: Resume testing

Legend:
✨ New file
✏️ Modified file
```

---

### Key Classes and Functions

#### `WarmupCosineScheduler` (ml/training/schedulers.py)
```python
class WarmupCosineScheduler:
    """Learning rate scheduler with warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_iterations, total_iterations,
                 lr_min=1e-5, lr_max=1e-3):
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.total_iterations = total_iterations
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.current_iteration = 0

    def step(self):
        """Update learning rate."""
        if self.current_iteration < self.warmup_iterations:
            # Linear warmup
            lr = self.lr_max * (self.current_iteration / self.warmup_iterations)
        else:
            # Cosine annealing
            progress = (self.current_iteration - self.warmup_iterations) / \
                       (self.total_iterations - self.warmup_iterations)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
                 (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_iteration += 1

    def get_last_lr(self):
        """Return current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Serialize scheduler state."""
        return {
            'current_iteration': self.current_iteration,
            'warmup_iterations': self.warmup_iterations,
            'total_iterations': self.total_iterations,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
        }

    def load_state_dict(self, state_dict):
        """Restore scheduler state."""
        self.current_iteration = state_dict['current_iteration']
        self.warmup_iterations = state_dict['warmup_iterations']
        self.total_iterations = state_dict['total_iterations']
        self.lr_min = state_dict['lr_min']
        self.lr_max = state_dict['lr_max']
```

---

#### `ConfigScheduler` (ml/scheduling/config_scheduler.py)
```python
class ConfigScheduler:
    """Manages iteration-based configuration updates."""

    def __init__(self, schedule_file: str):
        with open(schedule_file, 'r') as f:
            self.schedule = json.load(f)

        self.validate_schedule()
        self.phase_boundaries = self._compute_boundaries()

    def validate_schedule(self):
        """Ensure schedule is valid."""
        # Check for overlapping ranges, gaps, invalid params
        pass

    def get_config_for_iteration(self, iteration: int) -> dict:
        """Return config overrides for this iteration."""
        for phase in self.schedule['schedule']:
            start, end = self._parse_range(phase['iterations'])
            if start <= iteration <= end:
                return phase['overrides']

        raise ValueError(f"No config found for iteration {iteration}")

    def _parse_range(self, range_str: str) -> Tuple[int, int]:
        """Parse '1-50' → (1, 50)."""
        start, end = range_str.split('-')
        return int(start), int(end)

    def _compute_boundaries(self) -> List[int]:
        """Return list of iterations where config changes."""
        boundaries = []
        for phase in self.schedule['schedule']:
            start, end = self._parse_range(phase['iterations'])
            boundaries.append(start)
        return sorted(set(boundaries))

    def preview(self) -> str:
        """Generate human-readable schedule table."""
        lines = []
        lines.append("Configuration Schedule:")
        lines.append("-" * 80)
        lines.append(f"{'Iterations':<15} {'Determinizations':<20} {'Simulations':<15} {'Games/Iter':<15}")
        lines.append("-" * 80)

        for phase in self.schedule['schedule']:
            iters = phase['iterations']
            det = phase['overrides'].get('num_determinizations', '-')
            sims = phase['overrides'].get('simulations_per_determinization', '-')
            games = phase['overrides'].get('games_per_iteration', '-')
            lines.append(f"{iters:<15} {det:<20} {sims:<15} {games:<15}")

        return '\n'.join(lines)
```

---

#### `CheckpointRetentionPolicy` (ml/training/checkpoint_manager.py)
```python
class CheckpointRetentionPolicy:
    """Manages checkpoint retention based on tiered rules."""

    def __init__(self, checkpoint_dir: str, current_iteration: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.current_iteration = current_iteration

    def should_keep(self, checkpoint_iteration: int) -> bool:
        """Determine if checkpoint should be kept."""
        age = self.current_iteration - checkpoint_iteration

        # Rule 1: Keep last 10 iterations
        if age <= 10:
            return True

        # Rule 2: Keep every 2nd from 11-30 iterations ago
        if 11 <= age <= 30:
            return (checkpoint_iteration % 2) == 0

        # Rule 3: Keep every 5th from 31-50 iterations ago
        if 31 <= age <= 50:
            return (checkpoint_iteration % 5) == 0

        # Rule 4: Keep every 10th for 51+ iterations ago
        if age > 50:
            return (checkpoint_iteration % 10) == 0

        return False

    def cleanup(self):
        """Delete checkpoints that don't match retention rules."""
        deleted_count = 0

        for checkpoint_file in self.checkpoint_dir.glob('checkpoint_*.pth'):
            # Extract iteration number from filename
            match = re.search(r'checkpoint_(\d+)\.pth', checkpoint_file.name)
            if not match:
                continue

            checkpoint_iter = int(match.group(1))

            # Never delete best_model.pth or latest checkpoint
            if checkpoint_iter == self.current_iteration:
                continue

            if not self.should_keep(checkpoint_iter):
                checkpoint_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted checkpoint {checkpoint_file.name} (retention policy)")

        logger.info(f"Checkpoint cleanup: removed {deleted_count} old checkpoints")
```

---

#### Signal Handler (ml/train.py)
```python
import signal
import sys

shutdown_requested = False

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested

    logger.info(f"\nReceived signal {sig}, initiating graceful shutdown...")
    shutdown_requested = True

    # Don't exit immediately; let training loop save checkpoint

def setup_signal_handlers():
    """Register signal handlers."""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)  # Terminal close (Unix)

# In main():
setup_signal_handlers()

# In training loop:
for iteration in range(start_iter, config.num_iterations):
    # Check shutdown flag after each game
    if shutdown_requested:
        logger.info("Shutdown requested, saving checkpoint...")
        pipeline.save_checkpoint(reason="shutdown")
        logger.info("Checkpoint saved. Training can be resumed with --resume")
        break

    # ... training logic ...
```

---

## Testing & Validation

### Phase 1 Tests: Checkpoint Resume

**Test Suite:** `ml/tests/test_checkpoint_resume.py`

```python
class TestCheckpointResume:
    def test_interrupt_at_iteration_boundary(self):
        """Test interrupting between iterations."""
        # Start training for 5 iterations
        # Interrupt after iteration 3 completes
        # Resume and verify starts at iteration 4

    def test_interrupt_mid_iteration(self):
        """Test interrupting during self-play."""
        # Start training
        # Interrupt after 3000/10000 games
        # Resume and verify continues from game 3001

    def test_interrupt_during_training(self):
        """Test interrupting during NN training."""
        # Interrupt during epoch 5/10
        # Resume and verify completes remaining epochs

    def test_multiple_interrupts(self):
        """Test multiple interrupt/resume cycles."""
        # Start → interrupt → resume → interrupt → resume
        # Verify no state corruption

    def test_checkpoint_retention(self):
        """Test retention policy deletes correct checkpoints."""
        # Create 50 dummy checkpoints
        # Run cleanup
        # Verify correct checkpoints remain
```

---

### Phase 2 Tests: Learning Rate Schedule

**Test Suite:** `ml/tests/test_scheduler.py`

```python
class TestWarmupCosineScheduler:
    def test_warmup_phase(self):
        """Test LR increases linearly during warmup."""
        scheduler = WarmupCosineScheduler(...)
        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Verify linear increase
        assert lrs[0] == 0.0
        assert lrs[9] == 0.001
        assert all(lrs[i] < lrs[i+1] for i in range(9))

    def test_cosine_annealing(self):
        """Test LR decreases smoothly after warmup."""
        scheduler = WarmupCosineScheduler(...)
        # Run warmup
        for _ in range(10):
            scheduler.step()

        # Check cosine phase
        lrs = []
        for _ in range(490):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Verify smooth decrease
        assert lrs[0] == 0.001
        assert lrs[-1] == pytest.approx(0.00001, rel=1e-4)
        assert all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1))

    def test_state_dict(self):
        """Test scheduler state saves/loads correctly."""
        scheduler = WarmupCosineScheduler(...)
        for _ in range(50):
            scheduler.step()

        state = scheduler.state_dict()

        new_scheduler = WarmupCosineScheduler(...)
        new_scheduler.load_state_dict(state)

        # Verify same LR after restore
        assert scheduler.get_last_lr() == new_scheduler.get_last_lr()
```

---

### Phase 3 Tests: Config Scheduler

**Test Suite:** `ml/tests/test_config_scheduler.py`

```python
class TestConfigScheduler:
    def test_schedule_loading(self):
        """Test schedule loads from JSON."""
        scheduler = ConfigScheduler('schedules/aggressive_mcts.json')
        assert scheduler.schedule is not None

    def test_config_for_iteration(self):
        """Test correct config returned for each iteration."""
        scheduler = ConfigScheduler('schedules/aggressive_mcts.json')

        config_1 = scheduler.get_config_for_iteration(1)
        assert config_1['num_determinizations'] == 1

        config_100 = scheduler.get_config_for_iteration(100)
        assert config_100['num_determinizations'] == 2

        config_500 = scheduler.get_config_for_iteration(500)
        assert config_500['num_determinizations'] == 5

    def test_schedule_validation(self):
        """Test invalid schedules are rejected."""
        # Test overlapping ranges
        # Test gaps in coverage
        # Test invalid parameter names
```

---

### Phase 4 Tests: Game Distribution

**Test Suite:** `ml/tests/test_game_distribution.py`

```python
class TestGameDistribution:
    def test_weighted_sampling(self):
        """Test game configs match target distribution."""
        distribution = {
            (5, 7): 0.50,
            (4, 6): 0.15,
            (4, 7): 0.10,
            (6, 6): 0.10,
            (6, 7): 0.10,
            (3, 5): 0.05,
        }

        # Sample 10,000 games
        samples = [sample_game_config(distribution) for _ in range(10000)]

        # Count distribution
        counts = Counter(samples)

        # Verify matches target (±2%)
        assert 4800 <= counts[(5, 7)] <= 5200
        assert 1300 <= counts[(4, 6)] <= 1700

    def test_50_percent_constraint(self):
        """Test at least 50% of games are 5p/7c."""
        # Run actual self-play for 1000 games
        # Count (5, 7) games
        # Assert >= 500
```

---

### Phase 5 Tests: Dashboard API

**Test Suite:** `ml/tests/test_training_api.py`

```python
class TestTrainingAPI:
    def test_status_endpoint(self):
        """Test /api/status returns valid data."""
        response = client.get('/api/status')
        assert response.status_code == 200
        data = response.json()
        assert 'state' in data
        assert 'iteration' in data

    def test_control_start(self):
        """Test start control sets flag correctly."""
        response = client.post('/api/control/start')
        assert response.status_code == 200
        assert training_state.state == 'running'

    def test_control_pause(self):
        """Test pause control sets flag correctly."""
        response = client.post('/api/control/pause')
        assert response.status_code == 200
        # State should transition to 'paused' after current game

    def test_websocket_updates(self):
        """Test WebSocket sends updates."""
        with client.websocket_connect('/ws/updates') as ws:
            # Trigger state change
            training_state.iteration = 10

            # Receive update
            data = ws.receive_json()
            assert data['type'] == 'status_update'
            assert data['data']['iteration'] == 10
```

---

## Timeline & Dependencies

### Critical Path

```
Phase 1.1 → Phase 1.2 → Phase 1.3 → Phase 1.4
                                        ↓
                        Phase 2.1 → Phase 2.2
                                        ↓
        Phase 3.1 → Phase 3.2 → Phase 3.3
                        ↓
        Phase 4.1 → Phase 4.2 → Phase 4.3
                                        ↓
Phase 5.1 → Phase 5.2 → Phase 5.3 → Phase 5.4
```

### Parallel Work Opportunities

**Can be done in parallel:**
- Phase 2 (LR scheduler) and Phase 3.1 (config scheduler infrastructure)
- Phase 4 (game distribution) and Phase 3.3 (MCTS benchmarking)
- Phase 5 (dashboard) can start after Phase 1.2 (state manager exists)

**Must be sequential:**
- Phase 1.1 → 1.2 → 1.3 (checkpointing builds on itself)
- Phase 3.1 → 3.2 (schedule infrastructure before MCTS schedule)

### Recommended Order

**Week 1 (High Priority):**
- Phase 1.1, 1.2, 1.3, 1.4 (graceful shutdown - CRITICAL)
- Phase 2.1, 2.2 (LR warmup - quick win)

**Week 2 (Biggest Impact):**
- Phase 3.1, 3.2, 3.3 (dynamic MCTS - time savings)
- Phase 4.1, 4.2 (game distribution - convergence speed)

**Week 3 (Quality of Life):**
- Phase 4.3 (validation)
- Phase 5.1, 5.2, 5.3, 5.4 (dashboard - monitoring)

### Minimum Viable Implementation

**If time-constrained, implement:**
1. **Phase 1.1, 1.2** (signal handlers + intra-iteration checkpoints) - CRITICAL
2. **Phase 3.2** (aggressive MCTS schedule) - BIGGEST IMPACT
3. **Phase 4.1** (weighted game distribution) - CONVERGENCE SPEED

**Can defer:**
- Phase 1.3 (checkpoint retention - disk space management)
- Phase 2 (LR warmup - nice to have, not critical)
- Phase 5 (dashboard - quality of life)

---

## Expected Outcomes

### Training Time Reduction

**Baseline (current):**
- 3 det × 30 sims, 10K games/iter for 500 iterations
- **Total: ~3264h (~136 days)**

**With TOMS (aggressive schedule):**
- Variable MCTS: 1→2→3→4→5 det
- Variable games: 5K→8K→10K→12K→15K
- **Total: ~2301h (~96 days)**
- **Reduction: 40 days (29% faster)**

**With TOMS (conservative schedule):**
- Variable MCTS: 2→3→4 det
- Fixed games: 10K
- **Total: ~2800h (~117 days)**
- **Reduction: 19 days (14% faster)**

### Safety Improvements

**Before TOMS:**
- Checkpoint every 10 iterations (24-48h of work at risk)
- No intra-iteration checkpoints (1h per iteration at risk)
- Manual interrupt could lose progress

**After TOMS:**
- Checkpoint every iteration (2-4h of work at risk)
- Intra-iteration checkpoints every game (<1min at risk)
- Graceful shutdown guarantees zero data loss
- Smart retention prevents disk overflow

### Convergence Speed

**Game Distribution Optimization:**
- 50%+ games at optimal configuration (5p/7c)
- Model learns target scenario faster
- Reduced variance from edge cases
- Expected ELO improvement: +50-100 ELO at same iteration count

### Monitoring & Control

**Before TOMS:**
- No visibility into training progress (check logs manually)
- No control (must SSH and kill process)
- No real-time metrics

**After TOMS:**
- Live dashboard with all key metrics
- One-click pause/resume
- Real-time graphs (ELO, loss, speed)
- Estimated time remaining

---

## Future Enhancements (Post-TOMS)

### Adaptive Scheduling
- Adjust MCTS complexity based on loss plateau (not just iteration count)
- Increase games/iteration when ELO improvement slows
- Dynamic learning rate based on gradient magnitudes

### Distributed Training
- Multi-GPU support for self-play workers
- Distributed training across multiple machines
- Cloud integration (GCP, AWS, Azure)

### Advanced Monitoring
- TensorBoard integration
- Weights & Biases (W&B) logging
- MCTS tree visualizations
- Belief state heatmaps

### Experiment Management
- Track multiple training runs simultaneously
- A/B testing different schedules
- Hyperparameter sweeps with Ray Tune
- Automatic checkpoint comparison

### Production Features
- Model versioning and rollback
- A/B testing new models against production
- Inference performance profiling
- ONNX optimization and quantization

---

## Conclusion

The Training Optimization & Monitoring System (TOMS) represents a comprehensive enhancement to the BlobMaster training pipeline. By implementing graceful shutdown, dynamic scheduling, optimized game distribution, and real-time monitoring, we expect to:

1. **Reduce training time by ~40 days** (29% faster) via aggressive MCTS curriculum
2. **Eliminate progress loss** via intra-iteration checkpointing
3. **Accelerate convergence** by focusing on 5p/7c configuration (50%+ of games)
4. **Improve developer experience** via web dashboard with full control

The 16-session implementation plan provides a clear roadmap for achieving these goals, with each session delivering tangible value. The modular design allows for prioritization (implement Phase 1 and 3 first if time-constrained) while maintaining the option to complete all phases for maximum benefit.

**Recommended Start:** Begin with Phase 1.1 (signal handlers) to enable safe interrupts immediately, then proceed through phases sequentially or leverage parallel work opportunities based on available development resources.

---

**Document End**

For questions or clarifications, refer to:
- Implementation code in `ml/` directory
- Test suites in `ml/tests/`
- Performance benchmarks in `docs/performance/`
- Dashboard guide in `docs/training/DASHBOARD_GUIDE.md`
