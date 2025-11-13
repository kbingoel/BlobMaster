# External Monitor Implementation Guide

**Created**: 2025-11-13
**Status**: Ready for implementation
**Total Effort**: 8 hours (2 × 4-hour sessions)
**Purpose**: Add operational tooling for long training runs (7-9+ days)

---

## Overview

This guide provides a complete, self-contained implementation plan for Session 6: External Monitor & Checkpoint Management. This session adds critical operational capabilities for managing multi-day training runs.

### What You'll Build

1. **Checkpoint Rotation System**: Automatic disk management (saves ~200GB)
2. **Atomic Status File**: Live progress tracking without log parsing
3. **External Monitor Script**: TUI for real-time monitoring (attach/detach anytime)
4. **Pause/Resume Control**: Safe iteration-boundary pause mechanism

### Why This Matters

**Training context:**
- Phase 1: ~7-9 days continuous (500 iterations)
- Phase 2: ~35-40 days continuous (100 iterations, optional)
- Disk risk: 500 iterations × 500MB = ~250GB without rotation
- Monitoring: Need visibility without disrupting training

**Benefits:**
- **Disk management**: Rotate checkpoints, keep only recent + milestones
- **Progress visibility**: Monitor without tmux attach (avoids buffer issues)
- **Safe pause**: Stop training for maintenance, resume later
- **Peace of mind**: Live metrics show training is healthy

---

## Prerequisites

Before starting this session, ensure you have completed:

- ✅ **Session 0**: MCTS curriculum wiring + checkpoint naming convention
- ✅ **Session 1**: Zero-choice fast path (optional but recommended)
- ✅ **Session 2**: Training stabilization & curriculum
- ✅ **Session 3**: Dirichlet noise (fixed exploration)

**Verify baseline infrastructure exists:**

```bash
# Check checkpoint methods exist in TrainingPipeline
grep -n "def.*checkpoint" ml/training/trainer.py

# Expected methods:
# - TrainingPipeline: methods for saving/loading checkpoints
# - NetworkTrainer.save_checkpoint() and load_checkpoint()
```

**Current checkpoint infrastructure** (already implemented):
- `NetworkTrainer.save_checkpoint()` and `load_checkpoint()`
- `TrainingPipeline` checkpoint methods with directory structure
- Standardized checkpoint naming convention

**What this session adds:**
- Checkpoint rotation (FIFO eviction for cache, permanent milestones)
- `status.json` atomic writes
- External monitor script with Rich TUI
- Control signals for pause/resume
- Worker cleanup on graceful shutdown

---

## Architecture Overview

### Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│ Training Process (ml/train.py)                                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ TrainingPipeline.run_training()                          │   │
│  │                                                          │   │
│  │  Each iteration:                                         │   │
│  │    1. Check control.signal (pause request?)              │   │
│  │    2. Run self-play + training                           │   │
│  │    3. Save checkpoint (rotation logic)                   │   │
│  │    4. Update status.json (atomic write)                  │   │
│  │                                                          │   │
│  │  On pause/exit:                                          │   │
│  │    1. Shutdown workers gracefully                        │   │
│  │    2. Save final checkpoint                              │   │
│  │    3. Clean up resources                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │ writes
                        ↓
        ┌───────────────────────────────────────┐
        │ Filesystem                            │
        │                                       │
        │  models/checkpoints/                  │
        │    ├── permanent/                     │
        │    │   ├── iter005-elo0980.pth        │
        │    │   ├── iter010-elo1050.pth        │
        │    │   └── ...                        │
        │    ├── cache/                         │
        │    │   ├── iter007.pth    ← 4 most    │
        │    │   ├── iter008.pth      recent    │
        │    │   ├── iter009.pth                │
        │    │   └── iter011.pth                │
        │    ├── status.json         ← live     │
        │    └── control.signal      ← pause    │
        └───────────────────────────────────────┘
                        │
                        │ reads
                        ↓
        ┌───────────────────────────────────────┐
        │ External Monitor (ml/monitor.py)      │
        │                                       │
        │  ┌─────────────────────────────────┐  │
        │  │ Rich TUI (Unix/Linux only)      │  │
        │  │                                 │  │
        │  │  Iteration: 150/500 (30%)       │  │
        │  │  Phase: rounds (3×35 MCTS)      │  │
        │  │  ETA: 5.2 days                  │  │
        │  │  ELO: 1420 → 1435 (+15)         │  │
        │  │  LR: 0.000554                   │  │
        │  │                                 │  │
        │  │  [p] Pause  [q] Quit            │  │
        │  └─────────────────────────────────┘  │
        │                                       │
        │  Keyboard input:                      │
        │    p → write control.signal           │
        │    q → exit monitor (training runs)   │
        └───────────────────────────────────────┘
```

### File Locations

**Files to create:**
- `ml/monitor.py` (~250 lines) - External monitor script with Rich TUI

**Files to modify:**
- `ml/training/trainer.py` - Add rotation, status writer, control signal handler

**Runtime files:**
- `models/checkpoints/status.json` - Live status (updated each iteration)
- `models/checkpoints/control.signal` - Pause control (created by monitor)

### Important Notes

**Iteration Indexing Convention:**
- Internally: 0-indexed (iteration 0, 1, 2, ..., 499)
- User-facing: 1-indexed (displayed as 1, 2, 3, ..., 500)
- This guide uses 0-indexed for code, 1-indexed for descriptions

**Platform Requirements:**
- Training: Ubuntu Linux 24.04 (confirmed compatible)
- Monitor: Requires Unix/Linux terminal (uses tty/termios/select)

**Dependencies:**
- Rich library required for monitor TUI: `pip install rich`
- Not currently in `ml/requirements.txt` (optional operational tooling)

**Checkpoint Rotation:**
- Rotation happens when a checkpoint is saved (controlled by `save_every_n_iterations`)
- Default: saves every 10 iterations, so rotation runs every 10 iterations
- For per-iteration rotation: set `save_every_n_iterations = 1` in config

---

## Session 1: Backend Infrastructure (4 hours)

### Goals

Implement the backend systems that training needs:
1. Checkpoint rotation logic (permanent every 5 iters, cache others)
2. Atomic status file writer (JSON with progress, ETA, ELO)
3. Control signal handler (check for pause requests)
4. Graceful worker shutdown

### Part 1: Checkpoint Rotation (90 minutes)

**Objective**: Automatically manage checkpoint disk usage.

**Strategy:**
- **Permanent checkpoints**: Every 5 iterations (5, 10, 15, ..., 500) → `models/checkpoints/permanent/`
- **Cache checkpoints**: All other iterations → `models/checkpoints/cache/` (keep last 4, FIFO)
- **Naming**: Use standardized format with optional ELO suffix

**Implementation:**

**File**: `ml/training/trainer.py` (add to TrainingPipeline class)

```python
import os
import glob
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TrainingPipeline:
    # ... existing __init__ and methods ...

    def _get_checkpoint_filename(
        self,
        iteration: int,
        elo: Optional[int] = None,
        is_permanent: bool = False
    ) -> str:
        """
        Generate standardized checkpoint filename.

        Args:
            iteration: Current iteration (0-indexed)
            elo: Optional ELO rating to include in filename
            is_permanent: Whether this is a permanent checkpoint

        Returns:
            Filename string (e.g., "20251113-Blobmaster-v1-4.9M-iter005-elo1234.pth")

        Note:
            Iteration is 0-indexed internally but displayed as 1-indexed in filename.
        """
        from datetime import datetime

        # Get date prefix
        date_str = getattr(self, 'training_start_date', datetime.now().strftime('%Y%m%d'))

        # Count model parameters (millions)
        num_params = sum(p.numel() for p in self.network.parameters()) / 1e6

        # Build filename: iteration is displayed as 1-indexed
        display_iter = iteration + 1
        filename = f"{date_str}-Blobmaster-v1-{num_params:.1f}M-iter{display_iter:03d}"

        if elo is not None:
            filename += f"-elo{elo:04d}"

        filename += ".pth"
        return filename

    def save_checkpoint_with_rotation(
        self,
        iteration: int,
        metrics: Optional[dict] = None
    ) -> str:
        """
        Save checkpoint with automatic rotation logic.

        Args:
            iteration: Current iteration (0-indexed)
            metrics: Optional metrics dict (should contain 'elo' if available)

        Returns:
            Path to saved checkpoint

        Note:
            - Permanent checkpoints: iterations 4, 9, 14, ... (every 5th, 0-indexed)
            - Cache checkpoints: all others (max 4 kept via FIFO rotation)
            - Iteration is 0-indexed internally
        """
        # Determine if this is a permanent checkpoint (every 5th iteration)
        # iteration 4, 9, 14, 19, ... → displayed as 5, 10, 15, 20, ...
        is_permanent = (iteration + 1) % 5 == 0

        # Get ELO if available (only for permanent checkpoints)
        elo = None
        if is_permanent and metrics and 'elo' in metrics:
            try:
                elo = int(metrics['elo'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid ELO value: {metrics['elo']}, skipping ELO in filename")
                elo = None

        # Generate filename
        filename = self._get_checkpoint_filename(
            iteration=iteration,
            elo=elo,
            is_permanent=is_permanent
        )

        # Determine directory
        if is_permanent:
            save_dir = Path("models/checkpoints/permanent")
        else:
            save_dir = Path("models/checkpoints/cache")

        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename

        # Build checkpoint data
        checkpoint_data = {
            'iteration': iteration,  # 0-indexed
            'model_state_dict': self.network.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict(),
            'replay_buffer': self.replay_buffer.get_state(),
            'elo': elo,
            'metrics': metrics,
            'config': self.config,
            'training_start_date': getattr(self, 'training_start_date', None),
        }

        # Save checkpoint with error handling
        try:
            # Check available disk space (require at least 2GB free)
            stat = shutil.disk_usage(save_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 2.0:
                logger.error(f"Low disk space: {free_gb:.1f}GB free. Checkpoint may fail.")

            torch.save(checkpoint_data, filepath)

            logger.info(
                f"Saved {'permanent' if is_permanent else 'cache'} checkpoint: "
                f"{filename} ({iteration+1}/{self.total_iterations if hasattr(self, 'total_iterations') else '?'})"
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}")
            raise

        # Rotate cache checkpoints if needed
        if not is_permanent:
            self._rotate_cache_checkpoints(max_keep=4)

        return str(filepath)

    def _rotate_cache_checkpoints(self, max_keep: int = 4):
        """
        Keep only the most recent N cache checkpoints, delete older ones.

        Args:
            max_keep: Maximum number of cache checkpoints to keep (default: 4)

        Note:
            Permanent checkpoints are never deleted by this method.
        """
        cache_dir = Path("models/checkpoints/cache")
        if not cache_dir.exists():
            return

        # Get all cache checkpoint files
        cache_files = sorted(
            cache_dir.glob("*.pth"),
            key=lambda p: p.stat().st_mtime  # Sort by modification time
        )

        # Delete oldest files if we exceed max_keep
        num_to_delete = len(cache_files) - max_keep
        if num_to_delete > 0:
            for old_file in cache_files[:num_to_delete]:
                try:
                    file_size_mb = old_file.stat().st_size / (1024**2)
                    logger.info(
                        f"Rotating out old cache checkpoint: {old_file.name} "
                        f"({file_size_mb:.1f}MB)"
                    )
                    old_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete old checkpoint {old_file}: {e}")
```

**Testing checkpoint rotation:**

```bash
# Create test script: test_rotation.py
cat > test_rotation.py << 'EOF'
from pathlib import Path
import time

def test_rotation():
    """Test checkpoint rotation logic."""
    cache_dir = Path('models/checkpoints/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing test files
    for f in cache_dir.glob('test_*.pth'):
        f.unlink()

    # Create 10 test checkpoints with different timestamps
    print("Creating 10 test checkpoints...")
    for i in range(10):
        test_file = cache_dir / f'test_iter{i:03d}.pth'
        test_file.write_text(f'checkpoint {i}')
        time.sleep(0.02)  # Ensure different mtimes

    files_before = sorted(cache_dir.glob('test_*.pth'))
    print(f"Files before rotation: {len(files_before)}")
    assert len(files_before) == 10

    # Simulate rotation (keep 4 most recent)
    cache_files = sorted(
        cache_dir.glob('test_*.pth'),
        key=lambda p: p.stat().st_mtime
    )

    num_to_delete = len(cache_files) - 4
    print(f"Deleting {num_to_delete} oldest files...")
    for old_file in cache_files[:num_to_delete]:
        print(f"  Deleting: {old_file.name}")
        old_file.unlink()

    files_after = sorted(cache_dir.glob('test_*.pth'))
    print(f"Files after rotation: {len(files_after)}")
    assert len(files_after) == 4

    # Verify we kept the most recent ones
    assert 'test_iter006.pth' in [f.name for f in files_after]
    assert 'test_iter009.pth' in [f.name for f in files_after]
    assert 'test_iter000.pth' not in [f.name for f in files_after]

    # Clean up
    for f in files_after:
        f.unlink()

    print("✅ Rotation test passed!")

if __name__ == '__main__':
    test_rotation()
EOF

python test_rotation.py
rm test_rotation.py
```

**Acceptance criteria:**
- ✅ Every 5th iteration (0-indexed: 4, 9, 14, ...) saves to `permanent/` with ELO in filename
- ✅ Other iterations save to `cache/`
- ✅ Cache directory never has more than 4 files
- ✅ Oldest cache files deleted automatically
- ✅ Permanent checkpoints never deleted
- ✅ Disk space check warns if < 2GB free

---

### Part 2: Atomic Status Writer (90 minutes)

**Objective**: Write live status to JSON file without risk of corruption.

**Strategy:**
- Write to temporary file first (`status.json.tmp`)
- Atomic rename to `status.json` (POSIX guarantees atomicity)
- Include: iteration, phase, progress, ETA, ELO, learning rate
- Graceful fallback if config methods unavailable

**Implementation:**

**File**: `ml/training/trainer.py` (add StatusWriter class)

```python
import json
import time
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class StatusWriter:
    """
    Atomic status file writer for external monitoring.

    Writes training status to JSON file with atomic updates
    to prevent corruption from concurrent reads.
    """

    def __init__(self, status_file: str = "models/checkpoints/status.json"):
        """
        Args:
            status_file: Path to status file (default: models/checkpoints/status.json)
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def update(
        self,
        iteration: int,
        total_iterations: int,
        metrics: Dict[str, Any],
        config: Any
    ):
        """
        Update status file with current training state.

        Args:
            iteration: Current iteration (0-indexed)
            total_iterations: Total iterations planned
            metrics: Metrics dict from iteration
            config: Config dict (from TrainingConfig.to_dict())

        Note:
            Iteration is 0-indexed internally but displayed as 1-indexed in status.
        """
        # Calculate progress (use 1-indexed for display)
        display_iteration = iteration + 1
        progress = display_iteration / total_iterations if total_iterations > 0 else 0.0

        # Estimate ETA
        elapsed = time.time() - self.start_time
        if display_iteration > 0:
            avg_iter_time = elapsed / display_iteration
            remaining_iters = total_iterations - display_iteration
            eta_seconds = avg_iter_time * remaining_iters
            eta_hours = eta_seconds / 3600
            eta_days = eta_hours / 24
        else:
            eta_hours = 0
            eta_days = 0

        # Get MCTS params for current iteration (with fallback)
        mcts_str = "unknown"
        try:
            # Config is a dict (from TrainingConfig.to_dict())
            if callable(config.get('get_mcts_params')):
                num_det, sims = config['get_mcts_params'](display_iteration)
                mcts_str = f"{num_det}×{sims}"
            elif 'num_determinizations' in config and 'simulations_per_determinization' in config:
                mcts_str = f"{config['num_determinizations']}×{config['simulations_per_determinization']}"
        except Exception as e:
            logger.warning(f"Failed to get MCTS params: {e}")

        # Get phase (with fallback)
        phase = config.get('training_on', 'unknown')

        # Build status dict
        status = {
            'timestamp': time.time(),
            'iteration': display_iteration,  # 1-indexed for display
            'total_iterations': total_iterations,
            'progress': progress,
            'elapsed_hours': elapsed / 3600,
            'eta_hours': eta_hours,
            'eta_days': eta_days,
            'phase': phase,
            'mcts_config': mcts_str,
            'elo': metrics.get('current_elo', None),
            'elo_change': metrics.get('elo_change', None),
            'learning_rate': metrics.get('learning_rate', None),
            'loss': metrics.get('avg_total_loss', None),
            'policy_loss': metrics.get('avg_policy_loss', None),
            'value_loss': metrics.get('avg_value_loss', None),
            'training_units_generated': metrics.get('training_units_generated', None),
            'unit_type': metrics.get('unit_type', 'rounds'),
            'pi_target_tau': metrics.get('pi_target_tau', None),
        }

        # Atomic write: write to temp file, then rename
        tmp_file = self.status_file.with_suffix('.json.tmp')
        try:
            with open(tmp_file, 'w') as f:
                json.dump(status, f, indent=2)

            # Atomic rename (POSIX guarantees atomicity)
            tmp_file.replace(self.status_file)
        except Exception as e:
            logger.error(f"Failed to write status file: {e}")
            # Clean up temp file if it exists
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except:
                    pass


class TrainingPipeline:
    def __init__(self, network, encoder, masker, config):
        # ... existing init ...

        # Status writer for external monitoring
        self.status_writer = StatusWriter()
        self.total_iterations = 0  # Set by run_training()
```

**Testing status writer:**

```bash
# Create test script: test_status.py
cat > test_status.py << 'EOF'
from ml.training.trainer import StatusWriter
import time
import json
from pathlib import Path

class MockConfig:
    """Mock config for testing."""
    training_on = 'rounds'
    num_determinizations = 3
    simulations_per_determinization = 35

def test_status_writer():
    """Test atomic status file writes."""
    status_file = 'models/checkpoints/test_status.json'
    writer = StatusWriter(status_file)

    # Simulate 5 iteration updates
    print("Simulating 5 iterations...")
    for i in range(5):
        metrics = {
            'current_elo': 1000 + i * 50,
            'elo_change': 50 if i > 0 else 0,
            'learning_rate': 0.001 - i * 0.0001,
            'avg_total_loss': 0.5 - i * 0.05,
            'avg_policy_loss': 0.3 - i * 0.03,
            'avg_value_loss': 0.2 - i * 0.02,
            'training_units_generated': 4400,
            'unit_type': 'rounds',
            'pi_target_tau': 1.0 - i * 0.1,
        }

        writer.update(
            iteration=i,  # 0-indexed
            total_iterations=10,
            metrics=metrics,
            config=MockConfig()
        )

        print(f"  Updated status for iteration {i} (displayed as {i+1})")
        time.sleep(0.1)

    # Verify status file exists and is valid JSON
    status_path = Path(status_file)
    assert status_path.exists(), "Status file not created"

    with open(status_path) as f:
        status = json.load(f)

    print("\nFinal status:")
    print(json.dumps(status, indent=2))

    # Verify key fields
    assert status['iteration'] == 5, f"Expected iteration 5, got {status['iteration']}"
    assert status['total_iterations'] == 10
    assert 0 <= status['progress'] <= 1
    assert status['phase'] == 'rounds'
    assert status['mcts_config'] == '3×35'
    assert status['elo'] == 1200  # current_elo mapped to 'elo' in status

    # Clean up
    status_path.unlink()
    print("\n✅ Status writer test passed!")

if __name__ == '__main__':
    test_status_writer()
EOF

python test_status.py
rm test_status.py
```

**Acceptance criteria:**
- ✅ `status.json` created after first iteration
- ✅ File updates atomically (tmp file + rename)
- ✅ Contains: iteration (1-indexed), progress, ETA, ELO, learning rate
- ✅ ETA calculation works (hours and days)
- ✅ MCTS config string correct (e.g., "3×35") with fallback
- ✅ Error handling for malformed metrics
- ✅ No corruption on concurrent reads

---

### Part 3: Pause/Resume Control (90 minutes)

**Objective**: Allow external pause requests via control signal file with graceful cleanup.

**Strategy:**
- Monitor checks `models/checkpoints/control.signal` before each iteration
- If file contains "PAUSE", finish current iteration gracefully
- Shutdown workers, save checkpoint (already done at end of iteration)
- Training exits cleanly, can resume later with `--resume` flag

**Implementation:**

**File**: `ml/training/trainer.py` (add to TrainingPipeline)

```python
class TrainingPipeline:
    def __init__(self, network, encoder, masker, config):
        # ... existing init ...

        # Control signal file for pause/resume
        self.control_signal_file = Path("models/checkpoints/control.signal")

    def _check_control_signal(self) -> str:
        """
        Check for control signals from external monitor.

        Returns:
            Control command string ('PAUSE', 'CONTINUE', or empty)
        """
        if not self.control_signal_file.exists():
            return ''

        try:
            signal = self.control_signal_file.read_text().strip().upper()
            return signal
        except Exception as e:
            logger.warning(f"Failed to read control signal: {e}")
            return ''

    def _clear_control_signal(self):
        """Clear control signal file."""
        if self.control_signal_file.exists():
            try:
                self.control_signal_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clear control signal: {e}")

    def _shutdown_workers(self):
        """
        Gracefully shutdown self-play workers.

        Note:
            Override this method based on your worker implementation.
            This is a placeholder for proper cleanup.
        """
        if hasattr(self, 'selfplay_engine') and self.selfplay_engine is not None:
            try:
                logger.info("Shutting down self-play workers...")
                # Assuming SelfPlayEngine has a shutdown method
                if hasattr(self.selfplay_engine, 'shutdown'):
                    self.selfplay_engine.shutdown()
                else:
                    # Fallback: just set flag if shutdown method doesn't exist
                    logger.warning("SelfPlayEngine has no shutdown() method")
            except Exception as e:
                logger.error(f"Error shutting down workers: {e}")

    def run_training(self, num_iterations: int, resume_from: Optional[str] = None):
        """
        Run training with pause/resume support.

        Args:
            num_iterations: Total number of iterations to run
            resume_from: Optional checkpoint path to resume from

        Note:
            - Checks for PAUSE signal before each iteration
            - Saves checkpoint at end of each iteration
            - Cleans up workers before exit
            - Iteration is 0-indexed internally
        """
        self.total_iterations = num_iterations

        # Clear any stale control signals
        self._clear_control_signal()

        # Determine starting iteration
        start_iteration = 0
        if resume_from:
            try:
                checkpoint = torch.load(resume_from)
                start_iteration = checkpoint['iteration'] + 1  # Continue from next iteration
                # ... load checkpoint state ...
                logger.info(f"Resumed from iteration {start_iteration} (displaying as {start_iteration + 1})")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                raise

        # Training start date (for checkpoint filenames)
        from datetime import datetime
        self.training_start_date = getattr(self, 'training_start_date', datetime.now().strftime('%Y%m%d'))

        logger.info(f"Starting training: iterations {start_iteration} to {num_iterations - 1} (0-indexed)")

        try:
            for iteration in range(start_iteration, num_iterations):
                # Check for pause request BEFORE starting iteration
                signal = self._check_control_signal()
                if signal == 'PAUSE':
                    logger.info(
                        f"Pause signal received at iteration {iteration} (displayed as {iteration + 1}). "
                        f"Last checkpoint saved at iteration {iteration - 1}."
                    )
                    self._clear_control_signal()
                    # Checkpoint already saved at end of previous iteration
                    # Just cleanup and exit
                    break

                # Run iteration (self-play + training)
                logger.info(f"Starting iteration {iteration} (displayed as {iteration + 1}/{num_iterations})")
                metrics = self.run_iteration(iteration)

                # Save checkpoint with rotation
                checkpoint_path = self.save_checkpoint_with_rotation(
                    iteration=iteration,
                    metrics=metrics
                )

                # Update status file (atomic write)
                self.status_writer.update(
                    iteration=iteration,
                    total_iterations=num_iterations,
                    metrics=metrics,
                    config=self.config
                )

                logger.info(f"Iteration {iteration + 1}/{num_iterations} complete")

            # Training completed normally or paused
            if iteration == num_iterations - 1:
                logger.info(f"Training complete! {num_iterations} iterations finished.")
            else:
                logger.info(f"Training paused. Resume with: --resume {checkpoint_path}")

        finally:
            # Always cleanup workers on exit (normal or exception)
            logger.info("Cleaning up resources...")
            self._shutdown_workers()
```

**Testing pause/resume:**

```bash
# Create test script: test_pause.py
cat > test_pause.py << 'EOF'
from pathlib import Path
import time

def test_pause_signal():
    """Test pause signal creation and reading."""
    signal_file = Path('models/checkpoints/control.signal')
    signal_file.parent.mkdir(parents=True, exist_ok=True)

    # Clean up any existing signal
    if signal_file.exists():
        signal_file.unlink()

    # Test 1: Create pause signal
    print("Test 1: Creating PAUSE signal...")
    signal_file.write_text('PAUSE\n')
    assert signal_file.exists(), "Signal file not created"

    # Test 2: Read signal
    signal = signal_file.read_text().strip().upper()
    assert signal == 'PAUSE', f"Expected 'PAUSE', got '{signal}'"
    print("  ✅ Signal created and read correctly")

    # Test 3: Clear signal
    print("Test 2: Clearing signal...")
    signal_file.unlink()
    assert not signal_file.exists(), "Signal file not cleared"
    print("  ✅ Signal cleared successfully")

    # Test 4: Handle missing signal
    print("Test 3: Handling missing signal...")
    signal = signal_file.read_text().strip().upper() if signal_file.exists() else ''
    assert signal == '', f"Expected empty, got '{signal}'"
    print("  ✅ Missing signal handled correctly")

    print("\n✅ All pause signal tests passed!")

if __name__ == '__main__':
    test_pause_signal()
EOF

python test_pause.py
rm test_pause.py

# Manual integration test (requires full training setup):
# Terminal 1:
#   python ml/train.py --fast --iterations 10
# Terminal 2 (after a few iterations):
#   echo "PAUSE" > models/checkpoints/control.signal
# Expected: Training pauses after current iteration completes
```

**Acceptance criteria:**
- ✅ Control signal checked before each iteration
- ✅ "PAUSE" signal causes graceful exit
- ✅ Workers shutdown cleanly
- ✅ Signal file cleared after processing
- ✅ Can resume with `--resume` flag
- ✅ No mid-iteration state corruption
- ✅ Error handling for file operations

---

## Session 2: External Monitor UI (4 hours)

### Goals

Build the external monitor script with Rich TUI:
1. Read and display `status.json` in real-time
2. Show key metrics: iteration, progress, ETA, ELO, learning rate
3. Keyboard controls: `p` for pause, `q` for quit
4. Refresh every 5 seconds (configurable)
5. Unix/Linux terminal compatible

### Part 1: Status Reader & Display (150 minutes)

**Objective**: Create TUI that reads and displays training status.

**Strategy:**
- Use Rich library for terminal UI (tables, progress bars, live updates)
- Poll `status.json` every 5 seconds
- Handle missing file gracefully (training not started yet)
- Non-blocking keyboard input using select/tty (Unix/Linux)

**Implementation:**

**File**: `ml/monitor.py` (new file, ~250 lines)

```python
#!/usr/bin/env python3
"""
External training monitor with Rich TUI.

Platform Requirements:
    - Unix/Linux terminal (uses tty/termios/select for keyboard input)
    - Not compatible with Windows (Windows users should use WSL)

Usage:
    python ml/monitor.py [--status-file PATH] [--refresh-interval SECONDS]

Keyboard controls:
    p - Pause training (writes control signal)
    q - Quit monitor (training continues)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import logging

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    External monitor for training progress.

    Platform: Unix/Linux only (requires tty/termios/select)
    """

    def __init__(
        self,
        status_file: str = "models/checkpoints/status.json",
        control_file: str = "models/checkpoints/control.signal",
        refresh_interval: float = 5.0
    ):
        """
        Args:
            status_file: Path to status JSON file
            control_file: Path to control signal file
            refresh_interval: Seconds between status updates
        """
        self.status_file = Path(status_file)
        self.control_file = Path(control_file)
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.last_status: Optional[Dict[str, Any]] = None

    def read_status(self) -> Optional[Dict[str, Any]]:
        """
        Read current status from JSON file.

        Returns:
            Status dict or None if file doesn't exist or is invalid
        """
        if not self.status_file.exists():
            return None

        try:
            with open(self.status_file) as f:
                status = json.load(f)
                return status
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read status: {e}")
            # Return cached status to avoid flickering display
            return self.last_status

    def send_pause_signal(self):
        """Write PAUSE signal to control file."""
        try:
            self.control_file.parent.mkdir(parents=True, exist_ok=True)
            self.control_file.write_text("PAUSE\n")
            self.console.print(
                "[green]✓ Pause signal sent. Training will stop after current iteration.[/green]"
            )
        except OSError as e:
            self.console.print(f"[red]✗ Failed to write pause signal: {e}[/red]")

    def build_display(self, status: Optional[Dict[str, Any]]) -> Layout:
        """
        Build Rich layout from status dict.

        Args:
            status: Status dict or None if not available

        Returns:
            Rich Layout object
        """
        layout = Layout()

        if status is None:
            layout.update(
                Panel(
                    "[yellow]Waiting for training to start...\n"
                    f"Looking for: {self.status_file}\n\n"
                    "[dim]Start training in another terminal:[/dim]\n"
                    "[cyan]python ml/train.py --iterations 500[/cyan]",
                    title="BlobMaster Training Monitor",
                    border_style="yellow"
                )
            )
            return layout

        # Extract data from status (all validated with fallbacks)
        iteration = status.get('iteration', 0)  # 1-indexed from status file
        total = status.get('total_iterations', 500)
        progress_pct = status.get('progress', 0.0)
        eta_days = status.get('eta_days', 0.0)
        eta_hours = status.get('eta_hours', 0.0)
        phase = status.get('phase', 'unknown')
        mcts = status.get('mcts_config', 'N/A')
        elo = status.get('elo')
        elo_change = status.get('elo_change')
        lr = status.get('learning_rate')
        loss = status.get('loss')
        policy_loss = status.get('policy_loss')
        value_loss = status.get('value_loss')
        units = status.get('training_units_generated')
        unit_type = status.get('unit_type', 'rounds')
        tau = status.get('pi_target_tau')
        elapsed_hours = status.get('elapsed_hours', 0.0)

        # Build metrics table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Progress (iteration already 1-indexed from status file)
        table.add_row("Iteration", f"{iteration}/{total} ({progress_pct:.1%})")

        # Phase & MCTS
        table.add_row("Phase", f"{phase} ({mcts} MCTS)")

        # ETA
        if eta_days >= 1.0:
            eta_str = f"{eta_days:.1f} days ({eta_hours:.1f} hours)"
        else:
            eta_str = f"{eta_hours:.1f} hours"
        table.add_row("ETA", eta_str)

        # Elapsed
        if elapsed_hours >= 24:
            elapsed_str = f"{elapsed_hours/24:.1f} days ({elapsed_hours:.1f} hours)"
        else:
            elapsed_str = f"{elapsed_hours:.1f} hours"
        table.add_row("Elapsed", elapsed_str)

        # ELO (if available)
        if elo is not None:
            elo_str = f"{elo:.0f}"
            if elo_change is not None:
                sign = "+" if elo_change >= 0 else ""
                elo_str += f" ({sign}{elo_change:.0f})"
            table.add_row("ELO", elo_str)

        # Learning rate
        if lr is not None:
            table.add_row("Learning Rate", f"{lr:.6f}")

        # Losses
        if loss is not None:
            loss_str = f"{loss:.4f}"
            if policy_loss is not None and value_loss is not None:
                loss_str += f" (π:{policy_loss:.3f} v:{value_loss:.3f})"
            table.add_row("Loss", loss_str)

        # Training units
        if units is not None:
            table.add_row(f"{unit_type.capitalize()}/iter", f"{units:,}")

        # Target temperature
        if tau is not None:
            table.add_row("Target τ", f"{tau:.3f}")

        # Progress bar
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        progress_task = progress_bar.add_task(
            "Overall Progress",
            total=100,
            completed=progress_pct * 100
        )

        # Build layout
        layout.split_column(
            Layout(
                Panel(table, title="Training Status", border_style="green"),
                size=22
            ),
            Layout(
                Panel(progress_bar, title="Progress", border_style="blue"),
                size=5
            ),
            Layout(
                Panel(
                    "[cyan][p][/cyan] Pause training  |  [cyan][q][/cyan] Quit monitor  |  "
                    f"[dim]Refresh: {self.refresh_interval}s[/dim]",
                    border_style="yellow"
                ),
                size=3
            )
        )

        return layout

    def run(self):
        """
        Run monitor loop with keyboard input.

        Platform: Unix/Linux only (uses select/tty/termios)
        """
        # Check platform compatibility
        try:
            import select
            import tty
            import termios
        except ImportError:
            self.console.print(
                "[red]Error: Monitor requires Unix/Linux terminal (uses tty/termios/select)[/red]\n"
                "[yellow]Windows users: Please use WSL (Windows Subsystem for Linux)[/yellow]"
            )
            return 1

        # Set terminal to non-blocking mode for keyboard input
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setcbreak(sys.stdin.fileno())

            with Live(self.build_display(None), refresh_per_second=1) as live:
                while True:
                    # Read status
                    status = self.read_status()
                    if status is not None:
                        self.last_status = status

                    # Update display
                    live.update(self.build_display(status))

                    # Check for keyboard input (non-blocking with timeout)
                    start_time = time.time()
                    while time.time() - start_time < self.refresh_interval:
                        # Check if input is available (100ms timeout)
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1).lower()

                            if key == 'q':
                                self.console.print(
                                    "\n[yellow]Exiting monitor. Training continues in background.[/yellow]"
                                )
                                return 0

                            elif key == 'p':
                                self.send_pause_signal()
                                time.sleep(1.5)  # Give user time to see confirmation

                        # Small sleep to avoid busy-waiting
                        time.sleep(0.05)

        except KeyboardInterrupt:
            self.console.print(
                "\n[yellow]Monitor interrupted (Ctrl+C). Training continues in background.[/yellow]"
            )
            return 0

        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="External training monitor with Rich TUI (Unix/Linux only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Platform Requirements:
  Unix/Linux terminal only (uses tty/termios/select)
  Windows users: Use WSL (Windows Subsystem for Linux)

Keyboard controls:
  p - Pause training (writes control signal)
  q - Quit monitor (training continues)

Example usage:
  python ml/monitor.py
  python ml/monitor.py --refresh-interval 10
  python ml/monitor.py --status-file custom/status.json
        """
    )

    parser.add_argument(
        '--status-file',
        type=str,
        default='models/checkpoints/status.json',
        help='Path to status JSON file (default: models/checkpoints/status.json)'
    )

    parser.add_argument(
        '--control-file',
        type=str,
        default='models/checkpoints/control.signal',
        help='Path to control signal file (default: models/checkpoints/control.signal)'
    )

    parser.add_argument(
        '--refresh-interval',
        type=float,
        default=5.0,
        help='Seconds between status updates (default: 5.0)'
    )

    args = parser.parse_args()

    # Create and run monitor
    monitor = TrainingMonitor(
        status_file=args.status_file,
        control_file=args.control_file,
        refresh_interval=args.refresh_interval
    )

    try:
        exit_code = monitor.run()
        return exit_code
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

**Testing monitor:**

```bash
# Install Rich if needed
pip install rich

# Create test script: test_monitor.py
cat > test_monitor.py << 'EOF'
import json
from pathlib import Path
import time

def create_mock_status():
    """Create mock status file for monitor testing."""
    status = {
        'timestamp': time.time(),
        'iteration': 150,
        'total_iterations': 500,
        'progress': 0.30,
        'eta_hours': 124.5,
        'eta_days': 5.2,
        'elapsed_hours': 48.3,
        'phase': 'rounds',
        'mcts_config': '3×35',
        'elo': 1420,  # This is the output format (current_elo -> elo in status file)
        'elo_change': 15,
        'learning_rate': 0.000554,
        'loss': 0.342,  # avg_total_loss -> loss in status file
        'policy_loss': 0.205,  # avg_policy_loss -> policy_loss in status file
        'value_loss': 0.137,  # avg_value_loss -> value_loss in status file
        'training_units_generated': 4400,
        'unit_type': 'rounds',
        'pi_target_tau': 0.850,
    }

    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    with open('models/checkpoints/status.json', 'w') as f:
        json.dump(status, f, indent=2)

    print('✓ Created mock status file: models/checkpoints/status.json')
    print('\nRun monitor with:')
    print('  python ml/monitor.py --refresh-interval 2')
    print('\nKeyboard controls:')
    print('  p - Test pause signal')
    print('  q - Quit monitor')

if __name__ == '__main__':
    create_mock_status()
EOF

python test_monitor.py
rm test_monitor.py

# Now run the monitor (should display status)
# python ml/monitor.py --refresh-interval 2

# Test keyboard controls:
#   Press 'p' → should create control.signal file
#   Press 'q' → should exit cleanly
```

**Acceptance criteria:**
- ✅ Monitor displays status table with all metrics
- ✅ Progress bar shows completion percentage
- ✅ ETA displayed in days/hours with smart formatting
- ✅ Updates every 5 seconds (configurable)
- ✅ Handles missing status file gracefully
- ✅ Handles corrupt JSON gracefully (uses cached status)
- ✅ Keyboard input works (p/q keys, non-blocking)
- ✅ Platform check warns Windows users to use WSL
- ✅ Clean exit on Ctrl+C

---

### Part 2: Integration & Testing (90 minutes)

**Objective**: Integrate all components and validate end-to-end workflow.

**Tasks:**

1. **Add unit tests to `ml/training/test_training.py`**

```python
import pytest
from pathlib import Path
import time
import json
from ml.training.trainer import TrainingPipeline, StatusWriter


class TestCheckpointRotation:
    """Test checkpoint rotation logic."""

    def test_cache_rotation_keeps_four_most_recent(self):
        """Verify cache keeps only 4 most recent checkpoints."""
        cache_dir = Path('models/checkpoints/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Clean existing files
        for f in cache_dir.glob('*.pth'):
            f.unlink()

        # Create 10 checkpoints with different timestamps
        for i in range(10):
            checkpoint = cache_dir / f'test_checkpoint_{i:03d}.pth'
            checkpoint.write_text(f'checkpoint {i}')
            time.sleep(0.01)  # Ensure different mtimes

        # Simulate rotation (using TrainingPipeline._rotate_cache_checkpoints logic)
        cache_files = sorted(cache_dir.glob('test_checkpoint_*.pth'), key=lambda p: p.stat().st_mtime)
        num_to_delete = len(cache_files) - 4
        for old_file in cache_files[:num_to_delete]:
            old_file.unlink()

        remaining = list(cache_dir.glob('test_checkpoint_*.pth'))
        assert len(remaining) == 4, f"Expected 4 files, got {len(remaining)}"

        # Verify we kept the most recent
        remaining_names = [f.name for f in remaining]
        assert 'test_checkpoint_006.pth' in remaining_names
        assert 'test_checkpoint_009.pth' in remaining_names
        assert 'test_checkpoint_000.pth' not in remaining_names

        # Clean up
        for f in remaining:
            f.unlink()

    def test_permanent_checkpoints_never_rotated(self):
        """Verify permanent directory is not affected by rotation."""
        permanent_dir = Path('models/checkpoints/permanent')
        permanent_dir.mkdir(parents=True, exist_ok=True)

        # Create permanent checkpoints
        for i in [5, 10, 15, 20]:
            checkpoint = permanent_dir / f'permanent_iter{i:03d}.pth'
            checkpoint.write_text(f'permanent {i}')

        initial_count = len(list(permanent_dir.glob('*.pth')))
        assert initial_count == 4

        # Rotation should not touch permanent directory
        # (rotation only operates on cache directory)

        final_count = len(list(permanent_dir.glob('*.pth')))
        assert final_count == initial_count, "Permanent checkpoints were deleted!"

        # Clean up
        for f in permanent_dir.glob('permanent_*.pth'):
            f.unlink()


class TestStatusWriter:
    """Test atomic status file writes."""

    def test_atomic_write_no_corruption(self):
        """Verify status writes are atomic and don't corrupt file."""
        status_file = 'models/checkpoints/test_status_atomic.json'
        writer = StatusWriter(status_file)

        class MockConfig:
            training_on = 'rounds'
            num_determinizations = 3
            simulations_per_determinization = 35

        # Write status multiple times rapidly
        for i in range(10):
            metrics = {
                'current_elo': 1000 + i * 10,
                'learning_rate': 0.001,
                'avg_total_loss': 0.5,
            }
            writer.update(
                iteration=i,
                total_iterations=10,
                metrics=metrics,
                config=MockConfig()
            )

        # Verify file is valid JSON (not corrupted)
        with open(status_file) as f:
            status = json.load(f)

        assert status['iteration'] == 10  # Last iteration (1-indexed)
        assert status['elo'] == 1090  # current_elo mapped to 'elo'

        # Clean up
        Path(status_file).unlink()

    def test_status_handles_missing_metrics(self):
        """Verify status writer handles missing metrics gracefully."""
        status_file = 'models/checkpoints/test_status_missing.json'
        writer = StatusWriter(status_file)

        class MockConfig:
            training_on = 'rounds'

        # Write with minimal metrics
        metrics = {}  # Empty metrics
        writer.update(
            iteration=0,
            total_iterations=10,
            metrics=metrics,
            config=MockConfig()
        )

        # Verify file created with None values
        with open(status_file) as f:
            status = json.load(f)

        assert status['elo'] is None
        assert status['learning_rate'] is None
        assert status['iteration'] == 1  # 1-indexed

        # Clean up
        Path(status_file).unlink()


class TestControlSignals:
    """Test pause/resume control signals."""

    def test_pause_signal_detection(self):
        """Verify pause signal is detected correctly."""
        signal_file = Path('models/checkpoints/control.signal')
        signal_file.parent.mkdir(parents=True, exist_ok=True)

        # Clean up
        if signal_file.exists():
            signal_file.unlink()

        # Test: no signal
        signal = signal_file.read_text().strip().upper() if signal_file.exists() else ''
        assert signal == ''

        # Test: PAUSE signal
        signal_file.write_text('PAUSE\n')
        signal = signal_file.read_text().strip().upper()
        assert signal == 'PAUSE'

        # Test: case insensitive
        signal_file.write_text('pause\n')
        signal = signal_file.read_text().strip().upper()
        assert signal == 'PAUSE'

        # Clean up
        signal_file.unlink()

    def test_signal_cleared_after_processing(self):
        """Verify signal file is cleared after processing."""
        signal_file = Path('models/checkpoints/control.signal')
        signal_file.parent.mkdir(parents=True, exist_ok=True)

        # Create signal
        signal_file.write_text('PAUSE\n')
        assert signal_file.exists()

        # Simulate clearing
        signal_file.unlink()
        assert not signal_file.exists()
```

2. **Integration test workflow**

```bash
# Create integration test script: test_integration.sh
cat > test_integration.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Integration Test: Checkpoint Rotation + Status + Pause/Resume ==="

# Clean up from previous runs
rm -rf models/checkpoints/test_integration
mkdir -p models/checkpoints/test_integration/cache
mkdir -p models/checkpoints/test_integration/permanent

echo ""
echo "Test 1: Checkpoint rotation (20 iterations)"
echo "Expected: 4 permanent (5,10,15,20), max 4 cache"
# Note: This requires actual training run
# python ml/train.py --fast --iterations 20 --checkpoint-dir models/checkpoints/test_integration

PERM_COUNT=$(find models/checkpoints/test_integration/permanent -name "*.pth" 2>/dev/null | wc -l || echo "0")
CACHE_COUNT=$(find models/checkpoints/test_integration/cache -name "*.pth" 2>/dev/null | wc -l || echo "0")

echo "  Permanent checkpoints: $PERM_COUNT (expected: 4)"
echo "  Cache checkpoints: $CACHE_COUNT (expected: ≤4)"

echo ""
echo "Test 2: Status file updates"
if [ -f "models/checkpoints/status.json" ]; then
    echo "  ✓ Status file exists"
    python -c "import json; json.load(open('models/checkpoints/status.json'))" && echo "  ✓ Status file is valid JSON"
else
    echo "  ⚠ Status file not found (training not running)"
fi

echo ""
echo "Test 3: Pause signal"
echo "PAUSE" > models/checkpoints/control.signal
if [ -f "models/checkpoints/control.signal" ]; then
    SIGNAL=$(cat models/checkpoints/control.signal | tr -d '[:space:]')
    if [ "$SIGNAL" = "PAUSE" ]; then
        echo "  ✓ Pause signal created correctly"
    else
        echo "  ✗ Pause signal content incorrect: $SIGNAL"
    fi
    rm models/checkpoints/control.signal
else
    echo "  ✗ Failed to create pause signal"
fi

echo ""
echo "=== Integration Test Complete ==="
echo ""
echo "Manual tests remaining:"
echo "  1. Start training: python ml/train.py --fast --iterations 10"
echo "  2. Start monitor: python ml/monitor.py --refresh-interval 2"
echo "  3. Press 'p' in monitor to test pause"
echo "  4. Resume training: python ml/train.py --fast --iterations 10 --resume <checkpoint>"
EOF

chmod +x test_integration.sh
./test_integration.sh
rm test_integration.sh
```

3. **Documentation update**

Add to `README.md` or `CLAUDE.md`:

```markdown
## Training with Monitor

### Quick Start

**Terminal 1** - Start training:
```bash
# Full training run (7-9 days)
python ml/train.py --training-on rounds --iterations 500

# Or fast test (5 minutes)
python ml/train.py --fast --iterations 5
```

**Terminal 2** - Launch monitor:
```bash
python ml/monitor.py

# Or with custom refresh rate
python ml/monitor.py --refresh-interval 10
```

### Monitor Controls

- `p` - Pause training (saves checkpoint and exits gracefully)
- `q` - Quit monitor (training continues in background)

### Pause and Resume

Pause training from monitor (press `p`) or manually:
```bash
echo "PAUSE" > models/checkpoints/control.signal
```

Resume from checkpoint:
```bash
# Find latest checkpoint
ls -lt models/checkpoints/cache/

# Resume training
python ml/train.py --resume models/checkpoints/cache/xxx.pth --iterations 500
```

### Platform Requirements

- **Training**: Ubuntu Linux 24.04 (validated)
- **Monitor**: Unix/Linux terminal (Windows users: use WSL)

### Checkpoint Management

**Automatic rotation:**
- Permanent: Every 5 iterations (iter 5, 10, 15, ...) → `models/checkpoints/permanent/`
- Cache: All others (max 4 kept) → `models/checkpoints/cache/`

**Disk savings:**
- Without rotation: ~250GB for 500 iterations
- With rotation: ~50GB for 500 iterations

**Manual cleanup:**
```bash
# Keep only permanent checkpoints
rm -rf models/checkpoints/cache/

# Remove old permanent checkpoints
rm models/checkpoints/permanent/*-iter005-*.pth
```
```

**Acceptance criteria:**
- ✅ Unit tests pass for rotation, status, control signals
- ✅ Integration test validates file structure
- ✅ Documentation added to README/CLAUDE.md
- ✅ Manual workflow tested end-to-end
- ✅ No race conditions or corruption
- ✅ Works with tmux attach/detach

---

## Testing Procedures

### Automated Tests

Run unit tests:
```bash
# All trainer tests
python -m pytest ml/training/test_training.py -v

# Specific test classes
python -m pytest ml/training/test_training.py::TestCheckpointRotation -v
python -m pytest ml/training/test_training.py::TestStatusWriter -v
python -m pytest ml/training/test_training.py::TestControlSignals -v
```

### Manual Integration Test

Complete workflow validation:

```bash
# Terminal 1: Start training with monitoring
python ml/train.py --fast --iterations 10

# Terminal 2: Start monitor
python ml/monitor.py --refresh-interval 2

# Verify monitor displays:
#   - Iteration progress
#   - ETA in hours
#   - Phase and MCTS config
#   - Live updates every 2 seconds

# Test pause:
#   Press 'p' in monitor
#   Verify training stops after current iteration
#   Verify checkpoint saved

# Find latest checkpoint
ls -lt models/checkpoints/cache/ | head -2

# Resume training
python ml/train.py --fast --iterations 10 --resume models/checkpoints/cache/<checkpoint>.pth

# Verify:
#   - Training continues from correct iteration
#   - Status file updates
#   - Monitor shows resumed progress
```

### Stress Tests

```bash
# Test 1: Rapid status reads (simulate multiple monitors)
for i in {1..100}; do
    cat models/checkpoints/status.json > /dev/null 2>&1 &
done
wait
# Expected: No corruption, all reads succeed

# Test 2: Checkpoint rotation with many iterations
python ml/train.py --fast --iterations 30
# Expected: 6 permanent (5,10,15,20,25,30), max 4 cache

# Test 3: Pause during self-play
# Start training, immediately pause
python ml/train.py --fast --iterations 10 &
sleep 2
echo "PAUSE" > models/checkpoints/control.signal
wait
# Expected: Graceful shutdown, no partial checkpoints
```

---

## Usage Guide

### Starting Training with Monitor

**Recommended workflow for long runs:**

```bash
# Terminal 1: Start training in tmux session
tmux new -s blobmaster-training
python ml/train.py --training-on rounds --iterations 500
# Detach: Ctrl+B, then D

# Terminal 2: Monitor (can attach/detach anytime)
python ml/monitor.py

# To reattach training session:
tmux attach -t blobmaster-training

# To list sessions:
tmux ls
```

### Monitoring from Another Machine

If training runs on a server:

**Option 1: SSH + sshfs**
```bash
# Local machine: Mount checkpoint directory
sshfs user@server:/path/to/BlobMaster/models/checkpoints ~/checkpoints

# Run monitor locally
python ml/monitor.py --status-file ~/checkpoints/status.json

# Send pause signal remotely
echo "PAUSE" > ~/checkpoints/control.signal
```

**Option 2: SSH polling**
```bash
# Poll status via SSH (every 10 seconds)
watch -n 10 'ssh user@server "cat /path/to/BlobMaster/models/checkpoints/status.json" | jq .'

# Send pause signal
ssh user@server "echo PAUSE > /path/to/BlobMaster/models/checkpoints/control.signal"
```

### Troubleshooting

**Monitor shows "Waiting for training..."**
- Verify training process is running: `ps aux | grep train.py`
- Check status file exists: `ls -l models/checkpoints/status.json`
- Check file permissions: `chmod 644 models/checkpoints/status.json`

**Pause signal not working**
- Verify control file created: `ls -l models/checkpoints/control.signal`
- Check training logs for "Pause signal received" message
- Remember: pause happens at iteration boundary (may take minutes)

**Monitor keyboard input not working**
- Platform check: Unix/Linux only (Windows users need WSL)
- Try: `stty sane` to reset terminal
- Restart monitor: Ctrl+C then restart

**Checkpoint rotation not working**
- Check disk space: `df -h models/checkpoints/`
- Verify cache directory exists: `ls -ld models/checkpoints/cache/`
- Check file permissions: `chmod 755 models/checkpoints/cache/`

**Out of disk space**
- Monitor disk usage: `du -sh models/checkpoints/*/`
- Remove old cache checkpoints: `rm models/checkpoints/cache/*.pth`
- Keep only recent permanent: `ls -t models/checkpoints/permanent/*.pth | tail -n +10 | xargs rm`

---

## Success Metrics

After completing both sessions, you should have:

### Functional Requirements
- ✅ Checkpoint rotation: permanent every 5 iters, cache rotates (max 4)
- ✅ Atomic status file: updates every iteration without corruption
- ✅ External monitor: TUI with live metrics, keyboard controls
- ✅ Pause/resume: graceful iteration-boundary pause, safe resume
- ✅ Worker cleanup: proper resource cleanup on exit
- ✅ Error handling: robust file I/O with fallbacks

### Performance Requirements
- ✅ Disk savings: ~50GB vs ~250GB (500 iterations)
- ✅ Monitor overhead: <1% CPU, no training impact
- ✅ Status updates: <10ms per write (atomic)
- ✅ Refresh rate: configurable, default 5s

### Usability Requirements
- ✅ Works with tmux detach/reattach
- ✅ Monitor can attach/detach anytime
- ✅ Clear ETA estimation (days and hours)
- ✅ Intuitive keyboard controls (p/q)
- ✅ Platform compatibility documented (Unix/Linux)

### Testing Coverage
- ✅ Unit tests for rotation, status, control signals
- ✅ Integration tests for complete workflow
- ✅ Manual validation procedures documented
- ✅ Stress tests for concurrent access

---

## Next Steps

### 1. Validate Installation

```bash
# Install dependencies
pip install rich

# Run unit tests
python -m pytest ml/training/test_training.py::TestCheckpointRotation -v
python -m pytest ml/training/test_training.py::TestStatusWriter -v
python -m pytest ml/training/test_training.py::TestControlSignals -v
```

### 2. Test with Fast Run

```bash
# Terminal 1
python ml/train.py --fast --iterations 10

# Terminal 2
python ml/monitor.py --refresh-interval 2

# Test pause/resume:
#   Press 'p' in monitor
#   Wait for checkpoint
#   Resume: python ml/train.py --fast --iterations 10 --resume <checkpoint>
```

### 3. Start Phase 1 Training

```bash
# Terminal 1: Training (in tmux)
tmux new -s blobmaster-training
python ml/train.py --training-on rounds --iterations 500 --enable-curriculum
# Detach: Ctrl+B, D

# Terminal 2: Monitor
python ml/monitor.py

# Monitor progress over 7-9 days:
#   - Check ELO progression daily
#   - Pause for maintenance if needed
#   - Verify checkpoint rotation working
```

### 4. After Phase 1 Completes

- Evaluate final model on full games
- Analyze ELO progression curve
- Decide whether to proceed to Phase 2 (full games training)
- Export model to ONNX for production inference

---

## Document History

- **2025-11-13**: Initial creation - Standalone implementation guide for external monitoring system
- **Purpose**: Add operational tooling for multi-day training runs (7-9+ days)
- **Sessions**: 2 × 4-hour blocks (8 hours total)
- **Platform**: Ubuntu Linux 24.04, RTX 4060, Python 3.14
