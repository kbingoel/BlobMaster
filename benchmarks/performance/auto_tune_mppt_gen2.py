#!/usr/bin/env python3
"""
MPPT-Style Auto-Tune for BlobMaster Training Curriculum (Generation 2)

This script performs gradient-ascent (MPPT-style) optimization to find the best
configuration for each MCTS curriculum stage. Gen2 improvements:
- Workers fixed at 32 (removed from optimization)
- Hill-climbing fine search with verification (continues until true peak)
- Faster evaluation (200 rounds for all tests - consistent comparison)
- Batch size coarse step reduced to ±5

Key Features:
- Optimizes for each curriculum stage (1×15, 2×25, 3×35, 4×45, 5×50)
- Coarse search: ±5/±1 steps → Fine hill-climbing until drop + verification
- Tests parameter combinations to detect interactions
- No artificial limits - let hardware define boundaries
- Checkpoint/resume capability

Usage:
    python benchmarks/performance/auto_tune_mppt_gen2.py
    python benchmarks/performance/auto_tune_mppt_gen2.py --resume results/auto_tune_mppt_gen2_20251114
    python benchmarks/performance/auto_tune_mppt_gen2.py --stages 1,2,3  # Only stages 1-3
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import os

import torch
import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class MPPTConfig:
    """Configuration for MPPT optimization"""
    workers: int
    parallel_batch_size: int
    num_determinizations: int
    simulations_per_det: int
    batch_timeout_ms: int = 10

    def to_hash(self) -> str:
        """Generate MD5 hash for deduplication"""
        config_str = f"{self.workers}_{self.parallel_batch_size}_{self.num_determinizations}_{self.simulations_per_det}_{self.batch_timeout_ms}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def total_sims_per_move(self) -> int:
        """Calculate total MCTS simulations per move"""
        return self.num_determinizations * self.simulations_per_det

    def __str__(self) -> str:
        """Human-readable representation"""
        return f"{self.workers}w × {self.parallel_batch_size}batch × {self.num_determinizations}×{self.simulations_per_det} × {self.batch_timeout_ms}ms"


# ============================================================================
# Curriculum Definition
# ============================================================================

# MCTS curriculum stages from ml/config.py
CURRICULUM_STAGES = [
    (1, 15, "Stage 1: Early Training"),
    (2, 25, "Stage 2: Mid-Early Training"),
    (3, 35, "Stage 3: Mid Training"),
    (4, 45, "Stage 4: Late-Mid Training"),
    (5, 50, "Stage 5: Final Training")
]

# Baseline configuration
BASELINE_WORKERS = 32  # Fixed - not optimized in gen2
BASELINE_BATCH_SIZE = 30
BASELINE_TIMEOUT_MS = 10

# Step sizes for coarse and fine search (gen2: workers removed, batch_size reduced)
PARAM_STEPS = {
    'parallel_batch_size': {'coarse': 5, 'fine': 5},  # Reduced from 10
    'batch_timeout_ms': {'coarse': 5, 'fine': 1}
}

# Minimum values (hardware/logic constraints)
PARAM_MINIMUMS = {
    'parallel_batch_size': 1,
    'batch_timeout_ms': 1
}


# ============================================================================
# JSON-based Checkpoint Management
# ============================================================================

class MPPTCheckpointManager:
    """Manages JSON files for MPPT results and checkpoints"""

    def __init__(self, base_path: str):
        self.base_dir = Path(base_path).parent
        self.experiments_file = self.base_dir / "experiments.json"
        self.checkpoint_file = self.base_dir / "checkpoint.json"

        # Ensure directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data or initialize
        self.experiments = self._load_json(self.experiments_file, [])
        self.checkpoint_data = self._load_json(self.checkpoint_file, None)

    def _load_json(self, path: Path, default):
        """Load JSON file or return default if not exists"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                return default
        return default

    def _save_json(self, path: Path, data):
        """Save data to JSON file"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_result(self, stage_num: int, phase: str, config: MPPTConfig,
                    num_rounds: int, elapsed_sec: float,
                    rounds_per_min: Optional[float],
                    success: bool, error_msg: Optional[str] = None,
                    variance: Optional[float] = None,
                    examples_per_round: Optional[float] = None,
                    cpu_percent: Optional[float] = None,
                    gpu_memory_mb: Optional[float] = None,
                    param_name: Optional[str] = None,
                    search_type: Optional[str] = None):
        """Save experiment result to JSON file"""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'stage_num': stage_num,
            'stage_mcts': f"{config.num_determinizations}×{config.simulations_per_det}",
            'phase': phase,
            'param_name': param_name,
            'search_type': search_type,
            'config_hash': config.to_hash(),
            'num_rounds': num_rounds,
            'elapsed_sec': elapsed_sec,
            'rounds_per_min': rounds_per_min,
            'variance': variance,
            'success': success,
            'error_msg': error_msg,
            'workers': config.workers,
            'parallel_batch_size': config.parallel_batch_size,
            'num_determinizations': config.num_determinizations,
            'simulations_per_det': config.simulations_per_det,
            'batch_timeout_ms': config.batch_timeout_ms,
            'examples_per_round': examples_per_round,
            'cpu_percent': cpu_percent,
            'gpu_memory_mb': gpu_memory_mb
        }

        self.experiments.append(experiment)
        self._save_json(self.experiments_file, self.experiments)

    def save_checkpoint(self, stage_num: int, phase: str,
                       last_config: MPPTConfig, best_config: MPPTConfig,
                       best_perf: float):
        """Save checkpoint for resume capability"""
        self.checkpoint_data = {
            'stage_num': stage_num,
            'last_phase': phase,
            'last_config': last_config.to_dict(),
            'current_best_config': best_config.to_dict(),
            'current_best_perf': best_perf,
            'timestamp': datetime.now().isoformat()
        }
        self._save_json(self.checkpoint_file, self.checkpoint_data)

    def load_checkpoint(self) -> Optional[Tuple[int, str, MPPTConfig, MPPTConfig, float]]:
        """Load last checkpoint if exists"""
        if self.checkpoint_data:
            stage_num = self.checkpoint_data['stage_num']
            last_phase = self.checkpoint_data['last_phase']
            last_config = MPPTConfig(**self.checkpoint_data['last_config'])
            best_config = MPPTConfig(**self.checkpoint_data['current_best_config'])
            best_perf = self.checkpoint_data['current_best_perf']
            return stage_num, last_phase, last_config, best_config, best_perf
        return None

    def get_completed_configs(self, stage_num: Optional[int] = None,
                            phase: Optional[str] = None) -> Set[str]:
        """Get hashes of completed configurations"""
        hashes = set()
        for exp in self.experiments:
            if exp['success']:
                if (stage_num is None or exp['stage_num'] == stage_num) and \
                   (phase is None or exp['phase'] == phase):
                    hashes.add(exp['config_hash'])
        return hashes

    def get_stage_results(self, stage_num: int, phase: Optional[str] = None) -> List[Dict]:
        """Get all successful results for a specific stage"""
        results = []
        for exp in self.experiments:
            if exp['success'] and exp['rounds_per_min'] is not None and \
               exp['stage_num'] == stage_num:
                if phase is None or exp['phase'] == phase:
                    config = MPPTConfig(
                        workers=exp['workers'],
                        parallel_batch_size=exp['parallel_batch_size'],
                        num_determinizations=exp['num_determinizations'],
                        simulations_per_det=exp['simulations_per_det'],
                        batch_timeout_ms=exp['batch_timeout_ms']
                    )
                    results.append({
                        'config': config,
                        'rounds_per_min': exp['rounds_per_min'],
                        'variance': exp.get('variance', 0.0),
                        'num_rounds': exp['num_rounds'],
                        'param_name': exp.get('param_name'),
                        'search_type': exp.get('search_type')
                    })

        results.sort(key=lambda x: x['rounds_per_min'], reverse=True)
        return results

    def close(self):
        """No-op for JSON-based manager"""
        pass


# ============================================================================
# Network Creation
# ============================================================================

def create_network(device: str) -> Tuple[BlobNet, StateEncoder, ActionMasker]:
    """Create neural network with encoder and masker"""
    encoder = StateEncoder()
    masker = ActionMasker()

    network = BlobNet(
        state_dim=encoder.state_dim,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,
        max_bid=13,
        max_cards=52
    ).to(device)

    network.eval()
    return network, encoder, masker


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_benchmark(
    config: MPPTConfig,
    num_rounds: int,
    device: str,
    warmup_rounds: int = 5,
    cards_to_deal: int = 5,
    num_players: int = 4
) -> Dict:
    """
    Run benchmark for a single configuration

    Returns dict with:
        - success: bool
        - rounds_per_min: float (if successful)
        - elapsed_sec: float
        - error: str (if failed)
        - examples_per_round: float
        - cpu_percent: float
        - gpu_memory_mb: float
    """
    try:
        # Create network
        network, encoder, masker = create_network(device)

        # Create self-play engine
        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=config.workers,
            num_determinizations=config.num_determinizations,
            simulations_per_determinization=config.simulations_per_det,
            device=device,
            use_batched_evaluator=True,
            batch_size=512,
            batch_timeout_ms=config.batch_timeout_ms,
            use_thread_pool=False,
            use_parallel_expansion=True,
            parallel_batch_size=config.parallel_batch_size
        )

        # Warmup
        print(f"  Warming up ({warmup_rounds} rounds)...", end='', flush=True)
        engine.generate_games(
            num_games=warmup_rounds,
            num_players=num_players,
            cards_to_deal=cards_to_deal
        )
        torch.cuda.empty_cache()
        print(" done")

        # Benchmark
        print(f"  Benchmarking ({num_rounds} rounds)...", end='', flush=True)
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)

        examples = engine.generate_games(
            num_games=num_rounds,
            num_players=num_players,
            cards_to_deal=cards_to_deal
        )

        elapsed_sec = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=None)
        cpu_percent = (cpu_before + cpu_after) / 2

        # Get GPU memory usage
        gpu_memory_mb = 0.0
        if device == 'cuda' and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        rounds_per_min = (num_rounds / elapsed_sec) * 60
        examples_per_round = len(examples) / num_rounds if examples else 0.0

        print(f" {rounds_per_min:.1f} r/min")

        # Cleanup
        engine.shutdown()
        torch.cuda.empty_cache()
        time.sleep(2)

        return {
            'success': True,
            'rounds_per_min': rounds_per_min,
            'elapsed_sec': elapsed_sec,
            'examples_per_round': examples_per_round,
            'cpu_percent': cpu_percent,
            'gpu_memory_mb': gpu_memory_mb
        }

    except torch.cuda.OutOfMemoryError as e:
        error_msg = f"CUDA OOM: {str(e)}"
        print(f" FAILED: {error_msg}")
        torch.cuda.empty_cache()
        return {
            'success': False,
            'error': error_msg,
            'elapsed_sec': 0.0
        }

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f" FAILED: {error_msg}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {
            'success': False,
            'error': error_msg,
            'elapsed_sec': 0.0
        }


def run_multi_benchmark(
    config: MPPTConfig,
    num_runs: int,
    rounds_per_run: int,
    device: str
) -> Dict:
    """
    Run config multiple times and return average with variance

    Returns dict with:
        - success: bool
        - rounds_per_min: float (average)
        - variance: float (std dev as percentage)
        - elapsed_sec: float (total)
        - runs: List[float] (individual rounds_per_min)
    """
    results = []
    total_elapsed = 0.0

    print(f"  Running {num_runs} validation runs...")

    for run_idx in range(num_runs):
        print(f"    Run {run_idx + 1}/{num_runs}:", end=' ')
        result = run_benchmark(config, rounds_per_run, device)

        if not result['success']:
            return result

        results.append(result['rounds_per_min'])
        total_elapsed += result['elapsed_sec']

    # Calculate statistics
    avg = sum(results) / len(results)
    variance_val = (sum((x - avg) ** 2 for x in results) / len(results)) ** 0.5
    variance_pct = (variance_val / avg) * 100

    print(f"  Average: {avg:.1f} r/min, Variance: {variance_pct:.1f}%")

    return {
        'success': True,
        'rounds_per_min': avg,
        'variance': variance_pct,
        'elapsed_sec': total_elapsed,
        'runs': results
    }


# ============================================================================
# MPPT Optimization for Single Stage (Gen2: Hill-Climbing Fine Search)
# ============================================================================

def optimize_parameter_mppt(
    param_name: str,
    baseline_config: MPPTConfig,
    baseline_perf: float,
    checkpoint_mgr: MPPTCheckpointManager,
    stage_num: int,
    device: str
) -> Tuple[int, float]:
    """
    MPPT optimization for a single parameter using coarse-to-fine search
    Gen2: Fine search uses hill-climbing with verification

    Returns (optimal_value, optimal_perf)
    """
    print(f"\n--- Optimizing: {param_name} ---")

    baseline_value = getattr(baseline_config, param_name)
    coarse_step = PARAM_STEPS[param_name]['coarse']
    fine_step = PARAM_STEPS[param_name]['fine']
    min_value = PARAM_MINIMUMS[param_name]

    print(f"Baseline: {baseline_value}, Coarse step: ±{coarse_step}, Fine step: ±{fine_step}")

    # Phase 1: Coarse search - test ±coarse_step
    print(f"\n[Coarse Search] Testing ±{coarse_step} from baseline...")

    up_value = baseline_value + coarse_step
    down_value = max(min_value, baseline_value - coarse_step)

    # Test up
    config_up = MPPTConfig(
        workers=BASELINE_WORKERS,  # Fixed at 32
        parallel_batch_size=baseline_config.parallel_batch_size,
        num_determinizations=baseline_config.num_determinizations,
        simulations_per_det=baseline_config.simulations_per_det,
        batch_timeout_ms=baseline_config.batch_timeout_ms
    )
    setattr(config_up, param_name, up_value)

    print(f"  Testing {param_name}={up_value} (+{coarse_step}):")
    result_up = run_benchmark(config_up, num_rounds=100, device=device)
    checkpoint_mgr.save_result(
        stage_num=stage_num,
        phase='individual_coarse',
        config=config_up,
        num_rounds=100,
        elapsed_sec=result_up['elapsed_sec'],
        rounds_per_min=result_up.get('rounds_per_min'),
        success=result_up['success'],
        error_msg=result_up.get('error'),
        param_name=param_name,
        search_type='coarse_up'
    )

    perf_up = result_up['rounds_per_min'] if result_up['success'] else 0.0

    # Test down
    config_down = MPPTConfig(
        workers=BASELINE_WORKERS,  # Fixed at 32
        parallel_batch_size=baseline_config.parallel_batch_size,
        num_determinizations=baseline_config.num_determinizations,
        simulations_per_det=baseline_config.simulations_per_det,
        batch_timeout_ms=baseline_config.batch_timeout_ms
    )
    setattr(config_down, param_name, down_value)

    print(f"  Testing {param_name}={down_value} (-{coarse_step}):")
    result_down = run_benchmark(config_down, num_rounds=100, device=device)
    checkpoint_mgr.save_result(
        stage_num=stage_num,
        phase='individual_coarse',
        config=config_down,
        num_rounds=100,
        elapsed_sec=result_down['elapsed_sec'],
        rounds_per_min=result_down.get('rounds_per_min'),
        success=result_down['success'],
        error_msg=result_down.get('error'),
        param_name=param_name,
        search_type='coarse_down'
    )

    perf_down = result_down['rounds_per_min'] if result_down['success'] else 0.0

    # Determine direction
    print(f"\n  Baseline: {baseline_perf:.1f} r/min")
    print(f"  Up (+{coarse_step}): {perf_up:.1f} r/min")
    print(f"  Down (-{coarse_step}): {perf_down:.1f} r/min")

    # Follow better direction
    best_value = baseline_value
    best_perf = baseline_perf

    if perf_up > baseline_perf and perf_up > perf_down:
        direction = 'up'
        current_value = up_value
        current_perf = perf_up
        print(f"  → Following UP direction")
    elif perf_down > baseline_perf and perf_down > perf_up:
        direction = 'down'
        current_value = down_value
        current_perf = perf_down
        print(f"  → Following DOWN direction")
    else:
        direction = None
        print(f"  → No improvement in coarse search, keeping baseline")

    # Continue in promising direction
    if direction:
        print(f"\n[Coarse Hill-Climbing] Following {direction} direction...")
        best_value = current_value
        best_perf = current_perf

        while True:
            # Calculate next value
            if direction == 'up':
                next_value = current_value + coarse_step
            else:
                next_value = max(min_value, current_value - coarse_step)

            # Stop if we've hit minimum and going down
            if direction == 'down' and next_value <= min_value and current_value == min_value:
                print(f"  Reached minimum value ({min_value}), stopping")
                break

            # Test next value
            config_next = MPPTConfig(
                workers=BASELINE_WORKERS,  # Fixed at 32
                parallel_batch_size=baseline_config.parallel_batch_size,
                num_determinizations=baseline_config.num_determinizations,
                simulations_per_det=baseline_config.simulations_per_det,
                batch_timeout_ms=baseline_config.batch_timeout_ms
            )
            setattr(config_next, param_name, next_value)

            print(f"  Testing {param_name}={next_value}:")
            result_next = run_benchmark(config_next, num_rounds=100, device=device)
            checkpoint_mgr.save_result(
                stage_num=stage_num,
                phase='individual_coarse',
                config=config_next,
                num_rounds=100,
                elapsed_sec=result_next['elapsed_sec'],
                rounds_per_min=result_next.get('rounds_per_min'),
                success=result_next['success'],
                error_msg=result_next.get('error'),
                param_name=param_name,
                search_type=f'coarse_{direction}'
            )

            if not result_next['success']:
                print(f"    Failed: {result_next['error']}")
                print(f"  Stopping at {param_name}={current_value}")
                break

            perf_next = result_next['rounds_per_min']

            if perf_next <= current_perf:
                print(f"    Performance dropped ({perf_next:.1f} <= {current_perf:.1f})")
                print(f"  Stopping at {param_name}={current_value}")
                break
            else:
                print(f"    Improved! {perf_next:.1f} > {current_perf:.1f}")
                current_value = next_value
                current_perf = perf_next
                best_value = current_value
                best_perf = current_perf

    print(f"\n[Coarse Result] Best: {param_name}={best_value}, Perf: {best_perf:.1f} r/min")

    # Phase 2: Fine search with hill-climbing (Gen2: continues until drop + verification)
    print(f"\n[Fine Search with Hill-Climbing] Testing around {best_value}...")

    # Start by testing ±fine_step from coarse optimum (200 rounds each)
    fine_down = max(min_value, best_value - fine_step)
    fine_up = best_value + fine_step

    # Track all tested values
    tested_values = {}  # value -> perf

    # Test down
    if fine_down != best_value:
        config_fine_down = MPPTConfig(
            workers=BASELINE_WORKERS,  # Fixed at 32
            parallel_batch_size=baseline_config.parallel_batch_size,
            num_determinizations=baseline_config.num_determinizations,
            simulations_per_det=baseline_config.simulations_per_det,
            batch_timeout_ms=baseline_config.batch_timeout_ms
        )
        setattr(config_fine_down, param_name, fine_down)

        print(f"  Testing {param_name}={fine_down} (-{fine_step}):")
        result = run_benchmark(config_fine_down, num_rounds=200, device=device)
        checkpoint_mgr.save_result(
            stage_num=stage_num,
            phase='individual_fine',
            config=config_fine_down,
            num_rounds=200,
            elapsed_sec=result['elapsed_sec'],
            rounds_per_min=result.get('rounds_per_min'),
            success=result['success'],
            error_msg=result.get('error'),
            param_name=param_name,
            search_type='fine_initial_down'
        )

        if result['success']:
            tested_values[fine_down] = result['rounds_per_min']

    # Validate current (coarse optimum) with 200 rounds
    print(f"  Validating {param_name}={best_value} (coarse optimum):")
    config_current = MPPTConfig(
        workers=BASELINE_WORKERS,  # Fixed at 32
        parallel_batch_size=baseline_config.parallel_batch_size,
        num_determinizations=baseline_config.num_determinizations,
        simulations_per_det=baseline_config.simulations_per_det,
        batch_timeout_ms=baseline_config.batch_timeout_ms
    )
    setattr(config_current, param_name, best_value)

    result = run_benchmark(config_current, num_rounds=200, device=device)
    checkpoint_mgr.save_result(
        stage_num=stage_num,
        phase='individual_fine',
        config=config_current,
        num_rounds=200,
        elapsed_sec=result['elapsed_sec'],
        rounds_per_min=result.get('rounds_per_min'),
        success=result['success'],
        error_msg=result.get('error'),
        param_name=param_name,
        search_type='fine_initial_center'
    )

    if result['success']:
        tested_values[best_value] = result['rounds_per_min']
        # Update best_perf with validated value
        best_perf = result['rounds_per_min']

    # Test up
    config_fine_up = MPPTConfig(
        workers=BASELINE_WORKERS,  # Fixed at 32
        parallel_batch_size=baseline_config.parallel_batch_size,
        num_determinizations=baseline_config.num_determinizations,
        simulations_per_det=baseline_config.simulations_per_det,
        batch_timeout_ms=baseline_config.batch_timeout_ms
    )
    setattr(config_fine_up, param_name, fine_up)

    print(f"  Testing {param_name}={fine_up} (+{fine_step}):")
    result = run_benchmark(config_fine_up, num_rounds=200, device=device)
    checkpoint_mgr.save_result(
        stage_num=stage_num,
        phase='individual_fine',
        config=config_fine_up,
        num_rounds=200,
        elapsed_sec=result['elapsed_sec'],
        rounds_per_min=result.get('rounds_per_min'),
        success=result['success'],
        error_msg=result.get('error'),
        param_name=param_name,
        search_type='fine_initial_up'
    )

    if result['success']:
        tested_values[fine_up] = result['rounds_per_min']

    # Determine direction for hill-climbing
    if not tested_values:
        print(f"\n[Fine Result] All tests failed, keeping coarse optimum")
        return best_value, best_perf

    # Find current best from initial tests
    current_best_value = max(tested_values.keys(), key=lambda v: tested_values[v])
    current_best_perf = tested_values[current_best_value]

    print(f"\n  Initial fine search results:")
    for val in sorted(tested_values.keys()):
        perf = tested_values[val]
        marker = "★" if val == current_best_value else " "
        print(f"    {marker} {param_name}={val}: {perf:.1f} r/min")

    # Determine hill-climbing direction
    if current_best_value == fine_down:
        fine_direction = 'down'
        print(f"\n  → Hill-climbing DOWN from {current_best_value}")
    elif current_best_value == fine_up:
        fine_direction = 'up'
        print(f"\n  → Hill-climbing UP from {current_best_value}")
    else:
        fine_direction = None
        print(f"\n  → Peak found at {current_best_value}, no hill-climbing needed")

    # Hill-climb in the promising direction
    if fine_direction:
        current_value = current_best_value
        current_perf = current_best_perf
        best_value = current_value
        best_perf = current_perf

        while True:
            # Calculate next value
            if fine_direction == 'up':
                next_value = current_value + fine_step
            else:
                next_value = max(min_value, current_value - fine_step)

            # Stop if we've hit minimum and going down
            if fine_direction == 'down' and next_value <= min_value and current_value == min_value:
                print(f"  Reached minimum value ({min_value}), stopping")
                break

            # Test next value
            config_next = MPPTConfig(
                workers=BASELINE_WORKERS,  # Fixed at 32
                parallel_batch_size=baseline_config.parallel_batch_size,
                num_determinizations=baseline_config.num_determinizations,
                simulations_per_det=baseline_config.simulations_per_det,
                batch_timeout_ms=baseline_config.batch_timeout_ms
            )
            setattr(config_next, param_name, next_value)

            print(f"  Testing {param_name}={next_value}:")
            result_next = run_benchmark(config_next, num_rounds=200, device=device)
            checkpoint_mgr.save_result(
                stage_num=stage_num,
                phase='individual_fine',
                config=config_next,
                num_rounds=200,
                elapsed_sec=result_next['elapsed_sec'],
                rounds_per_min=result_next.get('rounds_per_min'),
                success=result_next['success'],
                error_msg=result_next.get('error'),
                param_name=param_name,
                search_type=f'fine_climb_{fine_direction}'
            )

            if not result_next['success']:
                print(f"    Failed: {result_next['error']}")
                print(f"  Stopping at {param_name}={current_value} (error encountered)")
                break

            perf_next = result_next['rounds_per_min']

            # Check if performance dropped
            if perf_next < current_perf:
                print(f"    Performance dropped ({perf_next:.1f} < {current_perf:.1f})")
                print(f"  Verifying with one more step...")

                # Verification step
                if fine_direction == 'up':
                    verify_value = next_value + fine_step
                else:
                    verify_value = max(min_value, next_value - fine_step)

                # Skip verification if we hit minimum
                if fine_direction == 'down' and verify_value <= min_value and next_value == min_value:
                    print(f"    At minimum, confirmed peak at {param_name}={current_value}")
                    break

                config_verify = MPPTConfig(
                    workers=BASELINE_WORKERS,  # Fixed at 32
                    parallel_batch_size=baseline_config.parallel_batch_size,
                    num_determinizations=baseline_config.num_determinizations,
                    simulations_per_det=baseline_config.simulations_per_det,
                    batch_timeout_ms=baseline_config.batch_timeout_ms
                )
                setattr(config_verify, param_name, verify_value)

                print(f"  Verification: Testing {param_name}={verify_value}:")
                result_verify = run_benchmark(config_verify, num_rounds=200, device=device)
                checkpoint_mgr.save_result(
                    stage_num=stage_num,
                    phase='individual_fine',
                    config=config_verify,
                    num_rounds=200,
                    elapsed_sec=result_verify['elapsed_sec'],
                    rounds_per_min=result_verify.get('rounds_per_min'),
                    success=result_verify['success'],
                    error_msg=result_verify.get('error'),
                    param_name=param_name,
                    search_type=f'fine_verify_{fine_direction}'
                )

                if not result_verify['success']:
                    print(f"    Verification failed: {result_verify['error']}")
                    print(f"  Confirmed peak at {param_name}={current_value}")
                    break

                perf_verify = result_verify['rounds_per_min']

                if perf_verify < current_perf:
                    print(f"    Verification confirms drop ({perf_verify:.1f} < {current_perf:.1f})")
                    print(f"  Confirmed peak at {param_name}={current_value}")
                    break
                else:
                    print(f"    Verification shows improvement! ({perf_verify:.1f} >= {current_perf:.1f})")
                    print(f"    Previous drop was outlier, continuing from {param_name}={verify_value}")
                    current_value = verify_value
                    current_perf = perf_verify
                    best_value = current_value
                    best_perf = current_perf
                    continue

            else:
                # Performance improved or stable, continue
                print(f"    Improved! {perf_next:.1f} >= {current_perf:.1f}")
                current_value = next_value
                current_perf = perf_next
                best_value = current_value
                best_perf = current_perf

    else:
        # No hill-climbing needed, use best from initial tests
        best_value = current_best_value
        best_perf = current_best_perf

    print(f"\n{'='*60}")
    print(f"FINAL OPTIMAL: {param_name}={best_value}, Perf: {best_perf:.1f} r/min")
    print(f"Improvement: {((best_perf / baseline_perf) - 1) * 100:+.1f}% vs baseline")
    print(f"{'='*60}")

    return best_value, best_perf


def optimize_stage(
    stage_num: int,
    num_det: int,
    sims_per_det: int,
    stage_desc: str,
    checkpoint_mgr: MPPTCheckpointManager,
    device: str
) -> Dict:
    """
    Run full MPPT optimization for a single curriculum stage
    Gen2: Workers fixed at 32, only optimize batch_size and timeout

    Returns dict with:
        - baseline_config: MPPTConfig
        - baseline_perf: float
        - optimal_config: MPPTConfig
        - optimal_perf: float
        - param_optimals: Dict[str, int]
    """
    print("\n" + "="*80)
    print(f"CURRICULUM {stage_desc}: {num_det}×{sims_per_det} MCTS")
    print("="*80)

    # Phase 1: Baseline
    print("\n--- Phase 1: Baseline Validation ---")
    baseline_config = MPPTConfig(
        workers=BASELINE_WORKERS,  # Fixed at 32
        parallel_batch_size=BASELINE_BATCH_SIZE,
        num_determinizations=num_det,
        simulations_per_det=sims_per_det,
        batch_timeout_ms=BASELINE_TIMEOUT_MS
    )

    print(f"Config: {baseline_config}")
    result = run_benchmark(baseline_config, num_rounds=200, device=device)

    checkpoint_mgr.save_result(
        stage_num=stage_num,
        phase='baseline',
        config=baseline_config,
        num_rounds=200,
        elapsed_sec=result['elapsed_sec'],
        rounds_per_min=result.get('rounds_per_min'),
        success=result['success'],
        error_msg=result.get('error')
    )

    if not result['success']:
        print(f"\n❌ Baseline FAILED: {result['error']}")
        return None

    baseline_perf = result['rounds_per_min']
    print(f"\n✓ Baseline: {baseline_perf:.1f} r/min")

    # Phase 2: Individual parameter optimization (Gen2: only batch_size and timeout)
    print("\n" + "="*80)
    print("Phase 2: Individual Parameter Optimization (MPPT Gen2)")
    print("Workers fixed at 32 - optimizing batch_size and timeout only")
    print("="*80)

    param_optimals = {'workers': BASELINE_WORKERS}  # Fixed
    best_config = baseline_config
    best_perf = baseline_perf

    # Optimize each parameter (Gen2: workers removed)
    for param_name in ['parallel_batch_size', 'batch_timeout_ms']:
        optimal_value, optimal_perf = optimize_parameter_mppt(
            param_name=param_name,
            baseline_config=baseline_config,
            baseline_perf=baseline_perf,
            checkpoint_mgr=checkpoint_mgr,
            stage_num=stage_num,
            device=device
        )

        param_optimals[param_name] = optimal_value

        # Create config with this optimal value
        test_config = MPPTConfig(
            workers=BASELINE_WORKERS,  # Fixed at 32
            parallel_batch_size=baseline_config.parallel_batch_size,
            num_determinizations=num_det,
            simulations_per_det=sims_per_det,
            batch_timeout_ms=baseline_config.batch_timeout_ms
        )
        setattr(test_config, param_name, optimal_value)

        if optimal_perf > best_perf:
            best_perf = optimal_perf
            best_config = test_config

        checkpoint_mgr.save_checkpoint(stage_num, 'individual', best_config, best_config, best_perf)

    print(f"\n{'='*80}")
    print(f"Individual Optimization Complete")
    print(f"Optimal values: workers={param_optimals['workers']} (fixed), " +
          f"batch_size={param_optimals['parallel_batch_size']}, " +
          f"timeout={param_optimals['batch_timeout_ms']}ms")
    print(f"{'='*80}")

    # Phase 3: Combination testing (Gen2: 200 rounds, only vary 2 params)
    print("\n" + "="*80)
    print("Phase 3: Combination Testing")
    print("="*80)

    # Test all optimal values together
    combined_config = MPPTConfig(
        workers=BASELINE_WORKERS,  # Fixed at 32
        parallel_batch_size=param_optimals['parallel_batch_size'],
        num_determinizations=num_det,
        simulations_per_det=sims_per_det,
        batch_timeout_ms=param_optimals['batch_timeout_ms']
    )

    print(f"\n[Combined Optimum] Testing: {combined_config}")
    result = run_benchmark(combined_config, num_rounds=200, device=device)

    checkpoint_mgr.save_result(
        stage_num=stage_num,
        phase='combination',
        config=combined_config,
        num_rounds=200,
        elapsed_sec=result['elapsed_sec'],
        rounds_per_min=result.get('rounds_per_min'),
        success=result['success'],
        error_msg=result.get('error'),
        search_type='combined'
    )

    if result['success']:
        combined_perf = result['rounds_per_min']
        print(f"  Combined: {combined_perf:.1f} r/min")

        if combined_perf > best_perf:
            best_perf = combined_perf
            best_config = combined_config
            print(f"  ★ NEW BEST from combination!")
    else:
        combined_perf = 0.0
        print(f"  ✗ Combined test failed: {result['error']}")

    # Fine-tune around combined optimum (Gen2: only vary 2 params, 200 rounds)
    print(f"\n[Fine-Tuning] Testing variations around combined optimum...")

    fine_tune_tests = [
        # Vary batch_size ±5
        MPPTConfig(BASELINE_WORKERS, param_optimals['parallel_batch_size'] - 5,
                  num_det, sims_per_det, param_optimals['batch_timeout_ms']),
        MPPTConfig(BASELINE_WORKERS, param_optimals['parallel_batch_size'] + 5,
                  num_det, sims_per_det, param_optimals['batch_timeout_ms']),
        # Vary timeout ±1
        MPPTConfig(BASELINE_WORKERS, param_optimals['parallel_batch_size'],
                  num_det, sims_per_det, param_optimals['batch_timeout_ms'] - 1),
        MPPTConfig(BASELINE_WORKERS, param_optimals['parallel_batch_size'],
                  num_det, sims_per_det, param_optimals['batch_timeout_ms'] + 1),
    ]

    for idx, config in enumerate(fine_tune_tests):
        # Skip if any value is below minimum
        if config.parallel_batch_size < PARAM_MINIMUMS['parallel_batch_size'] or \
           config.batch_timeout_ms < PARAM_MINIMUMS['batch_timeout_ms']:
            continue

        print(f"  [{idx+1}/{len(fine_tune_tests)}] Testing: {config}")
        result = run_benchmark(config, num_rounds=200, device=device)

        checkpoint_mgr.save_result(
            stage_num=stage_num,
            phase='combination',
            config=config,
            num_rounds=200,
            elapsed_sec=result['elapsed_sec'],
            rounds_per_min=result.get('rounds_per_min'),
            success=result['success'],
            error_msg=result.get('error'),
            search_type='fine_tune'
        )

        if result['success']:
            perf = result['rounds_per_min']
            if perf > best_perf:
                best_perf = perf
                best_config = config
                print(f"    ★ NEW BEST: {best_perf:.1f} r/min")
            else:
                print(f"    {perf:.1f} r/min ({(perf/best_perf-1)*100:+.1f}% vs best)")

    checkpoint_mgr.save_checkpoint(stage_num, 'combination', best_config, best_config, best_perf)

    # Phase 4: Final validation (Gen2: 3×200 rounds)
    print("\n" + "="*80)
    print("Phase 4: Final Validation")
    print("="*80)
    print(f"Best config: {best_config}")

    result_final = run_multi_benchmark(best_config, num_runs=3, rounds_per_run=200, device=device)

    if result_final['success']:
        for run_idx, perf in enumerate(result_final['runs']):
            checkpoint_mgr.save_result(
                stage_num=stage_num,
                phase='final',
                config=best_config,
                num_rounds=200,
                elapsed_sec=result_final['elapsed_sec'] / len(result_final['runs']),
                rounds_per_min=perf,
                success=True,
                variance=result_final['variance']
            )

        final_perf = result_final['rounds_per_min']
        final_variance = result_final['variance']

        print(f"\n{'='*80}")
        print(f"STAGE {stage_num} COMPLETE")
        print(f"{'='*80}")
        print(f"Baseline: {baseline_perf:.1f} r/min")
        print(f"Optimal:  {final_perf:.1f} r/min ± {final_variance:.1f}%")
        print(f"Improvement: {((final_perf / baseline_perf) - 1) * 100:+.1f}%")
        print(f"Config: {best_config}")
        print(f"{'='*80}")

        checkpoint_mgr.save_checkpoint(stage_num, 'final', best_config, best_config, final_perf)

        return {
            'baseline_config': baseline_config,
            'baseline_perf': baseline_perf,
            'optimal_config': best_config,
            'optimal_perf': final_perf,
            'optimal_variance': final_variance,
            'param_optimals': param_optimals
        }
    else:
        print(f"\n❌ Final validation failed: {result_final.get('error', 'Unknown')}")
        return None


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MPPT-style parameter optimization for BlobMaster training curriculum (Gen2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_tune_mppt_gen2.py                                        # All 5 stages
  python auto_tune_mppt_gen2.py --stages 1,2,3                         # Only stages 1-3
  python auto_tune_mppt_gen2.py --resume results/auto_tune_mppt_gen2_20251114  # Resume
        """
    )

    parser.add_argument('--stages', type=str, default='1,2,3,4,5',
                       help='Comma-separated list of stages to run (1-5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/auto_tune_mppt_gen2_YYYYMMDD_HHMMSS/)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.resume:
        output_dir = Path(args.resume)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'results/auto_tune_mppt_gen2_{timestamp}')

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_dir / 'mppt_results.db')

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available, switching to CPU")
        args.device = 'cpu'

    print("="*80)
    print("BlobMaster MPPT-Style Auto-Tune (Generation 2)")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Stages: {args.stages}")
    print("\nGen2 Improvements:")
    print("  - Workers fixed at 32 (not optimized)")
    print("  - Hill-climbing fine search with verification")
    print("  - Consistent: 200 rounds for all tests (fair comparison)")
    print()

    # Parse stages
    stages_to_run = [int(s.strip()) for s in args.stages.split(',')]

    # Initialize checkpoint manager
    checkpoint_mgr = MPPTCheckpointManager(checkpoint_path)

    # Run optimization for each stage
    stage_results = {}

    try:
        for stage_idx in stages_to_run:
            if stage_idx < 1 or stage_idx > 5:
                print(f"WARNING: Invalid stage {stage_idx}, skipping")
                continue

            num_det, sims_per_det, stage_desc = CURRICULUM_STAGES[stage_idx - 1]

            result = optimize_stage(
                stage_num=stage_idx,
                num_det=num_det,
                sims_per_det=sims_per_det,
                stage_desc=stage_desc,
                checkpoint_mgr=checkpoint_mgr,
                device=args.device
            )

            if result:
                stage_results[stage_idx] = result

        # Summary
        print("\n" + "="*80)
        print("ALL STAGES COMPLETE")
        print("="*80)

        for stage_idx in sorted(stage_results.keys()):
            result = stage_results[stage_idx]
            num_det, sims_per_det, _ = CURRICULUM_STAGES[stage_idx - 1]
            improvement = ((result['optimal_perf'] / result['baseline_perf']) - 1) * 100

            print(f"\nStage {stage_idx} ({num_det}×{sims_per_det} MCTS):")
            print(f"  Baseline: {result['baseline_perf']:.1f} r/min")
            print(f"  Optimal:  {result['optimal_perf']:.1f} r/min ± {result['optimal_variance']:.1f}%")
            print(f"  Improvement: {improvement:+.1f}%")
            print(f"  Config: {result['optimal_config']}")

        print(f"\nResults saved to: {output_dir}")
        print(f"  - experiments.json: All experiment results")
        print(f"  - checkpoint.json: Resume checkpoint")
        print(f"\nGenerate report with:")
        print(f"  python benchmarks/performance/auto_tune_mppt_report.py {output_dir}")

        checkpoint_mgr.close()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Progress has been saved to checkpoint")
        print(f"Resume with: python auto_tune_mppt_gen2.py --resume {output_dir}")
        checkpoint_mgr.close()
        return 130

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        checkpoint_mgr.close()
        return 1


if __name__ == '__main__':
    sys.exit(main())
