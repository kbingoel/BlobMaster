#!/usr/bin/env python3
"""
Intelligent Auto-Tune Parameter Sweep for BlobMaster Training

This script performs systematic exploration of training hyperparameters to find
optimal configurations for Phase 1 (Independent Rounds) training performance.

Features:
- Baseline validation with fail-fast
- Smart parameter exploration with early termination
- Statistical validation with multiple runs
- Checkpoint/resume capability
- Comprehensive reporting and visualization
- Unattended operation with progress tracking

Usage:
    python benchmarks/performance/auto_tune.py
    python benchmarks/performance/auto_tune.py --time-budget 360
    python benchmarks/performance/auto_tune.py --resume results/auto_tune_20251114_120000
    python benchmarks/performance/auto_tune.py --quick
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import os

import torch
import psutil

# No longer using SQLite - using JSON files instead

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class SweepConfig:
    """Configuration for a single sweep test"""
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
        return f"{self.workers}w × {self.parallel_batch_size}batch × {self.num_determinizations}×{self.simulations_per_det} MCTS"


# ============================================================================
# JSON-based Checkpoint Management
# ============================================================================

class CheckpointManager:
    """Manages JSON files for results and checkpoints"""

    def __init__(self, base_path: str):
        # base_path is like "results/auto_tune_20251114/auto_tune_results.db"
        # Convert to directory path
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

    def save_result(self, phase: str, config: SweepConfig, num_rounds: int,
                    elapsed_sec: float, rounds_per_min: Optional[float],
                    success: bool, error_msg: Optional[str] = None,
                    variance: Optional[float] = None,
                    examples_per_round: Optional[float] = None,
                    cpu_percent: Optional[float] = None,
                    gpu_memory_mb: Optional[float] = None):
        """Save experiment result to JSON file"""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
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

    def save_checkpoint(self, phase: str, last_config: SweepConfig,
                       best_config: SweepConfig, best_perf: float):
        """Save checkpoint for resume capability"""
        self.checkpoint_data = {
            'last_phase': phase,
            'last_config': last_config.to_dict(),
            'current_best_config': best_config.to_dict(),
            'current_best_perf': best_perf,
            'timestamp': datetime.now().isoformat()
        }
        self._save_json(self.checkpoint_file, self.checkpoint_data)

    def load_checkpoint(self) -> Optional[Tuple[str, SweepConfig, SweepConfig, float]]:
        """Load last checkpoint if exists"""
        if self.checkpoint_data:
            last_phase = self.checkpoint_data['last_phase']
            last_config = SweepConfig(**self.checkpoint_data['last_config'])
            best_config = SweepConfig(**self.checkpoint_data['current_best_config'])
            best_perf = self.checkpoint_data['current_best_perf']
            return last_phase, last_config, best_config, best_perf
        return None

    def get_completed_configs(self, phase: Optional[str] = None) -> Set[str]:
        """Get hashes of completed configurations"""
        hashes = set()
        for exp in self.experiments:
            if exp['success']:
                if phase is None or exp['phase'] == phase:
                    hashes.add(exp['config_hash'])
        return hashes

    def get_successful_results(self, phase: Optional[str] = None) -> List[Dict]:
        """Get all successful experiment results"""
        results = []

        for exp in self.experiments:
            if exp['success'] and exp['rounds_per_min'] is not None:
                if phase is None or exp['phase'] == phase:
                    config = SweepConfig(
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
                        'num_rounds': exp['num_rounds']
                    })

        # Sort by rounds_per_min descending
        results.sort(key=lambda x: x['rounds_per_min'], reverse=True)
        return results

    def get_failed_results(self) -> List[Dict]:
        """Get all failed experiment results"""
        results = []

        for exp in self.experiments:
            if not exp['success']:
                config = SweepConfig(
                    workers=exp['workers'],
                    parallel_batch_size=exp['parallel_batch_size'],
                    num_determinizations=exp['num_determinizations'],
                    simulations_per_det=exp['simulations_per_det'],
                    batch_timeout_ms=exp['batch_timeout_ms']
                )
                results.append({
                    'config': config,
                    'error': exp.get('error_msg', 'Unknown error'),
                    'phase': exp['phase']
                })

        return results

    def close(self):
        """No-op for JSON-based manager (files are saved immediately)"""
        pass


# ============================================================================
# Progress Tracking and ETA Estimation
# ============================================================================

class ETAEstimator:
    """Estimate remaining time based on actual runtime"""

    def __init__(self):
        self.phase_times: Dict[str, List[float]] = {}
        self.config_times: List[float] = []
        self.start_time = time.time()

    def update_config(self, elapsed: float):
        """Record time for a single config test"""
        self.config_times.append(elapsed)

    def update_phase(self, phase: str, elapsed: float):
        """Record time for entire phase"""
        if phase not in self.phase_times:
            self.phase_times[phase] = []
        self.phase_times[phase].append(elapsed)

    def estimate_remaining(self, configs_remaining: int) -> float:
        """Estimate remaining time based on average config time"""
        if not self.config_times:
            return 0.0

        avg_time_per_config = sum(self.config_times) / len(self.config_times)
        return avg_time_per_config * configs_remaining

    def format_eta(self, seconds: float) -> str:
        """Format seconds as human-readable duration"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def elapsed(self) -> float:
        """Get total elapsed time"""
        return time.time() - self.start_time


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
        dropout=0.1,
        max_bid=13,
        max_cards=52
    ).to(device)

    # Initialize with random weights (not training, just benchmarking)
    network.eval()

    return network, encoder, masker


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_config_benchmark(
    config: SweepConfig,
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
        # Check hardware limits
        if config.workers > 32:
            return {
                'success': False,
                'error': 'Exceeds known VRAM limit (max 32 workers for RTX 4060 8GB)',
                'elapsed_sec': 0.0
            }

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
            batch_size=256,
            batch_timeout_ms=config.batch_timeout_ms,
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
        time.sleep(2)  # Cooldown

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


def run_config_multi(
    config: SweepConfig,
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
        result = run_config_benchmark(config, rounds_per_run, device)

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
# Phase 1: Baseline Validation
# ============================================================================

def run_baseline_phase(
    checkpoint_mgr: CheckpointManager,
    device: str,
    baseline_config: SweepConfig,
    expected_perf: float,
    tolerance: float
) -> Tuple[bool, float]:
    """
    Phase 1: Validate baseline configuration

    Returns (success, baseline_perf)
    """
    print("\n" + "="*80)
    print("Phase 1/4: Baseline Validation")
    print("="*80)
    print(f"Config: {baseline_config}")
    print(f"Expected: {expected_perf:.1f} r/min ±{tolerance*100:.0f}%")
    print()

    # Run baseline with high confidence (500 rounds)
    result = run_config_benchmark(baseline_config, num_rounds=500, device=device)

    # Save to database
    checkpoint_mgr.save_result(
        phase='baseline',
        config=baseline_config,
        num_rounds=500,
        elapsed_sec=result['elapsed_sec'],
        rounds_per_min=result.get('rounds_per_min'),
        success=result['success'],
        error_msg=result.get('error'),
        examples_per_round=result.get('examples_per_round'),
        cpu_percent=result.get('cpu_percent'),
        gpu_memory_mb=result.get('gpu_memory_mb')
    )

    if not result['success']:
        print(f"\n❌ Baseline FAILED: {result['error']}")
        return False, 0.0

    baseline_perf = result['rounds_per_min']
    diff_pct = ((baseline_perf - expected_perf) / expected_perf) * 100

    print(f"\n{'✓' if abs(diff_pct) <= tolerance * 100 else '✗'} Result: {baseline_perf:.1f} r/min ({diff_pct:+.1f}% vs expected)")

    if abs(diff_pct) > tolerance * 100:
        print(f"\n⚠️  WARNING: Baseline differs by {abs(diff_pct):.1f}% from expected!")
        if abs(diff_pct) > 20:
            print("❌ ABORT: System performance significantly degraded. Check:")
            print("   - GPU driver")
            print("   - Background processes")
            print("   - Hardware health")
            return False, baseline_perf
        else:
            print("   Continuing, but results may differ from historical baselines.")

    # Save checkpoint
    checkpoint_mgr.save_checkpoint('baseline', baseline_config, baseline_config, baseline_perf)

    return True, baseline_perf


# ============================================================================
# Phase 2: Individual Parameter Sweeps
# ============================================================================

def run_individual_sweeps(
    checkpoint_mgr: CheckpointManager,
    device: str,
    baseline_config: SweepConfig,
    baseline_perf: float,
    param_space: Dict,
    early_stop_threshold: float,
    eta_estimator: ETAEstimator
) -> Tuple[SweepConfig, float]:
    """
    Phase 2: Sweep each parameter individually

    Returns (best_config, best_perf)
    """
    print("\n" + "="*80)
    print("Phase 2/4: Individual Parameter Sweeps")
    print("="*80)
    print(f"Baseline: {baseline_perf:.1f} r/min")
    print(f"Early stop threshold: {early_stop_threshold*100:.0f}% slower")
    print()

    best_config = baseline_config
    best_perf = baseline_perf

    # Order of parameter exploration (most impactful first)
    param_order = ['workers', 'parallel_batch_size', 'num_determinizations', 'simulations_per_det', 'batch_timeout_ms']

    for param_name in param_order:
        if param_name not in param_space:
            continue

        print(f"\n--- Sweeping: {param_name} ---")
        param_values = param_space[param_name]
        baseline_value = getattr(baseline_config, param_name)

        # Test each value
        for value in param_values:
            # Skip baseline value (already tested)
            if value == baseline_value:
                print(f"  {param_name}={value}: (baseline)")
                continue

            # Create config with this parameter value
            config = SweepConfig(
                workers=baseline_config.workers,
                parallel_batch_size=baseline_config.parallel_batch_size,
                num_determinizations=baseline_config.num_determinizations,
                simulations_per_det=baseline_config.simulations_per_det,
                batch_timeout_ms=baseline_config.batch_timeout_ms
            )
            setattr(config, param_name, value)

            # Skip if already tested
            if config.to_hash() in checkpoint_mgr.get_completed_configs('individual'):
                print(f"  {param_name}={value}: (already tested, skipping)")
                continue

            print(f"  {param_name}={value}:")

            # Quick screening (100 rounds)
            config_start = time.time()
            result = run_config_benchmark(config, num_rounds=100, device=device)
            config_elapsed = time.time() - config_start
            eta_estimator.update_config(config_elapsed)

            # Save result
            checkpoint_mgr.save_result(
                phase='individual',
                config=config,
                num_rounds=100,
                elapsed_sec=result['elapsed_sec'],
                rounds_per_min=result.get('rounds_per_min'),
                success=result['success'],
                error_msg=result.get('error'),
                examples_per_round=result.get('examples_per_round'),
                cpu_percent=result.get('cpu_percent'),
                gpu_memory_mb=result.get('gpu_memory_mb')
            )

            if not result['success']:
                print(f"    ✗ Failed: {result['error']}")
                continue

            perf = result['rounds_per_min']
            perf_vs_best = perf / best_perf

            # Early termination check
            if perf_vs_best < (1 - early_stop_threshold):
                print(f"    ✗ {perf:.1f} r/min ({(1-perf_vs_best)*100:.1f}% slower, skipping)")
                continue

            # Promising config - extended validation
            if perf_vs_best >= 0.95:  # Within 5% of best
                print(f"    ✓ {perf:.1f} r/min (promising, validating with 500 rounds)")
                result_ext = run_config_benchmark(config, num_rounds=500, device=device)

                checkpoint_mgr.save_result(
                    phase='individual',
                    config=config,
                    num_rounds=500,
                    elapsed_sec=result_ext['elapsed_sec'],
                    rounds_per_min=result_ext.get('rounds_per_min'),
                    success=result_ext['success'],
                    error_msg=result_ext.get('error'),
                    examples_per_round=result_ext.get('examples_per_round'),
                    cpu_percent=result_ext.get('cpu_percent'),
                    gpu_memory_mb=result_ext.get('gpu_memory_mb')
                )

                if result_ext['success']:
                    perf = result_ext['rounds_per_min']
                    if perf > best_perf:
                        best_perf = perf
                        best_config = config
                        print(f"    ★ NEW BEST: {best_perf:.1f} r/min")
            else:
                print(f"    ○ {perf:.1f} r/min ({(perf_vs_best-1)*100:+.1f}% vs best)")

        # Save checkpoint after each parameter
        checkpoint_mgr.save_checkpoint('individual', best_config, best_config, best_perf)
        print(f"\nCurrent best after {param_name}: {best_config} = {best_perf:.1f} r/min")

    eta_estimator.update_phase('individual', eta_estimator.elapsed())
    return best_config, best_perf


# ============================================================================
# Phase 3: Interaction Exploration
# ============================================================================

def run_interaction_phase(
    checkpoint_mgr: CheckpointManager,
    device: str,
    baseline_config: SweepConfig,
    best_individual_config: SweepConfig,
    best_perf: float,
    param_space: Dict,
    eta_estimator: ETAEstimator
) -> Tuple[SweepConfig, float]:
    """
    Phase 3: Explore promising parameter combinations

    Returns (best_config, best_perf)
    """
    print("\n" + "="*80)
    print("Phase 3/4: Interaction Exploration")
    print("="*80)
    print(f"Current best: {best_perf:.1f} r/min")
    print()

    # Get top values for each parameter from Phase 2
    individual_results = checkpoint_mgr.get_successful_results('individual')

    if not individual_results:
        print("No successful individual results, skipping interaction phase")
        return best_individual_config, best_perf

    # Find top 3 values for each parameter
    top_params = {}
    for param_name in ['workers', 'parallel_batch_size', 'num_determinizations', 'simulations_per_det']:
        param_perfs = {}
        for result in individual_results:
            config = result['config']
            value = getattr(config, param_name)
            perf = result['rounds_per_min']

            if value not in param_perfs or perf > param_perfs[value]:
                param_perfs[value] = perf

        # Sort by performance and take top 3
        sorted_values = sorted(param_perfs.items(), key=lambda x: x[1], reverse=True)
        top_params[param_name] = [v for v, _ in sorted_values[:3]]

        print(f"Top {param_name}: {top_params[param_name]}")

    # Generate promising combinations
    # Strategy: Combine top values from different parameters
    combinations = []

    # Best from each parameter
    if len(top_params['workers']) > 0 and len(top_params['parallel_batch_size']) > 0:
        for workers in top_params['workers'][:2]:
            for batch_size in top_params['parallel_batch_size'][:2]:
                combinations.append(SweepConfig(
                    workers=workers,
                    parallel_batch_size=batch_size,
                    num_determinizations=best_individual_config.num_determinizations,
                    simulations_per_det=best_individual_config.simulations_per_det,
                    batch_timeout_ms=best_individual_config.batch_timeout_ms
                ))

    # Test MCTS combinations with best workers/batch_size
    if len(top_params['num_determinizations']) > 0 and len(top_params['simulations_per_det']) > 0:
        for num_det in top_params['num_determinizations'][:2]:
            for sims in top_params['simulations_per_det'][:2]:
                combinations.append(SweepConfig(
                    workers=best_individual_config.workers,
                    parallel_batch_size=best_individual_config.parallel_batch_size,
                    num_determinizations=num_det,
                    simulations_per_det=sims,
                    batch_timeout_ms=best_individual_config.batch_timeout_ms
                ))

    # Remove duplicates
    seen_hashes = {best_individual_config.to_hash()}
    unique_combinations = []
    for config in combinations:
        if config.to_hash() not in seen_hashes:
            seen_hashes.add(config.to_hash())
            unique_combinations.append(config)

    print(f"\nTesting {len(unique_combinations)} promising combinations...")

    best_config = best_individual_config

    for idx, config in enumerate(unique_combinations):
        print(f"\n[{idx+1}/{len(unique_combinations)}] {config}:")

        # Skip if already tested
        if config.to_hash() in checkpoint_mgr.get_completed_configs():
            print("  (already tested, skipping)")
            continue

        config_start = time.time()
        result = run_config_benchmark(config, num_rounds=500, device=device)
        config_elapsed = time.time() - config_start
        eta_estimator.update_config(config_elapsed)

        checkpoint_mgr.save_result(
            phase='interaction',
            config=config,
            num_rounds=500,
            elapsed_sec=result['elapsed_sec'],
            rounds_per_min=result.get('rounds_per_min'),
            success=result['success'],
            error_msg=result.get('error'),
            examples_per_round=result.get('examples_per_round'),
            cpu_percent=result.get('cpu_percent'),
            gpu_memory_mb=result.get('gpu_memory_mb')
        )

        if not result['success']:
            print(f"  ✗ Failed: {result['error']}")
            continue

        perf = result['rounds_per_min']
        if perf > best_perf:
            best_perf = perf
            best_config = config
            print(f"  ★ NEW BEST: {best_perf:.1f} r/min")
        else:
            print(f"  ○ {perf:.1f} r/min ({(perf/best_perf-1)*100:+.1f}% vs best)")

    eta_estimator.update_phase('interaction', eta_estimator.elapsed())
    checkpoint_mgr.save_checkpoint('interaction', best_config, best_config, best_perf)

    return best_config, best_perf


# ============================================================================
# Phase 4: Final Validation
# ============================================================================

def run_final_validation(
    checkpoint_mgr: CheckpointManager,
    device: str,
    best_config: SweepConfig,
    best_perf: float,
    eta_estimator: ETAEstimator
) -> Dict:
    """
    Phase 4: Final validation of top configurations

    Returns dict with top configs and their statistics
    """
    print("\n" + "="*80)
    print("Phase 4/4: Final Validation")
    print("="*80)
    print(f"Current best: {best_config} = {best_perf:.1f} r/min")
    print()

    # Get top 5 configs from all phases
    all_results = checkpoint_mgr.get_successful_results()
    top_5 = sorted(all_results, key=lambda x: x['rounds_per_min'], reverse=True)[:5]

    print(f"Validating top {len(top_5)} configurations with 1000 rounds each...\n")

    final_results = []

    for idx, result in enumerate(top_5):
        config = result['config']
        print(f"[{idx+1}/{len(top_5)}] {config}:")

        # Run multiple times for variance
        config_start = time.time()
        multi_result = run_config_multi(config, num_runs=3, rounds_per_run=1000, device=device)
        config_elapsed = time.time() - config_start
        eta_estimator.update_config(config_elapsed)

        if not multi_result['success']:
            print(f"  ✗ Failed: {multi_result['error']}")
            continue

        # Save each run
        for run_idx, perf in enumerate(multi_result['runs']):
            checkpoint_mgr.save_result(
                phase='final',
                config=config,
                num_rounds=1000,
                elapsed_sec=multi_result['elapsed_sec'] / len(multi_result['runs']),
                rounds_per_min=perf,
                success=True,
                variance=multi_result['variance']
            )

        final_results.append({
            'config': config,
            'rounds_per_min': multi_result['rounds_per_min'],
            'variance': multi_result['variance'],
            'runs': multi_result['runs']
        })

        print(f"  ✓ {multi_result['rounds_per_min']:.1f} r/min ± {multi_result['variance']:.1f}%")

    eta_estimator.update_phase('final', eta_estimator.elapsed())

    # Find overall best
    if final_results:
        best_final = max(final_results, key=lambda x: x['rounds_per_min'])
        checkpoint_mgr.save_checkpoint('final', best_final['config'], best_final['config'], best_final['rounds_per_min'])

    return {
        'top_configs': final_results,
        'best': final_results[0] if final_results else None
    }


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent auto-tune parameter sweep for BlobMaster training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_tune.py                                        # Full sweep
  python auto_tune.py --time-budget 360                      # 6 hour time limit
  python auto_tune.py --resume results/auto_tune_20251114    # Resume from checkpoint
  python auto_tune.py --quick                                # Quick mode (reduced space)
        """
    )

    parser.add_argument('--time-budget', type=int, default=None,
                       help='Maximum runtime in minutes (default: unlimited)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory (e.g., results/auto_tune_20251114_120000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/auto_tune_YYYYMMDD_HHMMSS/)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced parameter space')
    parser.add_argument('--phases', type=str, default='baseline,individual,interaction,final',
                       help='Comma-separated list of phases to run')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--baseline-tolerance', type=float, default=0.10,
                       help='Baseline validation tolerance (default: 0.10 = ±10%%)')
    parser.add_argument('--early-stop-threshold', type=float, default=0.30,
                       help='Early termination threshold (default: 0.30 = 30%% slower)')
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
        output_dir = Path(f'results/auto_tune_{timestamp}')

    output_dir.mkdir(parents=True, exist_ok=True)

    # For compatibility with CheckpointManager that expects a "db path"
    # We pass a dummy path and let CheckpointManager use its parent directory
    checkpoint_path = str(output_dir / 'auto_tune_results.db')

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available, switching to CPU")
        args.device = 'cpu'

    print("="*80)
    print("BlobMaster Auto-Tune Parameter Sweep")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Phases: {args.phases}")
    if args.time_budget:
        print(f"Time budget: {args.time_budget} minutes")
    print()

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_path)
    eta_estimator = ETAEstimator()

    # Define parameter space
    if args.quick:
        param_space = {
            'workers': [16, 24, 32],
            'parallel_batch_size': [20, 30, 40],
            'num_determinizations': [2, 3, 4],
            'simulations_per_det': [20, 30, 40],
            'batch_timeout_ms': [10]
        }
    else:
        param_space = {
            'workers': [8, 16, 24, 32],
            'parallel_batch_size': [10, 20, 30, 40, 50],
            'num_determinizations': [2, 3, 4, 5],
            'simulations_per_det': [15, 20, 25, 30, 35, 40, 50],
            'batch_timeout_ms': [5, 10, 15, 20]
        }

    # Baseline configuration
    baseline_config = SweepConfig(
        workers=32,
        parallel_batch_size=30,
        num_determinizations=3,
        simulations_per_det=30,
        batch_timeout_ms=10
    )
    expected_baseline_perf = 741.0  # From validation 2025-11-13

    # Parse phases
    phases_to_run = [p.strip() for p in args.phases.split(',')]

    try:
        # Phase 1: Baseline
        if 'baseline' in phases_to_run:
            success, baseline_perf = run_baseline_phase(
                checkpoint_mgr, args.device, baseline_config,
                expected_baseline_perf, args.baseline_tolerance
            )

            if not success:
                print("\n❌ Baseline validation failed, aborting")
                return 1

            best_config = baseline_config
            best_perf = baseline_perf
        else:
            # Load from checkpoint
            checkpoint = checkpoint_mgr.load_checkpoint()
            if checkpoint:
                _, _, best_config, best_perf = checkpoint
                print(f"Loaded from checkpoint: {best_config} = {best_perf:.1f} r/min")
            else:
                print("ERROR: Skipping baseline but no checkpoint found")
                return 1

        # Phase 2: Individual sweeps
        if 'individual' in phases_to_run:
            best_config, best_perf = run_individual_sweeps(
                checkpoint_mgr, args.device, baseline_config, best_perf,
                param_space, args.early_stop_threshold, eta_estimator
            )

        # Phase 3: Interactions
        if 'interaction' in phases_to_run:
            best_config, best_perf = run_interaction_phase(
                checkpoint_mgr, args.device, baseline_config, best_config,
                best_perf, param_space, eta_estimator
            )

        # Phase 4: Final validation
        final_results = None
        if 'final' in phases_to_run:
            final_results = run_final_validation(
                checkpoint_mgr, args.device, best_config, best_perf, eta_estimator
            )

        # Summary
        print("\n" + "="*80)
        print("SWEEP COMPLETE")
        print("="*80)
        print(f"Total runtime: {eta_estimator.format_eta(eta_estimator.elapsed())}")
        print(f"\nBest configuration: {best_config}")
        print(f"Performance: {best_perf:.1f} r/min")
        print(f"Improvement: {((best_perf / expected_baseline_perf) - 1) * 100:+.1f}%")

        # Print top 5
        all_results = checkpoint_mgr.get_successful_results()
        top_5 = sorted(all_results, key=lambda x: x['rounds_per_min'], reverse=True)[:5]

        print("\nTop 5 Configurations:")
        for idx, result in enumerate(top_5):
            config = result['config']
            perf = result['rounds_per_min']
            variance = result.get('variance', 0.0)
            print(f"{idx+1}. {config} = {perf:.1f} r/min (variance: {variance:.1f}%)")

        # Print failures
        failures = checkpoint_mgr.get_failed_results()
        if failures:
            print(f"\n{len(failures)} Failed Configurations:")
            for failure in failures[:10]:  # Show first 10
                config = failure['config']
                error = failure['error']
                print(f"  - {config}: {error}")

        print(f"\nResults saved to: {output_dir}")
        print(f"  - experiments.json: All experiment results")
        print(f"  - checkpoint.json: Resume checkpoint")
        print(f"\nGenerate report with:")
        print(f"  python benchmarks/performance/auto_tune_report.py {output_dir}")

        checkpoint_mgr.close()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Progress has been saved to checkpoint")
        print(f"Resume with: python auto_tune.py --resume {output_dir}")
        checkpoint_mgr.close()
        return 130

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        checkpoint_mgr.close()
        return 1


if __name__ == '__main__':
    sys.exit(main())
