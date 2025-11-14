#!/usr/bin/env python3
"""
Bayesian Optimization autotune using Optuna TPE sampler.

This script optimizes parallel_batch_size and batch_timeout_ms for each
curriculum stage using Optuna's Tree-structured Parzen Estimator (TPE).

Usage:
    # Full run (all 5 stages, 15 trials each)
    python benchmarks/performance/auto_tune_bo.py

    # Specific stages
    python benchmarks/performance/auto_tune_bo.py --stages 1,2,3

    # Resume from checkpoint
    python benchmarks/performance/auto_tune_bo.py --resume results/auto_tune_bo_20251114

    # Custom trial count
    python benchmarks/performance/auto_tune_bo.py --n-trials 20 --n-startup 8
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: Optuna not installed. Please run: pip install optuna")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BOConfig:
    """Configuration for a single Bayesian Optimization trial."""
    workers: int
    parallel_batch_size: int
    num_determinizations: int
    simulations_per_det: int
    batch_timeout_ms: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = f"{self.workers}_{self.parallel_batch_size}_{self.num_determinizations}_{self.simulations_per_det}_{self.batch_timeout_ms}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# Curriculum stages: (num_determinizations, simulations_per_det, description)
CURRICULUM_STAGES = [
    (1, 15, "Stage 1: Early Training"),
    (2, 25, "Stage 2: Mid-Early Training"),
    (3, 35, "Stage 3: Mid Training"),
    (4, 45, "Stage 4: Late-Mid Training"),
    (5, 50, "Stage 5: Final Training"),
]

# Baseline configuration (strong starting point)
BASELINE_WORKERS = 32
BASELINE_BATCH_SIZE = 30
BASELINE_TIMEOUT_MS = 10

# Optuna search space bounds
BATCH_SIZE_MIN = 5
BATCH_SIZE_MAX = 100
TIMEOUT_MS_MIN = 1
TIMEOUT_MS_MAX = 50

# Default Optuna parameters
DEFAULT_N_TRIALS = 15
DEFAULT_N_STARTUP_TRIALS = 5  # Random trials before TPE kicks in
DEFAULT_ROUNDS_PER_TRIAL = 200
DEFAULT_VALIDATION_ROUNDS = 200
DEFAULT_VALIDATION_RUNS = 3


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_benchmark(config: BOConfig, num_rounds: int, verbose: bool = False) -> Optional[float]:
    """
    Run a benchmark with the given configuration.

    Args:
        config: Configuration to test
        num_rounds: Number of rounds to run
        verbose: Print detailed output

    Returns:
        Rounds per minute, or None if benchmark failed
    """
    cmd = [
        sys.executable,
        "benchmarks/performance/benchmark_phase2_iteration.py",
        "--workers", str(config.workers),
        "--parallel-batch-size", str(config.parallel_batch_size),
        "--num-determinizations", str(config.num_determinizations),
        "--simulations-per-det", str(config.simulations_per_det),
        "--batch-timeout-ms", str(config.batch_timeout_ms),
        "--num-games", str(num_rounds),
        "--device", "cuda",
    ]

    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"  ERROR: Benchmark failed with return code {result.returncode}")
            if verbose:
                print(f"  STDERR: {result.stderr}")
            return None

        # Parse output for rounds/min
        for line in result.stdout.splitlines():
            if "Games per minute:" in line:
                games_per_min = float(line.split(":")[-1].strip())
                return games_per_min

        print(f"  ERROR: Could not parse rounds/min from output")
        return None

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Benchmark timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"  ERROR: Benchmark failed with exception: {e}")
        return None


# =============================================================================
# Optuna Optimization
# =============================================================================

def create_objective(stage_det: int, stage_sims: int, num_rounds: int, results_log: List[Dict]):
    """
    Create an Optuna objective function for a specific curriculum stage.

    Args:
        stage_det: Number of determinizations for this stage
        stage_sims: Simulations per determinization for this stage
        num_rounds: Number of rounds per trial
        results_log: List to append trial results to

    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        # Suggest INTEGER parameters (clean values)
        parallel_batch_size = trial.suggest_int('parallel_batch_size', BATCH_SIZE_MIN, BATCH_SIZE_MAX)
        batch_timeout_ms = trial.suggest_int('batch_timeout_ms', TIMEOUT_MS_MIN, TIMEOUT_MS_MAX)

        config = BOConfig(
            workers=BASELINE_WORKERS,  # Fixed
            parallel_batch_size=parallel_batch_size,
            num_determinizations=stage_det,
            simulations_per_det=stage_sims,
            batch_timeout_ms=batch_timeout_ms,
        )

        print(f"\n  Trial {trial.number}: batch_size={parallel_batch_size}, timeout={batch_timeout_ms}ms")

        # Run benchmark
        rounds_per_min = run_benchmark(config, num_rounds, verbose=False)

        if rounds_per_min is None:
            # Failed trial - return very low value
            print(f"    FAILED")
            rounds_per_min = 0.0
            trial.set_user_attr("failed", True)
        else:
            print(f"    Result: {rounds_per_min:.1f} r/min")
            trial.set_user_attr("failed", False)

        # Log result
        results_log.append({
            "trial_number": trial.number,
            "parallel_batch_size": parallel_batch_size,
            "batch_timeout_ms": batch_timeout_ms,
            "rounds_per_min": rounds_per_min,
            "state": trial.state.name,
        })

        return rounds_per_min

    return objective


def optimize_stage(
    stage_num: int,
    stage_det: int,
    stage_sims: int,
    stage_desc: str,
    n_trials: int,
    n_startup_trials: int,
    rounds_per_trial: int,
    output_dir: Path,
) -> Dict:
    """
    Run Bayesian Optimization for a single curriculum stage.

    Args:
        stage_num: Stage number (1-5)
        stage_det: Number of determinizations
        stage_sims: Simulations per determinization
        stage_desc: Stage description
        n_trials: Total number of trials
        n_startup_trials: Number of random startup trials
        rounds_per_trial: Rounds per trial
        output_dir: Output directory for results

    Returns:
        Dictionary with optimization results
    """
    print(f"\n{'='*80}")
    print(f"{stage_desc} ({stage_det}×{stage_sims} MCTS)")
    print(f"{'='*80}")
    print(f"Running {n_trials} trials ({n_startup_trials} random startup + {n_trials - n_startup_trials} TPE)")

    # Create Optuna study with TPE sampler
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,  # Reproducibility
    )

    study = optuna.create_study(
        direction='maximize',  # Maximize rounds/min
        sampler=sampler,
        study_name=f"stage_{stage_num}",
    )

    # Enqueue baseline as first trial
    print(f"\nEnqueuing baseline: batch_size={BASELINE_BATCH_SIZE}, timeout={BASELINE_TIMEOUT_MS}ms")
    study.enqueue_trial({
        'parallel_batch_size': BASELINE_BATCH_SIZE,
        'batch_timeout_ms': BASELINE_TIMEOUT_MS,
    })

    # Results log for this stage
    results_log = []

    # Create objective function
    objective = create_objective(stage_det, stage_sims, rounds_per_trial, results_log)

    # Run optimization
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    end_time = time.time()

    # Extract results
    best_trial = study.best_trial
    best_config = BOConfig(
        workers=BASELINE_WORKERS,
        parallel_batch_size=best_trial.params['parallel_batch_size'],
        num_determinizations=stage_det,
        simulations_per_det=stage_sims,
        batch_timeout_ms=best_trial.params['batch_timeout_ms'],
    )

    # Get baseline performance (trial 0)
    baseline_perf = results_log[0]['rounds_per_min'] if results_log else 0.0
    improvement = ((best_trial.value - baseline_perf) / baseline_perf * 100) if baseline_perf > 0 else 0.0

    print(f"\n{'='*80}")
    print(f"Stage {stage_num} Optimization Complete!")
    print(f"{'='*80}")
    print(f"Best Configuration:")
    print(f"  parallel_batch_size: {best_config.parallel_batch_size}")
    print(f"  batch_timeout_ms: {best_config.batch_timeout_ms}")
    print(f"Best Performance: {best_trial.value:.1f} r/min")
    print(f"Baseline Performance: {baseline_perf:.1f} r/min")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Optimization Runtime: {(end_time - start_time) / 60:.1f} minutes")

    return {
        "stage_num": stage_num,
        "stage_mcts": f"{stage_det}×{stage_sims}",
        "stage_desc": stage_desc,
        "n_trials": n_trials,
        "n_startup_trials": n_startup_trials,
        "best_config": best_config.to_dict(),
        "best_performance": best_trial.value,
        "baseline_performance": baseline_perf,
        "improvement_pct": improvement,
        "trials": results_log,
        "runtime_seconds": end_time - start_time,
        "timestamp": datetime.now().isoformat(),
    }


def run_validation(config: BOConfig, num_rounds: int, num_runs: int) -> Dict:
    """
    Run validation trials for a configuration.

    Args:
        config: Configuration to validate
        num_rounds: Rounds per validation run
        num_runs: Number of validation runs

    Returns:
        Validation statistics
    """
    print(f"\nRunning {num_runs} validation runs ({num_rounds} rounds each)...")

    results = []
    for i in range(num_runs):
        print(f"  Validation run {i+1}/{num_runs}...", end=" ")
        perf = run_benchmark(config, num_rounds, verbose=False)
        if perf is not None:
            results.append(perf)
            print(f"{perf:.1f} r/min")
        else:
            print("FAILED")

    if not results:
        return {
            "success": False,
            "num_runs": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    import statistics
    mean = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    std_pct = (std / mean * 100) if mean > 0 else 0.0

    print(f"\nValidation Results:")
    print(f"  Mean: {mean:.1f} r/min")
    print(f"  Std Dev: {std:.1f} r/min (±{std_pct:.1f}%)")
    print(f"  Range: [{min(results):.1f}, {max(results):.1f}]")

    return {
        "success": True,
        "num_runs": len(results),
        "mean": mean,
        "std": std,
        "std_pct": std_pct,
        "min": min(results),
        "max": max(results),
        "all_results": results,
    }


# =============================================================================
# Results Management
# =============================================================================

def save_results(output_dir: Path, stage_results: Dict, validation_results: Dict):
    """
    Save stage results to JSON files.

    Args:
        output_dir: Output directory
        stage_results: Stage optimization results
        validation_results: Validation results
    """
    experiments_file = output_dir / "experiments.json"

    # Load existing experiments
    if experiments_file.exists():
        with open(experiments_file, 'r') as f:
            experiments = json.load(f)
    else:
        experiments = []

    # Add trials from this stage
    stage_num = stage_results["stage_num"]
    stage_mcts = stage_results["stage_mcts"]
    baseline_perf = stage_results["baseline_performance"]

    for trial in stage_results["trials"]:
        # Determine trial type
        if trial["trial_number"] == 0:
            trial_type = "baseline"
        elif trial["trial_number"] < stage_results["n_startup_trials"]:
            trial_type = "random"
        else:
            trial_type = "tpe"

        # Calculate improvement
        perf = trial["rounds_per_min"]
        improvement = ((perf - baseline_perf) / baseline_perf * 100) if baseline_perf > 0 else 0.0

        # Check if best so far
        is_best_so_far = (perf == max([t["rounds_per_min"] for t in stage_results["trials"][:trial["trial_number"]+1]]))

        experiments.append({
            "timestamp": stage_results["timestamp"],
            "stage_num": stage_num,
            "stage_mcts": stage_mcts,
            "phase": "bo_trial",
            "trial_num": trial["trial_number"],
            "trial_type": trial_type,
            "parallel_batch_size": trial["parallel_batch_size"],
            "batch_timeout_ms": trial["batch_timeout_ms"],
            "rounds_per_min": perf,
            "is_best_so_far": is_best_so_far,
            "improvement_vs_baseline": improvement,
            "workers": BASELINE_WORKERS,
            "num_determinizations": stage_results["best_config"]["num_determinizations"],
            "simulations_per_det": stage_results["best_config"]["simulations_per_det"],
            "optuna_state": trial["state"],
        })

    # Add validation runs
    if validation_results["success"]:
        for i, val_perf in enumerate(validation_results["all_results"]):
            experiments.append({
                "timestamp": datetime.now().isoformat(),
                "stage_num": stage_num,
                "stage_mcts": stage_mcts,
                "phase": "validation",
                "trial_num": i,
                "trial_type": "validation",
                "parallel_batch_size": stage_results["best_config"]["parallel_batch_size"],
                "batch_timeout_ms": stage_results["best_config"]["batch_timeout_ms"],
                "rounds_per_min": val_perf,
                "is_best_so_far": False,
                "improvement_vs_baseline": ((val_perf - baseline_perf) / baseline_perf * 100) if baseline_perf > 0 else 0.0,
                "workers": BASELINE_WORKERS,
                "num_determinizations": stage_results["best_config"]["num_determinizations"],
                "simulations_per_det": stage_results["best_config"]["simulations_per_det"],
                "optuna_state": "COMPLETE",
            })

    # Save experiments
    with open(experiments_file, 'w') as f:
        json.dump(experiments, f, indent=2)

    print(f"\nSaved results to {experiments_file}")


def save_checkpoint(output_dir: Path, completed_stages: List[int], stage_results: List[Dict]):
    """
    Save checkpoint for resume capability.

    Args:
        output_dir: Output directory
        completed_stages: List of completed stage numbers
        stage_results: List of stage result dictionaries
    """
    checkpoint_file = output_dir / "checkpoint.json"

    # Build best configs per stage
    best_per_stage = {}
    for result in stage_results:
        stage_num = result["stage_num"]
        best_per_stage[str(stage_num)] = {
            "parallel_batch_size": result["best_config"]["parallel_batch_size"],
            "batch_timeout_ms": result["best_config"]["batch_timeout_ms"],
            "rounds_per_min": result["best_performance"],
            "improvement": f"{result['improvement_pct']:+.1f}%",
        }

    checkpoint = {
        "completed_stages": completed_stages,
        "current_best_per_stage": best_per_stage,
        "timestamp": datetime.now().isoformat(),
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_file}")


def load_checkpoint(output_dir: Path) -> Optional[Dict]:
    """
    Load checkpoint if it exists.

    Args:
        output_dir: Output directory

    Returns:
        Checkpoint data or None
    """
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization autotune using Optuna TPE")
    parser.add_argument("--stages", type=str, default="1,2,3,4,5", help="Comma-separated stage numbers to run (default: 1,2,3,4,5)")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help=f"Number of trials per stage (default: {DEFAULT_N_TRIALS})")
    parser.add_argument("--n-startup", type=int, default=DEFAULT_N_STARTUP_TRIALS, help=f"Number of random startup trials (default: {DEFAULT_N_STARTUP_TRIALS})")
    parser.add_argument("--rounds-per-trial", type=int, default=DEFAULT_ROUNDS_PER_TRIAL, help=f"Rounds per trial (default: {DEFAULT_ROUNDS_PER_TRIAL})")
    parser.add_argument("--validation-rounds", type=int, default=DEFAULT_VALIDATION_ROUNDS, help=f"Rounds per validation run (default: {DEFAULT_VALIDATION_ROUNDS})")
    parser.add_argument("--validation-runs", type=int, default=DEFAULT_VALIDATION_RUNS, help=f"Number of validation runs (default: {DEFAULT_VALIDATION_RUNS})")
    parser.add_argument("--resume", type=str, help="Resume from results directory")

    args = parser.parse_args()

    # Parse stages
    stages_to_run = [int(s) for s in args.stages.split(",")]

    # Create output directory
    if args.resume:
        output_dir = Path(args.resume)
        print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/auto_tune_bo_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(output_dir) if args.resume else None
    completed_stages = checkpoint["completed_stages"] if checkpoint else []

    # Filter stages to run
    stages_to_run = [s for s in stages_to_run if s not in completed_stages]
    if not stages_to_run:
        print("All requested stages already completed!")
        return

    print(f"\nConfiguration:")
    print(f"  Stages to run: {stages_to_run}")
    print(f"  Trials per stage: {args.n_trials}")
    print(f"  Startup trials: {args.n_startup}")
    print(f"  Rounds per trial: {args.rounds_per_trial}")
    print(f"  Validation: {args.validation_runs} runs × {args.validation_rounds} rounds")

    # Run optimization for each stage
    all_stage_results = []

    for stage_num in stages_to_run:
        stage_det, stage_sims, stage_desc = CURRICULUM_STAGES[stage_num - 1]

        # Optimize stage
        stage_results = optimize_stage(
            stage_num=stage_num,
            stage_det=stage_det,
            stage_sims=stage_sims,
            stage_desc=stage_desc,
            n_trials=args.n_trials,
            n_startup_trials=args.n_startup,
            rounds_per_trial=args.rounds_per_trial,
            output_dir=output_dir,
        )

        # Validate best config
        best_config = BOConfig(**stage_results["best_config"])
        validation_results = run_validation(
            config=best_config,
            num_rounds=args.validation_rounds,
            num_runs=args.validation_runs,
        )

        # Save results
        save_results(output_dir, stage_results, validation_results)

        # Update checkpoint
        completed_stages.append(stage_num)
        all_stage_results.append(stage_results)
        save_checkpoint(output_dir, completed_stages, all_stage_results)

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"\nGenerate report with:")
    print(f"  python benchmarks/performance/auto_tune_bo_report.py {output_dir}")


if __name__ == "__main__":
    main()
