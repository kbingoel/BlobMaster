#!/usr/bin/env python3
"""
JSON-based version of auto_tune.py (no SQLite dependency)
Quick workaround for Python builds without sqlite3 support
"""

import json
import sys
from pathlib import Path

# Import the original auto_tune
sys.path.insert(0, str(Path(__file__).parent))
from auto_tune import *

# Override CheckpointManager to use JSON instead of SQLite
class JSONCheckpointManager:
    """JSON-based checkpoint manager (no SQLite required)"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / 'results.json'
        self.checkpoint_file = self.output_dir / 'checkpoint.json'

        # Load existing results
        self.results = self._load_results()

    def _load_results(self):
        """Load results from JSON file"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []

    def _save_results(self):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def save_result(self, phase: str, config: SweepConfig, num_rounds: int,
                    elapsed_sec: float, rounds_per_min: Optional[float],
                    success: bool, error_msg: Optional[str] = None,
                    variance: Optional[float] = None,
                    examples_per_round: Optional[float] = None,
                    cpu_percent: Optional[float] = None,
                    gpu_memory_mb: Optional[float] = None):
        """Save experiment result"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'config_hash': config.to_hash(),
            'num_rounds': num_rounds,
            'elapsed_sec': elapsed_sec,
            'rounds_per_min': rounds_per_min,
            'variance': variance,
            'success': success,
            'error_msg': error_msg,
            'config': config.to_dict(),
            'examples_per_round': examples_per_round,
            'cpu_percent': cpu_percent,
            'gpu_memory_mb': gpu_memory_mb
        }
        self.results.append(result)
        self._save_results()

    def save_checkpoint(self, phase: str, last_config: SweepConfig,
                       best_config: SweepConfig, best_perf: float):
        """Save checkpoint"""
        checkpoint = {
            'last_phase': phase,
            'last_config': last_config.to_dict(),
            'current_best_config': best_config.to_dict(),
            'current_best_perf': best_perf,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self) -> Optional[Tuple[str, SweepConfig, SweepConfig, float]]:
        """Load checkpoint"""
        if not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        return (
            checkpoint['last_phase'],
            SweepConfig(**checkpoint['last_config']),
            SweepConfig(**checkpoint['current_best_config']),
            checkpoint['current_best_perf']
        )

    def get_completed_configs(self, phase: Optional[str] = None) -> Set[str]:
        """Get hashes of completed configs"""
        completed = set()
        for result in self.results:
            if result['success'] and (phase is None or result['phase'] == phase):
                completed.add(result['config_hash'])
        return completed

    def get_successful_results(self, phase: Optional[str] = None) -> List[Dict]:
        """Get successful results"""
        successful = []
        for result in self.results:
            if not result['success'] or result['rounds_per_min'] is None:
                continue
            if phase and result['phase'] != phase:
                continue

            config = SweepConfig(**result['config'])
            successful.append({
                'config': config,
                'rounds_per_min': result['rounds_per_min'],
                'variance': result.get('variance', 0.0),
                'num_rounds': result['num_rounds'],
                'phase': result['phase'],
                'elapsed_sec': result['elapsed_sec'],
                'examples_per_round': result.get('examples_per_round'),
                'cpu_percent': result.get('cpu_percent'),
                'gpu_memory_mb': result.get('gpu_memory_mb')
            })

        # Sort by performance
        successful.sort(key=lambda x: x['rounds_per_min'], reverse=True)
        return successful

    def get_failed_results(self) -> List[Dict]:
        """Get failed results"""
        failed = []
        for result in self.results:
            if result['success']:
                continue

            config = SweepConfig(**result['config'])
            failed.append({
                'config': config,
                'error': result['error_msg'],
                'phase': result['phase']
            })
        return failed

    def close(self):
        """No-op for JSON (file already saved)"""
        pass


# Monkey-patch the main function to use JSON manager
original_main = main

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent auto-tune parameter sweep (JSON version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a JSON-based version that doesn't require SQLite.
Results are saved to JSON files instead of a database.

Examples:
  python auto_tune_json.py --quick
  python auto_tune_json.py --time-budget 360
        """
    )

    parser.add_argument('--time-budget', type=int, default=None,
                       help='Maximum runtime in minutes (default: unlimited)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory')
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

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available, switching to CPU")
        args.device = 'cpu'

    print("="*80)
    print("BlobMaster Auto-Tune Parameter Sweep (JSON version)")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Phases: {args.phases}")
    if args.time_budget:
        print(f"Time budget: {args.time_budget} minutes")
    print()

    # Initialize JSON checkpoint manager
    checkpoint_mgr = JSONCheckpointManager(str(output_dir))
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
    expected_baseline_perf = 741.0

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
        top_5 = all_results[:5]

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
            for failure in failures[:10]:
                config = failure['config']
                error = failure['error']
                print(f"  - {config}: {error}")

        print(f"\nResults saved to: {output_dir}")
        print(f"  - results.json (all results)")
        print(f"  - checkpoint.json (resume data)")

        checkpoint_mgr.close()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Progress has been saved to checkpoint")
        print(f"Resume with: python auto_tune_json.py --resume {output_dir}")
        checkpoint_mgr.close()
        return 130

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        checkpoint_mgr.close()
        return 1


if __name__ == '__main__':
    sys.exit(main())
