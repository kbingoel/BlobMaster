"""
Full diagnostic suite runner.

This script runs all diagnostic tools in sequence:
1. Profiling to identify bottlenecks
2. Comprehensive benchmarking across configurations
3. Visualization and analysis
4. Decision document generation

Run this to get a complete understanding of BlobNet training performance
and identify the optimal configuration for your hardware.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and report results."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("="*80 + "\n")

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )

        elapsed = time.time() - start
        print(f"\n✓ {description} completed in {elapsed:.1f}s")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n✗ {description} failed after {elapsed:.1f}s")
        print(f"Error: {e}")

        return False


def main():
    """Run full diagnostic suite."""

    print("="*80)
    print("BlobNet Training Performance Diagnostic Suite")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Change to ml directory
    ml_dir = Path(__file__).parent
    print(f"Working directory: {ml_dir}")

    total_start = time.time()

    # Step 1: Profiling
    print("\n" + "="*80)
    print("STEP 1: Profiling Self-Play Pipeline")
    print("="*80)
    print("This will identify where time is being spent in the code.")
    print("Expected duration: 5-10 minutes")

    success = run_command(
        [sys.executable, "profile_selfplay.py"],
        "Profiling Analysis"
    )

    if not success:
        print("\nWarning: Profiling failed. Continuing anyway...")

    # Step 2: Comprehensive Benchmarking
    print("\n" + "="*80)
    print("STEP 2: Comprehensive Benchmarking")
    print("="*80)
    print("This will test different worker counts and configurations.")
    print("Expected duration: 20-40 minutes")
    print()

    user_input = input("Continue with comprehensive benchmark? This may take 30+ minutes. (y/n): ")

    if user_input.lower() != 'y':
        print("\nSkipping comprehensive benchmark.")
        print("You can run it later with: python ml/benchmark_diagnostic.py")
    else:
        success = run_command(
            [sys.executable, "benchmark_diagnostic.py"],
            "Comprehensive Benchmark Suite"
        )

        if not success:
            print("\nERROR: Benchmark failed!")
            print("Please check the error messages above.")
            return

        # Step 3: Visualization
        print("\n" + "="*80)
        print("STEP 3: Visualizing Results")
        print("="*80)

        success = run_command(
            [sys.executable, "visualize_diagnostic.py"],
            "Result Visualization"
        )

        if not success:
            print("\nWarning: Visualization failed.")

    # Summary
    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("DIAGNOSTIC SUITE COMPLETE")
    print("="*80)
    print(f"Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nGenerated files:")
    files = [
        ("benchmark_diagnostic_results.csv", "Raw benchmark data"),
        ("benchmark_diagnostic_results.json", "Detailed benchmark data"),
        ("benchmark_worker_scaling.png", "Worker scaling plots"),
        ("benchmark_batching_comparison.png", "Batching comparison plots"),
        ("benchmark_game_complexity.png", "Game complexity plots"),
        ("profile_*.prof", "Profiling data"),
    ]

    for filename, description in files:
        path = Path(filename)
        if '*' in filename:
            # Glob pattern
            matching = list(ml_dir.glob(filename))
            if matching:
                for f in matching:
                    print(f"  ✓ {f.name:<40} - {description}")
        elif path.exists():
            print(f"  ✓ {filename:<40} - {description}")

    print("\nNext steps:")
    print("  1. Review the PNG visualizations to understand performance trends")
    print("  2. Check the CSV/JSON files for detailed data")
    print("  3. Read the recommendations in the visualization output")
    print("  4. Implement suggested architectural changes")
    print("  5. Re-run benchmarks to validate improvements")

    print("\nKey questions to answer:")
    print("  - What worker count gives best GPU utilization?")
    print("  - Does batching help or hurt at different scales?")
    print("  - What's the expected training time with optimal config?")
    print("  - Where are the bottlenecks (profiling data)?")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
