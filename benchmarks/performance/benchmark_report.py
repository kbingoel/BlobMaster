"""
Aggregate benchmark results and generate comprehensive report.

Reads results from:
- benchmark_selfplay_results.csv
- benchmark_training_results.json
- benchmark_iteration_results.json

Generates:
- BENCHMARK-Results.md (comprehensive markdown report)
- Console summary with recommendations

Usage:
    python ml/benchmark_report.py
    python ml/benchmark_report.py --selfplay path/to/selfplay.csv --training path/to/training.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class BenchmarkReportGenerator:
    """Generate comprehensive benchmark report from individual benchmark results."""

    def __init__(
        self,
        selfplay_csv: str,
        training_json: str,
        iteration_json: str,
    ):
        """
        Initialize report generator.

        Args:
            selfplay_csv: Path to self-play benchmark CSV
            training_json: Path to training benchmark JSON
            iteration_json: Path to iteration benchmark JSON
        """
        self.selfplay_csv = selfplay_csv
        self.training_json = training_json
        self.iteration_json = iteration_json

        # Load data
        self.selfplay_results = self._load_selfplay_results()
        self.training_results = self._load_training_results()
        self.iteration_results = self._load_iteration_results()

    def _load_selfplay_results(self) -> List[Dict[str, Any]]:
        """Load self-play benchmark results from CSV."""
        if not Path(self.selfplay_csv).exists():
            print(f"WARNING: Self-play results not found: {self.selfplay_csv}")
            return []

        results = []
        with open(self.selfplay_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                result = {
                    "num_workers": int(row["num_workers"]),
                    "mcts_config": row["mcts_config"],
                    "num_determinizations": int(row["num_determinizations"]),
                    "simulations_per_det": int(row["simulations_per_det"]),
                    "total_sims_per_move": int(row["total_sims_per_move"]),
                    "games_per_minute": float(row["games_per_minute"]),
                    "seconds_per_game": float(row["seconds_per_game"]),
                    "cpu_percent": float(row["cpu_percent"]),
                }
                results.append(result)

        print(f"Loaded {len(results)} self-play results from {self.selfplay_csv}")
        return results

    def _load_training_results(self) -> List[Dict[str, Any]]:
        """Load training benchmark results from JSON."""
        if not Path(self.training_json).exists():
            print(f"WARNING: Training results not found: {self.training_json}")
            return []

        with open(self.training_json, "r") as f:
            results = json.load(f)

        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]

        print(f"Loaded {len(valid_results)} training results from {self.training_json}")
        return valid_results

    def _load_iteration_results(self) -> Optional[Dict[str, Any]]:
        """Load iteration benchmark results from JSON."""
        if not Path(self.iteration_json).exists():
            print(f"WARNING: Iteration results not found: {self.iteration_json}")
            return None

        with open(self.iteration_json, "r") as f:
            results = json.load(f)

        print(f"Loaded iteration results from {self.iteration_json}")
        return results

    def generate_report(self) -> str:
        """
        Generate comprehensive benchmark report.

        Returns:
            Markdown report string
        """
        lines = []

        # Header
        lines.append("# BlobMaster Training Pipeline Performance Benchmark")
        lines.append("")
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("**Hardware**: AMD 7950X (16 cores/32 threads) + RTX 4060 (8GB VRAM)")
        lines.append("**Phase**: 4 Sessions 1-5 Complete (Training Pipeline Implemented)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Self-Play Performance
        if self.selfplay_results:
            lines.extend(self._generate_selfplay_section())
        else:
            lines.append("## Self-Play Performance")
            lines.append("")
            lines.append("*No self-play benchmark results available*")
            lines.append("")

        # Training Performance
        if self.training_results:
            lines.extend(self._generate_training_section())
        else:
            lines.append("## Training Performance")
            lines.append("")
            lines.append("*No training benchmark results available*")
            lines.append("")

        # Iteration Performance
        if self.iteration_results:
            lines.extend(self._generate_iteration_section())
        else:
            lines.append("## Iteration Performance")
            lines.append("")
            lines.append("*No iteration benchmark results available*")
            lines.append("")

        # Recommendations
        lines.extend(self._generate_recommendations_section())

        # Conclusion
        lines.extend(self._generate_conclusion_section())

        return "\n".join(lines)

    def _generate_selfplay_section(self) -> List[str]:
        """Generate self-play performance section."""
        lines = []
        lines.append("## Self-Play Performance")
        lines.append("")

        # Table header
        lines.append("| Workers | MCTS Config | Sims/Move | Games/Min | Sec/Game | CPU% |")
        lines.append("|---------|-------------|-----------|-----------|----------|------|")

        # Table rows (show medium config for all worker counts)
        medium_results = [r for r in self.selfplay_results if r["mcts_config"] == "medium"]
        for r in medium_results:
            lines.append(f"| {r['num_workers']:<7} | "
                        f"{r['mcts_config']:<11} | "
                        f"{r['total_sims_per_move']:<9} | "
                        f"{r['games_per_minute']:<9.1f} | "
                        f"{r['seconds_per_game']:<8.2f} | "
                        f"{r['cpu_percent']:<4.0f} |")

        lines.append("")

        # Find best worker configuration
        if medium_results:
            best = max(medium_results, key=lambda x: x["games_per_minute"])
            lines.append(f"**Best Worker Configuration**: {best['num_workers']} workers")
            lines.append(f"- Games per minute: {best['games_per_minute']:.1f}")
            lines.append(f"- CPU utilization: {best['cpu_percent']:.0f}%")
            lines.append("")

            # Compare with 32 workers if available
            workers_32 = next((r for r in medium_results if r["num_workers"] == 32), None)
            workers_16 = next((r for r in medium_results if r["num_workers"] == 16), None)
            if workers_32 and workers_16:
                speedup = workers_32["games_per_minute"] / workers_16["games_per_minute"]
                lines.append(f"**Hyperthreading Impact**: 32 workers is {speedup:.2f}x faster than 16 workers")
                if speedup < 1.15:
                    lines.append(f"- Recommendation: Use 16 workers (minimal benefit from hyperthreading)")
                else:
                    lines.append(f"- Recommendation: Use 32 workers (significant hyperthreading benefit)")
                lines.append("")

        # MCTS configuration comparison (at best worker count)
        if medium_results:
            best_workers = best["num_workers"]
            worker_results = [r for r in self.selfplay_results if r["num_workers"] == best_workers]
            worker_results_sorted = sorted(worker_results, key=lambda x: x["total_sims_per_move"])

            lines.append(f"**MCTS Configuration Impact** (at {best_workers} workers):")
            lines.append("")
            lines.append("| Config | Sims/Move | Games/Min | Relative Speed |")
            lines.append("|--------|-----------|-----------|----------------|")

            baseline = worker_results_sorted[0]["games_per_minute"] if worker_results_sorted else 1.0
            for r in worker_results_sorted:
                relative_speed = r["games_per_minute"] / baseline
                lines.append(f"| {r['mcts_config']:<6} | "
                            f"{r['total_sims_per_move']:<9} | "
                            f"{r['games_per_minute']:<9.1f} | "
                            f"{relative_speed:<14.2f}x |")

            lines.append("")

        return lines

    def _generate_training_section(self) -> List[str]:
        """Generate training performance section."""
        lines = []
        lines.append("## Training Performance")
        lines.append("")

        # Table header
        lines.append("| Device | Batch Size | Precision | Batches/Sec | Examples/Sec | VRAM (MB) |")
        lines.append("|--------|------------|-----------|-------------|--------------|-----------|")

        # Table rows
        for r in self.training_results:
            vram_str = f"{r['peak_memory_mb']:.0f}" if r['peak_memory_mb'] > 0 else "N/A"
            lines.append(f"| {r['device']:<6} | "
                        f"{r['batch_size']:<10} | "
                        f"{r['precision']:<9} | "
                        f"{r['batches_per_second']:<11.2f} | "
                        f"{r['examples_per_second']:<12.1f} | "
                        f"{vram_str:<9} |")

        lines.append("")

        # Find best CUDA configuration
        cuda_results = [r for r in self.training_results if r["device"] == "cuda"]
        if cuda_results:
            best_cuda = max(cuda_results, key=lambda x: x["examples_per_second"])
            lines.append(f"**Best CUDA Configuration**: batch_size={best_cuda['batch_size']}, "
                        f"precision={best_cuda['precision']}")
            lines.append(f"- Throughput: {best_cuda['examples_per_second']:.0f} examples/sec")
            lines.append(f"- Peak VRAM: {best_cuda['peak_memory_mb']:.0f} MB")
            lines.append("")

            # FP16 vs FP32 comparison
            fp32_results = [r for r in cuda_results if r["precision"] == "fp32"]
            fp16_results = [r for r in cuda_results if r["precision"] == "fp16"]

            if fp32_results and fp16_results:
                lines.append("**FP16 vs FP32 Speedup**:")
                lines.append("")
                for fp32 in fp32_results:
                    batch = fp32["batch_size"]
                    fp16 = next((r for r in fp16_results if r["batch_size"] == batch), None)
                    if fp16:
                        speedup = fp16["examples_per_second"] / fp32["examples_per_second"]
                        vram_reduction = fp32["peak_memory_mb"] - fp16["peak_memory_mb"]
                        lines.append(f"- Batch {batch}: {speedup:.2f}x faster, "
                                    f"{vram_reduction:.0f} MB less VRAM")
                lines.append("")

        # GPU vs CPU comparison
        cpu_results = [r for r in self.training_results if r["device"] == "cpu"]
        if cpu_results and cuda_results:
            best_cpu = max(cpu_results, key=lambda x: x["examples_per_second"])
            best_cuda = max(cuda_results, key=lambda x: x["examples_per_second"])
            speedup = best_cuda["examples_per_second"] / best_cpu["examples_per_second"]
            lines.append(f"**GPU vs CPU**: {speedup:.1f}x faster on CUDA")
            lines.append("")

        return lines

    def _generate_iteration_section(self) -> List[str]:
        """Generate iteration performance section."""
        lines = []
        lines.append("## End-to-End Iteration Performance")
        lines.append("")

        if not self.iteration_results:
            return lines

        summary = self.iteration_results["iteration_summary"]
        proj = self.iteration_results["projections"]

        # Current iteration breakdown
        lines.append("### Iteration Breakdown (1/10th scale)")
        lines.append("")
        lines.append(f"- **Self-play**: {summary['selfplay_time_min']:.1f} minutes "
                    f"({summary['selfplay_percentage']:.0f}% of total)")
        lines.append(f"- **Training**: {summary['training_time_min']:.1f} minutes "
                    f"({summary['training_percentage']:.0f}% of total)")
        lines.append(f"- **Total**: {summary['total_time_min']:.1f} minutes")
        lines.append("")

        # Projections
        lines.append("### Projected Full-Scale Performance")
        lines.append("")
        lines.append(f"**Single Iteration** (10,000 games):")
        lines.append(f"- Self-play: {proj['projected_selfplay_time_min']:.1f} minutes")
        lines.append(f"- Training: {proj['projected_training_time_min']:.1f} minutes")
        lines.append(f"- **Total**: {proj['projected_iteration_time_min']:.1f} minutes")
        lines.append("")

        lines.append(f"**Full Training Run** ({proj['total_iterations']} iterations):")
        lines.append(f"- Total time: {proj['projected_total_time_hours']:.1f} hours")
        lines.append(f"- Total time: **{proj['projected_total_time_days']:.1f} days**")
        lines.append("")

        # Bottleneck analysis
        lines.append("### Bottleneck Analysis")
        lines.append("")
        selfplay_pct = summary['selfplay_percentage']
        if selfplay_pct > 70:
            lines.append(f"- **Self-play is the primary bottleneck** ({selfplay_pct:.0f}% of iteration time)")
            lines.append(f"- GPU-batched MCTS would significantly reduce training time")
        elif selfplay_pct > 50:
            lines.append(f"- **Self-play is the main bottleneck** ({selfplay_pct:.0f}% of iteration time)")
            lines.append(f"- GPU-batched MCTS optimization would be beneficial")
        else:
            lines.append(f"- Self-play and training are balanced")
            lines.append(f"- Current configuration is reasonable")
        lines.append("")

        return lines

    def _generate_recommendations_section(self) -> List[str]:
        """Generate recommendations section."""
        lines = []
        lines.append("## Recommendations")
        lines.append("")

        # Self-play recommendations
        if self.selfplay_results:
            medium_results = [r for r in self.selfplay_results if r["mcts_config"] == "medium"]
            if medium_results:
                best = max(medium_results, key=lambda x: x["games_per_minute"])
                lines.append(f"### Self-Play Configuration")
                lines.append(f"- **Workers**: {best['num_workers']}")
                lines.append(f"- **MCTS**: {best['mcts_config']} ({best['total_sims_per_move']} sims/move)")
                lines.append(f"- **Expected throughput**: {best['games_per_minute']:.0f} games/minute")
                lines.append("")

        # Training recommendations
        if self.training_results:
            cuda_results = [r for r in self.training_results if r["device"] == "cuda"]
            if cuda_results:
                best_cuda = max(cuda_results, key=lambda x: x["examples_per_second"])
                lines.append(f"### Training Configuration")
                lines.append(f"- **Device**: CUDA")
                lines.append(f"- **Batch size**: {best_cuda['batch_size']}")
                lines.append(f"- **Precision**: {best_cuda['precision'].upper()}")
                lines.append(f"- **Expected throughput**: {best_cuda['examples_per_second']:.0f} examples/sec")
                lines.append("")

        # GPU-batched MCTS recommendation
        if self.iteration_results:
            proj = self.iteration_results["projections"]
            summary = self.iteration_results["iteration_summary"]

            lines.append(f"### GPU-Batched MCTS Priority")
            lines.append("")

            days = proj['projected_total_time_days']
            selfplay_pct = summary['selfplay_percentage']

            if days > 20 or selfplay_pct > 70:
                lines.append("**PRIORITY: HIGH**")
                lines.append("")
                lines.append(f"- Current estimate: {days:.1f} days for 500 iterations")
                lines.append(f"- Self-play bottleneck: {selfplay_pct:.0f}% of iteration time")
                lines.append(f"- GPU-batched MCTS could reduce to ~{days / 3:.1f} days (3-6x speedup)")
                lines.append(f"- **Recommendation**: Implement GPU-batched MCTS before starting training")
            elif days > 12 or selfplay_pct > 60:
                lines.append("**PRIORITY: MEDIUM**")
                lines.append("")
                lines.append(f"- Current estimate: {days:.1f} days for 500 iterations")
                lines.append(f"- Self-play bottleneck: {selfplay_pct:.0f}% of iteration time")
                lines.append(f"- GPU-batched MCTS could reduce to ~{days / 3:.1f} days")
                lines.append(f"- **Recommendation**: Consider GPU-batched MCTS if you want <10 day training")
            else:
                lines.append("**PRIORITY: LOW**")
                lines.append("")
                lines.append(f"- Current estimate: {days:.1f} days for 500 iterations")
                lines.append(f"- Training time is acceptable")
                lines.append(f"- **Recommendation**: Proceed with current implementation, optimize later if needed")

            lines.append("")

        return lines

    def _generate_conclusion_section(self) -> List[str]:
        """Generate conclusion section."""
        lines = []
        lines.append("## Conclusion")
        lines.append("")

        if self.iteration_results:
            proj = self.iteration_results["projections"]
            lines.append(f"Based on these benchmarks, a full training run of 500 iterations is estimated to take "
                        f"**{proj['projected_total_time_days']:.1f} days** on your hardware (AMD 7950X + RTX 4060).")
            lines.append("")

        lines.append("**Next Steps**:")
        lines.append("")
        lines.append("1. Review the recommendations above")
        lines.append("2. Decide on GPU-batched MCTS implementation priority")
        lines.append("3. Configure training pipeline with optimal hyperparameters")
        lines.append("4. Start initial training run")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Report generated by BlobMaster benchmark suite*")

        return lines

    def print_console_summary(self):
        """Print summary to console."""
        print("\n" + "="*70)
        print("BENCHMARK REPORT SUMMARY")
        print("="*70 + "\n")

        # Self-play
        if self.selfplay_results:
            medium_results = [r for r in self.selfplay_results if r["mcts_config"] == "medium"]
            if medium_results:
                best = max(medium_results, key=lambda x: x["games_per_minute"])
                print(f"Self-Play (best config):")
                print(f"  - Workers: {best['num_workers']}")
                print(f"  - Games/minute: {best['games_per_minute']:.1f}")
                print(f"  - CPU: {best['cpu_percent']:.0f}%")
                print()

        # Training
        if self.training_results:
            cuda_results = [r for r in self.training_results if r["device"] == "cuda"]
            if cuda_results:
                best = max(cuda_results, key=lambda x: x["examples_per_second"])
                print(f"Training (best config):")
                print(f"  - Batch size: {best['batch_size']}")
                print(f"  - Precision: {best['precision']}")
                print(f"  - Examples/sec: {best['examples_per_second']:.0f}")
                print(f"  - VRAM: {best['peak_memory_mb']:.0f} MB")
                print()

        # Iteration
        if self.iteration_results:
            proj = self.iteration_results["projections"]
            summary = self.iteration_results["iteration_summary"]
            print(f"Full Training Estimate:")
            print(f"  - Minutes/iteration: {proj['projected_iteration_time_min']:.1f}")
            print(f"  - Total time: {proj['projected_total_time_days']:.1f} days")
            print(f"  - Bottleneck: Self-play ({summary['selfplay_percentage']:.0f}%)")
            print()

        print("="*70 + "\n")


def main():
    """Generate benchmark report."""
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--selfplay",
        type=str,
        default="benchmark_selfplay_results.csv",
        help="Self-play results CSV",
    )
    parser.add_argument(
        "--training",
        type=str,
        default="benchmark_training_results.json",
        help="Training results JSON",
    )
    parser.add_argument(
        "--iteration",
        type=str,
        default="benchmark_iteration_results.json",
        help="Iteration results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="BENCHMARK-Results.md",
        help="Output markdown file",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("BENCHMARK REPORT GENERATOR")
    print("="*70)
    print(f"Self-play results: {args.selfplay}")
    print(f"Training results: {args.training}")
    print(f"Iteration results: {args.iteration}")
    print(f"Output: {args.output}")
    print("="*70 + "\n")

    # Generate report
    generator = BenchmarkReportGenerator(
        selfplay_csv=args.selfplay,
        training_json=args.training,
        iteration_json=args.iteration,
    )

    report = generator.generate_report()

    # Save to file
    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {args.output}")

    # Print console summary
    generator.print_console_summary()

    print("="*70)
    print("REPORT GENERATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
