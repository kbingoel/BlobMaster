#!/usr/bin/env python3
"""
Generate comprehensive report from MPPT auto-tune results

This script reads the JSON files from auto_tune_mppt.py and generates:
- Markdown summary report with tables for each curriculum stage
- Individual parameter optimization progression tables
- Combination testing results
- Final recommendations

Usage:
    python auto_tune_mppt_report.py results/auto_tune_mppt_20251114_153000
    python auto_tune_mppt_report.py results/auto_tune_mppt_20251114_153000 --output custom_report.md
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class MPPTConfig:
    """Configuration for a single test"""
    workers: int
    parallel_batch_size: int
    num_determinizations: int
    simulations_per_det: int
    batch_timeout_ms: int = 10

    def __str__(self) -> str:
        return f"{self.workers}w × {self.parallel_batch_size}batch × {self.num_determinizations}×{self.simulations_per_det} × {self.batch_timeout_ms}ms"


def load_results_from_json(results_dir: str) -> List[Dict]:
    """Load all experiment results from JSON file"""
    results_path = Path(results_dir)
    experiments_file = results_path / "experiments.json"

    if not experiments_file.exists():
        print(f"ERROR: experiments.json not found in {results_dir}")
        return []

    with open(experiments_file, 'r') as f:
        experiments = json.load(f)

    return experiments


def group_by_stage(experiments: List[Dict]) -> Dict[int, List[Dict]]:
    """Group experiments by curriculum stage"""
    stages = defaultdict(list)

    for exp in experiments:
        stage_num = exp.get('stage_num')
        if stage_num is not None:
            stages[stage_num].append(exp)

    return dict(stages)


def generate_stage_report(stage_num: int, experiments: List[Dict]) -> List[str]:
    """Generate report section for a single curriculum stage"""
    report = []

    # Get MCTS configuration from any experiment
    if not experiments:
        return report

    num_det = experiments[0]['num_determinizations']
    sims_per_det = experiments[0]['simulations_per_det']
    total_sims = num_det * sims_per_det

    report.append(f"\n## Stage {stage_num}: {num_det}×{sims_per_det} MCTS ({total_sims} total simulations)")
    report.append("")

    # Group by phase
    baseline_exps = [e for e in experiments if e['phase'] == 'baseline']
    individual_coarse = [e for e in experiments if e['phase'] == 'individual_coarse']
    individual_fine = [e for e in experiments if e['phase'] == 'individual_fine']
    combination = [e for e in experiments if e['phase'] == 'combination']
    final = [e for e in experiments if e['phase'] == 'final']

    # 1. Baseline
    if baseline_exps:
        baseline = baseline_exps[0]
        baseline_perf = baseline.get('rounds_per_min', 0.0)
        baseline_config = MPPTConfig(
            workers=baseline['workers'],
            parallel_batch_size=baseline['parallel_batch_size'],
            num_determinizations=baseline['num_determinizations'],
            simulations_per_det=baseline['simulations_per_det'],
            batch_timeout_ms=baseline['batch_timeout_ms']
        )

        report.append("### Baseline Configuration")
        report.append("")
        report.append(f"- **Config**: {baseline_config}")
        report.append(f"- **Performance**: {baseline_perf:.1f} rounds/min")
        report.append("")

    # 2. Individual parameter optimization
    report.append("### Individual Parameter Optimization")
    report.append("")

    # Group individual results by parameter
    param_results = defaultdict(lambda: {'coarse': [], 'fine': []})

    for exp in individual_coarse:
        param_name = exp.get('param_name')
        if param_name:
            param_results[param_name]['coarse'].append(exp)

    for exp in individual_fine:
        param_name = exp.get('param_name')
        if param_name:
            param_results[param_name]['fine'].append(exp)

    # Generate tables for each parameter
    for param_name in ['workers', 'parallel_batch_size', 'batch_timeout_ms']:
        if param_name not in param_results:
            continue

        report.append(f"#### Parameter: `{param_name}`")
        report.append("")

        # Coarse search results
        coarse_results = param_results[param_name]['coarse']
        if coarse_results:
            report.append("**Coarse Search (±4 steps):**")
            report.append("")
            report.append("| Value | Search Type | Rounds/Min | vs Baseline | Status |")
            report.append("|-------|-------------|------------|-------------|--------|")

            for exp in sorted(coarse_results, key=lambda x: x.get(param_name, 0)):
                value = exp.get(param_name, 0)
                search_type = exp.get('search_type', 'unknown')
                perf = exp.get('rounds_per_min', 0.0) if exp['success'] else 0.0
                status = "✓" if exp['success'] else f"✗ {exp.get('error_msg', 'Failed')[:30]}"

                if exp['success'] and baseline_perf > 0:
                    vs_baseline = ((perf / baseline_perf) - 1) * 100
                    vs_baseline_str = f"{vs_baseline:+.1f}%"
                    perf_str = f"{perf:.1f}"
                else:
                    vs_baseline_str = "—"
                    perf_str = "—"

                report.append(f"| {value} | {search_type} | {perf_str} | {vs_baseline_str} | {status} |")

            report.append("")

        # Fine search results
        fine_results = param_results[param_name]['fine']
        if fine_results:
            report.append("**Fine Search (±1 step):**")
            report.append("")
            report.append("| Value | Search Type | Rounds/Min | vs Baseline | Status |")
            report.append("|-------|-------------|------------|-------------|--------|")

            for exp in sorted(fine_results, key=lambda x: x.get(param_name, 0)):
                value = exp.get(param_name, 0)
                search_type = exp.get('search_type', 'unknown')
                perf = exp.get('rounds_per_min', 0.0) if exp['success'] else 0.0
                status = "✓" if exp['success'] else f"✗ {exp.get('error_msg', 'Failed')[:30]}"

                if exp['success'] and baseline_perf > 0:
                    vs_baseline = ((perf / baseline_perf) - 1) * 100
                    vs_baseline_str = f"{vs_baseline:+.1f}%"
                    perf_str = f"{perf:.1f}"
                else:
                    vs_baseline_str = "—"
                    perf_str = "—"

                report.append(f"| {value} | {search_type} | {perf_str} | {vs_baseline_str} | {status} |")

            report.append("")

            # Find optimal value
            successful_fine = [e for e in fine_results if e['success']]
            if successful_fine:
                best_fine = max(successful_fine, key=lambda x: x['rounds_per_min'])
                best_value = best_fine.get(param_name, 0)
                best_perf = best_fine['rounds_per_min']
                improvement = ((best_perf / baseline_perf) - 1) * 100 if baseline_perf > 0 else 0

                report.append(f"**Optimal `{param_name}`: {best_value}** ({best_perf:.1f} r/min, {improvement:+.1f}% vs baseline)")
                report.append("")

    # 3. Combination testing
    if combination:
        report.append("### Combination Testing")
        report.append("")
        report.append("| Config | Search Type | Rounds/Min | vs Baseline | Status |")
        report.append("|--------|-------------|------------|-------------|--------|")

        for exp in sorted(combination, key=lambda x: x.get('rounds_per_min', 0.0) if x['success'] else 0.0, reverse=True):
            config = MPPTConfig(
                workers=exp['workers'],
                parallel_batch_size=exp['parallel_batch_size'],
                num_determinizations=exp['num_determinizations'],
                simulations_per_det=exp['simulations_per_det'],
                batch_timeout_ms=exp['batch_timeout_ms']
            )
            config_str = f"{exp['workers']}w×{exp['parallel_batch_size']}b×{exp['batch_timeout_ms']}ms"
            search_type = exp.get('search_type', 'unknown')
            perf = exp.get('rounds_per_min', 0.0) if exp['success'] else 0.0
            status = "✓" if exp['success'] else f"✗ {exp.get('error_msg', 'Failed')[:30]}"

            if exp['success'] and baseline_perf > 0:
                vs_baseline = ((perf / baseline_perf) - 1) * 100
                vs_baseline_str = f"{vs_baseline:+.1f}%"
                perf_str = f"{perf:.1f}"
            else:
                vs_baseline_str = "—"
                perf_str = "—"

            report.append(f"| {config_str} | {search_type} | {perf_str} | {vs_baseline_str} | {status} |")

        report.append("")

    # 4. Final validation
    if final:
        report.append("### Final Validation (3 runs × 1000 rounds)")
        report.append("")

        # Get config from first final experiment
        final_config = MPPTConfig(
            workers=final[0]['workers'],
            parallel_batch_size=final[0]['parallel_batch_size'],
            num_determinizations=final[0]['num_determinizations'],
            simulations_per_det=final[0]['simulations_per_det'],
            batch_timeout_ms=final[0]['batch_timeout_ms']
        )

        report.append(f"**Final Config**: {final_config}")
        report.append("")

        # Calculate statistics
        perfs = [e['rounds_per_min'] for e in final if e['success']]
        if perfs:
            avg_perf = sum(perfs) / len(perfs)
            variance = (sum((x - avg_perf) ** 2 for x in perfs) / len(perfs)) ** 0.5
            variance_pct = (variance / avg_perf) * 100 if avg_perf > 0 else 0
            improvement = ((avg_perf / baseline_perf) - 1) * 100 if baseline_perf > 0 else 0

            report.append("| Run | Rounds/Min |")
            report.append("|-----|------------|")
            for idx, perf in enumerate(perfs):
                report.append(f"| {idx+1} | {perf:.1f} |")

            report.append("")
            report.append(f"- **Average**: {avg_perf:.1f} rounds/min")
            report.append(f"- **Variance**: ±{variance_pct:.1f}%")
            report.append(f"- **Improvement vs Baseline**: {improvement:+.1f}%")
            report.append("")

    # 5. Summary
    report.append("### Stage Summary")
    report.append("")

    if baseline_exps and final:
        final_perfs = [e['rounds_per_min'] for e in final if e['success']]
        if final_perfs:
            baseline_perf = baseline_exps[0].get('rounds_per_min', 0.0)
            final_perf = sum(final_perfs) / len(final_perfs)
            improvement = ((final_perf / baseline_perf) - 1) * 100 if baseline_perf > 0 else 0

            final_config = MPPTConfig(
                workers=final[0]['workers'],
                parallel_batch_size=final[0]['parallel_batch_size'],
                num_determinizations=final[0]['num_determinizations'],
                simulations_per_det=final[0]['simulations_per_det'],
                batch_timeout_ms=final[0]['batch_timeout_ms']
            )

            report.append("| Metric | Baseline | Optimal | Change |")
            report.append("|--------|----------|---------|--------|")
            report.append(f"| Performance | {baseline_perf:.1f} r/min | {final_perf:.1f} r/min | {improvement:+.1f}% |")
            report.append(f"| Workers | {baseline_exps[0]['workers']} | {final[0]['workers']} | {final[0]['workers'] - baseline_exps[0]['workers']:+d} |")
            report.append(f"| Batch Size | {baseline_exps[0]['parallel_batch_size']} | {final[0]['parallel_batch_size']} | {final[0]['parallel_batch_size'] - baseline_exps[0]['parallel_batch_size']:+d} |")
            report.append(f"| Timeout (ms) | {baseline_exps[0]['batch_timeout_ms']} | {final[0]['batch_timeout_ms']} | {final[0]['batch_timeout_ms'] - baseline_exps[0]['batch_timeout_ms']:+d} |")
            report.append("")

            # Recommendation
            if improvement > 10:
                recommendation = f"✅ **Strong improvement**: {improvement:.1f}% speedup achieved"
            elif improvement > 5:
                recommendation = f"✓ **Moderate improvement**: {improvement:.1f}% speedup achieved"
            elif improvement > 0:
                recommendation = f"○ **Marginal improvement**: {improvement:.1f}% speedup achieved"
            else:
                recommendation = f"⚠️ **No improvement**: Baseline configuration is optimal"

            report.append(f"**Recommendation**: {recommendation}")
            report.append("")

    report.append("---")
    report.append("")

    return report


def generate_full_report(results_dir: str, output_path: str):
    """Generate comprehensive markdown report"""
    experiments = load_results_from_json(results_dir)

    if not experiments:
        print("ERROR: No experiments found")
        return

    stages = group_by_stage(experiments)

    report = []

    # Header
    report.append("# BlobMaster MPPT Auto-Tune Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("This report shows MPPT (Maximum Power Point Tracking) style optimization results for each curriculum stage.")
    report.append("Each stage uses coarse-to-fine gradient ascent to find optimal parameters.")
    report.append("")

    # Overview
    report.append("## Overview")
    report.append("")
    report.append(f"- **Total experiments**: {len(experiments)}")
    report.append(f"- **Stages optimized**: {len(stages)}")
    report.append(f"- **Successful tests**: {sum(1 for e in experiments if e['success'])}")
    report.append(f"- **Failed tests**: {sum(1 for e in experiments if not e['success'])}")
    report.append("")

    # Overall summary table
    report.append("## Summary Across All Stages")
    report.append("")
    report.append("| Stage | MCTS | Baseline (r/min) | Optimal (r/min) | Improvement | Optimal Config |")
    report.append("|-------|------|------------------|-----------------|-------------|----------------|")

    for stage_num in sorted(stages.keys()):
        stage_exps = stages[stage_num]
        baseline = [e for e in stage_exps if e['phase'] == 'baseline']
        final = [e for e in stage_exps if e['phase'] == 'final' and e['success']]

        if baseline and final:
            num_det = baseline[0]['num_determinizations']
            sims_per_det = baseline[0]['simulations_per_det']
            baseline_perf = baseline[0]['rounds_per_min']
            final_perfs = [e['rounds_per_min'] for e in final]
            final_perf = sum(final_perfs) / len(final_perfs)
            improvement = ((final_perf / baseline_perf) - 1) * 100

            config_str = f"{final[0]['workers']}w×{final[0]['parallel_batch_size']}b×{final[0]['batch_timeout_ms']}ms"

            report.append(f"| {stage_num} | {num_det}×{sims_per_det} | {baseline_perf:.1f} | {final_perf:.1f} | {improvement:+.1f}% | {config_str} |")

    report.append("")

    # Detailed stage reports
    report.append("---")
    report.append("")
    report.append("# Detailed Stage Reports")
    report.append("")

    for stage_num in sorted(stages.keys()):
        stage_report = generate_stage_report(stage_num, stages[stage_num])
        report.extend(stage_report)

    # Recommendations
    report.append("---")
    report.append("")
    report.append("## Overall Recommendations")
    report.append("")

    # Find best improvements
    improvements = []
    for stage_num in sorted(stages.keys()):
        stage_exps = stages[stage_num]
        baseline = [e for e in stage_exps if e['phase'] == 'baseline']
        final = [e for e in stage_exps if e['phase'] == 'final' and e['success']]

        if baseline and final:
            baseline_perf = baseline[0]['rounds_per_min']
            final_perfs = [e['rounds_per_min'] for e in final]
            final_perf = sum(final_perfs) / len(final_perfs)
            improvement = ((final_perf / baseline_perf) - 1) * 100
            improvements.append((stage_num, improvement, final[0]))

    if improvements:
        improvements.sort(key=lambda x: x[1], reverse=True)

        report.append("### Key Findings")
        report.append("")

        # Best improvement
        best_stage, best_improvement, _ = improvements[0]
        report.append(f"1. **Best improvement**: Stage {best_stage} achieved {best_improvement:+.1f}% speedup")

        # Average improvement
        avg_improvement = sum(imp for _, imp, _ in improvements) / len(improvements)
        report.append(f"2. **Average improvement**: {avg_improvement:+.1f}% across all stages")

        # Consistency
        variances = []
        for stage_num in sorted(stages.keys()):
            final = [e for e in stages[stage_num] if e['phase'] == 'final' and e['success']]
            if final:
                variance = final[0].get('variance', 0.0)
                variances.append(variance)

        if variances:
            avg_variance = sum(variances) / len(variances)
            report.append(f"3. **Stability**: Average variance {avg_variance:.1f}% across final validations")

        report.append("")

        # Export configurations
        report.append("### Recommended Configurations")
        report.append("")
        report.append("Add to `ml/config.py`:")
        report.append("")
        report.append("```python")
        report.append(f"# MPPT Auto-tuned configurations (generated {datetime.now().strftime('%Y-%m-%d')})")
        report.append("")

        for stage_num in sorted(stages.keys()):
            final = [e for e in stages[stage_num] if e['phase'] == 'final' and e['success']]
            if final:
                exp = final[0]
                num_det = exp['num_determinizations']
                sims_per_det = exp['simulations_per_det']

                report.append(f"# Stage {stage_num}: {num_det}×{sims_per_det} MCTS")
                report.append(f"STAGE_{stage_num}_WORKERS = {exp['workers']}")
                report.append(f"STAGE_{stage_num}_BATCH_SIZE = {exp['parallel_batch_size']}")
                report.append(f"STAGE_{stage_num}_TIMEOUT_MS = {exp['batch_timeout_ms']}")
                report.append("")

        report.append("```")
        report.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Report written to: {output_path}")


def generate_summary_csv(results_dir: str, output_path: str):
    """Generate summary CSV for quick reference"""
    experiments = load_results_from_json(results_dir)

    if not experiments:
        return

    stages = group_by_stage(experiments)

    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Stage', 'MCTS_Config', 'Phase', 'Workers', 'Batch_Size', 'Timeout_ms',
            'Rounds_Per_Min', 'Variance_%', 'Success', 'Param_Name', 'Search_Type'
        ])

        for stage_num in sorted(stages.keys()):
            for exp in stages[stage_num]:
                mcts_config = f"{exp['num_determinizations']}x{exp['simulations_per_det']}"
                writer.writerow([
                    stage_num,
                    mcts_config,
                    exp['phase'],
                    exp['workers'],
                    exp['parallel_batch_size'],
                    exp['batch_timeout_ms'],
                    exp.get('rounds_per_min', 0.0),
                    exp.get('variance', 0.0),
                    exp['success'],
                    exp.get('param_name', ''),
                    exp.get('search_type', '')
                ])

    print(f"✓ Summary CSV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate report from MPPT auto-tune results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('results_dir', type=str,
                       help='Path to results directory (e.g., results/auto_tune_mppt_20251114_153000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output markdown file (default: mppt_report.md in results dir)')

    args = parser.parse_args()

    # Check results directory exists
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    experiments_file = results_dir / "experiments.json"
    if not experiments_file.exists():
        print(f"ERROR: experiments.json not found in {results_dir}")
        return 1

    # Setup output paths
    if args.output:
        report_path = args.output
    else:
        report_path = str(results_dir / 'mppt_report.md')

    summary_csv_path = str(results_dir / 'mppt_summary.csv')

    print(f"Loading results from: {results_dir}")

    # Generate report
    print("Generating markdown report...")
    generate_full_report(str(results_dir), report_path)

    # Generate summary CSV
    print("Generating summary CSV...")
    generate_summary_csv(str(results_dir), summary_csv_path)

    print("\n✓ Report generation complete!")
    print(f"  Report: {report_path}")
    print(f"  Summary CSV: {summary_csv_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
