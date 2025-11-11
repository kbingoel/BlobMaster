#!/usr/bin/env python3
"""Analyze profile files and generate bottleneck report."""

import pstats
from pstats import SortKey
import sys
from pathlib import Path

def analyze_profile(profile_file):
    """Analyze a profile file and print detailed statistics."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {profile_file}")
    print(f"{'='*80}\n")

    stats = pstats.Stats(profile_file)

    # Total time
    print(f"\n{'='*80}")
    print("TOTAL TIME BREAKDOWN")
    print(f"{'='*80}")
    stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)

    # Time per function (not including subcalls)
    print(f"\n{'='*80}")
    print("TOP FUNCTIONS BY INTERNAL TIME (excluding subcalls)")
    print(f"{'='*80}")
    stats.strip_dirs().sort_stats(SortKey.TIME).print_stats(30)

    # Most called functions
    print(f"\n{'='*80}")
    print("MOST FREQUENTLY CALLED FUNCTIONS")
    print(f"{'='*80}")
    stats.strip_dirs().sort_stats(SortKey.CALLS).print_stats(30)

    # Callers of expensive functions
    print(f"\n{'='*80}")
    print("CALLERS/CALLEES ANALYSIS (Top 10 by cumulative time)")
    print(f"{'='*80}")
    stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_callers(10)

def main():
    # Find all profile files
    profile_files = sorted(Path('.').glob('profile_*.prof'), key=lambda p: p.stat().st_mtime, reverse=True)

    if not profile_files:
        print("ERROR: No profile files found!")
        print("Run profiling first: python ml/profile_selfplay.py")
        sys.exit(1)

    # Analyze the most recent profile file
    most_recent = profile_files[0]
    analyze_profile(str(most_recent))

    print(f"\n{'='*80}")
    print("SUMMARY OF AVAILABLE PROFILE FILES")
    print(f"{'='*80}")
    for pf in profile_files:
        print(f"  - {pf.name} ({pf.stat().st_size / 1024:.1f} KB)")

if __name__ == '__main__':
    main()
