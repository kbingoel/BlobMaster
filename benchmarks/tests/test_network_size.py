"""
Quick script to calculate and compare network sizes.

Compares the baseline network configuration from test_action_plan_windows.py
with the current Phase 1 benchmark network.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.network.model import BlobNet


def count_parameters(network: BlobNet) -> int:
    """Count total trainable parameters in network."""
    return sum(p.numel() for p in network.parameters())


def main():
    print("="*80)
    print("Network Size Comparison")
    print("="*80)

    # Baseline configuration (from test_action_plan_windows.py)
    print("\n1. BASELINE Configuration (test_action_plan_windows.py)")
    print("   - Used for 80.8 / 43.3 games/min results")
    baseline_net = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,
    )
    baseline_params = count_parameters(baseline_net)
    print(f"   Parameters: {baseline_params:,}")
    print(f"   Architecture: emb=256, layers=6, heads=8, ffn=1024")

    # Current Phase 1 benchmark configuration
    print("\n2. CURRENT Phase 1 Benchmark Configuration")
    print("   - Used in our recent tests (5.9-11.1 games/min)")
    current_net = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    current_params = count_parameters(current_net)
    print(f"   Parameters: {current_params:,}")
    print(f"   Architecture: emb=128, layers=2, heads=4, ffn=256")

    # Original benchmark_selfplay.py configuration
    print("\n3. Original benchmark_selfplay.py Configuration")
    print("   - Default small network for fast inference")
    original_net = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    original_params = count_parameters(original_net)
    print(f"   Parameters: {original_params:,}")
    print(f"   Architecture: emb=128, layers=2, heads=4, ffn=256")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Baseline vs Current: {baseline_params / current_params:.2f}x larger")
    print(f"   ({baseline_params:,} vs {current_params:,} parameters)")
    print()
    print("Dimension Comparison:")
    print(f"   Embedding: 256 vs 128 (2.0x)")
    print(f"   Layers: 6 vs 2 (3.0x)")
    print(f"   Heads: 8 vs 4 (2.0x)")
    print(f"   FFN: 1024 vs 256 (4.0x)")
    print()
    print("HYPOTHESIS:")
    print("   Larger network (baseline) may perform BETTER because:")
    print("   1. Better GPU utilization (more FLOPS per call)")
    print("   2. Amortizes GPU kernel launch overhead")
    print("   3. RTX 4060 has 3,072 CUDA cores - wants larger workloads")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
