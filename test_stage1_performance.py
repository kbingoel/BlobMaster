#!/usr/bin/env python3
"""Quick performance test for Stage 1 (1×15 MCTS) configuration.

Expected performance from MPPT report:
- Stage 1 (1×15 MCTS, 32w×30b×10ms): ~1614.8 rounds/min
"""

import time
import torch
from ml.config import TrainingConfig
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


def test_stage1_performance(num_rounds: int = 1000):
    """Test Stage 1 performance with MPPT-optimized config."""

    print("=" * 80)
    print("BlobMaster Stage 1 Performance Test")
    print("=" * 80)
    print()

    # Create config with Stage 1 settings (1×15 MCTS)
    config = TrainingConfig(
        num_workers=32,
        games_per_iteration=num_rounds,
        num_determinizations=1,
        simulations_per_determinization=15,
        parallel_batch_size=30,
        batch_timeout_ms=10.0,
        device="cuda",
        training_on="rounds",  # Train on individual rounds, not full games
    )

    print(f"Configuration:")
    print(f"  • Workers: {config.num_workers}")
    print(f"  • Rounds to play: {num_rounds}")
    print(f"  • MCTS: {config.num_determinizations}×{config.simulations_per_determinization}")
    print(f"  • Batch size: {config.parallel_batch_size}")
    print(f"  • Batch timeout: {config.batch_timeout_ms}ms")
    print(f"  • Device: {config.device}")
    print(f"  • Training mode: {config.training_on}")
    print()
    print(f"Expected performance: ~1614.8 rounds/min (from MPPT report)")
    print()
    print("-" * 80)
    print()

    # Create network
    print("Initializing neural network...")
    network = BlobNet(
        embedding_dim=config.embedding_dim,
        num_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(config.device)
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    print(f"Network created: {sum(p.numel() for p in network.parameters()):,} parameters")
    print()

    # Create self-play engine
    print("Initializing self-play engine...")
    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        config=config,
    )

    print(f"Self-play engine created with {config.num_workers} workers")
    print()
    print("=" * 80)
    print("Starting self-play test...")
    print("=" * 80)
    print()

    # Run self-play and measure performance
    start_time = time.time()

    games = engine.generate_games(config.games_per_iteration)

    elapsed = time.time() - start_time

    # Calculate performance
    rounds_played = len(games)
    rounds_per_min = rounds_played / elapsed * 60

    print()
    print("=" * 80)
    print("Performance Results")
    print("=" * 80)
    print()
    print(f"Rounds played: {rounds_played}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Performance: {rounds_per_min:.1f} rounds/min")
    print()
    print(f"Expected: ~1614.8 rounds/min")
    print(f"Difference: {rounds_per_min - 1614.8:+.1f} rounds/min ({(rounds_per_min / 1614.8 - 1) * 100:+.1f}%)")
    print()

    # Interpret results
    variance = abs(rounds_per_min / 1614.8 - 1)
    if variance < 0.05:
        print("✅ Performance matches expected (within 5%)")
    elif variance < 0.10:
        print("⚠️  Performance close to expected (within 10%)")
    else:
        print("❌ Performance differs significantly (>10% variance)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    import sys

    num_rounds = 1000
    if len(sys.argv) > 1:
        num_rounds = int(sys.argv[1])

    test_stage1_performance(num_rounds)
