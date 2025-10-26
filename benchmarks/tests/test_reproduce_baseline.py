"""
Reproduce the baseline results from baseline_results.csv.

This script uses the EXACT configuration from the original baseline test:
- Large network (256 emb, 6 layers, 1024 FFN) = 4.9M params
- 32 workers, multiprocessing, no batching
- Light/Medium/Heavy MCTS configs
- Device: CUDA

Expected results (from baseline_results.csv):
- Light (2×20=40 sims):   80.8 games/min
- Medium (3×30=90 sims):  43.3 games/min
- Heavy (5×50=250 sims):  25.0 games/min
"""

import sys
from pathlib import Path
import time
import csv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


def count_parameters(network):
    """Count network parameters."""
    return sum(p.numel() for p in network.parameters())


def run_baseline_test(
    config_name: str,
    num_determinizations: int,
    simulations_per_det: int,
    num_games: int = 20,
):
    """
    Run baseline test with specific MCTS configuration.

    Args:
        config_name: "light", "medium", or "heavy"
        num_determinizations: Number of determinizations
        simulations_per_det: Simulations per determinization
        num_games: Number of games to test (default: 20)

    Returns:
        dict with results
    """
    total_sims = num_determinizations * simulations_per_det

    print(f"\n{'='*80}")
    print(f"BASELINE TEST: {config_name.upper()} MCTS")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Workers: 32")
    print(f"  - Parallelism: multiprocessing")
    print(f"  - Batching: no (use_batched_evaluator=False)")
    print(f"  - MCTS: {num_determinizations} det × {simulations_per_det} sims = {total_sims} sims/move")
    print(f"  - Games: {num_games}")
    print(f"{'='*80}\n")

    # Create LARGE network (baseline configuration)
    print("Creating network...")
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,      # Baseline: 256 (not 128!)
        num_layers=6,           # Baseline: 6 (not 2!)
        num_heads=8,
        feedforward_dim=1024,   # Baseline: 1024 (not 256!)
        dropout=0.0,
    )
    params = count_parameters(network)
    print(f"  Network parameters: {params:,}")
    print(f"  Architecture: emb=256, layers=6, heads=8, ffn=1024")

    network.to("cuda")
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create self-play engine (BASELINE configuration)
    print("\nCreating self-play engine...")
    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=32,                     # Baseline: 32
        num_determinizations=num_determinizations,
        simulations_per_determinization=simulations_per_det,
        device="cuda",
        use_batched_evaluator=False,        # Baseline: False!
        use_thread_pool=False,              # Baseline: multiprocessing
    )

    # Warm-up run
    print("Warming up (2 games)...")
    engine.generate_games(num_games=2, num_players=4, cards_to_deal=3)

    # Benchmark run
    print(f"\nGenerating {num_games} games...")
    start_time = time.time()

    examples = engine.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=3,
    )

    elapsed = time.time() - start_time
    games_per_min = (num_games / elapsed) * 60.0
    sec_per_game = elapsed / num_games

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Games/minute: {games_per_min:.2f}")
    print(f"Seconds/game: {sec_per_game:.2f}")
    print(f"Training examples: {len(examples)}")
    print(f"Examples/minute: {(len(examples) / elapsed) * 60.0:.1f}")

    # Clean up
    del network
    del engine
    torch.cuda.empty_cache()

    return {
        'config': config_name,
        'num_determinizations': num_determinizations,
        'simulations_per_det': simulations_per_det,
        'total_sims_per_move': total_sims,
        'num_games': num_games,
        'elapsed_time_sec': elapsed,
        'games_per_minute': games_per_min,
        'seconds_per_game': sec_per_game,
        'num_examples': len(examples),
        'examples_per_minute': (len(examples) / elapsed) * 60.0,
    }


def main():
    """Run all baseline tests."""

    print("\n" + "="*80)
    print("BASELINE REPRODUCTION TEST")
    print("Reproducing results from baseline_results.csv")
    print("="*80)

    # Check GPU
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available! This test requires a GPU.")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print("\nExpected results (from baseline_results.csv):")
    print("  - Light (2×20):   80.8 games/min")
    print("  - Medium (3×30):  43.3 games/min")
    print("  - Heavy (5×50):   25.0 games/min")

    results = []

    # Test 1: Light MCTS (2×20 = 40 sims/move)
    light_result = run_baseline_test(
        config_name="light",
        num_determinizations=2,
        simulations_per_det=20,
        num_games=20,
    )
    results.append(light_result)

    # Test 2: Medium MCTS (3×30 = 90 sims/move)
    medium_result = run_baseline_test(
        config_name="medium",
        num_determinizations=3,
        simulations_per_det=30,
        num_games=20,
    )
    results.append(medium_result)

    # Test 3: Heavy MCTS (5×50 = 250 sims/move)
    heavy_result = run_baseline_test(
        config_name="heavy",
        num_determinizations=5,
        simulations_per_det=50,
        num_games=20,
    )
    results.append(heavy_result)

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    print(f"\n{'Config':<10} {'Expected':<15} {'Actual':<15} {'Match?':<10}")
    print("-"*50)

    expected = {
        'light': 80.8,
        'medium': 43.3,
        'heavy': 25.0,
    }

    for result in results:
        config = result['config']
        actual = result['games_per_minute']
        exp = expected[config]
        diff_pct = ((actual - exp) / exp) * 100

        if abs(diff_pct) < 10:
            match = "[YES]"
        elif abs(diff_pct) < 20:
            match = "[CLOSE]"
        else:
            match = "[NO]"

        print(f"{config:<10} {exp:>6.1f} g/min   {actual:>6.1f} g/min   {match:<10} ({diff_pct:+.1f}%)")

    # Save results
    output_file = Path(__file__).parent.parent.parent / "results" / "baseline_reproduction.csv"
    print(f"\nSaving results to: {output_file}")

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Results saved")


if __name__ == "__main__":
    main()
