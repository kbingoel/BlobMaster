#!/usr/bin/env python3
"""
Quick test to measure performance with optimized BatchedEvaluator and correct config.
"""
import time
import torch
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine

# Enable TF32 for faster inference (if available)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create network
network = BlobNet()
network.to(device)
network.eval()

# Create encoder/masker
encoder = StateEncoder()
masker = ActionMasker()

# Create engine with optimal config (matching baseline)
engine = SelfPlayEngine(
    network=network,
    encoder=encoder,
    masker=masker,
    num_workers=32,
    num_determinizations=3,
    simulations_per_determinization=30,
    device=device,
    use_batched_evaluator=True,
    batch_size=512,
    batch_timeout_ms=10.0,
    use_parallel_expansion=False,  # ← Test without parallel expansion first
    parallel_batch_size=30,
    use_thread_pool=False,  # ← Force multiprocessing to match "Holy mackerel" measurement
)

print(f"Using thread pool: {engine.use_thread_pool}")
print(f"Using batched evaluator: {engine.use_batched_evaluator}")
print(f"Parallel expansion: {engine.use_parallel_expansion}")
print(f"Parallel batch size: {engine.parallel_batch_size}")
print()
print("Generating 100 games with 32 workers...")
print("Config: 3 det × 30 sims, parallel_batch_size=30, batched evaluator")
print()

start = time.time()
examples = engine.generate_games(
    num_games=100,
    num_players=4,
    cards_to_deal=5,
)
elapsed = time.time() - start

games_per_min = (100 / elapsed) * 60.0
print(f"\nResults:")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Games/minute: {games_per_min:.1f}")
print(f"  Training examples: {len(examples)}")

engine.shutdown()
