"""
Minimal test to diagnose GPU worker issue.

This script tests whether multiprocessing workers can successfully use CUDA
by creating a simple network, spawning a worker, and checking device placement.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.multiprocessing as mp
from ml.network.model import BlobNet


def test_gpu_in_worker(worker_id: int, network_state: dict, device: str):
    """
    Worker function to test GPU access in multiprocessing.

    Args:
        worker_id: Worker ID
        network_state: Network state dict
        device: Target device ('cpu' or 'cuda')
    """
    print(f"\n[Worker {worker_id}] === Starting GPU Test ===")
    print(f"[Worker {worker_id}] CUDA available: {torch.cuda.is_available()}")
    print(f"[Worker {worker_id}] Requested device: {device}")

    if torch.cuda.is_available():
        print(f"[Worker {worker_id}] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Worker {worker_id}] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[Worker {worker_id}] CUDA device name: {torch.cuda.get_device_name(0)}")

    # Create network from state dict
    state_dim = network_state["input_embedding.weight"].shape[1]
    embedding_dim = network_state["input_embedding.weight"].shape[0]

    # Count transformer layers
    num_layers = 0
    while f"transformer.layers.{num_layers}.self_attn.in_proj_weight" in network_state:
        num_layers += 1

    feedforward_dim = network_state["transformer.layers.0.linear1.weight"].shape[0]

    network = BlobNet(
        state_dim=state_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=8,
        feedforward_dim=feedforward_dim,
        dropout=0.1,
    )

    print(f"[Worker {worker_id}] Network created, device: {next(network.parameters()).device}")

    network.load_state_dict(network_state)
    print(f"[Worker {worker_id}] State dict loaded, device: {next(network.parameters()).device}")

    network.to(device)
    actual_device = next(network.parameters()).device
    print(f"[Worker {worker_id}] After .to({device}), device: {actual_device}")

    # Test inference
    print(f"[Worker {worker_id}] Testing inference...")
    network.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, state_dim).to(device)
        print(f"[Worker {worker_id}] Dummy input device: {dummy_input.device}")

        policy, value = network(dummy_input)
        print(f"[Worker {worker_id}] Policy output device: {policy.device}")
        print(f"[Worker {worker_id}] Value output device: {value.device}")

    # Final check
    if device == "cuda" and actual_device.type == "cpu":
        print(f"[Worker {worker_id}] ❌ FAILED: Network still on CPU despite .to('cuda')")
        return False
    elif device == "cuda" and actual_device.type == "cuda":
        print(f"[Worker {worker_id}] ✅ SUCCESS: Network successfully moved to CUDA")
        return True
    else:
        print(f"[Worker {worker_id}] ✅ CPU test passed")
        return True


def main():
    """Run the GPU worker test."""
    print("=" * 80)
    print("GPU Worker Test - Multiprocessing Diagnostics")
    print("=" * 80)

    # Check main process CUDA
    print("\n[Main] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"[Main] CUDA device: {torch.cuda.get_device_name(0)}")

    # Create a small network
    print("\n[Main] Creating test network...")
    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=8,
        feedforward_dim=256,
        dropout=0.1,
    )

    # Test with CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Testing with device: {device}")

    # Move to device in main process
    network.to(device)
    print(f"[Main] Network device in main process: {next(network.parameters()).device}")

    # Get state dict
    network_state = network.state_dict()

    # Test single worker
    print("\n" + "=" * 80)
    print("TEST 1: Single Worker")
    print("=" * 80)

    # Use spawn method for Windows compatibility
    mp.set_start_method("spawn", force=True)

    process = mp.Process(
        target=test_gpu_in_worker,
        args=(0, network_state, device),
    )
    process.start()
    process.join()

    print("\n" + "=" * 80)
    print("TEST 2: Multiple Workers (2)")
    print("=" * 80)

    processes = []
    for i in range(2):
        p = mp.Process(
            target=test_gpu_in_worker,
            args=(i, network_state, device),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
