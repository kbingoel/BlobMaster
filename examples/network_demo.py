"""
Demo script showing BlobNet neural network in action.

This script demonstrates:
1. Creating a game state
2. Encoding the state for neural network input
3. Running inference to get action probabilities and value prediction
4. Training the network on dummy data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ml.game.blob import BlobGame
from ml.network import StateEncoder, ActionMasker, create_model, create_trainer


def demo_inference():
    """Demonstrate inference on a game state."""
    print("=" * 70)
    print("DEMO 1: Neural Network Inference")
    print("=" * 70)

    # Create components
    encoder = StateEncoder()
    masker = ActionMasker()
    model = create_model()
    model.eval()  # Evaluation mode

    print(f"\nModel parameters: {model.get_num_parameters():,}")

    # Create game
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    print(f"\nGame setup:")
    print(f"  Players: {game.num_players}")
    print(f"  Cards dealt: 5")
    print(f"  Player has {len(player.hand)} cards in hand")

    # Encode state
    state = encoder.encode(game, player)
    print(f"\nState encoding:")
    print(f"  Shape: {state.shape}")
    print(f"  Non-zero elements: {(state != 0).sum().item()}/{state.numel()}")

    # Create legal action mask for bidding
    is_dealer = player.position == game.dealer_position
    mask = masker.create_bidding_mask(
        cards_dealt=5,
        is_dealer=is_dealer,
        forbidden_bid=None
    )

    legal_bids = [i for i in range(6) if mask[i] == 1.0]
    print(f"\nLegal bids: {legal_bids}")

    # Run inference
    with torch.no_grad():
        policy, value = model(state, mask)

    print(f"\nInference results:")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value prediction: {value.item():.4f}")

    # Show bid probabilities
    print(f"\nBid probabilities:")
    for bid in legal_bids:
        prob = policy[bid].item()
        print(f"  Bid {bid}: {prob:.4f} ({prob*100:.1f}%)")

    # Verify illegal bids have zero probability
    print(f"\nIllegal bid probabilities (should be 0):")
    for bid in range(6, 14):
        prob = policy[bid].item()
        print(f"  Bid {bid}: {prob:.6f}")

    print()


def demo_training():
    """Demonstrate training on dummy data."""
    print("=" * 70)
    print("DEMO 2: Neural Network Training")
    print("=" * 70)

    # Create model and trainer
    model = create_model()
    trainer = create_trainer(model, learning_rate=0.01)

    print(f"\nModel parameters: {model.get_num_parameters():,}")

    # Generate dummy training data
    print(f"\nGenerating training data from 8 game states...")

    encoder = StateEncoder()
    masker = ActionMasker()

    states = []
    target_policies = []
    target_values = []
    masks = []

    for i in range(8):
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Encode state
        state = encoder.encode(game, player)
        states.append(state)

        # Create mask
        is_dealer = player.position == game.dealer_position
        mask = masker.create_bidding_mask(
            cards_dealt=5,
            is_dealer=is_dealer,
            forbidden_bid=None
        )
        masks.append(mask)

        # Create dummy target policy (uniform over legal actions)
        target_policy = mask.clone()
        target_policy = target_policy / target_policy.sum()
        target_policies.append(target_policy)

        # Create dummy target value (random score)
        target_values.append(torch.tensor([torch.rand(1).item() * 2 - 1]))

    # Stack into batches
    state_batch = torch.stack(states)
    policy_batch = torch.stack(target_policies)
    value_batch = torch.stack(target_values)
    mask_batch = torch.stack(masks)

    print(f"  Batch size: {state_batch.shape[0]}")
    print(f"  State shape: {state_batch.shape}")

    # Train for multiple steps
    print(f"\nTraining for 20 iterations...")
    losses = []

    for i in range(20):
        loss_dict = trainer.train_step(
            state_batch, policy_batch, value_batch, mask_batch
        )
        losses.append(loss_dict['total_loss'])

        if i % 5 == 0:
            print(f"  Iteration {i:2d}: Loss = {loss_dict['total_loss']:.4f} "
                  f"(policy: {loss_dict['policy_loss']:.4f}, "
                  f"value: {loss_dict['value_loss']:.4f})")

    # Show improvement
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nTraining results:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {improvement:.1f}%")

    print()


def demo_performance():
    """Benchmark inference performance."""
    print("=" * 70)
    print("DEMO 3: Performance Benchmark")
    print("=" * 70)

    import time

    model = create_model()
    model.eval()

    # Warmup
    state = torch.randn(256)
    with torch.no_grad():
        for _ in range(10):
            model(state)

    # Single inference benchmark
    num_runs = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            policy, value = model(state)

    elapsed_ms = (time.time() - start_time) * 1000
    avg_time_ms = elapsed_ms / num_runs

    print(f"\nSingle state inference:")
    print(f"  Runs: {num_runs}")
    print(f"  Total time: {elapsed_ms:.2f} ms")
    print(f"  Average time: {avg_time_ms:.2f} ms per forward pass")
    print(f"  Throughput: {1000/avg_time_ms:.0f} inferences/second")

    # Batch inference benchmark
    batch_sizes = [1, 4, 8, 16, 32]

    print(f"\nBatch inference performance:")
    print(f"  {'Batch Size':>12} | {'Time (ms)':>10} | {'Per State (ms)':>15} | {'Speedup':>8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*15}-+-{'-'*8}")

    baseline_time = None

    for batch_size in batch_sizes:
        state_batch = torch.randn(batch_size, 256)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(state_batch)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                policy, value = model(state_batch)

        elapsed_ms = (time.time() - start_time) * 1000
        time_per_state = elapsed_ms / (num_runs * batch_size)

        if baseline_time is None:
            baseline_time = time_per_state
            speedup = 1.0
        else:
            speedup = baseline_time / time_per_state

        print(f"  {batch_size:12d} | {elapsed_ms:10.2f} | {time_per_state:15.2f} | {speedup:8.2f}x")

    print()


def main():
    """Run all demos."""
    print()
    print("+" + "=" * 68 + "+")
    print("|" + " " * 18 + "BlobNet Neural Network Demo" + " " * 23 + "|")
    print("+" + "=" * 68 + "+")
    print()

    demo_inference()
    demo_training()
    demo_performance()

    print("=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
