from ml.mcts.search import MCTS
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame

network = BlobNet()
encoder = StateEncoder()
masker = ActionMasker()
mcts = MCTS(network, encoder, masker, num_simulations=32)

# Run the test 10 times to see if it's flaky
failures = 0
for run in range(10):
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=3)

    print(f"\nRun {run + 1}:")
    print(f"  Dealer position: {game.dealer_position}")

    # All players use search (in position order, NOT bidding order)
    for i, player in enumerate(game.players):
        is_dealer = player.position == game.dealer_position
        total_bids_before = sum(p.bid for p in game.players if p.bid is not None)

        action_probs = mcts.search_batched(game, player, batch_size=8)
        bid = max(action_probs, key=action_probs.get)
        player.make_bid(bid)

        print(f"  Player {i} (pos={player.position}, dealer={is_dealer}): "
              f"total_before={total_bids_before}, bid={bid}")

    total_bids = sum(p.bid for p in game.players)
    print(f"  Total bids: {total_bids}")

    if total_bids == 3:
        print(f"  FAIL: Total bids equals cards dealt!")
        failures += 1
    else:
        print(f"  PASS")

print(f"\n\nSummary: {failures}/10 runs had total_bids == 3")
