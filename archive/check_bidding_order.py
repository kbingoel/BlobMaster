from ml.game.blob import BlobGame

game = BlobGame(num_players=4)
game.setup_round(cards_to_deal=3)

print(f'Dealer position: {game.dealer_position}')

bidding_order_start = (game.dealer_position + 1) % 4
print(f'Bidding order starts at: {bidding_order_start}')

bidding_order = [(bidding_order_start + i) % 4 for i in range(4)]
print(f'Bidding order: {bidding_order}')
print(f'Dealer bids at index: {bidding_order.index(game.dealer_position)}')

print("\nWhen iterating through game.players in order:")
for i, player in enumerate(game.players):
    print(f"  i={i}, player.position={player.position}, is_dealer={player.position == game.dealer_position}")
