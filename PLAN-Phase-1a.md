# PLAN-Phase-1a.md
# Implementation Plan: Core Game Engine (`ml/game/blob.py`)

**Phase**: 1a - Core Game Engine
**Timeline**: Weekend 1 (Days 1-2)
**Goal**: Bulletproof game logic that handles all edge cases

---

## Overview

Implement the complete game engine for **Blob** (trick-taking bidding game) in Python. This is the foundation for the entire AI training pipeline and must handle all game rules correctly for 3-8 players.

## File Structure to Create

```
ml/game/
├── __init__.py          # Package initialization
├── constants.py         # Game constants (suits, ranks, scoring)
├── blob.py              # Main game logic (CORE DELIVERABLE)
└── test_blob.py         # Comprehensive unit tests
```

---

## Game Rules Reference

### Setup
- **Players**: 3-8 players (variable per game)
- **Deck**: Standard 52-card deck
- **Rounds**: Variable cards dealt per round
  - Typical: Start with N cards → decrease → 1 card × num_players → increase back to N
- **Trump**: Rotates through ♠ → ♥ → ♣ → ♦ → None (no-trump) → repeat

### Bidding Phase
- Players bid **sequentially** on exact number of tricks they expect to win
- **Last Bidder Constraint**: Dealer cannot bid such that `sum(all_bids) == cards_dealt`
  - Creates strategic tension: dealer must force "over" or "under" bidding
- Valid bids: `0` to `cards_dealt` (inclusive), except forbidden dealer bid

### Playing Phase
- Standard trick-taking rules:
  1. **Follow suit**: Must play same suit as lead card if possible
  2. **Trump beats non-trump**: Trump card beats any non-trump card
  3. **Highest in suit wins**: If no trump, highest card in led suit wins
  4. **Winner leads next**: Trick winner leads the next trick
  5. **No-trump rounds**: Led card's suit becomes "trump" for that trick only

### Scoring Phase
- **Exact bids only**: `score = (tricks_won == bid) ? (10 + bid) : 0`
- Examples:
  - Bid 2, won 2 → **12 points**
  - Bid 3, won 4 → **0 points** (bust)
  - Bid 0, won 0 → **10 points** (risky!)

### Anti-Cheat / Validation
- **Illegal card detection**: Track which cards each player should have based on dealing and plays
- **Suit elimination tracking**: When player doesn't follow suit, mark that they have no cards in that suit
- **Card counting**: Maintain global state of all cards played, remaining cards per suit
- **Cheat detection**: Raise exception if player plays card they shouldn't have or fails to follow suit when able
- **Purpose**: Critical for multiplayer games against real humans who may make mistakes or attempt to cheat

---

## Implementation Plan

### 1. Constants (`ml/game/constants.py`)

```python
"""
Game constants for Blob card game.
"""

# Card definitions
SUITS = ['♠', '♥', '♣', '♦']
SUIT_NAMES = {
    '♠': 'Spades',
    '♥': 'Hearts',
    '♣': 'Clubs',
    '♦': 'Diamonds'
}

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {rank: idx for idx, rank in enumerate(RANKS, start=2)}

# Trump rotation cycle
TRUMP_ROTATION = ['♠', '♥', '♣', '♦', None]  # None = no-trump rounds

# Game constraints
MIN_PLAYERS = 3
MAX_PLAYERS = 8
DECK_SIZE = 52

# Scoring
SCORE_BASE = 10  # Base score for making exact bid

# Round structure
def generate_round_structure(starting_cards: int, num_players: int) -> List[int]:
    """
    Generate round structure for a game.

    Args:
        starting_cards: Number of cards dealt in first round (e.g., 7)
        num_players: Number of players in game (3-8)

    Returns:
        List of cards to deal per round
        Example: starting_cards=5, num_players=3 → [5,4,3,2,1,1,1,2,3,4,5]

    Raises:
        ValueError: If starting_cards * num_players > 52
    """
    # Validate we don't exceed deck size
    if starting_cards * num_players > DECK_SIZE:
        raise ValueError(
            f"Cannot deal {starting_cards} cards to {num_players} players "
            f"(requires {starting_cards * num_players} cards, deck has {DECK_SIZE})"
        )

    # Descending phase: starting_cards down to 1
    descending = list(range(starting_cards, 0, -1))

    # One-card rounds: num_players rounds with 1 card each
    one_card_rounds = [1] * num_players

    # Ascending phase: 2 back up to starting_cards
    ascending = list(range(2, starting_cards + 1))

    return descending + one_card_rounds + ascending
```

**Purpose**: Centralized configuration, easy to modify for game variants.

**Key Addition**: `generate_round_structure()` dynamically creates round structure based on player count and starting cards, with validation to prevent exceeding 52-card deck limit.

---

### 2. Core Classes (`ml/game/blob.py`)

#### 2.0 Custom Exceptions

```python
class BlobGameException(Exception):
    """Base exception for Blob game errors."""
    pass

class IllegalPlayException(BlobGameException):
    """Raised when player attempts an illegal card play."""
    def __init__(self, player_name: str, card: Card, reason: str):
        self.player_name = player_name
        self.card = card
        self.reason = reason
        super().__init__(f"{player_name} played {card} illegally: {reason}")

class InvalidBidException(BlobGameException):
    """Raised when player attempts an invalid bid."""
    pass

class GameStateException(BlobGameException):
    """Raised when game is in invalid state for requested action."""
    pass
```

**Purpose**: Clear error handling for anti-cheat system and game validation.

---

#### 2.1 `Card` Class

**Properties**:
- `suit: str` - One of ♠, ♥, ♣, ♦
- `rank: str` - One of 2-10, J, Q, K, A
- `value: int` - Numeric value for comparison (2=2, J=11, A=14)

**Methods**:
- `__init__(suit, rank)`
- `__str__()` → "A♠"
- `__repr__()` → "Card('A', '♠')"
- `__eq__(other)` → Compare by suit and rank
- `__lt__(other)` → Enable sorting by value
- `__hash__()` → Allow use in sets/dicts

**Design Notes**:
- Immutable (frozen dataclass or namedtuple)
- Enable sorting for hand display
- hashable for set operations

---

#### 2.2 `Deck` Class

**Properties**:
- `cards: List[Card]` - All 52 cards
- `dealt_cards: Set[Card]` - Track what's been dealt

**Methods**:
- `__init__()` → Create standard 52-card deck
- `shuffle()` → Randomize card order
- `deal(num_cards: int, num_players: int) -> List[List[Card]]`
  - Deal `num_cards` to each of `num_players`
  - Validate: `num_cards * num_players <= 52`
  - Return list of hands
- `reset()` → Return all cards to deck, clear dealt_cards
- `remaining_cards() -> int` → Cards left in deck

**Edge Cases**:
- Deal validation: ensure enough cards
- Track dealt cards for game state queries

---

#### 2.3 `Player` Class

**Properties**:
- `name: str` - Player identifier
- `hand: List[Card]` - Current cards
- `bid: Optional[int]` - Bid for current round (None if not yet bid)
- `tricks_won: int` - Tricks won this round
- `total_score: int` - Cumulative score across all rounds
- `position: int` - Seat position (0-indexed)
- `known_void_suits: Set[str]` - Suits player has revealed they don't have (for card counting)
- `cards_played: List[Card]` - All cards this player has played this round (for validation)

**Methods**:
- `__init__(name, position)`
- `receive_cards(cards: List[Card])` → Add cards to hand
- `play_card(card: Card) -> Card` → Remove and return card from hand, add to cards_played
- `make_bid(bid: int)` → Set bid for round
- `win_trick()` → Increment tricks_won
- `calculate_round_score() -> int` → Check if bid met, return score
- `reset_round()` → Clear hand, bid, tricks_won, known_void_suits, cards_played
- `sort_hand()` → Sort cards by suit then rank
- `mark_void_suit(suit: str)` → Add suit to known_void_suits (when they don't follow suit)
- `has_suit(suit: str) -> bool` → Check if player has any cards in suit (for validation)

**Design Notes**:
- Player manages own state including anti-cheat tracking
- Score calculation method for clarity
- Keep hand sorted for UI display
- Track void suits for card counting and validation

---

#### 2.4 `Trick` Class

**Properties**:
- `cards_played: List[Tuple[Player, Card]]` - (player, card) pairs in play order
- `led_suit: Optional[str]` - Suit of first card played
- `trump_suit: Optional[str]` - Current round's trump (or None)
- `winner: Optional[Player]` - Trick winner (after determination)

**Methods**:
- `__init__(trump_suit: Optional[str])`
- `add_card(player: Player, card: Card)`
  - Track play order
  - Set led_suit on first card
- `determine_winner() -> Player`
  - Apply trump/suit rules
  - Return winning player
  - Cache result in `self.winner`
- `get_winning_card() -> Card` → Return highest card
- `clear()` → Reset for next trick
- `is_complete(num_players: int) -> bool` → Check if all players played

**Winner Determination Logic**:
```python
def determine_winner(self) -> Player:
    # Filter trump cards if trump suit exists
    trump_cards = [(p, c) for p, c in self.cards_played
                   if c.suit == self.trump_suit]

    if trump_cards:
        # Highest trump wins
        winner = max(trump_cards, key=lambda x: x[1].value)
    else:
        # Highest card in led suit wins
        led_suit_cards = [(p, c) for p, c in self.cards_played
                          if c.suit == self.led_suit]
        winner = max(led_suit_cards, key=lambda x: x[1].value)

    self.winner = winner[0]
    return self.winner
```

---

#### 2.5 `BlobGame` Class (Main Orchestrator)

**Properties**:
- `num_players: int` - Number of players (3-8)
- `players: List[Player]` - Player objects
- `deck: Deck` - Game deck
- `current_round: int` - Round counter
- `trump_suit: Optional[str]` - Current trump suit
- `dealer_position: int` - Current dealer (rotates each round)
- `current_trick: Optional[Trick]` - Active trick
- `tricks_history: List[Trick]` - Completed tricks this round
- `game_phase: str` - One of: 'setup', 'bidding', 'playing', 'scoring', 'complete'
- `cards_played_this_round: List[Card]` - All cards played this round (for card counting)
- `cards_remaining_by_suit: Dict[str, int]` - Count of unplayed cards per suit (for card counting)

**Methods** (detailed below):

##### Setup Phase
```python
def __init__(self, num_players: int, player_names: List[str] = None):
    """
    Initialize game with N players.
    - Validate num_players in [3, 8]
    - Create Player objects
    - Initialize Deck
    - Set dealer_position = 0
    """

def setup_round(self, cards_to_deal: int) -> None:
    """
    Prepare for new round:
    1. Reset deck and player round state
    2. Determine trump suit for this round
    3. Deal cards to all players
    4. Sort player hands
    5. Set game_phase = 'bidding'
    """

def determine_trump(self) -> Optional[str]:
    """
    Rotate trump based on current_round:
    - Round 0 → ♠, Round 1 → ♥, Round 2 → ♣, Round 3 → ♦, Round 4 → None
    - Cycles: round % len(TRUMP_ROTATION)
    """
```

##### Bidding Phase
```python
def bidding_phase(self) -> None:
    """
    Collect bids from all players sequentially:
    1. Loop through players starting left of dealer
    2. For each player except dealer: accept any bid [0, cards_dealt]
    3. For dealer: calculate forbidden bid, enforce constraint
    4. Store bids in player.bid
    5. Set game_phase = 'playing'
    """

def get_forbidden_bid(self, current_total_bids: int, cards_dealt: int) -> Optional[int]:
    """
    Calculate dealer's forbidden bid:
    - forbidden = cards_dealt - current_total_bids
    - Return None if forbidden < 0 or > cards_dealt (edge case)
    """

def is_valid_bid(self, bid: int, is_dealer: bool,
                 current_total_bids: int, cards_dealt: int) -> bool:
    """
    Validate bid:
    - General: 0 <= bid <= cards_dealt
    - Dealer: bid != get_forbidden_bid(...)
    """
```

##### Playing Phase
```python
def playing_phase(self) -> None:
    """
    Play all tricks for the round:
    1. While tricks remain: play_trick()
    2. Set game_phase = 'scoring'
    """

def play_trick(self) -> None:
    """
    Execute one complete trick:
    1. Create new Trick with current trump
    2. Loop through players (starting with lead player)
    3. Get legal plays for each player
    4. Player selects card (manual or bot)
    5. Add card to trick
    6. Determine winner after all cards played
    7. Winner.win_trick()
    8. Update lead_player for next trick
    9. Add to tricks_history
    """

def get_legal_plays(self, player: Player, led_suit: Optional[str]) -> List[Card]:
    """
    Return cards player can legally play:
    - If led_suit is None (first card): return all cards in hand
    - If player has led_suit: return only those cards
    - Else: return all cards (can't follow suit)
    """

def is_valid_play(self, card: Card, player: Player, led_suit: Optional[str]) -> bool:
    """
    Check if card play is legal:
    - Card must be in player's hand
    - If led_suit exists and player has that suit: card.suit must match
    """

def validate_play_with_anti_cheat(self, card: Card, player: Player, led_suit: Optional[str]) -> None:
    """
    Strict validation with anti-cheat detection:
    1. Verify card is in player's hand
    2. If led_suit exists:
       a. Check if player has cards in led_suit
       b. If yes and card.suit != led_suit: RAISE IllegalPlayException (cheating detected!)
       c. If no and card.suit != led_suit: mark player.mark_void_suit(led_suit)
    3. Update cards_played_this_round and cards_remaining_by_suit

    Raises:
        IllegalPlayException: If player attempts illegal move
    """

def update_card_counting(self, card: Card, player: Player, led_suit: Optional[str]) -> None:
    """
    Update card counting state after valid play:
    - Add card to cards_played_this_round
    - Decrement cards_remaining_by_suit[card.suit]
    - If player didn't follow suit and led_suit exists: player.mark_void_suit(led_suit)
    """
```

##### Scoring Phase
```python
def scoring_phase(self) -> None:
    """
    Calculate scores for all players:
    1. For each player: call calculate_round_score()
    2. Update player.total_score
    3. Log scores
    4. Set game_phase = 'complete' (or 'setup' for next round)
    """
```

##### Game Flow
```python
def play_round(self, cards_to_deal: int) -> Dict:
    """
    Execute complete round:
    1. setup_round(cards_to_deal)
    2. bidding_phase()
    3. playing_phase()
    4. scoring_phase()
    5. Return round summary (bids, tricks, scores)
    """

def play_full_game(self, round_structure: List[int]) -> Dict:
    """
    Play multiple rounds with varying cards:
    - round_structure: [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7]
    - For each: play_round(cards)
    - Return final scores and game history
    """
```

##### State Queries (for AI/UI)
```python
def get_game_state(self) -> Dict:
    """
    Return complete observable game state:
    {
        'phase': str,
        'round': int,
        'trump': Optional[str],
        'dealer_position': int,
        'players': [
            {
                'name': str,
                'hand': List[Card],  # Only for that player (or all if debug)
                'bid': Optional[int],
                'tricks_won': int,
                'total_score': int
            }, ...
        ],
        'current_trick': {
            'cards_played': [...],
            'led_suit': Optional[str]
        },
        'tricks_history': [...]
    }
    """

def get_legal_actions(self, player: Player) -> List[Union[int, Card]]:
    """
    Return valid actions for player based on game phase:
    - If 'bidding': return valid bids (list of ints)
    - If 'playing': return legal cards (list of Cards)
    """
```

---

### 3. CLI Interface (for Testing)

Add at bottom of `blob.py`:

```python
if __name__ == "__main__":
    """
    Interactive CLI game for manual testing.
    """
    import random

    def random_bot_bid(player, cards_dealt, is_dealer, forbidden_bid):
        """Bot makes random valid bid."""
        valid_bids = [b for b in range(cards_dealt + 1)
                      if not (is_dealer and b == forbidden_bid)]
        return random.choice(valid_bids)

    def random_bot_play(player, legal_cards):
        """Bot plays random legal card."""
        return random.choice(legal_cards)

    def human_bid(player, cards_dealt, is_dealer, forbidden_bid):
        """Human player inputs bid."""
        print(f"\n{player.name}'s hand: {sorted(player.hand)}")
        while True:
            bid = int(input(f"Your bid (0-{cards_dealt}): "))
            if is_dealer and bid == forbidden_bid:
                print(f"Forbidden! Cannot bid {forbidden_bid} (dealer constraint)")
                continue
            if 0 <= bid <= cards_dealt:
                return bid
            print("Invalid bid!")

    def human_play(player, legal_cards):
        """Human player selects card."""
        print(f"\n{player.name}'s hand: {sorted(player.hand)}")
        print(f"Legal plays: {legal_cards}")
        # Show current trick state...
        while True:
            card_str = input("Play card (e.g., 'A♠'): ")
            # Parse and validate...

    # Main game loop
    num_players = 4
    starting_cards = 5
    game = BlobGame(num_players=num_players, player_names=["You", "Bot1", "Bot2", "Bot3"])

    # Generate round structure dynamically
    from ml.game.constants import generate_round_structure
    try:
        round_structure = generate_round_structure(starting_cards, num_players)
        print(f"Round structure: {round_structure}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    for round_num, cards in enumerate(round_structure):
        print(f"\n{'='*50}")
        print(f"ROUND {round_num + 1}: {cards} cards, Trump: {game.trump_suit}")
        print(f"{'='*50}")

        # Bidding phase
        # ... call human_bid for player 0, random_bot_bid for others

        # Playing phase
        # ... display trick, get plays

        # Scoring phase
        # ... show scores

    print("\n\nFINAL SCORES:")
    for player in game.players:
        print(f"{player.name}: {player.total_score}")
```

**Purpose**: Manual validation of game rules, debugging, demo.

---

### 4. Unit Tests (`ml/game/test_blob.py`)

#### Test Structure

```python
import pytest
from ml.game.blob import Card, Deck, Player, Trick, BlobGame
from ml.game.constants import SUITS, RANKS

class TestCard:
    def test_card_creation(self):
        """Test Card initialization and properties."""

    def test_card_equality(self):
        """Test Card comparison."""

    def test_card_sorting(self):
        """Test Cards sort by value correctly."""

    def test_card_string_representation(self):
        """Test __str__ and __repr__."""

class TestDeck:
    def test_deck_initialization(self):
        """Deck has 52 unique cards."""

    def test_deck_shuffle(self):
        """Shuffling changes order."""

    def test_deck_deal(self):
        """Dealing distributes cards correctly."""

    def test_deck_deal_validation(self):
        """Cannot deal more cards than available."""

    def test_deck_reset(self):
        """Reset returns all cards."""

class TestPlayer:
    def test_player_creation(self):
        """Player initializes correctly."""

    def test_receive_and_play_card(self):
        """Can receive and play cards."""

    def test_score_calculation_exact_bid(self):
        """Correct score when bid matches tricks."""

    def test_score_calculation_missed_bid(self):
        """Zero score when bid doesn't match."""

class TestTrick:
    def test_trick_winner_no_trump(self):
        """Highest card in led suit wins (no trump)."""

    def test_trick_winner_with_trump(self):
        """Trump card beats non-trump."""

    def test_trick_winner_multiple_trumps(self):
        """Highest trump wins."""

    def test_led_suit_determination(self):
        """First card sets led suit."""

class TestBlobGame:
    def test_game_initialization(self):
        """Game initializes with correct player count."""

    def test_trump_rotation(self):
        """Trump cycles through suits correctly."""

    def test_forbidden_bid_calculation(self):
        """Dealer's forbidden bid is correct."""

    def test_valid_bid_enforcement(self):
        """Bid validation works for dealer and non-dealer."""

    def test_legal_plays_must_follow_suit(self):
        """Player must follow suit if able."""

    def test_legal_plays_no_suit_restriction(self):
        """Player can play any card if can't follow suit."""

    def test_full_round_3_players(self):
        """Complete round with 3 players."""

    def test_full_round_8_players(self):
        """Complete round with 8 players."""

    def test_scoring_all_players_make_bid(self):
        """All players get points when bids met."""

    def test_scoring_no_players_make_bid(self):
        """All players get zero when bids missed."""

    def test_zero_bid_success(self):
        """Player bidding 0 and winning 0 gets 10 points."""

class TestEdgeCases:
    def test_last_player_cannot_bid_forbidden(self):
        """Dealer cannot make total equal cards dealt."""

    def test_all_players_bid_zero(self):
        """Handle all zeros scenario."""

    def test_no_trump_round(self):
        """No-trump round uses led suit only."""

    def test_single_card_round(self):
        """1-card round works correctly."""

class TestRoundStructure:
    def test_generate_round_structure_basic(self):
        """Generate correct round structure for valid inputs."""

    def test_generate_round_structure_3_players(self):
        """3 players, 7 cards: [7,6,5,4,3,2,1,1,1,2,3,4,5,6,7]"""

    def test_generate_round_structure_4_players(self):
        """4 players, 5 cards: [5,4,3,2,1,1,1,1,1,2,3,4,5]"""

    def test_generate_round_structure_exceeds_deck(self):
        """Raise ValueError if starting_cards * num_players > 52."""

    def test_generate_round_structure_max_valid(self):
        """Find max starting cards for each player count."""

class TestAntiCheat:
    def test_detect_illegal_card_not_in_hand(self):
        """Raise exception if player plays card they don't have."""

    def test_detect_illegal_suit_violation(self):
        """Raise exception if player has led suit but plays different suit."""

    def test_suit_elimination_tracking(self):
        """Mark player void in suit when they don't follow suit."""

    def test_card_counting_updates(self):
        """cards_remaining_by_suit updates correctly after plays."""

    def test_valid_play_different_suit(self):
        """Allow different suit if player has none of led suit."""

    def test_known_void_suits_prevent_false_positives(self):
        """Don't raise exception if player known to be void in suit."""
```

**Coverage Goals**:
- Unit tests: >90% code coverage
- Edge cases: All identified scenarios
- Integration tests: Full game flows

---

### 5. Edge Cases & Special Scenarios

#### Dealer Constraint Edge Cases
```python
# Example: 3 players, 5 cards dealt
# Player 1 bids: 2
# Player 2 bids: 1
# Dealer (Player 3) CANNOT bid: 2 (because 2 + 1 + 2 = 5)
# Valid bids: 0, 1, 3, 4, 5
```

#### No-Trump Round Behavior
```python
# In no-trump rounds:
# - First card of EACH trick sets "trump" for that trick
# - Example: Lead card is 5♥ → Hearts is trump for THIS trick only
# - Next trick: New lead card sets new trump
```

#### 0-Bid Strategy
```python
# Bidding 0 is valid and strategic:
# - If you win 0 tricks: score = 10 points
# - If you win any tricks: score = 0 points
# - Requires careful card play to avoid winning
```

#### Suit Elimination Tracking (Implemented in Phase 1)
```python
# When player doesn't follow suit:
# - Can deduce they lack that suit
# - Mark player.known_void_suits.add(suit)
# - Used for AI belief state (Phase 3)
# - Used for anti-cheat validation (Phase 1)
# - Example: Player plays ♣ when ♥ was led → They have no Hearts
```

#### Maximum Cards Per Player (Deck Limit)
```python
# 52-card deck limits maximum starting cards based on player count:
# 3 players: max 17 cards each (17*3=51)
# 4 players: max 13 cards each (13*4=52)
# 5 players: max 10 cards each (10*5=50)
# 6 players: max 8 cards each (8*6=48)
# 7 players: max 7 cards each (7*7=49)
# 8 players: max 6 cards each (6*8=48)

# generate_round_structure() will raise ValueError if exceeded:
# Example: 8 players, 7 cards starting → 8*7=56 > 52 → ValueError!
```

---

## Implementation Checklist

### Phase 1a.1: Setup (30 min)
- [ ] Create `ml/game/__init__.py`
- [ ] Create `ml/game/constants.py`:
  - [ ] Card constants (suits, ranks, values)
  - [ ] Game constraints (min/max players, deck size)
  - [ ] `generate_round_structure()` function with deck limit validation
- [ ] Setup pytest configuration if needed

### Phase 1a.2: Basic Classes (2 hours)
- [ ] Implement custom exceptions (BlobGameException, IllegalPlayException, etc.)
- [ ] Implement `Card` class with tests
- [ ] Implement `Deck` class with tests
- [ ] Implement `Player` class with tests:
  - [ ] Include `known_void_suits` tracking
  - [ ] Include `cards_played` tracking
  - [ ] Add `mark_void_suit()` and `has_suit()` methods
- [ ] Implement `Trick` class with tests
- [ ] Run tests: `pytest ml/game/test_blob.py -v`

### Phase 1a.3: Game Logic - Setup (1 hour)
- [ ] Implement `BlobGame.__init__()`
- [ ] Implement `BlobGame.setup_round()`
- [ ] Implement `BlobGame.determine_trump()`
- [ ] Test round setup

### Phase 1a.4: Game Logic - Bidding (1.5 hours)
- [ ] Implement `BlobGame.bidding_phase()`
- [ ] Implement `BlobGame.get_forbidden_bid()`
- [ ] Implement `BlobGame.is_valid_bid()`
- [ ] Test bidding logic, especially dealer constraint

### Phase 1a.5: Game Logic - Playing (2.5 hours)
- [ ] Implement `BlobGame.playing_phase()`
- [ ] Implement `BlobGame.play_trick()`
- [ ] Implement `BlobGame.get_legal_plays()`
- [ ] Implement `BlobGame.is_valid_play()`
- [ ] Implement `BlobGame.validate_play_with_anti_cheat()`:
  - [ ] Check card in hand
  - [ ] Detect illegal suit violations
  - [ ] Raise IllegalPlayException when appropriate
- [ ] Implement `BlobGame.update_card_counting()`:
  - [ ] Update cards_played_this_round
  - [ ] Update cards_remaining_by_suit
  - [ ] Mark void suits when detected
- [ ] Test trick playing with various scenarios
- [ ] Test anti-cheat detection

### Phase 1a.6: Game Logic - Scoring (30 min)
- [ ] Implement `BlobGame.scoring_phase()`
- [ ] Test score calculation (exact bid vs. missed)

### Phase 1a.7: Game Flow (1 hour)
- [ ] Implement `BlobGame.play_round()`
- [ ] Implement `BlobGame.play_full_game()`
- [ ] Implement `BlobGame.get_game_state()`
- [ ] Test full game flow

### Phase 1a.8: CLI Interface (1.5 hours)
- [ ] Implement CLI game loop
- [ ] Add human input handlers
- [ ] Add random bot players
- [ ] Add display formatting
- [ ] Manual playtesting

### Phase 1a.9: Testing & Validation (2.5 hours)
- [ ] Write comprehensive unit tests
- [ ] Test `generate_round_structure()` with various player counts
- [ ] Test deck limit validation (e.g., 8 players × 7 cards = error)
- [ ] Test anti-cheat exception raising
- [ ] Test suit elimination tracking
- [ ] Test card counting updates
- [ ] Achieve >90% code coverage
- [ ] Test all edge cases
- [ ] Run code quality checks: `black`, `flake8`, `mypy`

### Phase 1a.10: Documentation (30 min)
- [ ] Add docstrings to all classes/methods
- [ ] Add type hints throughout
- [ ] Add inline comments for complex logic
- [ ] Update README if needed

---

## Success Criteria

### Functional Requirements
✅ Game correctly handles 3-8 players
✅ Bidding phase enforces last-player constraint
✅ Playing phase enforces follow-suit rules
✅ Trick winner determination is correct (trump/no-trump)
✅ Scoring calculates exact bid matching
✅ Trump rotates correctly across rounds
✅ CLI game is playable end-to-end
✅ Round structure generation works with deck limit validation
✅ Anti-cheat detection catches illegal plays
✅ Suit elimination tracking works correctly
✅ Card counting updates after each play

### Code Quality
✅ All pytest tests pass
✅ Code coverage >90%
✅ Passes `black ml/game/`
✅ Passes `flake8 ml/game/`
✅ Passes `mypy ml/game/`
✅ All public methods have docstrings
✅ Type hints on all function signatures

### Deliverables
✅ `ml/game/constants.py` - Game configuration
✅ `ml/game/blob.py` - Core game engine (~600 lines)
✅ `ml/game/test_blob.py` - Unit tests (~400 lines)
✅ Working CLI demo: `python ml/game/blob.py`

### Ready for Phase 2
✅ Game state is serializable (for AI state encoding)
✅ Legal action queries work (for AI action space)
✅ Game logic is deterministic (for MCTS simulation)
✅ No bugs found in manual playtesting

---

## Timeline Estimate

**Total Time**: ~13-15 hours (Weekend 1)

| Task | Time |
|------|------|
| Setup & constants (with round generation) | 0.5h |
| Basic classes + tests (with anti-cheat fields) | 2h |
| Game setup logic | 1h |
| Bidding logic | 1.5h |
| Playing logic + anti-cheat validation | 2.5h |
| Scoring logic | 0.5h |
| Game flow | 1h |
| CLI interface | 1.5h |
| Testing & validation (including anti-cheat) | 2.5h |
| Documentation | 0.5h |

---

## Next Steps (Phase 1b)

After completing `blob.py`:

1. **Code Review**: Validate against game rules
2. **Manual Testing**: Play 10+ games via CLI to find bugs
3. **Edge Case Validation**: Test all identified edge cases
4. **Performance Check**: Ensure game runs efficiently (target: <10ms per move validation)
5. **Integration Prep**: Ensure game state is ready for Phase 2 (state encoding)

Then proceed to **Phase 2: MCTS + Neural Network**.

---

## Notes & Considerations

### Design Decisions
- **Immutable Cards**: Prevents bugs from modifying card state
- **Player manages own state**: Clear separation of concerns
- **Trick as separate class**: Encapsulates winner determination logic
- **Game phases**: Clear state machine for game flow

### Future Extensibility
- Game state should be easily serializable for AI training
- Legal action queries should be efficient (called frequently by MCTS)
- Consider adding `copy()` method to BlobGame for MCTS simulation
- Belief state tracking (Phase 3) will extend Player class

### Testing Strategy
- Test smallest units first (Card, Deck)
- Build up to integration tests (full games)
- Property-based testing for invariants (e.g., card conservation)
- Manual CLI testing for rule validation

---

## References

- [README.md](README.md) - Full project overview
- [CLAUDE.md](CLAUDE.md) - Development guidelines
- Phase 2 will build on this foundation for state encoding

---

## Summary of Key Features

### 1. **Dynamic Round Structure Generation**
- `generate_round_structure(starting_cards, num_players)` function
- Automatically creates: descending → (num_players × 1-card rounds) → ascending
- Example: 4 players, 5 starting cards → `[5,4,3,2,1,1,1,1,1,2,3,4,5]`
- **Deck limit validation**: Raises `ValueError` if `starting_cards * num_players > 52`
- Prevents impossible games (e.g., 8 players × 7 cards = 56 cards needed)

### 2. **Anti-Cheat System**
- **Player tracking**:
  - `known_void_suits`: Suits player has proven they don't have
  - `cards_played`: All cards player has played this round
- **Game tracking**:
  - `cards_played_this_round`: Global card history
  - `cards_remaining_by_suit`: Card counting per suit
- **Validation methods**:
  - `validate_play_with_anti_cheat()`: Strict enforcement with exceptions
  - Detects: cards not in hand, illegal suit violations
  - Raises: `IllegalPlayException` with detailed reason
- **Use cases**:
  - Detect human mistakes in multiplayer games
  - Catch attempted cheating
  - Provide data for AI belief state (Phase 3)

### 3. **Suit Elimination Tracking**
- Automatically marks when player doesn't follow suit
- Updates `player.known_void_suits.add(suit)`
- Used for:
  - Anti-cheat validation (prevent false positives)
  - Card counting (deduce remaining cards)
  - Future AI belief state (Phase 3)

### 4. **Maximum Cards Calculation**
```
Player Count | Max Starting Cards | Calculation
-------------|-------------------|-------------
3 players    | 17 cards          | 17 × 3 = 51
4 players    | 13 cards          | 13 × 4 = 52
5 players    | 10 cards          | 10 × 5 = 50
6 players    | 8 cards           | 8 × 6 = 48
7 players    | 7 cards           | 7 × 7 = 49
8 players    | 6 cards           | 6 × 8 = 48
```

### 5. **Custom Exceptions**
- `BlobGameException`: Base exception class
- `IllegalPlayException`: Illegal card play (with player, card, reason)
- `InvalidBidException`: Invalid bid
- `GameStateException`: Invalid game state for action

---

**Last Updated**: 2025-10-22
**Status**: Ready for implementation
**Version**: 2.0 (with anti-cheat and dynamic rounds)
