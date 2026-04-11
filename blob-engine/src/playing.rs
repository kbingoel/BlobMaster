//! Trick-taking phase: legal-play generation, play application, and
//! per-round scoring.
//!
//! Follows the classical Blob rules:
//!
//! - If the current player has any cards of the led suit, they must play one
//!   of them; otherwise they may play any card (including trump).
//! - The first player of a trick may play any card in hand.
//! - The trick winner is the player of the highest trump if any trump was
//!   played; otherwise the player of the highest card of the led suit. In
//!   `NoTrump` rounds there is no trump override, so led suit always wins.
//!
//! `legal_plays` returns a `u64` bitmask (~5ns) rather than a `Vec<u8>` —
//! MCTS expansion leans on this being allocation-free.

use crate::card::NUM_RANKS;
use crate::round::NO_TRUMP;
use crate::state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS};

/// Bitmask of legal cards the current player may play.
///
/// Returns `0` outside the playing phase. Within the playing phase:
/// - First card of the trick: entire hand.
/// - Subsequent card, led suit present in hand: only cards of the led suit.
/// - Subsequent card, void in led suit: entire hand.
#[inline]
pub fn legal_plays(state: &BlobState) -> u64 {
    if state.phase() != GamePhase::Playing {
        return 0;
    }
    let hand = state.hands[state.current_player as usize];
    if state.trick_cards_played == 0 {
        return hand;
    }
    let led_card = state.trick_play_order[0];
    let led_suit = led_card / NUM_RANKS;
    // 13-bit mask for that suit, sitting at `suit*13` in the u64 layout.
    let suit_mask: u64 = 0x1FFFu64 << (led_suit * NUM_RANKS);
    let of_suit = hand & suit_mask;
    if of_suit != 0 {
        of_suit
    } else {
        hand
    }
}

/// Apply `card` as the current player's play and advance state.
///
/// Removes the card from the player's hand, records it in the in-progress
/// trick, and maintains `played_this_round` incrementally (read by the
/// encoder for `cards_above_remaining` / `is_highest_in_suit`).
///
/// When the trick completes (all seats have played): determines the winner,
/// records the trick into `trick_history`, increments `tricks_won[winner]`,
/// sets `trick_leader = current_player = winner`, clears the in-progress
/// trick buffers, and increments `tricks_completed`. If this was the last
/// trick of the round, transitions to `GamePhase::Scoring`.
///
/// Panics in debug if `card` is not in `legal_plays(state)`.
pub fn apply_play(state: &mut BlobState, card: u8) {
    debug_assert_eq!(state.phase(), GamePhase::Playing);
    debug_assert!(
        card < 52 && (legal_plays(state) >> card) & 1 == 1,
        "illegal play {card} for player {} (legal mask = {:064b})",
        state.current_player,
        legal_plays(state)
    );

    let p = state.current_player as usize;
    let bit = 1u64 << card;
    state.hands[p] &= !bit;
    state.played_this_round |= bit;
    let slot = state.trick_cards_played as usize;
    state.trick_play_order[slot] = card;
    state.trick_cards_played += 1;

    if state.trick_cards_played < state.num_players {
        state.current_player = (state.current_player + 1) % state.num_players;
        return;
    }

    // Trick is complete — decide winner and advance round state.
    let (winner_slot, suit_led) = winning_slot(state);
    let winner = (state.trick_leader + winner_slot) % state.num_players;

    let mut cards = [(0u8, 0u8); MAX_PLAYERS];
    for (i, slot) in cards
        .iter_mut()
        .enumerate()
        .take(state.num_players as usize)
    {
        let player = (state.trick_leader + i as u8) % state.num_players;
        *slot = (player, state.trick_play_order[i]);
    }
    let rec = TrickRecord {
        cards,
        num_played: state.num_players,
        winner,
        suit_led,
    };
    let t = state.tricks_completed as usize;
    state.trick_history[t] = rec;
    state.tricks_completed += 1;
    state.tricks_won[winner as usize] += 1;

    state.trick_play_order = [0; MAX_PLAYERS];
    state.trick_cards_played = 0;
    state.trick_leader = winner;
    state.current_player = winner;

    if state.tricks_completed == state.cards_dealt {
        state.game_phase = GamePhase::Scoring as u8;
    }
}

/// Returns `(winner_slot_within_play_order, suit_led)` for a completed trick.
///
/// Precondition: `state.trick_cards_played == state.num_players`.
fn winning_slot(state: &BlobState) -> (u8, u8) {
    let lead = state.trick_play_order[0];
    let suit_led = lead / NUM_RANKS;
    let trump = state.trump_suit;
    let trump_active = trump != NO_TRUMP;

    let mut best_slot: u8 = 0;
    let mut best_rank: u8 = lead % NUM_RANKS;
    let mut best_is_trump = trump_active && suit_led == trump;

    for i in 1..state.num_players as usize {
        let c = state.trick_play_order[i];
        let c_suit = c / NUM_RANKS;
        let c_rank = c % NUM_RANKS;
        let c_is_trump = trump_active && c_suit == trump;

        let takes_lead = if best_is_trump {
            c_is_trump && c_rank > best_rank
        } else if c_is_trump {
            true
        } else {
            c_suit == suit_led && c_rank > best_rank
        };

        if takes_lead {
            best_slot = i as u8;
            best_rank = c_rank;
            best_is_trump = c_is_trump;
        }
    }

    (best_slot, suit_led)
}

/// Compute per-round scores and accumulate them into `cumulative_scores`.
///
/// All-or-nothing: `score[i] = 10 + bid[i]` iff `tricks_won[i] == bid[i]`,
/// else 0. Returns the per-round score array (slots beyond `num_players`
/// stay zero). Phase must be `Scoring`.
pub fn score_round(state: &mut BlobState) -> [u8; MAX_PLAYERS] {
    debug_assert_eq!(state.phase(), GamePhase::Scoring);
    let mut out = [0u8; MAX_PLAYERS];
    for (i, slot) in out
        .iter_mut()
        .enumerate()
        .take(state.num_players as usize)
    {
        if state.tricks_won[i] == state.bids[i] {
            let s = 10 + state.bids[i];
            *slot = s;
            state.cumulative_scores[i] += s as u16;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Suit};
    use crate::round::NO_TRUMP;
    use crate::state::{BlobState, GamePhase};

    fn card(s: Suit, r: u8) -> u8 {
        Card::new(s, r).index()
    }

    /// Build a state mid-round in the playing phase with a given trump.
    ///
    /// Hands are left empty — individual tests populate them.
    fn playing_state(num_players: u8, cards_dealt: u8, dealer: u8, trump: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = num_players;
        s.cards_dealt = cards_dealt;
        s.dealer = dealer;
        s.trump_suit = trump;
        s.game_phase = GamePhase::Playing as u8;
        s.trick_leader = (dealer + 1) % num_players;
        s.current_player = s.trick_leader;
        s
    }

    fn set_hand(s: &mut BlobState, player: u8, cards: &[u8]) {
        let mut h: u64 = 0;
        for &c in cards {
            h |= 1u64 << c;
        }
        s.hands[player as usize] = h;
    }

    fn legal_set(s: &BlobState) -> Vec<u8> {
        let mask = legal_plays(s);
        (0..52u8).filter(|c| (mask >> c) & 1 == 1).collect()
    }

    // -- legal_plays ---------------------------------------------------------

    #[test]
    fn legal_plays_returns_zero_outside_playing() {
        let mut s = playing_state(4, 5, 0, Suit::Spades as u8);
        set_hand(&mut s, 1, &[card(Suit::Hearts, 0), card(Suit::Spades, 12)]);
        s.game_phase = GamePhase::Bidding as u8;
        assert_eq!(legal_plays(&s), 0);
        s.game_phase = GamePhase::Scoring as u8;
        assert_eq!(legal_plays(&s), 0);
    }

    #[test]
    fn legal_plays_first_card_full_hand() {
        // First card of the first trick — every card in hand is legal.
        let mut s = playing_state(3, 5, 0, Suit::Spades as u8);
        let hand = [
            card(Suit::Hearts, 0),
            card(Suit::Hearts, 5),
            card(Suit::Clubs, 1),
            card(Suit::Diamonds, 11),
            card(Suit::Spades, 12),
        ];
        set_hand(&mut s, 1, &hand);
        // current_player == 1 (left of dealer 0).
        let legal = legal_set(&s);
        assert_eq!(legal.len(), 5);
        for c in hand {
            assert!(legal.contains(&c));
        }
    }

    #[test]
    fn legal_plays_must_follow_suit() {
        // Hearts led — player with hearts may only play hearts.
        let mut s = playing_state(3, 5, 0, Suit::Spades as u8);
        set_hand(
            &mut s,
            2,
            &[
                card(Suit::Hearts, 3),
                card(Suit::Hearts, 6),
                card(Suit::Clubs, 1),
                card(Suit::Diamonds, 10),
            ],
        );
        // Player 1 led with 2♥.
        s.trick_leader = 1;
        s.trick_play_order[0] = card(Suit::Hearts, 0);
        s.trick_cards_played = 1;
        s.current_player = 2;
        let legal = legal_set(&s);
        assert_eq!(legal.len(), 2);
        assert!(legal.contains(&card(Suit::Hearts, 3)));
        assert!(legal.contains(&card(Suit::Hearts, 6)));
    }

    #[test]
    fn legal_plays_void_can_play_any() {
        // Hearts led, player void in hearts — entire hand legal (including trump).
        let mut s = playing_state(3, 5, 0, Suit::Spades as u8);
        set_hand(
            &mut s,
            2,
            &[
                card(Suit::Clubs, 1),
                card(Suit::Diamonds, 10),
                card(Suit::Spades, 12),
            ],
        );
        s.trick_play_order[0] = card(Suit::Hearts, 0);
        s.trick_cards_played = 1;
        s.current_player = 2;
        let legal = legal_set(&s);
        assert_eq!(legal.len(), 3);
    }

    #[test]
    fn legal_plays_trump_not_mandatory_when_void() {
        // Rust rules match legacy: void-in-led-suit means *any* card, not
        // "must play trump if available". Player with only trump and off-suit
        // cards can still pick the off-suit.
        let mut s = playing_state(3, 5, 0, Suit::Spades as u8);
        set_hand(
            &mut s,
            2,
            &[card(Suit::Clubs, 11), card(Suit::Spades, 4)],
        );
        s.trick_play_order[0] = card(Suit::Hearts, 7);
        s.trick_cards_played = 1;
        s.current_player = 2;
        let legal = legal_set(&s);
        assert_eq!(legal.len(), 2);
    }

    #[test]
    fn legal_plays_exact_suit_mask_shape() {
        // Sanity: the returned mask for "must follow ♦" really is ∩ ♦ suit.
        let mut s = playing_state(4, 6, 0, NO_TRUMP);
        let diamonds_mask: u64 = 0x1FFFu64 << (3 * NUM_RANKS);
        set_hand(
            &mut s,
            1,
            &[
                card(Suit::Diamonds, 0),
                card(Suit::Diamonds, 6),
                card(Suit::Diamonds, 12),
                card(Suit::Spades, 12),
            ],
        );
        s.trick_play_order[0] = card(Suit::Diamonds, 1);
        s.trick_cards_played = 1;
        s.current_player = 1;
        assert_eq!(legal_plays(&s), s.hands[1] & diamonds_mask);
    }

    // -- winning_slot / apply_play trick winner -----------------------------

    /// Play a synthetic trick: set each player's hand to exactly one card and
    /// drive `apply_play` through all seats. Returns the finished state.
    fn play_one_trick(
        num_players: u8,
        trump: u8,
        leader: u8,
        plays: &[(u8, u8)], // (player, card) in play order, starting with leader
    ) -> BlobState {
        assert_eq!(plays.len(), num_players as usize);
        assert_eq!(plays[0].0, leader);

        let mut s = playing_state(num_players, num_players, (leader + num_players - 1) % num_players, trump);
        s.trick_leader = leader;
        s.current_player = leader;
        // cards_dealt = num_players here so that a single trick finishes the round.
        s.cards_dealt = 1;
        // Hands: each player holds only their one card.
        for &(p, c) in plays {
            s.hands[p as usize] = 1u64 << c;
        }
        // But cards_dealt was 1 — each hand has exactly 1 card, so one trick
        // ends the round.
        for &(_, c) in plays {
            apply_play(&mut s, c);
        }
        s
    }

    #[test]
    fn winner_no_trump_highest_led_suit_wins() {
        // No-trump round, hearts led. Alice plays A♥ and wins.
        let s = play_one_trick(
            3,
            NO_TRUMP,
            0,
            &[
                (0, card(Suit::Hearts, 12)), // A♥
                (1, card(Suit::Hearts, 11)), // K♥
                (2, card(Suit::Hearts, 10)), // Q♥
            ],
        );
        assert_eq!(s.trick_history[0].winner, 0);
        assert_eq!(s.tricks_won[..3], [1, 0, 0]);
    }

    #[test]
    fn winner_off_suit_cannot_beat_led_suit() {
        // Hearts led. A♠ off-suit cannot beat even 2♥.
        let s = play_one_trick(
            3,
            NO_TRUMP,
            0,
            &[
                (0, card(Suit::Hearts, 0)),  // 2♥
                (1, card(Suit::Spades, 12)), // A♠ (off suit, no trump)
                (2, card(Suit::Hearts, 1)),  // 3♥
            ],
        );
        assert_eq!(s.trick_history[0].winner, 2);
    }

    #[test]
    fn winner_trump_beats_non_trump() {
        // Spades trump, hearts led — lowest trump beats highest non-trump.
        let s = play_one_trick(
            3,
            Suit::Spades as u8,
            0,
            &[
                (0, card(Suit::Hearts, 12)), // A♥ led
                (1, card(Suit::Spades, 0)),  // 2♠ trump
                (2, card(Suit::Hearts, 11)), // K♥
            ],
        );
        assert_eq!(s.trick_history[0].winner, 1);
    }

    #[test]
    fn winner_highest_trump_wins_among_trumps() {
        let s = play_one_trick(
            3,
            Suit::Spades as u8,
            0,
            &[
                (0, card(Suit::Spades, 3)),  // 5♠ trump
                (1, card(Suit::Spades, 11)), // K♠ trump
                (2, card(Suit::Hearts, 12)), // A♥
            ],
        );
        assert_eq!(s.trick_history[0].winner, 1);
    }

    #[test]
    fn winner_trump_is_led_suit() {
        // Trump suit *is* led. Highest trump in play wins — both trick winner
        // paths collapse to the same comparison.
        let s = play_one_trick(
            2,
            Suit::Spades as u8,
            0,
            &[
                (0, card(Suit::Spades, 11)), // K♠ led
                (1, card(Suit::Spades, 12)), // A♠
            ],
        );
        assert_eq!(s.trick_history[0].winner, 1);
    }

    #[test]
    fn winner_no_trump_round_has_no_override() {
        // NoTrump round: spade plays don't act as trump.
        let s = play_one_trick(
            3,
            NO_TRUMP,
            0,
            &[
                (0, card(Suit::Hearts, 11)), // K♥ led
                (1, card(Suit::Spades, 12)), // A♠ (no trump, so ignored)
                (2, card(Suit::Hearts, 12)), // A♥ wins
            ],
        );
        assert_eq!(s.trick_history[0].winner, 2);
    }

    #[test]
    fn winner_records_suit_led_in_history() {
        let s = play_one_trick(
            3,
            Suit::Spades as u8,
            0,
            &[
                (0, card(Suit::Diamonds, 5)),
                (1, card(Suit::Diamonds, 8)),
                (2, card(Suit::Diamonds, 3)),
            ],
        );
        assert_eq!(s.trick_history[0].suit_led, Suit::Diamonds as u8);
    }

    // -- apply_play state updates --------------------------------------------

    #[test]
    fn apply_play_advances_current_player_mid_trick() {
        let mut s = playing_state(4, 3, 0, Suit::Spades as u8);
        set_hand(
            &mut s,
            1,
            &[
                card(Suit::Hearts, 0),
                card(Suit::Clubs, 2),
                card(Suit::Diamonds, 5),
            ],
        );
        set_hand(&mut s, 2, &[card(Suit::Hearts, 2), card(Suit::Clubs, 0), card(Suit::Clubs, 3)]);
        set_hand(&mut s, 3, &[card(Suit::Hearts, 4), card(Suit::Clubs, 5), card(Suit::Clubs, 6)]);
        set_hand(&mut s, 0, &[card(Suit::Hearts, 6), card(Suit::Clubs, 9), card(Suit::Clubs, 10)]);

        // Player 1 leads the 2♥.
        apply_play(&mut s, card(Suit::Hearts, 0));
        assert_eq!(s.current_player, 2);
        assert_eq!(s.trick_cards_played, 1);
        assert_eq!(s.trick_leader, 1);
        assert_eq!(s.trick_play_order[0], card(Suit::Hearts, 0));
        assert_eq!(s.hands[1] & (1u64 << card(Suit::Hearts, 0)), 0);
        assert!(s.played_this_round & (1u64 << card(Suit::Hearts, 0)) != 0);
    }

    #[test]
    fn apply_play_wraps_current_player_around_last_seat() {
        let mut s = playing_state(4, 2, 2, Suit::Spades as u8);
        // trick_leader = 3 (left of dealer 2); current_player = 3.
        set_hand(&mut s, 3, &[card(Suit::Hearts, 0), card(Suit::Hearts, 1)]);
        set_hand(&mut s, 0, &[card(Suit::Hearts, 2), card(Suit::Hearts, 3)]);
        set_hand(&mut s, 1, &[card(Suit::Hearts, 4), card(Suit::Hearts, 5)]);
        set_hand(&mut s, 2, &[card(Suit::Hearts, 6), card(Suit::Hearts, 7)]);
        apply_play(&mut s, card(Suit::Hearts, 0));
        assert_eq!(s.current_player, 0);
        apply_play(&mut s, card(Suit::Hearts, 2));
        assert_eq!(s.current_player, 1);
    }

    #[test]
    fn apply_play_completes_trick_and_resets_in_progress_state() {
        let mut s = playing_state(3, 2, 0, Suit::Spades as u8);
        // trick_leader = 1.
        set_hand(&mut s, 1, &[card(Suit::Hearts, 0), card(Suit::Clubs, 0)]);
        set_hand(&mut s, 2, &[card(Suit::Hearts, 5), card(Suit::Clubs, 1)]);
        set_hand(&mut s, 0, &[card(Suit::Hearts, 3), card(Suit::Clubs, 2)]);

        apply_play(&mut s, card(Suit::Hearts, 0)); // player 1
        apply_play(&mut s, card(Suit::Hearts, 5)); // player 2
        apply_play(&mut s, card(Suit::Hearts, 3)); // player 0

        // Trick complete. Player 2 wins (5♥ beats 3♥ and 2♥).
        assert_eq!(s.tricks_completed, 1);
        assert_eq!(s.tricks_won[2], 1);
        assert_eq!(s.tricks_won[0], 0);
        assert_eq!(s.tricks_won[1], 0);
        assert_eq!(s.trick_leader, 2);
        assert_eq!(s.current_player, 2);
        assert_eq!(s.trick_cards_played, 0);
        assert_eq!(s.trick_play_order, [0; MAX_PLAYERS]);
        // Still in playing phase — 1 trick of 2 remaining.
        assert_eq!(s.phase(), GamePhase::Playing);
    }

    #[test]
    fn apply_play_records_trick_history_in_play_order() {
        let mut s = playing_state(3, 1, 0, Suit::Spades as u8);
        set_hand(&mut s, 1, &[card(Suit::Hearts, 7)]);
        set_hand(&mut s, 2, &[card(Suit::Hearts, 2)]);
        set_hand(&mut s, 0, &[card(Suit::Hearts, 4)]);

        apply_play(&mut s, card(Suit::Hearts, 7)); // player 1 leads
        apply_play(&mut s, card(Suit::Hearts, 2)); // player 2
        apply_play(&mut s, card(Suit::Hearts, 4)); // player 0

        let rec = &s.trick_history[0];
        assert_eq!(rec.num_played, 3);
        assert_eq!(rec.suit_led, Suit::Hearts as u8);
        assert_eq!(rec.winner, 1);
        // Cards stored in play order starting at trick_leader.
        assert_eq!(rec.cards[0], (1, card(Suit::Hearts, 7)));
        assert_eq!(rec.cards[1], (2, card(Suit::Hearts, 2)));
        assert_eq!(rec.cards[2], (0, card(Suit::Hearts, 4)));
    }

    #[test]
    fn apply_play_transitions_to_scoring_after_last_trick() {
        // 3 players × 1 card — the round ends after one trick.
        let mut s = playing_state(3, 1, 0, Suit::Spades as u8);
        set_hand(&mut s, 1, &[card(Suit::Hearts, 7)]);
        set_hand(&mut s, 2, &[card(Suit::Spades, 0)]); // trump
        set_hand(&mut s, 0, &[card(Suit::Hearts, 4)]);
        apply_play(&mut s, card(Suit::Hearts, 7));
        apply_play(&mut s, card(Suit::Spades, 0));
        assert_eq!(s.phase(), GamePhase::Playing);
        apply_play(&mut s, card(Suit::Hearts, 4));
        assert_eq!(s.phase(), GamePhase::Scoring);
        assert_eq!(s.tricks_won[2], 1);
    }

    #[test]
    fn apply_play_full_round_multi_trick() {
        // 3 players × 2 cards — two tricks to complete the round.
        let mut s = playing_state(3, 2, 0, Suit::Spades as u8);
        // trick_leader = 1, current_player = 1.
        set_hand(&mut s, 1, &[card(Suit::Hearts, 0), card(Suit::Clubs, 0)]);
        set_hand(&mut s, 2, &[card(Suit::Hearts, 5), card(Suit::Clubs, 1)]);
        set_hand(&mut s, 0, &[card(Suit::Hearts, 3), card(Suit::Clubs, 2)]);

        // Trick 1 (hearts led): winner = player 2 (5♥).
        apply_play(&mut s, card(Suit::Hearts, 0));
        apply_play(&mut s, card(Suit::Hearts, 5));
        apply_play(&mut s, card(Suit::Hearts, 3));
        assert_eq!(s.trick_leader, 2);

        // Trick 2 (clubs led by player 2): winner = player 0 (2♣).
        apply_play(&mut s, card(Suit::Clubs, 1));
        apply_play(&mut s, card(Suit::Clubs, 2));
        apply_play(&mut s, card(Suit::Clubs, 0));

        assert_eq!(s.phase(), GamePhase::Scoring);
        assert_eq!(s.tricks_completed, 2);
        assert_eq!(s.tricks_won[0], 1);
        assert_eq!(s.tricks_won[1], 0);
        assert_eq!(s.tricks_won[2], 1);
        // All hands empty.
        assert_eq!(s.hands[0], 0);
        assert_eq!(s.hands[1], 0);
        assert_eq!(s.hands[2], 0);
        // played_this_round accumulates everything dealt.
        assert_eq!(s.played_this_round.count_ones(), 6);
    }

    #[test]
    #[should_panic(expected = "illegal play")]
    fn apply_play_panics_on_card_not_in_hand() {
        let mut s = playing_state(3, 3, 0, Suit::Spades as u8);
        set_hand(&mut s, 1, &[card(Suit::Hearts, 0)]);
        // Attempting to play a card the player doesn't hold.
        apply_play(&mut s, card(Suit::Spades, 12));
    }

    #[test]
    #[should_panic(expected = "illegal play")]
    fn apply_play_panics_on_suit_violation() {
        let mut s = playing_state(3, 3, 0, Suit::Spades as u8);
        set_hand(
            &mut s,
            1,
            &[card(Suit::Hearts, 0), card(Suit::Clubs, 0)],
        );
        set_hand(
            &mut s,
            2,
            &[card(Suit::Hearts, 4), card(Suit::Clubs, 4)],
        );
        set_hand(
            &mut s,
            0,
            &[card(Suit::Hearts, 7), card(Suit::Clubs, 7)],
        );
        // Player 1 leads 2♥.
        apply_play(&mut s, card(Suit::Hearts, 0));
        // Player 2 has hearts but tries to play clubs.
        apply_play(&mut s, card(Suit::Clubs, 4));
    }

    // -- score_round ---------------------------------------------------------

    #[test]
    fn score_round_exact_bid_earns_ten_plus_bid() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 5;
        s.game_phase = GamePhase::Scoring as u8;
        s.bids[0] = 3;
        s.tricks_won[0] = 3;
        s.bids[1] = 1;
        s.tricks_won[1] = 0;
        s.bids[2] = 0;
        s.tricks_won[2] = 0;

        let round = score_round(&mut s);
        assert_eq!(round[0], 13);
        assert_eq!(round[1], 0);
        assert_eq!(round[2], 10);
        assert_eq!(s.cumulative_scores[0], 13);
        assert_eq!(s.cumulative_scores[1], 0);
        assert_eq!(s.cumulative_scores[2], 10);
    }

    #[test]
    fn score_round_over_and_under_both_score_zero() {
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 3;
        s.game_phase = GamePhase::Scoring as u8;
        s.bids[..4].copy_from_slice(&[2, 1, 0, 3]);
        s.tricks_won[..4].copy_from_slice(&[1, 2, 1, 2]); // all miss
        let round = score_round(&mut s);
        assert_eq!(round[..4], [0, 0, 0, 0]);
        assert_eq!(s.cumulative_scores[..4], [0, 0, 0, 0]);
    }

    #[test]
    fn score_round_zero_bid_success_yields_ten() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 5;
        s.game_phase = GamePhase::Scoring as u8;
        s.bids[0] = 0;
        s.tricks_won[0] = 0;
        s.bids[1] = 2;
        s.tricks_won[1] = 2;
        s.bids[2] = 3;
        s.tricks_won[2] = 3;
        let round = score_round(&mut s);
        assert_eq!(round[0], 10);
        assert_eq!(round[1], 12);
        assert_eq!(round[2], 13);
    }

    #[test]
    fn score_round_ignores_unused_player_slots() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 4;
        s.game_phase = GamePhase::Scoring as u8;
        s.bids[0] = 1;
        s.tricks_won[0] = 1;
        // Slots [3..8] uninitialized (all zero) and must stay zeroed.
        let round = score_round(&mut s);
        assert_eq!(round[0], 11);
        for (i, &slot) in round.iter().enumerate().skip(3) {
            assert_eq!(slot, 0);
            assert_eq!(s.cumulative_scores[i], 0);
        }
    }

    #[test]
    fn score_round_accumulates_across_multiple_rounds() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 3;
        s.game_phase = GamePhase::Scoring as u8;

        s.bids[0] = 2;
        s.tricks_won[0] = 2;
        score_round(&mut s);
        assert_eq!(s.cumulative_scores[0], 12);

        // Next round: bid 1, make it. Total = 12 + 11 = 23.
        s.bids[0] = 1;
        s.tricks_won[0] = 1;
        score_round(&mut s);
        assert_eq!(s.cumulative_scores[0], 23);

        // Next round: bid 3, win 2 → miss. Total unchanged.
        s.bids[0] = 3;
        s.tricks_won[0] = 2;
        score_round(&mut s);
        assert_eq!(s.cumulative_scores[0], 23);
    }

    #[test]
    #[should_panic]
    fn score_round_panics_outside_scoring_phase() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.game_phase = GamePhase::Playing as u8;
        score_round(&mut s);
    }

    // -- integration with bidding + dealing ---------------------------------

    #[test]
    fn full_round_deal_bid_play_score_integration() {
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        // Deal a 3p×4c round, have everyone bid 0, then play out by always
        // choosing the lowest legal card. Verify round scores make sense.
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 4;
        s.dealer = 0;
        s.trump_suit = Suit::Spades as u8;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        start_round(&mut s, &mut rng);
        assert_eq!(s.phase(), GamePhase::Bidding);

        // Everyone bids 0 (dealer must verify legality).
        for _ in 0..3 {
            let mask = legal_bids(&s);
            assert!(mask != 0);
            // Prefer 0; if dealer's forbidden value is 0, pick 1.
            let bid = if (mask & 1) == 1 { 0 } else { 1 };
            apply_bid(&mut s, bid);
        }
        assert_eq!(s.phase(), GamePhase::Playing);
        assert_eq!(s.tricks_completed, 0);

        // Play out — lowest legal card each turn.
        for _ in 0..(s.cards_dealt as usize * s.num_players as usize) {
            let mask = legal_plays(&s);
            assert!(mask != 0, "no legal plays mid-round");
            let c = mask.trailing_zeros() as u8;
            apply_play(&mut s, c);
        }

        assert_eq!(s.phase(), GamePhase::Scoring);
        assert_eq!(s.tricks_completed as u32, s.cards_dealt as u32);
        // Tricks won sum to cards dealt (invariant).
        let sum: u32 = s.tricks_won[..s.num_players as usize]
            .iter()
            .map(|&t| t as u32)
            .sum();
        assert_eq!(sum, s.cards_dealt as u32);

        // All hands empty after the round.
        for i in 0..s.num_players as usize {
            assert_eq!(s.hands[i], 0);
        }

        let round = score_round(&mut s);
        // Scoring is all-or-nothing: at least one player missed (someone
        // won a trick but bid 0), and at most one player made their bid.
        for (i, &r) in round.iter().enumerate().take(s.num_players as usize) {
            if s.tricks_won[i] == s.bids[i] {
                assert_eq!(r, 10 + s.bids[i]);
            } else {
                assert_eq!(r, 0);
            }
        }
    }
}
