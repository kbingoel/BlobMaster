//! Per-round utility (gen-2.md §5.1).
//!
//! Each round is a fresh deal and round scores add up, so search and the
//! value target score a seat by the round being played alone:
//!
//! ```text
//! u_s = ŝ_s − λ · mean_{j≠s} ŝ_j          ŝ = round points / (10 + cards dealt)
//! ```
//!
//! - Round points are `10 + bid` for an exact bid and 0 otherwise, so ŝ lies
//!   in [0, 1] on a fixed scale: nothing is clipped and nothing depends on
//!   the rest of the game.
//! - λ = 1 ([`DEFAULT_LAMBDA`]) is "my points minus the table's", so spoiling
//!   an opponent's bid has value; λ = 0 is "my points only".
//! - The value net predicts ŝ for every seat and search applies λ, so λ can
//!   change without retraining.

use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// Default λ: my points minus the table's mean.
pub const DEFAULT_LAMBDA: f32 = 1.0;

/// Points each seat scores in a finished round: `10 + bid` if it took
/// exactly its bid, else 0. Slots `>= num_players` are 0.
pub fn round_points(state: &BlobState) -> [u8; MAX_PLAYERS] {
    debug_assert!(
        matches!(state.phase(), GamePhase::Scoring | GamePhase::Complete),
        "round_points on an unfinished round ({:?})",
        state.phase()
    );
    let mut out = [0u8; MAX_PLAYERS];
    for (i, slot) in out.iter_mut().enumerate().take(state.num_players as usize) {
        if state.tricks_won[i] == state.bids[i] {
            *slot = 10 + state.bids[i];
        }
    }
    out
}

/// The most a seat can score in a round with `cards_dealt` cards; ŝ is
/// round points divided by this.
#[inline]
pub fn score_scale(cards_dealt: u8) -> f32 {
    10.0 + cards_dealt as f32
}

/// ŝ for every seat from its round points.
pub fn normalized_scores(points: &[u8; MAX_PLAYERS], cards_dealt: u8) -> [f32; MAX_PLAYERS] {
    let scale = score_scale(cards_dealt);
    points.map(|p| p as f32 / scale)
}

/// `u_s` for every seat from each seat's ŝ (expected or actual). Slots
/// `>= num_players` are 0.
pub fn utilities(s_hat: &[f32; MAX_PLAYERS], num_players: u8, lambda: f32) -> [f32; MAX_PLAYERS] {
    let n = num_players as usize;
    let total: f32 = s_hat[..n].iter().sum();
    let others = n.max(2) as f32 - 1.0;
    let mut u = [0.0f32; MAX_PLAYERS];
    for s in 0..n {
        u[s] = s_hat[s] - lambda * (total - s_hat[s]) / others;
    }
    u
}

/// Exact `u_s` for every seat of a finished round.
pub fn terminal_utilities(state: &BlobState, lambda: f32) -> [f32; MAX_PLAYERS] {
    let s_hat = normalized_scores(&round_points(state), state.cards_dealt);
    utilities(&s_hat, state.num_players, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished(bids: &[u8], tricks: &[u8], cards_dealt: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = bids.len() as u8;
        s.cards_dealt = cards_dealt;
        s.game_phase = GamePhase::Scoring as u8;
        s.bids[..bids.len()].copy_from_slice(bids);
        s.tricks_won[..tricks.len()].copy_from_slice(tricks);
        s
    }

    #[test]
    fn round_points_pay_exact_bids_only() {
        let s = finished(&[2, 1, 0, 3], &[2, 0, 0, 2], 5);
        assert_eq!(round_points(&s)[..5], [12, 0, 10, 0, 0]);
    }

    #[test]
    fn utility_is_my_share_minus_the_tables_mean() {
        // 4 seats, 5 cards: scale 15. Points 12, 0, 10, 0.
        let s = finished(&[2, 1, 0, 3], &[2, 0, 0, 2], 5);
        let u = terminal_utilities(&s, 1.0);
        let s_hat = [12.0 / 15.0, 0.0, 10.0 / 15.0, 0.0];
        for seat in 0..4 {
            let others: f32 = (0..4).filter(|&j| j != seat).map(|j| s_hat[j]).sum::<f32>() / 3.0;
            assert!((u[seat] - (s_hat[seat] - others)).abs() < 1e-6, "seat {seat}");
        }
        assert!(u[4..].iter().all(|&v| v == 0.0));
        // λ = 0 is my points only.
        let mine = terminal_utilities(&s, 0.0);
        assert_eq!(mine[..4], s_hat);
    }

    #[test]
    fn utilities_sum_to_zero_at_lambda_one() {
        // Each ŝ_j appears in n − 1 means with weight 1/(n − 1).
        let s_hat = [0.9, 0.1, 0.0, 0.7, 0.35, 0.0, 0.0, 0.0];
        let u = utilities(&s_hat, 5, 1.0);
        assert!(u.iter().sum::<f32>().abs() < 1e-6);
    }

    #[test]
    fn normalized_scores_lie_in_unit_interval() {
        // The best possible round: bid every card and make it.
        let s = finished(&[7, 0, 0], &[7, 0, 0], 7);
        let s_hat = normalized_scores(&round_points(&s), 7);
        assert_eq!(s_hat[0], 1.0);
        assert!((s_hat[1] - 10.0 / 17.0).abs() < 1e-6);
    }
}
