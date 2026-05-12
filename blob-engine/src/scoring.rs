//! C2-fix helpers: in-tree terminal value computation aligned with
//! `blob_nn::self_play::backfill_values` so MCTS terminal leaves and the
//! training-target values live on a single coherent scale.
//!
//! See [fix-mcts-plan.md](../../../fix-mcts-plan.md) Step 2 for the
//! rationale; the bundled C2a/C2b/C2c callouts in that plan map to:
//! - **C2a** — multi-seat backprop: covered by
//!   [`crate::mcts::backprop_terminal`], which consumes the per-seat
//!   vector this module emits.
//! - **C2b** — round-boundary truncation: out of scope. MCTS's
//!   `apply_action` no-ops on `Scoring`/`Complete`, so a round-1 search
//!   sees no signal from rounds 2..N. [`terminal_z_scores`] returns the
//!   closest single-statistic approximation available without rolling a
//!   fresh deal forward.
//! - **C2c** — scale alignment: [`z_score_clip`] is the single source of
//!   truth for the z-score statistic; both [`terminal_z_scores`] and
//!   `backfill_values` call it on `cumulative_scores`-derived inputs so
//!   in-tree Q and the training value target cannot drift in scale.

use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// Floor on the standard-deviation denominator. Mirrors
/// `blob_nn::self_play::backfill_values` so this helper produces
/// identical output for any final state.
pub const Z_SCORE_EPS: f32 = 1e-6;

/// Z-score `scores[..n]` and clip to `[-1, 1]`. Returns all-zero when
/// the std underflows `Z_SCORE_EPS` (all-equal scores: z-score is
/// undefined). Slots `>= n` stay zero.
///
/// Shared with `backfill_values` (single source of truth for C2c).
#[inline]
pub fn z_score_clip(scores: &[f32; MAX_PLAYERS], n: usize) -> [f32; MAX_PLAYERS] {
    let mut z = [0.0f32; MAX_PLAYERS];
    if n == 0 {
        return z;
    }
    debug_assert!(n <= MAX_PLAYERS);
    let mean: f32 = scores[..n].iter().sum::<f32>() / n as f32;
    let var: f32 =
        scores[..n].iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    if std < Z_SCORE_EPS {
        return z;
    }
    let denom = std.max(Z_SCORE_EPS);
    for i in 0..n {
        z[i] = ((scores[i] - mean) / denom).clamp(-1.0, 1.0);
    }
    z
}

/// Per-seat z-scored value for a terminal MCTS leaf, on the same scale
/// as the training value target.
///
/// - `Complete`: z-score `cumulative_scores[..n]` directly.
/// - `Scoring`: z-score `cumulative_scores[..n] + this_round_score[..n]`,
///   where `this_round_score[i] = (tricks_won[i] == bid[i]) ? 10 + bid[i]
///   : 0`. `advance_round` is what folds the round into
///   `cumulative_scores`; MCTS hits the `Scoring` boundary *before* that
///   call ([self_play.rs:209-211](../../../blob-nn/src/self_play.rs)),
///   so we pre-add the payout locally to keep the in-tree Q consistent
///   with what `backfill_values` would emit if the game ended now.
///
/// Slots `>= num_players` are zero. Non-terminal phases return all-zero
/// — callers are expected to gate this behind [`crate::mcts::is_terminal`].
pub fn terminal_z_scores(state: &BlobState) -> [f32; MAX_PLAYERS] {
    let n = state.num_players as usize;
    debug_assert!(n <= MAX_PLAYERS);

    let mut scores = [0.0f32; MAX_PLAYERS];
    match state.phase() {
        GamePhase::Complete => {
            for i in 0..n {
                scores[i] = state.cumulative_scores[i] as f32;
            }
        }
        GamePhase::Scoring => {
            for i in 0..n {
                let round_payout = if state.tricks_won[i] == state.bids[i] {
                    10 + state.bids[i] as u32
                } else {
                    0
                };
                scores[i] = state.cumulative_scores[i] as f32 + round_payout as f32;
            }
        }
        GamePhase::Bidding | GamePhase::Playing => return scores,
    }

    z_score_clip(&scores, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(n: usize, vals: &[u16]) -> [u16; MAX_PLAYERS] {
        let mut out = [0u16; MAX_PLAYERS];
        for (i, &v) in vals.iter().take(n).enumerate() {
            out[i] = v;
        }
        out
    }

    fn terminal_state(
        phase: GamePhase,
        num_players: u8,
        cumulative: [u16; MAX_PLAYERS],
        bids: [u8; MAX_PLAYERS],
        tricks_won: [u8; MAX_PLAYERS],
    ) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = num_players;
        s.cumulative_scores = cumulative;
        s.bids = bids;
        s.tricks_won = tricks_won;
        s.game_phase = phase as u8;
        s
    }

    #[test]
    fn z_score_clip_returns_zero_when_all_equal() {
        let mut s = [0.0f32; MAX_PLAYERS];
        for i in 0..4 {
            s[i] = 7.0;
        }
        let z = z_score_clip(&s, 4);
        for i in 0..4 {
            assert_eq!(z[i], 0.0, "seat {i}");
        }
    }

    #[test]
    fn z_score_clip_is_zero_mean_unit_variance_before_clip() {
        let mut s = [0.0f32; MAX_PLAYERS];
        // Spread: 10, 20, 30, 40 → mean 25, std sqrt(125) ≈ 11.18
        for (i, v) in [10.0, 20.0, 30.0, 40.0].iter().enumerate() {
            s[i] = *v;
        }
        let z = z_score_clip(&s, 4);
        let sum: f32 = z[..4].iter().sum();
        assert!(sum.abs() < 1e-4, "mean of z={sum}, expected ~0");
        // Symmetric pairs.
        assert!((z[0] + z[3]).abs() < 1e-5);
        assert!((z[1] + z[2]).abs() < 1e-5);
        for zi in &z[..4] {
            assert!(*zi >= -1.0 && *zi <= 1.0);
        }
    }

    /// Complete phase: scores already final; z-score directly.
    #[test]
    fn terminal_z_scores_complete_matches_z_score_clip() {
        let cum = scores(4, &[30, 0, 10, 50]);
        let s = terminal_state(
            GamePhase::Complete,
            4,
            cum,
            [0; MAX_PLAYERS],
            [0; MAX_PLAYERS],
        );
        let got = terminal_z_scores(&s);
        let mut expected_in = [0.0f32; MAX_PLAYERS];
        for i in 0..4 {
            expected_in[i] = cum[i] as f32;
        }
        let expected = z_score_clip(&expected_in, 4);
        for i in 0..MAX_PLAYERS {
            assert!((got[i] - expected[i]).abs() < 1e-6, "seat {i}");
        }
    }

    /// Scoring phase: must pre-add the just-finished round's payout
    /// before z-scoring (cumulative_scores hasn't absorbed it yet).
    #[test]
    fn terminal_z_scores_scoring_pre_adds_round_payout() {
        // 4-player Scoring snapshot.
        // - Seat 0 bid 3, won 3 → +13
        // - Seat 1 bid 2, won 1 → 0
        // - Seat 2 bid 0, won 0 → +10
        // - Seat 3 bid 4, won 4 → +14
        let cum = scores(4, &[20, 30, 20, 0]); // pre-round totals
        let bids_arr = {
            let mut b = [0u8; MAX_PLAYERS];
            b[0] = 3;
            b[1] = 2;
            b[2] = 0;
            b[3] = 4;
            b
        };
        let tw_arr = {
            let mut t = [0u8; MAX_PLAYERS];
            t[0] = 3;
            t[1] = 1;
            t[2] = 0;
            t[3] = 4;
            t
        };
        let s = terminal_state(GamePhase::Scoring, 4, cum, bids_arr, tw_arr);
        let got = terminal_z_scores(&s);

        // Reference: build the post-round totals by hand and z-score them.
        let mut post = [0.0f32; MAX_PLAYERS];
        post[0] = 20.0 + 13.0; // 33
        post[1] = 30.0 + 0.0; // 30
        post[2] = 20.0 + 10.0; // 30
        post[3] = 0.0 + 14.0; // 14
        let expected = z_score_clip(&post, 4);
        for i in 0..MAX_PLAYERS {
            assert!(
                (got[i] - expected[i]).abs() < 1e-6,
                "seat {i}: got {} expected {}",
                got[i],
                expected[i]
            );
        }
        // Slot beyond num_players stays zero.
        for i in 4..MAX_PLAYERS {
            assert_eq!(got[i], 0.0);
        }
    }

    #[test]
    fn terminal_z_scores_returns_zero_for_non_terminal_phase() {
        let s = terminal_state(
            GamePhase::Playing,
            4,
            scores(4, &[10, 20, 30, 40]),
            [0; MAX_PLAYERS],
            [0; MAX_PLAYERS],
        );
        let z = terminal_z_scores(&s);
        for v in z.iter() {
            assert_eq!(*v, 0.0);
        }
    }
}
