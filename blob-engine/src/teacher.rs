//! Teacher rounds for the supervised warm start (gen-2.md §6 Phase 4).
//!
//! Rule bot 2 is the teacher. Single rounds are played and stored in the
//! replay format: P learns to imitate rule bot 2's policy at every
//! decision, and V learns the points every seat actually scored.
//!
//! - **Seats.** Each seat of a round is rule bot 2, or the rule bot with
//!   chance `rule_bot_share` (drawn per seat and round), so V also sees
//!   tables that don't all play alike.
//! - **Exploration.** With chance `explore`, an unforced move is drawn from
//!   the teacher policy instead of being the seat's own choice. The data then
//!   also covers positions that rule bot 2 would not steer into.
//! - **Targets.** Every decision is labelled with rule bot 2's policy for
//!   its state, whoever sits in the seat ([`teacher_policy`]). Forced moves
//!   are stored too: V trains on every state, P's learner skips them.
//! - **Deterministic.** Round `i` is seeded from `(seed, i)`, so a buffer
//!   doesn't depend on the thread count ([`fill_buffer`]).

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::bidding::legal_bids;
use crate::dealing::{new_round, RoundParams};
use crate::encoder::hand_card_indices;
use crate::mcts::apply_action;
use crate::replay::{Decision, ReplayBuffer, SparsePolicy};
use crate::round::RoundMix;
use crate::rule_bot::rule_bot_action;
use crate::rule_bot_2::{bid_chances, play_chances, rule_bot_2_action};
use crate::state::{BlobState, GamePhase};

/// How teacher rounds are played and labelled. Unknown keys are an error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TeacherConfig {
    /// The rounds to play.
    pub mix: RoundMix,
    /// Chance that a seat is played by the rule bot instead of rule bot 2.
    pub rule_bot_share: f32,
    /// Chance that an unforced move is drawn from the teacher policy.
    pub explore: f32,
    /// Softmax temperature over each move's expected points, in points.
    pub temperature: f32,
    /// Weight of rule bot 2's own move in the target; the softmax gets the
    /// rest. At 0.5 or more the target's top move is always rule bot 2's;
    /// below, a near-tie it breaks toward the weaker move can flip it.
    pub argmax_weight: f32,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            mix: RoundMix::default(),
            rule_bot_share: 0.2,
            explore: 0.1,
            temperature: 1.0,
            argmax_weight: 0.5,
        }
    }
}

impl TeacherConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.mix.validate().map_err(|e| format!("teacher.mix: {e:?}"))?;
        let unit = |name: &str, x: f32| {
            if (0.0..=1.0).contains(&x) {
                Ok(())
            } else {
                Err(format!("teacher.{name} must be in [0, 1], got {x}"))
            }
        };
        unit("rule_bot_share", self.rule_bot_share)?;
        unit("explore", self.explore)?;
        unit("argmax_weight", self.argmax_weight)?;
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            return Err(format!("teacher.temperature must be > 0, got {}", self.temperature));
        }
        Ok(())
    }
}

/// Rule bot 2's policy for the seat to move, as the replay buffer stores
/// it: bids by value, plays by hand position.
///
/// `argmax_weight` goes to rule bot 2's own move ([`rule_bot_2_action`]);
/// the rest is a softmax at `temperature` over each legal move's expected
/// points: `(10 + b) · P(make b)` for a bid, `(10 + bid) · P(make)` after a
/// card ([`bid_chances`], [`play_chances`]). Every legal move is listed.
pub fn teacher_policy(state: &BlobState, cfg: &TeacherConfig) -> SparsePolicy {
    teacher_move(state, cfg).0
}

/// [`teacher_policy`] and rule bot 2's move as a policy label.
fn teacher_move(state: &BlobState, cfg: &TeacherConfig) -> (SparsePolicy, u8) {
    let me = state.current_player as usize;
    let best = rule_bot_2_action(state);
    let (mut policy, best_label): (SparsePolicy, u8) = match state.phase() {
        GamePhase::Bidding => {
            let legal = legal_bids(state);
            let ev = bid_chances(state)
                .iter()
                .enumerate()
                .filter(|&(b, _)| (legal >> b) & 1 == 1)
                .map(|(b, &p)| (b as u8, (10 + b) as f32 * p))
                .collect();
            (ev, best)
        }
        GamePhase::Playing => {
            let hand = hand_card_indices(state, state.current_player);
            let pos = |card: u8| hand.iter().position(|&c| c == card).expect("card in hand") as u8;
            let worth = (10 + state.bids[me]) as f32;
            let ev = play_chances(state).iter().map(|&(card, p)| (pos(card), worth * p)).collect();
            (ev, pos(best))
        }
        phase => panic!("teacher asked to act in {phase:?}"),
    };
    let max = policy.iter().map(|&(_, ev)| ev).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, x) in policy.iter_mut() {
        *x = ((*x - max) / cfg.temperature).exp();
        sum += *x;
    }
    for (action, x) in policy.iter_mut() {
        *x *= (1.0 - cfg.argmax_weight) / sum;
        if *action == best_label {
            *x += cfg.argmax_weight;
        }
    }
    (policy, best_label)
}

/// Draw a label from `policy`.
fn sample_label<R: Rng + ?Sized>(policy: &SparsePolicy, rng: &mut R) -> u8 {
    let mut x = rng.gen::<f32>() * policy.iter().map(|&(_, p)| p).sum::<f32>();
    for &(label, p) in policy {
        if x < p {
            return label;
        }
        x -= p;
    }
    policy[policy.len() - 1].0
}

/// Play one round with `params` and return its decisions, labelled with
/// [`teacher_policy`], and the finished state.
pub fn teacher_round<R: Rng + ?Sized>(
    params: RoundParams,
    cfg: &TeacherConfig,
    rng: &mut R,
) -> (Vec<Decision>, BlobState) {
    let mut s = new_round(params, rng).expect("valid round parameters");
    let rule_bot_seats: Vec<bool> =
        (0..params.num_players).map(|_| rng.gen::<f32>() < cfg.rule_bot_share).collect();
    let mut decisions = Vec::new();
    while matches!(s.phase(), GamePhase::Bidding | GamePhase::Playing) {
        let (policy, best) = teacher_move(&s, cfg);
        let label = if policy.len() > 1 && rng.gen::<f32>() < cfg.explore {
            Some(sample_label(&policy, rng))
        } else if rule_bot_seats[s.current_player as usize] {
            None
        } else {
            Some(best)
        };
        // A label is a bid, or a hand position; `apply_action` takes a card.
        let action = match (label, s.phase()) {
            (None, _) => rule_bot_action(&s),
            (Some(b), GamePhase::Bidding) => b,
            (Some(pos), _) => hand_card_indices(&s, s.current_player)[pos as usize],
        };
        decisions.push(Decision { state: s, policy });
        apply_action(&mut s, action);
    }
    (decisions, s)
}

fn round_seed(seed: u64, round: u64) -> u64 {
    let mut x = seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Rounds generated in parallel before they are pushed in order.
const WAVE: u64 = 1 << 16;

/// Play `rounds` teacher rounds on `threads` threads and push them into
/// `buf` in round order. Round `i` draws its parameters from `cfg.mix` and
/// plays with an RNG seeded from `(seed, i)`, so the buffer depends only on
/// `cfg`, `rounds` and `seed`.
pub fn fill_buffer(buf: &mut ReplayBuffer, cfg: &TeacherConfig, rounds: u64, seed: u64, threads: usize) {
    let threads = threads.max(1) as u64;
    let mut start = 0u64;
    while start < rounds {
        let end = (start + WAVE).min(rounds);
        let per = (end - start).div_ceil(threads);
        let parts: Vec<Vec<(Vec<Decision>, BlobState)>> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = ((start + t * per).min(end), (start + (t + 1) * per).min(end));
                    sc.spawn(move || {
                        (lo..hi)
                            .map(|i| {
                                let mut rng = Xoshiro256PlusPlus::seed_from_u64(round_seed(seed, i));
                                let params = cfg.mix.sample(&mut rng);
                                teacher_round(params, cfg, &mut rng)
                            })
                            .collect()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("teacher thread panicked")).collect()
        });
        for (decisions, end_state) in parts.iter().flatten() {
            buf.push_round(decisions, end_state);
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::apply_bid;
    use crate::playing::{apply_play, legal_plays};
    use crate::rule_bot_2::rule_bot_2_play;

    fn rng(seed: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(seed)
    }

    fn params(cards: u8) -> RoundParams {
        RoundParams { num_players: 5, cards_dealt: cards, trump: 1, dealer: 2 }
    }

    #[test]
    fn default_config_is_valid_and_rejects_unknown_keys() {
        assert!(TeacherConfig::default().validate().is_ok());
        let parsed: TeacherConfig = toml::from_str("explore = 0.0\n[mix]\nplayers = [4]\nstart_cards = 8\n").unwrap();
        assert_eq!((parsed.explore, parsed.mix.players.clone()), (0.0, vec![4]));
        assert_eq!(parsed.temperature, TeacherConfig::default().temperature);
        assert!(toml::from_str::<TeacherConfig>("explor = 0.1\n").is_err());
        let bad = TeacherConfig { temperature: 0.0, ..Default::default() };
        assert!(bad.validate().is_err());
    }

    /// Each target is a distribution over exactly the legal moves, and its
    /// top move is rule bot 2's.
    #[test]
    fn policy_covers_the_legal_moves_and_peaks_at_rule_bot_2s_move() {
        let cfg = TeacherConfig { explore: 0.0, rule_bot_share: 0.0, ..Default::default() };
        let mut r = rng(1);
        let mut checked = [0usize; 2];
        for round in 0..40u64 {
            let (decisions, _) = teacher_round(params(1 + (round % 7) as u8), &cfg, &mut r);
            for d in &decisions {
                let s = &d.state;
                let sum: f32 = d.policy.iter().map(|&(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
                let top = d.policy.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
                match s.phase() {
                    GamePhase::Bidding => {
                        let mut labels: Vec<u8> = d.policy.iter().map(|&(b, _)| b).collect();
                        labels.sort();
                        let legal: Vec<u8> = (0..14).filter(|b| (legal_bids(s) >> b) & 1 == 1).collect();
                        assert_eq!(labels, legal);
                        assert_eq!(top, rule_bot_2_action(s));
                        checked[0] += 1;
                    }
                    _ => {
                        let hand = hand_card_indices(s, s.current_player);
                        let mut cards: Vec<u8> = d.policy.iter().map(|&(pos, _)| hand[pos as usize]).collect();
                        cards.sort();
                        let legal: Vec<u8> = (0..52).filter(|c| (legal_plays(s) >> c) & 1 == 1).collect();
                        assert_eq!(cards, legal);
                        assert_eq!(hand[top as usize], rule_bot_2_play(s));
                        checked[1] += 1;
                    }
                }
            }
        }
        assert!(checked[0] > 100 && checked[1] > 300, "{checked:?}");
    }

    /// Without exploration or rule-bot seats, the round is rule bot 2
    /// playing itself, and every state is recorded.
    #[test]
    fn plain_round_is_rule_bot_2_self_play() {
        let cfg = TeacherConfig { explore: 0.0, rule_bot_share: 0.0, ..Default::default() };
        let (decisions, end) = teacher_round(params(6), &cfg, &mut rng(7));
        assert_eq!(decisions.len(), 5 + 5 * 6);
        let mut s = decisions[0].state;
        for d in &decisions {
            assert_eq!(d.state, s);
            let action = rule_bot_2_action(&s);
            match s.phase() {
                GamePhase::Bidding => apply_bid(&mut s, action),
                _ => apply_play(&mut s, action),
            }
        }
        assert_eq!(s, end);
        assert_eq!(end.phase(), GamePhase::Scoring);
    }

    #[test]
    fn exploration_changes_moves() {
        let quiet = TeacherConfig { explore: 0.0, rule_bot_share: 0.0, ..Default::default() };
        let loud = TeacherConfig { explore: 1.0, ..quiet.clone() };
        let differs = (0..20u64).filter(|&seed| {
            let a = teacher_round(params(7), &quiet, &mut rng(seed)).1;
            let b = teacher_round(params(7), &loud, &mut rng(seed)).1;
            (a.bids, a.tricks_won) != (b.bids, b.tricks_won)
        });
        assert!(differs.count() >= 10);
    }

    #[test]
    fn buffer_is_the_same_on_any_thread_count() {
        let cfg = TeacherConfig::default();
        let fill = |threads| {
            let mut buf = ReplayBuffer::new(100_000);
            fill_buffer(&mut buf, &cfg, 300, 42, threads);
            buf
        };
        let (one, many) = (fill(1), fill(7));
        assert_eq!(one.rounds_pushed(), 300);
        assert_eq!(one.len(), many.len());
        for i in (0..one.len()).step_by(37) {
            assert_eq!(one.state(i), many.state(i));
            assert_eq!(one.round_id(i), many.round_id(i));
        }
        // Round 0 replays from its seed alone.
        let mut r = rng(round_seed(42, 0));
        let (decisions, _) = teacher_round(cfg.mix.sample(&mut r), &cfg, &mut r);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!((d.state, one.round_id(i)), (*one.state(i), 0));
        }
        assert_eq!(one.round_id(decisions.len()), 1);
    }
}
