//! Learner building blocks salvaged from the gen-1 training driver
//! (gen-2.md §6 Phase 2): replay batches → training tensors, the
//! validation split, and losses on held-out data.
//!
//! The learner itself (alternating P / V steps, LR keyed to learner steps,
//! metrics rows, checkpoints) is built on these in Phase 4.

use blob_engine::bidding::legal_bids;
use blob_engine::encoder::{encode, TOKEN_TYPE_HAND};
use blob_engine::playing::legal_plays;
use blob_engine::replay::{BidBatch, PlayBatch, ReplayBuffer};
use tch::{Device, Tensor};

use crate::heads::NUM_BIDS;
use crate::input::pad_batch;
use crate::model::BlobNet;
use crate::train::{policy_cross_entropy, value_mse, Phase, TrainBatch};

/// Whether round `round_id` is held out for validation.
///
/// Split by round, not by position: positions from one round share a label,
/// so a position-level split would leave near-copies of every held-out
/// label in training (gen-2.md §5.6). Deterministic, and a larger
/// `fraction` holds out a superset.
pub fn is_validation_round(round_id: u64, fraction: f64) -> bool {
    let mut x = round_id ^ 0x5A11_DA7E_0F0F_0F0F;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64 / (1u64 << 53) as f64) < fraction
}

/// Losses of the current weights (eval mode, no dropout) over a set of
/// examples. `NaN` where the set has no examples of that phase.
#[derive(Debug, Clone, Copy)]
pub struct HeldOutLosses {
    pub examples: usize,
    pub bid_policy_loss: f64,
    pub play_policy_loss: f64,
    pub value_loss: f64,
    /// Value MSE of always predicting 0, for scale (gen-2.md §2.2).
    pub value_loss_predict0: f64,
}

impl Default for HeldOutLosses {
    fn default() -> Self {
        Self {
            examples: 0,
            bid_policy_loss: f64::NAN,
            play_policy_loss: f64::NAN,
            value_loss: f64::NAN,
            value_loss_predict0: f64::NAN,
        }
    }
}

/// Convert a `BidBatch` from the replay buffer into a `TrainBatch` ready
/// for `train_step`.
pub fn bid_train_batch(batch: &BidBatch, device: Device) -> Option<TrainBatch> {
    let n = batch.states.len();
    if n == 0 {
        return None;
    }
    let encs: Vec<_> = batch.states.iter().map(|s| encode(s, s.current_player)).collect();
    let input = pad_batch(&encs, device);

    let mut mask = vec![false; n * NUM_BIDS as usize];
    for (row, state) in batch.states.iter().enumerate() {
        let legal = legal_bids(state);
        for b in 0..NUM_BIDS as usize {
            if (legal >> b) & 1 == 1 {
                mask[row * NUM_BIDS as usize + b] = true;
            }
        }
    }
    let legal_mask =
        Tensor::from_slice(&mask).view([n as i64, NUM_BIDS]).to_device(device);
    let policy_target = Tensor::from_slice(&batch.policies)
        .view([n as i64, NUM_BIDS])
        .to_device(device);
    let value_target = Tensor::from_slice(&batch.values)
        .view([n as i64])
        .to_device(device);

    Some(TrainBatch {
        input,
        phase: Phase::Bidding,
        legal_mask,
        policy_target,
        value_target,
    })
}

/// Convert a `PlayBatch` into a `TrainBatch`.
///
/// Play-head policies in the replay buffer are indexed by **hand position**
/// (0..hand_size). The play head outputs one score per **sequence
/// position**. This helper scatters each row's hand-indexed policy onto
/// the sequence positions whose `token_types[i] == TOKEN_TYPE_HAND`, in
/// encoder emission order (matches `EncodedState::hand_card_indices`).
pub fn play_train_batch(batch: &PlayBatch, device: Device) -> Option<TrainBatch> {
    let n = batch.states.len();
    if n == 0 {
        return None;
    }
    let encs: Vec<_> = batch.states.iter().map(|s| encode(s, s.current_player)).collect();
    let input = pad_batch(&encs, device);
    let seq_len = input.attention_mask.size()[1] as usize;

    let mut mask = vec![false; n * seq_len];
    let mut target = vec![0.0f32; n * seq_len];
    for (row, (state, enc)) in batch.states.iter().zip(encs.iter()).enumerate() {
        let legal = legal_plays(state);
        let mut hand_slot = 0usize;
        for (seq_i, &tt) in enc.token_types.iter().enumerate() {
            if tt != TOKEN_TYPE_HAND {
                continue;
            }
            let card_idx = enc.hand_card_indices[hand_slot];
            let pol_base = row * batch.max_hand_size;
            // `max_hand_size` in a PlayBatch is the largest *nonzero* hand
            // position observed — may be smaller than the encoder's actual
            // hand-token count. Treat out-of-range positions as zero prob.
            if hand_slot < batch.max_hand_size {
                target[row * seq_len + seq_i] = batch.policies[pol_base + hand_slot];
            }
            if (legal >> card_idx) & 1 == 1 {
                mask[row * seq_len + seq_i] = true;
            }
            hand_slot += 1;
        }
    }

    let legal_mask =
        Tensor::from_slice(&mask).view([n as i64, seq_len as i64]).to_device(device);
    let policy_target = Tensor::from_slice(&target)
        .view([n as i64, seq_len as i64])
        .to_device(device);
    let value_target = Tensor::from_slice(&batch.values)
        .view([n as i64])
        .to_device(device);

    Some(TrainBatch {
        input,
        phase: Phase::Playing,
        legal_mask,
        policy_target,
        value_target,
    })
}

/// Losses of `model` on `buf[indices]`, in eval mode (no dropout) and
/// without gradients, in `batch_size` chunks.
///
/// Compare a validation set with the same measurement on an equally large
/// training sample, never with losses logged during training: those are
/// averaged over many steps with dropout on (gen-2.md §5.6).
pub fn held_out_losses(
    model: &BlobNet,
    buf: &ReplayBuffer,
    indices: &[usize],
    batch_size: usize,
    device: Device,
) -> HeldOutLosses {
    let (mut bid_ce, mut play_ce, mut v_se, mut v_zero) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut n_bid, mut n_play) = (0usize, 0usize);
    let mut add = |tb: &TrainBatch, values: &[f32], is_bid: bool| {
        let n = values.len();
        let (probs, value_pred) = tch::no_grad(|| match tb.phase {
            Phase::Bidding => model.forward_bid(&tb.input, &tb.legal_mask, false),
            Phase::Playing => model.forward_play(&tb.input, &tb.legal_mask, false),
        });
        let ce = policy_cross_entropy(&probs, &tb.policy_target).double_value(&[]) * n as f64;
        v_se += value_mse(&value_pred, &tb.value_target).double_value(&[]) * n as f64;
        v_zero += values.iter().map(|&t| (t as f64).powi(2)).sum::<f64>();
        if is_bid {
            bid_ce += ce;
            n_bid += n;
        } else {
            play_ce += ce;
            n_play += n;
        }
    };
    for chunk in indices.chunks(batch_size.max(1)) {
        let (bid, play) = buf.batch_from_indices(chunk);
        if let Some(tb) = bid_train_batch(&bid, device) {
            add(&tb, &bid.values, true);
        }
        if let Some(tb) = play_train_batch(&play, device) {
            add(&tb, &play.values, false);
        }
    }
    let n = n_bid + n_play;
    let mean = |sum: f64, k: usize| if k > 0 { sum / k as f64 } else { f64::NAN };
    HeldOutLosses {
        examples: n,
        bid_policy_loss: mean(bid_ce, n_bid),
        play_policy_loss: mean(play_ce, n_play),
        value_loss: mean(v_se, n),
        value_loss_predict0: mean(v_zero, n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::bidding::apply_bid;
    use blob_engine::dealing::deal;
    use blob_engine::game::new_game;
    use blob_engine::replay::SparsePolicy;
    use blob_engine::state::{BlobState, GamePhase};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use smallvec::smallvec;
    use tch::nn;

    fn bidding_state(seed: u64) -> BlobState {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        s
    }

    /// First trick, first card: every card in the leader's hand is legal.
    fn playing_state(seed: u64) -> BlobState {
        let mut s = bidding_state(seed);
        while s.phase() == GamePhase::Bidding {
            let bid = legal_bids(&s).trailing_zeros() as u8;
            apply_bid(&mut s, bid);
        }
        s
    }

    fn half_half() -> SparsePolicy {
        smallvec![(0u8, 0.5f32), (1, 0.5)]
    }

    #[test]
    fn validation_split_is_deterministic_and_near_fraction() {
        let held: usize = (0..100_000u64).filter(|&r| is_validation_round(r, 0.03)).count();
        assert!((2_700..3_300).contains(&held), "held out {held} of 100k");
        for r in 0..1000u64 {
            assert_eq!(is_validation_round(r, 0.03), is_validation_round(r, 0.03));
            assert!(!is_validation_round(r, 0.0));
            assert!(is_validation_round(r, 1.0));
        }
        // A larger fraction holds out a superset.
        assert!((0..10_000u64).all(|r| !is_validation_round(r, 0.03) || is_validation_round(r, 0.1)));
    }

    #[test]
    fn bid_batch_masks_exactly_the_legal_bids() {
        let mut buf = ReplayBuffer::new(8);
        let states: Vec<BlobState> = (0..3).map(bidding_state).collect();
        for s in &states {
            buf.push(*s, half_half(), 0.25, GamePhase::Bidding);
        }
        let (bid, play) = buf.batch_from_indices(&[0, 1, 2]);
        assert!(play_train_batch(&play, Device::Cpu).is_none());
        let tb = bid_train_batch(&bid, Device::Cpu).unwrap();
        assert_eq!(tb.phase, Phase::Bidding);
        assert_eq!(tb.legal_mask.size(), vec![3, NUM_BIDS]);
        let mask: Vec<bool> = tb.legal_mask.flatten(0, -1).try_into().unwrap();
        for (row, s) in states.iter().enumerate() {
            let legal = legal_bids(s);
            for b in 0..NUM_BIDS as usize {
                assert_eq!(mask[row * NUM_BIDS as usize + b], (legal >> b) & 1 == 1);
            }
        }
        let target: Vec<f32> = tb.policy_target.flatten(0, -1).try_into().unwrap();
        assert_eq!(&target[..3], &[0.5, 0.5, 0.0]);
        let values: Vec<f32> = tb.value_target.try_into().unwrap();
        assert_eq!(values, vec![0.25; 3]);
    }

    /// Hand-position policies land on the matching hand-card tokens, in
    /// `hand_card_indices` order, and the mask covers exactly the legal ones.
    #[test]
    fn play_batch_scatters_policy_onto_hand_tokens() {
        let mut buf = ReplayBuffer::new(8);
        let states: Vec<BlobState> = (10..13).map(playing_state).collect();
        let policy: SparsePolicy = smallvec![(1u8, 0.25f32), (3, 0.75)];
        for s in &states {
            buf.push(*s, policy.clone(), -0.5, GamePhase::Playing);
        }
        let (_, play) = buf.batch_from_indices(&[0, 1, 2]);
        let tb = play_train_batch(&play, Device::Cpu).unwrap();
        assert_eq!(tb.phase, Phase::Playing);
        let seq_len = tb.legal_mask.size()[1] as usize;
        let mask: Vec<bool> = tb.legal_mask.flatten(0, -1).try_into().unwrap();
        let target: Vec<f32> = tb.policy_target.flatten(0, -1).try_into().unwrap();
        for (row, s) in states.iter().enumerate() {
            let enc = encode(s, s.current_player);
            let legal = legal_plays(s);
            let hand_positions: Vec<usize> = (0..enc.num_tokens)
                .filter(|&i| enc.token_types[i] == TOKEN_TYPE_HAND)
                .collect();
            assert_eq!(hand_positions.len(), enc.hand_card_indices.len());
            for i in 0..seq_len {
                let slot = hand_positions.iter().position(|&p| p == i);
                let want_target = match slot {
                    Some(1) => 0.25,
                    Some(3) => 0.75,
                    _ => 0.0,
                };
                assert_eq!(target[row * seq_len + i], want_target, "row {row} pos {i}");
                let want_mask =
                    slot.is_some_and(|h| (legal >> enc.hand_card_indices[h]) & 1 == 1);
                assert_eq!(mask[row * seq_len + i], want_mask, "row {row} pos {i}");
            }
        }
    }

    #[test]
    fn held_out_losses_cover_every_example() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        let mut buf = ReplayBuffer::new(16);
        for i in 0..5u64 {
            buf.push(playing_state(i), half_half(), 0.5, GamePhase::Playing);
        }
        for i in 0..4u64 {
            buf.push(bidding_state(100 + i), half_half(), -0.5, GamePhase::Bidding);
        }
        let idx: Vec<usize> = (0..buf.len()).collect();
        let l = held_out_losses(&model, &buf, &idx, 3, Device::Cpu);
        assert_eq!(l.examples, 9);
        assert!(l.bid_policy_loss.is_finite() && l.play_policy_loss.is_finite());
        assert!(l.value_loss.is_finite());
        assert!((l.value_loss_predict0 - 0.25).abs() < 1e-9);

        // Chunking doesn't change the result.
        let whole = held_out_losses(&model, &buf, &idx, 64, Device::Cpu);
        assert!((whole.value_loss - l.value_loss).abs() < 1e-5);
        assert!((whole.play_policy_loss - l.play_policy_loss).abs() < 1e-5);

        // An empty set reports NaN, not 0.
        let empty = held_out_losses(&model, &buf, &[], 3, Device::Cpu);
        assert_eq!(empty.examples, 0);
        assert!(empty.value_loss.is_nan());
    }
}
