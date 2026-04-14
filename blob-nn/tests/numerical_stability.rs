//! Session 6.3 — numerical-stability gate.
//!
//! 50 training iterations on randomly-generated batches. Asserts:
//! - every loss component is finite (no NaN/Inf) every step;
//! - every predicted value sits in `[-1, 1]` (ValueHead tanh contract);
//! - model parameters stay finite after the run.
//!
//! Slow (~15 s on CPU). Ignored by default; run with:
//! `cargo test -p blob-nn --release -- --ignored numerical_stability`

use blob_engine::encoder::{encode, TOKEN_TYPE_HAND};
use blob_engine::{dealing::deal, game::new_game};
use blob_nn::heads::NUM_BIDS;
use blob_nn::input::pad_batch;
use blob_nn::model::BlobNet;
use blob_nn::train::{
    build_optimizer, policy_cross_entropy, train_step, value_mse, Phase, TrainBatch,
};
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn random_play_batch(seed: u64, batch_size: usize) -> TrainBatch {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut encs = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let mut s = new_game(5, 7).unwrap();
        deal(&mut s, &mut rng);
        encs.push(encode(&s, s.current_player));
    }
    let input = pad_batch(&encs, Device::Cpu);
    let hand_mask = input.token_types.eq(TOKEN_TYPE_HAND as i64);
    let counts = hand_mask
        .to_kind(Kind::Float)
        .sum_dim_intlist(&[-1i64][..], true, Kind::Float);
    let target = hand_mask.to_kind(Kind::Float) / counts.clamp_min(1.0);
    let b = batch_size as i64;
    let vt: Vec<f32> = (0..b)
        .map(|i| ((i as f32 * 0.173 + seed as f32 * 0.01) % 2.0 - 1.0).clamp(-1.0, 1.0))
        .collect();
    let value_target = Tensor::from_slice(&vt).view([b]);
    TrainBatch {
        input,
        phase: Phase::Playing,
        legal_mask: hand_mask,
        policy_target: target,
        value_target,
    }
}

fn random_bid_batch(seed: u64, batch_size: usize) -> TrainBatch {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0xBEEF);
    let mut encs = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let mut s = new_game(5, 7).unwrap();
        deal(&mut s, &mut rng);
        encs.push(encode(&s, s.current_player));
    }
    let input = pad_batch(&encs, Device::Cpu);
    let b = batch_size as i64;
    let legal = Tensor::ones([b, NUM_BIDS], (Kind::Bool, Device::Cpu));
    let mut tgt = vec![0.0f32; (b * NUM_BIDS) as usize];
    for i in 0..b {
        let col = ((seed.wrapping_add(i as u64)) % (NUM_BIDS as u64)) as i64;
        tgt[(i * NUM_BIDS + col) as usize] = 1.0;
    }
    let policy_target = Tensor::from_slice(&tgt).view([b, NUM_BIDS]);
    let vt: Vec<f32> = (0..b).map(|i| (i as f32 * 0.21 - 0.4).clamp(-1.0, 1.0)).collect();
    let value_target = Tensor::from_slice(&vt).view([b]);
    TrainBatch {
        input,
        phase: Phase::Bidding,
        legal_mask: legal,
        policy_target,
        value_target,
    }
}

#[test]
#[ignore]
fn fifty_iterations_no_nan_or_inf() {
    tch::manual_seed(0xC0DE);
    let vs = VarStore::new(Device::Cpu);
    let model = BlobNet::new(&vs.root());
    let mut opt = build_optimizer(&vs).unwrap();
    opt.set_lr(3e-4);

    for i in 0..50u64 {
        let batch = if i % 2 == 0 {
            random_play_batch(i, 4)
        } else {
            random_bid_batch(i, 4)
        };

        let losses = train_step(&model, &mut opt, &batch);
        assert!(
            losses.policy.is_finite() && losses.value.is_finite() && losses.total.is_finite(),
            "non-finite loss at iter {i}: {losses:?}"
        );

        // Re-run forward (no_grad) to inspect predictions.
        let (policy, value) = match batch.phase {
            Phase::Bidding => model.forward_bid(&batch.input, &batch.legal_mask, false),
            Phase::Playing => model.forward_play(&batch.input, &batch.legal_mask, false),
        };

        // ValueHead is tanh → must stay in [-1, 1].
        let v_min = value.min().double_value(&[]);
        let v_max = value.max().double_value(&[]);
        assert!(v_min >= -1.0 - 1e-5 && v_max <= 1.0 + 1e-5,
            "value out of range at iter {i}: [{v_min}, {v_max}]");
        assert!(v_min.is_finite() && v_max.is_finite(), "value non-finite at iter {i}");

        let p_sum = policy.sum(Kind::Float).double_value(&[]);
        assert!(p_sum.is_finite(), "policy non-finite at iter {i}");

        let (pmin, pmax) = (
            policy.min().double_value(&[]),
            policy.max().double_value(&[]),
        );
        assert!(pmin >= -1e-5 && pmax <= 1.0 + 1e-5,
            "policy prob out of [0,1] at iter {i}: [{pmin}, {pmax}]");

        // Compute loss via public helpers and reassert finiteness — guards
        // against regression where `train_step` might hide a non-finite
        // forward behind clipping.
        let pl = policy_cross_entropy(&policy, &batch.policy_target).double_value(&[]);
        let vl = value_mse(&value, &batch.value_target).double_value(&[]);
        assert!(pl.is_finite() && vl.is_finite(), "recomputed loss NaN at iter {i}");
    }

    // Final parameter sanity: no NaN/Inf anywhere in the model.
    for (name, tensor) in vs.variables() {
        let finite = tensor.isfinite().all().int64_value(&[]);
        assert_eq!(finite, 1, "parameter {name} contains NaN/Inf after 50 iters");
    }
}
