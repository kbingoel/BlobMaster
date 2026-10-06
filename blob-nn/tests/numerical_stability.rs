//! Numerical-stability gate.
//!
//! 50 learner steps on rule-bot-2 rounds at a high learning rate. Asserts:
//! - both training losses are finite every step;
//! - P's probabilities and V's values stay in `[0, 1]`;
//! - every parameter is finite after the run.
//!
//! Slow (~15 s on CPU). Ignored by default; run with:
//! `cargo test -p blob-nn --release -- --ignored numerical_stability`

use blob_engine::{fill_buffer, ReplayBuffer, RoundMix, TeacherConfig};
use blob_nn::learner::{bid_policy_batch, is_forced, value_batch, Learner, LearnerConfig};
use blob_nn::train::policy_probs;
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
use tch::Device;

#[test]
#[ignore]
fn numerical_stability() {
    tch::manual_seed(0xC0DE);
    let teacher = TeacherConfig { mix: RoundMix { players: vec![5], start_cards: 7, large_round_exponent: 0.0 }, ..Default::default() };
    let mut buf = ReplayBuffer::new(20_000);
    fill_buffer(&mut buf, &teacher, 300, 3, 8);
    let all: Vec<usize> = (0..buf.len()).collect();
    let unforced: Vec<usize> = all.iter().copied().filter(|&i| !is_forced(buf.state(i))).collect();

    let cfg = LearnerConfig { device: "cpu".into(), batch_size: 32, steps: 50, warmup_steps: 1, peak_lr: 3e-3, ..Default::default() };
    let mut learner = Learner::new(&cfg).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    for i in 0..cfg.steps {
        let p = buf.sample_batch_from(&unforced, cfg.batch_size, &mut rng, true);
        let v = buf.sample_batch_from(&all, cfg.batch_size, &mut rng, true);
        let (pl, vl) = learner.train_step(&p, &v);
        let (pl, vl) = (pl.unwrap().double_value(&[]), vl.unwrap().double_value(&[]));
        assert!(pl.is_finite() && vl.is_finite(), "non-finite loss at step {i}: {pl} {vl}");

        if let Some(pb) = bid_policy_batch(&p.0, Device::Cpu) {
            let probs = tch::no_grad(|| policy_probs(&learner.policy, &pb, false));
            let (lo, hi) = (probs.min().double_value(&[]), probs.max().double_value(&[]));
            assert!(lo >= -1e-6 && hi <= 1.0 + 1e-6, "bid probs out of [0,1] at step {i}: [{lo}, {hi}]");
        }
        let vb = value_batch(&v.0, &v.1, Device::Cpu).unwrap();
        let values = tch::no_grad(|| learner.value.forward(&vb.input, false));
        let (lo, hi) = (values.min().double_value(&[]), values.max().double_value(&[]));
        assert!(lo >= 0.0 && hi <= 1.0, "values out of [0,1] at step {i}: [{lo}, {hi}]");
    }

    let dir = std::env::temp_dir().join(format!("blob-stability-{}", std::process::id()));
    learner.save(&dir).unwrap();
    let mut check = Learner::new(&cfg).unwrap();
    check.resume(&dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let v = check.value_held_out(&buf, &all[..2000], 256);
    assert!(v.mse.is_finite() && v.correlation.is_finite(), "{v:?}");
    let p = check.policy_held_out(&buf, &unforced[..2000], 256);
    assert!(p.bid_loss.is_finite() && p.play_loss.is_finite(), "{p:?}");
}
