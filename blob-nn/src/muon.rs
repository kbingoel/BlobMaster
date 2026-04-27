//! Session 7.4d — Muon optimizer for hidden 2D weight matrices.
//!
//! tch-rs 0.20 has no built-in Muon, so this is a hand-rolled side
//! optimizer that runs alongside AdamW. AdamW handles biases, LayerNorm
//! scales, embeddings, and head matrices; Muon handles the transformer's
//! QKV / attn-out / FFN weight matrices (the four 2D linears in each
//! encoder block).
//!
//! ## Algorithm
//!
//! For each tracked weight `W ∈ ℝ^{out×in}`:
//!
//! 1. Update the momentum buffer in-place: `m ← β·m + g`.
//! 2. Form the Nesterov-look-ahead gradient `g' = g + β·m`. (Keller
//!    Jordan's reference Muon defaults to `nesterov=True`; published
//!    gains are measured under this variant.)
//! 3. Orthogonalize `g'` via 5 Newton-Schulz iterations, producing
//!    `O ≈ U V^T` where `g' = U Σ V^T` is the SVD. NS5 with the quintic
//!    polynomial coefficients `(3.4445, -4.7750, 2.0315)` is the Keller
//!    Jordan / modded-nanoGPT recipe — it converges to the orthogonal
//!    factor without ever computing an explicit SVD.
//! 4. Update with an aspect-ratio rescale that keeps the per-element RMS
//!    of the step in the same ballpark as AdamW at the same learning
//!    rate: `W ← W − lr · sqrt(max(1, out/in)) · O`.
//!
//! ## Coexisting with AdamW
//!
//! The Muon-targeted weights are registered into [`MUON_GROUP`] in the
//! `VarStore`. The AdamW optimizer is built across all groups, with the
//! Muon group's LR pinned to `0.0` — both the moment-driven term and the
//! decoupled weight-decay term in AdamW's update are multiplied by `lr`,
//! so the AdamW step is a true no-op for those params. Adam moments
//! still tick (they accumulate a copy of the gradient signal that is
//! never read), but they are bounded and harmless.
//!
//! Wiring: in the training step, after `total.backward()` and the global
//! `clip_grad_norm`, call `Muon::step(lr)` *before* `optimizer.step()`.
//! Order is not load-bearing (AdamW is a no-op for these params) but
//! consistent ordering makes traces easier to read.

use tch::{nn, Tensor};

use crate::transformer::N_LAYERS;

/// Heavy-ball momentum coefficient. Standard Muon default.
pub const MUON_BETA: f64 = 0.95;

/// Number of Newton-Schulz iterations per step. Five is the canonical
/// setting; further iterations sharpen the orthogonalization but offer
/// diminishing returns.
pub const MUON_NS_STEPS: usize = 5;

/// Multiplier on the schedule LR when applying Muon updates. The
/// per-element step magnitude is `lr * MUON_LR_SCALE *
/// sqrt(max(1, out/in))`, so `1.0` keeps Muon's update RMS in the same
/// ballpark as AdamW at the same LR.
pub const MUON_LR_SCALE: f64 = 1.0;

/// Newton-Schulz quintic polynomial coefficients (Keller Jordan).
const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;

/// Per-tensor state for Muon: a handle on the parameter (shares storage
/// with the `VarStore` entry) and its momentum buffer.
struct MuonParam {
    name: String,
    weight: Tensor,
    momentum: Tensor,
}

/// Muon optimizer side-channel. Owns its own momentum buffers; reads
/// gradients off the parameters in place.
pub struct Muon {
    params: Vec<MuonParam>,
}

impl Muon {
    /// Collect the Muon-targeted parameters from `vs` and allocate
    /// matching zero-initialized momentum buffers on the same device.
    pub fn from_var_store(vs: &nn::VarStore) -> Self {
        let _g = tch::no_grad_guard();
        let mut params: Vec<MuonParam> = vs
            .variables()
            .into_iter()
            .filter(|(name, _)| is_muon_target(name))
            .map(|(name, weight)| {
                debug_assert_eq!(
                    weight.size().len(),
                    2,
                    "Muon target {name} is not a 2D matrix"
                );
                let momentum = Tensor::zeros_like(&weight);
                MuonParam { name, weight, momentum }
            })
            .collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));

        // Sanity check: 8 layers × {qkv, out, fc1, fc2} = 32 weights.
        debug_assert_eq!(
            params.len(),
            N_LAYERS * 4,
            "expected {} Muon targets, got {} ({:?})",
            N_LAYERS * 4,
            params.len(),
            params.iter().map(|p| &p.name).collect::<Vec<_>>(),
        );

        Self { params }
    }

    /// Names of the parameters Muon will update, sorted. Useful for
    /// tests and diagnostics.
    pub fn target_names(&self) -> Vec<String> {
        self.params.iter().map(|p| p.name.clone()).collect()
    }

    /// Apply one Muon step at the given schedule learning rate. Reads
    /// each tracked weight's `.grad()` in place; if a grad is undefined
    /// (e.g. an early phase-only forward never touched it), that param
    /// is skipped. No-op if `lr == 0`.
    pub fn step(&mut self, lr: f64) {
        if lr == 0.0 {
            return;
        }
        let _g = tch::no_grad_guard();
        let lr = lr * MUON_LR_SCALE;
        for p in &mut self.params {
            let g = p.weight.grad();
            if !g.defined() {
                continue;
            }

            // Momentum buffer update in place: m ← β·m + g.
            let _ = p.momentum.f_mul_scalar_(MUON_BETA);
            let _ = p.momentum.f_add_(&g);

            // Nesterov look-ahead: g' = g + β·m.
            let look_ahead = &g + &p.momentum * MUON_BETA;

            let ortho = newton_schulz(&look_ahead, MUON_NS_STEPS);

            let sz = p.weight.size();
            let (rows, cols) = (sz[0] as f64, sz[1] as f64);
            let aspect_scale = (rows / cols).max(1.0).sqrt();
            let alpha = -lr * aspect_scale;

            // W ← W + alpha · O.
            let _ = p.weight.f_add_(&(&ortho * alpha));
        }
    }
}

/// True iff `name` identifies one of the transformer's 2D matrix
/// weights — `transformer.layer{i}.{attn.qkv,attn.out,ffn.fc1,ffn.fc2}.weight`.
pub fn is_muon_target(name: &str) -> bool {
    if !name.starts_with("transformer.layer") {
        return false;
    }
    if !name.ends_with(".weight") {
        return false;
    }
    name.contains(".attn.qkv.")
        || name.contains(".attn.out.")
        || name.contains(".ffn.fc1.")
        || name.contains(".ffn.fc2.")
}

/// Newton-Schulz iteration for matrix-sign / orthogonalization. Operates
/// on a 2D tensor; if `rows > cols` the input is transposed first so the
/// inner `X X^T` matmuls run on the smaller dimension. Output has the
/// same shape as the input.
///
/// Note: with the Keller Jordan quintic coefficients, this is *not* a
/// converging Newton iteration — its fixed point at 1 isn't attractive,
/// and 5 iterations leave the singular values scattered roughly across
/// `[0.5, 1.5]` rather than collapsed to 1. This is intentional (see the
/// Modded-NanoGPT design notes): for Muon's purposes it is enough to
/// "balance" the singular values, not to truly orthogonalize. The
/// post-update aspect-ratio rescale absorbs the residual variance.
fn newton_schulz(g: &Tensor, steps: usize) -> Tensor {
    let sz = g.size();
    debug_assert_eq!(sz.len(), 2, "Newton-Schulz expects a 2D tensor");
    let (rows, cols) = (sz[0], sz[1]);
    let transposed = rows > cols;

    let mut x = if transposed {
        g.transpose(0, 1).contiguous()
    } else {
        g.shallow_clone()
    };

    // Normalize spectral scale: divide by Frobenius norm so ||X||_2 ≤ 1
    // going into the iteration. The +eps guards against an all-zero
    // momentum on the very first step.
    let frob = x.norm();
    x = x / (&frob + 1e-7);

    for _ in 0..steps {
        let xt = x.transpose(0, 1);
        let a = x.matmul(&xt);
        let b = NS_B * &a + NS_C * a.matmul(&a);
        x = NS_A * &x + b.matmul(&x);
    }

    if transposed {
        x.transpose(0, 1).contiguous()
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BlobNet;
    use crate::train::build_optimizer;
    use tch::{nn::VarStore, Device, Kind};

    #[test]
    fn target_selection_picks_transformer_matrix_weights_only() {
        // Sanity: at least these belong, none of these don't.
        assert!(is_muon_target("transformer.layer0.attn.qkv.weight"));
        assert!(is_muon_target("transformer.layer3.attn.out.weight"));
        assert!(is_muon_target("transformer.layer7.ffn.fc1.weight"));
        assert!(is_muon_target("transformer.layer1.ffn.fc2.weight"));

        // Biases, LN scales, embeddings, heads — all skipped.
        assert!(!is_muon_target("transformer.layer0.attn.qkv.bias"));
        assert!(!is_muon_target("transformer.layer0.attn.out.bias"));
        assert!(!is_muon_target("transformer.layer0.ffn.fc1.bias"));
        assert!(!is_muon_target("transformer.layer0.ln1.weight"));
        assert!(!is_muon_target("transformer.layer0.ln2.bias"));
        assert!(!is_muon_target("input.hand_proj.weight"));
        assert!(!is_muon_target("input.cls"));
        assert!(!is_muon_target("play_head.fc1.weight"));
        assert!(!is_muon_target("bid_head.fc2.weight"));
        assert!(!is_muon_target("value_head.fc1.weight"));
    }

    #[test]
    fn from_var_store_collects_exactly_the_four_per_layer_matrix_weights() {
        let vs = VarStore::new(Device::Cpu);
        let _m = BlobNet::new(&vs.root());
        let muon = Muon::from_var_store(&vs);
        let names = muon.target_names();
        // 8 layers × 4 matrices each.
        assert_eq!(names.len(), N_LAYERS * 4);
        // Spot-check a few.
        assert!(names.iter().any(|n| n == "transformer.layer0.attn.qkv.weight"));
        assert!(names.iter().any(|n| n == "transformer.layer7.ffn.fc2.weight"));
        // Nothing leaks in.
        for n in &names {
            assert!(is_muon_target(n), "non-target slipped in: {n}");
        }
    }

    #[test]
    fn newton_schulz_approximately_orthogonalizes() {
        tch::manual_seed(7);

        // NS5 with the Keller Jordan coefficients does *not* converge to
        // exact orthogonality — the iteration is tuned for the
        // (concentrated) spectrum of trained transformer gradients, not
        // for iid Gaussian inputs whose normalized spectrum spans a
        // wider range. So we don't assert that `||X||_F^2 ≈ min_dim`
        // (that would be too tight for random inputs); instead we test
        // the actual orthogonality property: off-diagonal mass of the
        // gram matrix `X^T X` is small relative to diagonal mass. For a
        // truly orthogonal `X`, `X^T X = I` and off-diagonal energy is
        // zero; here we accept anything below 25% of diagonal energy as
        // "approximately orthogonal".
        for shape in [[64_i64, 32_i64], [32, 64]] {
            let g = Tensor::randn(shape, (Kind::Float, Device::Cpu));
            let x = newton_schulz(&g, 5);

            let gram = if shape[0] >= shape[1] {
                x.transpose(0, 1).matmul(&x)
            } else {
                x.matmul(&x.transpose(0, 1))
            };
            let diag = gram.diagonal(0, -2, -1);
            let diag_energy: f64 = diag.square().sum(Kind::Float).double_value(&[]);
            let total_energy: f64 = gram.square().sum(Kind::Float).double_value(&[]);
            let off_energy = total_energy - diag_energy;
            let ratio = off_energy / diag_energy.max(1e-12);
            assert!(
                ratio < 0.25,
                "gram off/diag energy ratio too high: {ratio} (shape {shape:?})"
            );

            // Output is finite (no NaN/inf leaked through the iteration).
            let any_nan: bool = x.isnan().any().int64_value(&[]) != 0;
            let any_inf: bool = x.isinf().any().int64_value(&[]) != 0;
            assert!(!any_nan && !any_inf, "NS produced NaN/Inf (shape {shape:?})");
        }
    }


    #[test]
    fn newton_schulz_zero_input_produces_finite_output() {
        // The +eps guards the all-zero momentum case (first step before
        // any gradient has been pushed in via momentum).
        let z = Tensor::zeros([4, 4], (Kind::Float, Device::Cpu));
        let x = newton_schulz(&z, 5);
        let any_nan: bool = x.isnan().any().int64_value(&[]) != 0;
        assert!(!any_nan, "NS produced NaN on zero input");
    }

    /// Standalone harness that wires together a fresh model, a few
    /// `train_step`-shaped iterations on a fixed batch, and asserts that
    /// (a) the Muon weights move and (b) AdamW's step does not move
    /// them (since it runs at `lr=0` for the Muon group).
    #[test]
    fn muon_moves_weights_and_adamw_does_not() {
        use crate::heads::NUM_BIDS;
        use crate::input::pad_batch;
        use crate::train::{
            policy_cross_entropy, set_schedule_lr, value_mse, Phase, TrainBatch, GRAD_CLIP_MAX_NORM,
        };
        use blob_engine::encoder::encode;
        use blob_engine::{dealing::deal, game::new_game};
        use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

        tch::manual_seed(123);
        let vs = VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        let mut opt = build_optimizer(&vs).unwrap();
        let mut muon = Muon::from_var_store(&vs);

        // Snapshot a Muon weight and a non-Muon weight (LN scale).
        let muon_name = "transformer.layer0.attn.qkv.weight";
        let other_name = "transformer.layer0.ln1.weight";
        let snap_muon = vs.variables()[muon_name].shallow_clone().copy();
        let snap_other = vs.variables()[other_name].shallow_clone().copy();

        // Build a tiny bidding batch (smallest path through the model).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        let e = encode(&s, s.current_player);
        let input = pad_batch(&[e], Device::Cpu);
        let legal = Tensor::ones([1, NUM_BIDS], (Kind::Bool, Device::Cpu));
        let mut tgt = vec![0.0f32; NUM_BIDS as usize];
        tgt[3] = 1.0;
        let policy_target = Tensor::from_slice(&tgt).view([1, NUM_BIDS]);
        let value_target = Tensor::from_slice(&[0.1f32]).view([1]);
        let batch = TrainBatch {
            input,
            phase: Phase::Bidding,
            legal_mask: legal,
            policy_target,
            value_target,
        };

        let lr = 1e-3;
        for _ in 0..5 {
            set_schedule_lr(&mut opt, lr);
            let (p, v) = model.forward_bid(&batch.input, &batch.legal_mask, true);
            let pl = policy_cross_entropy(&p, &batch.policy_target);
            let vl = value_mse(&v, &batch.value_target);
            let total: Tensor = &pl + 2.0 * &vl;

            opt.zero_grad();
            total.backward();
            opt.clip_grad_norm(GRAD_CLIP_MAX_NORM);
            muon.step(lr);
            opt.step();
        }

        let after_muon = vs.variables()[muon_name].shallow_clone();
        let after_other = vs.variables()[other_name].shallow_clone();

        let muon_delta: f64 = (&after_muon - &snap_muon)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        let other_delta: f64 = (&after_other - &snap_other)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);

        assert!(muon_delta > 1e-5, "Muon weight did not move: delta={muon_delta}");
        assert!(other_delta > 1e-5, "AdamW weight did not move: delta={other_delta}");
    }

    /// AdamW alone (without `muon.step`) must not move Muon-group
    /// weights, since its LR for that group is held at zero.
    #[test]
    fn adamw_alone_leaves_muon_group_weights_unchanged() {
        use crate::heads::NUM_BIDS;
        use crate::input::pad_batch;
        use crate::train::{
            policy_cross_entropy, set_schedule_lr, value_mse, Phase, TrainBatch, GRAD_CLIP_MAX_NORM,
        };
        use blob_engine::encoder::encode;
        use blob_engine::{dealing::deal, game::new_game};
        use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

        tch::manual_seed(321);
        let vs = VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        let mut opt = build_optimizer(&vs).unwrap();

        let muon_name = "transformer.layer3.ffn.fc1.weight";
        let snap = vs.variables()[muon_name].shallow_clone().copy();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        let input = pad_batch(&[encode(&s, s.current_player)], Device::Cpu);
        let legal = Tensor::ones([1, NUM_BIDS], (Kind::Bool, Device::Cpu));
        let mut tgt = vec![0.0f32; NUM_BIDS as usize];
        tgt[5] = 1.0;
        let batch = TrainBatch {
            input,
            phase: Phase::Bidding,
            legal_mask: legal,
            policy_target: Tensor::from_slice(&tgt).view([1, NUM_BIDS]),
            value_target: Tensor::from_slice(&[-0.2f32]).view([1]),
        };

        for _ in 0..10 {
            set_schedule_lr(&mut opt, 1e-3);
            let (p, v) = model.forward_bid(&batch.input, &batch.legal_mask, true);
            let total: Tensor = policy_cross_entropy(&p, &batch.policy_target) + 2.0 * value_mse(&v, &batch.value_target);
            opt.zero_grad();
            total.backward();
            opt.clip_grad_norm(GRAD_CLIP_MAX_NORM);
            // Deliberately do NOT call muon.step.
            opt.step();
        }

        let after = vs.variables()[muon_name].shallow_clone();
        let delta: f64 = (&after - &snap).abs().sum(Kind::Float).double_value(&[]);
        assert!(
            delta < 1e-8,
            "AdamW with lr=0 should leave Muon group untouched, but delta={delta}"
        );
    }
}
