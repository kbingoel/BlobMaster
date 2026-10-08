//! blob-nn — the networks and their training code (Linux / libtorch).
//!
//! - `model`: the policy net P and the value net V (gen-2.md §5.3), built
//!   from `input`, `transformer` and `heads`.
//! - `train`: losses, AdamW, the LR schedule, checkpoints.
//! - `learner`: replay batches → tensors, the alternating P / V learner,
//!   held-out measurements.
//!
//! The only crate that depends on `tch`. The entity encoder lives in
//! `blob-engine::encoder`, so search and inference never need libtorch.

pub mod heads;
pub mod input;
pub mod learner;
pub mod model;
pub mod train;
pub mod transformer;

/// libtorch's RNG is global: tests that build networks from a seed take
/// this lock, so another test's networks can't shift their init.
#[cfg(test)]
pub(crate) static TORCH_RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());
