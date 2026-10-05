//! blob-nn — the network and its training code (Linux / libtorch).
//!
//! The only crate that depends on `tch`. The entity encoder lives in
//! `blob-engine::encoder`, so search and inference never need libtorch.

pub mod heads;
pub mod input;
pub mod learner;
pub mod model;
pub mod train;
pub mod transformer;
