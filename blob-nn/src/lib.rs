//! blob-nn — neural network training crate (Linux / libtorch).
//!
//! Populated in Section 3 of `development-plan.md`. Session 1.1 leaves this
//! crate as an empty stub so the workspace compiles.
//!
//! The entity encoder lives in `blob-engine::encoder` (moved there from
//! blob-nn after Section 2 so that MCTS in blob-engine can encode states
//! without creating a circular crate dependency).
//!
//! Re-exported here for backwards compatibility during the transition.

pub use blob_engine::encoder;

pub mod input;
pub mod transformer;
pub mod heads;
pub mod model;
pub mod train;
