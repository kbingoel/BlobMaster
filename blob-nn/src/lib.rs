//! blob-nn — neural network training crate (Linux / libtorch).
//!
//! Populated in Section 3 of `development-plan.md`. Session 1.1 leaves this
//! crate as an empty stub so the workspace compiles.
//!
//! Session 2.1 adds the entity encoder's hand-card token encoder
//! (`encoder` module): raw 30-dim feature vectors per card in hand.

pub mod encoder;
