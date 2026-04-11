//! blob-engine — card, hand, and state primitives for BlobMaster.
//!
//! Session 1.1 covers the low-level card representation (`Card`, `Suit`),
//! `u64` bitmask hand ops (`Hand`), and the `BlobState` stack struct along
//! with its supporting types (`GamePhase`, `TrickRecord`). Gameplay logic
//! (dealing, bidding, trick-taking, scoring) arrives in Sessions 1.2–1.4.

pub mod card;
pub mod hand;
pub mod state;

pub use card::{Card, Suit, MAX_CARDS_DEALT, NUM_CARDS, NUM_RANKS, NUM_SUITS};
pub use hand::Hand;
pub use state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS};
