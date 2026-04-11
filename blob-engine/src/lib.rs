//! blob-engine — card, hand, and state primitives for BlobMaster.
//!
//! Session 1.1 covers the low-level card representation (`Card`, `Suit`),
//! `u64` bitmask hand ops (`Hand`), and the `BlobState` stack struct along
//! with its supporting types (`GamePhase`, `TrickRecord`).
//!
//! Session 1.2 adds dealing, the bidding phase, and trump rotation
//! (`round`, `dealing`, `bidding` modules). Trick-taking and scoring land
//! in Sessions 1.3–1.4.

pub mod bidding;
pub mod card;
pub mod dealing;
pub mod hand;
pub mod round;
pub mod state;

pub use bidding::{apply_bid, forbidden_bid, legal_bids};
pub use card::{Card, Suit, MAX_CARDS_DEALT, NUM_CARDS, NUM_RANKS, NUM_SUITS};
pub use dealing::{deal, start_round};
pub use hand::Hand;
pub use round::{
    cards_dealt_for_round, round_structure, total_rounds, trump_for_round, validate_round_params,
    RoundParamsError, NO_TRUMP, TRUMP_CYCLE_LEN,
};
pub use state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS, MIN_PLAYERS};
