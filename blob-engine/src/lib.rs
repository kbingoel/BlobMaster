//! blob-engine — card, hand, and state primitives for BlobMaster.
//!
//! Session 1.1 covers the low-level card representation (`Card`, `Suit`),
//! `u64` bitmask hand ops (`Hand`), and the `BlobState` stack struct along
//! with its supporting types (`GamePhase`, `TrickRecord`).
//!
//! Session 1.2 adds dealing, the bidding phase, and trump rotation
//! (`round`, `dealing`, `bidding` modules).
//!
//! Session 1.3 adds trick-taking and per-round scoring (`playing` module):
//! `legal_plays`, `apply_play`, and `score_round`.
//!
//! Session 1.4 wires the per-phase primitives into the full multi-round game
//! loop (`game` module): `new_game`, `advance_round`, and `is_game_over`.
//!
//! Session 2.1–2.3 adds the entity encoder (`encoder` module): variable-length
//! token sequences for neural network input, with hand card, played card,
//! player state, and context tokens.

pub mod belief;
pub mod bidding;
pub mod card;
pub mod dealing;
pub mod encoder;
pub mod evaluator;
pub mod game;
pub mod hand;
pub mod mcts;
pub mod onnx;
pub mod playing;
pub mod round;
pub mod state;

pub use bidding::{apply_bid, forbidden_bid, legal_bids};
pub use evaluator::{DummyEvaluator, Evaluator, NUM_BIDS};
pub use onnx::OnnxEvaluator;
pub use card::{Card, Suit, MAX_CARDS_DEALT, NUM_CARDS, NUM_RANKS, NUM_SUITS};
pub use dealing::{deal, start_round};
pub use game::{advance_round, is_game_over, new_game};
pub use hand::Hand;
pub use belief::{determinize, void_suits, VoidTable, DEFAULT_DETERMINIZE_ATTEMPTS};
pub use mcts::{
    adaptive_budget, apply_action, backprop, expand, is_terminal, mcts_search, root_action_probs,
    run_search, select_best_child, select_leaf, signal_ratio, ucb1_score, MctsArena, MctsConfig,
    MctsNode, MctsResult, DEFAULT_ARENA_CAPACITY, DEFAULT_C_PUCT,
};
pub use playing::{apply_play, legal_plays, score_round};
pub use round::{
    cards_dealt_for_round, round_structure, total_rounds, trump_for_round, validate_round_params,
    RoundParamsError, NO_TRUMP, TRUMP_CYCLE_LEN,
};
pub use state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS, MIN_PLAYERS};
