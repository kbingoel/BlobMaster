//! blob-engine — game rules, encoder, search and inference for BlobMaster.
//!
//! - Rules: `card`, `hand` (`u64` bitmask), `state` (`BlobState`, a `Copy`
//!   stack struct), `round`, `dealing`, `bidding`, `playing`, `game`.
//!   `dealing::new_round` and `round::RoundMix` start single rounds.
//! - Objective: `scoring` (the per-round utility `u_s`, gen-2.md §5.1).
//! - Network input: `encoder` (variable-length typed token sequences, P and
//!   V modes, one guarded layout).
//! - Search: `mcts` over sampled deals from `belief`, guided by a policy and
//!   a value evaluator (`evaluator`; `onnx::OnnxEvaluator` in production).
//! - Training data: `replay`, with suit-permutation augmentation from
//!   `augment`; `teacher` plays the warm start's rule-bot-2 rounds.
//! - Yardsticks: `rule_bot`, `rule_bot_2` and `bench` (gen-2.md §5.7).
//!
//! Pure Rust plus `ort`; never depends on `tch` (that lives in `blob-nn`).

pub mod augment;
pub mod belief;
pub mod bench;
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
pub mod profiling;
pub mod replay;
pub mod round;
pub mod rule_bot;
pub mod rule_bot_2;
pub mod scoring;
pub mod state;
pub mod teacher;

pub use bidding::{apply_bid, forbidden_bid, legal_bids};
pub use evaluator::{DummyEvaluator, PolicyEvaluator, ValueEvaluator, NUM_BIDS};
pub use onnx::{OnnxEvaluator, OnnxPolicy, OnnxValue};
pub use card::{Card, Suit, MAX_CARDS_DEALT, NUM_CARDS, NUM_RANKS, NUM_SUITS};
pub use dealing::{deal, new_round, start_round, RoundParams};
pub use game::{advance_round, is_game_over, new_game};
pub use hand::Hand;
pub use belief::{determinize, void_suits, VoidTable, DEFAULT_DETERMINIZE_ATTEMPTS};
pub use mcts::{
    apply_action, backup, expand, is_terminal, mcts_search, root_action_probs, run_search,
    select_best_child, select_leaf, signal_ratio, ucb1_score, MctsArena, MctsConfig, MctsNode,
    MctsResult, SearchBudget, DEFAULT_ARENA_CAPACITY, DEFAULT_C_PUCT,
};
pub use rule_bot::{rule_bot_action, rule_bot_bid, rule_bot_play};
pub use rule_bot_2::{rule_bot_2_action, rule_bot_2_bid, rule_bot_2_play};
pub use scoring::{round_points, terminal_utilities, utilities, DEFAULT_LAMBDA};
pub use playing::{apply_play, legal_plays, score_round};
pub use replay::{
    BidBatch, Decision, PlayBatch, ReplayBuffer, SharedReplay, SparsePolicy, MAX_BID_ACTIONS,
    MAX_PLAY_ACTIONS,
};
pub use round::{
    cards_dealt_for_round, round_structure, total_rounds, trump_for_round, validate_round_params,
    RoundMix, RoundParamsError, NO_TRUMP, TRUMP_CYCLE_LEN,
};
pub use state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS, MIN_PLAYERS};
pub use teacher::{fill_buffer, teacher_policy, teacher_round, TeacherConfig};
