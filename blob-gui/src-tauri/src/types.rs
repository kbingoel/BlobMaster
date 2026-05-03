//! Serde + specta types crossing the Tauri IPC bridge.
//!
//! Every type here is the wire format between the Rust core and the SvelteKit
//! frontend. Card encoding is `card_index = suit * 13 + rank` (♠=0 ♥=1 ♣=2 ♦=3,
//! ranks 2=0 … A=12) — same as `blob_engine::card`. Bitmask types from the
//! engine (legal-plays `u64`, legal-bids `u16`) are flattened to `Vec<u8>` of
//! card / bid indices on the way out, so the frontend never juggles BigInts.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use specta::Type;

use blob_engine::state::GamePhase as EnginePhase;

/// IPC mirror of `blob_engine::state::GamePhase`. Wrapped here so we can
/// derive `specta::Type` (the engine crate has no specta dependency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum GamePhase {
    Bidding,
    Playing,
    Scoring,
    Complete,
}

impl From<EnginePhase> for GamePhase {
    fn from(p: EnginePhase) -> Self {
        match p {
            EnginePhase::Bidding => GamePhase::Bidding,
            EnginePhase::Playing => GamePhase::Playing,
            EnginePhase::Scoring => GamePhase::Scoring,
            EnginePhase::Complete => GamePhase::Complete,
        }
    }
}

// ---- configuration -------------------------------------------------------

/// Trump policy for the game. `AutoRotate` follows
/// [`blob_engine::round::trump_for_round`] (the engine's default cycle);
/// the four fixed-suit variants and `NoTrump` are surfaced for completeness
/// but require an engine extension to take effect — for Session 9.2 they are
/// accepted by `new_game` and recorded on the session, but the engine
/// continues to use auto-rotate. Wiring the override is a follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum TrumpMode {
    AutoRotate,
    Spades,
    Hearts,
    Clubs,
    Diamonds,
    NoTrump,
}

impl Default for TrumpMode {
    fn default() -> Self {
        TrumpMode::AutoRotate
    }
}

/// Pre-game configuration (the form on `/setup`).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct GameConfig {
    /// 4..=7 in the GUI; engine accepts 3..=8.
    pub num_players: u8,
    /// Cards per player in the first round (`C` in plan-speak).
    pub start_cards: u8,
    /// Seat the human occupies. 0..num_players.
    pub human_seat: u8,
    /// Initial dealer seat. 0..num_players.
    pub dealer: u8,
    /// Per-seat display names. Length must equal `num_players`.
    pub player_names: Vec<String>,
    /// Trump policy. See [`TrumpMode`].
    pub trump_mode: TrumpMode,
}

/// Engine-side knobs that change between calls but live for the session.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct EngineSettings {
    /// Softmax temperature applied to policy outputs (0.0–2.0). 1.0 = identity.
    pub temperature: f32,
    /// MCTS simulation budget. 0 = pure-policy mode (no tree).
    pub mcts_simulations: u32,
    /// Number of belief determinizations to average per AI suggestion.
    pub determinization_samples: u8,
    /// Optional fixed seed — `None` means time-seeded each call.
    pub deterministic_seed: Option<u64>,
    /// Display mode for inline AI eval (cycled by `E`).
    pub eval_display: EvalDisplay,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            mcts_simulations: 400,
            determinization_samples: 8,
            deterministic_seed: None,
            eval_display: EvalDisplay::WinRate,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum EvalDisplay {
    WinRate,
    Policy,
    MctsVisits,
    Value,
    Off,
}

// ---- model ---------------------------------------------------------------

/// Metadata about a discovered or loaded ONNX model file.
///
/// `d_model`, `n_layers`, `n_heads` are populated only after a successful
/// `load_model` call (read from the session's input shapes); discovery via
/// `list_models` leaves them `None`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub file_name: String,
    pub size_bytes: u64,
    /// Last-modified time as a Unix epoch in seconds. `None` if the
    /// filesystem doesn't expose it (rare on Windows/Linux/macOS).
    pub modified_unix_secs: Option<u64>,
    pub d_model: Option<u32>,
    pub n_layers: Option<u32>,
    pub n_heads: Option<u32>,
}

// ---- snapshot ------------------------------------------------------------

/// Frontend-facing view of a `GameSession`. Authoritative state lives in
/// Rust — this struct is rebuilt on every state-mutating command and the
/// frontend treats it as immutable for rendering.
///
/// Bid/play masks (`legal_bids`, `legal_plays`) are pre-flattened to
/// card/bid index lists. They are populated only when the **human** is the
/// active player; for opponent turns the frontend shouldn't be making play
/// suggestions anyway, and the engine validates legality on submission.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SessionSnapshot {
    // --- meta ---
    pub phase: GamePhase,
    pub num_players: u8,
    pub cards_dealt: u8,
    pub start_cards: u8,
    pub round_idx: u8,
    /// Total rounds in this game (derived from `start_cards` + `num_players`).
    pub total_rounds: u8,
    pub trump_suit: u8,
    pub dealer: u8,
    pub current_player: u8,
    pub trick_leader: u8,

    // --- per-player ---
    /// `bids[i] = Some(b)` once seat `i` has bid this round, else `None`.
    pub bids: Vec<Option<u8>>,
    pub tricks_won: Vec<u8>,
    pub cumulative_scores: Vec<u16>,
    pub player_names: Vec<String>,

    // --- human view ---
    pub human_seat: u8,
    /// Card indices currently in the human's hand (sorted ascending).
    pub human_hand: Vec<u8>,

    // --- trick state ---
    pub trick_in_progress: Vec<TrickPlay>,
    /// Completed tricks for the **current round only**.
    pub trick_history: Vec<CompletedTrick>,

    // --- legality (only populated when current_player == human_seat) ---
    pub legal_bids: Option<Vec<u8>>,
    pub legal_plays: Option<Vec<u8>>,
    /// Dealer's forbidden bid value when currently at the dealer's turn,
    /// per [`blob_engine::bidding::forbidden_bid`].
    pub forbidden_bid: Option<u8>,

    // --- bookkeeping ---
    pub event_log_len: u32,
    pub model_loaded: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Type)]
pub struct TrickPlay {
    pub seat: u8,
    pub card: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CompletedTrick {
    pub leader: u8,
    pub winner: u8,
    pub suit_led: u8,
    /// In play order, length == num_players.
    pub plays: Vec<TrickPlay>,
}

// ---- AI suggestion -------------------------------------------------------

/// Per-card eval annotation rendered on the bottom-left magnified hand and,
/// optionally, on the right CardGrid (Session 9.7).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Type)]
pub struct CardEval {
    pub card: u8,
    /// Network policy ∈ [0, 1], renormalized over the legal subset.
    pub policy: f32,
    /// MCTS visits at the root child for this card. 0 if `mcts_simulations == 0`.
    pub mcts_visits: u32,
    /// MCTS Q-value ∈ [-1, 1] for the acting player.
    pub mcts_value: f32,
    /// Determinization-averaged win-rate ∈ [0, 1].
    pub win_rate: f32,
}

/// Reply from `request_ai_suggestion`. Either bidding or playing.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(tag = "phase", rename_all = "kebab-case")]
pub enum AiSuggestion {
    Bidding {
        /// Index `b` ∈ 0..=cards_dealt (illegal entries clamped to 0).
        policy: Vec<f32>,
        recommended_bid: u8,
        value_estimate: f32,
    },
    Playing {
        per_card: Vec<CardEval>,
        recommended_card: u8,
        value_estimate: f32,
        sims_completed: u32,
        depth: u32,
    },
}

// ---- event log -----------------------------------------------------------

/// Reversible session-level event. Kept on `GameSession.event_log` to power
/// `undo_last_event` (Session 9.2 stub) and saved with the session.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum GameEvent {
    SetHumanHand { cards: Vec<u8> },
    Bid { seat: u8, bid: u8 },
    Play { seat: u8, card: u8 },
    AdvanceRound,
}

// ---- error ---------------------------------------------------------------

/// Tagged error type crossing the bridge. `anyhow::Error` is intentionally
/// not used: the frontend needs to discriminate on `kind`.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize, Type)]
#[serde(tag = "kind", content = "message", rename_all = "kebab-case")]
pub enum GuiError {
    #[error("no game session has been started")]
    NoSession,
    #[error("game configuration is invalid: {0}")]
    InvalidConfig(String),
    #[error("model has not been loaded")]
    ModelNotLoaded,
    #[error("failed to load model: {0}")]
    ModelLoadFailed(String),
    #[error("illegal action for current state: {0}")]
    IllegalAction(String),
    #[error("unexpected game phase: {0}")]
    WrongPhase(String),
    #[error("filesystem error: {0}")]
    Io(String),
    #[error("session file not found: {0}")]
    SessionNotFound(String),
    #[error("not yet implemented: {0}")]
    NotImplemented(String),
}

impl From<std::io::Error> for GuiError {
    fn from(e: std::io::Error) -> Self {
        GuiError::Io(e.to_string())
    }
}

pub type GuiResult<T> = std::result::Result<T, GuiError>;
