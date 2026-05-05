//! `GameSession` — the GUI-side wrapper around `BlobState`.
//!
//! Authoritative game state lives here and is mutated only by Rust commands.
//! The frontend receives serialized [`SessionSnapshot`]s and never mutates
//! locally. Per-session bookkeeping that the engine doesn't track lives on
//! `GameSession`: human seat, model handle, engine knobs, event log for undo.

use serde::{Deserialize, Serialize};

use blob_engine::bidding::{forbidden_bid, legal_bids};
use blob_engine::card::NUM_CARDS;
use blob_engine::onnx::OnnxEvaluator;
use blob_engine::playing::legal_plays;
use blob_engine::round::total_rounds;
use blob_engine::state::{BlobState, GamePhase as EnginePhase, MAX_PLAYERS};

use crate::types::{
    CompletedTrick, EngineSettings, GameConfig, GameEvent, GuiError, GuiResult,
    SessionSnapshot, TrickPlay, TrumpMode,
};

/// On-disk shape of a [`GameSession`]. Mirrors the live struct minus the
/// non-serializable `evaluator`. Versioned so future format changes can be
/// handled gracefully — bump on any breaking schema change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSession {
    pub version: u32,
    pub state: BlobState,
    pub config: GameConfig,
    pub human_seat: u8,
    pub human_hand: u64,
    pub trump_mode: TrumpMode,
    pub player_names: Vec<String>,
    pub engine_settings: EngineSettings,
    pub event_log: Vec<GameEvent>,
    pub bid_placed: [bool; MAX_PLAYERS],
}

const PERSISTED_SESSION_VERSION: u32 = 1;

/// Live session held in `tauri::State<Mutex<Option<GameSession>>>`.
pub struct GameSession {
    pub state: BlobState,
    pub config: GameConfig,
    pub human_seat: u8,
    /// Bitmask of cards the human has declared they hold this round. Same
    /// encoding as `BlobState.hands[i]`. Mirrors `state.hands[human_seat]`
    /// once `set_human_hand` has been called for the round.
    pub human_hand: u64,
    pub trump_mode: TrumpMode,
    pub player_names: Vec<String>,
    pub evaluator: Option<OnnxEvaluator>,
    pub engine_settings: EngineSettings,
    pub event_log: Vec<GameEvent>,
    /// Tracks which seats have placed their bid this round (in dealer-order
    /// from `dealer + 1`). `BlobState.bids` zero-initializes so we can't
    /// tell "bid 0" from "not yet bid" without this side-channel.
    pub bid_placed: [bool; MAX_PLAYERS],
}

impl GameSession {
    pub fn from_config(config: GameConfig) -> GuiResult<Self> {
        validate_config(&config)?;
        let mut state = blob_engine::game::new_game(config.num_players, config.start_cards)
            .map_err(|e| GuiError::InvalidConfig(format!("{e:?}")))?;
        // GUI-configurable starting dealer (engine defaults to seat 0).
        state.dealer = config.dealer;
        state.current_player = (config.dealer + 1) % config.num_players;
        state.trick_leader = state.current_player;
        Ok(Self {
            state,
            human_seat: config.human_seat,
            human_hand: 0,
            trump_mode: config.trump_mode,
            player_names: config.player_names.clone(),
            config,
            evaluator: None,
            engine_settings: EngineSettings::default(),
            event_log: Vec::new(),
            bid_placed: [false; MAX_PLAYERS],
        })
    }

    /// Snapshot used by `save_session`. Drops the loaded ONNX evaluator —
    /// model selection is per-launch and the user re-picks on resume.
    pub fn to_persisted(&self) -> PersistedSession {
        PersistedSession {
            version: PERSISTED_SESSION_VERSION,
            state: self.state,
            config: self.config.clone(),
            human_seat: self.human_seat,
            human_hand: self.human_hand,
            trump_mode: self.trump_mode,
            player_names: self.player_names.clone(),
            engine_settings: self.engine_settings,
            event_log: self.event_log.clone(),
            bid_placed: self.bid_placed,
        }
    }

    /// Inverse of `to_persisted`. The evaluator is left empty — the caller
    /// installs one via `load_model` if AI suggestions are needed.
    pub fn from_persisted(p: PersistedSession) -> GuiResult<Self> {
        if p.version != PERSISTED_SESSION_VERSION {
            return Err(GuiError::InvalidConfig(format!(
                "session file version {} not supported (expected {})",
                p.version, PERSISTED_SESSION_VERSION
            )));
        }
        Ok(Self {
            state: p.state,
            config: p.config,
            human_seat: p.human_seat,
            human_hand: p.human_hand,
            trump_mode: p.trump_mode,
            player_names: p.player_names,
            evaluator: None,
            engine_settings: p.engine_settings,
            event_log: p.event_log,
            bid_placed: p.bid_placed,
        })
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        let np = self.state.num_players as usize;
        let phase = self.state.phase();

        let bids = (0..np)
            .map(|i| {
                if self.bid_placed[i] {
                    Some(self.state.bids[i])
                } else {
                    None
                }
            })
            .collect();

        let trick_in_progress = (0..self.state.trick_cards_played as usize)
            .map(|i| TrickPlay {
                seat: (self.state.trick_leader + i as u8) % self.state.num_players,
                card: self.state.trick_play_order[i],
            })
            .collect();

        let trick_history = (0..self.state.tricks_completed as usize)
            .map(|t| {
                let rec = &self.state.trick_history[t];
                let plays = (0..rec.num_played as usize)
                    .map(|i| TrickPlay {
                        seat: rec.cards[i].0,
                        card: rec.cards[i].1,
                    })
                    .collect();
                CompletedTrick {
                    leader: rec.cards[0].0,
                    winner: rec.winner,
                    suit_led: rec.suit_led,
                    plays,
                }
            })
            .collect();

        let human_active = self.state.current_player == self.human_seat;
        let legal_bids_vec = if human_active && phase == EnginePhase::Bidding {
            Some(mask_to_indices_u16(legal_bids(&self.state)))
        } else {
            None
        };
        let legal_plays_vec = if phase == EnginePhase::Playing {
            let mask = if human_active {
                legal_plays(&self.state)
            } else {
                opponent_legal_mask(&self.state, self.human_hand)
            };
            Some(mask_to_indices_u64(mask))
        } else {
            None
        };
        let forbidden = if phase == EnginePhase::Bidding
            && self.state.current_player == self.state.dealer
        {
            forbidden_bid(&self.state)
        } else {
            None
        };

        SessionSnapshot {
            phase: phase.into(),
            num_players: self.state.num_players,
            cards_dealt: self.state.cards_dealt,
            start_cards: self.state.start_cards,
            round_idx: self.state.round_idx,
            total_rounds: total_rounds(self.state.start_cards, self.state.num_players),
            trump_suit: self.state.trump_suit,
            dealer: self.state.dealer,
            current_player: self.state.current_player,
            trick_leader: self.state.trick_leader,
            bids,
            tricks_won: self.state.tricks_won[..np].to_vec(),
            cumulative_scores: self.state.cumulative_scores[..np].to_vec(),
            player_names: self.player_names.clone(),
            human_seat: self.human_seat,
            human_hand: hand_to_indices(self.human_hand),
            trick_in_progress,
            trick_history,
            legal_bids: legal_bids_vec,
            legal_plays: legal_plays_vec,
            forbidden_bid: forbidden,
            event_log_len: self.event_log.len() as u32,
            model_loaded: self.evaluator.is_some(),
        }
    }
}

fn validate_config(c: &GameConfig) -> GuiResult<()> {
    if !(2..=8).contains(&c.num_players) {
        return Err(GuiError::InvalidConfig(format!(
            "num_players {} out of range",
            c.num_players
        )));
    }
    if c.start_cards == 0 || c.start_cards > 13 {
        return Err(GuiError::InvalidConfig(format!(
            "start_cards {} out of range (1..=13)",
            c.start_cards
        )));
    }
    let total = c.num_players as usize * c.start_cards as usize;
    if total > NUM_CARDS as usize {
        return Err(GuiError::InvalidConfig(format!(
            "num_players × start_cards = {total} exceeds 52"
        )));
    }
    if c.human_seat >= c.num_players {
        return Err(GuiError::InvalidConfig(format!(
            "human_seat {} out of range",
            c.human_seat
        )));
    }
    if c.dealer >= c.num_players {
        return Err(GuiError::InvalidConfig(format!(
            "dealer {} out of range",
            c.dealer
        )));
    }
    if c.player_names.len() != c.num_players as usize {
        return Err(GuiError::InvalidConfig(format!(
            "player_names length {} != num_players {}",
            c.player_names.len(),
            c.num_players
        )));
    }
    Ok(())
}

/// Opponent legality from the GUI's public-knowledge perspective. We don't
/// know the opponent's hand, so any card not yet played and not in the
/// human's hand is potentially in theirs. The engine's strict
/// follow-suit mask isn't reachable without belief sampling — and the
/// user (who can see the table) is the source of truth for the opponent's
/// actual play. We surface the permissive mask and trust the click.
fn opponent_legal_mask(s: &BlobState, human_hand: u64) -> u64 {
    let all = (1u64 << NUM_CARDS) - 1;
    all & !s.played_this_round & !human_hand
}

fn mask_to_indices_u16(mask: u16) -> Vec<u8> {
    (0..16u8).filter(|i| (mask >> i) & 1 == 1).collect()
}

fn mask_to_indices_u64(mask: u64) -> Vec<u8> {
    (0..64u8).filter(|i| (mask >> i) & 1 == 1).collect()
}

pub(crate) fn hand_to_indices(mask: u64) -> Vec<u8> {
    (0..NUM_CARDS).filter(|i| (mask >> i) & 1 == 1).collect()
}

pub(crate) fn indices_to_hand(cards: &[u8]) -> GuiResult<u64> {
    let mut mask: u64 = 0;
    for &c in cards {
        if c >= NUM_CARDS {
            return Err(GuiError::InvalidConfig(format!(
                "card index {c} out of range (0..52)"
            )));
        }
        let bit = 1u64 << c;
        if mask & bit != 0 {
            return Err(GuiError::InvalidConfig(format!(
                "duplicate card index {c}"
            )));
        }
        mask |= bit;
    }
    Ok(mask)
}
