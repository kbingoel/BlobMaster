//! Tauri command surface. Every entry point the frontend can reach lives
//! here. Bodies marked **stub** return well-formed mock data — wiring the
//! real engine call is deferred to the session that owns the feature.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use blob_engine::bidding::{apply_bid, legal_bids};
use blob_engine::dealing::start_round;
use blob_engine::onnx::OnnxEvaluator;
use blob_engine::state::GamePhase as EnginePhase;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tauri::State;

use crate::session::{indices_to_hand, GameSession};
use crate::types::{
    AiSuggestion, EngineSettings, GameConfig, GameEvent, GuiError, GuiResult, ModelInfo,
    SessionSnapshot,
};

/// `tauri::State` payload. `Mutex` is fine for our single-user GUI — every
/// command takes the lock for the duration of one call.
pub type AppState = Mutex<Option<GameSession>>;

// ---- helpers -------------------------------------------------------------

fn with_session<R>(
    state: &State<'_, AppState>,
    f: impl FnOnce(&mut GameSession) -> GuiResult<R>,
) -> GuiResult<R> {
    let mut guard = state.lock().expect("AppState mutex poisoned");
    let session = guard.as_mut().ok_or(GuiError::NoSession)?;
    f(session)
}

fn rng_for(session: &GameSession) -> Xoshiro256PlusPlus {
    match session.engine_settings.deterministic_seed {
        Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
        None => Xoshiro256PlusPlus::from_entropy(),
    }
}

fn checkpoints_dir() -> PathBuf {
    // The repo layout puts the GUI under `blob-gui/src-tauri`; the
    // `checkpoints/` directory sits at the workspace root, two levels up
    // from the binary's `CARGO_MANIFEST_DIR`. At runtime we look relative
    // to the current working directory and fall back to the env-known
    // manifest path, which keeps `bun tauri dev` working.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let candidate = cwd.join("checkpoints");
    if candidate.is_dir() {
        return candidate;
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("checkpoints"))
        .unwrap_or(candidate)
}

fn modified_unix_secs(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

// ---- commands ------------------------------------------------------------

/// Smoke command from Session 9.1 — kept alive so the existing
/// `+page.svelte` button still has something to call.
#[tauri::command]
#[specta::specta]
pub fn engine_version() -> String {
    format!(
        "blob-engine {} — {} cards · {} suits · max {} dealt",
        env!("CARGO_PKG_VERSION"),
        blob_engine::NUM_CARDS,
        blob_engine::NUM_SUITS,
        blob_engine::MAX_CARDS_DEALT,
    )
}

/// Scan `checkpoints/` for `*.onnx` files plus the recents list.
///
/// Stub: just enumerates files in the workspace `checkpoints/` directory.
/// The recents-list integration lands in Session 9.3.
#[tauri::command]
#[specta::specta]
pub fn list_models() -> GuiResult<Vec<ModelInfo>> {
    let dir = checkpoints_dir();
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("onnx") {
            continue;
        }
        let meta = entry.metadata()?;
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        out.push(ModelInfo {
            path: path.clone(),
            file_name,
            size_bytes: meta.len(),
            modified_unix_secs: modified_unix_secs(&meta),
            d_model: None,
            n_layers: None,
            n_heads: None,
        });
    }
    out.sort_by(|a, b| b.modified_unix_secs.cmp(&a.modified_unix_secs));
    Ok(out)
}

/// Load an ONNX model into the active session.
///
/// `OnnxEvaluator::from_file` does the real work; on success the session's
/// `evaluator` slot is populated and subsequent `request_ai_suggestion`
/// calls have a model to use. If no session exists yet, the load still
/// happens and the model is dropped — the next `new_game` will need to
/// reload. (Session 9.3 will refactor this so the model lives outside the
/// session.)
#[tauri::command]
#[specta::specta]
pub fn load_model(state: State<'_, AppState>, path: PathBuf) -> GuiResult<ModelInfo> {
    let evaluator = OnnxEvaluator::from_file(&path)
        .map_err(|e| GuiError::ModelLoadFailed(e.to_string()))?;
    let meta = std::fs::metadata(&path)?;
    let info = ModelInfo {
        file_name: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string(),
        path: path.clone(),
        size_bytes: meta.len(),
        modified_unix_secs: modified_unix_secs(&meta),
        // Session-shape introspection lands in Session 9.3.
        d_model: None,
        n_layers: None,
        n_heads: None,
    };
    if let Ok(mut guard) = state.lock() {
        if let Some(session) = guard.as_mut() {
            session.evaluator = Some(evaluator);
        }
    }
    Ok(info)
}

/// Initialize a fresh game from the setup screen and replace any prior
/// session. Returns the initial snapshot (Bidding phase, no hands dealt
/// to opponents — opponents stay empty until belief sampling).
#[tauri::command]
#[specta::specta]
pub fn new_game(
    state: State<'_, AppState>,
    config: GameConfig,
) -> GuiResult<SessionSnapshot> {
    let session = GameSession::from_config(config)?;
    let snap = session.snapshot();
    let mut guard = state.lock().expect("AppState mutex poisoned");
    *guard = Some(session);
    Ok(snap)
}

/// Record the human's hand for the current round.
///
/// Validates count == `cards_dealt`, stores the bitmask on
/// `session.human_hand`, and writes through to `state.hands[human_seat]`.
/// Other seats remain empty — belief sampling will fill them at AI time.
/// Also seeds the engine RNG so opponents' "hands" can be drawn for
/// determinization later (Session 9.6).
#[tauri::command]
#[specta::specta]
pub fn set_human_hand(
    state: State<'_, AppState>,
    cards: Vec<u8>,
) -> GuiResult<SessionSnapshot> {
    with_session(&state, |session| {
        let cards_dealt = session.state.cards_dealt as usize;
        if cards.len() != cards_dealt {
            return Err(GuiError::InvalidConfig(format!(
                "expected {cards_dealt} cards, got {}",
                cards.len()
            )));
        }
        let mask = indices_to_hand(&cards)?;
        session.human_hand = mask;
        let seat = session.human_seat as usize;
        session.state.hands[seat] = mask;
        // Initialize the deck shuffle for opponents-unknown by running
        // start_round once. The engine's `start_round` resets state and
        // would clobber our hand, so we don't call it here — instead we
        // leave opponents empty (the belief sampler fills them later).
        session.event_log.push(GameEvent::SetHumanHand { cards });
        Ok(session.snapshot())
    })
}

/// Submit the active player's bid. Wraps [`apply_bid`].
#[tauri::command]
#[specta::specta]
pub fn submit_bid(
    state: State<'_, AppState>,
    seat: u8,
    bid: u8,
) -> GuiResult<SessionSnapshot> {
    with_session(&state, |session| {
        if session.state.phase() != EnginePhase::Bidding {
            return Err(GuiError::WrongPhase(format!(
                "phase is {:?}, not Bidding",
                session.state.phase()
            )));
        }
        if seat != session.state.current_player {
            return Err(GuiError::IllegalAction(format!(
                "seat {seat} bid out of turn (current={})",
                session.state.current_player
            )));
        }
        let mask = legal_bids(&session.state);
        if (mask >> bid) & 1 == 0 {
            return Err(GuiError::IllegalAction(format!(
                "bid {bid} not in legal mask {mask:b}"
            )));
        }
        apply_bid(&mut session.state, bid);
        session.bid_placed[seat as usize] = true;
        session.event_log.push(GameEvent::Bid { seat, bid });
        Ok(session.snapshot())
    })
}

/// Record a card played by `seat`. Public-state-only wrapper around
/// [`apply_play`].
///
/// **Stub** — the public-state-only variant of `apply_play` lands in
/// Session 9.6. For now this surfaces a structured `NotImplemented` error
/// so the frontend can wire the call site without crashing.
#[tauri::command]
#[specta::specta]
pub fn record_card_played(
    state: State<'_, AppState>,
    seat: u8,
    card: u8,
) -> GuiResult<SessionSnapshot> {
    with_session(&state, |_session| {
        Err(GuiError::NotImplemented(format!(
            "record_card_played(seat={seat}, card={card}) — wired in Session 9.6"
        )))
    })
}

/// Run an AI suggestion on the current state.
///
/// **Stub** — returns a deterministic mock payload so the frontend can
/// render the eval surface end-to-end. Real MCTS + belief wiring lands in
/// Session 9.7.
#[tauri::command]
#[specta::specta]
pub fn request_ai_suggestion(state: State<'_, AppState>) -> GuiResult<AiSuggestion> {
    with_session(&state, |session| {
        let phase = session.state.phase();
        match phase {
            EnginePhase::Bidding => {
                let cards_dealt = session.state.cards_dealt as usize;
                let mut policy = vec![0.0f32; cards_dealt + 1];
                if !policy.is_empty() {
                    let mid = policy.len() / 2;
                    policy[mid] = 1.0;
                }
                Ok(AiSuggestion::Bidding {
                    policy,
                    recommended_bid: (cards_dealt / 2) as u8,
                    value_estimate: 0.0,
                })
            }
            EnginePhase::Playing => Ok(AiSuggestion::Playing {
                per_card: Vec::new(),
                recommended_card: 0,
                value_estimate: 0.0,
                sims_completed: 0,
                depth: 0,
            }),
            other => Err(GuiError::WrongPhase(format!(
                "AI suggestion not available in phase {other:?}"
            ))),
        }
    })
}

#[tauri::command]
#[specta::specta]
pub fn update_engine_settings(
    state: State<'_, AppState>,
    settings: EngineSettings,
) -> GuiResult<()> {
    with_session(&state, |session| {
        session.engine_settings = settings;
        Ok(())
    })
}

/// Pop the last event from the session log.
///
/// **Stub** — replay-from-checkpoint logic lands in Session 9.8. For now
/// the event is removed from the log but state is not rewound, so this
/// only signals "I tried to undo" to the frontend.
#[tauri::command]
#[specta::specta]
pub fn undo_last_event(state: State<'_, AppState>) -> GuiResult<SessionSnapshot> {
    with_session(&state, |session| {
        if session.event_log.pop().is_none() {
            return Err(GuiError::IllegalAction("nothing to undo".into()));
        }
        Ok(session.snapshot())
    })
}

#[tauri::command]
#[specta::specta]
pub fn save_session(state: State<'_, AppState>) -> GuiResult<PathBuf> {
    with_session(&state, |_session| {
        Err(GuiError::NotImplemented(
            "save_session — wired in Session 9.8".into(),
        ))
    })
}

#[tauri::command]
#[specta::specta]
pub fn load_session(_path: PathBuf) -> GuiResult<SessionSnapshot> {
    Err(GuiError::NotImplemented(
        "load_session — wired in Session 9.8".into(),
    ))
}

// ---- internal helpers (currently unused — will land with later sessions)

#[allow(dead_code)]
fn _ensure_round_dealt(session: &mut GameSession) {
    let mut rng = rng_for(session);
    if session.state.hands[session.human_seat as usize] == 0 {
        // Used by Session 9.6 once the belief-aware `start_round` wrapper
        // exists. Today, `start_round` would clobber the human hand the
        // user just entered, so we leave this path dormant.
        let _ = (start_round::<Xoshiro256PlusPlus>, &mut rng);
    }
}

#[allow(dead_code)]
fn _checkpoints_for_test() -> &'static Path {
    Path::new("checkpoints")
}
