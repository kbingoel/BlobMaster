//! Tauri command surface. Every entry point the frontend can reach lives
//! here. Bodies marked **stub** return well-formed mock data — wiring the
//! real engine call is deferred to the session that owns the feature.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use blob_engine::belief::{determinize, void_suits, DEFAULT_DETERMINIZE_ATTEMPTS};
use blob_engine::bidding::{apply_bid, legal_bids};
use blob_engine::card::NUM_CARDS;
use blob_engine::dealing::start_round;
use blob_engine::encoder::encode;
use blob_engine::evaluator::{Evaluator, HeuristicEvaluator};
use blob_engine::game::advance_round as engine_advance_round;
use blob_engine::mcts::{run_search, MctsArena};
use blob_engine::onnx::OnnxEvaluator;
use blob_engine::playing::{apply_play, legal_plays};
use blob_engine::round::{cards_dealt_for_round, total_rounds, trump_for_round};
use blob_engine::state::{GamePhase as EnginePhase, MAX_PLAYERS};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tauri::State;

use crate::session::{indices_to_hand, GameSession, PersistedSession};
use crate::types::{
    AiSuggestion, AppSettings, CardEval, EngineSettings, GameConfig, GameEvent, GuiError,
    GuiResult, ModelInfo, RoundScoreRow, RoundStructureEntry, RoundSummary, SavedSessionInfo,
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

/// `~/.blobmaster/`. Created on first write.
fn config_dir() -> GuiResult<PathBuf> {
    let base = dirs::home_dir()
        .ok_or_else(|| GuiError::Io("could not resolve user home directory".into()))?;
    Ok(base.join(".blobmaster"))
}

fn settings_path() -> GuiResult<PathBuf> {
    Ok(config_dir()?.join("settings.json"))
}

fn recents_path() -> GuiResult<PathBuf> {
    Ok(config_dir()?.join("recents.json"))
}

fn sessions_dir() -> GuiResult<PathBuf> {
    Ok(config_dir()?.join("sessions"))
}

/// Build a `ModelInfo` from a path. Returns `None` if the file is missing
/// or unreadable (used for filtering recents whose target was deleted).
fn model_info_for(path: &Path) -> Option<ModelInfo> {
    let meta = std::fs::metadata(path).ok()?;
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    Some(ModelInfo {
        path: path.to_path_buf(),
        file_name,
        size_bytes: meta.len(),
        modified_unix_secs: modified_unix_secs(&meta),
        d_model: None,
        n_layers: None,
        n_heads: None,
    })
}

/// Read the recents list (most-recent first), dropping entries whose file
/// no longer exists.
fn read_recents() -> Vec<PathBuf> {
    let Ok(p) = recents_path() else { return Vec::new() };
    let Ok(buf) = std::fs::read_to_string(&p) else { return Vec::new() };
    serde_json::from_str::<Vec<PathBuf>>(&buf).unwrap_or_default()
}

fn write_recents(paths: &[PathBuf]) -> GuiResult<()> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    let p = recents_path()?;
    let buf = serde_json::to_string_pretty(paths)
        .map_err(|e| GuiError::Io(format!("serialize recents: {e}")))?;
    std::fs::write(&p, buf)?;
    Ok(())
}

/// Cap the recents list at this many entries. The setup screen only shows
/// a handful, and the file is meant to stay human-skimmable.
const RECENTS_CAP: usize = 10;

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

/// Scan `checkpoints/` for `*.onnx` files. Sorted by modified-time, newest
/// first. The recents list is queried separately via [`list_recent_models`]
/// so the setup screen can render the two in distinct sections.
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
        if let Some(info) = model_info_for(&path) {
            out.push(info);
        }
    }
    out.sort_by(|a, b| b.modified_unix_secs.cmp(&a.modified_unix_secs));
    Ok(out)
}

/// Recently loaded models, most-recent first. Entries whose underlying
/// file is missing are silently dropped.
#[tauri::command]
#[specta::specta]
pub fn list_recent_models() -> GuiResult<Vec<ModelInfo>> {
    let mut out = Vec::new();
    for p in read_recents() {
        if let Some(info) = model_info_for(&p) {
            out.push(info);
        }
    }
    Ok(out)
}

/// Promote `path` to the head of the recents list. Idempotent — calling
/// with the same path twice doesn't duplicate it.
#[tauri::command]
#[specta::specta]
pub fn add_recent_model(path: PathBuf) -> GuiResult<Vec<ModelInfo>> {
    let mut paths = read_recents();
    paths.retain(|p| p != &path);
    paths.insert(0, path);
    paths.truncate(RECENTS_CAP);
    write_recents(&paths)?;
    list_recent_models()
}

/// Load an ONNX model. Validates the file by constructing an
/// `OnnxEvaluator`; on success it's installed onto the active session
/// (if any) and the path is promoted to the head of the recents list.
///
/// Architecture metadata (`d_model`, `n_layers`, `n_heads`) is not derivable
/// from the public ONNX I/O contract — those dims live in initializer
/// shapes the `ort` crate doesn't surface. Returned as `None`; the setup
/// screen displays only the file size + modified time when they are.
#[tauri::command]
#[specta::specta]
pub fn load_model(state: State<'_, AppState>, path: PathBuf) -> GuiResult<ModelInfo> {
    let evaluator = OnnxEvaluator::from_file(&path)
        .map_err(|e| GuiError::ModelLoadFailed(e.to_string()))?;
    let info = model_info_for(&path)
        .ok_or_else(|| GuiError::Io(format!("could not stat {}", path.display())))?;
    if let Ok(mut guard) = state.lock() {
        if let Some(session) = guard.as_mut() {
            session.evaluator = Some(evaluator);
        }
    }
    // Best-effort recents update — failure to write the file shouldn't
    // block the user from playing.
    let mut paths = read_recents();
    paths.retain(|p| p != &path);
    paths.insert(0, path);
    paths.truncate(RECENTS_CAP);
    let _ = write_recents(&paths);
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
/// For the human seat we hold the real hand and the engine's `legal_plays`
/// mask is authoritative. For opponents we don't know their hand at all —
/// we synthesize a one-card hand consisting of just the played card so
/// `apply_play` can run its trick bookkeeping uniformly. The card is
/// validated against public knowledge (not already played, not in the
/// human's hand, in range) before being recorded.
#[tauri::command]
#[specta::specta]
pub fn record_card_played(
    state: State<'_, AppState>,
    seat: u8,
    card: u8,
) -> GuiResult<SessionSnapshot> {
    with_session(&state, |session| {
        if session.state.phase() != EnginePhase::Playing {
            return Err(GuiError::WrongPhase(format!(
                "phase is {:?}, not Playing",
                session.state.phase()
            )));
        }
        if seat != session.state.current_player {
            return Err(GuiError::IllegalAction(format!(
                "seat {seat} played out of turn (current={})",
                session.state.current_player
            )));
        }
        if card >= NUM_CARDS {
            return Err(GuiError::IllegalAction(format!(
                "card index {card} out of range (0..{NUM_CARDS})"
            )));
        }
        let bit = 1u64 << card;
        if session.state.played_this_round & bit != 0 {
            return Err(GuiError::IllegalAction(format!(
                "card {card} already played this round"
            )));
        }

        let is_human = seat == session.human_seat;
        if is_human {
            if session.human_hand & bit == 0 {
                return Err(GuiError::IllegalAction(format!(
                    "card {card} not in human's hand"
                )));
            }
            let mask = legal_plays(&session.state);
            if (mask >> card) & 1 == 0 {
                return Err(GuiError::IllegalAction(format!(
                    "card {card} fails follow-suit constraint"
                )));
            }
        } else {
            // Opponent's play: card cannot be one the human is known to hold.
            if session.human_hand & bit != 0 {
                return Err(GuiError::IllegalAction(format!(
                    "card {card} is in the human's hand"
                )));
            }
            // Synthesize a single-card hand so apply_play's debug_assert
            // and legal_plays accept the move uniformly.
            session.state.hands[seat as usize] = bit;
        }

        apply_play(&mut session.state, card);

        if is_human {
            session.human_hand &= !bit;
        }
        // Opponent hand is now 0 again (apply_play removed the bit).

        session.event_log.push(GameEvent::Play { seat, card });
        Ok(session.snapshot())
    })
}

/// Run an AI suggestion on the current state.
///
/// Uses the loaded `OnnxEvaluator` if one is installed on the session,
/// otherwise falls back to `HeuristicEvaluator` so the AI surface stays
/// usable without a trained model. Per-card metrics are averaged across
/// `determinization_samples` belief samples; visit counts are summed.
///
/// Streaming `ai-thinking` progress events and a CancellationToken are
/// out of scope for this pass — the heuristic path completes well under
/// the 1.5 s budget and the on-NN path can be split into a tokio task in
/// a follow-up without touching the result shape.
#[tauri::command]
#[specta::specta]
pub fn request_ai_suggestion(state: State<'_, AppState>) -> GuiResult<AiSuggestion> {
    with_session(&state, |session| {
        let phase = session.state.phase();
        match phase {
            EnginePhase::Bidding => compute_bidding_suggestion(session),
            EnginePhase::Playing => compute_playing_suggestion(session),
            other => Err(GuiError::WrongPhase(format!(
                "AI suggestion not available in phase {other:?}"
            ))),
        }
    })
}

/// Adapter so we can call the same orchestration code on either the
/// loaded ONNX model or the heuristic fallback without dynamic dispatch
/// at every leaf.
fn run_with_evaluator<R>(session: &GameSession, f: impl FnOnce(&dyn Evaluator) -> R) -> R {
    match &session.evaluator {
        Some(onnx) => f(onnx as &dyn Evaluator),
        None => f(&HeuristicEvaluator as &dyn Evaluator),
    }
}

fn compute_bidding_suggestion(session: &GameSession) -> GuiResult<AiSuggestion> {
    let s = &session.state;
    let cards_dealt = s.cards_dealt as usize;
    let mask = legal_bids(s);
    if mask == 0 {
        return Err(GuiError::IllegalAction(
            "no legal bids available".into(),
        ));
    }

    // Bidding is fully observable from the bidder's perspective (their own
    // hand was just entered), so a single evaluator call is enough — no
    // determinization needed for the policy/value at the bidding root.
    let (raw_policy, value) = run_with_evaluator(session, |eval| eval.evaluate(s));

    // Project onto the legal mask + 0..=cards_dealt window the frontend
    // expects, renormalize.
    let mut policy = vec![0.0f32; cards_dealt + 1];
    let mut total = 0.0f32;
    for b in 0..=cards_dealt as u8 {
        if (mask >> b) & 1 == 1 {
            let p = raw_policy.get(b as usize).copied().unwrap_or(0.0).max(0.0);
            policy[b as usize] = p;
            total += p;
        }
    }
    if total > 0.0 {
        for v in policy.iter_mut() {
            *v /= total;
        }
    } else {
        // Evaluator returned no mass over the legal window (edge case for
        // OnnxEvaluator with degenerate output) — fall back to uniform.
        let n = (0..=cards_dealt as u8).filter(|b| (mask >> *b) & 1 == 1).count();
        let p = 1.0 / n as f32;
        for b in 0..=cards_dealt as u8 {
            if (mask >> b) & 1 == 1 {
                policy[b as usize] = p;
            }
        }
    }

    let recommended_bid = policy
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u8)
        .unwrap_or(0);

    Ok(AiSuggestion::Bidding {
        policy,
        recommended_bid,
        value_estimate: value,
    })
}

fn compute_playing_suggestion(session: &GameSession) -> GuiResult<AiSuggestion> {
    let s = &session.state;
    let perspective = s.current_player;
    let legal_mask = legal_plays(s);
    if legal_mask == 0 {
        return Err(GuiError::IllegalAction(
            "no legal plays available".into(),
        ));
    }

    let cfg = &session.engine_settings;
    let num_dets = cfg.determinization_samples.max(1) as usize;
    let sims_per = cfg.mcts_simulations;
    let voids = void_suits(s);

    // Per-card aggregators, indexed by card 0..52 to keep the math obvious.
    // Only entries with `samples[c] > 0` are meaningful.
    let mut visit_sum = [0u32; NUM_CARDS as usize];
    let mut value_sum = [0.0f32; NUM_CARDS as usize];
    let mut value_n = [0u32; NUM_CARDS as usize];
    let mut policy_sum = [0.0f32; NUM_CARDS as usize];
    let mut policy_n = [0u32; NUM_CARDS as usize];
    let mut root_value_sum = 0.0f32;
    let mut root_value_n = 0u32;
    let mut total_visits = 0u32;
    let mut max_depth = 0u32;

    let c_puct = blob_engine::mcts::DEFAULT_C_PUCT;
    let mut rng = rng_for(session);

    // We need the legal cards in absolute index form for renormalization
    // of the evaluator's hand-position-indexed policy.
    let legal_cards: Vec<u8> = (0..NUM_CARDS).filter(|c| (legal_mask >> *c) & 1 == 1).collect();

    run_with_evaluator(session, |eval| {
        for _ in 0..num_dets {
            let det_state = determinize(s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);

            // Encoder hand-card-index map for THIS determinization (the
            // perspective's hand is preserved, so it's the same every
            // iteration in practice — but recompute defensively).
            let enc = encode(&det_state, perspective);
            let (raw_policy, root_value) = eval.evaluate(&det_state);

            // Renormalize policy across the legal subset and accumulate
            // into the per-card buckets.
            let mut policy_total = 0.0f32;
            let mut per_card_policy = [0.0f32; NUM_CARDS as usize];
            for (pos, &card) in enc.hand_card_indices.iter().enumerate() {
                if (legal_mask >> card) & 1 == 0 {
                    continue;
                }
                let p = raw_policy.get(pos).copied().unwrap_or(0.0).max(0.0);
                per_card_policy[card as usize] = p;
                policy_total += p;
            }
            if policy_total > 0.0 {
                for &card in &legal_cards {
                    per_card_policy[card as usize] /= policy_total;
                }
            } else {
                let p = 1.0 / legal_cards.len() as f32;
                for &card in &legal_cards {
                    per_card_policy[card as usize] = p;
                }
            }
            for &card in &legal_cards {
                policy_sum[card as usize] += per_card_policy[card as usize];
                policy_n[card as usize] += 1;
            }

            root_value_sum += root_value;
            root_value_n += 1;

            if sims_per == 0 {
                // Pure-policy mode: visits stay zero, win-rate falls back
                // to root value projected to [0,1] (handled below).
                continue;
            }

            // Run MCTS on this determinization. We use the per-det
            // `run_search` rather than the cross-det lockstep driver
            // because we need direct access to each arena's root children
            // to extract per-card visits/Q.
            let mut arena = MctsArena::new(perspective);
            run_search(&mut arena, &det_state, eval, sims_per, c_puct);

            let depth = arena_depth(&arena);
            if depth > max_depth {
                max_depth = depth;
            }

            for &child_idx in arena.root().children.iter() {
                let child = arena.node(child_idx);
                if (legal_mask >> child.action) & 1 == 0 {
                    continue;
                }
                let card = child.action as usize;
                visit_sum[card] += child.visit_count;
                total_visits = total_visits.saturating_add(child.visit_count);
                if child.value_counts[perspective as usize] > 0 {
                    value_sum[card] += child.q(perspective);
                    value_n[card] += 1;
                }
            }
        }
    });

    let root_value = if root_value_n > 0 {
        root_value_sum / root_value_n as f32
    } else {
        0.0
    };

    let mut per_card: Vec<CardEval> = legal_cards
        .iter()
        .map(|&card| {
            let i = card as usize;
            let policy = if policy_n[i] > 0 { policy_sum[i] / policy_n[i] as f32 } else { 0.0 };
            let mcts_value = if value_n[i] > 0 { value_sum[i] / value_n[i] as f32 } else { 0.0 };
            // Win-rate uses MCTS Q where available (proper imperfect-info
            // estimate), and falls back to the root value when MCTS was
            // skipped (mcts_simulations == 0). Q ∈ [-1,1] → [0,1].
            let v_for_card = if value_n[i] > 0 { mcts_value } else { root_value };
            let win_rate = ((v_for_card + 1.0) * 0.5).clamp(0.0, 1.0);
            CardEval {
                card,
                policy,
                mcts_visits: visit_sum[i],
                mcts_value,
                win_rate,
            }
        })
        .collect();

    // Recommended card: highest visit count (MCTS available), else highest
    // policy probability, with deterministic tie-break on the lowest card
    // index. Falls back to the first legal card if everything is zero.
    let recommended_card = if total_visits > 0 {
        per_card
            .iter()
            .max_by(|a, b| {
                a.mcts_visits
                    .cmp(&b.mcts_visits)
                    .then_with(|| b.card.cmp(&a.card))
            })
            .map(|e| e.card)
            .unwrap_or(legal_cards[0])
    } else {
        per_card
            .iter()
            .max_by(|a, b| {
                a.policy
                    .partial_cmp(&b.policy)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.card.cmp(&a.card))
            })
            .map(|e| e.card)
            .unwrap_or(legal_cards[0])
    };

    // Sort entries by card index for stable rendering on the frontend.
    per_card.sort_by_key(|e| e.card);

    Ok(AiSuggestion::Playing {
        per_card,
        recommended_card,
        value_estimate: root_value,
        sims_completed: total_visits,
        depth: max_depth,
    })
}

/// BFS depth of an arena rooted at node 0. Used as the displayed search
/// depth — the deepest path reached by any descent in `run_search`.
fn arena_depth(arena: &MctsArena) -> u32 {
    let mut depth = 0u32;
    let mut frontier: Vec<u32> = vec![0];
    while !frontier.is_empty() {
        let mut next: Vec<u32> = Vec::new();
        for &idx in &frontier {
            for &child in arena.node(idx).children.iter() {
                next.push(child);
            }
        }
        if next.is_empty() {
            break;
        }
        depth += 1;
        frontier = next;
    }
    depth
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

/// Compute the round-end summary for the just-finished round.
///
/// Must be called while the engine is in the `Scoring` phase — i.e. after
/// the last `apply_play` but before `advance_round`. Pure: does not mutate
/// state or commit scores.
#[tauri::command]
#[specta::specta]
pub fn round_summary(state: State<'_, AppState>) -> GuiResult<RoundSummary> {
    with_session(&state, |session| {
        let phase = session.state.phase();
        // We allow Complete here too, so the end-of-game screen can show
        // the final round's breakdown (cumulative_after == cumulative_scores).
        let np = session.state.num_players as usize;
        let mut rows = Vec::with_capacity(np);
        for seat in 0..np {
            let bid = session.state.bids[seat];
            let tricks = session.state.tricks_won[seat];
            let round_score = if bid == tricks { 10 + bid } else { 0 };
            // In Scoring the engine has not yet folded the round into
            // cumulative_scores; in Complete it has. Compute the "after"
            // value from whichever side of advance_round we're sitting on.
            let cumulative_after = match phase {
                EnginePhase::Scoring => session.state.cumulative_scores[seat] + round_score as u16,
                _ => session.state.cumulative_scores[seat],
            };
            rows.push(RoundScoreRow {
                seat: seat as u8,
                bid,
                tricks_won: tricks,
                round_score,
                cumulative_after,
            });
        }
        let total = total_rounds(session.state.start_cards, session.state.num_players);
        let is_final_round = session.state.round_idx + 1 >= total;
        Ok(RoundSummary {
            round_idx: session.state.round_idx,
            cards_dealt: session.state.cards_dealt,
            trump_suit: session.state.trump_suit,
            dealer: session.state.dealer,
            player_names: session.player_names.clone(),
            rows,
            is_final_round,
        })
    })
}

/// Score the just-finished round and either deal the next round (Bidding
/// phase) or transition to `Complete`. Wraps [`engine_advance_round`].
///
/// Resets the per-round bookkeeping the GUI tracks alongside the engine —
/// `human_hand` (cleared so `set_human_hand` is required again next round)
/// and `bid_placed`. Appends an [`GameEvent::AdvanceRound`] event to the log.
#[tauri::command]
#[specta::specta]
pub fn advance_round(state: State<'_, AppState>) -> GuiResult<SessionSnapshot> {
    with_session(&state, |session| {
        if session.state.phase() != EnginePhase::Scoring {
            return Err(GuiError::WrongPhase(format!(
                "advance_round requires Scoring, got {:?}",
                session.state.phase()
            )));
        }
        let mut rng = rng_for(session);
        engine_advance_round(&mut session.state, &mut rng);
        // The engine's `start_round` deals fresh hands to *every* seat,
        // including the human's. We overwrite the human seat's hand with 0
        // so the GUI re-prompts via `set_human_hand` (the user is the
        // source of truth for what's actually been dealt at the table).
        if session.state.phase() == EnginePhase::Bidding {
            let seat = session.human_seat as usize;
            session.state.hands[seat] = 0;
            // Engine also dealt cards to the other seats — wipe them too,
            // belief sampling will re-fill them at AI time.
            for s in 0..session.state.num_players as usize {
                if s != seat {
                    session.state.hands[s] = 0;
                }
            }
        }
        session.human_hand = 0;
        session.bid_placed = [false; MAX_PLAYERS];
        session.event_log.push(GameEvent::AdvanceRound);
        Ok(session.snapshot())
    })
}

/// Deterministic round structure for the current game. Used to render the
/// persistent round-progress strip — one entry per round in play order.
#[tauri::command]
#[specta::specta]
pub fn round_structure(state: State<'_, AppState>) -> GuiResult<Vec<RoundStructureEntry>> {
    with_session(&state, |session| {
        let total = total_rounds(session.state.start_cards, session.state.num_players);
        let mut out = Vec::with_capacity(total as usize);
        for r in 0..total {
            out.push(RoundStructureEntry {
                round_idx: r,
                cards_dealt: cards_dealt_for_round(r, session.state.start_cards, session.state.num_players),
                trump_suit: trump_for_round(r as u32),
            });
        }
        Ok(out)
    })
}

/// Persist the current session to `~/.blobmaster/sessions/<timestamp>.json`.
/// Returns the resolved path so the frontend can show a "saved to …" toast.
///
/// Uses Unix-epoch seconds for the filename. Sessions are append-only —
/// there's no in-place overwrite across launches; resuming from one and
/// saving again writes a new file.
#[tauri::command]
#[specta::specta]
pub fn save_session(state: State<'_, AppState>) -> GuiResult<PathBuf> {
    with_session(&state, |session| {
        let dir = sessions_dir()?;
        std::fs::create_dir_all(&dir)?;
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = dir.join(format!("{stamp}.json"));
        let persisted = session.to_persisted();
        let buf = serde_json::to_string_pretty(&persisted)
            .map_err(|e| GuiError::Io(format!("serialize session: {e}")))?;
        std::fs::write(&path, buf)?;
        Ok(path)
    })
}

/// Restore a session from disk, replacing any in-memory session. The
/// returned snapshot is the standard `SessionSnapshot` — the frontend
/// routes based on `phase` (Bidding → /hand-entry if no human hand yet,
/// otherwise /play; Scoring → /round-summary; Complete → /end).
#[tauri::command]
#[specta::specta]
pub fn load_session(state: State<'_, AppState>, path: PathBuf) -> GuiResult<SessionSnapshot> {
    let buf = std::fs::read_to_string(&path)
        .map_err(|e| GuiError::SessionNotFound(format!("{}: {e}", path.display())))?;
    let persisted: PersistedSession = serde_json::from_str(&buf)
        .map_err(|e| GuiError::Io(format!("parse session {}: {e}", path.display())))?;
    let session = GameSession::from_persisted(persisted)?;
    let snap = session.snapshot();
    let mut guard = state.lock().expect("AppState mutex poisoned");
    *guard = Some(session);
    Ok(snap)
}

/// List saved sessions, newest first. Each entry parses the file header to
/// surface enough metadata for the Resume list (player count, round, leader).
/// Files that fail to parse are silently skipped.
#[tauri::command]
#[specta::specta]
pub fn list_sessions() -> GuiResult<Vec<SavedSessionInfo>> {
    let dir = sessions_dir()?;
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(buf) = std::fs::read_to_string(&path) else { continue };
        let Ok(p) = serde_json::from_str::<PersistedSession>(&buf) else { continue };
        let saved_unix_secs = std::fs::metadata(&path)
            .ok()
            .and_then(|m| modified_unix_secs(&m));
        let np = p.state.num_players as usize;
        let total = total_rounds(p.state.start_cards, p.state.num_players);
        // Leader: highest cumulative score, ties go to lowest seat for
        // deterministic display. None if the entire scoreboard is 0.
        let (leader_name, leader_score) = {
            let mut best: Option<(usize, u16)> = None;
            for s in 0..np {
                let sc = p.state.cumulative_scores[s];
                match best {
                    Some((_, b)) if sc <= b => {}
                    _ => best = Some((s, sc)),
                }
            }
            match best {
                Some((s, sc)) if sc > 0 => (
                    p.player_names.get(s).cloned(),
                    sc,
                ),
                _ => (None, 0),
            }
        };
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        out.push(SavedSessionInfo {
            path: path.clone(),
            file_name,
            saved_unix_secs,
            num_players: p.state.num_players,
            start_cards: p.state.start_cards,
            round_idx: p.state.round_idx,
            total_rounds: total,
            phase: p.state.phase().into(),
            leader_name,
            leader_score,
            player_names: p.player_names,
        });
    }
    out.sort_by(|a, b| b.saved_unix_secs.cmp(&a.saved_unix_secs));
    Ok(out)
}

/// Delete a saved-session file. Used by the setup screen's Resume list to
/// prune dead entries; missing files are treated as success.
#[tauri::command]
#[specta::specta]
pub fn delete_session(path: PathBuf) -> GuiResult<()> {
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(GuiError::Io(format!("delete {}: {e}", path.display()))),
    }
}

/// Serialize the in-memory session as JSON and write it to `path`. Used by
/// the end-of-game "Export log" affordance — the frontend collects `path`
/// from a Save dialog and hands it off here so the file write happens
/// in-process (no JS-side `fs` plugin required).
#[tauri::command]
#[specta::specta]
pub fn export_session_log(state: State<'_, AppState>, path: PathBuf) -> GuiResult<PathBuf> {
    with_session(&state, |session| {
        let persisted = session.to_persisted();
        let buf = serde_json::to_string_pretty(&persisted)
            .map_err(|e| GuiError::Io(format!("serialize session: {e}")))?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path, buf)?;
        Ok(path.clone())
    })
}

/// Load persisted setup-screen form values from
/// `~/.blobmaster/settings.json`. Missing or corrupt files yield
/// [`AppSettings::default`] without erroring — first launch is the common
/// case.
#[tauri::command]
#[specta::specta]
pub fn load_app_settings() -> GuiResult<AppSettings> {
    let p = settings_path()?;
    let Ok(buf) = std::fs::read_to_string(&p) else {
        return Ok(AppSettings::default());
    };
    Ok(serde_json::from_str(&buf).unwrap_or_default())
}

/// Persist setup-screen form values. Creates `~/.blobmaster/` on first
/// write.
#[tauri::command]
#[specta::specta]
pub fn save_app_settings(settings: AppSettings) -> GuiResult<()> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    let p = settings_path()?;
    let buf = serde_json::to_string_pretty(&settings)
        .map_err(|e| GuiError::Io(format!("serialize settings: {e}")))?;
    std::fs::write(&p, buf)?;
    Ok(())
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
