# BlobMaster GUI Development Plan

> **Companion to** [development-plan.md](development-plan.md). Section 9 placeholder is replaced by this document. The core RL system (Sections 1–8) is the upstream dependency: an ONNX model file is the only artifact the GUI needs from training.

> **Terminology** mirrors the engine — "game" = full multi-round session, "round" = one deal-bid-play-score cycle, "trick" = each player plays one card. See [README.md](README.md).

---

## Goals & Non-Goals

**Vision.** The GUI is a fast *tracking surface* for a real-table game, not a card-game simulator. The dominant UI element is a **52-card grid** (4 suit rows × 13 rank columns, plus a header row and label column = 5×14) that fills most of the screen. Every card-related interaction — entering your hand, recording opponents' plays, playing your own card — happens by clicking a cell in this grid. AI evaluation overlays each legal cell like engine lines next to candidate moves in a chess UI.

**Goals**
- Load any ONNX model exported by [scripts/export_onnx.py](scripts/export_onnx.py) and use it as a copilot during a real-table 5-player game.
- One click per played card, regardless of whose turn it is. The active player is implicit context; the click identifies the card.
- Engine handles all rules, scoring, trump rotation, card counting, legality, and inference — the UI is a thin shell.
- AI eval (policy %, value, MCTS visits) rendered inline on the grid for every legal card, refreshed live as state advances.
- Game settings (num_players, starting cards C, trump, dealer, model, MCTS depth, temperature) configurable per session and persisted between launches.
- Architecture-agnostic model loading — input/output names from [blob-engine/src/onnx.rs:11-20](blob-engine/src/onnx.rs#L11-L20) are the only contract; model dims, layer count, and quantization are inferred at runtime.

**Non-Goals (initial scope)**
- Multiplayer over the network — single local user, table is offline.
- Training UI / metric dashboards — TensorBoard already serves that.
- Mobile / responsive design — desktop only, optimized for a 13"+ laptop screen.
- Replay/import from arbitrary game logs — only the in-progress session is tracked.
- Photo-real card rendering, animations, transitions — Unicode suit symbols (♠♥♣♦) and CSS state classes are sufficient. Practicality over polish.

**Prerequisites from [development-plan.md](development-plan.md)**
- Bid success rate > 50%, win rate vs random > 65% (Section 9 gate). Currently iter 32 from training run 4; ship target is iter 229.
- A valid ONNX file under [checkpoints/](checkpoints/) loadable by [blob-engine/src/onnx.rs](blob-engine/src/onnx.rs).

---

## Architecture Overview

**Stack**: Tauri 2 (Rust core) + SvelteKit (static adapter) + TypeScript + Bun (toolchain) + Tailwind CSS 4.

**Process model**: single Tauri binary. WebView2 renders the Svelte SPA; all engine calls cross the Tauri IPC bridge as `#[tauri::command]` invocations. No HTTP server, no second process.

**Crate layout**
```
blob-gui/
├── src-tauri/
│   ├── Cargo.toml         # depends on blob-engine (path), tauri, specta, ts-rs
│   ├── tauri.conf.json
│   └── src/
│       ├── main.rs        # Tauri builder + command registration
│       ├── commands.rs    # #[tauri::command] surface
│       ├── session.rs     # GameSession state held in tauri::State
│       └── types.rs       # serde + ts-rs types crossing the bridge
└── src/                   # SvelteKit
    ├── routes/
    ├── lib/
    │   ├── api.ts         # generated bindings (specta output)
    │   ├── stores/        # game state, settings stores
    │   └── components/
    └── app.html
```

**State ownership**
- Authoritative game state (`BlobState` from [blob-engine/src/state.rs](blob-engine/src/state.rs)) lives in Rust, wrapped in `Mutex<GameSession>` inside `tauri::State`.
- Frontend holds only a serialized snapshot for rendering — never mutates state locally. Every action is a command round-trip.
- `GameSession` adds GUI-only fields the engine doesn't track: human seat index, model handle, engine settings (temperature, MCTS sims), event log for undo.

**Knowledge model (important)**
- Engine's `BlobState.hands: [u64; 8]` assumes perfect information. The GUI only knows the human's hand.
- For AI suggestions and opponent-card-played handling, we use the **belief / determinization** path already in [blob-engine/src/belief.rs:86](blob-engine/src/belief.rs#L86) and [mcts.rs:353](blob-engine/src/mcts.rs#L353). The session stores a `KnownHand` (human's seat) and a `PlayedCards` history; opponent hands are sampled at AI-decision time and constrained by what's been played.
- `apply_card_played(seat, card)` always succeeds at the engine level by mutating only the public state (trick history, played mask, current_player) without requiring opponent hand contents. A small wrapper around [playing.rs:60](blob-engine/src/playing.rs#L60) handles this — to be added in Session 2.

**Independence from model architecture**
- All model-specific knowledge is encapsulated in [blob-engine/src/onnx.rs](blob-engine/src/onnx.rs). The GUI calls `OnnxEvaluator::new(path)` with whatever file the user picks — `d_model`, `num_layers`, `num_heads` are read from the graph.
- Model picker is a simple file dialog (`.onnx` filter) plus a recents list; no metadata file required.

**Screen layout (three-pane, fixed)**

The play screen is split into three panes with no global header:

```
┌───────────────────────────────┬───────────────────────────────┐
│                               │                               │
│  TOP-LEFT (50% × 50%)         │                               │
│  Players panel                │                               │
│  - One row per player         │                               │
│  - Won/bid, total, rank       │                               │
│  - Played cards revealed,     │                               │
│    unplayed slots face-down   │  RIGHT (50% × 100%)           │
│  - Active player highlighted  │  Master CardGrid              │
│                               │  - 4 cols (♠ ♥ ♣ ♦)           │
├───────────────────────────────┤  - 13 rows (A → 2)            │
│                               │  - Every cell is rank+suit    │
│  BOTTOM-LEFT (50% × 50%)      │    in the suit color          │
│  My hand (magnified)          │  - Played cells greyed with   │
│  - Inference status row:      │    seat + trick footer        │
│    MCTS sims, depth, value,   │  - Click = record played card │
│    current score              │  - Keyboard navigable         │
│  - Hand row: each card big,   │                               │
│    eval (winrate/policy)      │                               │
│    inline below               │                               │
│  - Recommended card lifted    │                               │
│  - Placed-this-round strip    │                               │
│                               │                               │
└───────────────────────────────┴───────────────────────────────┘
```

**Right pane — Master CardGrid (`<CardGrid />`)**
- 4 columns × 13 rows, suit columns left-to-right ♠ ♥ ♣ ♦, ranks descending A at top → 2 at bottom (matches a poker tracker's mental model).
- Each cell renders rank text + suit Unicode glyph in the suit color (red for ♥/♦, near-black for ♠/♣). No images, no SVG.
- Cell-state CSS classes derived from `SessionSnapshot`:
  - `in-hand` — the user holds this card (suit-tinted background, bold border).
  - `played-by-N` — already played; greyed + footer showing seat (`P3`) and round/trick (`R7.t3`).
  - `legal` / `illegal` — for the active player only; illegal is low opacity, unclickable.
  - `recommended` — top AI choice (subtle halo).
- Click records a played card for the active player (engine validates legality). Used primarily for opponents' plays — the user normally plays from the bottom-left magnified hand.
- Optional compact eval annotation per legal cell (small policy %, off by default to avoid clutter; full eval lives in the bottom-left pane).

**Top-left pane — Players panel**
- One horizontal row per player (4–7 rows depending on `num_players`). Active player row highlighted.
- Per row:
  - Seat number + name + cumulative-rank badge (1st / 2nd / …).
  - `Won/Bid` chip (e.g. `3/4`), turning red if mathematically impossible to make bid, green if exactly on track.
  - Total cumulative score.
  - This-round card slots: C placeholders. Played cards reveal in suit color; unplayed slots show face-down (`▒` block or generic back). Order = play order within the round.

**Bottom-left pane — My hand (magnified)**
- Top status row: live inference metrics — MCTS sims completed, search depth, root value estimate, my current cumulative score, my round score. This is the "engine readout" strip.
- Hand row: my unplayed cards rendered side-by-side, large. Under each card:
  - Two-line eval overlay (default win-rate primary, policy % secondary — see Session 9.7).
  - Background tinted on a green→red gradient by relative win-rate within the legal subset.
- Recommended card is visually "lifted" (translateY) and outlined.
- Placed strip (smaller, below the hand row): cards I've played this round, in play order, dimmed. Empties at round transition.

**Reuse**
- The `<CardGrid />` component also drives the hand-entry route (Session 9.4) and a future read-only review mode.
- The bottom-left magnified-hand panel is a separate component (`<MyHandPanel />`); used only on `/play`.

---

## Overarching Themes

| Theme | Sessions | Purpose |
|---|---|---|
| A. Foundation | 1, 2 | Tauri+Svelte scaffold, specta-typed IPC, engine bridge |
| B. The CardGrid | 3, 4 | Setup screen, build the central CardGrid component, hand entry mode |
| C. Play loop | 5, 6 | Bidding keypad, trick-play via the grid, click-per-card |
| D. AI eval lines | 7 | Per-cell policy/value/MCTS overlays (chess-engine style) |
| E. Multi-round + persistence | 8 | Round transitions, scoring, save/resume, undo |
| F. Polish & ship | 9 | Error states, MSI installer, log viewer |

**Total: 9 sessions × ~3h = ~27h.** Each session ends in a runnable, demo-able state.

---

## Session 9.1 — Tauri+Svelte scaffold and engine bridge

Stand up the project, prove the IPC plumbing works end-to-end with a single non-trivial engine call.

- `bun create tauri-app blob-gui` → choose SvelteKit + TypeScript. Verify `bun tauri dev` opens a window.
- Add `blob-engine` as a path dependency in `blob-gui/src-tauri/Cargo.toml`. Confirm Windows release build compiles (no `tch` leak — same constraint as [blob-bin](blob-bin/)).
- Configure SvelteKit static adapter (`@sveltejs/adapter-static`) with `prerender = true` and SPA fallback. Tauri serves the built bundle from `dist/`.
- Install Tailwind 4 (`bun add -d tailwindcss @tailwindcss/vite`) with the Vite plugin; replace default styles.
- First command: `#[tauri::command] fn engine_version() -> String` returning a string from `blob_engine`. Wire up a button in `+page.svelte` that calls it via `@tauri-apps/api/core::invoke` and renders the result.
- Add `tauri.conf.json` window config: 1280×800 default, min 1024×640, title "BlobMaster".
- Clean up the legacy `backend/node_modules` directory at repo root (orphan from prior experiment).
- **Done when**: `bun tauri dev` shows a button that round-trips a string through the Rust core.

---

## Session 9.2 — Type-safe IPC contract and command surface

Define the full set of engine-facing commands the UI will need, with auto-generated TypeScript types **and typed command bindings** so the frontend can't drift from Rust signatures or argument shapes.

- Add `specta` + `tauri-specta`. Decision rationale: closes the IPC type gap including command names and argument shapes (ts-rs only generates types, leaves `invoke()` calls unchecked). Document the integration in `src-tauri/README.md`.
- Configure the build script to emit `src/lib/api.ts` containing both type definitions and a typed `commands` object (e.g. `commands.newGame(config)` instead of `invoke("new_game", { config })`).
- Define `GameSession` struct in [src-tauri/src/session.rs](blob-gui/src-tauri/src/session.rs) with: `state: BlobState`, `human_seat: u8`, `human_hand: u64`, `evaluator: Option<OnnxEvaluator>`, `engine_settings: EngineSettings`, `event_log: Vec<GameEvent>`.
- Implement the command surface (signatures only this session — bodies stub-return mock data where engine wiring is non-trivial):
  - `list_models() -> Vec<ModelInfo>` — scans [checkpoints/](checkpoints/) and recents.
  - `load_model(path: PathBuf) -> Result<ModelInfo>` — calls `OnnxEvaluator::new`.
  - `new_game(config: GameConfig) -> SessionSnapshot` — wraps [game.rs:34](blob-engine/src/game.rs#L34).
  - `set_human_hand(cards: Vec<u8>) -> Result<SessionSnapshot>`.
  - `submit_bid(seat: u8, bid: u8) -> Result<SessionSnapshot>` — wraps [bidding.rs:63](blob-engine/src/bidding.rs#L63).
  - `record_card_played(seat: u8, card: u8) -> Result<SessionSnapshot>` — public-state-only wrapper around [playing.rs:60](blob-engine/src/playing.rs#L60).
  - `request_ai_suggestion() -> AiSuggestion` — uses [mcts.rs:353](blob-engine/src/mcts.rs#L353) with current engine settings.
  - `update_engine_settings(settings: EngineSettings) -> ()`.
  - `undo_last_event() -> Result<SessionSnapshot>`.
  - `save_session() / load_session()` — JSON to `~/.blobmaster/sessions/`.
- Define error type `GuiError` mapping engine errors to a tagged enum ts-rs can export. No `anyhow::Error` across the bridge.
- **Done when**: every command compiles, generated `api.ts` has full types, `bun run typecheck` passes, and a smoke test calls `new_game` from the frontend and prints the snapshot.

---

## Session 9.3 — Game setup screen

Build the pre-game configuration UI. This is the first screen the user sees.

- Route: `/setup` (default landing).
- Form fields with Tailwind components:
  - **Number of players**: 4–7, segmented control. Default 5.
  - **Starting cards C**: 1–13, slider with label. Default 7. Validation: `num_players × C ≤ 52` enforced live.
  - **Trump suit / rotation**: dropdown with [Spades, Hearts, Clubs, Diamonds, NoTrump, Auto-rotate]. Default Auto-rotate (matches [development-plan.md:35](development-plan.md#L35)).
  - **Dealer / starting player**: integer 0..num_players, with player-name labels.
  - **Player names**: 4–7 text inputs labeled "You" + opponents. Persisted per-num_players.
  - **Human seat**: which seat number the user occupies. Default 0.
- Model picker section:
  - File dialog (`@tauri-apps/plugin-dialog`) filtered to `*.onnx`.
  - Recents list from `~/.blobmaster/recents.json`.
  - Shows resolved metadata after `load_model`: file size, modified date, `d_model`, `n_layers` (read from session input shapes).
- Engine settings section (collapsible "Advanced"):
  - Temperature: 0.0–2.0 slider, default 1.0.
  - MCTS simulations: 0 (pure policy), 100, 400, 1600, 6400 — preset buttons + custom.
  - Determinization samples: 1–32 slider, default 8.
  - Deterministic-RNG toggle + seed input.
- "Start Game" button: validates, calls `new_game`, persists settings to `~/.blobmaster/settings.json`, navigates to `/hand-entry`.
- **Done when**: a fresh launch lets the user pick a model, configure a 5-player C=7 game, and arrive at the next screen with a populated `SessionSnapshot`.

---

## Session 9.4 — CardGrid component and hand-entry mode

Build the central CardGrid once. Every subsequent session reuses it.

- `lib/components/CardGrid.svelte`: 4-column × 13-row CSS grid. Columns are suits ♠ ♥ ♣ ♦ (left → right), rows are ranks A → 2 (top → bottom). No header/label rows — the suit/rank are visible inside each cell. No SVG, no card images.
- Cell content: rank text (2–10, J, Q, K, A) and suit glyph (♠ U+2660, ♥ U+2665, ♣ U+2663, ♦ U+2666) rendered together in the suit color (red for ♥/♦, near-black for ♠/♣). Compact two-line layout: rank on top, suit below.
- Cell-state derivation: a single `cellState(card_index, snapshot, mode)` pure function returns one of `in-hand | legal | played | illegal | empty`. Drives a class binding; CSS does the rest.
- Mode prop: `"hand-entry" | "play" | "review"`. Click handler is mode-dependent, emitted as a typed Svelte event `on:cardclick={ detail: { card_index } }`.
- Played-cell annotation: greyed background, footer line with seat (`P3`) and round-trick index (`R7.t3`) so the full played history is visible at a glance.
- Sizing: fills its parent container. On the `/play` route, parent is the right pane (50% viewport width × 100% height) — cells become tall rectangles, fine for readability since rank+suit are stacked vertically.
- Route: `/hand-entry` is a special full-screen layout — the CardGrid takes the full window in `"hand-entry"` mode (no three-pane split during hand entry).
  - Click toggles `in-hand`. Counter shows `selected / cards_dealt`. Submit disabled until equal.
  - Sticky strip at top: round number, trump suit symbol, dealer indicator, cards-this-round.
  - Keyboard: rank keys 2–10, J, Q, K, A pre-arm a rank (highlights the row); suit keys S/H/C/D toggle the cell at the intersection. Esc clears the pre-arm.
- "Confirm hand" calls `set_human_hand`. On success navigate to `/play` (three-pane layout).
- "Back to setup" link discards the not-yet-started game.
- **Important**: hand-entry is reused at every round transition (Session 9.8) since `cards_dealt` changes round-to-round.
- **Done when**: the CardGrid component renders with all five cell states; user can select exactly C cards via mouse or keyboard and submit; the same component renders correctly in the right pane of `/play` (with `mode="play"`).

---

## Session 9.5 — Bidding phase

Capture every player's bid in dealer-then-clockwise order, in the three-pane layout.

- Route: `/play` enters in `Bidding` phase per `BlobState.game_phase`. The three-pane layout is the same as trick play; only the bottom-left pane changes content.
- Right pane: CardGrid showing my hand (`in-hand` cells highlighted) — passive during bidding, used as a reference.
- Top-left pane: Players panel, with the bid column showing `—` for not-yet-bid players and the value for those who have bid. Active bidder row highlighted.
- Bottom-left pane: Bidding keypad replacing the magnified hand. Buttons 0..cards_dealt arranged horizontally; active player name prominent above (`P3 (Alice) bids:`).
  - Dealer-restriction rule from [bidding.rs](blob-engine/src/bidding.rs) surfaced visually: forbidden value greyed with tooltip ("would make total bids = cards_dealt").
  - Running tally line: `sum so far / cards_dealt`, color-coded (green if dealer can still pick a legal value, amber otherwise).
  - When the human is the active bidder: AI bid suggestion strip shows top-3 bids with probabilities; recommended bid lifted; Enter accepts.
- Each bid = one `submit_bid` round-trip; UI never advances locally without engine confirmation.
- Phase transitions to `Playing` when all bids are in; the bottom-left pane swaps to the magnified-hand view (Session 9.6).
- **Done when**: a full bidding round for 5 players including the dealer constraint can be entered with one click per bid, the human's turn shows the AI suggestion, and Enter submits the recommended bid.

---

## Session 9.6 — Trick-play in the three-pane layout

The heart of the app. Single click per played card, regardless of whose turn it is. Build out the three panes in parallel.

- Route: `/play` in `Playing` phase. Layout per the diagram in Architecture Overview.

**Top-left — Players panel** (`<PlayersPanel />`)
- One row per player. Sourced from `SessionSnapshot.players[]`.
- Columns per row: rank badge, name + seat, won/bid chip, total score, this-round card slots.
- Card slots: render `cards_dealt` placeholders. Played cards reveal in suit color in play order (read from `BlobState.trick_history` and the in-progress `trick_play_order`); unplayed slots show as face-down (`▒`).
- Active player row: highlighted background + left-edge accent.
- "Will-make-bid" indicator: green if `tricks_won == bid` already (over-trick risk if more come), amber if can still hit, red if mathematically impossible.

**Bottom-left — My hand panel** (`<MyHandPanel />`)
- Inference status row across the top: MCTS sims completed, root value estimate (e.g. `+0.42`), search depth (max ply), my round score, my cumulative score. Updates live via the `ai-thinking` event.
- Hand row: my unplayed cards rendered side-by-side, large. Below each card: two-line eval (Session 9.7).
- The recommended card is "lifted" (translateY -8px) and outlined.
- Placed-this-round strip below the hand row: dimmed thumbnails of the cards I've already played this round, in play order. Empties on `advance_round`.
- Click on a hand card = play it. Enter = play the recommended card.

**Right — Master CardGrid** (`<CardGrid mode="play" />`)
- Renders the full 52-card state. Played cells greyed with seat + trick footer.
- Click on a `legal` cell records the active player's play via `record_card_played`. Used primarily for opponents — the grid is unambiguous about which card; the active player is implicit context.
- For an opponent's turn: legal cells are cards not yet played, not in the human's hand, following suit if required (engine-computed mask).
- For the human's turn: legal cells correspond to my hand's legal subset; clicking either the right pane or the bottom-left pane plays the same card.

**Engine wiring**
- Click handler: `record_card_played(active_seat, card_index)`. Engine validates, updates `current_player`, returns new snapshot. UI re-renders. No optimistic updates.
- Trick winner: snapshot includes the winner's seat. The Players panel row briefly pulses (single 200ms flash, no continuous animation).
- Undo: Ctrl+Z pops the last event from the session event log and replays from a checkpoint. Single-action by default; configurable depth in settings (Session 9.8).

- **Done when**: a full hand of 7 tricks for 5 players plays through with one click per card; click works from both the right grid and the bottom-left hand for the human's turn; legal masks, `tricks_won`, and the placed-strip all match the engine throughout.

---

## Session 9.7 — Inline AI eval (chess-engine style) in the bottom-left pane

Turn the bottom-left pane into a chess-engine evaluation readout. Eval lives **primarily in the magnified-hand panel**, with optional secondary surfacing on the right grid.

- Async command `request_ai_suggestion` runs on a dedicated tokio task and emits a Tauri event `ai-thinking` with progress (`{ sims_completed, depth, root_value }`) so the inference status row updates live without blocking the UI thread.
- Cancellation: if the user plays a card or the active player changes before AI finishes, abort the search via a `CancellationToken` stored in the session.
- Suggestion payload `AiSuggestion`:
  - For Bidding: `{ policy: Vec<f32>, recommended_bid: u8, value_estimate: f32 }` — drives the bidding panel from Session 9.5.
  - For Playing: `{ per_card: HashMap<u8, CardEval>, recommended_card: u8, value_estimate: f32, sims_completed: u32, depth: u32 }` where `CardEval = { policy: f32, mcts_visits: u32, mcts_value: f32, win_rate: f32 }`.
- **Bottom-left pane — primary eval surface**:
  - Inference status row (top of pane): `MCTS: 1620/6400 · depth 9 · v=+0.42 · score 14 (+3 round)`. All values stream live from `ai-thinking`.
  - Below each hand card, two lines (font ~60% of card label):
    - Line 1: primary metric (default win-rate %, e.g. `54%`).
    - Line 2: secondary metric (default policy %, e.g. `π 18%`).
  - Background tint on each card: green→red gradient by win-rate percentile within the legal subset.
  - Recommended card lifted (translateY -8px) + outlined; pressing Enter plays it.
- **Right pane — secondary eval (compact)**:
  - On `legal` cells (only for the human's turn), a single small annotation in the bottom corner: the primary metric only (no two-line breakdown). Avoids visual clutter on the 52-cell grid.
  - Off by default; opt-in via settings ("Show eval on master grid").
- Display-mode toggle: `E` cycles `{ Win-rate, Policy, MCTS-visits, Value, Off }`. Off hides eval everywhere — for table play without AI hints.
- Auto-trigger: every snapshot change with the human as active player triggers a fresh suggestion. Opponent turns can opt in via setting "show eval for all players" (table-coaching mode).
- Belief sampling: AI uses [belief.rs:86](blob-engine/src/belief.rs#L86) determinization with `EngineSettings.determinization_samples` (default 8). Per-card metrics are averaged across samples; `mcts_visits` summed.
- Performance budget: iter-229 model + 400 MCTS sims × 8 determinizations on Intel iGPU CPU fallback, target < 1.5s per suggestion. Baseline against [self-play-profile.md](self-play-profile.md).
- Engine-settings live editor: compact form in the bottom-left pane footer (collapsible) — temperature, MCTS sims, determinization samples, eval-display mode. Changes apply to the next call.
- **Done when**: on every snapshot change with the human active, every card in the magnified hand shows live two-line eval; the inference status row streams updates as MCTS progresses; Enter plays the recommended card; eval-display mode cycles via `E`; mid-game suggestion completes in < 1.5s.

---

## Session 9.8 — Multi-round flow, scoring, save/resume

Wire round transitions, the cumulative scoreboard, and durable session state.

- Round-end screen between rounds:
  - Per-player table: bid, tricks won, round score (per the existing scoring logic in [blob-engine/src/round.rs](blob-engine/src/round.rs)), cumulative score.
  - "Continue to next round" button — calls `advance_round` ([game.rs:67](blob-engine/src/game.rs#L67)) and routes back to `/hand-entry` with the new round's parameters (cards_dealt, trump).
- Round-progress strip persistent at top: shows the [C, C−1, …, 1, 1, …, 1, 2, …, C] structure for `num_players × C ≤ 52` (formula at [development-plan.md:40](development-plan.md#L40)), with current round highlighted and trump suit per round.
- Save format: serialize `GameSession` (with event log) as JSON to `~/.blobmaster/sessions/<timestamp>.json` on every round end and on app close.
- Load: setup screen gains a "Resume game" section listing recent sessions with timestamp, num_players, current round, current cumulative leader.
- Undo across round boundaries: explicitly disabled — once the round summary is dismissed, the round is final. Surface this in the UI with a confirmation dialog on "Continue".
- End-of-game screen: full scoreboard, winner announcement, "Start new game" / "Export log" buttons.
- **Done when**: a full 17-round game (5 players, C=7) plays through cold-launch → save mid-game → quit → resume → finish, with all scores matching the engine.

---

## Session 9.9 — Practical polish, error handling, packaging

Tighten the UX where it matters at the table; ship a Windows installer. Skip cosmetic animation work.

- **Error UX**:
  - Toast system (component in `lib/components/Toast.svelte`) for engine errors (illegal play attempted, model load failure, file IO).
  - Modal recovery flow if `OnnxEvaluator` fails mid-game (e.g. bad model file): keep the game session, prompt to reload model.
- **Loading states**:
  - Per-cell spinner on the CardGrid while AI suggestion is in flight (no blocking overlay).
  - Inline progress text on model load (`Loading model... 14/16 layers`) using the `ai-thinking` event channel.
- **Keyboard-first table use**: the GUI is designed to be driven primarily from the keyboard during a real table game. Mouse is fully supported but never required.
  - **Global**:
    - `Enter` — accept and play the recommended card (during my turn in `Playing`); accept the recommended bid (during my turn in `Bidding`).
    - `Ctrl+Z` — undo the last recorded action (single-step by default; depth configurable).
    - `Ctrl+Shift+Z` / `Ctrl+Y` — redo.
    - `Esc` — cancel an in-flight AI search; clear pre-armed rank in hand-entry.
    - `?` or `F1` — overlay listing all bindings.
    - `E` — cycle eval display mode `{ Win-rate, Policy, MCTS-visits, Value, Off }`.
    - `T` — cycle which player's eval to show (toggle table-coaching mode).
  - **Hand entry**:
    - Rank keys `2`–`9`, `T`, `J`, `Q`, `K`, `A` — pre-arm a rank (highlights the row).
    - Suit keys `S`, `H`, `C`, `D` — toggle the cell at the pre-armed rank in that suit.
  - **Bidding**:
    - Number keys `0`–`9` (and `Shift+` for 10–13 if `cards_dealt > 9`) — record the active bidder's bid.
  - **Trick play**:
    - `←` / `→` — move selection within my hand (bottom-left pane).
    - `Space` — play the currently selected card from my hand.
    - For recording opponents' plays: rank+suit combo (same as hand entry) records the active opponent's card; or click in the right grid.
  - **Round transitions**:
    - `N` — continue to next round from the round-summary screen.
  - All bindings are documented in [blob-gui/README.md](blob-gui/README.md) and the in-app `?` overlay; the overlay is generated from a single `KEYMAP` constant so docs cannot drift.
- **Settings persistence** via `tauri-plugin-store`: window size, last-used model, engine defaults, player names, eval display mode.
- **Logging**: `tracing` in Rust → file at `~/.blobmaster/logs/gui.log` with rotation. Log viewer route at `/debug/logs` (gated behind a "Developer mode" toggle).
- **Build**: configure `tauri.conf.json` bundle for Windows MSI + portable EXE. Bundle no model — user picks at runtime.
- **README**: short [blob-gui/README.md](blob-gui/README.md) — install, dev loop, build, model-file expectations, the CardGrid contract.
- **Smoke checklist**: cold install on a clean Windows VM, run a full game with the iter-229 ONNX, verify no missing DLLs, no telemetry surprises, eval lines render in < 1.5s.
- **Explicitly skipped**: card-flip / dealing animations, sound effects, themes. Practicality first.
- **Done when**: an MSI installer produces a working app on a fresh Windows install with no Rust/Node toolchain present.

---

## Cross-Cutting Concerns

**Testing strategy**
- Rust commands: unit-test each `#[tauri::command]` body by calling the underlying function directly (Tauri-free).
- Integration: a `headless_session` test in [src-tauri/tests/](blob-gui/src-tauri/tests/) drives a full game through the command surface.
- Frontend: skip framework testing libs initially. Vitest + Testing Library only if a component grows non-trivial (Session 9.6 trick-play view is the candidate).

**Type-safety contract**
- Single source of truth: Rust types with `#[derive(specta::Type)]`, command bindings via `tauri-specta`. `bun run gen-types` runs as part of `bun tauri dev` via a Cargo build script.
- Card encoding (`card_index = suit * 13 + rank`, suits S=0 H=1 C=2 D=3) shared between Rust and a single TS helper module — no parallel definitions.

**Visual style**
- Pure HTML + Tailwind, no card images, no SVG card faces. Unicode suit glyphs (♠ ♥ ♣ ♦) and rank text only.
- Color: red for hearts/diamonds, near-black for spades/clubs. Cell-state classes (`in-hand`, `legal`, `played`, `illegal`, `recommended`) drive borders and backgrounds. Eval-line tinting uses a green→red gradient on cell background.
- One CSS file (`app.css`) plus Tailwind utilities. No CSS-in-JS, no styling libraries.

**Independence from model**
- The GUI never reads model dims, layer counts, or training metadata. `OnnxEvaluator` is the only consumer of those.
- A model picked from disk that doesn't conform to the I/O contract in [onnx.rs:11-20](blob-engine/src/onnx.rs#L11-L20) fails fast at `load_model` with a structured error surfaced in the setup screen.

**Out-of-scope but worth noting**
- Online play / spectator mode — would need a real WS server.
- Mobile — Tauri Mobile exists but iGPU + ONNX path is desktop-only.
- Training observability — keep using TensorBoard.
- Replay viewer — the event log exists from Session 9.2; a viewer route is a future session.

---

**Total estimated GUI sessions: 9 sessions (~27 hours).** Successor work (replay viewer, online play, mobile) deferred until end-user feedback from real table play.
