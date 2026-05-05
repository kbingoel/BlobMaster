# BlobMaster GUI

Tauri 2 + SvelteKit copilot for Blob, the trick-taking card game. Tracks a real-table game, surfaces AI suggestions from a loaded ONNX model, persists sessions, and stays out of your way at the table.

## Install & dev loop

Prereqs: Bun, Rust toolchain, Tauri prerequisites for your OS (WebView2 on Windows). See the [Tauri prerequisites guide](https://v2.tauri.app/start/prerequisites/) once.

```sh
cd blob-gui
bun install
bun tauri dev      # hot-reload dev build; regenerates src/lib/api.ts on launch
```

## Build

```sh
bun tauri build    # produces a release exe under ../target/release/
```

The Windows installer step is intentionally skipped in the current release scope.

## Model files

The GUI takes any ONNX model exported by [`scripts/export_onnx.py`](../scripts/export_onnx.py). The setup screen has a file picker plus a recents list and a `checkpoints/` scanner. The architecture (d_model, n_layers, n_heads) is read at load time from the graph — no metadata file required. Without a model, the AI surface falls back to the engine's heuristic evaluator and games still play.

## Card-grid contract

The right pane on `/play` is a 4-column × 13-row CSS grid (suits ♠♥♣♦ left→right, ranks A→2 top→bottom). Clicking a `legal` cell records the active player's card for that turn — engine validates legality. The same component drives `/hand-entry` (toggle cells until count = `cards_dealt`).

Card encoding: `card_index = suit * 13 + rank`. Suits ♠=0 ♥=1 ♣=2 ♦=3, ranks 2=0 … A=12.

## Trump overrides

The round-progress strip at the top of `/play`, `/hand-entry`, and `/round-summary` doubles as a trump editor. Click **Edit trumps** → digits **1**–**5** set the cursor round's trump (♠ ♥ ♣ ♦ NT) and advance to the next round. Past rounds are locked. **Enter** saves; **Esc** cancels. Manual overrides survive `advance_round` and are persisted with the session.

## Keyboard shortcuts

Press **?** at any time to bring up the live keyboard-help overlay, generated from a single [`KEYMAP`](src/lib/keymap.ts) constant so docs can't drift from behavior.

| Context | Keys |
|---|---|
| Global | `?` toggle help · `Esc` cancel/close |
| Hand entry | `2`–`9` `T J Q K A` pre-arm rank · `S H C D` toggle cell at armed rank |
| Bidding | `0`–`9` record bid · `Enter` accept AI's recommended bid |
| Trick play | `Enter` play recommended card · `E` cycle eval display |
| Trump editor | `1`-`5` set trump and advance · `←`/`→` move cursor · `Enter` save · `Esc` cancel |
| Round summary | `N` continue to next round |

## Architecture

Authoritative game state is `BlobState` in `blob-engine`, wrapped in `Mutex<GameSession>` inside `tauri::State`. The frontend never mutates locally — every action is a Tauri command round-trip that returns a fresh `SessionSnapshot`. Type-safe IPC via `tauri-specta`; `src/lib/api.ts` is regenerated on every dev launch.

Sessions persist as JSON to `~/.blobmaster/sessions/<unix_secs>.json` on every round end and on app close. Setup-screen form values live at `~/.blobmaster/settings.json`.

## Layout cheat sheet

```
src-tauri/src/
  commands.rs    — every #[tauri::command] entry point
  session.rs     — GameSession (held in tauri::State)
  types.rs       — wire types crossing the IPC bridge

src/
  routes/
    setup/         pre-game form + Resume saved sessions
    hand-entry/    enter the human's hand for the round
    play/          three-pane play screen (Players · MyHand · CardGrid)
    round-summary/ scoring & continue
    end/           final standings, export log
  lib/
    api.ts                 (generated)
    cardUtils.ts           card encoding helpers
    evalUtils.ts           eval display helpers
    keymap.ts              shared keyboard-shortcut catalog
    components/
      CardGrid.svelte
      PlayersPanel.svelte
      MyHandPanel.svelte
      BiddingKeypad.svelte
      RoundProgressStrip.svelte (also the trump editor)
      KeymapOverlay.svelte
      Toast.svelte
    stores/
      session.ts            current SessionSnapshot
      toast.ts              transient notifications
      trumpEditing.ts       global flag claimed by the trump editor
```

See [../gui-development-plan.md](../gui-development-plan.md) for the full session-by-session record.
