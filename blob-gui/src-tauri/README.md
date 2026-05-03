# blob-gui — Tauri core

Rust core for the BlobMaster GUI. Hosts the engine bridge as
`#[tauri::command]` handlers and ships authoritative game state in
`tauri::State`. The SvelteKit frontend sits in `../src/`.

## Type-safe IPC contract (Session 9.2)

The bridge is type-checked end-to-end with [`tauri-specta`]. Every command
body carries both `#[tauri::command]` and `#[specta::specta]`; argument and
return types derive `specta::Type` alongside `Serialize` / `Deserialize`.

The single source of truth is [`build_specta_builder()`](src/lib.rs), which
both the live binary and the `export-bindings` helper consume. From it we
emit `../src/lib/api.ts` containing:

- a typed `commands` object (`commands.newGame(config)`, etc.) — replaces
  raw `invoke('new_game', { config })` calls,
- type aliases for every IPC-crossing struct (`SessionSnapshot`,
  `GameConfig`, `AiSuggestion`, `GuiError`, …).

### Regenerating bindings

Two paths, both equivalent:

```sh
# from blob-gui/, after any signature change:
bun run gen-types

# or, automatically — debug builds re-export on every launch:
bun tauri dev
```

Release builds skip the export so the file checked into git is what ships.

### Pinned versions

Specta v2 is still in release-candidate. `2.0.0-rc.24` (current latest as of
2026-05) requires nightly-only `fmt::from_fn` / `TypeId::of` const features
that haven't reached stable Rust 1.90. We pin to the last set known to build
on stable:

```toml
specta            = "=2.0.0-rc.22"
specta-typescript = "=0.0.9"
tauri-specta      = "=2.0.0-rc.21"
```

Bump only after confirming compilation on the project's pinned toolchain.

### Adding a new command

1. Write the function in [`commands.rs`](src/commands.rs):
   ```rust
   #[tauri::command]
   #[specta::specta]
   pub fn my_command(state: State<'_, AppState>, arg: u32) -> GuiResult<u32> { ... }
   ```
2. Add it to `collect_commands![...]` in [`lib.rs`](src/lib.rs).
3. Run `bun run gen-types` (or just `bun tauri dev`).
4. Use it from Svelte: `await commands.myCommand(42)` (snake_case → camelCase
   on the wire).

Argument and return types must `derive(specta::Type, Serialize, Deserialize)`.
For `u64`, configure `BigIntExportBehavior::Number` on the exporter — already
set in both [`bin/export_bindings.rs`](src/bin/export_bindings.rs) and the
debug-mode export in `lib.rs`.

### Errors

Commands that can fail return `GuiResult<T> = Result<T, GuiError>`. `GuiError`
is a tagged enum (see [`types.rs`](src/types.rs)) so the frontend can
discriminate on `kind` instead of parsing strings. `?` from `std::io::Error`
maps automatically.

## Persisted state (Session 9.3)

Two JSON files under the user's home directory survive across launches:

- `~/.blobmaster/settings.json` — setup-screen form values (`AppSettings`).
  Read on mount via `commands.loadAppSettings()`, written by
  `commands.saveAppSettings(form)` when **Start Game** is clicked.
- `~/.blobmaster/recents.json` — recently loaded model paths, most-recent
  first, capped at 10 entries. `commands.loadModel(path)` auto-bumps;
  `commands.listRecentModels()` reads (and silently drops missing files).

The setup screen sources its model picker from three places: a Tauri file
dialog (filtered to `*.onnx`), the recents list, and a scan of
`<workspace>/checkpoints/`.

## Layout

```
src-tauri/
├── Cargo.toml
├── tauri.conf.json
├── build.rs          # tauri_build::build()
└── src/
    ├── main.rs       # binary entry — defers to lib::run()
    ├── lib.rs        # specta builder + tauri::Builder
    ├── commands.rs   # #[tauri::command] surface
    ├── session.rs    # GameSession + snapshot construction
    ├── types.rs      # IPC types crossing the bridge
    └── bin/
        └── export_bindings.rs  # one-shot api.ts generator
```

## Rebuild + run

```sh
# from blob-gui/:
bun tauri dev          # dev mode (regenerates api.ts each launch)
bun tauri build        # release MSI/EXE (Session 9.9)
cargo check -p blob-gui
```
