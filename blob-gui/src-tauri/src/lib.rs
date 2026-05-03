//! Tauri shell entrypoint. Builder construction is split into
//! [`build_specta_builder`] so the `export-bindings` binary can drive the
//! same configuration.

pub mod commands;
pub mod session;
pub mod types;

use std::sync::Mutex;

use tauri_specta::{collect_commands, Builder as SpectaBuilder};

use crate::commands::{
    engine_version, list_models, load_model, load_session, new_game, record_card_played,
    request_ai_suggestion, save_session, set_human_hand, submit_bid, undo_last_event,
    update_engine_settings, AppState,
};

/// Build the tauri-specta `Builder` with every command registered. Shared
/// between `run()` and `src/bin/export_bindings.rs` so the generated
/// `api.ts` always matches what the live binary serves.
pub fn build_specta_builder() -> SpectaBuilder<tauri::Wry> {
    SpectaBuilder::<tauri::Wry>::new().commands(collect_commands![
        engine_version,
        list_models,
        load_model,
        new_game,
        set_human_hand,
        submit_bid,
        record_card_played,
        request_ai_suggestion,
        update_engine_settings,
        undo_last_event,
        save_session,
        load_session,
    ])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let specta_builder = build_specta_builder();

    // Regenerate bindings on every dev launch so frontend types track the
    // Rust signatures without a manual step. Release builds skip this —
    // the file checked into the repo is what ships.
    #[cfg(debug_assertions)]
    {
        use specta_typescript::{BigIntExportBehavior, Typescript};
        let cfg = Typescript::default().bigint(BigIntExportBehavior::Number);
        if let Err(e) = specta_builder.export(cfg, "../src/lib/api.ts") {
            eprintln!("warning: failed to export TS bindings: {e}");
        }
    }

    let app_state: AppState = Mutex::new(None);

    tauri::Builder::default()
        .manage(app_state)
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(specta_builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
