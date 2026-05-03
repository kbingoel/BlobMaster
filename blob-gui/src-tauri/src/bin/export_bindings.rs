//! One-shot bindings generator. Run via:
//!
//! ```text
//! cargo run -p blob-gui --bin export-bindings
//! ```
//!
//! Writes `blob-gui/src/lib/api.ts` from the same `Builder` the live binary
//! uses, so a fresh checkout can run `bun run check` without first having
//! to launch `bun tauri dev`.

use specta_typescript::{BigIntExportBehavior, Typescript};

fn main() {
    let builder = blob_gui_lib::build_specta_builder();
    let path = "../src/lib/api.ts";
    let cfg = Typescript::default().bigint(BigIntExportBehavior::Number);
    builder
        .export(cfg, path)
        .expect("failed to export typescript bindings");
    println!("wrote {path}");
}
