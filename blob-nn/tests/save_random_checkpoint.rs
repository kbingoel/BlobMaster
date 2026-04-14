//! Helper: save a random-init BlobNet to a directory specified by
//! `BLOB_SAVE_CKPT_DIR`. Only runs when the env var is set.
//!
//! `cargo test -p blob-nn --release --test save_random_checkpoint -- --ignored save_random_init`

use std::path::PathBuf;

use blob_nn::model::BlobNet;
use blob_nn::train::save_checkpoint;
use tch::{nn::VarStore, Device};

#[test]
#[ignore]
fn save_random_init() {
    let Ok(dir) = std::env::var("BLOB_SAVE_CKPT_DIR") else {
        eprintln!("BLOB_SAVE_CKPT_DIR unset; skipping");
        return;
    };
    tch::manual_seed(0);
    let vs = VarStore::new(Device::Cpu);
    let _ = BlobNet::new(&vs.root());
    save_checkpoint(&vs, 0, PathBuf::from(&dir)).expect("save");
    eprintln!("[save_random_init] wrote checkpoint to {dir}");
}
