//! Helper: save a random-init P and V checkpoint to the directory in
//! `BLOB_SAVE_CKPT_DIR`. Only runs when the env var is set.
//!
//! `cargo test -p blob-nn --release --test save_random_checkpoint -- --ignored save_random_init`

use blob_nn::model::{PolicyNet, ValueNet};
use blob_nn::train::{save_checkpoint, CheckpointMeta};
use tch::{nn::VarStore, Device};

#[test]
#[ignore]
fn save_random_init() {
    let Ok(dir) = std::env::var("BLOB_SAVE_CKPT_DIR") else {
        eprintln!("BLOB_SAVE_CKPT_DIR unset; skipping");
        return;
    };
    tch::manual_seed(0);
    let (p, v) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
    let _ = (PolicyNet::new(&p.root()), ValueNet::new(&v.root()));
    save_checkpoint(&dir, &p, &v, CheckpointMeta { learner_step: 0 }).expect("save");
    eprintln!("[save_random_init] wrote checkpoint to {dir}");
}
