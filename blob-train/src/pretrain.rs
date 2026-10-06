//! `pretrain`: the supervised warm start (gen-2.md §6 Phase 4).
//!
//! 1. Play `data.rounds` teacher rounds (rule bot 2, `blob_engine::teacher`)
//!    into a replay buffer and split it by round into training and
//!    validation.
//! 2. Train P and V on the training rounds, P on decisions with a choice
//!    and V on every state. Loader threads build batches while the GPU
//!    trains.
//! 3. Log to `metrics.jsonl`: a training row every `log.every` steps, and a
//!    held-out row every `log.eval_every` steps (validation against an
//!    equally large training sample, dropout off).
//! 4. At the end, save the checkpoint, run the held-out measurement on
//!    every validation example (`held_out.json`, with G1) and export the
//!    model directory.
//!
//! Run directory: `config.toml` (every default filled in), `metrics.jsonl`,
//! `checkpoint/`, `held_out.json`, `model/`. A `STOP` file makes the run
//! save and exit at its next training row. `--resume` continues from the
//! checkpoint with the run's own config: the teacher data replays from the
//! seed, the optimizers start afresh.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::time::Instant;

use blob_engine::{fill_buffer, ReplayBuffer};
use blob_nn::learner::{is_forced, is_validation_round, Learner, StepBatches};
use rand::seq::index;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};

use crate::config::PretrainConfig;
use crate::export::export;

/// G1 (gen-2.md §7): V's correlation with the actual outcome on held-out
/// rounds, and its RMSE where the deal decides the outcome.
pub const G1_MIN_CORRELATION: f64 = 0.7;
pub const G1_MAX_ONE_CARD_RMSE: f64 = 0.05;

pub const CONFIG_FILE: &str = "config.toml";

struct Run {
    dir: PathBuf,
}

impl Run {
    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
    fn checkpoint(&self) -> PathBuf {
        self.path("checkpoint")
    }
}

/// Examples to measure on: validation slots and an equally large sample of
/// training slots, for P (unforced decisions) and V (every state).
struct EvalSets {
    valid_p: Vec<usize>,
    train_p: Vec<usize>,
    valid_v: Vec<usize>,
    train_v: Vec<usize>,
}

/// The training and validation slots of a buffer.
struct Split {
    train_p: Vec<usize>,
    train_v: Vec<usize>,
    valid_p: Vec<usize>,
    valid_v: Vec<usize>,
}

impl Split {
    fn new(buf: &ReplayBuffer, fraction: f64) -> Self {
        let mut s = Split { train_p: vec![], train_v: vec![], valid_p: vec![], valid_v: vec![] };
        for i in 0..buf.len() {
            let (p, v) = if is_validation_round(buf.round_id(i), fraction) {
                (&mut s.valid_p, &mut s.valid_v)
            } else {
                (&mut s.train_p, &mut s.train_v)
            };
            v.push(i);
            if !is_forced(buf.state(i)) {
                p.push(i);
            }
        }
        s
    }

    /// At most `n` validation slots per network (all of them if `n` is
    /// `None`), with as many training slots, drawn with `rng`.
    fn eval_sets(&self, n: Option<usize>, rng: &mut Xoshiro256PlusPlus) -> EvalSets {
        let mut pick = |slots: &[usize], k: usize| -> Vec<usize> {
            let mut out: Vec<usize> =
                index::sample(rng, slots.len(), k.min(slots.len())).into_iter().map(|i| slots[i]).collect();
            out.sort_unstable();
            out
        };
        let valid_p = pick(&self.valid_p, n.unwrap_or(usize::MAX));
        let valid_v = pick(&self.valid_v, n.unwrap_or(usize::MAX));
        EvalSets {
            train_p: pick(&self.train_p, valid_p.len()),
            train_v: pick(&self.train_v, valid_v.len()),
            valid_p,
            valid_v,
        }
    }
}

/// Largest round the mix can deal, in decisions.
fn max_decisions_per_round(cfg: &PretrainConfig) -> usize {
    let mix = &cfg.teacher.mix;
    mix.players.iter().map(|&n| n as usize * (1 + mix.start_cards as usize)).max().unwrap_or(1)
}

fn held_out(learner: &Learner, buf: &ReplayBuffer, sets: &EvalSets, chunk: usize) -> Value {
    json!({
        "validation": {
            "policy": learner.policy_held_out(buf, &sets.valid_p, chunk),
            "value": learner.value_held_out(buf, &sets.valid_v, chunk),
        },
        "train_sample": {
            "policy": learner.policy_held_out(buf, &sets.train_p, chunk),
            "value": learner.value_held_out(buf, &sets.train_v, chunk),
        },
    })
}

fn short(report: &Value) -> String {
    let f = |v: &Value| v.as_f64().map_or("NaN".to_string(), |x| format!("{x:.4}"));
    let (vp, vv) = (&report["validation"]["policy"], &report["validation"]["value"]);
    let (tp, tv) = (&report["train_sample"]["policy"], &report["train_sample"]["value"]);
    format!(
        "held out (valid / train): P bid {} / {}, play {} / {}, agree {} / {}; V mse {} / {} (var {}), corr {} / {}, 1-card mse {}",
        f(&vp["bid_loss"]), f(&tp["bid_loss"]), f(&vp["play_loss"]), f(&tp["play_loss"]),
        f(&vp["play_agreement"]), f(&tp["play_agreement"]), f(&vv["mse"]), f(&tv["mse"]),
        f(&vv["variance"]), f(&vv["correlation"]), f(&tv["correlation"]), f(&vv["one_card_mse"]),
    )
}

/// G1 on V's validation measurement (a serialized `ValueHeldOut`).
fn g1(v: &Value) -> Value {
    let corr = v["correlation"].as_f64().unwrap_or(f64::NAN);
    let rmse = v["one_card_mse"].as_f64().unwrap_or(f64::NAN).sqrt();
    json!({
        "correlation": corr,
        "one_card_rmse": rmse,
        "pass": corr > G1_MIN_CORRELATION && rmse < G1_MAX_ONE_CARD_RMSE,
        "criteria": format!("correlation > {G1_MIN_CORRELATION}, 1-card RMSE < {G1_MAX_ONE_CARD_RMSE}"),
    })
}

fn write_row(file: &mut File, row: &Value) -> Result<(), String> {
    writeln!(file, "{row}").and_then(|_| file.flush()).map_err(|e| format!("metrics.jsonl: {e}"))
}

/// Run (or, with `resume`, continue) a warm start in `dir`.
pub fn pretrain(cfg: PretrainConfig, dir: &Path, resume: bool) -> Result<(), String> {
    let started = Instant::now();
    let run = Run { dir: dir.to_path_buf() };
    if resume {
        if !run.checkpoint().is_dir() {
            return Err(format!("{}: no checkpoint to resume", run.checkpoint().display()));
        }
    } else {
        if run.checkpoint().exists() {
            return Err(format!("{} already holds a run; continue it with --resume", dir.display()));
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        std::fs::write(run.path(CONFIG_FILE), cfg.to_toml()).map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    }

    tch::manual_seed(cfg.data.seed as i64);
    let mut learner = Learner::new(&cfg.learner)?;
    if resume {
        learner.resume(&run.checkpoint())?;
        eprintln!("[pretrain] resumed at learner step {}", learner.step);
    }
    eprintln!("[pretrain] device {:?}", learner.device);

    let threads = match cfg.data.threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
        n => n,
    };
    let t = Instant::now();
    let mut buf = ReplayBuffer::new(cfg.data.rounds as usize * max_decisions_per_round(&cfg));
    fill_buffer(&mut buf, &cfg.teacher, cfg.data.rounds, cfg.data.seed, threads);
    let split = Split::new(&buf, cfg.data.validation_fraction);
    eprintln!(
        "[pretrain] teacher: {} rounds, {} decisions in {:.1} s; training {} (P {}), validation {} (P {})",
        buf.rounds_pushed(),
        buf.len(),
        t.elapsed().as_secs_f64(),
        split.train_v.len(),
        split.train_p.len(),
        split.valid_v.len(),
        split.valid_p.len(),
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.data.seed ^ 0xE7A1_5E75);
    let periodic = split.eval_sets(Some(cfg.log.eval_examples), &mut rng);
    let chunk = 2 * cfg.learner.batch_size;

    let mut metrics = OpenOptions::new()
        .create(true)
        .append(true)
        .open(run.path("metrics.jsonl"))
        .map_err(|e| format!("metrics.jsonl: {e}"))?;

    let (tx, rx) = sync_channel::<StepBatches>(2 * cfg.data.loader_threads);
    let stopped = std::thread::scope(|sc| {
        for t in 0..cfg.data.loader_threads {
            let (tx, buf, split, cfg) = (tx.clone(), &buf, &split, &cfg);
            sc.spawn(move || {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.data.seed.wrapping_add(1 + t as u64));
                let (n, augment) = (cfg.learner.batch_size, cfg.learner.augment);
                loop {
                    let p = buf.sample_batch_from(&split.train_p, n, &mut rng, augment);
                    let v = buf.sample_batch_from(&split.train_v, n, &mut rng, augment);
                    if tx.send(StepBatches::build(&p, &v, Device::Cpu)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let r = train(&cfg, &run, &mut learner, &rx, &buf, &periodic, chunk, &mut metrics, started);
        drop(rx); // loaders stop at their next send
        r
    })?;
    learner.save(&run.checkpoint())?;
    if stopped {
        eprintln!("[pretrain] STOP: saved the checkpoint at step {}; continue with --resume", learner.step);
        return Ok(());
    }

    eprintln!("[pretrain] held-out measurement on every validation example");
    let all = split.eval_sets(None, &mut rng);
    let report = held_out(&learner, &buf, &all, chunk);
    let out = json!({
        "learner_step": learner.step,
        "secs": started.elapsed().as_secs_f64(),
        "g1": g1(&report["validation"]["value"]),
        "validation": report["validation"],
        "train_sample": report["train_sample"],
    });
    let text = serde_json::to_string_pretty(&out).expect("report serializes") + "\n";
    std::fs::write(run.path("held_out.json"), text).map_err(|e| format!("held_out.json: {e}"))?;
    eprintln!("[pretrain] {}", short(&report));
    eprintln!("[pretrain] G1: {}", out["g1"]);

    export(Some(&run.checkpoint()), &run.path("model"), false)?;
    eprintln!("[pretrain] done in {:.0} s: model in {}", started.elapsed().as_secs_f64(), run.path("model").display());
    Ok(())
}

/// The training loop. Returns whether a STOP file ended it early.
#[allow(clippy::too_many_arguments)]
fn train(
    cfg: &PretrainConfig,
    run: &Run,
    learner: &mut Learner,
    rx: &Receiver<StepBatches>,
    buf: &ReplayBuffer,
    periodic: &EvalSets,
    chunk: usize,
    metrics: &mut File,
    started: Instant,
) -> Result<bool, String> {
    let total = cfg.learner.steps;
    let device = learner.device;
    let zero = || Tensor::zeros([], (Kind::Float, device));
    let (mut p_sum, mut v_sum, mut n) = (zero(), zero(), 0u32);
    let mut interval = Instant::now();
    // Seconds of the interval spent waiting for batches, and measuring or
    // saving (left out of steps/s).
    let (mut waited, mut paused) = (0.0f64, 0.0f64);
    while learner.step < total {
        let lr = learner.lr();
        let w = Instant::now();
        let batches = rx.recv().map_err(|_| "batch loaders stopped".to_string())?;
        waited += w.elapsed().as_secs_f64();
        let (p, v) = learner.train_on(&batches.to_device(device));
        p_sum += p.unwrap_or_else(zero);
        v_sum += v.unwrap_or_else(zero);
        n += 1;
        let step = learner.step;

        if step.is_multiple_of(cfg.log.every) || step == total {
            let secs = interval.elapsed().as_secs_f64() - paused;
            let row = json!({
                "kind": "train",
                "step": step,
                "secs": started.elapsed().as_secs_f64(),
                "lr": lr,
                "policy_loss": p_sum.double_value(&[]) / n as f64,
                "value_loss": v_sum.double_value(&[]) / n as f64,
                "steps_per_sec": n as f64 / secs,
                "loader_wait": waited / secs,
            });
            write_row(metrics, &row)?;
            eprintln!(
                "[pretrain] step {step}/{total} lr {lr:.2e}: P {:.4}, V {:.4}; {:.1} steps/s, waiting on batches {:.0}%",
                row["policy_loss"].as_f64().unwrap_or(f64::NAN),
                row["value_loss"].as_f64().unwrap_or(f64::NAN),
                n as f64 / secs,
                100.0 * waited / secs,
            );
            (p_sum, v_sum, n, waited, paused, interval) = (zero(), zero(), 0, 0.0, 0.0, Instant::now());
            let stop = run.path("STOP");
            if stop.exists() {
                let _ = std::fs::remove_file(&stop);
                return Ok(true);
            }
        }
        let pause = Instant::now();
        if step < total && cfg.log.eval_every > 0 && step.is_multiple_of(cfg.log.eval_every) {
            let report = held_out(learner, buf, periodic, chunk);
            eprintln!("[pretrain] step {step}: {}", short(&report));
            let row = json!({
                "kind": "held_out",
                "step": step,
                "secs": started.elapsed().as_secs_f64(),
                "validation": report["validation"],
                "train_sample": report["train_sample"],
            });
            write_row(metrics, &row)?;
        }
        if step < total && cfg.log.checkpoint_every > 0 && step.is_multiple_of(cfg.log.checkpoint_every) {
            learner.save(&run.checkpoint())?;
        }
        paused += pause.elapsed().as_secs_f64();
    }
    Ok(false)
}
