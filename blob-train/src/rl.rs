//! `train`: async self-play RL (gen-2.md §5.6, §6 Phase 5).
//!
//! Two processes. ONNX Runtime and libtorch don't share one: an in-process
//! search bench crashed in ONNX Runtime while the learner ran.
//! - **The actor process** (`blobmaster selfplay`, ONNX only; this driver
//!   starts it, restarts it if it dies, and stops it by closing its stdin):
//!   `run.actors` threads play single rounds with P + V search and write
//!   them to `replay/` in chunks. It plays the model named in `model.json`.
//!   With `rollout.actors` it also plays rollout rounds (policy iteration
//!   by rollouts, `blob_engine::rollout`) into `replay-pi/`; P then trains
//!   on those instead of on search targets, and `run.actors` may be 0.
//! - **This process** (libtorch):
//!   - **ingest** tails `replay/` and splits the rounds by round id into the
//!     training and validation buffers; with `value_stream.actors`, it also
//!     reads (and deletes) `replay-v/`, the rounds of P alone for V, into
//!     their own pair of buffers; with `rollout.actors`, `replay-pi/` into
//!     the rollout buffers (and each valued move's next state into V's);
//!   - **loaders** build batches from the training buffer, P's from the
//!     decisions with a choice;
//!   - **learner** (the main thread, GPU): one P and one V update per step,
//!     held back by the replay-ratio governor; while it waits, V-only updates
//!     on the rounds of P alone (their own ratio cap); held-out rows;
//!     publishes;
//!   - **publisher** exports a published checkpoint to a model directory
//!     (Python) and points `model.json` at it;
//!   - **evaluator** runs `blobmaster bench`: network-only benches of every
//!     published model beside the actors, and a search bench every
//!     `eval.search_every_hours` with the actor process frozen (SIGSTOP);
//!   - **monitor**: control files, the running clock, the actor process,
//!     `status.md` and the self-play rows.
//!
//! Run directory:
//! - `config.toml` (every default filled in), `state.json` (running time,
//!   the actors' model), `metrics.jsonl`, `status.md` (rewritten every
//!   `log.status_secs`: the run at a glance), `train.log` is yours to
//!   redirect to, `selfplay.log` (the actor process);
//! - `checkpoint/` (the resume point), `models/step-NNNNNN/` (`checkpoint/`
//!   and `model/` of each publish), `replay/chunk-*.bin`, `bench/`;
//! - `selfplay.json`, `model.json`: the actor process's settings and model.
//!
//! Control files in the run directory:
//! - `STOP`: the actors finish their rounds, the learner saves, the run
//!   exits; `--resume` continues (the buffer reloads from `replay/`).
//! - `PAUSE`: actors and learner idle until it is removed; the clock stops.
//! - `FINISH`: end now, as at `run.hours`: publish, then the final benches.
//!
//! A kill loses at most the steps since the last checkpoint
//! (`log.checkpoint_minutes`) and the rounds of the chunk being filled.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use blob_engine::bench::DEFAULT_SEED;
use blob_engine::mcts::OneCardBids;
use blob_engine::replay::{Decision, SparsePolicy};
use blob_engine::rollout::{PiReplay, RolloutStats};
use blob_engine::selfplay::{
    chunk_files, chunk_files_in, read_chunk, read_json, read_rollout_chunk, write_json, ActorsConfig, ModelPointer,
    ACTORS_CONFIG_FILE, MODEL_POINTER_FILE, REPLAY_DIR, ROLLOUT_REPLAY_DIR, SELFPLAY_BUCKETS, VALUE_REPLAY_DIR,
};
use blob_engine::{fill_buffer, BlobState, GamePhase, ReplayBuffer, SelfPlayStats, SharedReplay, TeacherConfig};
use blob_nn::learner::{
    bid_policy_batch, is_forced, is_validation_round, play_policy_batch, value_batch, FrozenPolicy, Learner, PiStep,
    StepBatches,
};
use blob_nn::train::{policy_probs, recover_checkpoint, ValueBatch};
use rand::seq::index;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};

use crate::config::TrainConfig;
use crate::export::export_with_threads;
use crate::pretrain::CONFIG_FILE;

const STATE_FILE: &str = "state.json";
const STATUS_FILE: &str = "status.md";
const METRICS_FILE: &str = "metrics.jsonl";
const PROBE_SEED: u64 = 0x9E0B_E5E7;
/// Python threads of an export beside the actors.
const EXPORT_THREADS: usize = 4;
/// Restarts of the actor process before the run stops.
const MAX_ACTOR_RESTARTS: u64 = 20;

/// What survives a restart besides the checkpoint and the buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunState {
    /// Running seconds (pauses excluded), across resumes.
    active_secs: f64,
    /// The actors' model directory and the learner step it holds.
    model: PathBuf,
    model_step: u64,
    /// Running seconds at which the next mid-run search bench is due.
    next_search_secs: f64,
    finished: bool,
}

#[derive(Default)]
struct Status {
    phase: String,
    learner: Value,
    held_out: Value,
    probes: Vec<Value>,
    selfplay: Value,
    benches: Vec<Value>,
    events: Vec<String>,
}

struct Shared {
    cfg: TrainConfig,
    dir: PathBuf,
    started: Instant,
    /// The learner winds down and the actor process is stopped.
    stop: AtomicBool,
    /// After the wind-down: the final publish and benches.
    finish: AtomicBool,
    /// A `PAUSE` file is present.
    paused: AtomicBool,
    /// The evaluator wants every core: the actor process is frozen.
    hold: AtomicBool,
    /// The actor process, whether it is frozen (SIGSTOP), its restarts.
    child: Mutex<Option<Child>>,
    frozen: Mutex<bool>,
    restarts: AtomicU64,
    /// Ingest stops tailing `replay/` (after a last sweep).
    ingest_done: AtomicBool,
    state: Mutex<RunState>,
    train: SharedReplay,
    valid: SharedReplay,
    /// Training examples produced since the run began (the governor's
    /// denominator).
    produced: AtomicU64,
    rounds: AtomicU64,
    examples: AtomicU64,
    /// Self-play statistics since the last self-play row.
    window: Mutex<SelfPlayStats>,
    /// Rounds of P alone for V (`value_stream`): their buffers, training
    /// states produced, rounds and states read, V-only samples trained,
    /// and the rounds and states read since the last self-play row.
    vtrain: SharedReplay,
    vvalid: SharedReplay,
    vproduced: AtomicU64,
    vrounds: AtomicU64,
    vsamples: AtomicU64,
    vwindow: Mutex<(u64, u64)>,
    /// Rollout samples (`rollout`): their buffers, training samples
    /// produced, rounds read, and their statistics since the last
    /// self-play row.
    rtrain: RwLock<PiReplay>,
    rvalid: RwLock<PiReplay>,
    rproduced: AtomicU64,
    rrounds: AtomicU64,
    rwindow: Mutex<RolloutStats>,
    /// The fixed V check (`eval.value_rounds_from`), if any.
    fixed: Option<FixedRounds>,
    /// The learner's V update count at the last publish: V-only updates
    /// change V without a learner step, and the final publish exports them.
    published_value_updates: AtomicU64,
    metrics: Mutex<File>,
    status: Mutex<Status>,
}

/// Seconds east of UTC, from `date +%z`; 0 if that fails.
fn utc_offset() -> i64 {
    static OFFSET: OnceLock<i64> = OnceLock::new();
    *OFFSET.get_or_init(|| {
        let out = Command::new("date").arg("+%z").output().ok();
        let s = out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        if s.len() != 5 {
            return 0;
        }
        let sign = if s.starts_with('-') { -1 } else { 1 };
        let (h, m) = (s[1..3].parse::<i64>().unwrap_or(0), s[3..5].parse::<i64>().unwrap_or(0));
        sign * (h * 3600 + m * 60)
    })
}

/// Local wall-clock time, `HH:MM`.
fn clock() -> String {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0) + utc_offset();
    let day = t.rem_euclid(86_400);
    format!("{:02}:{:02}", day / 3600, (day % 3600) / 60)
}

fn unix_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl Shared {
    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// P trains on rollout samples, not on search targets.
    fn pi_on(&self) -> bool {
        self.cfg.rollout.actors > 0
    }

    /// V has its own buffers: the V stream, or the rollouts' next states.
    fn v_on(&self) -> bool {
        self.cfg.value_stream.actors > 0 || (self.pi_on() && self.cfg.rollout.value_children)
    }

    fn secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    fn active_hours(&self) -> f64 {
        self.state.lock().unwrap().active_secs / 3600.0
    }

    fn log(&self, row: &Value) {
        let mut f = self.metrics.lock().unwrap();
        let _ = writeln!(f, "{row}").and_then(|_| f.flush());
    }

    fn event(&self, msg: impl Into<String>) {
        let msg = msg.into();
        eprintln!("[train {}] {msg}", clock());
        self.log(&json!({ "kind": "event", "secs": self.secs(), "unix": unix_secs(), "message": msg }));
        let mut st = self.status.lock().unwrap();
        st.events.push(format!("{} {msg}", clock()));
        let n = st.events.len();
        if n > 15 {
            st.events.drain(..n - 15);
        }
    }

    fn set_phase(&self, phase: &str) {
        self.status.lock().unwrap().phase = phase.to_string();
    }

    fn save_state(&self) {
        let st = self.state.lock().unwrap().clone();
        if let Err(e) = write_json(&self.path(STATE_FILE), &st) {
            eprintln!("[train] {STATE_FILE}: {e}");
        }
    }
}

// ---- the actor process ----------------------------------------------------------

/// `blobmaster`, built beside this binary.
fn blobmaster() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("blobmaster")))
        .unwrap_or_else(|| PathBuf::from("blobmaster"))
}

/// A `blobmaster` command without libtorch in its environment.
fn blobmaster_cmd() -> Command {
    let mut cmd = Command::new(blobmaster());
    cmd.env_remove("LD_PRELOAD").env_remove("LD_LIBRARY_PATH");
    cmd
}

fn spawn_actors(sh: &Shared) -> Result<Child, String> {
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(sh.path("selfplay.log"))
        .map_err(|e| format!("selfplay.log: {e}"))?;
    let err = log.try_clone().map_err(|e| format!("selfplay.log: {e}"))?;
    blobmaster_cmd()
        .arg("selfplay")
        .arg("--run")
        .arg(&sh.dir)
        .stdin(Stdio::piped())
        .stdout(log)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("starting {} selfplay: {e}", blobmaster().display()))
}

/// Freeze (SIGSTOP) or thaw (SIGCONT) the actor process to match
/// `paused || hold`.
fn sync_frozen(sh: &Shared) {
    // Never freeze it once stopping: it must read the end of its input.
    let want = !sh.stop.load(SeqCst) && (sh.paused.load(SeqCst) || sh.hold.load(SeqCst));
    let mut frozen = sh.frozen.lock().unwrap();
    if *frozen == want {
        return;
    }
    if let Some(c) = sh.child.lock().unwrap().as_ref() {
        let sig = if want { "-STOP" } else { "-CONT" };
        let _ = Command::new("kill").arg(sig).arg(c.id().to_string()).status();
    }
    *frozen = want;
}

/// Restart the actor process if it exited while the run goes on.
fn supervise_actors(sh: &Shared) {
    if sh.stop.load(SeqCst) {
        return;
    }
    let exited = {
        let mut c = sh.child.lock().unwrap();
        match c.as_mut().map(|c| c.try_wait()) {
            Some(Ok(Some(status))) => Some(status.to_string()),
            None => Some("not running".to_string()),
            _ => None,
        }
    };
    let Some(status) = exited else { return };
    let n = sh.restarts.fetch_add(1, SeqCst) + 1;
    if n > MAX_ACTOR_RESTARTS {
        sh.event(format!("actor process {status}; {MAX_ACTOR_RESTARTS} restarts used: stopping the run"));
        sh.stop.store(true, SeqCst);
        return;
    }
    sh.event(format!("actor process {status}; restart {n} (selfplay.log has its output)"));
    match spawn_actors(sh) {
        Ok(child) => {
            *sh.child.lock().unwrap() = Some(child);
            *sh.frozen.lock().unwrap() = false;
            sync_frozen(sh);
        }
        Err(e) => sh.event(e),
    }
}

/// Close the actor process's stdin and wait for it to finish its rounds
/// and write its last chunk; kill it after `timeout`.
fn stop_actors(sh: &Shared, timeout: Duration) {
    *sh.frozen.lock().unwrap() = true;
    sh.hold.store(false, SeqCst);
    sh.paused.store(false, SeqCst);
    sync_frozen(sh);
    let Some(mut child) = sh.child.lock().unwrap().take() else { return };
    drop(child.stdin.take());
    let t = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if t.elapsed() < timeout => thread::sleep(Duration::from_millis(200)),
            _ => {
                sh.event("actor process didn't stop in time: killed");
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
        }
    }
}

// ---- ingest -----------------------------------------------------------------------

/// Load the chunks of `replay/` from round `next_id` on into the buffers.
/// Returns the chunks read.
fn ingest_new(sh: &Shared, next_id: &mut u64) -> usize {
    let fraction = sh.cfg.replay.validation_fraction;
    let mut read = 0;
    for (first, path) in chunk_files(&sh.dir) {
        if first < *next_id {
            continue;
        }
        let chunk = match read_chunk(&path) {
            Ok(c) => c,
            Err(e) => {
                sh.event(format!("skipping {e}"));
                *next_id = first + 1;
                continue;
            }
        };
        for r in &chunk.rounds {
            let n = r.decisions.len() as u64;
            if is_validation_round(r.id, fraction) {
                sh.valid.push_round(&r.decisions, &r.end);
            } else {
                sh.train.push_round(&r.decisions, &r.end);
                sh.produced.fetch_add(n, SeqCst);
            }
            sh.rounds.fetch_add(1, SeqCst);
            sh.examples.fetch_add(n, SeqCst);
            *next_id = (*next_id).max(r.id + 1);
        }
        sh.window.lock().unwrap().merge(&chunk.stats);
        read += 1;
    }
    read
}

/// Load the chunks of `replay-v/` into the V stream's buffers and delete
/// them. Returns the chunks read.
fn ingest_value(sh: &Shared) -> usize {
    let fraction = sh.cfg.replay.validation_fraction;
    let mut read = 0;
    for (_, path) in chunk_files_in(&sh.dir, VALUE_REPLAY_DIR) {
        let chunk = read_chunk(&path);
        let _ = std::fs::remove_file(&path);
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                sh.event(format!("skipping {e}"));
                continue;
            }
        };
        let (mut rounds, mut states) = (0u64, 0u64);
        for r in &chunk.rounds {
            if r.decisions.is_empty() {
                continue;
            }
            let n = r.decisions.len() as u64;
            if is_validation_round(r.id, fraction) {
                sh.vvalid.push_round(&r.decisions, &r.end);
            } else {
                sh.vtrain.push_round(&r.decisions, &r.end);
                sh.vproduced.fetch_add(n, SeqCst);
            }
            rounds += 1;
            states += n;
        }
        sh.vrounds.fetch_add(rounds, SeqCst);
        let mut w = sh.vwindow.lock().unwrap();
        *w = (w.0 + rounds, w.1 + states);
        read += 1;
    }
    read
}

/// Load the chunks of `replay-pi/` into the rollout buffers and delete
/// them; with `rollout.value_children`, the state after each valued move
/// goes to V's buffers with its outcome. Returns the chunks read.
fn ingest_rollout(sh: &Shared) -> usize {
    let fraction = sh.cfg.replay.validation_fraction;
    let children = sh.cfg.rollout.value_children;
    let mut read = 0;
    for (_, path) in chunk_files_in(&sh.dir, ROLLOUT_REPLAY_DIR) {
        let chunk = read_rollout_chunk(&path);
        let _ = std::fs::remove_file(&path);
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                sh.event(format!("skipping {e}"));
                continue;
            }
        };
        for r in chunk.rounds {
            let valid = is_validation_round(r.id, fraction);
            let (n, mut kids) = (r.samples.len() as u64, 0u64);
            for smp in r.samples {
                if children {
                    for (state, points) in smp.children() {
                        let d = Decision { state, policy: SparsePolicy::from_slice(&[(0u8, 1.0f32)]) };
                        if valid {
                            sh.vvalid.push_scored(&[d], points);
                        } else {
                            sh.vtrain.push_scored(&[d], points);
                            kids += 1;
                        }
                    }
                }
                if valid {
                    sh.rvalid.write().unwrap().push(r.id, smp);
                } else {
                    sh.rtrain.write().unwrap().push(r.id, smp);
                }
            }
            if !valid {
                sh.rproduced.fetch_add(n, SeqCst);
                sh.vproduced.fetch_add(kids, SeqCst);
            }
            sh.rrounds.fetch_add(1, SeqCst);
        }
        sh.rwindow.lock().unwrap().merge(&chunk.stats);
        read += 1;
    }
    read
}

fn ingest(sh: Arc<Shared>, mut next_id: u64) {
    while !sh.ingest_done.load(SeqCst) {
        if ingest_new(&sh, &mut next_id) + ingest_value(&sh) + ingest_rollout(&sh) == 0 {
            thread::sleep(Duration::from_secs(1));
        }
    }
    ingest_new(&sh, &mut next_id);
}

/// One learner step's batches: search targets, or rollout samples.
enum Batches {
    Search(StepBatches),
    Pi(PiStep),
}

fn loader(sh: Arc<Shared>, tx: SyncSender<Batches>, seed: u64) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let (n, augment) = (sh.cfg.learner.batch_size, sh.cfg.learner.augment);
    loop {
        if sh.stop.load(SeqCst) {
            return;
        }
        if sh.train.len() < sh.cfg.replay.min_examples {
            thread::sleep(Duration::from_millis(200));
            continue;
        }
        let (p, v) = {
            let buf = sh.train.read();
            (buf.sample_batch_where(n, &mut rng, augment, |s| !is_forced(s)), buf.sample_batch(n, &mut rng, augment))
        };
        if tx.send(Batches::Search(StepBatches::build(&p, &v, Device::Cpu))).is_err() {
            return;
        }
    }
}

/// Policy-iteration batches: rollout samples for P, and V's batch from its
/// own buffers once they hold `value_stream.min_examples` states.
fn pi_loader(sh: Arc<Shared>, tx: SyncSender<Batches>, seed: u64) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let (n, augment) = (sh.cfg.learner.batch_size, sh.cfg.learner.augment);
    let lambda = sh.cfg.selfplay.search.lambda;
    loop {
        if sh.stop.load(SeqCst) {
            return;
        }
        if sh.rtrain.read().unwrap().len() < sh.cfg.rollout.min_examples {
            thread::sleep(Duration::from_millis(200));
            continue;
        }
        let p = sh.rtrain.read().unwrap().sample_batch(n, &mut rng, augment, lambda);
        let v = (sh.v_on() && sh.vtrain.len() >= sh.cfg.value_stream.min_examples)
            .then(|| sh.vtrain.read().sample_batch(n, &mut rng, augment));
        if tx.send(Batches::Pi(PiStep::build(&p, v.as_ref(), Device::Cpu))).is_err() {
            return;
        }
    }
}

/// V-only batches from the V stream's training buffer, once it holds
/// `value_stream.min_examples` states.
fn value_loader(sh: Arc<Shared>, tx: SyncSender<ValueBatch>, seed: u64) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let (n, augment) = (sh.cfg.learner.batch_size, sh.cfg.learner.augment);
    loop {
        if sh.stop.load(SeqCst) {
            return;
        }
        if sh.vtrain.len() < sh.cfg.value_stream.min_examples {
            thread::sleep(Duration::from_millis(200));
            continue;
        }
        let (bid, play) = sh.vtrain.read().sample_batch(n, &mut rng, augment);
        if let Some(vb) = value_batch(&bid, &play, Device::Cpu) {
            if tx.send(vb).is_err() {
                return;
            }
        }
    }
}

// ---- fixed V check --------------------------------------------------------------

/// The last validation rounds of an earlier run (`eval.value_rounds_from`):
/// the same states at every held-out row, unseen by every model trained in
/// either run. V's error there is comparable across steps and runs; the
/// moving validation window is not.
struct FixedRounds {
    from: String,
    buf: ReplayBuffer,
    all: Vec<usize>,
    bids: Vec<usize>,
    plays: Vec<usize>,
}

impl FixedRounds {
    fn load(from: &str, rounds: usize) -> Result<Option<Self>, String> {
        if from.is_empty() || rounds == 0 {
            return Ok(None);
        }
        let dir = Path::new(from);
        let text = std::fs::read_to_string(dir.join(CONFIG_FILE)).map_err(|e| format!("eval.value_rounds_from: {from}/{CONFIG_FILE}: {e}"))?;
        let cfg: toml::Value = toml::from_str(&text).map_err(|e| format!("{from}/{CONFIG_FILE}: {e}"))?;
        let fraction = cfg.get("replay").and_then(|r| r.get("validation_fraction")).and_then(|f| f.as_float()).unwrap_or(0.05);
        let mut picked = Vec::new();
        for (_, path) in chunk_files(dir).iter().rev() {
            let Ok(chunk) = read_chunk(path) else { continue };
            picked.extend(chunk.rounds.into_iter().rev().filter(|r| is_validation_round(r.id, fraction)));
            if picked.len() >= rounds {
                break;
            }
        }
        picked.truncate(rounds);
        if picked.is_empty() {
            return Err(format!("eval.value_rounds_from: no validation rounds in {from}/{REPLAY_DIR}"));
        }
        let mut buf = ReplayBuffer::new(picked.iter().map(|r| r.decisions.len()).sum::<usize>().max(1));
        for r in picked.iter().rev() {
            buf.push_round(&r.decisions, &r.end);
        }
        let all: Vec<usize> = (0..buf.len()).collect();
        let (bids, plays) = all.iter().partition(|&&i| buf.state(i).phase() == GamePhase::Bidding);
        Ok(Some(Self { from: from.to_string(), buf, all, bids, plays }))
    }

    fn measure(&self, l: &Learner, chunk: usize) -> Value {
        json!({
            "from": self.from,
            "all": l.value_held_out(&self.buf, &self.all, chunk),
            "bids": l.value_held_out(&self.buf, &self.bids, chunk),
            "plays": l.value_held_out(&self.buf, &self.plays, chunk),
        })
    }
}

// ---- probe set ----------------------------------------------------------------

/// Fixed teacher rounds: the same states at every measurement.
struct Probe {
    buf: ReplayBuffer,
    p_idx: Vec<usize>,
    v_idx: Vec<usize>,
    chunk: usize,
}

/// P's policies and V's ŝ (with the seat mask) on the probe states, on the
/// CPU.
struct ProbeOut {
    bid: Vec<Tensor>,
    play: Vec<Tensor>,
    value: Vec<(Tensor, Tensor)>,
}

impl Probe {
    fn new(rounds: u64, chunk: usize) -> Self {
        let mut buf = ReplayBuffer::new((rounds as usize).max(1) * 64);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        fill_buffer(&mut buf, &TeacherConfig::default(), rounds.max(1), PROBE_SEED, threads);
        let v_idx: Vec<usize> = (0..buf.len()).collect();
        let p_idx = v_idx.iter().copied().filter(|&i| !is_forced(buf.state(i))).collect();
        Self { buf, p_idx, v_idx, chunk }
    }

    fn outputs(&self, l: &Learner) -> ProbeOut {
        let dev = l.device;
        let mut out = ProbeOut { bid: vec![], play: vec![], value: vec![] };
        tch::no_grad(|| {
            for idx in self.p_idx.chunks(self.chunk) {
                let (bid, play) = self.buf.batch_from_indices(idx);
                if let Some(pb) = bid_policy_batch(&bid, dev) {
                    out.bid.push(policy_probs(&l.policy, &pb, false).to_device(Device::Cpu));
                }
                if let Some(pb) = play_policy_batch(&play, dev) {
                    out.play.push(policy_probs(&l.policy, &pb, false).to_device(Device::Cpu));
                }
            }
            for idx in self.v_idx.chunks(self.chunk) {
                let (bid, play) = self.buf.batch_from_indices(idx);
                if let Some(vb) = value_batch(&bid, &play, dev) {
                    let k = vb.target.size()[1];
                    let v = l.value.seat_values(&vb.input, k, false).to_device(Device::Cpu);
                    out.value.push((v, vb.seat_mask.to_device(Device::Cpu)));
                }
            }
        });
        out
    }
}

/// Mean KL(a ‖ b) over the rows of matching probability tensors.
fn mean_kl(a: &[Tensor], b: &[Tensor]) -> f64 {
    let (mut sum, mut rows) = (0.0f64, 0i64);
    for (p, q) in a.iter().zip(b) {
        let k = (p * ((p + 1e-8).log() - (q + 1e-8).log())).sum(Kind::Double);
        sum += k.double_value(&[]);
        rows += p.size()[0];
    }
    sum / rows.max(1) as f64
}

/// Mean |a − b| of V's ŝ over the real seats.
fn mean_abs_change(a: &[(Tensor, Tensor)], b: &[(Tensor, Tensor)]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0.0f64);
    for ((x, m), (y, _)) in a.iter().zip(b) {
        let m = m.to_kind(Kind::Float);
        sum += ((x - y).abs() * &m).sum(Kind::Double).double_value(&[]);
        n += m.sum(Kind::Double).double_value(&[]);
    }
    sum / n.max(1.0)
}

fn probe_change(step: u64, now: &ProbeOut, prev: &ProbeOut, init: &ProbeOut) -> Value {
    json!({
        "kind": "probe",
        "step": step,
        "kl_bid_vs_prev": mean_kl(&prev.bid, &now.bid),
        "kl_play_vs_prev": mean_kl(&prev.play, &now.play),
        "kl_bid_vs_start": mean_kl(&init.bid, &now.bid),
        "kl_play_vs_start": mean_kl(&init.play, &now.play),
        "v_change_vs_prev": mean_abs_change(&prev.value, &now.value),
        "v_change_vs_start": mean_abs_change(&init.value, &now.value),
    })
}

// ---- learner ------------------------------------------------------------------

/// `n` slots of `buf` whose state passes `keep`, drawn at random.
fn pick_where(buf: &ReplayBuffer, n: usize, rng: &mut Xoshiro256PlusPlus, keep: impl Fn(&BlobState) -> bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    if buf.is_empty() {
        return out;
    }
    for _ in 0..64 * n {
        if out.len() == n {
            break;
        }
        let i = rng.gen_range(0..buf.len());
        if keep(buf.state(i)) {
            out.push(i);
        }
    }
    out.sort_unstable();
    out
}

/// Held out: self-play validation rounds against an equal sample of
/// training rounds, and the teacher probe set.
fn held_out(sh: &Shared, l: &Learner, probe: &Probe, start: Option<&FrozenPolicy>, rng: &mut Xoshiro256PlusPlus) -> Value {
    let chunk = probe.chunk;
    let n = sh.cfg.log.eval_examples;
    let (valid, train) = {
        let vb = sh.valid.read();
        let tb = sh.train.read();
        let mut vv: Vec<usize> = index::sample(rng, vb.len(), n.min(vb.len())).into_vec();
        vv.sort_unstable();
        let unforced: Vec<usize> = (0..vb.len()).filter(|&i| !is_forced(vb.state(i))).collect();
        let mut vp: Vec<usize> =
            index::sample(rng, unforced.len(), n.min(unforced.len())).into_iter().map(|i| unforced[i]).collect();
        vp.sort_unstable();
        let tv = pick_where(&tb, vv.len(), rng, |_| true);
        let tp = pick_where(&tb, vp.len(), rng, |s| !is_forced(s));
        (
            json!({ "policy": l.policy_held_out(&vb, &vp, chunk), "value": l.value_held_out(&vb, &vv, chunk) }),
            json!({ "policy": l.policy_held_out(&tb, &tp, chunk), "value": l.value_held_out(&tb, &tv, chunk) }),
        )
    };
    let teacher = json!({
        "policy": l.policy_held_out(&probe.buf, &probe.p_idx, chunk),
        "value": l.value_held_out(&probe.buf, &probe.v_idx, chunk),
    });
    // The V stream: its validation rounds against an equal training sample.
    let stream = {
        let (vb, tb) = (sh.vvalid.read(), sh.vtrain.read());
        if vb.is_empty() || tb.is_empty() {
            Value::Null
        } else {
            let mut vv: Vec<usize> = index::sample(rng, vb.len(), n.min(vb.len())).into_vec();
            vv.sort_unstable();
            let tv = pick_where(&tb, vv.len(), rng, |_| true);
            json!({ "validation": l.value_held_out(&vb, &vv, chunk), "train_sample": l.value_held_out(&tb, &tv, chunk) })
        }
    };
    // Policy iteration: rollout validation samples against an equal training
    // sample, and the newest validation samples (the latest models' play).
    let rollout = {
        let (vb, tb) = (sh.rvalid.read().unwrap(), sh.rtrain.read().unwrap());
        if vb.is_empty() || tb.is_empty() {
            Value::Null
        } else {
            let (t, eps, lambda) = (sh.cfg.rollout.temperature, sh.cfg.rollout.epsilon, sh.cfg.selfplay.search.lambda);
            let mut vv: Vec<usize> = index::sample(rng, vb.len(), n.min(vb.len())).into_vec();
            vv.sort_unstable();
            let mut tv: Vec<usize> = index::sample(rng, tb.len(), vv.len().min(tb.len())).into_vec();
            tv.sort_unstable();
            let recent = vb.recent((n / 4).max(1));
            json!({
                "validation": l.pi_held_out(&vb, &vv, chunk, t, eps, lambda, None),
                "train_sample": l.pi_held_out(&tb, &tv, chunk, t, eps, lambda, None),
                "recent": l.pi_held_out(&vb, &recent, chunk, t, eps, lambda, start),
            })
        }
    };
    json!({
        "kind": "held_out",
        "step": l.step,
        "secs": sh.secs(),
        "active_hours": sh.active_hours(),
        "validation": valid,
        "train_sample": train,
        "teacher_probe": teacher,
        "value_stream": stream,
        "rollout": rollout,
        "fixed_value": sh.fixed.as_ref().map_or(Value::Null, |f| f.measure(l, chunk)),
    })
}

fn num(v: &Value) -> String {
    v.as_f64().map_or("-".to_string(), |x| format!("{x:.4}"))
}

fn short_held_out(r: &Value) -> String {
    let (v, t, p) = (&r["validation"], &r["train_sample"], &r["teacher_probe"]);
    let fixed = &r["fixed_value"];
    let ro = &r["rollout"];
    let extra = format!(
        "; fixed V mse {} (bids {}, plays {}); V stream mse {} / {}; rollouts: gain bids {} plays {} (newest {} / {}), over the start {} / {}, V's pick {} / {}",
        num(&fixed["all"]["mse"]),
        num(&fixed["bids"]["mse"]),
        num(&fixed["plays"]["mse"]),
        num(&r["value_stream"]["validation"]["mse"]),
        num(&r["value_stream"]["train_sample"]["mse"]),
        num(&ro["validation"]["bids"]["gain"]),
        num(&ro["validation"]["plays"]["gain"]),
        num(&ro["recent"]["bids"]["gain"]),
        num(&ro["recent"]["plays"]["gain"]),
        num(&ro["recent"]["bids"]["gain_vs_start"]),
        num(&ro["recent"]["plays"]["gain_vs_start"]),
        num(&ro["validation"]["bids"]["v_gain"]),
        num(&ro["validation"]["plays"]["v_gain"]),
    );
    format!(
        "P bid {} / {}, play {} / {}, top = search's {} / {}; V mse {} / {} (var {}), corr {} / {}; teacher probe: P top = rule bot 2's {} / {}, V last-trick mse {}",
        num(&v["policy"]["bid_loss"]), num(&t["policy"]["bid_loss"]), num(&v["policy"]["play_loss"]), num(&t["policy"]["play_loss"]),
        num(&v["policy"]["play_agreement"]), num(&t["policy"]["play_agreement"]), num(&v["value"]["mse"]), num(&t["value"]["mse"]),
        num(&v["value"]["variance"]), num(&v["value"]["correlation"]), num(&t["value"]["correlation"]),
        num(&p["policy"]["bid_agreement"]), num(&p["policy"]["play_agreement"]), num(&p["value"]["last_trick_mse"]),
    ) + &extra
}

fn publish(sh: &Shared, l: &Learner, tx: &Sender<(u64, PathBuf)>) -> Result<(), String> {
    sh.published_value_updates.store(l.value_updates, SeqCst);
    let mdir = sh.path("models").join(format!("step-{:06}", l.step));
    l.save(&mdir.join("checkpoint"))?;
    l.save(&sh.path("checkpoint"))?;
    sh.save_state();
    let _ = tx.send((l.step, mdir));
    Ok(())
}

/// The training loop; returns when `stop` is set.
fn learn(
    sh: &Shared,
    l: &mut Learner,
    rx: &Receiver<Batches>,
    vrx: &Receiver<ValueBatch>,
    probe: &Probe,
    init: &ProbeOut,
    start: Option<&FrozenPolicy>,
    pub_tx: &Sender<(u64, PathBuf)>,
) -> Result<(), String> {
    let cfg = &sh.cfg;
    let b = cfg.learner.batch_size as f64;
    let start_step = l.step;
    let start_value_updates = l.value_updates;
    let resumed = start_step > 0;
    let ramp = cfg.learner.resume_warmup_steps.max(1) as f64;
    let device = l.device;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.run.seed ^ 0x4E1D ^ start_step);
    let zero = || Tensor::zeros([], (Kind::Float, device));
    let (mut p_sum, mut v_sum, mut n) = (zero(), zero(), 0u32);
    let mut interval = Instant::now();
    let (mut gov_wait, mut load_wait, mut side) = (0.0f64, 0.0f64, 0.0f64);
    let mut last_ckpt = Instant::now();
    let mut training = false;
    // The starting held-out row comes before any update, P's or V's.
    let mut first_row = resumed;
    let mut prev = probe.outputs(l);
    let v_on = sh.v_on();
    let pi_on = sh.pi_on();
    let (mut vo_sum, mut vo_n, mut vo_time) = (zero(), 0u32, 0.0f64);
    let mut lr = l.lr();
    // A training row every `log.every` learner steps, and at least every
    // two minutes while anything trains (V-only updates may run for a long
    // time before P's first step).
    let row_secs = 120.0;
    while !sh.stop.load(SeqCst) {
        // P's data and its governor: rollout samples, or search targets.
        let (produced, have, need, ratio) = if pi_on {
            let r = &cfg.rollout;
            (sh.rproduced.load(SeqCst) as f64, sh.rtrain.read().unwrap().len(), r.min_examples, r.ratio)
        } else {
            (sh.produced.load(SeqCst) as f64, sh.train.len(), cfg.replay.min_examples, cfg.replay.replay_ratio)
        };
        let ready = have >= need;
        let mut stepped = false;
        // Rollout buffers start empty on a resume: count P's samples from
        // this start (search's buffer reloads, so its count spans the run).
        let steps = l.step - if pi_on { start_step } else { 0 };
        let per_step = if pi_on { b * cfg.rollout.micro_batches.max(1) as f64 } else { b };
        if !ready || sh.paused.load(SeqCst) || (steps + 1) as f64 * per_step > ratio * produced {
            // P waits for search data: V learns from the rounds of P alone
            // meanwhile, within its own ratio cap, from the start.
            let v_room = (sh.vsamples.load(SeqCst) as f64 + b) <= cfg.value_stream.ratio * sh.vproduced.load(SeqCst) as f64;
            let vb = if v_on && !sh.paused.load(SeqCst) && v_room { vrx.try_recv().ok() } else { None };
            if let Some(vb) = vb {
                if !first_row {
                    first_row = true;
                    let row = held_out(sh, l, probe, start, &mut rng);
                    eprintln!("[train] step {} (before any update): {}", l.step, short_held_out(&row));
                    sh.log(&row);
                    sh.status.lock().unwrap().held_out = row;
                    (interval, gov_wait, load_wait, side) = (Instant::now(), 0.0, 0.0, 0.0);
                }
                let w = Instant::now();
                if resumed {
                    l.value_lr_scale = ((l.value_updates - start_value_updates + 1) as f64 / ramp).min(1.0);
                }
                vo_sum += l.train_value_on(&vb.to_device(device));
                vo_n += 1;
                sh.vsamples.fetch_add(b as u64, SeqCst);
                vo_time += w.elapsed().as_secs_f64();
            } else {
                let w = Instant::now();
                thread::sleep(Duration::from_millis(if v_on && first_row { 5 } else { 50 }));
                gov_wait += w.elapsed().as_secs_f64();
                if !ready {
                    let what = if pi_on { "rollout samples" } else { "training examples" };
                    sh.set_phase(&format!("waiting for data ({have} / {need} {what})"));
                }
            }
        } else {
            if !training {
                training = true;
                sh.set_phase("training");
                let what = if pi_on { "rollout samples" } else { "training examples" };
                sh.event(format!("learner starts at step {} on {have} {what}", l.step));
                if !first_row {
                    first_row = true;
                    let row = held_out(sh, l, probe, start, &mut rng);
                    eprintln!("[train] step {}: {}", l.step, short_held_out(&row));
                    sh.log(&row);
                    sh.status.lock().unwrap().held_out = row;
                }
                (interval, gov_wait, load_wait, side) = (Instant::now(), 0.0, 0.0, 0.0);
            }
            if resumed {
                l.lr_scale = ((l.step - start_step + 1) as f64 / ramp).min(1.0);
                l.value_lr_scale = ((l.value_updates - start_value_updates + 1) as f64 / ramp).min(1.0);
            }
            lr = l.lr();
            let w = Instant::now();
            let batches = rx.recv().map_err(|_| "batch loaders stopped".to_string())?;
            load_wait += w.elapsed().as_secs_f64();
            let (p, v) = match batches {
                Batches::Search(b) => l.train_on(&b.to_device(device)),
                Batches::Pi(b) => {
                    // `rollout.micro_batches` batches per P update.
                    let mut parts = vec![b.to_device(device)];
                    while parts.len() < cfg.rollout.micro_batches.max(1) {
                        match rx.recv().map_err(|_| "batch loaders stopped".to_string())? {
                            Batches::Pi(b) => parts.push(b.to_device(device)),
                            Batches::Search(_) => unreachable!("one kind of loader per run"),
                        }
                    }
                    l.train_pi_on(&parts, cfg.rollout.temperature, cfg.rollout.epsilon)
                }
            };
            p_sum += p.unwrap_or_else(zero);
            v_sum += v.unwrap_or_else(zero);
            n += 1;
            stepped = true;
        }
        let step = l.step;

        let secs = interval.elapsed().as_secs_f64();
        if (stepped && step.is_multiple_of(cfg.log.every)) || (n + vo_n > 0 && secs >= row_secs) {
            let busy = (secs - gov_wait - load_wait - side - vo_time).max(1e-9);
            let vproduced = sh.vproduced.load(SeqCst) as f64;
            let mean = |t: &Tensor, k: u32| if k > 0 { t.double_value(&[]) / k as f64 } else { f64::NAN };
            let row = json!({
                "kind": "train",
                "step": step,
                "secs": sh.secs(),
                "active_hours": sh.active_hours(),
                "lr": lr,
                "policy_loss": mean(&p_sum, n),
                "value_loss": mean(&v_sum, n),
                "steps_per_hour": n as f64 / secs * 3600.0,
                "gpu_steps_per_sec": n as f64 / busy,
                "governor_wait": gov_wait / secs,
                "loader_wait": load_wait / secs,
                "examples_produced": produced,
                "replay_ratio": (step - if pi_on { start_step } else { 0 }) as f64 * per_step / produced.max(1.0),
                "train_examples": sh.train.len(),
                "valid_examples": sh.valid.len(),
                "model_step": sh.state.lock().unwrap().model_step,
                "value_lr": l.value_lr(),
                "value_only_steps": vo_n,
                "value_only_loss": mean(&vo_sum, vo_n),
                "value_only_steps_per_hour": vo_n as f64 / secs * 3600.0,
                "value_only_busy": vo_time / secs,
                "value_stream_produced": vproduced,
                "value_stream_ratio": sh.vsamples.load(SeqCst) as f64 / vproduced.max(1.0),
                "value_stream_examples": sh.vtrain.len(),
                "rollout_produced": sh.rproduced.load(SeqCst),
                "rollout_examples": sh.rtrain.read().unwrap().len(),
            });
            eprintln!(
                "[train] step {step} lr {lr:.2e}: P {:.4}, V {:.4}; {:.0} steps/h, replay ratio {:.2}, governor wait {:.0}%; V-only {:.0}/h, V-only loss {:.4}",
                row["policy_loss"].as_f64().unwrap_or(f64::NAN),
                row["value_loss"].as_f64().unwrap_or(f64::NAN),
                row["steps_per_hour"].as_f64().unwrap_or(0.0),
                row["replay_ratio"].as_f64().unwrap_or(0.0),
                100.0 * gov_wait / secs,
                row["value_only_steps_per_hour"].as_f64().unwrap_or(0.0),
                row["value_only_loss"].as_f64().unwrap_or(f64::NAN),
            );
            sh.log(&row);
            sh.status.lock().unwrap().learner = row;
            (p_sum, v_sum, n, gov_wait, load_wait, side, interval) = (zero(), zero(), 0, 0.0, 0.0, 0.0, Instant::now());
            (vo_sum, vo_n, vo_time) = (zero(), 0, 0.0);
        }
        let t = Instant::now();
        if stepped && step.is_multiple_of(cfg.log.eval_every) {
            let row = held_out(sh, l, probe, start, &mut rng);
            eprintln!("[train] step {step}: {}", short_held_out(&row));
            sh.log(&row);
            sh.status.lock().unwrap().held_out = row;
        }
        if stepped && step.is_multiple_of(cfg.log.publish_every) {
            publish(sh, l, pub_tx)?;
            let now = probe.outputs(l);
            let row = probe_change(step, &now, &prev, init);
            sh.log(&row);
            sh.status.lock().unwrap().probes.push(row);
            prev = now;
            last_ckpt = Instant::now();
        } else if first_row && last_ckpt.elapsed().as_secs_f64() > cfg.log.checkpoint_minutes * 60.0 {
            l.save(&sh.path("checkpoint"))?;
            sh.save_state();
            last_ckpt = Instant::now();
        }
        side += t.elapsed().as_secs_f64();
    }
    // The run's last held-out row, before the final publish.
    if sh.finish.load(SeqCst) && first_row {
        let row = held_out(sh, l, probe, start, &mut rng);
        eprintln!("[train] step {} (final): {}", l.step, short_held_out(&row));
        sh.log(&row);
        sh.status.lock().unwrap().held_out = row;
    }
    Ok(())
}


// ---- publisher and evaluator --------------------------------------------------

fn publisher(sh: Arc<Shared>, rx: Receiver<(u64, PathBuf)>, eval_tx: Sender<(u64, PathBuf)>) {
    for (step, mdir) in rx {
        let t = Instant::now();
        let model = mdir.join("model");
        match export_with_threads(Some(&mdir.join("checkpoint")), &model, false, Some(EXPORT_THREADS)) {
            Ok(()) => {
                {
                    let mut st = sh.state.lock().unwrap();
                    st.model = model.clone();
                    st.model_step = step;
                }
                if let Err(e) = write_json(&sh.path(MODEL_POINTER_FILE), &ModelPointer { model: model.clone(), step }) {
                    sh.event(format!("{MODEL_POINTER_FILE}: {e}"));
                }
                sh.save_state();
                let secs = t.elapsed().as_secs_f64();
                sh.log(&json!({ "kind": "publish", "step": step, "secs": sh.secs(), "export_secs": secs }));
                sh.event(format!("published step {step} ({secs:.0} s export); the actors switch within seconds"));
                let _ = eval_tx.send((step, model));
            }
            Err(e) => sh.event(format!("export of step {step} failed: {e}")),
        }
    }
}

/// A per-deal file of `bench --per-deal-out`: its header and each deal's
/// difference.
fn read_per_deal(path: &Path) -> Result<(String, Vec<f64>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default().to_string();
    let mut out = Vec::new();
    for line in lines.skip(1) {
        let (d, x) = line.split_once(',').ok_or_else(|| format!("{}: bad line {line:?}", path.display()))?;
        let d: usize = d.parse().map_err(|_| format!("{}: bad deal {d:?}", path.display()))?;
        if out.len() <= d {
            out.resize(d + 1, f64::NAN);
        }
        out[d] = x.parse().unwrap_or(f64::NAN);
    }
    Ok((header, out))
}

/// Mean and 95% half-width of the finite values.
fn mean_ci(xs: &[f64]) -> (f64, f64, usize) {
    let v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    let k = v.len() as f64;
    let mean = v.iter().sum::<f64>() / k;
    if v.len() < 2 {
        return (mean, f64::NAN, v.len());
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k - 1.0);
    (mean, 1.96 * (var / k).sqrt(), v.len())
}

/// `a − b` deal by deal, for two per-deal files of the same deal list.
fn paired(a: &Path, b: &Path) -> Result<(f64, f64, usize), String> {
    let ((ha, xa), (hb, xb)) = (read_per_deal(a)?, read_per_deal(b)?);
    if ha != hb {
        return Err(format!("{} was played on another deal list ({hb:?}, not {ha:?})", b.display()));
    }
    let d: Vec<f64> = xa.iter().zip(&xb).map(|(x, y)| x - y).collect();
    Ok(mean_ci(&d))
}

/// The most recent `bench/<name>-step-*.csv` before `step`, if any.
fn previous_csv(sh: &Shared, name: &str, step: u64) -> Option<PathBuf> {
    let prefix = format!("{name}-step-");
    let mut found: Vec<(u64, PathBuf)> = std::fs::read_dir(sh.path("bench"))
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|p| {
            let stem = p.file_stem()?.to_str()?.to_string();
            let s: u64 = stem.strip_prefix(&prefix)?.parse().ok()?;
            (p.extension()? == "csv" && s < step).then_some((s, p))
        })
        .collect();
    found.sort();
    found.pop().map(|x| x.1)
}

/// What a bench report says: bids made by 1 / 2–4 / 5–8 cards, 0-bids in
/// 5–8-card rounds, win share and seconds.
fn parse_report(text: &str) -> (Vec<f64>, f64, f64, f64) {
    let mut made = vec![f64::NAN; 3];
    let (mut zero, mut win, mut secs) = (f64::NAN, f64::NAN, f64::NAN);
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        let num = |i: usize| t.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(f64::NAN);
        match t.first().copied() {
            Some("1") if t.get(1) == Some(&"card") => made[0] = num(3),
            Some("2-4") => made[1] = num(3),
            Some("5-8") => {
                made[2] = num(3);
                zero = num(4);
            }
            Some("win") => win = num(2),
            Some("table") if t.len() >= 2 => secs = num(t.len() - 2),
            _ => {}
        }
    }
    (made, zero, win, secs)
}

/// `bench` flags for the self-play search settings, so a search bench
/// measures the search that makes the training targets.
fn search_args(sh: &Shared) -> Vec<String> {
    let c = &sh.cfg.selfplay.search;
    let one_card = match c.one_card_bids {
        OneCardBids::Exact => "exact",
        OneCardBids::Policy => "policy",
        OneCardBids::Search => "search",
    };
    [
        ("--c-puct", c.c_puct.to_string()),
        ("--bid-dets", c.bid_budget.determinizations.to_string()),
        ("--bid-sims", c.bid_budget.sims_per_determinization.to_string()),
        ("--dets", c.play_budget.determinizations.to_string()),
        ("--sims", c.play_budget.sims_per_determinization.to_string()),
        ("--one-card-bids", one_card.to_string()),
        ("--bid-candidates", c.bid_weighting.candidates.to_string()),
        ("--bid-noise", c.bid_weighting.noise.to_string()),
        ("--root", c.root_rule.to_string()),
        ("--q-temp", c.q_temperature.to_string()),
    ]
    .into_iter()
    .flat_map(|(k, v)| [k.to_string(), v])
    .collect()
}

/// Run `blobmaster bench` on `model`: report and per-deal file in
/// `bench/<name>-step-NNNNNN.{txt,csv}`, compared paired with each of
/// `baselines` that exists, logged as a bench row.
#[allow(clippy::too_many_arguments)]
fn bench(sh: &Shared, step: u64, name: &str, model: &Path, mode: &str, opponent: &str, deals: usize, seed: u64, threads: Option<usize>, extra: &[String], baselines: &[(String, PathBuf)]) {
    let stem = format!("{name}-step-{step:06}");
    let (txt, csv) = (sh.path("bench").join(format!("{stem}.txt")), sh.path("bench").join(format!("{stem}.csv")));
    let mut cmd = blobmaster_cmd();
    cmd.arg("bench")
        .arg(model)
        .args(["--mode", mode, "--opponent", opponent])
        .args(["--deals", &deals.to_string(), "--seed", &seed.to_string()])
        .arg("--per-deal-out")
        .arg(&csv)
        .args(extra);
    if let Some(t) = threads {
        cmd.args(["--threads", &t.to_string()]);
    }
    let t = Instant::now();
    let out = match cmd.output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            sh.event(format!("bench {stem} failed ({}): {}", o.status, err.lines().last().unwrap_or("")));
            return;
        }
        Err(e) => {
            sh.event(format!("bench {stem}: {e}"));
            return;
        }
    };
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    let (made, zero, win, secs) = parse_report(&text);
    let (diff, ci) = match read_per_deal(&csv) {
        Ok((_, xs)) => {
            let (m, c, _) = mean_ci(&xs);
            (m, c)
        }
        Err(e) => {
            sh.event(format!("bench {stem}: {e}"));
            (f64::NAN, f64::NAN)
        }
    };
    let mut pairs = Vec::new();
    for (label, base) in baselines {
        if !base.is_file() {
            continue;
        }
        match paired(&csv, base) {
            Ok((d, c, n)) => {
                text.push_str(&format!("paired vs {label} ({}): diff {d:+.2} ± {c:.2} (95% CI over {n} deals)\n", base.display()));
                pairs.push(json!({ "vs": label, "file": base, "diff": d, "ci": c, "deals": n }));
            }
            Err(e) => sh.event(format!("bench {stem}: {e}")),
        }
    }
    let _ = std::fs::write(&txt, &text);
    let row = json!({
        "kind": "bench",
        "step": step,
        "name": name,
        "secs": sh.secs(),
        "active_hours": sh.active_hours(),
        "deals": deals,
        "diff": diff,
        "ci": ci,
        "win_share": win,
        "made": made,
        "zero_bids_5_8": zero,
        "bench_secs": if secs.is_finite() { secs } else { t.elapsed().as_secs_f64() },
        "paired": pairs,
    });
    sh.log(&row);
    let pair = pairs.first().map_or(String::new(), |p| {
        format!(", paired vs {} {:+.2} ± {:.2}", p["vs"].as_str().unwrap_or(""), p["diff"].as_f64().unwrap_or(f64::NAN), p["ci"].as_f64().unwrap_or(f64::NAN))
    });
    sh.event(format!("bench {name} step {step}: {diff:+.2} ± {ci:.2}{pair} ({:.0} s)", t.elapsed().as_secs_f64()));
    sh.status.lock().unwrap().benches.push(row);
}

/// Name of the network bench on the search bench's deals.
fn net_search_deals_name(sh: &Shared) -> String {
    format!("net-rb2-s{}", sh.cfg.eval.search_seed)
}

/// The network benches of a publish, on `threads` (None: every core).
fn net_benches(sh: &Shared, step: u64, model: &Path, threads: Option<usize>) {
    let e = &sh.cfg.eval;
    let start_model = std::fs::canonicalize(&sh.cfg.run.init_model).unwrap_or_else(|_| PathBuf::from(&sh.cfg.run.init_model));
    let mut benches = vec![
        ("net-rb2".to_string(), "rulebot2".to_string(), e.net_deals_rule_bot_2, DEFAULT_SEED),
        ("net-rb".to_string(), "rulebot".to_string(), e.net_deals_rule_bot, DEFAULT_SEED),
        // The search bench's deals: search's margin over P alone, paired.
        (net_search_deals_name(sh), "rulebot2".to_string(), e.search_deals, e.search_seed),
        // Four copies of the run's starting P: progress in self-play's
        // own setting (the start scores 0 there).
        ("net-vs0".to_string(), start_model.to_string_lossy().to_string(), if step > 0 { e.net_deals_vs_start } else { 0 }, DEFAULT_SEED),
    ];
    // The panel of fixed opponents, step 0 included for the paired change.
    for m in &e.panel {
        let model = std::fs::canonicalize(&m.model).unwrap_or_else(|_| PathBuf::from(&m.model));
        benches.push((format!("net-vs-{}", m.name), model.to_string_lossy().to_string(), e.net_deals_panel, DEFAULT_SEED));
    }
    for (name, opponent, deals, seed) in benches {
        if deals == 0 {
            continue;
        }
        let mut baselines = vec![];
        let start = sh.path("bench").join(format!("{name}-step-000000.csv"));
        if step > 0 {
            if let Some(p) = previous_csv(sh, &name, step).filter(|p| *p != start) {
                baselines.push(("previous".to_string(), p));
            }
            if start.is_file() {
                baselines.insert(0, ("start".to_string(), start));
            }
        }
        bench(sh, step, &name, model, "network", &opponent, deals, seed, threads, &[], &baselines);
    }
}

/// A search bench on every core, the actor process frozen while it runs.
#[allow(clippy::too_many_arguments)]
fn search_bench(sh: &Shared, step: u64, model: &Path, name: &str, opponent: &str, deals: usize, seed: u64, baseline: &str) {
    sh.hold.store(true, SeqCst);
    sync_frozen(sh);
    let phase = std::mem::take(&mut sh.status.lock().unwrap().phase);
    sh.set_phase(&format!("search bench {name} of step {step} (actors frozen)"));
    let mut baselines = vec![];
    // P alone on the same deals at the same step: the improvement margin.
    if opponent == "rulebot2" && seed == sh.cfg.eval.search_seed && deals == sh.cfg.eval.search_deals {
        let p_alone = sh.path("bench").join(format!("{}-step-{step:06}.csv", net_search_deals_name(sh)));
        baselines.push(("P alone".to_string(), p_alone));
    }
    if !baseline.is_empty() {
        baselines.push(("baseline".to_string(), PathBuf::from(baseline)));
    }
    if let Some(p) = previous_csv(sh, name, step) {
        baselines.push(("previous".to_string(), p));
    }
    bench(sh, step, name, model, "search", opponent, deals, seed, None, &search_args(sh), &baselines);
    sh.hold.store(false, SeqCst);
    sync_frozen(sh);
    sh.set_phase(&phase);
}

/// Search against four copies of the same model's P (`eval.search_vs_p_deals`).
fn search_vs_p(sh: &Shared, step: u64, model: &Path) {
    let e = &sh.cfg.eval;
    if e.search_vs_p_deals > 0 {
        search_bench(sh, step, model, "search-vsP", &model.to_string_lossy(), e.search_vs_p_deals, e.search_seed, "");
    }
}

fn evaluator(sh: Arc<Shared>, rx: Receiver<(u64, PathBuf)>) {
    let e = &sh.cfg.eval;
    while let Ok(mut next) = rx.recv() {
        // Publishes can come faster than the benches run: bench the newest.
        while let Ok(newer) = rx.try_recv() {
            next = newer;
        }
        let (step, model) = next;
        if sh.stop.load(SeqCst) && !sh.finish.load(SeqCst) {
            continue;
        }
        // Beside the actors on `eval.threads`; after they stopped (the final
        // publish), on every core.
        let threads = if sh.stop.load(SeqCst) { None } else { Some(e.threads) };
        net_benches(&sh, step, &model, threads);
        let due = {
            let mut st = sh.state.lock().unwrap();
            let due = e.search_every_hours > 0.0 && step > 0 && st.active_secs >= st.next_search_secs;
            if due {
                st.next_search_secs = st.active_secs + e.search_every_hours * 3600.0;
            }
            due
        };
        if due && !sh.stop.load(SeqCst) {
            search_bench(&sh, step, &model, "search-rb2", "rulebot2", e.search_deals, e.search_seed, &e.search_baseline);
            search_vs_p(&sh, step, &model);
        }
    }
}

// ---- monitor and status ---------------------------------------------------------

fn phase_stats(p: &blob_engine::selfplay::PhaseStats) -> Value {
    let n = p.decisions.max(1) as f64;
    json!({
        "decisions": p.decisions,
        "kl_target_prior": p.kl_sum / n,
        "top_differs": p.top_differs as f64 / n,
        "target_entropy": p.target_entropy_sum / n,
        "prior_entropy": p.prior_entropy_sum / n,
        "off_top_played": p.off_top_played as f64 / n,
    })
}

fn rollout_phase(rw: &RolloutStats, k: usize) -> Value {
    let n = rw.samples[k].max(1) as f64;
    json!({
        "samples": rw.samples[k],
        "moves_per_sample": rw.moves[k] as f64 / n,
        "top_not_best": rw.top_not_best[k] as f64 / n,
        "hindsight": rw.hindsight_sum[k] / n,
    })
}

fn selfplay_row(sh: &Shared, w: &SelfPlayStats, vw: (u64, u64), rw: &RolloutStats, secs: f64) -> Value {
    let share = |x: u64, n: u64| if n > 0 { x as f64 / n as f64 } else { f64::NAN };
    json!({
        "kind": "selfplay",
        "secs": sh.secs(),
        "active_hours": sh.active_hours(),
        "window_secs": secs,
        "rounds_per_hour": w.rounds as f64 / secs * 3600.0,
        "examples_per_hour": w.decisions as f64 / secs * 3600.0,
        "forced": share(w.forced, w.decisions),
        "bid": phase_stats(&w.bid),
        "play": phase_stats(&w.play),
        "made": (0..3).map(|i| share(w.made[i], w.seat_rounds[i])).collect::<Vec<_>>(),
        "zero_bids": (0..3).map(|i| share(w.zero_bids[i], w.seat_rounds[i])).collect::<Vec<_>>(),
        "rounds_total": sh.rounds.load(SeqCst),
        "examples_total": sh.examples.load(SeqCst),
        "train_examples": sh.train.len(),
        "valid_examples": sh.valid.len(),
        "actor_restarts": sh.restarts.load(SeqCst),
        "model_step": sh.state.lock().unwrap().model_step,
        "value_stream": {
            "rounds_per_hour": vw.0 as f64 / secs * 3600.0,
            "states_per_hour": vw.1 as f64 / secs * 3600.0,
            "rounds_total": sh.vrounds.load(SeqCst),
            "train_states": sh.vtrain.len(),
            "valid_states": sh.vvalid.len(),
        },
        "rollout": {
            "rounds_per_hour": rw.rounds as f64 / secs * 3600.0,
            "samples_per_hour": (rw.samples[0] + rw.samples[1]) as f64 / secs * 3600.0,
            "bid": rollout_phase(rw, 0),
            "play": rollout_phase(rw, 1),
            "rounds_total": sh.rrounds.load(SeqCst),
            "train_samples": sh.rtrain.read().unwrap().len(),
            "valid_samples": sh.rvalid.read().unwrap().len(),
        },
    })
}

fn f(v: &Value, digits: usize) -> String {
    v.as_f64().map_or("-".to_string(), |x| format!("{x:.digits$}"))
}

fn pct(v: &Value) -> String {
    v.as_f64().map_or("-".to_string(), |x| format!("{:.1}%", 100.0 * x))
}

fn render_status(sh: &Shared) -> String {
    let cfg = &sh.cfg;
    let (active_h, model_step) = {
        let st = sh.state.lock().unwrap();
        (st.active_secs / 3600.0, st.model_step)
    };
    let st = sh.status.lock().unwrap();
    let mut s = String::new();
    let mut w = |line: String| {
        s.push_str(&line);
        s.push('\n');
    };
    w(format!("# RL run `{}`", sh.dir.display()));
    w(String::new());
    w(format!(
        "Updated {} · running {:.2} h of {:.1} (wall {:.2} h) · **{}**",
        clock(),
        active_h,
        cfg.run.hours,
        sh.secs() / 3600.0,
        if st.phase.is_empty() { "starting" } else { st.phase.as_str() }
    ));
    w(String::new());
    w("Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).".into());
    w(String::new());

    w("## Strength".into());
    w(String::new());
    w("Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.".into());
    w(String::new());
    w("| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |".into());
    w("|---|---|---|---|---|---|---|---|".into());
    for b in &st.benches {
        let paired: Vec<String> = b["paired"]
            .as_array()
            .map(|a| a.iter().map(|p| format!("{} {:+.2} ± {:.2}", p["vs"].as_str().unwrap_or(""), p["diff"].as_f64().unwrap_or(f64::NAN), p["ci"].as_f64().unwrap_or(f64::NAN))).collect())
            .unwrap_or_default();
        let made = &b["made"];
        w(format!(
            "| {} | {} | {} | {} | {:+.2} ± {:.2} | {} | {} / {} / {} | {} |",
            b["step"],
            f(&b["active_hours"], 1),
            b["name"].as_str().unwrap_or(""),
            b["deals"],
            b["diff"].as_f64().unwrap_or(f64::NAN),
            b["ci"].as_f64().unwrap_or(f64::NAN),
            paired.join("; "),
            f(&made[0], 3),
            f(&made[1], 3),
            f(&made[2], 3),
            f(&b["zero_bids_5_8"], 3),
        ));
    }
    w(String::new());

    w("## Learner".into());
    w(String::new());
    let l = &st.learner;
    if l.is_null() {
        w("Not started.".into());
    } else {
        w(format!(
            "step {} · LR {} · P loss {} · V loss {} · replay ratio {} (target {}) · {} steps/h · waiting on the governor {} · actors' model: step {}",
            l["step"],
            l["lr"].as_f64().map_or("-".into(), |x| format!("{x:.2e}")),
            f(&l["policy_loss"], 4),
            f(&l["value_loss"], 4),
            f(&l["replay_ratio"], 2),
            if cfg.rollout.actors > 0 { cfg.rollout.ratio } else { cfg.replay.replay_ratio },
            f(&l["steps_per_hour"], 0),
            pct(&l["governor_wait"]),
            model_step,
        ));
        if cfg.rollout.actors > 0 {
            w(String::new());
            w(format!(
                "P trains on rollouts (T {}, ε {}): {} samples produced, buffer {} · replay ratio cap {}.",
                cfg.rollout.temperature,
                cfg.rollout.epsilon,
                l["rollout_produced"],
                l["rollout_examples"],
                cfg.rollout.ratio,
            ));
        }
        if sh.v_on() {
            w(String::new());
            w(format!(
                "V's own buffer (V stream, rollout next states): {} V-only updates/h (GPU {} of the time), V loss {} · ratio {} (cap {}) · buffer {} states",
                f(&l["value_only_steps_per_hour"], 0),
                pct(&l["value_only_busy"]),
                f(&l["value_only_loss"], 4),
                f(&l["value_stream_ratio"], 2),
                cfg.value_stream.ratio,
                l["value_stream_examples"],
            ));
        }
    }
    w(String::new());
    let h = &st.held_out;
    if !h.is_null() {
        let (v, t, p) = (&h["validation"], &h["train_sample"], &h["teacher_probe"]);
        w(format!("Held out at step {} (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.", h["step"]));
        w(String::new());
        w("| | validation | training sample |".into());
        w("|---|---|---|".into());
        let row = |name: &str, a: &Value, b: &Value| format!("| {name} | {} | {} |", f(a, 4), f(b, 4));
        w(row("P bid cross-entropy", &v["policy"]["bid_loss"], &t["policy"]["bid_loss"]));
        w(row("P play cross-entropy", &v["policy"]["play_loss"], &t["policy"]["play_loss"]));
        w(row("P top = search's top, bids", &v["policy"]["bid_agreement"], &t["policy"]["bid_agreement"]));
        w(row("P top = search's top, plays", &v["policy"]["play_agreement"], &t["policy"]["play_agreement"]));
        w(row("V MSE", &v["value"]["mse"], &t["value"]["mse"]));
        w(row("targets' variance", &v["value"]["variance"], &t["value"]["variance"]));
        w(row("V correlation", &v["value"]["correlation"], &t["value"]["correlation"]));
        w(row("V last-trick MSE", &v["value"]["last_trick_mse"], &t["value"]["last_trick_mse"]));
        let vs = &h["value_stream"];
        if !vs.is_null() {
            w(row("V stream: V MSE", &vs["validation"]["mse"], &vs["train_sample"]["mse"]));
            w(row("V stream: targets' variance", &vs["validation"]["variance"], &vs["train_sample"]["variance"]));
        }
        w(String::new());
        let ro = &h["rollout"];
        if !ro.is_null() {
            let (rv, rt, rn) = (&ro["validation"], &ro["train_sample"], &ro["recent"]);
            w("Rollout samples (utility units on each sample's own deal; gain = P's top move minus the playing P's, the rest of the round by that P):".into());
            w(String::new());
            w("| | validation | training sample | newest validation |".into());
            w("|---|---|---|---|".into());
            let pm = |v: &Value| format!("{} ± {}", f(&v["gain"], 4), f(&v["gain_ci"], 4));
            let vm = |v: &Value| format!("{} ± {}", f(&v["v_gain"], 4), f(&v["v_gain_ci"], 4));
            for (name, ph) in [("bids", "bids"), ("plays", "plays")] {
                w(format!("| PI loss, {name} | {} | {} | {} |", f(&rv[ph]["loss"], 4), f(&rt[ph]["loss"], 4), f(&rn[ph]["loss"], 4)));
                w(format!("| P's gain, {name} | {} | {} | {} |", pm(&rv[ph]), pm(&rt[ph]), pm(&rn[ph])));
                w(format!(
                    "| P's gain over the start's top move, {name} | | | {} ± {} ({} changed) |",
                    f(&rn[ph]["gain_vs_start"], 4),
                    f(&rn[ph]["gain_vs_start_ci"], 4),
                    pct(&rn[ph]["changed_vs_start"]),
                ));
                w(format!("| top move changed, {name} | {} | {} | {} |", pct(&rv[ph]["changed"]), pct(&rt[ph]["changed"]), pct(&rn[ph]["changed"])));
                w(format!("| V's pick gain, {name} | {} | {} | {} |", vm(&rv[ph]), vm(&rt[ph]), vm(&rn[ph])));
                w(format!("| hindsight gap, {name} | {} | {} | {} |", f(&rv[ph]["hindsight"], 4), f(&rt[ph]["hindsight"], 4), f(&rn[ph]["hindsight"], 4)));
            }
            w(String::new());
        }
        let fx = &h["fixed_value"];
        if !fx.is_null() {
            w(format!(
                "Fixed V check (validation rounds of `{}`, the same states every row): MSE {} (bids {}, plays {}), correlation {}.",
                fx["from"].as_str().unwrap_or(""),
                f(&fx["all"]["mse"], 4),
                f(&fx["bids"]["mse"], 4),
                f(&fx["plays"]["mse"], 4),
                f(&fx["all"]["correlation"], 3),
            ));
            w(String::new());
        }
        w(format!(
            "Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids {} / plays {}; V MSE {} (variance {}), correlation {}, last-trick RMSE {}.",
            f(&p["policy"]["bid_agreement"], 3),
            f(&p["policy"]["play_agreement"], 3),
            f(&p["value"]["mse"], 4),
            f(&p["value"]["variance"], 4),
            f(&p["value"]["correlation"], 3),
            p["value"]["last_trick_mse"].as_f64().map_or("-".into(), |x| format!("{:.3}", x.sqrt())),
        ));
        w(String::new());
    }
    if !st.probes.is_empty() {
        w("P's and V's change per publish, on the probe states (KL in nats; V in ŝ):".into());
        w(String::new());
        w("| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |".into());
        w("|---|---|---|---|".into());
        for p in st.probes.iter().rev().take(12).rev() {
            w(format!(
                "| {} | {} / {} | {} / {} | {} / {} |",
                p["step"],
                f(&p["kl_bid_vs_prev"], 4),
                f(&p["kl_play_vs_prev"], 4),
                f(&p["kl_bid_vs_start"], 4),
                f(&p["kl_play_vs_start"], 4),
                f(&p["v_change_vs_prev"], 4),
                f(&p["v_change_vs_start"], 4),
            ));
        }
        w(String::new());
    }

    w("## Self-play".into());
    w(String::new());
    let sp = &st.selfplay;
    if sp.is_null() {
        w("No rounds yet.".into());
    } else {
        w(format!(
            "Last {:.0} s: {} rounds/h, {} examples/h ({} forced); {} rounds, {} examples in all; buffers: training {} / {}, validation {}; {} actors, process restarts {}.",
            sp["window_secs"].as_f64().unwrap_or(0.0),
            f(&sp["rounds_per_hour"], 0),
            f(&sp["examples_per_hour"], 0),
            pct(&sp["forced"]),
            sp["rounds_total"],
            sp["examples_total"],
            sp["train_examples"],
            cfg.replay.capacity,
            sp["valid_examples"],
            cfg.run.actors,
            sp["actor_restarts"],
        ));
        w(String::new());
        if cfg.run.actors > 0 {
            w("| search health | bids | plays |".into());
            w("|---|---|---|".into());
            let (b, p) = (&sp["bid"], &sp["play"]);
            w(format!("| decisions with a choice | {} | {} |", b["decisions"], p["decisions"]));
            w(format!("| target's top ≠ P's top | {} | {} |", pct(&b["top_differs"]), pct(&p["top_differs"])));
            w(format!("| KL(target ‖ P) | {} | {} |", f(&b["kl_target_prior"], 3), f(&p["kl_target_prior"], 3)));
            w(format!("| entropy: target / P | {} / {} | {} / {} |", f(&b["target_entropy"], 3), f(&b["prior_entropy"], 3), f(&p["target_entropy"], 3), f(&p["prior_entropy"], 3)));
            w(format!("| move played ≠ target's top | {} | {} |", pct(&b["off_top_played"]), pct(&p["off_top_played"])));
            w(String::new());
        }
        let (m, z) = (&sp["made"], &sp["zero_bids"]);
        let ro = &sp["rollout"];
        if cfg.rollout.actors > 0 {
            w(format!(
                "Rollouts: {} rounds/h, {} valued decisions/h ({} bids, {} plays; moves per decision {} / {}); P's top move not the best on the deal {} / {}, hindsight gap {} / {}; {} rounds in all; buffers: training {} / {}, validation {}; {} actors.",
                f(&ro["rounds_per_hour"], 0),
                f(&ro["samples_per_hour"], 0),
                ro["bid"]["samples"],
                ro["play"]["samples"],
                f(&ro["bid"]["moves_per_sample"], 2),
                f(&ro["play"]["moves_per_sample"], 2),
                pct(&ro["bid"]["top_not_best"]),
                pct(&ro["play"]["top_not_best"]),
                f(&ro["bid"]["hindsight"], 4),
                f(&ro["play"]["hindsight"], 4),
                ro["rounds_total"],
                ro["train_samples"],
                cfg.rollout.capacity,
                ro["valid_samples"],
                cfg.rollout.actors,
            ));
            w(String::new());
        }
        let vs = &sp["value_stream"];
        if cfg.value_stream.actors > 0 {
            w(format!(
                "V stream: {} rounds/h, {} states/h; {} rounds in all; buffers: training {} / {}, validation {}; {} actors.",
                f(&vs["rounds_per_hour"], 0),
                f(&vs["states_per_hour"], 0),
                vs["rounds_total"],
                vs["train_states"],
                cfg.value_stream.capacity,
                vs["valid_states"],
                cfg.value_stream.actors,
            ));
            w(String::new());
        }
        w(format!(
            "Bids made by cards dealt {}: {} / {} / {}; 0-bids {} / {} / {}.",
            SELFPLAY_BUCKETS.join(" / "),
            f(&m[0], 3),
            f(&m[1], 3),
            f(&m[2], 3),
            f(&z[0], 3),
            f(&z[1], 3),
            f(&z[2], 3)
        ));
    }
    w(String::new());
    w("## Events".into());
    w(String::new());
    for e in st.events.iter().rev() {
        w(format!("- {e}"));
    }
    s
}

fn write_status(sh: &Shared) {
    let text = render_status(sh);
    let (path, tmp) = (sh.path(STATUS_FILE), sh.path("status.md.tmp"));
    if let Err(e) = std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, &path)) {
        eprintln!("[train] {STATUS_FILE}: {e}");
    }
}

/// Remove `path` if it exists; whether it did.
fn take(path: &Path) -> bool {
    path.exists() && std::fs::remove_file(path).is_ok()
}

fn monitor(sh: Arc<Shared>, done: Arc<AtomicBool>) {
    let tick = Duration::from_secs(2);
    let mut last = Instant::now();
    let mut window = Instant::now();
    while !done.load(SeqCst) {
        thread::sleep(tick);
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        if take(&sh.path("STOP")) {
            sh.event("STOP file: winding down; continue with --resume");
            sh.set_phase("stopping");
            sh.stop.store(true, SeqCst);
        }
        if take(&sh.path("FINISH")) {
            sh.event("FINISH file: winding down for the final evaluation");
            sh.finish.store(true, SeqCst);
            sh.stop.store(true, SeqCst);
        }
        let pause = sh.path("PAUSE").exists() && !sh.stop.load(SeqCst);
        if pause != sh.paused.load(SeqCst) {
            sh.paused.store(pause, SeqCst);
            sync_frozen(&sh);
            sh.event(if pause { "PAUSE file: actors frozen, learner idle" } else { "PAUSE removed: running" });
        }
        supervise_actors(&sh);
        let active = {
            let mut st = sh.state.lock().unwrap();
            if !pause {
                st.active_secs += dt;
            }
            st.active_secs
        };
        if !sh.stop.load(SeqCst) && active >= sh.cfg.run.hours * 3600.0 {
            sh.event(format!("{:.1} h reached: winding down for the final evaluation", sh.cfg.run.hours));
            sh.finish.store(true, SeqCst);
            sh.stop.store(true, SeqCst);
        }
        if window.elapsed().as_secs() >= sh.cfg.log.status_secs {
            let secs = window.elapsed().as_secs_f64();
            window = Instant::now();
            let w = std::mem::take(&mut *sh.window.lock().unwrap());
            let vw = std::mem::take(&mut *sh.vwindow.lock().unwrap());
            let rw = std::mem::take(&mut *sh.rwindow.lock().unwrap());
            if w.rounds > 0 || rw.rounds > 0 {
                let row = selfplay_row(&sh, &w, vw, &rw, secs);
                sh.log(&row);
                sh.status.lock().unwrap().selfplay = row;
            }
            sh.save_state();
            write_status(&sh);
        }
    }
}


// ---- the run ------------------------------------------------------------------

/// Run (or, with `resume`, continue) self-play RL in `dir`.
pub fn train(cfg: TrainConfig, dir: &Path, resume: bool) -> Result<(), String> {
    if !blobmaster().is_file() {
        return Err(format!("{} not found; build it: cargo build --release -p blob-bin", blobmaster().display()));
    }
    let run_ckpt = dir.join("checkpoint");
    recover_checkpoint(&run_ckpt).map_err(|e| format!("{}: {e}", run_ckpt.display()))?;
    let state: RunState = if resume {
        if !run_ckpt.is_dir() {
            return Err(format!("{}: no checkpoint to resume", run_ckpt.display()));
        }
        read_json(&dir.join(STATE_FILE))?
    } else {
        if dir.join(STATE_FILE).exists() || run_ckpt.exists() {
            return Err(format!("{} already holds a run; continue it with --resume", dir.display()));
        }
        for p in [&cfg.run.init_checkpoint, &cfg.run.init_model] {
            if !Path::new(p).is_dir() {
                return Err(format!("{p}: not a directory"));
            }
        }
        let ckpt: Value = read_json(&Path::new(&cfg.run.init_checkpoint).join("meta.json"))?;
        let model: Value = read_json(&Path::new(&cfg.run.init_model).join("meta.json"))?;
        if ckpt["learner_step"] != model["learner_step"] {
            return Err(format!(
                "run.init_model (step {}) is not run.init_checkpoint (step {}) exported",
                model["learner_step"], ckpt["learner_step"]
            ));
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        std::fs::write(dir.join(CONFIG_FILE), cfg.to_toml()).map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
        let every = cfg.eval.search_every_hours;
        RunState {
            active_secs: 0.0,
            model: std::fs::canonicalize(&cfg.run.init_model).map_err(|e| format!("{}: {e}", cfg.run.init_model))?,
            model_step: 0,
            next_search_secs: if every > 0.0 { every * 3600.0 } else { 1e18 },
            finished: false,
        }
    };
    for sub in [REPLAY_DIR, "models", "bench"] {
        std::fs::create_dir_all(dir.join(sub)).map_err(|e| format!("{}: {e}", dir.join(sub).display()))?;
    }
    let actors = ActorsConfig {
        actors: cfg.run.actors,
        seed: cfg.run.seed,
        mix: cfg.selfplay.mix.clone(),
        search: cfg.selfplay.search,
        chunk_rounds: cfg.replay.chunk_rounds,
        chunk_secs: cfg.replay.chunk_secs,
        value_actors: cfg.value_stream.actors,
        value_chunk_secs: cfg.value_stream.chunk_secs,
        rollout_actors: cfg.rollout.actors,
        rollout: cfg.rollout.round(),
        rollout_opponents: cfg
            .rollout
            .opponents
            .iter()
            .map(|m| match std::fs::canonicalize(m) {
                Ok(p) if p.join("policy.onnx").is_file() => Ok(p),
                Ok(p) => Err(format!("rollout opponent {}: no policy.onnx", p.display())),
                Err(e) => Err(format!("rollout opponent {m}: {e}")),
            })
            .collect::<Result<_, _>>()?,
        rollout_chunk_secs: cfg.rollout.chunk_secs,
    };
    write_json(&dir.join(ACTORS_CONFIG_FILE), &actors).map_err(|e| format!("{ACTORS_CONFIG_FILE}: {e}"))?;
    write_json(&dir.join(MODEL_POINTER_FILE), &ModelPointer { model: state.model.clone(), step: state.model_step })
        .map_err(|e| format!("{MODEL_POINTER_FILE}: {e}"))?;
    let metrics = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resume)
        .truncate(!resume)
        .open(dir.join(METRICS_FILE))
        .map_err(|e| format!("{METRICS_FILE}: {e}"))?;

    tch::manual_seed(cfg.run.seed as i64);
    tch::set_num_threads(2);
    let mut learner = Learner::new(&cfg.learner.learner())?;
    if resume {
        learner.resume(&run_ckpt)?;
    } else {
        learner.resume(Path::new(&cfg.run.init_checkpoint))?;
        learner.step = 0;
        learner.value_updates = 0;
    }
    eprintln!("[train] device {:?}, learner step {}", learner.device, learner.step);

    let t = Instant::now();
    let probe = Probe::new(cfg.eval.probe_rounds, 2 * cfg.learner.batch_size);
    let init = if resume {
        let mut first = Learner::new(&cfg.learner.learner())?;
        first.resume(Path::new(&cfg.run.init_checkpoint))?;
        probe.outputs(&first)
    } else {
        probe.outputs(&learner)
    };
    eprintln!(
        "[train] probe set: {} states ({} with a choice) from {} teacher rounds, {:.1} s",
        probe.v_idx.len(),
        probe.p_idx.len(),
        cfg.eval.probe_rounds,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let fixed = FixedRounds::load(&cfg.eval.value_rounds_from, cfg.eval.value_rounds)?;
    if let Some(f) = &fixed {
        eprintln!("[train] fixed V check: {} states from {} ({:.1} s)", f.all.len(), f.from, t.elapsed().as_secs_f64());
    }
    let start_policy = if cfg.rollout.actors > 0 {
        Some(FrozenPolicy::load(Path::new(&cfg.run.init_checkpoint), learner.device)?)
    } else {
        None
    };
    let cap = cfg.replay.capacity;
    let fr = cfg.replay.validation_fraction;
    let pi_on = cfg.rollout.actors > 0;
    let v_on = cfg.value_stream.actors > 0 || (pi_on && cfg.rollout.value_children);
    let vcap = if v_on { cfg.value_stream.capacity } else { 1 };
    let rcap = if pi_on { cfg.rollout.capacity } else { 1 };
    let valid_cap = |c: usize| ((c as f64 * fr / (1.0 - fr)).ceil() as usize).max(1000);
    let sh = Arc::new(Shared {
        cfg: cfg.clone(),
        dir: dir.to_path_buf(),
        started: Instant::now(),
        stop: AtomicBool::new(false),
        finish: AtomicBool::new(false),
        paused: AtomicBool::new(false),
        hold: AtomicBool::new(false),
        child: Mutex::new(None),
        frozen: Mutex::new(false),
        restarts: AtomicU64::new(0),
        ingest_done: AtomicBool::new(false),
        state: Mutex::new(state),
        train: SharedReplay::new(ReplayBuffer::new(cap)),
        valid: SharedReplay::new(ReplayBuffer::new(valid_cap(cap))),
        produced: AtomicU64::new(0),
        rounds: AtomicU64::new(0),
        examples: AtomicU64::new(0),
        window: Mutex::new(SelfPlayStats::default()),
        vtrain: SharedReplay::new(ReplayBuffer::new(vcap)),
        vvalid: SharedReplay::new(ReplayBuffer::new(valid_cap(vcap))),
        vproduced: AtomicU64::new(0),
        vrounds: AtomicU64::new(0),
        vsamples: AtomicU64::new(0),
        vwindow: Mutex::new((0, 0)),
        rtrain: RwLock::new(PiReplay::new(rcap)),
        rvalid: RwLock::new(PiReplay::new(valid_cap(rcap))),
        rproduced: AtomicU64::new(0),
        rrounds: AtomicU64::new(0),
        rwindow: Mutex::new(RolloutStats::default()),
        fixed,
        published_value_updates: AtomicU64::new(learner.value_updates),
        metrics: Mutex::new(metrics),
        status: Mutex::new(Status::default()),
    });
    let mut next_id = 0u64;
    if resume {
        let t = Instant::now();
        ingest_new(&sh, &mut next_id);
        *sh.window.lock().unwrap() = SelfPlayStats::default();
        eprintln!(
            "[train] reloaded {} rounds ({} training / {} validation examples kept) in {:.1} s",
            sh.rounds.load(SeqCst),
            sh.train.len(),
            sh.valid.len(),
            t.elapsed().as_secs_f64()
        );
        // The benches, probe rows and last rows so far, for status.md.
        if let Ok(text) = std::fs::read_to_string(dir.join(METRICS_FILE)) {
            let mut st = sh.status.lock().unwrap();
            for row in text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
                match row["kind"].as_str() {
                    Some("bench") => st.benches.push(row),
                    Some("probe") => st.probes.push(row),
                    Some("held_out") => st.held_out = row,
                    Some("train") => st.learner = row,
                    _ => {}
                }
            }
        }
    }
    sh.log(&json!({
        "kind": "start",
        "secs": 0.0,
        "unix": unix_secs(),
        "resume": resume,
        "step": learner.step,
        "train_examples": sh.train.len(),
        "valid_examples": sh.valid.len(),
        "rounds_reloaded": sh.rounds.load(SeqCst),
        "probe_states": probe.v_idx.len(),
    }));
    let model = sh.state.lock().unwrap().model.clone();
    sh.event(if resume {
        format!(
            "resumed at step {} with {} training / {} validation examples; actors' model {}",
            learner.step,
            sh.train.len(),
            sh.valid.len(),
            model.display()
        )
    } else {
        format!("started from {} (actors: {})", cfg.run.init_checkpoint, model.display())
    });
    sh.set_phase("starting the actors");

    *sh.child.lock().unwrap() = Some(spawn_actors(&sh)?);
    let done = Arc::new(AtomicBool::new(false));
    let mon = {
        let (sh, done) = (sh.clone(), done.clone());
        thread::spawn(move || monitor(sh, done))
    };
    let ingest_h = {
        let sh = sh.clone();
        thread::spawn(move || ingest(sh, next_id))
    };
    let (eval_tx, eval_rx) = channel::<(u64, PathBuf)>();
    let evaluator_h = {
        let sh = sh.clone();
        thread::spawn(move || evaluator(sh, eval_rx))
    };
    if !resume {
        let _ = eval_tx.send((0, model));
    }
    let (pub_tx, pub_rx) = channel::<(u64, PathBuf)>();
    let publisher_h = {
        let (sh, eval_tx) = (sh.clone(), eval_tx.clone());
        thread::spawn(move || publisher(sh, pub_rx, eval_tx))
    };
    let (batch_tx, batch_rx) = sync_channel::<Batches>(2 * cfg.replay.loader_threads);
    let loaders: Vec<JoinHandle<()>> = (0..cfg.replay.loader_threads)
        .map(|i| {
            let (sh, tx) = (sh.clone(), batch_tx.clone());
            let seed = cfg.run.seed ^ 0x10AD ^ ((learner.step << 8) + i as u64);
            if pi_on {
                thread::spawn(move || pi_loader(sh, tx, seed))
            } else {
                thread::spawn(move || loader(sh, tx, seed))
            }
        })
        .collect();
    drop(batch_tx);
    let (vbatch_tx, vbatch_rx) = sync_channel::<ValueBatch>(4);
    let value_loader_h = v_on.then(|| {
        let sh = sh.clone();
        let seed = cfg.run.seed ^ 0x7A1E ^ (learner.step << 8);
        thread::spawn(move || value_loader(sh, vbatch_tx, seed))
    });
    write_status(&sh);

    let result = learn(&sh, &mut learner, &batch_rx, &vbatch_rx, &probe, &init, start_policy.as_ref(), &pub_tx);
    sh.stop.store(true, SeqCst);
    drop(batch_rx);
    drop(vbatch_rx);
    if let Err(e) = &result {
        sh.event(format!("learner error: {e}"));
    }
    learner.save(&run_ckpt)?;
    sh.event(format!("saved the checkpoint at step {}; stopping the actors", learner.step));
    sh.set_phase("stopping the actors");
    stop_actors(&sh, Duration::from_secs(300));
    sh.ingest_done.store(true, SeqCst);
    let _ = ingest_h.join();
    for h in loaders.into_iter().chain(value_loader_h) {
        let _ = h.join();
    }
    sh.save_state();
    sh.event(format!("{} rounds, {} examples in all", sh.rounds.load(SeqCst), sh.examples.load(SeqCst)));
    result?;

    if sh.finish.load(SeqCst) {
        sh.set_phase("final evaluation");
        let last = sh.state.lock().unwrap().model_step;
        if learner.step > last || learner.value_updates > sh.published_value_updates.load(SeqCst) {
            publish(&sh, &learner, &pub_tx)?;
        }
        drop(pub_tx);
        let _ = publisher_h.join();
        drop(eval_tx);
        let _ = evaluator_h.join();
        let (step, model) = {
            let st = sh.state.lock().unwrap();
            (st.model_step, st.model.clone())
        };
        let e = &cfg.eval;
        if e.final_search {
            search_bench(&sh, step, &model, "search-rb2", "rulebot2", e.search_deals, e.search_seed, &e.search_baseline);
            search_vs_p(&sh, step, &model);
        }
        if e.final_rule_bot_deals > 0 {
            search_bench(&sh, step, &model, "search-rb", "rulebot", e.final_rule_bot_deals, DEFAULT_SEED, &e.final_rule_bot_baseline);
        }
        sh.state.lock().unwrap().finished = true;
        sh.set_phase("finished");
        sh.event(format!("finished: final model {}", model.display()));
    } else {
        // STOP: let a running export finish (seconds); a running bench is abandoned.
        drop(pub_tx);
        let _ = publisher_h.join();
        sh.set_phase("stopped (continue with --resume)");
        sh.event("stopped");
    }
    sh.save_state();
    done.store(true, SeqCst);
    let _ = mon.join();
    write_status(&sh);
    Ok(())
}
