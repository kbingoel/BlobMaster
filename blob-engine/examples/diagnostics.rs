//! Gen-1 diagnostics (2026-10-02) — the measurements behind gen-2.md §2.
//!
//! Read-only: loads ONNX checkpoints and replay buffers, never writes.
//! Fixed table: 5 players, 7 starting cards (the gen-1 training config).
//! Uses every available core; run with nothing else busy.
//!
//! ```text
//! cargo run --release -p blob-engine --example diagnostics -- <command> ...
//!
//!   match  <model.onnx> <focal> <opponent> <games>
//!       One focal seat (rotated over games) vs 4 opponents of one kind.
//!       Kinds: mcts (model + 5x100 search, greedy), raw (model policy, no
//!       search), rulebot, heuristic (HeuristicEvaluator, no search), random.
//!       Bots never run search. Reports points/game vs the opponent mean
//!       with a 95% CI, win share (fair = 0.20) and bids made.
//!
//!   value  <model.onnx> <buffer.bin> <games>
//!       Value-head error on replay-buffer positions vs fresh self-play
//!       (same recipe as run-2026-05-14), by game phase; correlation with
//!       the current round's score; bid calibration; and how often the
//!       root player's options get no value inside the search.
//!
//!   tokens <model.onnx>
//!       Mean tokens per decision today vs with every hand visible, and
//!       measured ONNX cost per batch-of-5 call by sequence length with all
//!       threads busy.
//! ```
//!
//! `match` was the prototype for `blobmaster bench` (`blob_engine::bench`),
//! which adds duplicate deals and per-hand-size bid stats; use that for new
//! measurements. This file stays to reproduce gen-2.md §2.

use blob_engine::mcts::{run_lockstep_search, TemperatureSchedule};
use blob_engine::*;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};

const NP: u8 = 5;
const NC: u8 = 7;

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

/// The run-2026-05-14 self-play recipe.
fn train_cfg() -> MctsConfig {
    MctsConfig {
        c_puct: 1.5,
        num_determinizations: 5,
        sims_per_determinization: 100,
        min_sims_floor: 60,
        temperature: 1.0,
        temperature_schedule: Some(TemperatureSchedule::HardStep { early: 1.0, late: 0.1, switch_at: 15 }),
        arena_capacity: 4096,
        target_batch: 5,
        root_dirichlet_alpha: 0.0,
        root_dirichlet_epsilon: 0.25,
    }
}

/// Same search, no root noise; the caller picks the most-visited move.
fn eval_cfg() -> MctsConfig {
    MctsConfig { temperature_schedule: None, root_dirichlet_epsilon: 0.0, ..train_cfg() }
}

fn sample(p: &[f32], rng: &mut impl Rng) -> usize {
    let t: f32 = p.iter().sum();
    let mut u = rng.gen::<f32>() * t;
    for (i, &x) in p.iter().enumerate() {
        u -= x;
        if u <= 0.0 && x > 0.0 {
            return i;
        }
    }
    p.iter().rposition(|&x| x > 0.0).unwrap_or(0)
}

/// First maximum (ties → lowest index).
fn argmax(p: &[f32]) -> usize {
    (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b })
}

fn hand_cards(s: &BlobState) -> Vec<u8> {
    Hand::new(s.hands[s.current_player as usize]).iter().map(|c| c.index()).collect()
}

fn legal_count(s: &BlobState) -> u32 {
    if s.phase() == GamePhase::Bidding {
        legal_bids(s).count_ones()
    } else {
        legal_plays(s).count_ones()
    }
}

fn apply(s: &mut BlobState, action: u8) {
    if s.phase() == GamePhase::Bidding {
        apply_bid(s, action)
    } else {
        apply_play(s, action)
    }
}

fn mse(a: &[(f32, f32)]) -> f64 {
    a.iter().map(|(p, t)| ((p - t) as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64
}

fn corr(a: &[(f32, f32)]) -> f64 {
    let n = a.len() as f64;
    let mx = a.iter().map(|x| x.0 as f64).sum::<f64>() / n;
    let my = a.iter().map(|x| x.1 as f64).sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in a {
        let (dx, dy) = (x as f64 - mx, y as f64 - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt()).max(1e-12)
}

/// Run `work(game_index, evaluator)` for `games` games across all cores,
/// one ONNX session per thread, collecting the results.
fn parallel_games<T: Send>(model: &str, games: usize, work: impl Fn(usize, &OnnxEvaluator) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let out = Mutex::new(Vec::with_capacity(games));
    std::thread::scope(|sc| {
        for _ in 0..threads() {
            sc.spawn(|| {
                let ev = OnnxEvaluator::from_file(model).expect("load onnx");
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let r = work(g, &ev);
                    out.lock().unwrap().push(r);
                }
            });
        }
    });
    out.into_inner().unwrap()
}

// ---------------------------------------------------------------------------
// value

struct Rec {
    round_idx: u8,
    phase: GamePhase,
    perspective: u8,
    v_pred: f32,
    target: f32,
    round_score: f32,
    bid: u8,
    tricks: u8,
}

#[derive(Default)]
struct QStats {
    bidding: bool,
    visited_children: u32,
    children_without_root_value: u32,
    root_visits: u32,
    root_value_count: u32,
}

/// One 5×100 search exactly like `mcts_search` (minus root noise), then
/// count how many root children received a value for the root player.
fn instrumented_search(state: &BlobState, eval: &OnnxEvaluator, rng: &mut impl Rng) -> QStats {
    let p = state.current_player;
    let voids = void_suits(state);
    let dets: Vec<BlobState> = (0..5).map(|_| determinize(state, p, &voids, rng, DEFAULT_DETERMINIZE_ATTEMPTS)).collect();
    let mut arenas: Vec<MctsArena> = (0..5).map(|_| MctsArena::with_capacity(p, 4096)).collect();
    run_lockstep_search(&mut arenas, &dets, eval, 100, 1.5, 5);
    let mut q = QStats { bidding: state.phase() == GamePhase::Bidding, ..Default::default() };
    for a in &arenas {
        let root = a.root();
        q.root_visits += root.visit_count;
        q.root_value_count += root.value_counts[p as usize];
        for &c in &root.children {
            let ch = a.node(c);
            if ch.visit_count > 0 {
                q.visited_children += 1;
                if ch.value_counts[p as usize] == 0 {
                    q.children_without_root_value += 1;
                }
            }
        }
    }
    q
}

fn play_value_game(eval: &OnnxEvaluator, rng: &mut Xoshiro256PlusPlus, instrument: bool) -> (Vec<Rec>, Vec<QStats>) {
    let cfg = train_cfg();
    let mut state = new_game(NP, NC).unwrap();
    start_round(&mut state, rng);
    let (mut recs, mut qs) = (Vec::new(), Vec::new());
    let (mut round_start, mut decision) = (0usize, 0usize);
    loop {
        match state.phase() {
            GamePhase::Bidding | GamePhase::Playing => {
                let (_, v) = eval.evaluate(&state);
                if instrument && legal_count(&state) > 1 {
                    qs.push(instrumented_search(&state, eval, rng));
                }
                let r = mcts_search(&state, eval, &cfg, rng, decision);
                decision += 1;
                let a = sample(&r.policy_sampling, rng);
                recs.push(Rec {
                    round_idx: state.round_idx,
                    phase: state.phase(),
                    perspective: state.current_player,
                    v_pred: v,
                    target: f32::NAN,
                    round_score: f32::NAN,
                    bid: 0,
                    tricks: 0,
                });
                let action = if state.phase() == GamePhase::Bidding { a as u8 } else { hand_cards(&state)[a] };
                apply(&mut state, action);
            }
            GamePhase::Scoring => {
                for r in &mut recs[round_start..] {
                    let s = r.perspective as usize;
                    r.bid = state.bids[s];
                    r.tricks = state.tricks_won[s];
                    r.round_score = if r.bid == r.tricks { 10.0 + r.bid as f32 } else { 0.0 };
                }
                round_start = recs.len();
                advance_round(&mut state, rng);
            }
            GamePhase::Complete => break,
        }
    }
    let mut scores = [0f32; MAX_PLAYERS];
    for i in 0..NP as usize {
        scores[i] = state.cumulative_scores[i] as f32;
    }
    let z = z_score_clip(&scores, NP as usize);
    for r in &mut recs {
        r.target = z[r.perspective as usize];
    }
    (recs, qs)
}

fn report_mse(label: &str, pairs: &[(f32, f32)]) {
    let zero: Vec<(f32, f32)> = pairs.iter().map(|&(_, t)| (0.0, t)).collect();
    println!(
        "{label}: MSE model {:.4}  MSE predict-0 {:.4}  corr {:.3}  n={}",
        mse(pairs),
        mse(&zero),
        corr(pairs),
        pairs.len()
    );
}

fn cmd_value(model: &str, buffer: &str, games: usize) {
    let eval = OnnxEvaluator::from_file(model).expect("load onnx");
    let buf = ReplayBuffer::load(buffer).expect("load buffer");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let mut inbuf: Vec<(f32, f32)> = Vec::new();
    for _ in 0..20 {
        let (b, p) = buf.sample_batch(1000, &mut rng);
        let states = b.states.iter().chain(p.states.iter());
        let targets = b.values.iter().chain(p.values.iter());
        for (s, &t) in states.zip(targets) {
            inbuf.push((eval.evaluate(s).1, t));
        }
    }
    report_mse("replay buffer (training data)", &inbuf);

    let results = parallel_games(model, games, |g, ev| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1000 + g as u64);
        play_value_game(ev, &mut rng, g < 16)
    });
    let mut recs: Vec<Rec> = Vec::new();
    let mut qs: Vec<QStats> = Vec::new();
    for (r, q) in results {
        recs.extend(r);
        qs.extend(q);
    }

    println!("\nfresh self-play, {games} games (never trained on):");
    report_mse("  all decisions ", &recs.iter().map(|r| (r.v_pred, r.target)).collect::<Vec<_>>());
    for (lo, hi, name) in [(0u8, 5u8, "  rounds 1-6   "), (6, 11, "  rounds 7-12  "), (12, 16, "  rounds 13-17 ")] {
        let sub: Vec<(f32, f32)> =
            recs.iter().filter(|r| r.round_idx >= lo && r.round_idx <= hi).map(|r| (r.v_pred, r.target)).collect();
        report_mse(name, &sub);
    }
    let rs: Vec<(f32, f32)> = recs.iter().map(|r| (r.v_pred, r.round_score)).collect();
    let rs_bid: Vec<(f32, f32)> =
        recs.iter().filter(|r| r.phase == GamePhase::Bidding).map(|r| (r.v_pred, r.round_score)).collect();
    let rt: Vec<(f32, f32)> = recs.iter().filter(|r| r.round_idx <= 5).map(|r| (r.round_score, r.target)).collect();
    println!("corr(value, this round's score)                  {:.3}", corr(&rs));
    println!("corr(value, this round's score), bid decisions   {:.3}", corr(&rs_bid));
    println!("corr(this round's score, game target), rounds 1-6 {:.3}", corr(&rt));

    let bids: Vec<&Rec> = recs.iter().filter(|r| r.phase == GamePhase::Bidding).collect();
    let share = |n: usize| n as f64 / bids.len() as f64;
    println!(
        "\nbid calibration (n={}): made {:.3}  won fewer {:.3}  won more {:.3}",
        bids.len(),
        share(bids.iter().filter(|r| r.bid == r.tricks).count()),
        share(bids.iter().filter(|r| r.tricks < r.bid).count()),
        share(bids.iter().filter(|r| r.tricks > r.bid).count()),
    );
    for b in 0..=NC {
        let sub: Vec<&&Rec> = bids.iter().filter(|r| r.bid == b).collect();
        if !sub.is_empty() {
            let made = sub.iter().filter(|r| r.bid == r.tricks).count();
            println!("  bid {b}: share {:.3}  made {:.3}", share(sub.len()), made as f64 / sub.len() as f64);
        }
    }

    for bidding in [true, false] {
        let sub: Vec<&QStats> = qs.iter().filter(|q| q.bidding == bidding).collect();
        if sub.is_empty() {
            continue;
        }
        let sum = |f: fn(&QStats) -> u32| sub.iter().map(|q| f(q)).sum::<u32>() as f64;
        println!(
            "\nsearch, {} decisions (n={}): visited root options with no value for the root player {:.1}%, \
             root sims that carried a root-player value {:.1}%",
            if bidding { "bid" } else { "play" },
            sub.len(),
            100.0 * sum(|q| q.children_without_root_value) / sum(|q| q.visited_children),
            100.0 * sum(|q| q.root_value_count) / sum(|q| q.root_visits),
        );
    }
}

// ---------------------------------------------------------------------------
// match

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Mcts,
    Raw,
    RuleBot,
    Heuristic,
    Random,
}

fn parse_kind(s: &str) -> Kind {
    match s {
        "mcts" => Kind::Mcts,
        "raw" => Kind::Raw,
        "rulebot" => Kind::RuleBot,
        "heuristic" => Kind::Heuristic,
        "random" => Kind::Random,
        _ => panic!("unknown player kind {s:?} (mcts | raw | rulebot | heuristic | random)"),
    }
}

fn decide(kind: Kind, s: &BlobState, ev: &OnnxEvaluator, rng: &mut Xoshiro256PlusPlus) -> u8 {
    let bidding = s.phase() == GamePhase::Bidding;
    let to_action = |idx: usize| if bidding { idx as u8 } else { hand_cards(s)[idx] };
    match kind {
        Kind::Mcts => to_action(argmax(&mcts_search(s, ev, &eval_cfg(), rng, 0).policy_target)),
        Kind::Raw => to_action(argmax(&ev.evaluate(s).0)),
        Kind::Heuristic => to_action(argmax(&HeuristicEvaluator.evaluate(s).0)),
        Kind::RuleBot => rule_bot_action(s),
        Kind::Random => {
            let mask = if bidding { legal_bids(s) as u64 } else { legal_plays(s) };
            let legal: Vec<u8> = (0..64u8).filter(|&a| (mask >> a) & 1 == 1).collect();
            legal[rng.gen_range(0..legal.len())]
        }
    }
}

struct GameOut {
    focal_score: f64,
    opp_mean_score: f64,
    focal_win: f64,
    focal_made: u32,
    opp_made: u32,
    rounds: u32,
}

fn play_match_game(focal: Kind, opp: Kind, seat: u8, ev: &OnnxEvaluator, rng: &mut Xoshiro256PlusPlus) -> GameOut {
    let mut s = new_game(NP, NC).unwrap();
    start_round(&mut s, rng);
    let (mut focal_made, mut opp_made, mut rounds) = (0, 0, 0);
    loop {
        match s.phase() {
            GamePhase::Bidding | GamePhase::Playing => {
                let kind = if s.current_player == seat { focal } else { opp };
                let a = decide(kind, &s, ev, rng);
                apply(&mut s, a);
            }
            GamePhase::Scoring => {
                rounds += 1;
                for i in 0..NP as usize {
                    let made = (s.bids[i] == s.tricks_won[i]) as u32;
                    if i == seat as usize {
                        focal_made += made;
                    } else {
                        opp_made += made;
                    }
                }
                advance_round(&mut s, rng);
            }
            GamePhase::Complete => break,
        }
    }
    let f = s.cumulative_scores[seat as usize] as f64;
    let others: Vec<f64> = (0..NP).filter(|&i| i != seat).map(|i| s.cumulative_scores[i as usize] as f64).collect();
    let best_other = others.iter().cloned().fold(f64::MIN, f64::max);
    GameOut {
        focal_score: f,
        opp_mean_score: others.iter().sum::<f64>() / others.len() as f64,
        focal_win: if f > best_other { 1.0 } else if f == best_other { 0.5 } else { 0.0 },
        focal_made,
        opp_made,
        rounds,
    }
}

fn cmd_match(model: &str, focal: Kind, opp: Kind, games: usize) {
    let o = parallel_games(model, games, |g, ev| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xABC0 + g as u64);
        play_match_game(focal, opp, (g % NP as usize) as u8, ev, &mut rng)
    });
    let n = o.len() as f64;
    let diffs: Vec<f64> = o.iter().map(|g| g.focal_score - g.opp_mean_score).collect();
    let mean = diffs.iter().sum::<f64>() / n;
    let sd = (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let rounds = o.iter().map(|g| g.rounds).sum::<u32>() as f64;
    println!(
        "{focal:?} vs 4x{opp:?}: games={}  focal {:.1}  opponents {:.1}  diff {:+.1} ± {:.1} (95%)  \
         win {:.3} (fair 0.200)  bids made focal {:.3} opponents {:.3}",
        o.len(),
        o.iter().map(|g| g.focal_score).sum::<f64>() / n,
        o.iter().map(|g| g.opp_mean_score).sum::<f64>() / n,
        mean,
        1.96 * sd / n.sqrt(),
        o.iter().map(|g| g.focal_win).sum::<f64>() / n,
        o.iter().map(|g| g.focal_made).sum::<u32>() as f64 / rounds,
        o.iter().map(|g| g.opp_made).sum::<u32>() as f64 / (rounds * (NP - 1) as f64),
    );
}

// ---------------------------------------------------------------------------
// tokens

fn cmd_tokens(model: &str) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut states: Vec<BlobState> = Vec::new();
    for _ in 0..300 {
        let mut s = new_game(NP, NC).unwrap();
        start_round(&mut s, &mut rng);
        loop {
            match s.phase() {
                GamePhase::Bidding | GamePhase::Playing => {
                    if legal_count(&s) > 1 {
                        states.push(s);
                    }
                    let a = rule_bot_action(&s);
                    apply(&mut s, a);
                }
                GamePhase::Scoring => advance_round(&mut s, &mut rng),
                GamePhase::Complete => break,
            }
        }
    }
    let today: Vec<usize> = states.iter().map(|s| encoder::encode(s, s.current_player).num_tokens).collect();
    let all_hands: Vec<usize> =
        states.iter().map(|s| (s.num_players * s.cards_dealt + s.num_players + 2) as usize).collect();
    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
    println!(
        "non-forced decisions {}: mean tokens today {:.1}, with every hand visible {:.1}",
        states.len(),
        mean(&today),
        mean(&all_hands)
    );

    let t = threads();
    for target in [12usize, 20, 26, 32, 37] {
        let pool: Vec<BlobState> =
            states.iter().zip(&today).filter(|(_, &n)| n == target).map(|(s, _)| *s).take(500).collect();
        if pool.len() < 10 {
            println!("tokens {target}: too few states");
            continue;
        }
        let calls = AtomicUsize::new(0);
        let barrier = Barrier::new(t);
        let busy_secs = Mutex::new(0.0f64);
        std::thread::scope(|sc| {
            for w in 0..t {
                let (pool, calls, barrier, busy_secs) = (&pool, &calls, &barrier, &busy_secs);
                sc.spawn(move || {
                    let ev = OnnxEvaluator::from_file(model).expect("load onnx");
                    barrier.wait();
                    let start = std::time::Instant::now();
                    let mut i = w * 5;
                    while start.elapsed().as_secs_f64() < 4.0 {
                        let batch: Vec<&BlobState> = (0..5).map(|k| &pool[(i + k) % pool.len()]).collect();
                        let _ = ev.evaluate_batch(&batch);
                        calls.fetch_add(1, Ordering::Relaxed);
                        i += 5;
                    }
                    *busy_secs.lock().unwrap() += start.elapsed().as_secs_f64();
                });
            }
        });
        let ms = 1000.0 * *busy_secs.lock().unwrap() / calls.load(Ordering::Relaxed) as f64;
        println!("tokens {target:>2}: {ms:.2} ms per batch-of-5 call ({t} threads busy)");
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize| a.get(i).map(String::as_str).unwrap_or_else(|| panic!("missing argument {i}; see file header"));
    match a.get(1).map(String::as_str) {
        Some("match") => cmd_match(arg(2), parse_kind(arg(3)), parse_kind(arg(4)), arg(5).parse().expect("games")),
        Some("value") => cmd_value(arg(2), arg(3), arg(4).parse().expect("games")),
        Some("tokens") => cmd_tokens(arg(2)),
        _ => eprintln!("usage: diagnostics match|value|tokens ... (see blob-engine/examples/diagnostics.rs header)"),
    }
}
