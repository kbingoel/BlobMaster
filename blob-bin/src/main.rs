//! blob-bin — inference / deployment CLI.
//!
//! Intentionally free of training-only dependencies (`tch`, `rayon`,
//! `indicatif`): this binary must load and run ONNX models without pulling
//! in libtorch, so it is safe to ship for the Windows + Intel iGPU target
//! in AGENTS.md.
//!
//! - `bench`: absolute strength vs fixed opponents on duplicate deals
//!   (gen-2.md §5.7, `blob_engine::bench`).
//! - `play`: one human against bots in the terminal (`play.rs`).
//!
//! A model is a directory (`policy.onnx`, `value.onnx`, `meta.json`;
//! gen-2.md §5.3), written by `blobmaster-train export`.

mod play;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use blob_engine::bench::{eval_mcts_config, run_bench, Agent, BenchConfig, Nets, DEFAULT_SEED};
use blob_engine::mcts::{MctsConfig, SearchBudget};
use blob_engine::rule_bot_2::Rollouts;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "blobmaster",
    about = "Blob inference and deployment CLI (ONNX only).",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Greedy search with the policy and value nets (`--bid-dets` ×
    /// `--bid-sims` for bids, `--dets` × `--sims` for plays).
    Search,
    /// The policy net alone, no search.
    Network,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Bot {
    Search,
    Network,
    Rulebot,
    Rulebot2,
    /// Rule bot 2 with rollouts (`--samples`, `--depth`).
    Rulebot2r,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// One focal player vs identical opponents on duplicate deals: every deal
    /// is played once from each seat. Reports points/game with a 95% CI over
    /// deals, win share and bid statistics by hand size.
    Bench {
        /// Focal player: a model directory, `rulebot`, `rulebot2` or
        /// `rulebot2r` (rule bot 2 with rollouts: `--samples`, `--depth`,
        /// `--play-only`).
        focal: String,
        /// How a focal model plays. Required unless the focal player is a rule bot.
        #[arg(long, value_enum)]
        mode: Option<Mode>,
        /// Opponents: `rulebot`, `rulebot2`, `rulebot2r`, or a model
        /// directory whose policy net plays greedily (only rule bot 2r looks
        /// ahead).
        #[arg(long, default_value = "rulebot")]
        opponent: String,
        /// Deal seeds; games = deals × players. Default 64 with search
        /// (~4.5 min for gen 1 at 5×100), 128 otherwise (~10 s).
        #[arg(long)]
        deals: Option<usize>,
        #[arg(long, default_value_t = 5)]
        players: u8,
        #[arg(long, default_value_t = 7)]
        cards: u8,
        #[command(flatten)]
        search: SearchArgs,
        #[command(flatten)]
        rollouts: RolloutArgs,
        /// Worker threads (default: every core).
        #[arg(long)]
        threads: Option<usize>,
        /// Base seed of the deal list. Keep it fixed to compare models on
        /// the same cards.
        #[arg(long, default_value_t = DEFAULT_SEED)]
        seed: u64,
    },
    /// Play a game in the terminal against bots.
    Play {
        /// Model directory for the bots. Without it the bots are rule bots.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Bot type (default: search with a model, else rulebot).
        #[arg(long, value_enum)]
        bot: Option<Bot>,
        #[arg(long, default_value_t = 5)]
        players: u8,
        #[arg(long, default_value_t = 7)]
        cards: u8,
        /// Your seat (seat 0 deals the first round and bids last).
        #[arg(long, default_value_t = 0)]
        seat: u8,
        /// Deal seed (default: from the clock).
        #[arg(long)]
        seed: Option<u64>,
        /// Print every bot decision's policy and search values. Reveals the
        /// bots' cards through their policies.
        #[arg(long)]
        show: bool,
        #[command(flatten)]
        search: SearchArgs,
        #[command(flatten)]
        rollouts: RolloutArgs,
        /// Disable ANSI colours (also off when NO_COLOR is set or stdout is
        /// not a terminal).
        #[arg(long)]
        no_color: bool,
    },
}

/// Search budgets (gen-2.md §5.4): bids default to more sampled deals and
/// fewer simulations, because a bid's value depends mostly on hidden cards.
#[derive(clap::Args, Debug, Clone, Copy)]
struct SearchArgs {
    /// Search, plays: sampled deals per decision.
    #[arg(long, default_value_t = 5)]
    dets: u32,
    /// Search, plays: simulations per sampled deal.
    #[arg(long, default_value_t = 100)]
    sims: u32,
    /// Search, bids: sampled deals per decision.
    #[arg(long, default_value_t = 20)]
    bid_dets: u32,
    /// Search, bids: simulations per sampled deal.
    #[arg(long, default_value_t = 25)]
    bid_sims: u32,
}

impl SearchArgs {
    fn config(self) -> MctsConfig {
        eval_mcts_config(SearchBudget::new(self.bid_dets, self.bid_sims), SearchBudget::new(self.dets, self.sims))
    }
}

/// Rule bot 2r settings (`blob_engine::rule_bot_2::Rollouts`).
#[derive(clap::Args, Debug, Clone, Copy)]
struct RolloutArgs {
    /// Rule bot 2r: sampled deals per decision.
    #[arg(long, default_value_t = Rollouts::default().samples)]
    samples: u32,
    /// Rule bot 2r: tricks each rollout looks ahead, counting the current
    /// one (default: to the end of the round).
    #[arg(long)]
    depth: Option<u8>,
    /// Rule bot 2r: roll out plays only; bids are rule bot 2's.
    #[arg(long)]
    play_only: bool,
}

impl RolloutArgs {
    fn config(self) -> Rollouts {
        Rollouts { samples: self.samples, depth: self.depth, bids: !self.play_only }
    }
}

fn parse_opponent(s: &str, rollouts: Rollouts) -> Agent {
    match s {
        "rulebot" => Agent::RuleBot,
        "rulebot2" => Agent::RuleBot2,
        "rulebot2r" => Agent::RuleBot2R(rollouts),
        _ => Agent::Network(PathBuf::from(s)),
    }
}

/// Exit early if `agent`'s model directory is missing or can't load (e.g.
/// it was trained on another encoder layout), instead of panicking in every
/// bench thread.
fn require_model(agent: &Agent) {
    if let Some(p) = agent.model() {
        if !p.is_dir() {
            eprintln!("error: model directory {} not found", p.display());
            std::process::exit(2);
        }
        if let Err(e) = Nets::load(agent) {
            eprintln!("error: load model {}: {e}", p.display());
            std::process::exit(2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_bench(
    focal: String,
    mode: Option<Mode>,
    opponent: String,
    deals: Option<usize>,
    players: u8,
    cards: u8,
    mcts: MctsConfig,
    rollouts: Rollouts,
    threads: Option<usize>,
    seed: u64,
) {
    let focal = match (focal.as_str(), mode) {
        ("rulebot", _) => Agent::RuleBot,
        ("rulebot2", _) => Agent::RuleBot2,
        ("rulebot2r", _) => Agent::RuleBot2R(rollouts),
        (p, Some(Mode::Search)) => Agent::Search(PathBuf::from(p)),
        (p, Some(Mode::Network)) => Agent::Network(PathBuf::from(p)),
        (_, None) => {
            eprintln!("error: --mode search|network is required for a model");
            std::process::exit(2);
        }
    };
    let opponent = parse_opponent(&opponent, rollouts);
    require_model(&focal);
    require_model(&opponent);
    let searching = matches!(focal, Agent::Search(_));
    let mut cfg = BenchConfig {
        num_players: players,
        start_cards: cards,
        deals: deals.unwrap_or(if searching { 64 } else { 128 }),
        seed,
        mcts,
        ..BenchConfig::default()
    };
    if let Some(t) = threads {
        cfg.threads = t;
    }
    if let Err(e) = blob_engine::new_game(players, cards) {
        eprintln!("error: invalid table {players} players / {cards} cards: {e:?}");
        std::process::exit(2);
    }
    eprintln!(
        "bench: {} deals x {players} seats = {} games on {} threads",
        cfg.deals,
        cfg.deals * players as usize,
        cfg.threads
    );
    let started = Instant::now();
    let last_tenth = AtomicUsize::new(0);
    let progress = |done: usize, total: usize| {
        let tenth = done * 10 / total;
        if tenth > last_tenth.fetch_max(tenth, Ordering::Relaxed) && done < total {
            eprintln!("bench: {done}/{total} games, {:.0} s", started.elapsed().as_secs_f64());
        }
    };
    let report = run_bench(&focal, &opponent, &cfg, &progress);
    println!("{report}");
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Bench { focal, mode, opponent, deals, players, cards, search, rollouts, threads, seed } => {
            cmd_bench(focal, mode, opponent, deals, players, cards, search.config(), rollouts.config(), threads, seed)
        }
        Command::Play { model, bot, players, cards, seat, seed, show, search, rollouts, no_color } => {
            let bot = match (bot, model) {
                (Some(Bot::Rulebot), _) | (None, None) => Agent::RuleBot,
                (Some(Bot::Rulebot2), _) => Agent::RuleBot2,
                (Some(Bot::Rulebot2r), _) => Agent::RuleBot2R(rollouts.config()),
                (Some(Bot::Search) | None, Some(m)) => Agent::Search(m),
                (Some(Bot::Network), Some(m)) => Agent::Network(m),
                (Some(_), None) => {
                    eprintln!("error: --bot search|network needs --model");
                    std::process::exit(2);
                }
            };
            require_model(&bot);
            let seed = seed.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            });
            let opts = play::Options {
                num_players: players,
                start_cards: cards,
                human_seat: seat,
                seed,
                bot,
                mcts: search.config(),
                show,
                color: !no_color && std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal(),
            };
            let stdin = std::io::stdin();
            if let Err(e) = play::run(&opts, stdin.lock(), std::io::stdout().lock()) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}
