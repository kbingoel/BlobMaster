//! `blobmaster play` — one human against bots in the terminal.
//!
//! The engine deals every hand, but bots only ever see their own view: the
//! network encodes the acting seat's perspective and search samples the
//! hidden hands (`belief.rs`). `--show` prints each bot's policy and value,
//! which reveals its cards through the policy; it is an analysis aid.
//!
//! The game loop is generic over its input and output so tests can drive
//! a whole game through it.

use std::io::{self, BufRead, Write};

use blob_engine::bench::{greedy_action, search_action, Agent};
use blob_engine::card::NUM_RANKS;
use blob_engine::mcts::{mcts_search, MctsConfig};
use blob_engine::rule_bot::{expected_tricks, rule_bot_action};
use blob_engine::rule_bot_2::{bid_chances, play_chances, rollout_values, rule_bot_2_action, rule_bot_2r_action};
use blob_engine::{
    advance_round, apply_bid, apply_play, legal_bids, legal_plays, new_game, start_round, total_rounds,
    BlobState, Evaluator, GamePhase, Hand, OnnxEvaluator, NO_TRUMP,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(Debug, Clone)]
pub struct Options {
    pub num_players: u8,
    pub start_cards: u8,
    pub human_seat: u8,
    pub seed: u64,
    /// Plays every other seat; also answers `hint` and `auto`.
    pub bot: Agent,
    pub mcts: MctsConfig,
    /// Print each bot decision's policy and value.
    pub show: bool,
    pub color: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The last round was scored.
    Complete,
    /// The human typed `quit` or input ended.
    Quit,
}

const HELP: &str = "\
Commands at your turn:
  <bid>        a number, e.g. 2
  <card>       rank + suit, e.g. AS, 10h, Td, q♥ (suit first works too: S10)
  <n>          while playing: the n-th card of the numbered list
  hint         what the bot would do in your seat, with its policy and value
  auto         let the bot make this move for you
  table        show bids, tricks and scores
  help         this text
  quit         leave the game
Scoring: make your bid exactly for 10 + bid points, otherwise 0.
The dealer bids last and may not bid so that the bids add up to the cards dealt.
\"value\" is the network's value estimate for that seat (−1..+1).";

enum Command {
    Number(u8),
    Card(u8),
    Hint,
    Auto,
    Table,
    Help,
    Quit,
}

fn parse_command(line: &str) -> Option<Command> {
    let t = line.trim();
    match t.to_ascii_lowercase().as_str() {
        "hint" | "?" => return Some(Command::Hint),
        "auto" => return Some(Command::Auto),
        "table" => return Some(Command::Table),
        "help" => return Some(Command::Help),
        "quit" | "exit" => return Some(Command::Quit),
        _ => {}
    }
    if let Ok(n) = t.parse::<u8>() {
        return Some(Command::Number(n));
    }
    parse_card(t).map(Command::Card)
}

/// Parse `AS`, `10h`, `Td`, `q♥`, `S10`, `♠a` … into a card index.
pub fn parse_card(s: &str) -> Option<u8> {
    let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).flat_map(char::to_uppercase).collect();
    if chars.len() < 2 {
        return None;
    }
    let suit_of = |c: char| match c {
        'S' | '♠' => Some(0u8),
        'H' | '♥' => Some(1),
        'C' | '♣' => Some(2),
        'D' | '♦' => Some(3),
        _ => None,
    };
    let rank_of = |r: &[char]| -> Option<u8> {
        match r {
            ['1', '0'] | ['T'] => Some(8),
            ['J'] => Some(9),
            ['Q'] => Some(10),
            ['K'] => Some(11),
            ['A'] => Some(12),
            [d @ '2'..='9'] => Some(*d as u8 - b'2'),
            _ => None,
        }
    };
    let last = chars.len() - 1;
    let (suit, rank) = match (suit_of(chars[last]), suit_of(chars[0])) {
        (Some(s), _) if rank_of(&chars[..last]).is_some() => (s, rank_of(&chars[..last])?),
        (_, Some(s)) => (s, rank_of(&chars[1..])?),
        _ => return None,
    };
    Some(suit * NUM_RANKS + rank)
}

const RANKS: [&str; 13] = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"];
const SUITS: [&str; 4] = ["♠", "♥", "♣", "♦"];

struct Table<R, W> {
    opts: Options,
    eval: Option<OnnxEvaluator>,
    state: BlobState,
    cards: Xoshiro256PlusPlus,
    rng: Xoshiro256PlusPlus,
    announced_round: Option<u8>,
    input: R,
    out: W,
}

impl<R: BufRead, W: Write> Table<R, W> {
    fn paint(&self, text: &str, suit: u8) -> String {
        if self.opts.color && (suit == 1 || suit == 3) {
            format!("\x1b[31m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    fn card(&self, c: u8) -> String {
        let (suit, rank) = (c / NUM_RANKS, c % NUM_RANKS);
        self.paint(&format!("{}{}", RANKS[rank as usize], SUITS[suit as usize]), suit)
    }

    fn suit(&self, suit: u8) -> String {
        if suit == NO_TRUMP {
            "none".to_string()
        } else {
            self.paint(SUITS[suit as usize], suit)
        }
    }

    /// `♠ A 10 4 · ♥ Q 7 · ♦ 3`, high cards first.
    fn hand(&self, bits: u64) -> String {
        let mut groups = Vec::new();
        for suit in 0..4u8 {
            let ranks: Vec<&str> = (0..NUM_RANKS)
                .rev()
                .filter(|r| (bits >> (suit * NUM_RANKS + r)) & 1 == 1)
                .map(|r| RANKS[r as usize])
                .collect();
            if !ranks.is_empty() {
                groups.push(self.paint(&format!("{} {}", SUITS[suit as usize], ranks.join(" ")), suit));
            }
        }
        if groups.is_empty() {
            "(empty)".to_string()
        } else {
            groups.join(" · ")
        }
    }

    fn name(&self, seat: u8) -> String {
        if seat == self.opts.human_seat {
            "You".to_string()
        } else {
            format!("P{seat}")
        }
    }

    fn has_bid(&self, seat: u8) -> bool {
        let s = &self.state;
        if s.phase() != GamePhase::Bidding {
            return true;
        }
        let n = s.num_players;
        let first = (s.dealer + 1) % n;
        let pos = |x: u8| (x + n - first) % n;
        pos(seat) < pos(s.current_player)
    }

    /// `You 1/2 · P1 0/3 …` (tricks won / bid), or just the bids with
    /// `–` for seats yet to bid.
    fn standing_line(&self, with_tricks: bool) -> String {
        let s = &self.state;
        (0..s.num_players)
            .map(|i| {
                let bid = if self.has_bid(i) { s.bids[i as usize].to_string() } else { "–".to_string() };
                if !with_tricks {
                    format!("{} {bid}", self.name(i))
                } else {
                    format!("{} {}/{bid}", self.name(i), s.tricks_won[i as usize])
                }
            })
            .collect::<Vec<_>>()
            .join(" · ")
    }

    fn scores_line(&self) -> String {
        let s = &self.state;
        (0..s.num_players)
            .map(|i| format!("{} {}", self.name(i), s.cumulative_scores[i as usize]))
            .collect::<Vec<_>>()
            .join(" · ")
    }

    fn current_trick(&self) -> String {
        let s = &self.state;
        if s.trick_cards_played == 0 {
            return "you lead".to_string();
        }
        let plays: Vec<String> = (0..s.trick_cards_played)
            .map(|i| {
                let seat = (s.trick_leader + i) % s.num_players;
                format!("{} {}", self.name(seat), self.card(s.trick_play_order[i as usize]))
            })
            .collect();
        format!("{}   ({} winning)", plays.join("  "), self.name(self.trick_winner_so_far()))
    }

    fn trick_winner_so_far(&self) -> u8 {
        let s = &self.state;
        let trump = s.trump_suit;
        let lead = s.trick_play_order[0];
        let (mut best_slot, mut best) = (0u8, lead);
        for i in 1..s.trick_cards_played {
            let c = s.trick_play_order[i as usize];
            let (cs, bs) = (c / NUM_RANKS, best / NUM_RANKS);
            let takes = if bs == trump && trump != NO_TRUMP {
                cs == trump && c % NUM_RANKS > best % NUM_RANKS
            } else if cs == trump && trump != NO_TRUMP {
                true
            } else {
                cs == lead / NUM_RANKS && c % NUM_RANKS > best % NUM_RANKS
            };
            if takes {
                best_slot = i;
                best = c;
            }
        }
        (s.trick_leader + best_slot) % s.num_players
    }

    fn legal_cards(&self) -> Vec<u8> {
        let legal = legal_plays(&self.state);
        (0..52u8).filter(|&c| (legal >> c) & 1 == 1).collect()
    }

    fn legal_bid_list(&self) -> Vec<u8> {
        let mask = legal_bids(&self.state);
        (0..=self.state.cards_dealt).filter(|&b| (mask >> b) & 1 == 1).collect()
    }

    /// Top legal actions of a dense policy: `K♠ 62% · 9♠ 30%` or `1 62% · 0 30%`.
    fn policy_line(&self, s: &BlobState, policy: &[f32]) -> String {
        let mut items: Vec<(String, f32)> = if s.phase() == GamePhase::Bidding {
            let mask = legal_bids(s);
            policy
                .iter()
                .enumerate()
                .filter(|(b, _)| (mask >> b) & 1 == 1)
                .map(|(b, &p)| (b.to_string(), p))
                .collect()
        } else {
            let legal = legal_plays(s);
            Hand::new(s.hands[s.current_player as usize])
                .iter()
                .zip(policy)
                .filter(|(c, _)| (legal >> c.index()) & 1 == 1)
                .map(|(c, &p)| (self.card(c.index()), p))
                .collect()
        };
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        items.iter().take(6).map(|(a, p)| format!("{a} {:.0}%", 100.0 * p)).collect::<Vec<_>>().join(" · ")
    }

    /// The bot's action for the current player, plus a description of its
    /// reasoning when `explain` is set.
    fn decide(&mut self, explain: bool) -> (u8, Option<String>) {
        let s = self.state;
        let p = s.current_player as usize;
        let bidding = s.phase() == GamePhase::Bidding;
        match &self.opts.bot {
            Agent::RuleBot => {
                let a = rule_bot_action(&s);
                let why = explain.then(|| {
                    if bidding {
                        format!("rule bot: expects {:.2} tricks", expected_tricks(&s))
                    } else if s.bids[p] > s.tricks_won[p] {
                        format!("rule bot: needs {} more trick(s), tries to win", s.bids[p] - s.tricks_won[p])
                    } else {
                        "rule bot: bid reached, tries to lose".to_string()
                    }
                });
                (a, why)
            }
            Agent::RuleBot2 => {
                let a = rule_bot_2_action(&s);
                let why = explain.then(|| {
                    let mut items: Vec<(String, f32)> = if bidding {
                        let mask = legal_bids(&s);
                        bid_chances(&s)
                            .iter()
                            .enumerate()
                            .filter(|(b, _)| (mask >> b) & 1 == 1)
                            .map(|(b, &p)| (format!("bid {b}"), p))
                            .collect()
                    } else {
                        play_chances(&s).iter().map(|&(c, p)| (self.card(c), p)).collect()
                    };
                    items.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let line: Vec<String> =
                        items.iter().take(6).map(|(a, p)| format!("{a} {:.0}%", 100.0 * p)).collect();
                    format!("rule bot 2: chance to make the bid: {}", line.join(" · "))
                });
                (a, why)
            }
            Agent::RuleBot2R(cfg) => {
                let cfg = *cfg;
                let a = rule_bot_2r_action(&s, &cfg, &mut self.rng);
                let why = explain.then(|| {
                    let mut items: Vec<(String, f32)> = rollout_values(&s, &cfg, &mut self.rng)
                        .iter()
                        .map(|&(x, v)| (if bidding { format!("bid {x}") } else { self.card(x) }, v))
                        .collect();
                    items.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let line: Vec<String> = items.iter().take(6).map(|(a, v)| format!("{a} {v:+.1}")).collect();
                    format!("rule bot 2r: points vs the table's mean: {}", line.join(" · "))
                });
                (a, why)
            }
            Agent::Network(_) => {
                let ev = self.eval.as_ref().expect("network bot has a model");
                let (policy, v) = ev.evaluate(&s);
                let a = greedy_action(&s, &policy);
                (a, explain.then(|| format!("network: {}   value {v:+.2}", self.policy_line(&s, &policy))))
            }
            Agent::Search(_) => {
                let ev = self.eval.as_ref().expect("search bot has a model");
                let r = mcts_search(&s, ev, &self.opts.mcts, &mut self.rng, 0);
                let a = search_action(&s, &r);
                let why = explain.then(|| {
                    let (prior, v) = ev.evaluate(&s);
                    if r.total_visits == 0 {
                        return format!("forced   network value {v:+.2}");
                    }
                    format!(
                        "search:  {}   ({} visits, value {:+.2})\n      network: {}   value {v:+.2}",
                        self.policy_line(&s, &r.policy_target),
                        r.total_visits,
                        r.value_estimate,
                        self.policy_line(&s, &prior),
                    )
                });
                (a, why)
            }
        }
    }

    fn announce_round(&mut self) -> io::Result<()> {
        let s = self.state;
        if self.announced_round == Some(s.round_idx) {
            return Ok(());
        }
        self.announced_round = Some(s.round_idx);
        let total = total_rounds(s.start_cards, s.num_players);
        writeln!(
            self.out,
            "\n── Round {} of {total} · {} card{} · trump {} · dealer {} ──",
            s.round_idx + 1,
            s.cards_dealt,
            if s.cards_dealt == 1 { "" } else { "s" },
            self.suit(s.trump_suit),
            self.name(s.dealer)
        )?;
        writeln!(self.out, "Your hand: {}", self.hand(s.hands[self.opts.human_seat as usize]))
    }

    /// Announce and apply `action` for the current player; `why` is printed
    /// under the action, before any trick or bidding summary it triggers.
    fn act(&mut self, action: u8, who: &str, why: Option<String>) -> io::Result<()> {
        let why = why.map(|w| format!("      {w}\n")).unwrap_or_default();
        if self.state.phase() == GamePhase::Bidding {
            write!(self.out, "  {who} bid{} {action}\n{why}", if who == "You" { "" } else { "s" })?;
            apply_bid(&mut self.state, action);
            if self.state.phase() == GamePhase::Playing {
                let sum: u32 = (0..self.state.num_players).map(|i| self.state.bids[i as usize] as u32).sum();
                writeln!(
                    self.out,
                    "Bids: {}   (total {sum} for {} tricks)",
                    self.standing_line(false),
                    self.state.cards_dealt
                )?;
            }
        } else {
            write!(self.out, "  {who} play{} {}\n{why}", if who == "You" { "" } else { "s" }, self.card(action))?;
            let before = self.state.tricks_completed;
            apply_play(&mut self.state, action);
            if self.state.tricks_completed > before {
                let t = self.state.trick_history[before as usize];
                let winning_card = t.cards[..t.num_played as usize].iter().find(|(p, _)| *p == t.winner).map(|x| x.1);
                writeln!(
                    self.out,
                    "  → {} win{} trick {} with {}   [{}]",
                    self.name(t.winner),
                    if t.winner == self.opts.human_seat { "" } else { "s" },
                    before + 1,
                    winning_card.map(|c| self.card(c)).unwrap_or_default(),
                    self.standing_line(true)
                )?;
            }
        }
        Ok(())
    }

    fn bot_turn(&mut self) -> io::Result<()> {
        let (a, why) = self.decide(self.opts.show);
        let who = self.name(self.state.current_player);
        self.act(a, &who, why)
    }

    fn prompt(&mut self) -> io::Result<()> {
        let s = self.state;
        let me = self.opts.human_seat as usize;
        if s.phase() == GamePhase::Bidding {
            let bids = self.legal_bid_list();
            let sum: u32 = (0..s.num_players).filter(|&i| self.has_bid(i)).map(|i| s.bids[i as usize] as u32).sum();
            writeln!(self.out, "Your hand: {}   (trump {})", self.hand(s.hands[me]), self.suit(s.trump_suit))?;
            writeln!(self.out, "Bids so far: {}   (total {sum} for {} tricks)", self.standing_line(false), s.cards_dealt)?;
            let list: Vec<String> = bids.iter().map(u8::to_string).collect();
            write!(self.out, "Your bid [{}]> ", list.join(" "))?;
        } else {
            writeln!(
                self.out,
                "Trick {} of {}: {}",
                s.tricks_completed + 1,
                s.cards_dealt,
                self.current_trick()
            )?;
            writeln!(self.out, "Tricks/bids: {}", self.standing_line(true))?;
            writeln!(self.out, "Your hand: {}   (trump {})", self.hand(s.hands[me]), self.suit(s.trump_suit))?;
            let list: Vec<String> =
                self.legal_cards().iter().enumerate().map(|(i, &c)| format!("{}) {}", i + 1, self.card(c))).collect();
            write!(self.out, "Play [{}]> ", list.join("  "))?;
        }
        self.out.flush()
    }

    /// Why `action` can't be played now, if it can't.
    fn reject(&self, cmd: &Command) -> Result<u8, String> {
        let s = &self.state;
        let me = self.opts.human_seat as usize;
        match (s.phase(), cmd) {
            (GamePhase::Bidding, Command::Number(b)) => {
                if *b > s.cards_dealt {
                    Err(format!("Bid 0 to {}.", s.cards_dealt))
                } else if (legal_bids(s) >> b) & 1 == 0 {
                    Err(format!(
                        "As dealer you can't bid {b}: the bids would add up to the {} cards dealt.",
                        s.cards_dealt
                    ))
                } else {
                    Ok(*b)
                }
            }
            (GamePhase::Bidding, Command::Card(_)) => Err("It's the bidding phase: type a number.".to_string()),
            (GamePhase::Playing, Command::Number(n)) => {
                let legal = self.legal_cards();
                match (*n as usize).checked_sub(1).and_then(|i| legal.get(i)) {
                    Some(&c) => Ok(c),
                    None => Err(format!("Pick 1 to {}, or type a card.", legal.len())),
                }
            }
            (GamePhase::Playing, Command::Card(c)) => {
                if (s.hands[me] >> c) & 1 == 0 {
                    Err(format!("You don't hold {}.", self.card(*c)))
                } else if (legal_plays(s) >> c) & 1 == 0 {
                    let led = s.trick_play_order[0] / NUM_RANKS;
                    Err(format!("You must follow {}.", self.suit(led)))
                } else {
                    Ok(*c)
                }
            }
            _ => Err("Not now.".to_string()),
        }
    }

    /// Ask until the human gives a legal action. `None` = quit.
    fn human_turn(&mut self) -> io::Result<Option<u8>> {
        let options = if self.state.phase() == GamePhase::Bidding {
            self.legal_bid_list()
        } else {
            self.legal_cards()
        };
        if options.len() == 1 {
            let what = if self.state.phase() == GamePhase::Bidding { "bid" } else { "play" };
            let shown =
                if what == "bid" { options[0].to_string() } else { self.card(options[0]) };
            writeln!(self.out, "Your only legal {what}: {shown}")?;
            return Ok(Some(options[0]));
        }
        loop {
            self.prompt()?;
            let mut line = String::new();
            if self.input.read_line(&mut line)? == 0 {
                writeln!(self.out, "\n(input closed)")?;
                return Ok(None);
            }
            if line.trim().is_empty() {
                continue;
            }
            let Some(cmd) = parse_command(&line) else {
                writeln!(self.out, "'{}' is not a bid, card or command (type help).", line.trim())?;
                continue;
            };
            match cmd {
                Command::Quit => return Ok(None),
                Command::Help => writeln!(self.out, "{HELP}")?,
                Command::Table => {
                    writeln!(self.out, "Scores: {}", self.scores_line())?;
                }
                Command::Hint => {
                    let (a, why) = self.decide(true);
                    let shown = if self.state.phase() == GamePhase::Bidding { a.to_string() } else { self.card(a) };
                    writeln!(self.out, "  hint: {shown}\n      {}", why.unwrap_or_default())?;
                }
                Command::Auto => {
                    let (a, _) = self.decide(false);
                    return Ok(Some(a));
                }
                Command::Number(_) | Command::Card(_) => match self.reject(&cmd) {
                    Ok(a) => return Ok(Some(a)),
                    Err(msg) => writeln!(self.out, "{msg}")?,
                },
            }
        }
    }

    fn round_result(&mut self) -> io::Result<()> {
        let s = self.state;
        writeln!(self.out, "Round {} result:", s.round_idx + 1)?;
        writeln!(self.out, "  {:<6}{:>5}{:>6}{:>8}{:>7}", "", "bid", "won", "points", "total")?;
        for i in 0..s.num_players as usize {
            let made = s.bids[i] == s.tricks_won[i];
            let points = if made { 10 + s.bids[i] as u16 } else { 0 };
            writeln!(
                self.out,
                "  {:<6}{:>5}{:>6}{:>8}{:>7}",
                self.name(i as u8),
                s.bids[i],
                s.tricks_won[i],
                points,
                s.cumulative_scores[i] + points
            )?;
        }
        Ok(())
    }

    fn final_standings(&mut self) -> io::Result<()> {
        let s = self.state;
        let mut order: Vec<u8> = (0..s.num_players).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(s.cumulative_scores[i as usize]));
        writeln!(self.out, "\nFinal standings after {} rounds:", total_rounds(s.start_cards, s.num_players))?;
        for (rank, &i) in order.iter().enumerate() {
            writeln!(self.out, "  {}. {:<5}{:>5}", rank + 1, self.name(i), s.cumulative_scores[i as usize])?;
        }
        Ok(())
    }
}

/// Play one game. Loads the bot's model if it has one.
pub fn run<R: BufRead, W: Write>(opts: &Options, input: R, mut out: W) -> io::Result<Finish> {
    let eval = match opts.bot.model() {
        Some(p) => Some(
            OnnxEvaluator::from_file(p)
                .map_err(|e| io::Error::other(format!("load ONNX model {}: {e}", p.display())))?,
        ),
        None => None,
    };
    let mut state = new_game(opts.num_players, opts.start_cards)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{e:?}")))?;
    if opts.human_seat >= opts.num_players {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "seat must be below the player count"));
    }
    let mut cards = Xoshiro256PlusPlus::seed_from_u64(opts.seed);
    start_round(&mut state, &mut cards);
    let bot = match &opts.bot {
        Agent::RuleBot => "the rule bot".to_string(),
        Agent::RuleBot2 => "rule bot 2".to_string(),
        Agent::RuleBot2R(cfg) => format!("rule bot 2r ({cfg})"),
        Agent::Network(p) => format!("{} (network only)", p.display()),
        Agent::Search(p) => format!(
            "{} with {}x{} search",
            p.display(),
            opts.mcts.num_determinizations,
            opts.mcts.sims_per_determinization
        ),
    };
    writeln!(
        out,
        "Blob: {} players, {} start cards, seed {}. You are seat {}; the other seats are {bot}.\nType help for commands.",
        opts.num_players, opts.start_cards, opts.seed, opts.human_seat
    )?;
    let mut t = Table {
        opts: opts.clone(),
        eval,
        state,
        cards,
        rng: Xoshiro256PlusPlus::seed_from_u64(opts.seed ^ 0x5EA2_C400),
        announced_round: None,
        input,
        out,
    };
    loop {
        match t.state.phase() {
            GamePhase::Bidding | GamePhase::Playing => {
                t.announce_round()?;
                if t.state.current_player == t.opts.human_seat {
                    match t.human_turn()? {
                        Some(a) => t.act(a, "You", None)?,
                        None => {
                            writeln!(t.out, "Game abandoned. Scores: {}", t.scores_line())?;
                            return Ok(Finish::Quit);
                        }
                    }
                } else {
                    t.bot_turn()?;
                }
            }
            GamePhase::Scoring => {
                t.round_result()?;
                advance_round(&mut t.state, &mut t.cards);
            }
            GamePhase::Complete => {
                t.final_standings()?;
                return Ok(Finish::Complete);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::bench::eval_mcts_config;

    fn opts(seed: u64) -> Options {
        Options {
            num_players: 5,
            start_cards: 7,
            human_seat: 0,
            seed,
            bot: Agent::RuleBot,
            mcts: eval_mcts_config(5, 100),
            show: true,
            color: false,
        }
    }

    #[test]
    fn parses_cards_both_ways() {
        assert_eq!(parse_card("AS"), Some(12));
        assert_eq!(parse_card("as"), Some(12));
        assert_eq!(parse_card("SA"), Some(12));
        assert_eq!(parse_card("2♠"), Some(0));
        assert_eq!(parse_card("10h"), Some(13 + 8));
        assert_eq!(parse_card("Th"), Some(13 + 8));
        assert_eq!(parse_card("h10"), Some(13 + 8));
        assert_eq!(parse_card("q♥"), Some(13 + 10));
        assert_eq!(parse_card("Kc"), Some(26 + 11));
        assert_eq!(parse_card("d3"), Some(39 + 1));
        assert_eq!(parse_card("DD"), None);
        assert_eq!(parse_card("1s"), None);
        assert_eq!(parse_card("11d"), None);
        assert_eq!(parse_card("x"), None);
    }

    fn play_auto(o: &Options, extra: &str) -> (Finish, String) {
        let input = format!("{extra}{}", "auto\n".repeat(400));
        let mut out = Vec::new();
        let f = run(o, input.as_bytes(), &mut out).unwrap();
        (f, String::from_utf8(out).unwrap())
    }

    #[test]
    fn full_game_completes_and_scores_match_rule_bot_self_play() {
        // With `auto` the human seat plays the rule bot too, so the game must
        // equal a plain rule-bot game on the same deal seed.
        let o = opts(11);
        let (f, text) = play_auto(&o, "");
        assert_eq!(f, Finish::Complete);
        assert!(text.contains("Final standings after 17 rounds"), "{text}");
        assert!(text.contains("── Round 17 of 17"));
        assert!(text.contains("rule bot: expects"), "--show output missing");

        let mut cards = Xoshiro256PlusPlus::seed_from_u64(11);
        let mut s = new_game(5, 7).unwrap();
        start_round(&mut s, &mut cards);
        while s.phase() != GamePhase::Complete {
            match s.phase() {
                GamePhase::Bidding => {
                    let b = rule_bot_action(&s);
                    apply_bid(&mut s, b)
                }
                GamePhase::Playing => {
                    let c = rule_bot_action(&s);
                    apply_play(&mut s, c)
                }
                _ => advance_round(&mut s, &mut cards),
            }
        }
        for i in 0..5u8 {
            let name = if i == 0 { "You".to_string() } else { format!("P{i}") };
            let line = format!("{name:<5}{:>5}", s.cumulative_scores[i as usize]);
            assert!(text.contains(&line), "missing final line {line:?}\n{text}");
        }
    }

    #[test]
    fn rejects_bad_input_then_accepts() {
        let o = opts(3);
        // Seat 0 is the round-1 dealer and bids last, so the first prompt
        // is a bid: a card, an out-of-range bid and junk are refused.
        let (f, text) = play_auto(&o, "AS\n9\nzzz\nhelp\nhint\ntable\n");
        assert_eq!(f, Finish::Complete);
        assert!(text.contains("It's the bidding phase"));
        assert!(text.contains("Bid 0 to 7."));
        assert!(text.contains("'zzz' is not a bid, card or command"));
        assert!(text.contains("Commands at your turn"));
        assert!(text.contains("hint: "));
        assert!(text.contains("Scores: You 0"));
    }

    #[test]
    fn explicit_moves_are_applied() {
        // Answer every prompt with the first listed option: "1" picks the
        // first legal card while playing; while bidding, the first legal bid
        // is written out explicitly by peeking at the same deal.
        let o = Options { show: false, ..opts(5) };
        let mut cards = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut s = new_game(5, 7).unwrap();
        start_round(&mut s, &mut cards);
        while s.current_player != 0 {
            let b = rule_bot_action(&s);
            apply_bid(&mut s, b);
        }
        let first_bid = (0..=7u8).find(|b| (legal_bids(&s) >> b) & 1 == 1).unwrap();
        let input = format!("{first_bid}\n{}", "1\n".repeat(200));
        let mut out = Vec::new();
        let f = run(&o, input.as_bytes(), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(&format!("  You bid {first_bid}")), "{text}");
        // Numbers are bids while bidding, so "1" may be refused by the
        // dealer rule; input then runs out mid-game, which counts as a quit.
        assert!(matches!(f, Finish::Complete | Finish::Quit));
        assert!(text.contains("  You play "));
    }

    #[test]
    fn quit_stops_the_game() {
        let o = opts(9);
        let mut out = Vec::new();
        let f = run(&o, "quit\n".as_bytes(), &mut out).unwrap();
        assert_eq!(f, Finish::Quit);
        assert!(String::from_utf8(out).unwrap().contains("Game abandoned"));
    }

    #[test]
    fn rejects_cards_not_held_or_not_following() {
        let o = opts(21);
        let mut t = Table {
            opts: o.clone(),
            eval: None,
            state: new_game(5, 7).unwrap(),
            cards: Xoshiro256PlusPlus::seed_from_u64(21),
            rng: Xoshiro256PlusPlus::seed_from_u64(0),
            announced_round: None,
            input: io::empty(),
            out: Vec::new(),
        };
        start_round(&mut t.state, &mut t.cards);
        while t.state.phase() == GamePhase::Bidding {
            let b = rule_bot_action(&t.state);
            apply_bid(&mut t.state, b);
        }
        // Play until it's seat 0's turn with a card already led.
        while !(t.state.current_player == 0 && t.state.trick_cards_played > 0) {
            let c = rule_bot_action(&t.state);
            apply_play(&mut t.state, c);
        }
        let mine = t.state.hands[0];
        let not_mine = (0..52u8).find(|c| (mine >> c) & 1 == 0).unwrap();
        assert!(t.reject(&Command::Card(not_mine)).unwrap_err().starts_with("You don't hold"));
        let legal = legal_plays(&t.state);
        if let Some(c) = (0..52u8).find(|c| (mine >> c) & 1 == 1 && (legal >> c) & 1 == 0) {
            assert!(t.reject(&Command::Card(c)).unwrap_err().starts_with("You must follow"));
        }
        let ok = (0..52u8).find(|c| (legal >> c) & 1 == 1).unwrap();
        assert_eq!(t.reject(&Command::Card(ok)), Ok(ok));
        assert_eq!(t.reject(&Command::Number(1)), Ok(ok));
    }
}
