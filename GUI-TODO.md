# GUI-TODO.md - Phases 6-7 Implementation Plan

**Status:** Planning Document
**Created:** 2025-11-15
**Target Platform:** Windows Laptop (Inference/Production)
**Dependencies:** Phase 5 ONNX Export Complete

**IMPORTANT - Implementation Scope:**
This document covers Phases 5-7 (ONNX export + GUI). However, note the following dependencies:

- ✅ **Sections 6.1-6.3** (Project Setup, Game Engine Port, State Encoder) can be implemented **NOW** while away from Ubuntu training machine
- ⚠️ **Sections 6.4-6.5** (ONNX Inference, MCTS) require trained model from Phase 1 (~3-5 days training)
- ⚠️ **Full implementation requires Phase 2** (multi-round game support, Sessions 4-5, ~8 hours implementation)
- 📖 **State Encoder:** See [docs/STATE_ENCODER_SPEC.md](../docs/STATE_ENCODER_SPEC.md) for complete 256-dimension specification

**Recommendation:** Focus on 6.1-6.3 first, then wait for Phase 1 training completion + Phase 2 implementation before proceeding with full GUI.

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Phase 5: ONNX Export Preparation](#phase-5-onnx-export-preparation)
3. [Phase 6: Backend Implementation](#phase-6-backend-implementation)
4. [Phase 7: Frontend Implementation](#phase-7-frontend-implementation)
5. [Critical Implementation Details](#critical-implementation-details)
6. [Testing Strategy](#testing-strategy)
7. [Task Breakdown & Timeline](#task-breakdown--timeline)
8. [Reference Information](#reference-information)

---

## Overview & Architecture

### Monorepo Strategy

**Decision:** Keep BlobMaster as single repository with training (Python) and production (TypeScript) code.

**Rationale:**
- ✅ Simpler for solo developer (single repo to manage)
- ✅ Documentation stays in sync
- ✅ Easy to reference Python code when porting to TypeScript
- ✅ DVC handles large model files elegantly
- ✅ Can split later if needed (after Phase 7 experience)

**Project Structure:**
```
BlobMaster/  (monorepo on both machines)
├── ml/                       # Python training (Ubuntu PC only)
├── models/
│   ├── checkpoints/
│   │   └── *.pth.dvc        # Pointers in Git, files in DVC remote
│   └── best_model.onnx.dvc  # Production model pointer
├── backend/                  # Bun + TypeScript (Windows laptop)
│   ├── src/
│   │   ├── game/            # Game engine port
│   │   ├── ml/              # State encoder + ONNX inference
│   │   ├── mcts/            # MCTS implementation
│   │   ├── api/             # REST + WebSocket endpoints
│   │   ├── db/              # SQLite queries
│   │   └── server.ts        # Entry point
│   ├── test/
│   │   ├── golden/          # Python-generated test cases
│   │   └── unit/            # Unit tests
│   ├── models/              # Symlink to ../models
│   ├── package.json
│   └── tsconfig.json
├── frontend/                 # Svelte (Windows laptop)
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/  # UI components
│   │   │   ├── stores/      # State management
│   │   │   └── api/         # API client
│   │   └── routes/          # SvelteKit pages
│   ├── package.json
│   └── svelte.config.js
├── docs/                     # Shared documentation
├── data/                     # SQLite database
└── .dvc/                     # DVC configuration
```

### Technology Stack

**Backend:**
- **Runtime:** Bun 1.x (fast startup, native TypeScript)
- **Web Framework:** Hono or Elysia (lightweight REST + WebSocket)
- **ML Inference:** `onnxruntime-node` (ONNX Runtime for Node.js/Bun)
- **Database:** `better-sqlite3` (embedded SQLite)
- **Validation:** Zod (runtime type checking)
- **Testing:** Bun test or Vitest

**Frontend:**
- **Framework:** SvelteKit (SSR, routing, type-safe)
- **State Management:** Svelte stores (reactive)
- **Styling:** TailwindCSS
- **Icons:** Lucide Svelte
- **WebSocket:** Native WebSocket or Socket.IO client
- **Testing:** Playwright (E2E), Vitest (unit)

**ML Infrastructure:**
- **Model Format:** ONNX (cross-platform)
- **Acceleration:** ONNX Runtime with OpenVINO EP (Intel iGPU) - future optimization
- **Initial Target:** CPU inference (<50ms per forward pass)

### Performance Targets

| Metric | Target | Maximum |
|--------|--------|---------|
| AI Move Time | <500ms | <1000ms |
| ONNX Inference | <50ms | <100ms |
| MCTS Simulations | 50-100 | 200 |
| WebSocket Latency | <100ms | <200ms |
| Page Load Time | <2s | <3s |

---

## Phase 5: ONNX Export Preparation

**Prerequisites:** Training complete on Ubuntu PC (Phase 4 done)

### 5.1 DVC Setup for Model Artifacts

**Goal:** Keep large model files out of Git, enable selective downloads

- [ ] **Install DVC on Ubuntu PC**
  ```bash
  pip install dvc
  ```

- [ ] **Initialize DVC in repository**
  ```bash
  dvc init
  git add .dvc/.gitignore .dvc/config
  git commit -m "Initialize DVC"
  ```

- [ ] **Configure remote storage**

  **Option A: Local external drive (simplest)**
  ```bash
  dvc remote add -d storage /mnt/usb/blobmaster_models
  dvc remote modify storage type local
  ```

  **Option B: NAS via SSH**
  ```bash
  dvc remote add -d storage ssh://nas.local/storage/blobmaster
  ```

  **Option C: AWS S3 (if available)**
  ```bash
  dvc remote add -d storage s3://my-bucket/blobmaster
  ```

- [ ] **Add model files to .gitignore**
  ```gitignore
  # .gitignore
  /models/checkpoints/*.pth
  /models/*.onnx
  ```

- [ ] **Track existing checkpoints (if any)**
  ```bash
  dvc add models/checkpoints/*.pth
  git add models/checkpoints/*.pth.dvc
  git commit -m "Track training checkpoints with DVC"
  dvc push
  ```

### 5.2 Create ONNX Export Script

**Reference:** `ml/network/model.py` (BlobNet architecture)

- [ ] **Create `ml/export_onnx.py`**

```python
"""
Export trained PyTorch model to ONNX format for production inference.

Usage:
    python ml/export_onnx.py --checkpoint models/checkpoints/best.pth --output models/best_model.onnx
"""

import argparse
import torch
import torch.onnx
from ml.network.model import BlobNet
from ml.config import get_production_config

def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    validate: bool = True
):
    """Export PyTorch model to ONNX format."""

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    config = get_production_config()
    model = BlobNet(
        state_dim=config.state_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        max_bid=config.max_bid,
        max_cards=config.max_cards
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy inputs
    print("Creating dummy inputs...")
    dummy_state = torch.randn(1, 256)  # Batch size 1, state dim 256
    dummy_mask = torch.ones(1, 52)     # Legal actions mask

    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_state, dummy_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['state', 'legal_actions_mask'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'legal_actions_mask': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

    print(f"✅ Model exported to {output_path}")

    # Validate export
    if validate:
        print("\nValidating ONNX model...")
        import onnx
        import onnxruntime as ort

        # Check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid")

        # Test inference
        session = ort.InferenceSession(output_path)

        # Compare outputs
        with torch.no_grad():
            pytorch_out = model(dummy_state, dummy_mask)

        onnx_out = session.run(
            None,
            {
                'state': dummy_state.numpy(),
                'legal_actions_mask': dummy_mask.numpy()
            }
        )

        # Check numerical equivalence
        policy_diff = torch.abs(pytorch_out[0] - torch.tensor(onnx_out[0])).max()
        value_diff = torch.abs(pytorch_out[1] - torch.tensor(onnx_out[1])).max()

        print(f"Policy max diff: {policy_diff:.6f}")
        print(f"Value max diff: {value_diff:.6f}")

        if policy_diff < 1e-5 and value_diff < 1e-5:
            print("✅ ONNX output matches PyTorch (within tolerance)")
        else:
            print("⚠️  Warning: ONNX output differs from PyTorch")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BlobNet to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best.pth",
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best_model.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step"
    )

    args = parser.parse_args()

    export_to_onnx(
        args.checkpoint,
        args.output,
        args.opset,
        validate=not args.no_validate
    )
```

- [ ] **Export final model**
  ```bash
  python ml/export_onnx.py --checkpoint models/checkpoints/best.pth --output models/best_model.onnx
  ```

- [ ] **Track ONNX model with DVC**
  ```bash
  dvc add models/best_model.onnx
  git add models/best_model.onnx.dvc
  git commit -m "Export production ONNX model"
  dvc push
  ```

### 5.3 Windows Laptop Setup

- [ ] **Clone repository on Windows**
  ```bash
  git clone <repo-url> C:\Github\BlobMaster
  cd C:\Github\BlobMaster
  ```

- [ ] **Install DVC on Windows**
  ```bash
  pip install dvc
  ```

- [ ] **Download production model only**
  ```bash
  dvc pull models/best_model.onnx
  ```

  This downloads **only** the ONNX file (~100MB), not all training checkpoints (~50GB).

- [ ] **Verify model file exists**
  ```bash
  ls models/best_model.onnx
  ```

---

## Phase 6: Backend Implementation

**Estimated Time:** 40-60 hours
**Platform:** Windows laptop with Bun

### 6.1 Project Setup (2 hours)

- [ ] **Install Bun**
  ```bash
  # Windows (PowerShell)
  irm bun.sh/install.ps1 | iex
  ```

- [ ] **Create backend directory**
  ```bash
  mkdir backend
  cd backend
  ```

- [ ] **Initialize Bun project**
  ```bash
  bun init
  ```

- [ ] **Install dependencies**
  ```bash
  # Core dependencies
  bun add hono  # or elysia for web framework
  bun add onnxruntime-node
  bun add better-sqlite3
  bun add zod

  # Development dependencies
  bun add -d @types/better-sqlite3
  bun add -d @types/node
  ```

- [ ] **Setup TypeScript config (`tsconfig.json`)**
  ```json
  {
    "compilerOptions": {
      "target": "ES2022",
      "module": "ESNext",
      "lib": ["ES2022"],
      "moduleResolution": "bundler",
      "strict": true,
      "esModuleInterop": true,
      "skipLibCheck": true,
      "forceConsistentCasingInFileNames": true,
      "resolveJsonModule": true,
      "outDir": "./dist",
      "rootDir": "./src",
      "types": ["bun-types"]
    },
    "include": ["src/**/*"],
    "exclude": ["node_modules"]
  }
  ```

- [ ] **Create directory structure**
  ```bash
  mkdir -p src/{game,ml,mcts,api,db}
  mkdir -p test/{golden,unit}
  ```

### 6.2 Game Engine Port (16 hours)

**Reference:** `ml/game/blob.py`, `ml/game/constants.py`

#### 6.2.1 Constants Module (30 min)

- [ ] **Create `src/game/constants.ts`**

```typescript
/**
 * Game constants matching Python implementation (ml/game/constants.py)
 */

export const SUITS = ["♠", "♥", "♣", "♦"] as const;
export type Suit = typeof SUITS[number];

export const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] as const;
export type Rank = typeof RANKS[number];

export const RANK_VALUES: Record<Rank, number> = {
  "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
  "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
};

export const TRUMP_ROTATION = ["♠", "♥", "♣", "♦", null] as const;
export type TrumpCard = typeof TRUMP_ROTATION[number];

export const MIN_PLAYERS = 3;
export const MAX_PLAYERS = 8;
export const SCORE_BASE = 10;

export const DECK_SIZE = 52;
export const NUM_SUITS = 4;
export const CARDS_PER_SUIT = 13;
```

#### 6.2.2 Card Class (1 hour)

**Reference:** `ml/game/blob.py` lines 64-108

- [ ] **Create `src/game/card.ts`**

```typescript
import { SUITS, RANKS, RANK_VALUES, type Suit, type Rank } from './constants.js';

export class Card {
  constructor(
    public readonly rank: Rank,
    public readonly suit: Suit
  ) {
    if (!SUITS.includes(suit)) {
      throw new Error(`Invalid suit: ${suit}`);
    }
    if (!RANKS.includes(rank)) {
      throw new Error(`Invalid rank: ${rank}`);
    }
  }

  /**
   * Numeric value for comparison (2-14)
   */
  get value(): number {
    return RANK_VALUES[this.rank];
  }

  /**
   * String representation (e.g., "A♠")
   */
  toString(): string {
    return `${this.rank}${this.suit}`;
  }

  /**
   * Equality comparison
   */
  equals(other: Card): boolean {
    return this.rank === other.rank && this.suit === other.suit;
  }

  /**
   * Convert to index (0-51) for neural network
   * Formula: suit_idx * 13 + rank_idx
   */
  toIndex(): number {
    const suitIdx = SUITS.indexOf(this.suit);
    const rankIdx = RANKS.indexOf(this.rank);
    return suitIdx * 13 + rankIdx;
  }

  /**
   * Create card from index (0-51)
   */
  static fromIndex(index: number): Card {
    if (index < 0 || index >= 52) {
      throw new Error(`Invalid card index: ${index}`);
    }
    const suitIdx = Math.floor(index / 13);
    const rankIdx = index % 13;
    return new Card(RANKS[rankIdx], SUITS[suitIdx]);
  }

  /**
   * Compare cards for sorting (by value, then suit)
   */
  compare(other: Card): number {
    if (this.value !== other.value) {
      return this.value - other.value;
    }
    return SUITS.indexOf(this.suit) - SUITS.indexOf(other.suit);
  }

  /**
   * Create a copy
   */
  clone(): Card {
    return new Card(this.rank, this.suit);
  }
}
```

- [ ] **Create `test/unit/card.test.ts`**
  - Test card creation
  - Test index conversion (to/from)
  - Test equality
  - Test sorting
  - Test invalid inputs

#### 6.2.3 Deck Class (1 hour)

**Reference:** `ml/game/blob.py` lines 115-174

- [ ] **Create `src/game/deck.ts`**

```typescript
import { Card } from './card.js';
import { SUITS, RANKS } from './constants.js';

export class Deck {
  private cards: Card[];
  private dealtCards: Set<string>;

  constructor() {
    this.cards = [];
    this.dealtCards = new Set();
    this.reset();
  }

  /**
   * Reset deck to full 52 cards
   */
  reset(): void {
    this.cards = [];
    this.dealtCards.clear();

    for (const suit of SUITS) {
      for (const rank of RANKS) {
        this.cards.push(new Card(rank, suit));
      }
    }
  }

  /**
   * Shuffle deck using Fisher-Yates algorithm
   */
  shuffle(): void {
    for (let i = this.cards.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.cards[i], this.cards[j]] = [this.cards[j], this.cards[i]];
    }
  }

  /**
   * Deal cards to players
   * @param numCards Cards per player
   * @param numPlayers Number of players
   * @returns Array of hands (each hand is array of cards)
   */
  deal(numCards: number, numPlayers: number): Card[][] {
    const totalNeeded = numCards * numPlayers;
    const remaining = this.remainingCards();

    if (totalNeeded > remaining) {
      throw new Error(
        `Cannot deal ${numCards} cards to ${numPlayers} players. Only ${remaining} cards remaining.`
      );
    }

    const hands: Card[][] = Array.from({ length: numPlayers }, () => []);

    for (let cardIdx = 0; cardIdx < numCards; cardIdx++) {
      for (let playerIdx = 0; playerIdx < numPlayers; playerIdx++) {
        const card = this.cards[this.dealtCards.size];
        this.dealtCards.add(card.toString());
        hands[playerIdx].push(card);
      }
    }

    return hands;
  }

  /**
   * Get number of undealt cards
   */
  remainingCards(): number {
    return this.cards.length - this.dealtCards.size;
  }

  /**
   * Clone the deck
   */
  clone(): Deck {
    const deck = new Deck();
    deck.cards = this.cards.map(c => c.clone());
    deck.dealtCards = new Set(this.dealtCards);
    return deck;
  }
}
```

- [ ] **Create tests for Deck**
  - Test dealing
  - Test shuffle randomness
  - Test remaining cards tracking

#### 6.2.4 Player Class (2 hours)

**Reference:** `ml/game/blob.py` lines 181-312

- [ ] **Create `src/game/player.ts`**

```typescript
import { Card } from './card.js';
import type { Suit } from './constants.js';
import { SCORE_BASE } from './constants.js';

export class Player {
  public hand: Card[] = [];
  public bid: number | null = null;
  public tricksWon: number = 0;
  public totalScore: number = 0;
  public knownVoidSuits: Set<Suit> = new Set();
  public cardsPlayed: Card[] = [];

  constructor(
    public readonly name: string,
    public readonly position: number
  ) {}

  /**
   * Add cards to hand
   */
  receiveCards(cards: Card[]): void {
    this.hand.push(...cards);
  }

  /**
   * Play a card from hand
   */
  playCard(card: Card): Card {
    const index = this.hand.findIndex(c => c.equals(card));
    if (index === -1) {
      throw new Error(`Card ${card} not in hand`);
    }

    const [playedCard] = this.hand.splice(index, 1);
    this.cardsPlayed.push(playedCard);
    return playedCard;
  }

  /**
   * Make a bid
   */
  makeBid(bid: number): void {
    if (bid < 0) {
      throw new Error(`Invalid bid: ${bid}`);
    }
    this.bid = bid;
  }

  /**
   * Increment tricks won
   */
  winTrick(): void {
    this.tricksWon++;
  }

  /**
   * Calculate score for current round
   * All-or-nothing: 10 + bid if exact, else 0
   */
  calculateRoundScore(): number {
    if (this.bid === null) {
      return 0;
    }

    return this.tricksWon === this.bid ? SCORE_BASE + this.bid : 0;
  }

  /**
   * Add round score to total
   */
  scoreRound(): void {
    const roundScore = this.calculateRoundScore();
    this.totalScore += roundScore;
  }

  /**
   * Reset state for new round (keep total score)
   */
  resetRound(): void {
    this.hand = [];
    this.bid = null;
    this.tricksWon = 0;
    this.knownVoidSuits.clear();
    this.cardsPlayed = [];
  }

  /**
   * Mark suit as void (player revealed they don't have it)
   */
  markVoidSuit(suit: Suit): void {
    this.knownVoidSuits.add(suit);
  }

  /**
   * Check if player has any cards of suit
   */
  hasSuit(suit: Suit): boolean {
    return this.hand.some(card => card.suit === suit);
  }

  /**
   * Clone the player
   */
  clone(): Player {
    const player = new Player(this.name, this.position);
    player.hand = this.hand.map(c => c.clone());
    player.bid = this.bid;
    player.tricksWon = this.tricksWon;
    player.totalScore = this.totalScore;
    player.knownVoidSuits = new Set(this.knownVoidSuits);
    player.cardsPlayed = this.cardsPlayed.map(c => c.clone());
    return player;
  }
}
```

- [ ] **Create tests for Player**
  - Test card dealing and playing
  - Test bidding
  - Test scoring (exact match vs miss)
  - Test void suit tracking

#### 6.2.5 Trick Class (2 hours)

**Reference:** `ml/game/blob.py` lines 319-448

- [ ] **Create `src/game/trick.ts`**

```typescript
import { Card } from './card.js';
import { Player } from './player.js';
import type { Suit } from './constants.js';

export class Trick {
  public cardsPlayed: Array<[Player, Card]> = [];
  public ledSuit: Suit | null = null;
  public winner: Player | null = null;

  constructor(public readonly trumpSuit: Suit | null = null) {}

  /**
   * Add a card to the trick
   */
  addCard(player: Player, card: Card): void {
    if (this.cardsPlayed.length === 0) {
      this.ledSuit = card.suit;
    }

    this.cardsPlayed.push([player, card]);
  }

  /**
   * Determine winner of completed trick
   *
   * Priority:
   * 1. Highest trump (if trump suit exists)
   * 2. Highest card in led suit
   */
  determineWinner(): Player {
    if (this.cardsPlayed.length === 0) {
      throw new Error("Cannot determine winner of empty trick");
    }

    let winningPlayer = this.cardsPlayed[0][0];
    let winningCard = this.cardsPlayed[0][1];

    for (let i = 1; i < this.cardsPlayed.length; i++) {
      const [player, card] = this.cardsPlayed[i];

      // Trump logic
      if (this.trumpSuit !== null) {
        const winningIsTrump = winningCard.suit === this.trumpSuit;
        const currentIsTrump = card.suit === this.trumpSuit;

        if (currentIsTrump && !winningIsTrump) {
          // Current card is trump, winning card is not
          winningPlayer = player;
          winningCard = card;
        } else if (currentIsTrump && winningIsTrump) {
          // Both are trump, higher value wins
          if (card.value > winningCard.value) {
            winningPlayer = player;
            winningCard = card;
          }
        } else if (!currentIsTrump && !winningIsTrump) {
          // Neither is trump, check led suit
          if (card.suit === this.ledSuit && card.value > winningCard.value) {
            winningPlayer = player;
            winningCard = card;
          }
        }
        // else: winning is trump, current is not -> no change
      } else {
        // No trump, highest in led suit wins
        if (card.suit === this.ledSuit && card.value > winningCard.value) {
          winningPlayer = player;
          winningCard = card;
        }
      }
    }

    this.winner = winningPlayer;
    return winningPlayer;
  }

  /**
   * Get the winning card
   */
  getWinningCard(): Card | null {
    if (this.winner === null) {
      return null;
    }

    for (const [player, card] of this.cardsPlayed) {
      if (player === this.winner) {
        return card;
      }
    }

    return null;
  }

  /**
   * Check if trick is complete
   */
  isComplete(numPlayers: number): boolean {
    return this.cardsPlayed.length === numPlayers;
  }

  /**
   * Clear trick for reuse
   */
  clear(): void {
    this.cardsPlayed = [];
    this.ledSuit = null;
    this.winner = null;
  }

  /**
   * Clone the trick
   */
  clone(): Trick {
    const trick = new Trick(this.trumpSuit);
    trick.cardsPlayed = this.cardsPlayed.map(([p, c]) => [p.clone(), c.clone()]);
    trick.ledSuit = this.ledSuit;
    trick.winner = this.winner?.clone() ?? null;
    return trick;
  }
}
```

- [ ] **Create tests for Trick**
  - Test winner determination (trump vs non-trump)
  - Test led suit tracking
  - Test edge cases (all trump, no trump)

#### 6.2.6 BlobGame Class (8 hours)

**Reference:** `ml/game/blob.py` lines 455-1808

This is the most complex class. Break into sub-tasks:

- [ ] **Create `src/game/blob-game.ts` - Part 1: Core state**

```typescript
import { Card } from './card.js';
import { Deck } from './deck.js';
import { Player } from './player.js';
import { Trick } from './trick.js';
import { TRUMP_ROTATION, type TrumpCard, type Suit } from './constants.js';

export type GamePhase = 'setup' | 'bidding' | 'playing' | 'scoring' | 'complete';

export interface RoundResult {
  roundNumber: number;
  trumpSuit: Suit | null;
  scores: number[];
  totalScores: number[];
}

export class BlobGame {
  public players: Player[];
  public deck: Deck;
  public currentRound: number = 0;
  public trumpSuit: Suit | null = null;
  public dealerPosition: number = 0;
  public currentTrick: Trick | null = null;
  public tricksHistory: Trick[] = [];
  public gamePhase: GamePhase = 'setup';
  public cardsPlayedThisRound: Card[] = [];
  public cardsRemainingBySuit: Map<Suit, number> = new Map();

  constructor(public readonly numPlayers: number) {
    if (numPlayers < 3 || numPlayers > 8) {
      throw new Error(`Invalid number of players: ${numPlayers}. Must be 3-8.`);
    }

    this.players = [];
    for (let i = 0; i < numPlayers; i++) {
      this.players.push(new Player(`Player ${i}`, i));
    }

    this.deck = new Deck();
  }

  /**
   * Determine trump suit based on round number
   * Rotation: ♠ → ♥ → ♣ → ♦ → None → repeat
   */
  determineTrump(): Suit | null {
    const trumpIndex = this.currentRound % 5;
    return TRUMP_ROTATION[trumpIndex];
  }

  // ... (continued in next section)
}
```

- [ ] **Part 2: Round setup and bidding**

```typescript
  /**
   * Setup round: reset, deal cards, determine trump
   */
  setupRound(cardsToDeal: number): void {
    // Reset deck and shuffle
    this.deck.reset();
    this.deck.shuffle();

    // Reset players for new round
    for (const player of this.players) {
      player.resetRound();
    }

    // Deal cards
    const hands = this.deck.deal(cardsToDeal, this.numPlayers);
    for (let i = 0; i < this.numPlayers; i++) {
      this.players[i].receiveCards(hands[i]);
    }

    // Determine trump
    this.trumpSuit = this.determineTrump();

    // Initialize tracking
    this.cardsPlayedThisRound = [];
    this.currentTrick = null;
    this.tricksHistory = [];

    // Count cards by suit
    this.cardsRemainingBySuit.clear();
    for (const player of this.players) {
      for (const card of player.hand) {
        const count = this.cardsRemainingBySuit.get(card.suit) ?? 0;
        this.cardsRemainingBySuit.set(card.suit, count + 1);
      }
    }

    // Set phase
    this.gamePhase = 'bidding';
  }

  /**
   * Get forbidden bid for dealer (bid that makes total = cards dealt)
   */
  getForbiddenBid(currentTotalBids: number, cardsDealt: number): number | null {
    const forbiddenBid = cardsDealt - currentTotalBids;

    if (forbiddenBid >= 0 && forbiddenBid <= cardsDealt) {
      return forbiddenBid;
    }

    return null;
  }

  /**
   * Check if bid is valid
   */
  isValidBid(
    bid: number,
    isDealer: boolean,
    currentTotalBids: number,
    cardsDealt: number
  ): boolean {
    // Range check
    if (bid < 0 || bid > cardsDealt) {
      return false;
    }

    // Dealer constraint
    if (isDealer) {
      const forbiddenBid = this.getForbiddenBid(currentTotalBids, cardsDealt);
      if (forbiddenBid !== null && bid === forbiddenBid) {
        return false;
      }
    }

    return true;
  }

  /**
   * Get legal bids for current player
   */
  getLegalBids(cardsDealt: number): number[] {
    const currentPlayer = this.getCurrentPlayer();
    if (!currentPlayer) {
      return [];
    }

    const isDealer = currentPlayer.position === this.dealerPosition;
    const currentTotalBids = this.players
      .filter(p => p.bid !== null)
      .reduce((sum, p) => sum + p.bid!, 0);

    const legalBids: number[] = [];
    for (let bid = 0; bid <= cardsDealt; bid++) {
      if (this.isValidBid(bid, isDealer, currentTotalBids, cardsDealt)) {
        legalBids.push(bid);
      }
    }

    return legalBids;
  }
```

- [ ] **Part 3: Card playing and legal moves**

```typescript
  /**
   * Get legal cards to play
   */
  getLegalPlays(player: Player, ledSuit: Suit | null): Card[] {
    // No led suit yet (first to play) -> any card
    if (ledSuit === null) {
      return [...player.hand];
    }

    // Must follow suit if possible
    const cardsInLedSuit = player.hand.filter(c => c.suit === ledSuit);

    if (cardsInLedSuit.length > 0) {
      return cardsInLedSuit;
    }

    // Can't follow suit -> any card
    return [...player.hand];
  }

  /**
   * Validate play with anti-cheat
   */
  validatePlay(card: Card, player: Player, ledSuit: Suit | null): void {
    // Check card in hand
    if (!player.hand.some(c => c.equals(card))) {
      throw new Error(`Player ${player.name} does not have card ${card}`);
    }

    // Check follow-suit rule
    if (ledSuit !== null && player.hasSuit(ledSuit) && card.suit !== ledSuit) {
      throw new Error(
        `Player ${player.name} must follow suit ${ledSuit} but played ${card.suit}`
      );
    }
  }
```

- [ ] **Part 4: Game state queries**

```typescript
  /**
   * Get current player whose turn it is
   */
  getCurrentPlayer(): Player | null {
    if (this.gamePhase === 'bidding') {
      // First player without a bid
      for (const player of this.players) {
        if (player.bid === null) {
          return player;
        }
      }
      return null; // All bids in
    }

    if (this.gamePhase === 'playing') {
      if (!this.currentTrick) {
        // Start new trick - winner of last trick leads (or dealer for first)
        if (this.tricksHistory.length === 0) {
          // First trick - player after dealer
          return this.players[(this.dealerPosition + 1) % this.numPlayers];
        } else {
          // Winner of last trick
          return this.tricksHistory[this.tricksHistory.length - 1].winner;
        }
      } else {
        // Continue current trick
        if (this.currentTrick.isComplete(this.numPlayers)) {
          return null; // Trick complete
        }

        // Next player in rotation from trick start
        const firstPlayer = this.currentTrick.cardsPlayed[0][0];
        const nextPosition =
          (firstPlayer.position + this.currentTrick.cardsPlayed.length) % this.numPlayers;
        return this.players[nextPosition];
      }
    }

    return null;
  }

  /**
   * Get legal actions for player (bids or cards)
   */
  getLegalActions(player: Player): (number | Card)[] {
    if (this.gamePhase === 'bidding') {
      // Determine cards dealt from hand size
      const cardsDealt = player.hand.length;
      return this.getLegalBids(cardsDealt);
    }

    if (this.gamePhase === 'playing') {
      const ledSuit = this.currentTrick?.ledSuit ?? null;
      return this.getLegalPlays(player, ledSuit);
    }

    return [];
  }

  /**
   * Get game state from player's perspective (imperfect information)
   */
  getGameState(playerPerspective: Player): any {
    return {
      numPlayers: this.numPlayers,
      currentRound: this.currentRound,
      trumpSuit: this.trumpSuit,
      dealerPosition: this.dealerPosition,
      phase: this.gamePhase,
      myHand: playerPerspective.hand.map(c => c.toString()),
      myBid: playerPerspective.bid,
      myTricksWon: playerPerspective.tricksWon,
      myTotalScore: playerPerspective.totalScore,
      players: this.players.map(p => ({
        name: p.name,
        position: p.position,
        bid: p.bid,
        tricksWon: p.tricksWon,
        totalScore: p.totalScore,
        cardsInHand: p.hand.length,
        knownVoidSuits: Array.from(p.knownVoidSuits)
      })),
      currentTrick: this.currentTrick ? {
        cardsPlayed: this.currentTrick.cardsPlayed.map(([p, c]) => ({
          player: p.position,
          card: c.toString()
        })),
        ledSuit: this.currentTrick.ledSuit,
        winner: this.currentTrick.winner?.position ?? null
      } : null,
      tricksHistory: this.tricksHistory.length,
      legalActions: this.getLegalActions(playerPerspective)
    };
  }
```

- [ ] **Part 5: Action application (for MCTS)**

```typescript
  /**
   * Apply action to game state (for MCTS simulation)
   * @param action - Bid value (0-13) or card index (0-51)
   * @param player - Player making the action
   */
  applyAction(action: number, player: Player): void {
    if (this.gamePhase === 'bidding') {
      // Action is bid value
      player.makeBid(action);

      // Check if all bids in
      if (this.players.every(p => p.bid !== null)) {
        this.gamePhase = 'playing';
      }
    } else if (this.gamePhase === 'playing') {
      // Action is card index (0-51)
      const card = Card.fromIndex(action);

      // Validate and play
      const ledSuit = this.currentTrick?.ledSuit ?? null;
      this.validatePlay(card, player, ledSuit);

      // Create trick if needed
      if (!this.currentTrick) {
        this.currentTrick = new Trick(this.trumpSuit);
      }

      // Play card
      const playedCard = player.playCard(card);
      this.currentTrick.addCard(player, playedCard);
      this.cardsPlayedThisRound.push(playedCard);

      // Check if trick complete
      if (this.currentTrick.isComplete(this.numPlayers)) {
        const winner = this.currentTrick.determineWinner();
        winner.winTrick();
        this.tricksHistory.push(this.currentTrick);
        this.currentTrick = null;

        // Check if round complete
        if (this.players.every(p => p.hand.length === 0)) {
          this.gamePhase = 'scoring';
        }
      }
    }
  }

  /**
   * Deep clone game state
   */
  clone(): BlobGame {
    const game = new BlobGame(this.numPlayers);
    game.players = this.players.map(p => p.clone());
    game.deck = this.deck.clone();
    game.currentRound = this.currentRound;
    game.trumpSuit = this.trumpSuit;
    game.dealerPosition = this.dealerPosition;
    game.currentTrick = this.currentTrick?.clone() ?? null;
    game.tricksHistory = this.tricksHistory.map(t => t.clone());
    game.gamePhase = this.gamePhase;
    game.cardsPlayedThisRound = this.cardsPlayedThisRound.map(c => c.clone());
    game.cardsRemainingBySuit = new Map(this.cardsRemainingBySuit);
    return game;
  }
}
```

- [ ] **Create comprehensive tests for BlobGame**
  - Test round setup
  - Test bidding (including dealer constraint)
  - Test card playing (follow suit rules)
  - Test scoring
  - Test phase transitions
  - Test game state queries
  - Test action application

### 6.3 State Encoder (8 hours)

**Reference:** `ml/network/encode.py`

- [ ] **Create `src/ml/state-encoder.ts`**

```typescript
import { BlobGame } from '../game/blob-game.js';
import { Player } from '../game/player.js';
import { Card } from '../game/card.js';
import { SUITS, RANKS } from '../game/constants.js';

/**
 * Encode game state into 256-dim tensor for neural network
 *
 * MUST match Python implementation exactly (ml/network/encode.py)
 */
export class StateEncoder {
  private static readonly STATE_DIM = 256;

  /**
   * Convert card to index (0-51)
   * Formula: suit_idx * 13 + rank_idx
   */
  private cardToIndex(card: Card): number {
    return card.toIndex();
  }

  /**
   * Normalize bid value to [-1, 1]
   * -1 if no bid yet, otherwise [0, 1]
   */
  private normalizeBid(bid: number | null, maxBid: number): number {
    if (bid === null) {
      return -1.0;
    }
    return bid / maxBid;
  }

  /**
   * Main encoding function
   * Returns Float32Array of 256 dimensions
   */
  encode(game: BlobGame, player: Player): Float32Array {
    const state = new Float32Array(StateEncoder.STATE_DIM);
    let offset = 0;

    // 1. My Hand (52-dim binary)
    for (const card of player.hand) {
      state[offset + this.cardToIndex(card)] = 1.0;
    }
    offset += 52;

    // 2. Cards Played This Trick (52-dim sequential)
    if (game.currentTrick) {
      for (let i = 0; i < game.currentTrick.cardsPlayed.length; i++) {
        const [_, card] = game.currentTrick.cardsPlayed[i];
        state[offset + this.cardToIndex(card)] = i + 1;
      }
    }
    offset += 52;

    // 3. All Cards Played This Round (52-dim binary)
    for (const card of game.cardsPlayedThisRound) {
      state[offset + this.cardToIndex(card)] = 1.0;
    }
    offset += 52;

    // 4. Player Bids (8-dim)
    const cardsDealt = player.hand.length + player.cardsPlayed.length;
    for (let i = 0; i < 8; i++) {
      if (i < game.numPlayers) {
        state[offset + i] = this.normalizeBid(game.players[i].bid, cardsDealt);
      } else {
        state[offset + i] = -1.0; // Padding for unused players
      }
    }
    offset += 8;

    // 5. Player Tricks Won (8-dim)
    for (let i = 0; i < 8; i++) {
      if (i < game.numPlayers) {
        const tricksNorm = cardsDealt > 0 ? game.players[i].tricksWon / cardsDealt : 0;
        state[offset + i] = tricksNorm;
      } else {
        state[offset + i] = 0.0; // Padding
      }
    }
    offset += 8;

    // 6. My Bid (1-dim)
    state[offset] = this.normalizeBid(player.bid, cardsDealt);
    offset += 1;

    // 7. My Tricks Won (1-dim)
    state[offset] = cardsDealt > 0 ? player.tricksWon / cardsDealt : 0;
    offset += 1;

    // 8. Round Metadata (8-dim)
    state[offset] = cardsDealt / 13.0; // Cards dealt normalized
    offset += 1;

    const trickNum = game.tricksHistory.length;
    state[offset] = cardsDealt > 0 ? trickNum / cardsDealt : 0; // Trick number normalized
    offset += 1;

    state[offset] = player.position / game.numPlayers; // Position normalized
    offset += 1;

    state[offset] = game.numPlayers / 8.0; // Num players normalized
    offset += 1;

    // Trump suit one-hot (4-dim)
    if (game.trumpSuit !== null) {
      const trumpIdx = SUITS.indexOf(game.trumpSuit);
      state[offset + trumpIdx] = 1.0;
    }
    offset += 4;

    // 9. Bidding Constraint (1-dim)
    const isDealer = player.position === game.dealerPosition;
    state[offset] = isDealer && game.gamePhase === 'bidding' ? 1.0 : 0.0;
    offset += 1;

    // 10. Game Phase (3-dim one-hot)
    if (game.gamePhase === 'bidding') {
      state[offset] = 1.0;
    } else if (game.gamePhase === 'playing') {
      state[offset + 1] = 1.0;
    } else if (game.gamePhase === 'scoring') {
      state[offset + 2] = 1.0;
    }
    offset += 3;

    // 11-12. Positional and Context Features (remaining dimensions)
    // IMPORTANT: See docs/STATE_ENCODER_SPEC.md for complete specification of all 256 dimensions
    // Reference implementation: ml/network/encode.py
    // Must match Python exactly - DO NOT pad with zeros, implement all features

    return state;
  }
}
```

- [ ] **Create golden tests for state encoding**
  - Generate test cases from Python
  - Compare encodings exactly (within 1e-6 tolerance)

- [ ] **Create `src/ml/action-masker.ts`**

```typescript
import { BlobGame } from '../game/blob-game.js';
import { Player } from '../game/player.js';
import type { Suit } from '../game/constants.js';

export class ActionMasker {
  /**
   * Create legal actions mask for bidding
   * Returns Float32Array of 52 dims (only use first cardsDealt+1)
   */
  createBiddingMask(
    cardsDealt: number,
    isDealer: boolean,
    forbiddenBid: number | null
  ): Float32Array {
    const mask = new Float32Array(52).fill(0);

    for (let bid = 0; bid <= cardsDealt; bid++) {
      if (bid !== forbiddenBid) {
        mask[bid] = 1.0;
      }
    }

    return mask;
  }

  /**
   * Create legal actions mask for playing
   * Returns Float32Array of 52 dims (1 for legal cards)
   */
  createPlayingMask(
    hand: Card[],
    ledSuit: Suit | null,
    encoder: StateEncoder
  ): Float32Array {
    const mask = new Float32Array(52).fill(0);

    // Get legal cards
    const legalCards = ledSuit !== null && hand.some(c => c.suit === ledSuit)
      ? hand.filter(c => c.suit === ledSuit)
      : hand;

    // Set mask
    for (const card of legalCards) {
      const idx = encoder['cardToIndex'](card); // Access private method
      mask[idx] = 1.0;
    }

    return mask;
  }
}
```

### 6.4 ONNX Inference (6 hours)

- [ ] **Install ONNX Runtime**
  ```bash
  bun add onnxruntime-node
  ```

- [ ] **Create `src/ml/onnx-inference.ts`**

```typescript
import * as ort from 'onnxruntime-node';
import { StateEncoder } from './state-encoder.js';
import { ActionMasker } from './action-masker.js';
import { BlobGame } from '../game/blob-game.js';
import { Player } from '../game/player.js';

export interface InferenceResult {
  policy: Float32Array;  // 52-dim action probabilities
  value: number;         // Score prediction [-1, 1]
}

export class ONNXInference {
  private session: ort.InferenceSession | null = null;
  private encoder: StateEncoder;
  private masker: ActionMasker;

  constructor() {
    this.encoder = new StateEncoder();
    this.masker = new ActionMasker();
  }

  /**
   * Load ONNX model
   */
  async load(modelPath: string): Promise<void> {
    console.log(`Loading ONNX model from ${modelPath}...`);

    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu'], // Start with CPU
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true
    });

    console.log('✅ ONNX model loaded successfully');
  }

  /**
   * Run inference on game state
   */
  async predict(
    game: BlobGame,
    player: Player
  ): Promise<InferenceResult> {
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Encode state
    const state = this.encoder.encode(game, player);

    // Create legal actions mask
    const mask = this.createMask(game, player);

    // Prepare inputs
    const feeds = {
      state: new ort.Tensor('float32', state, [1, 256]),
      legal_actions_mask: new ort.Tensor('float32', mask, [1, 52])
    };

    // Run inference
    const startTime = performance.now();
    const outputs = await this.session.run(feeds);
    const inferenceTime = performance.now() - startTime;

    if (inferenceTime > 100) {
      console.warn(`⚠️ Slow inference: ${inferenceTime.toFixed(1)}ms`);
    }

    // Extract results
    const policy = outputs.policy.data as Float32Array;
    const value = (outputs.value.data as Float32Array)[0];

    return { policy, value };
  }

  /**
   * Create legal actions mask based on game phase
   */
  private createMask(game: BlobGame, player: Player): Float32Array {
    if (game.gamePhase === 'bidding') {
      const cardsDealt = player.hand.length;
      const isDealer = player.position === game.dealerPosition;
      const currentTotalBids = game.players
        .filter(p => p.bid !== null)
        .reduce((sum, p) => sum + p.bid!, 0);
      const forbiddenBid = game.getForbiddenBid(currentTotalBids, cardsDealt);

      return this.masker.createBiddingMask(cardsDealt, isDealer, forbiddenBid);
    } else if (game.gamePhase === 'playing') {
      const ledSuit = game.currentTrick?.ledSuit ?? null;
      return this.masker.createPlayingMask(player.hand, ledSuit, this.encoder);
    }

    return new Float32Array(52).fill(0);
  }

  /**
   * Get best action (greedy)
   */
  getBestAction(policy: Float32Array, legalActions: (number | Card)[]): number {
    let maxProb = -Infinity;
    let bestAction = -1;

    for (const action of legalActions) {
      const actionIdx = typeof action === 'number' ? action : action.toIndex();
      if (policy[actionIdx] > maxProb) {
        maxProb = policy[actionIdx];
        bestAction = actionIdx;
      }
    }

    return bestAction;
  }

  /**
   * Sample action from policy (stochastic)
   */
  sampleAction(policy: Float32Array, legalActions: (number | Card)[], temperature: number = 1.0): number {
    // Apply temperature
    const logits = Array.from(policy).map((p, i) => {
      const isLegal = legalActions.some(a =>
        (typeof a === 'number' ? a : a.toIndex()) === i
      );
      return isLegal ? Math.log(p + 1e-10) / temperature : -Infinity;
    });

    // Softmax
    const maxLogit = Math.max(...logits.filter(l => l !== -Infinity));
    const exps = logits.map(l => l === -Infinity ? 0 : Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    const probs = exps.map(e => e / sumExps);

    // Sample
    const rand = Math.random();
    let cumProb = 0;
    for (let i = 0; i < probs.length; i++) {
      cumProb += probs[i];
      if (rand < cumProb) {
        return i;
      }
    }

    return legalActions[0] as number; // Fallback
  }
}
```

- [ ] **Test ONNX inference**
  - Load model successfully
  - Compare output with Python (same input → same output)
  - Measure inference time (<50ms target)

### 6.5 MCTS Implementation (12 hours)

**Reference:** `ml/mcts/`

#### 6.5.1 MCTS Node (2 hours)

- [ ] **Create `src/mcts/mcts-node.ts`**

```typescript
import { BlobGame } from '../game/blob-game.js';
import { Player } from '../game/player.js';

export class MCTSNode {
  public children: Map<number, MCTSNode> = new Map();
  public visitCount: number = 0;
  public totalValue: number = 0;
  public priorProb: number = 0;

  constructor(
    public gameState: BlobGame,
    public player: Player,
    public parent: MCTSNode | null = null
  ) {}

  /**
   * Mean value (Q-value)
   */
  get meanValue(): number {
    return this.visitCount > 0 ? this.totalValue / this.visitCount : 0;
  }

  /**
   * Is leaf node (not expanded)?
   */
  get isLeaf(): boolean {
    return this.children.size === 0;
  }

  /**
   * Select best child using UCB1 formula
   * UCB(child) = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
   */
  selectChild(cPuct: number): MCTSNode {
    let bestScore = -Infinity;
    let bestChild: MCTSNode | null = null;

    for (const child of this.children.values()) {
      const qValue = child.meanValue;
      const uValue = cPuct * child.priorProb *
        Math.sqrt(this.visitCount) / (1 + child.visitCount);
      const ucbScore = qValue + uValue;

      if (ucbScore > bestScore) {
        bestScore = ucbScore;
        bestChild = child;
      }
    }

    if (!bestChild) {
      throw new Error('No children to select from');
    }

    return bestChild;
  }

  /**
   * Expand node with children
   */
  expand(actionProbs: Map<number, number>, legalActions: number[]): void {
    for (const action of legalActions) {
      // Clone game state
      const childState = this.gameState.clone();

      // Apply action
      const currentPlayer = childState.getCurrentPlayer();
      if (!currentPlayer) {
        continue;
      }
      childState.applyAction(action, currentPlayer);

      // Create child node
      const child = new MCTSNode(childState, currentPlayer, this);
      child.priorProb = actionProbs.get(action) ?? 0;

      this.children.set(action, child);
    }
  }

  /**
   * Backpropagate value up the tree
   */
  backpropagate(value: number): void {
    let node: MCTSNode | null = this;

    while (node !== null) {
      node.visitCount++;
      node.totalValue += value;
      node = node.parent;
    }
  }

  /**
   * Get action probabilities based on visit counts
   */
  getActionProbabilities(temperature: number = 1.0): Map<number, number> {
    const probs = new Map<number, number>();

    if (this.children.size === 0) {
      return probs;
    }

    if (temperature === 0) {
      // Greedy: select most visited
      let maxVisits = -1;
      let bestAction = -1;

      for (const [action, child] of this.children) {
        if (child.visitCount > maxVisits) {
          maxVisits = child.visitCount;
          bestAction = action;
        }
      }

      probs.set(bestAction, 1.0);
    } else {
      // Apply temperature
      const visits = Array.from(this.children.entries()).map(([action, child]) => ({
        action,
        visits: Math.pow(child.visitCount, 1.0 / temperature)
      }));

      const totalVisits = visits.reduce((sum, v) => sum + v.visits, 0);

      for (const { action, visits: tempVisits } of visits) {
        probs.set(action, tempVisits / totalVisits);
      }
    }

    return probs;
  }
}
```

#### 6.5.2 Perfect Information MCTS (4 hours)

- [ ] **Create `src/mcts/perfect-mcts.ts`**

```typescript
import { MCTSNode } from './mcts-node.js';
import { BlobGame } from '../game/blob-game.js';
import { Player } from '../game/player.js';
import { ONNXInference } from '../ml/onnx-inference.js';

export interface MCTSConfig {
  numSimulations: number;
  cPuct: number;
  temperature: number;
}

export class PerfectMCTS {
  constructor(
    private inference: ONNXInference,
    private config: MCTSConfig
  ) {}

  /**
   * Run MCTS search and return action probabilities
   */
  async search(
    game: BlobGame,
    player: Player
  ): Promise<Map<number, number>> {
    // Create root node
    const root = new MCTSNode(game, player);

    // Run simulations
    for (let i = 0; i < this.config.numSimulations; i++) {
      await this.runSimulation(root);
    }

    // Return action probabilities
    return root.getActionProbabilities(this.config.temperature);
  }

  /**
   * Run single MCTS simulation
   * Steps: Selection → Expansion → Evaluation → Backpropagation
   */
  private async runSimulation(root: MCTSNode): Promise<void> {
    // 1. Selection: traverse tree using UCB1
    let node = root;
    while (!node.isLeaf) {
      node = node.selectChild(this.config.cPuct);
    }

    // 2. Expansion & Evaluation
    const currentPlayer = node.gameState.getCurrentPlayer();
    if (!currentPlayer) {
      // Terminal node (round complete)
      const value = this.evaluateTerminal(node.gameState, node.player);
      node.backpropagate(value);
      return;
    }

    // Get NN predictions
    const { policy, value } = await this.inference.predict(
      node.gameState,
      currentPlayer
    );

    // Get legal actions
    const legalActions = node.gameState.getLegalActions(currentPlayer);
    const legalIndices = legalActions.map(a =>
      typeof a === 'number' ? a : a.toIndex()
    );

    // Create action probabilities map
    const actionProbs = new Map<number, number>();
    let totalProb = 0;
    for (const action of legalIndices) {
      actionProbs.set(action, policy[action]);
      totalProb += policy[action];
    }

    // Normalize
    if (totalProb > 0) {
      for (const [action, prob] of actionProbs) {
        actionProbs.set(action, prob / totalProb);
      }
    }

    // Expand node
    node.expand(actionProbs, legalIndices);

    // 3. Backpropagation
    node.backpropagate(value);
  }

  /**
   * Evaluate terminal game state
   */
  private evaluateTerminal(game: BlobGame, player: Player): number {
    // Normalized score: (actual - expected) / max_possible
    const roundScore = player.calculateRoundScore();
    const cardsDealt = player.hand.length + player.cardsPlayed.length;
    const maxScore = 10 + cardsDealt;

    return roundScore / maxScore;
  }
}
```

#### 6.5.3 Belief State and Determinization (6 hours)

- [ ] **Create `src/mcts/belief-state.ts`**

```typescript
import { Card } from '../game/card.js';
import { Player } from '../game/player.js';
import type { Suit } from '../game/constants.js';

export interface PlayerConstraints {
  playerPosition: number;
  cardsInHand: number;
  cardsPlayed: Set<string>;
  cannotHaveSuits: Set<Suit>;
  mustHaveSuits: Set<Suit>;
}

export class BeliefState {
  private knownCards: Set<string>;
  private playedCards: Set<string>;
  private unseenCards: Set<string>;
  private playerConstraints: Map<number, PlayerConstraints>;

  constructor(
    myHand: Card[],
    cardsPlayed: Card[]
  ) {
    this.knownCards = new Set(myHand.map(c => c.toString()));
    this.playedCards = new Set(cardsPlayed.map(c => c.toString()));
    this.unseenCards = new Set();
    this.playerConstraints = new Map();

    // Initialize unseen cards (all cards - known - played)
    // ... (complete implementation)
  }

  /**
   * Update beliefs when player plays card
   */
  updateOnCardPlayed(
    playerPosition: number,
    card: Card,
    ledSuit: Suit | null
  ): void {
    // Mark card as played
    this.playedCards.add(card.toString());
    this.unseenCards.delete(card.toString());

    // Get constraints
    let constraints = this.playerConstraints.get(playerPosition);
    if (!constraints) {
      constraints = {
        playerPosition,
        cardsInHand: 0,
        cardsPlayed: new Set(),
        cannotHaveSuits: new Set(),
        mustHaveSuits: new Set()
      };
      this.playerConstraints.set(playerPosition, constraints);
    }

    // Track played card
    constraints.cardsPlayed.add(card.toString());

    // Suit inference
    if (ledSuit !== null && card.suit !== ledSuit) {
      // Player didn't follow suit -> they don't have it
      constraints.cannotHaveSuits.add(ledSuit);
    }

    if (card.suit) {
      constraints.mustHaveSuits.add(card.suit);
    }
  }

  /**
   * Get possible cards for player
   */
  getPossibleCards(playerPosition: number): Set<Card> {
    const constraints = this.playerConstraints.get(playerPosition);
    if (!constraints) {
      return new Set(Array.from(this.unseenCards).map(s => {
        // Parse card string back to Card
        // ... implementation
      }));
    }

    // Filter unseen cards by constraints
    // ... (complete implementation)
  }

  /**
   * Check if hand is consistent with constraints
   */
  isConsistentHand(playerPosition: number, hand: Card[]): boolean {
    const constraints = this.playerConstraints.get(playerPosition);
    if (!constraints) {
      return true;
    }

    // Check suit constraints
    for (const suit of constraints.cannotHaveSuits) {
      if (hand.some(c => c.suit === suit)) {
        return false;
      }
    }

    // ... (additional checks)

    return true;
  }
}
```

- [ ] **Create `src/mcts/determinizer.ts`**

```typescript
import { BlobGame } from '../game/blob-game.js';
import { Card } from '../game/card.js';
import { BeliefState } from './belief-state.js';

export class Determinizer {
  /**
   * Sample a consistent determinization (opponent hands)
   * Returns null if sampling fails after max attempts
   */
  sampleDeterminization(
    game: BlobGame,
    belief: BeliefState,
    maxAttempts: number = 100
  ): Map<number, Card[]> | null {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const hands = this.tryGenerateHands(game, belief);

      if (hands !== null) {
        return hands;
      }
    }

    return null; // Failed to generate valid determinization
  }

  /**
   * Try to generate consistent hands (rejection sampling)
   */
  private tryGenerateHands(
    game: BlobGame,
    belief: BeliefState
  ): Map<number, Card[]> | null {
    const hands = new Map<number, Card[]>();

    // Get unseen cards pool
    const unseenCards = belief['unseenCards']; // Access private field
    const pool = Array.from(unseenCards).map(s => {
      // Parse card from string
      // ... implementation
    });

    // Shuffle pool
    this.shuffle(pool);

    // Distribute to players
    // ... (implement with constraint checking)

    return hands;
  }

  /**
   * Fisher-Yates shuffle
   */
  private shuffle<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
}
```

- [ ] **Create `src/mcts/imperfect-mcts.ts`**

```typescript
import { PerfectMCTS, type MCTSConfig } from './perfect-mcts.js';
import { BeliefState } from './belief-state.js';
import { Determinizer } from './determinizer.js';
import { BlobGame } from '../game/blob-game.js';
import { Player } from '../game/player.js';
import { ONNXInference } from '../ml/onnx-inference.js';

export interface ImperfectMCTSConfig extends MCTSConfig {
  numDeterminizations: number;
  simulationsPerDeterminization: number;
}

export class ImperfectMCTS {
  private perfectMCTS: PerfectMCTS;
  private determinizer: Determinizer;

  constructor(
    private inference: ONNXInference,
    private config: ImperfectMCTSConfig
  ) {
    this.perfectMCTS = new PerfectMCTS(inference, {
      numSimulations: config.simulationsPerDeterminization,
      cPuct: config.cPuct,
      temperature: config.temperature
    });
    this.determinizer = new Determinizer();
  }

  /**
   * Run imperfect information MCTS
   * Sample multiple determinizations and aggregate results
   */
  async search(
    game: BlobGame,
    player: Player,
    belief: BeliefState
  ): Promise<Map<number, number>> {
    const actionCounts = new Map<number, number>();

    // Run MCTS on multiple determinizations
    for (let i = 0; i < this.config.numDeterminizations; i++) {
      // Sample consistent opponent hands
      const determinization = this.determinizer.sampleDeterminization(
        game,
        belief
      );

      if (!determinization) {
        console.warn(`Failed to generate determinization ${i}`);
        continue;
      }

      // Create determinized game state
      const detGame = this.applyDeterminization(game, determinization);

      // Run perfect MCTS on determinized state
      const actionProbs = await this.perfectMCTS.search(detGame, player);

      // Aggregate action counts
      for (const [action, prob] of actionProbs) {
        const count = actionCounts.get(action) ?? 0;
        actionCounts.set(action, count + prob);
      }
    }

    // Normalize to probabilities
    const totalCount = Array.from(actionCounts.values()).reduce((a, b) => a + b, 0);
    const actionProbs = new Map<number, number>();

    if (totalCount > 0) {
      for (const [action, count] of actionCounts) {
        actionProbs.set(action, count / totalCount);
      }
    }

    return actionProbs;
  }

  /**
   * Apply determinization to game state
   */
  private applyDeterminization(
    game: BlobGame,
    determinization: Map<number, Card[]>
  ): BlobGame {
    const detGame = game.clone();

    // Replace opponent hands with sampled hands
    for (const [position, hand] of determinization) {
      detGame.players[position].hand = hand.map(c => c.clone());
    }

    return detGame;
  }
}
```

### 6.6 API Endpoints (8 hours)

- [ ] **Create `src/api/game-routes.ts`**

```typescript
import { Hono } from 'hono';
import { z } from 'zod';
import { BlobGame } from '../game/blob-game.js';
import { ONNXInference } from '../ml/onnx-inference.js';
import { ImperfectMCTS } from '../mcts/imperfect-mcts.js';

const app = new Hono();

// In-memory game storage (replace with DB later)
const games = new Map<string, BlobGame>();

// AI inference engine
const inference = new ONNXInference();
await inference.load('../models/best_model.onnx');

// POST /api/game/create
app.post('/create', async (c) => {
  const body = await c.req.json();
  const schema = z.object({
    numPlayers: z.number().min(3).max(8)
  });

  const { numPlayers } = schema.parse(body);

  const game = new BlobGame(numPlayers);
  const gameId = crypto.randomUUID();
  games.set(gameId, game);

  return c.json({
    gameId,
    numPlayers,
    phase: game.gamePhase
  });
});

// GET /api/game/:id
app.get('/:id', (c) => {
  const gameId = c.req.param('id');
  const game = games.get(gameId);

  if (!game) {
    return c.json({ error: 'Game not found' }, 404);
  }

  // Return game state for player 0 (human)
  const player = game.players[0];
  return c.json(game.getGameState(player));
});

// POST /api/game/:id/action
app.post('/:id/action', async (c) => {
  const gameId = c.req.param('id');
  const game = games.get(gameId);

  if (!game) {
    return c.json({ error: 'Game not found' }, 404);
  }

  const body = await c.req.json();
  const schema = z.object({
    playerPosition: z.number(),
    action: z.number() // Bid or card index
  });

  const { playerPosition, action } = schema.parse(body);
  const player = game.players[playerPosition];

  // Apply action
  game.applyAction(action, player);

  return c.json(game.getGameState(player));
});

// POST /api/game/:id/ai-move
app.post('/:id/ai-move', async (c) => {
  const gameId = c.req.param('id');
  const game = games.get(gameId);

  if (!game) {
    return c.json({ error: 'Game not found' }, 404);
  }

  const currentPlayer = game.getCurrentPlayer();
  if (!currentPlayer) {
    return c.json({ error: 'No current player' }, 400);
  }

  // Run MCTS
  const mcts = new ImperfectMCTS(inference, {
    numDeterminizations: 3,
    simulationsPerDeterminization: 50,
    cPuct: 1.5,
    temperature: 0.0, // Greedy for production
    numSimulations: 150 // Total budget
  });

  // Create belief state (simplified for now)
  const belief = new BeliefState(
    currentPlayer.hand,
    game.cardsPlayedThisRound
  );

  const actionProbs = await mcts.search(game, currentPlayer, belief);

  // Select best action
  let bestAction = -1;
  let maxProb = -Infinity;
  for (const [action, prob] of actionProbs) {
    if (prob > maxProb) {
      maxProb = prob;
      bestAction = action;
    }
  }

  // Apply action
  game.applyAction(bestAction, currentPlayer);

  return c.json({
    action: bestAction,
    actionProbs: Object.fromEntries(actionProbs),
    gameState: game.getGameState(currentPlayer)
  });
});

export default app;
```

- [ ] **Create `src/api/websocket.ts`** (for real-time updates)
- [ ] **Create `src/server.ts`** (main entry point)

```typescript
import { Hono } from 'hono';
import { serve } from 'bun';
import gameRoutes from './api/game-routes.js';

const app = new Hono();

app.route('/api/game', gameRoutes);

app.get('/', (c) => c.text('BlobMaster API Server'));

const port = process.env.PORT || 3000;

console.log(`🚀 Server running on http://localhost:${port}`);

serve({
  fetch: app.fetch,
  port
});
```

- [ ] **Test API endpoints**
  - Create game
  - Get game state
  - Apply action
  - AI move

### 6.7 Database (4 hours)

- [ ] **Create `src/db/schema.sql`**

```sql
-- Games table
CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY,
  num_players INTEGER NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  status TEXT CHECK(status IN ('in_progress', 'completed')) DEFAULT 'in_progress'
);

-- Players table
CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  name TEXT NOT NULL,
  final_score INTEGER,
  FOREIGN KEY (game_id) REFERENCES games(id),
  UNIQUE(game_id, position)
);

-- Rounds table
CREATE TABLE IF NOT EXISTS rounds (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  round_number INTEGER NOT NULL,
  cards_dealt INTEGER NOT NULL,
  trump_suit TEXT,
  FOREIGN KEY (game_id) REFERENCES games(id),
  UNIQUE(game_id, round_number)
);

-- Moves table (for replay and analysis)
CREATE TABLE IF NOT EXISTS moves (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id TEXT NOT NULL,
  round_number INTEGER NOT NULL,
  player_position INTEGER NOT NULL,
  move_type TEXT CHECK(move_type IN ('bid', 'play')) NOT NULL,
  action INTEGER NOT NULL, -- Bid value or card index
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (game_id) REFERENCES games(id)
);

CREATE INDEX idx_moves_game ON moves(game_id);
CREATE INDEX idx_moves_round ON moves(game_id, round_number);
```

- [ ] **Create `src/db/queries.ts`**
- [ ] **Integrate database with API**

---

## Phase 7: Frontend Implementation

**Estimated Time:** 40-60 hours
**Platform:** Windows laptop with Bun

### 7.1 SvelteKit Setup (2 hours)

- [ ] **Create frontend project**
  ```bash
  cd ..  # Back to repo root
  bunx create-svelte@latest frontend
  # Choose: SvelteKit demo app, TypeScript, ESLint, Prettier
  ```

- [ ] **Install dependencies**
  ```bash
  cd frontend
  bun install
  bun add -D tailwindcss postcss autoprefixer
  bunx tailwindcss init -p
  bun add lucide-svelte
  ```

- [ ] **Configure TailwindCSS**

```javascript
// tailwind.config.js
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {}
  },
  plugins: []
};
```

```css
/* src/app.css */
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 7.2 Core Components (16 hours)

#### 7.2.1 Card Component (2 hours)

- [ ] **Create `src/lib/components/Card.svelte`**

```svelte
<script lang="ts">
  export let rank: string;
  export let suit: string;
  export let selectable: boolean = false;
  export let selected: boolean = false;
  export let onClick: (() => void) | undefined = undefined;

  const suitColors: Record<string, string> = {
    '♠': 'text-gray-900',
    '♣': 'text-gray-900',
    '♥': 'text-red-600',
    '♦': 'text-red-600'
  };

  const color = suitColors[suit] || 'text-gray-900';
</script>

<button
  class="card {color}"
  class:selectable
  class:selected
  on:click={onClick}
  disabled={!selectable}
>
  <div class="rank">{rank}</div>
  <div class="suit">{suit}</div>
</button>

<style>
  .card {
    width: 80px;
    height: 120px;
    background: white;
    border: 2px solid #333;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: default;
    transition: all 0.2s;
  }

  .card.selectable {
    cursor: pointer;
  }

  .card.selectable:hover {
    transform: translateY(-8px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
  }

  .card.selected {
    transform: translateY(-12px);
    border-color: #3b82f6;
    box-shadow: 0 12px 24px rgba(59, 130, 246, 0.4);
  }

  .rank {
    font-size: 24px;
    font-weight: bold;
  }

  .suit {
    font-size: 32px;
  }
</style>
```

#### 7.2.2 Hand Component (2 hours)

- [ ] **Create `src/lib/components/Hand.svelte`**

```svelte
<script lang="ts">
  import Card from './Card.svelte';

  export let cards: Array<{rank: string, suit: string}>;
  export let selectable: boolean = false;
  export let selectedCard: string | null = null;
  export let onCardClick: ((card: {rank: string, suit: string}) => void) | undefined = undefined;
</script>

<div class="hand">
  {#each cards as card}
    <Card
      rank={card.rank}
      suit={card.suit}
      selectable={selectable}
      selected={selectedCard === `${card.rank}${card.suit}`}
      onClick={() => onCardClick?.(card)}
    />
  {/each}
</div>

<style>
  .hand {
    display: flex;
    gap: -40px; /* Overlap cards */
    justify-content: center;
    padding: 20px;
  }
</style>
```

#### 7.2.3 BidSelector Component (2 hours)

- [ ] **Create `src/lib/components/BidSelector.svelte`**

```svelte
<script lang="ts">
  export let legalBids: number[];
  export let onBid: (bid: number) => void;
</script>

<div class="bid-selector">
  <h3>Make your bid:</h3>
  <div class="bids">
    {#each legalBids as bid}
      <button class="bid-button" on:click={() => onBid(bid)}>
        {bid}
      </button>
    {/each}
  </div>
</div>

<style>
  .bid-selector {
    text-align: center;
    padding: 20px;
  }

  .bids {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
  }

  .bid-button {
    width: 60px;
    height: 60px;
    font-size: 24px;
    font-weight: bold;
    border: 2px solid #333;
    border-radius: 8px;
    background: white;
    cursor: pointer;
    transition: all 0.2s;
  }

  .bid-button:hover {
    background: #3b82f6;
    color: white;
    transform: scale(1.1);
  }
</style>
```

#### 7.2.4 TrickDisplay Component (3 hours)

- [ ] **Create `src/lib/components/TrickDisplay.svelte`**
  - Show cards in current trick
  - Highlight winning card
  - Animate card plays

#### 7.2.5 ScoreBoard Component (2 hours)

- [ ] **Create `src/lib/components/ScoreBoard.svelte`**
  - Show all players
  - Display bids, tricks won, scores
  - Highlight current player

#### 7.2.6 GameBoard Component (3 hours)

- [ ] **Create `src/lib/components/GameBoard.svelte`**
  - Layout: Scoreboard top, trick center, hand bottom
  - Conditional rendering based on phase
  - Game flow orchestration

#### 7.2.7 AIThinking Component (2 hours)

- [ ] **Create `src/lib/components/AIThinking.svelte`**
  - Show progress bar during AI computation
  - Display estimated time remaining
  - Animate "thinking" state

### 7.3 State Management (8 hours)

- [ ] **Create `src/lib/stores/game-state.ts`**

```typescript
import { writable, derived } from 'svelte/store';

export interface GameState {
  gameId: string | null;
  phase: 'setup' | 'bidding' | 'playing' | 'scoring' | 'complete';
  numPlayers: number;
  currentPlayer: number;
  myPosition: number;
  myHand: Array<{rank: string, suit: string}>;
  myBid: number | null;
  myTricksWon: number;
  players: Array<{
    position: number;
    name: string;
    bid: number | null;
    tricksWon: number;
    totalScore: number;
  }>;
  currentTrick: Array<{player: number, card: {rank: string, suit: string}}> | null;
  legalActions: any[];
  trumpSuit: string | null;
}

export const gameState = writable<GameState | null>(null);

export const isMyTurn = derived(
  gameState,
  ($gameState) => $gameState && $gameState.currentPlayer === $gameState.myPosition
);

export const canMakeAction = derived(
  [gameState, isMyTurn],
  ([$gameState, $isMyTurn]) => $isMyTurn && $gameState?.legalActions && $gameState.legalActions.length > 0
);
```

- [ ] **Create `src/lib/stores/ai-state.ts`**

```typescript
import { writable } from 'svelte/store';

export interface AIState {
  thinking: boolean;
  progress: number; // 0-1
  estimatedTime: number; // ms
}

export const aiState = writable<AIState>({
  thinking: false,
  progress: 0,
  estimatedTime: 0
});
```

- [ ] **Create `src/lib/stores/settings.ts`**

```typescript
import { writable } from 'svelte/store';

export interface Settings {
  aiSpeed: 'fast' | 'medium' | 'slow';
  showHints: boolean;
  animationSpeed: 'fast' | 'normal' | 'slow';
}

export const settings = writable<Settings>({
  aiSpeed: 'medium',
  showHints: true,
  animationSpeed: 'normal'
});
```

### 7.4 API Client (4 hours)

- [ ] **Create `src/lib/api/client.ts`**

```typescript
const API_BASE = 'http://localhost:3000/api';

export class GameClient {
  async createGame(numPlayers: number): Promise<string> {
    const res = await fetch(`${API_BASE}/game/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ numPlayers })
    });

    if (!res.ok) {
      throw new Error('Failed to create game');
    }

    const data = await res.json();
    return data.gameId;
  }

  async getGameState(gameId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/game/${gameId}`);

    if (!res.ok) {
      throw new Error('Failed to fetch game state');
    }

    return res.json();
  }

  async makeAction(gameId: string, playerPosition: number, action: number): Promise<any> {
    const res = await fetch(`${API_BASE}/game/${gameId}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ playerPosition, action })
    });

    if (!res.ok) {
      throw new Error('Failed to make action');
    }

    return res.json();
  }

  async requestAIMove(gameId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/game/${gameId}/ai-move`, {
      method: 'POST'
    });

    if (!res.ok) {
      throw new Error('Failed to get AI move');
    }

    return res.json();
  }
}

export const gameClient = new GameClient();
```

- [ ] **Create WebSocket client for real-time updates**

### 7.5 Pages and Routing (8 hours)

#### 7.5.1 Home Page (2 hours)

- [ ] **Create `src/routes/+page.svelte`**
  - New game button
  - Number of players selector
  - Continue existing game (if any)
  - Settings link

#### 7.5.2 Game Page (4 hours)

- [ ] **Create `src/routes/game/[id]/+page.svelte`**
  - Load game state
  - Render GameBoard
  - Handle user actions (bids, card plays)
  - Request AI moves for other players
  - Show game over screen

#### 7.5.3 History Page (2 hours)

- [ ] **Create `src/routes/history/+page.svelte`**
  - List past games
  - View game details
  - Replay functionality

### 7.6 Polish and UX (8 hours)

- [ ] **Animations**
  - Card play animations
  - Trick winner reveal
  - Score updates
  - Phase transitions

- [ ] **Error Handling**
  - Network errors
  - Invalid moves
  - Timeout handling
  - Retry logic

- [ ] **Responsive Design**
  - Mobile layout
  - Tablet layout
  - Desktop layout

- [ ] **Tutorial/Help**
  - Game rules explanation
  - Interactive tutorial
  - Hints system

- [ ] **Accessibility**
  - Keyboard navigation
  - Screen reader support
  - High contrast mode

### 7.7 Testing (8 hours)

- [ ] **Unit Tests (Vitest)**
  - Component tests
  - Store tests
  - API client tests

- [ ] **E2E Tests (Playwright)**
  - Full game flow
  - Error scenarios
  - Multi-player interactions

---

## Critical Implementation Details

### Card Index Mapping (0-51 Scheme)

**Formula:** `index = suit_idx * 13 + rank_idx`

**Mapping:**
```
0-12:   ♠2-♠A (Spades)
13-25:  ♥2-♥A (Hearts)
26-38:  ♣2-♣A (Clubs)
39-51:  ♦2-♦A (Diamonds)
```

**TypeScript Implementation:**
```typescript
function cardToIndex(card: Card): number {
  const suitIdx = SUITS.indexOf(card.suit);
  const rankIdx = RANKS.indexOf(card.rank);
  return suitIdx * 13 + rankIdx;
}

function indexToCard(index: number): Card {
  const suitIdx = Math.floor(index / 13);
  const rankIdx = index % 13;
  return new Card(RANKS[rankIdx], SUITS[suitIdx]);
}
```

### State Encoding (256-Dimension Breakdown)

| Offset | Dimensions | Feature | Description |
|--------|------------|---------|-------------|
| 0 | 52 | My Hand | Binary: 1 if card in hand |
| 52 | 52 | Current Trick | Sequential: play order (1-8) |
| 104 | 52 | All Played This Round | Binary: 1 if played |
| 156 | 8 | Player Bids | Normalized: -1 if no bid, else [0,1] |
| 164 | 8 | Player Tricks Won | Normalized: tricks/cards_dealt |
| 172 | 1 | My Bid | Normalized: -1 or [0,1] |
| 173 | 1 | My Tricks Won | Normalized |
| 174 | 1 | Cards Dealt | Normalized: /13 |
| 175 | 1 | Trick Number | Normalized: /cards_dealt |
| 176 | 1 | My Position | Normalized: /num_players |
| 177 | 1 | Num Players | Normalized: /8 |
| 178 | 4 | Trump Suit | One-hot: [♠,♥,♣,♦] |
| 182 | 1 | Bidding Constraint | 1 if dealer during bidding |
| 183 | 3 | Game Phase | One-hot: [bidding, playing, scoring] |
| 186 | 70 | Context Features | Additional positional/game context |

**Total:** 256 dimensions

**CRITICAL:** TypeScript encoding must match Python **exactly**. Use golden tests to verify.

### Action Space (Unified 52-Dimension)

**Bidding Phase:**
- Use indices 0-13 for bid values
- Only use indices 0 to cards_dealt
- Apply dealer constraint masking

**Playing Phase:**
- Use indices 0-51 for card indices
- Map back to cards via `indexToCard()`
- Apply legal action masking (follow-suit rules)

### Legal Action Masking

**Bidding Mask:**
```typescript
function createBiddingMask(
  cardsDealt: number,
  isDealer: boolean,
  forbiddenBid: number | null
): Float32Array {
  const mask = new Float32Array(52).fill(0);

  for (let bid = 0; bid <= cardsDealt; bid++) {
    if (bid !== forbiddenBid) {
      mask[bid] = 1.0;
    }
  }

  return mask;
}
```

**Playing Mask:**
```typescript
function createPlayingMask(
  hand: Card[],
  ledSuit: Suit | null
): Float32Array {
  const mask = new Float32Array(52).fill(0);

  // Determine legal cards
  const legalCards = ledSuit !== null && hand.some(c => c.suit === ledSuit)
    ? hand.filter(c => c.suit === ledSuit)  // Must follow suit
    : hand;                                  // Any card

  for (const card of legalCards) {
    mask[card.toIndex()] = 1.0;
  }

  return mask;
}
```

### Game Logic Rules

#### Dealer Constraint (Forbidden Bid)

**Rule:** Dealer cannot bid such that total bids = cards dealt.

**Formula:**
```typescript
forbidden_bid = cards_dealt - current_total_bids;

// Only forbidden if in valid range [0, cards_dealt]
if (forbidden_bid >= 0 && forbidden_bid <= cards_dealt) {
  // This bid is forbidden for dealer
}
```

**Example:**
- 4 players, 5 cards each
- Players 0-2 bid: 2, 1, 0 (total = 3)
- Dealer (player 3) cannot bid 2 (would make total = 5 = cards dealt)
- Legal bids for dealer: [0, 1, 3, 4, 5]

#### Follow Suit Rule

**Rule:** Must play card in led suit if possible.

```typescript
function getLegalPlays(hand: Card[], ledSuit: Suit | null): Card[] {
  if (ledSuit === null) {
    // No led suit yet -> any card
    return hand;
  }

  const cardsInLedSuit = hand.filter(c => c.suit === ledSuit);

  if (cardsInLedSuit.length > 0) {
    // Must follow suit
    return cardsInLedSuit;
  }

  // Can't follow suit -> any card
  return hand;
}
```

**Void Suit Detection:**
- If player doesn't play led suit when they could have, record that they are void in that suit
- Update belief state accordingly

#### Trump Logic (Winner Determination)

**Priority:**
1. If trump suit exists: highest trump wins
2. Otherwise: highest card in led suit wins

```typescript
function determineWinner(trick: Trick): Player {
  let winningPlayer = trick.cardsPlayed[0].player;
  let winningCard = trick.cardsPlayed[0].card;

  for (const {player, card} of trick.cardsPlayed.slice(1)) {
    if (trumpSuit !== null) {
      const winningIsTrump = winningCard.suit === trumpSuit;
      const currentIsTrump = card.suit === trumpSuit;

      if (currentIsTrump && !winningIsTrump) {
        // Trump beats non-trump
        winningPlayer = player;
        winningCard = card;
      } else if (currentIsTrump && winningIsTrump) {
        // Both trump -> higher value wins
        if (card.value > winningCard.value) {
          winningPlayer = player;
          winningCard = card;
        }
      } else if (!currentIsTrump && !winningIsTrump) {
        // Neither trump -> check led suit
        if (card.suit === ledSuit && card.value > winningCard.value) {
          winningPlayer = player;
          winningCard = card;
        }
      }
    } else {
      // No trump -> led suit only
      if (card.suit === ledSuit && card.value > winningCard.value) {
        winningPlayer = player;
        winningCard = card;
      }
    }
  }

  return winningPlayer;
}
```

#### Scoring

**Rule:** All-or-nothing scoring.

```typescript
function calculateRoundScore(player: Player): number {
  if (player.bid === null) {
    return 0;
  }

  // Exact match: 10 + bid
  if (player.tricksWon === player.bid) {
    return 10 + player.bid;
  }

  // Missed bid: 0
  return 0;
}
```

**Examples:**
- Bid 3, won 3: score = 10 + 3 = 13
- Bid 3, won 2: score = 0
- Bid 3, won 4: score = 0
- Bid 0, won 0: score = 10 + 0 = 10

#### Trump Rotation

**Pattern:** ♠ → ♥ → ♣ → ♦ → None → repeat

```typescript
const TRUMP_ROTATION = ["♠", "♥", "♣", "♦", null];

function determineTrump(roundNumber: number): Suit | null {
  return TRUMP_ROTATION[roundNumber % 5];
}
```

**Example (5 players, C=7):**
- Round 0 (7 cards): ♠
- Round 1 (6 cards): ♥
- Round 2 (5 cards): ♣
- Round 3 (4 cards): ♦
- Round 4 (3 cards): None
- Round 5 (2 cards): ♠
- ...

---

## Testing Strategy

### Cross-Platform Validation

**Goal:** Ensure TypeScript game logic produces identical results to Python implementation.

#### Golden Test Generation (Python)

```python
# ml/tests/generate_golden_tests.py

import json
from ml.game.blob import BlobGame, Card
from ml.network.encode import StateEncoder
from ml.network.model import ActionMasker

def generate_test_case(seed: int) -> dict:
    """Generate a test case with known game state."""
    random.seed(seed)

    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)

    # Make some bids
    for i, player in enumerate(game.players):
        if i < 3:
            player.make_bid(i)  # Bids: 0, 1, 2

    # Get current player
    player = game.get_current_player()

    # Encode state
    encoder = StateEncoder()
    state = encoder.encode(game, player).tolist()

    # Get legal bids
    legal_bids = game.get_legal_bids(cards_to_deal=5)

    # Create mask
    masker = ActionMasker()
    is_dealer = player.position == game.dealer_position
    current_total = sum(p.bid for p in game.players if p.bid is not None)
    forbidden_bid = game.get_forbidden_bid(current_total, 5)
    mask = masker.create_bidding_mask(5, is_dealer, forbidden_bid).tolist()

    return {
        'seed': seed,
        'game_state': serialize_game(game),
        'player_position': player.position,
        'state_encoding': state,
        'legal_bids': legal_bids,
        'bidding_mask': mask,
        'expected_encoding_dims': 256
    }

def serialize_game(game: BlobGame) -> dict:
    """Serialize game to JSON-compatible dict."""
    return {
        'num_players': game.num_players,
        'current_round': game.current_round,
        'trump_suit': game.trump_suit,
        'dealer_position': game.dealer_position,
        'phase': game.game_phase,
        'players': [
            {
                'position': p.position,
                'hand': [str(c) for c in p.hand],
                'bid': p.bid,
                'tricks_won': p.tricks_won,
                'total_score': p.total_score
            }
            for p in game.players
        ]
    }

# Generate 100 test cases
for i in range(100):
    test_case = generate_test_case(i)
    with open(f'backend/test/golden/case_{i:03d}.json', 'w') as f:
        json.dump(test_case, f, indent=2)
```

#### Golden Test Validation (TypeScript)

```typescript
// backend/test/unit/golden.test.ts

import { describe, test, expect } from 'bun:test';
import { BlobGame } from '../../src/game/blob-game.js';
import { StateEncoder } from '../../src/ml/state-encoder.js';
import { ActionMasker } from '../../src/ml/action-masker.js';
import { readFileSync, readdirSync } from 'fs';

describe('Golden Tests - Cross-Platform Validation', () => {
  const goldenDir = './test/golden';
  const testFiles = readdirSync(goldenDir).filter(f => f.endsWith('.json'));

  for (const file of testFiles) {
    test(`Golden test: ${file}`, () => {
      const testCase = JSON.parse(readFileSync(`${goldenDir}/${file}`, 'utf-8'));

      // Deserialize game
      const game = deserializeGame(testCase.game_state);

      // Get player
      const player = game.players[testCase.player_position];

      // Encode state
      const encoder = new StateEncoder();
      const state = encoder.encode(game, player);

      // Compare encoding (within tolerance)
      expectArraysClose(
        Array.from(state),
        testCase.state_encoding,
        1e-6
      );

      // Compare legal actions
      const legalBids = game.getLegalBids(5);
      expect(legalBids.sort()).toEqual(testCase.legal_bids.sort());

      // Compare mask
      const masker = new ActionMasker();
      const isDealer = player.position === game.dealerPosition;
      const currentTotal = game.players
        .filter(p => p.bid !== null)
        .reduce((sum, p) => sum + p.bid!, 0);
      const forbiddenBid = game.getForbiddenBid(currentTotal, 5);
      const mask = masker.createBiddingMask(5, isDealer, forbiddenBid);

      expectArraysClose(
        Array.from(mask),
        testCase.bidding_mask,
        1e-6
      );
    });
  }
});

function deserializeGame(data: any): BlobGame {
  // Reconstruct game from serialized data
  // ... implementation
}

function expectArraysClose(actual: number[], expected: number[], tolerance: number) {
  expect(actual.length).toBe(expected.length);

  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tolerance) {
      throw new Error(
        `Arrays differ at index ${i}: ${actual[i]} vs ${expected[i]} (diff: ${diff})`
      );
    }
  }
}
```

### Test Coverage Targets

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Game Logic | 95%+ | Critical |
| State Encoder | 100% | Critical |
| Action Masker | 100% | Critical |
| MCTS | 90%+ | High |
| API Endpoints | 90%+ | High |
| UI Components | 80%+ | Medium |
| E2E Flows | 80%+ | High |

### Performance Benchmarks

```typescript
// backend/test/performance/inference.bench.ts

import { bench, describe } from 'bun:test';
import { ONNXInference } from '../../src/ml/onnx-inference.js';
import { BlobGame } from '../../src/game/blob-game.js';

describe('Inference Performance', () => {
  const inference = new ONNXInference();
  await inference.load('../models/best_model.onnx');

  const game = new BlobGame(4);
  game.setupRound(5);
  const player = game.players[0];

  bench('ONNX forward pass', async () => {
    await inference.predict(game, player);
  });
  // Target: <50ms per call
});

describe('MCTS Performance', () => {
  // ... benchmark MCTS search time
  // Target: <500ms for full AI move (3 det × 50 sims = 150 total)
});
```

---

## Task Breakdown & Timeline

### Phase 5: ONNX Export (Ubuntu PC)
**Estimated Time:** 4-6 hours

- [ ] **DVC Setup** (1-2 hours)
  - Install and initialize DVC
  - Configure remote storage
  - Track existing checkpoints

- [ ] **ONNX Export Script** (2-3 hours)
  - Create `ml/export_onnx.py`
  - Export model
  - Validate equivalence

- [ ] **Windows Setup** (1 hour)
  - Clone repo
  - Install DVC
  - Pull production model

### Phase 6: Backend (Windows Laptop)
**Estimated Time:** 40-60 hours

#### Week 1: Game Engine (16 hours)
- [ ] **Day 1** (4 hours): Constants, Card, Deck
- [ ] **Day 2** (4 hours): Player, Trick
- [ ] **Day 3** (6 hours): BlobGame (Part 1-3)
- [ ] **Day 4** (2 hours): BlobGame (Part 4-5), tests

#### Week 2: ML Infrastructure (16 hours)
- [ ] **Day 5** (6 hours): StateEncoder
- [ ] **Day 6** (2 hours): ActionMasker
- [ ] **Day 7** (6 hours): ONNX Inference
- [ ] **Day 8** (2 hours): Golden tests, validation

#### Week 3: MCTS (16 hours)
- [ ] **Day 9** (2 hours): MCTSNode
- [ ] **Day 10** (4 hours): PerfectMCTS
- [ ] **Day 11** (6 hours): BeliefState, Determinizer
- [ ] **Day 12** (4 hours): ImperfectMCTS, integration

#### Week 4: API & Database (12 hours)
- [ ] **Day 13** (6 hours): Game routes, endpoints
- [ ] **Day 14** (2 hours): WebSocket
- [ ] **Day 15** (2 hours): Database schema, queries
- [ ] **Day 16** (2 hours): Integration, testing

### Phase 7: Frontend (Windows Laptop)
**Estimated Time:** 40-60 hours

#### Week 5: Core Components (16 hours)
- [ ] **Day 17** (2 hours): SvelteKit setup, Card component
- [ ] **Day 18** (2 hours): Hand, BidSelector
- [ ] **Day 19** (3 hours): TrickDisplay
- [ ] **Day 20** (2 hours): ScoreBoard
- [ ] **Day 21** (3 hours): GameBoard
- [ ] **Day 22** (2 hours): AIThinking
- [ ] **Day 23** (2 hours): Component testing

#### Week 6: State & API (16 hours)
- [ ] **Day 24** (4 hours): Game state store
- [ ] **Day 25** (2 hours): AI state store, settings
- [ ] **Day 26** (4 hours): API client
- [ ] **Day 27** (2 hours): WebSocket client
- [ ] **Day 28** (4 hours): Integration testing

#### Week 7: Pages & Flow (16 hours)
- [ ] **Day 29** (2 hours): Home page
- [ ] **Day 30** (4 hours): Game page (Part 1)
- [ ] **Day 31** (4 hours): Game page (Part 2)
- [ ] **Day 32** (2 hours): History page
- [ ] **Day 33** (4 hours): Game flow, error handling

#### Week 8: Polish & Testing (12 hours)
- [ ] **Day 34** (4 hours): Animations, transitions
- [ ] **Day 35** (2 hours): Responsive design
- [ ] **Day 36** (2 hours): Tutorial/help
- [ ] **Day 37** (2 hours): Accessibility
- [ ] **Day 38** (2 hours): E2E testing

### Total Timeline
- **Phase 5:** 4-6 hours
- **Phase 6:** 40-60 hours (2-3 weeks)
- **Phase 7:** 40-60 hours (2-3 weeks)
- **Total:** 84-126 hours (4-6 weeks of focused work)

---

## Reference Information

### Python Files to Reference

**Game Engine:**
- [ml/game/blob.py](../ml/game/blob.py) - Main game logic (1808 lines)
- [ml/game/constants.py](../ml/game/constants.py) - Constants (86 lines)
- [ml/game/test_blob.py](../ml/game/test_blob.py) - Comprehensive tests (1400+ lines)

**ML Components:**
- [ml/network/encode.py](../ml/network/encode.py) - State encoding (693 lines)
- [ml/network/model.py](../ml/network/model.py) - Neural network (509 lines)
- [ml/config.py](../ml/config.py) - Training config (296 lines)

**MCTS:**
- [ml/mcts/search.py](../ml/mcts/search.py) - MCTS implementation (1438 lines)
- [ml/mcts/node.py](../ml/mcts/node.py) - Tree nodes (566 lines)
- [ml/mcts/belief_tracker.py](../ml/mcts/belief_tracker.py) - Belief tracking (542 lines)
- [ml/mcts/determinization.py](../ml/mcts/determinization.py) - Sampling (564 lines)

**Tests:**
- All `test_*.py` files for examples and edge cases

### Key Configuration Parameters

**Inference Config (from `ml/config.py`):**
```typescript
interface InferenceConfig {
  // MCTS
  num_determinizations: 3-5;
  simulations_per_determinization: 50-100;
  c_puct: 1.5;
  temperature: 0.0;  // Greedy for production

  // Model
  state_dim: 256;
  embedding_dim: 256;
  num_layers: 6;
  num_heads: 8;
  feedforward_dim: 1024;
  action_dim: 52;

  // Performance
  max_inference_time_ms: 500;
  enable_caching: true;
}
```

### Useful TypeScript Utilities

**Deep Copy:**
```typescript
function deepCopy<T>(obj: T): T {
  return structuredClone(obj); // Modern browsers/Bun
  // Or: JSON.parse(JSON.stringify(obj)) for simple objects
}
```

**Fisher-Yates Shuffle:**
```typescript
function shuffle<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}
```

**Softmax:**
```typescript
function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b);
  return exps.map(x => x / sumExps);
}
```

**UCB1 Score:**
```typescript
function ucb1Score(
  child: MCTSNode,
  parent: MCTSNode,
  cPuct: number
): number {
  const qValue = child.visitCount > 0
    ? child.totalValue / child.visitCount
    : 0;

  const uValue = cPuct * child.priorProb *
    Math.sqrt(parent.visitCount) / (1 + child.visitCount);

  return qValue + uValue;
}
```

---

## Appendix: Quick Start Commands

### Phase 5 (Ubuntu PC)
```bash
# Install DVC
pip install dvc

# Initialize
dvc init
dvc remote add -d storage /mnt/usb/blobmaster_models

# Export ONNX
python ml/export_onnx.py

# Track model
dvc add models/best_model.onnx
git add models/best_model.onnx.dvc
git commit -m "Export ONNX model"
dvc push
```

### Phase 6 (Windows Laptop)
```bash
# Clone and setup
git clone <repo> C:\Github\BlobMaster
cd C:\Github\BlobMaster

# Install DVC and download model
pip install dvc
dvc pull models/best_model.onnx

# Backend setup
mkdir backend
cd backend
bun init
bun add hono onnxruntime-node better-sqlite3 zod

# Start development
bun run src/server.ts
```

### Phase 7 (Windows Laptop)
```bash
# Frontend setup
cd ..
bunx create-svelte@latest frontend
cd frontend
bun install
bun add -D tailwindcss postcss autoprefixer
bun add lucide-svelte

# Start development
bun run dev
```

---

## Success Criteria

### Phase 6 Complete When:
- ✅ All game logic tests pass (including golden tests)
- ✅ ONNX inference matches Python (within 1e-6 tolerance)
- ✅ MCTS can make legal moves
- ✅ API endpoints return valid responses
- ✅ AI can play full games without errors
- ✅ Inference time <50ms, AI move time <500ms

### Phase 7 Complete When:
- ✅ User can create and play full games
- ✅ AI opponents make reasonable moves
- ✅ UI is responsive and intuitive
- ✅ Animations are smooth
- ✅ Error handling is robust
- ✅ E2E tests pass
- ✅ Works on mobile, tablet, desktop

---

## Next Steps After Phases 6-7

1. **Optimization:**
   - Enable OpenVINO execution provider for Intel iGPU
   - Model quantization (int8) for faster inference
   - MCTS tree reuse between moves
   - Batch inference for multiple determinizations

2. **Features:**
   - Multiplayer online (WebSocket)
   - Different difficulty levels (MCTS budget)
   - Replay and analysis tools
   - Statistics and ELO tracking

3. **Deployment:**
   - Package as Electron app
   - Create installer for Windows
   - Cloud deployment (optional)

4. **Training:**
   - Return to Ubuntu PC
   - Complete Phase 4 (full multi-round game training)
   - Iterate on model improvements

---

**End of GUI-TODO.md**
