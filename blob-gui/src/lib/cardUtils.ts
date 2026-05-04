// Card encoding: card_index = suit * 13 + rank
// Suits: S=0, H=1, C=2, D=3   Ranks: 2=0, 3=1, … K=11, A=12

export const SUITS = ['♠', '♥', '♣', '♦'] as const;

// Keyboard keys that arm each suit (index == suit number)
export const SUIT_KEYS = ['S', 'H', 'C', 'D'] as const;

// Display labels indexed by rank (rank 0 = '2', rank 12 = 'A')
export const RANK_LABELS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] as const;

// Keyboard keys that arm each rank (index == rank number)
export const RANK_KEYS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'] as const;

export const NUM_SUITS = 4;
export const NUM_RANKS = 13;
export const NUM_CARDS = 52;

export function cardIndex(suit: number, rank: number): number {
	return suit * 13 + rank;
}

export function cardSuit(index: number): number {
	return Math.floor(index / 13);
}

export function cardRank(index: number): number {
	return index % 13;
}

export function rankLabel(rank: number): string {
	return RANK_LABELS[rank];
}

export function suitGlyph(suit: number): string {
	return SUITS[suit] ?? '?';
}

// Hearts (1) and Diamonds (3) are red; Spades (0) and Clubs (2) are near-black.
export function isRed(suit: number): boolean {
	return suit === 1 || suit === 3;
}
