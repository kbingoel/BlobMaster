<script lang="ts">
	import { get } from 'svelte/store';
	import type { CardEval, EvalDisplay, SessionSnapshot } from '$lib/api';
	import {
		cardIndex,
		isRed,
		rankLabel,
		suitGlyph,
		SUIT_KEYS,
		RANK_KEYS
	} from '$lib/cardUtils';
	import { primaryMetric } from '$lib/evalUtils';
	import { trumpEditingStore } from '$lib/stores/trumpEditing';

	interface Props {
		snapshot?: SessionSnapshot | null;
		mode?: 'hand-entry' | 'play' | 'review';
		/** Card indices currently toggled as in-hand (hand-entry mode only). */
		selectedCards?: number[];
		onCardclick?: (cardIdx: number) => void;
		/**
		 * Optional per-card AI evals (Session 9.7 secondary surface). When
		 * non-empty and `evalMode !== 'off'`, legal cells render the primary
		 * metric in the bottom corner. Off by default — the right grid
		 * stays uncluttered unless the user opts in.
		 */
		perCardEvals?: CardEval[];
		evalMode?: EvalDisplay;
	}

	let {
		snapshot = null,
		mode = 'play',
		selectedCards = [],
		onCardclick = () => {},
		perCardEvals = [],
		evalMode = 'off'
	}: Props = $props();

	let evalByCard = $derived(new Map(perCardEvals.map((e) => [e.card, e])));

	type CellState = 'in-hand' | 'legal' | 'played' | 'illegal' | 'empty';

	interface PlayedInfo {
		seat: number;
		label: string; // e.g. "P3 R2.t1"
	}

	// Rank pre-armed by a keyboard rank key — highlights the entire row.
	let armedRank = $state<number | null>(null);

	// Pre-compute static arrays to avoid recreating them on each render.
	const ROWS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // row 0 = A (rank 12)
	const COLS = [0, 1, 2, 3]; // suit index

	function nameFor(seat: number): string {
		return snapshot?.player_names[seat] ?? `P${seat}`;
	}

	function getPlayedInfo(cardIdx: number): PlayedInfo | null {
		if (!snapshot) return null;

		for (let t = 0; t < snapshot.trick_history.length; t++) {
			for (const play of snapshot.trick_history[t].plays) {
				if (play.card === cardIdx) {
					return {
						seat: play.seat,
						label: `${nameFor(play.seat)} R${snapshot.round_idx + 1}.t${t + 1}`
					};
				}
			}
		}

		const tIdx = snapshot.trick_history.length + 1;
		for (const play of snapshot.trick_in_progress) {
			if (play.card === cardIdx) {
				return {
					seat: play.seat,
					label: `${nameFor(play.seat)} R${snapshot.round_idx + 1}.t${tIdx}`
				};
			}
		}

		return null;
	}

	function cellInfo(cardIdx: number): { state: CellState; playedInfo?: PlayedInfo } {
		if (mode === 'hand-entry') {
			return { state: selectedCards.includes(cardIdx) ? 'in-hand' : 'legal' };
		}

		if (!snapshot) return { state: 'empty' };

		const playedInfo = getPlayedInfo(cardIdx);
		if (playedInfo) return { state: 'played', playedInfo };

		if (snapshot.human_hand.includes(cardIdx)) {
			const isLegal = snapshot.legal_plays?.includes(cardIdx) ?? false;
			return { state: isLegal ? 'legal' : 'in-hand' };
		}

		// Opponent's turn: show engine-computed legal plays (null when it's not human's turn)
		if (
			snapshot.current_player !== snapshot.human_seat &&
			snapshot.legal_plays?.includes(cardIdx)
		) {
			return { state: 'legal' };
		}

		return { state: 'empty' };
	}

	function handleCellClick(cardIdx: number, state: CellState) {
		if (mode === 'hand-entry') {
			onCardclick(cardIdx);
		} else if (state === 'legal' || state === 'in-hand') {
			onCardclick(cardIdx);
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		// Surrender keyboard control to the trump editor when it's open.
		if (get(trumpEditingStore)) return;

		const key = e.key.toUpperCase();

		if (key === 'ESCAPE') {
			armedRank = null;
			return;
		}

		const suitIdx = (SUIT_KEYS as readonly string[]).indexOf(key);
		if (suitIdx >= 0 && armedRank !== null) {
			e.preventDefault();
			onCardclick(cardIndex(suitIdx, armedRank));
			armedRank = null;
			return;
		}

		const rankIdx = (RANK_KEYS as readonly string[]).indexOf(key);
		if (rankIdx >= 0) {
			e.preventDefault();
			armedRank = rankIdx;
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="card-grid" role="grid" aria-label="Card grid">
	{#each ROWS as row}
		{#each COLS as col}
			{@const rank = 12 - row}
			{@const cardIdx = cardIndex(col, rank)}
			{@const ci = cellInfo(cardIdx)}
			{@const red = isRed(col)}
			{@const rowArmed = mode === 'hand-entry' && armedRank === rank}
			{@const clickable =
				mode === 'hand-entry' || ci.state === 'legal' || ci.state === 'in-hand'}
			<div
				class="card-cell"
				class:state-in-hand={ci.state === 'in-hand'}
				class:state-legal={ci.state === 'legal'}
				class:state-played={ci.state === 'played'}
				class:state-illegal={ci.state === 'illegal'}
				class:state-empty={ci.state === 'empty'}
				class:suit-red={red}
				class:row-armed={rowArmed}
				class:clickable
				role="button"
				tabindex={clickable ? 0 : -1}
				aria-label="{rankLabel(rank)}{suitGlyph(col)}"
				aria-pressed={ci.state === 'in-hand'}
				onclick={() => clickable && handleCellClick(cardIdx, ci.state)}
				onkeydown={(e) => e.key === 'Enter' && clickable && handleCellClick(cardIdx, ci.state)}
			>
				<span class="rank-text">{rankLabel(rank)}</span>
				<span class="suit-text">{suitGlyph(col)}</span>
				{#if ci.state === 'played' && ci.playedInfo}
					<span class="played-label">{ci.playedInfo.label}</span>
				{:else if ci.state === 'legal' && evalMode !== 'off' && evalByCard.has(cardIdx)}
					<span class="grid-eval">{primaryMetric(evalByCard.get(cardIdx)!, evalMode)}</span>
				{/if}
			</div>
		{/each}
	{/each}
</div>

<style>
	.card-grid {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		grid-template-rows: repeat(13, 1fr);
		width: 100%;
		height: 100%;
		gap: 2px;
		background-color: #cbd5e1; /* gap color — slate-300 */
		border: 1px solid #cbd5e1;
		border-radius: 4px;
		overflow: hidden;
		box-sizing: border-box;
	}

	.card-cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		background-color: #f8fafc; /* slate-50 */
		color: #1e293b; /* slate-800 — default for ♠/♣ */
		cursor: default;
		user-select: none;
		position: relative;
		padding: 2px;
		box-sizing: border-box;
		transition: background-color 0.07s;
		outline: none;
		overflow: hidden;
	}

	/* ── Suit colours ─────────────────────────────────────────── */

	.suit-red .rank-text,
	.suit-red .suit-text {
		color: #dc2626; /* red-600 for ♥/♦ */
	}

	.rank-text {
		font-size: clamp(0.55rem, 1.4cqw, 1rem);
		font-weight: 700;
		line-height: 1.1;
	}

	.suit-text {
		font-size: clamp(0.6rem, 1.5cqw, 1.1rem);
		line-height: 1.1;
	}

	/* ── Cell states ──────────────────────────────────────────── */

	.state-in-hand {
		background-color: #dbeafe; /* blue-100 */
		box-shadow: inset 0 0 0 2px #3b82f6; /* blue-500 */
	}

	.state-in-hand.suit-red {
		background-color: #fee2e2; /* red-100 */
		box-shadow: inset 0 0 0 2px #ef4444; /* red-500 */
	}

	.state-legal {
		background-color: #f0fdf4; /* green-50 */
	}

	.state-played {
		background-color: #f1f5f9; /* slate-100 */
	}

	.state-played .rank-text,
	.state-played .suit-text {
		color: #94a3b8; /* slate-400 */
	}

	.state-illegal {
		opacity: 0.3;
	}

	/* state-empty stays at default slate-50 */

	/* ── Armed row highlight ───────────────────────────────────── */

	.row-armed {
		background-color: #fffbeb; /* amber-50 */
		box-shadow: inset 0 0 0 2px #f59e0b; /* amber-400 */
	}

	/* Armed overrides in-hand so the user can see which row is targeted */
	.row-armed.state-in-hand {
		background-color: #fef3c7; /* amber-100 */
		box-shadow: inset 0 0 0 2px #d97706; /* amber-600 */
	}

	/* ── Interactivity ────────────────────────────────────────── */

	.clickable {
		cursor: pointer;
	}

	.clickable:hover {
		background-color: #e2e8f0; /* slate-200 */
	}

	.state-in-hand.clickable:hover {
		background-color: #bfdbfe; /* blue-200 */
	}

	.state-in-hand.suit-red.clickable:hover {
		background-color: #fecaca; /* red-200 */
	}

	.row-armed.clickable:hover {
		background-color: #fde68a; /* amber-200 */
	}

	.card-cell:focus-visible {
		outline: 2px solid #3b82f6;
		outline-offset: -2px;
		z-index: 1;
	}

	/* ── Played annotation ────────────────────────────────────── */

	.played-label {
		position: absolute;
		bottom: 1px;
		left: 0;
		right: 0;
		text-align: center;
		font-size: clamp(0.4rem, 1cqw, 0.6rem);
		color: #64748b; /* slate-500 */
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		padding: 0 2px;
	}

	/* ── Secondary AI eval annotation (Session 9.7) ──────────────── */

	.grid-eval {
		position: absolute;
		bottom: 1px;
		right: 3px;
		font-size: clamp(0.4rem, 1cqw, 0.62rem);
		font-weight: 600;
		color: #166534;
		font-variant-numeric: tabular-nums;
		pointer-events: none;
	}
</style>
