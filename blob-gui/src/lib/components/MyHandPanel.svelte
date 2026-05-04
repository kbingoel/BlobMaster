<script lang="ts">
	import type { AiSuggestion, SessionSnapshot } from '$lib/api';
	import { isRed, rankLabel, suitGlyph, cardSuit, cardRank } from '$lib/cardUtils';

	interface Props {
		snapshot: SessionSnapshot;
		suggestion?: AiSuggestion | null;
		submitting?: boolean;
		errorMessage?: string | null;
		onPlay: (card: number) => void;
	}

	let {
		snapshot,
		suggestion = null,
		submitting = false,
		errorMessage = null,
		onPlay
	}: Props = $props();

	let isHumanTurn = $derived(snapshot.current_player === snapshot.human_seat);
	let humanSeat = $derived(snapshot.human_seat);

	// Sorted hand: by suit then rank ascending — the same order CardGrid uses.
	let handCards = $derived([...snapshot.human_hand].sort((a, b) => a - b));

	// Played-this-round cards by the human, in play order (across completed
	// tricks + the in-progress trick).
	let placedThisRound = $derived.by(() => {
		const out: number[] = [];
		for (const t of snapshot.trick_history) {
			for (const play of t.plays) {
				if (play.seat === humanSeat) out.push(play.card);
			}
		}
		for (const play of snapshot.trick_in_progress) {
			if (play.seat === humanSeat) out.push(play.card);
		}
		return out;
	});

	// AI suggestion details (playing variant only).
	let playSuggestion = $derived(
		suggestion && suggestion.phase === 'playing' ? suggestion : null
	);
	let recommendedCard = $derived(playSuggestion?.recommended_card ?? null);
	let valueEstimate = $derived(playSuggestion?.value_estimate ?? null);
	let simsCompleted = $derived(playSuggestion?.sims_completed ?? 0);
	let depth = $derived(playSuggestion?.depth ?? 0);

	let myRoundScore = $derived(snapshot.tricks_won[humanSeat] ?? 0);
	let myCumulativeScore = $derived(snapshot.cumulative_scores[humanSeat] ?? 0);
	let myBid = $derived(snapshot.bids[humanSeat] ?? null);

	function isLegal(card: number): boolean {
		if (!isHumanTurn) return false;
		return snapshot.legal_plays?.includes(card) ?? false;
	}

	function handleClick(card: number) {
		if (submitting) return;
		if (!isLegal(card)) return;
		onPlay(card);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		if (
			e.key === 'Enter' &&
			isHumanTurn &&
			recommendedCard !== null &&
			!submitting &&
			isLegal(recommendedCard)
		) {
			e.preventDefault();
			onPlay(recommendedCard);
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="hand-panel">
	<header class="status">
		<span class="status-cell"
			>MCTS: <strong>{simsCompleted}</strong></span
		>
		<span class="status-cell">depth <strong>{depth}</strong></span>
		<span class="status-cell"
			>v=<strong
				>{valueEstimate === null
					? '—'
					: (valueEstimate >= 0 ? '+' : '') + valueEstimate.toFixed(2)}</strong
			></span
		>
		<span class="status-cell"
			>round <strong>{myRoundScore}</strong>{myBid !== null
				? ` / ${myBid}`
				: ''}</span
		>
		<span class="status-cell">total <strong>{myCumulativeScore}</strong></span>
	</header>

	<div class="hand-row">
		{#if handCards.length === 0}
			<p class="empty-hand">Hand exhausted for this round.</p>
		{/if}
		{#each handCards as card (card)}
			{@const suit = cardSuit(card)}
			{@const rank = cardRank(card)}
			{@const legal = isLegal(card)}
			{@const recommended = isHumanTurn && recommendedCard === card}
			<button
				type="button"
				class="card"
				class:suit-red={isRed(suit)}
				class:legal
				class:illegal={isHumanTurn && !legal}
				class:recommended
				disabled={!legal || submitting}
				onclick={() => handleClick(card)}
			>
				<span class="card-rank">{rankLabel(rank)}</span>
				<span class="card-suit">{suitGlyph(suit)}</span>
				<span class="card-eval">
					<span class="eval-line">—</span>
					<span class="eval-line eval-line-2">—</span>
				</span>
			</button>
		{/each}
	</div>

	{#if placedThisRound.length > 0}
		<div class="placed-strip" aria-label="Cards I played this round">
			<span class="placed-label">played:</span>
			{#each placedThisRound as card (card)}
				{@const suit = cardSuit(card)}
				{@const rank = cardRank(card)}
				<span class="placed-card" class:suit-red={isRed(suit)}>
					{rankLabel(rank)}{suitGlyph(suit)}
				</span>
			{/each}
		</div>
	{/if}

	{#if !isHumanTurn}
		<p class="hint waiting-hint">
			Waiting for P{snapshot.current_player} to play — click a card on the
			right grid to record their move.
		</p>
	{:else if recommendedCard !== null}
		<p class="hint">
			Press <kbd>Enter</kbd> to play the recommended card.
		</p>
	{/if}

	{#if errorMessage}
		<p class="error">{errorMessage}</p>
	{/if}
</div>

<style>
	.hand-panel {
		display: flex;
		flex-direction: column;
		gap: 0.55rem;
		padding: 0.6rem 0.85rem;
		height: 100%;
		box-sizing: border-box;
		overflow-y: auto;
	}

	.status {
		display: flex;
		gap: 0.85rem;
		flex-wrap: wrap;
		font-size: 0.72rem;
		color: #475569;
		font-variant-numeric: tabular-nums;
		padding-bottom: 0.35rem;
		border-bottom: 1px solid #e2e8f0;
	}

	.status-cell strong {
		color: #0f172a;
		font-weight: 700;
	}

	.hand-row {
		display: flex;
		gap: 0.35rem;
		flex-wrap: wrap;
		padding-top: 0.3rem;
		min-height: 5.5rem;
	}

	.empty-hand {
		margin: 0;
		font-size: 0.78rem;
		color: #94a3b8;
		font-style: italic;
	}

	.card {
		flex: 0 0 auto;
		width: clamp(2.5rem, 5.5cqw, 3.6rem);
		min-width: 2.5rem;
		min-height: 4.5rem;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: flex-start;
		gap: 0.05rem;
		padding: 0.35rem 0.2rem 0.2rem;
		border: 1px solid #cbd5e1;
		border-radius: 5px;
		background: #fff;
		color: #0f172a;
		cursor: pointer;
		font-variant-numeric: tabular-nums;
		transition: transform 0.1s, background-color 0.07s, border-color 0.07s;
	}

	.card.suit-red {
		color: #dc2626;
	}

	.card-rank {
		font-size: 1.05rem;
		font-weight: 700;
		line-height: 1;
	}

	.card-suit {
		font-size: 1.1rem;
		line-height: 1;
	}

	.card-eval {
		margin-top: auto;
		display: flex;
		flex-direction: column;
		gap: 0.05rem;
		font-size: 0.55rem;
		color: #94a3b8;
	}

	.eval-line {
		line-height: 1;
	}

	.eval-line-2 {
		color: #cbd5e1;
	}

	.card:hover:not(:disabled) {
		background: #f1f5f9;
		border-color: #94a3b8;
	}

	.card.illegal {
		opacity: 0.35;
		cursor: not-allowed;
	}

	.card.recommended {
		transform: translateY(-8px);
		border-color: #15803d;
		box-shadow: 0 0 0 2px #15803d, 0 4px 8px -2px rgba(21, 128, 61, 0.4);
	}

	.placed-strip {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		flex-wrap: wrap;
		padding: 0.3rem 0.4rem;
		background: #f8fafc;
		border: 1px dashed #e2e8f0;
		border-radius: 4px;
		font-size: 0.75rem;
	}

	.placed-label {
		text-transform: uppercase;
		letter-spacing: 0.05em;
		font-size: 0.65rem;
		color: #94a3b8;
	}

	.placed-card {
		font-family: ui-monospace, 'SFMono-Regular', monospace;
		opacity: 0.55;
		color: #0f172a;
	}

	.placed-card.suit-red {
		color: #dc2626;
	}

	.hint {
		margin: 0;
		font-size: 0.75rem;
		color: #475569;
	}

	.waiting-hint {
		color: #94a3b8;
		font-style: italic;
	}

	kbd {
		font-family: ui-monospace, 'SFMono-Regular', monospace;
		font-size: 0.7rem;
		padding: 0 0.25rem;
		border: 1px solid #cbd5e1;
		border-bottom-width: 2px;
		border-radius: 3px;
		background: #fff;
	}

	.error {
		margin: 0;
		font-size: 0.78rem;
		color: #b91c1c;
	}
</style>
