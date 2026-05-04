<script lang="ts">
	import type { AiSuggestion, SessionSnapshot } from '$lib/api';

	interface Props {
		snapshot: SessionSnapshot;
		suggestion?: AiSuggestion | null;
		submitting?: boolean;
		errorMessage?: string | null;
		onSubmit: (bid: number) => void;
	}

	let {
		snapshot,
		suggestion = null,
		submitting = false,
		errorMessage = null,
		onSubmit
	}: Props = $props();

	let cardsDealt = $derived(snapshot.cards_dealt);
	let activeSeat = $derived(snapshot.current_player);
	let activeName = $derived(snapshot.player_names[activeSeat] ?? `P${activeSeat}`);
	let isHumanTurn = $derived(activeSeat === snapshot.human_seat);
	let isDealerTurn = $derived(activeSeat === snapshot.dealer);
	let forbidden = $derived(snapshot.forbidden_bid);

	// Sum of bids placed so far this round.
	let bidsPlacedSum = $derived(
		snapshot.bids.reduce((acc: number, b) => acc + (b ?? 0), 0)
	);

	// Tally line color: amber when dealer is currently constrained, green otherwise.
	let tallyAmber = $derived(isDealerTurn && forbidden !== null);

	// Build the value list 0..=cards_dealt.
	let bidValues = $derived(
		Array.from({ length: cardsDealt + 1 }, (_, i) => i)
	);

	function isLegal(bid: number): boolean {
		if (isHumanTurn) {
			// Engine-supplied legal mask is authoritative for the human.
			return snapshot.legal_bids?.includes(bid) ?? false;
		}
		// Opponent: every 0..cards_dealt is legal except the forbidden one
		// when this is the dealer's turn.
		if (isDealerTurn && forbidden === bid) return false;
		return bid >= 0 && bid <= cardsDealt;
	}

	// AI suggestion details (bidding variant only).
	let bidSuggestion = $derived(
		suggestion && suggestion.phase === 'bidding' ? suggestion : null
	);
	let recommendedBid = $derived(bidSuggestion?.recommended_bid ?? null);

	// Top-3 bids sorted by policy probability, filtered to non-zero entries.
	let topBids = $derived.by(() => {
		if (!bidSuggestion) return [] as { bid: number; prob: number }[];
		return bidSuggestion.policy
			.map((prob, bid) => ({ bid, prob }))
			.filter((e) => e.prob > 0)
			.sort((a, b) => b.prob - a.prob)
			.slice(0, 3);
	});

	function handleClick(bid: number) {
		if (submitting) return;
		if (!isLegal(bid)) return;
		onSubmit(bid);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

		if (e.key === 'Enter' && isHumanTurn && recommendedBid !== null && !submitting) {
			e.preventDefault();
			if (isLegal(recommendedBid)) onSubmit(recommendedBid);
			return;
		}

		// Number keys 0..9 record a bid for the active player (whoever it is).
		if (/^[0-9]$/.test(e.key) && !submitting) {
			const bid = parseInt(e.key, 10);
			if (bid <= cardsDealt && isLegal(bid)) {
				e.preventDefault();
				onSubmit(bid);
			}
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="keypad">
	<header class="header">
		<span class="active-label">
			<strong>P{activeSeat} ({activeName})</strong> bids:
		</span>
		{#if isDealerTurn}
			<span class="dealer-tag">dealer</span>
		{/if}
	</header>

	<p class="tally" class:tally-amber={tallyAmber} class:tally-green={!tallyAmber}>
		Total bids: <strong>{bidsPlacedSum}</strong> / {cardsDealt}
		{#if isDealerTurn && forbidden !== null}
			· dealer cannot bid <strong>{forbidden}</strong>
		{/if}
	</p>

	<div class="buttons">
		{#each bidValues as bid}
			{@const legal = isLegal(bid)}
			{@const isForbidden = isDealerTurn && forbidden === bid}
			{@const isRecommended = isHumanTurn && recommendedBid === bid}
			<button
				type="button"
				class="bid-btn"
				class:bid-illegal={!legal}
				class:bid-forbidden={isForbidden}
				class:bid-recommended={isRecommended}
				disabled={!legal || submitting}
				title={isForbidden ? 'Dealer constraint: would make total bids = cards_dealt' : ''}
				onclick={() => handleClick(bid)}
			>
				{bid}
			</button>
		{/each}
	</div>

	{#if isHumanTurn && bidSuggestion}
		<div class="ai-strip">
			<div class="ai-label">AI suggestion</div>
			<div class="ai-bids">
				{#each topBids as entry}
					<button
						type="button"
						class="ai-pill"
						class:ai-pill-recommended={entry.bid === recommendedBid}
						disabled={!isLegal(entry.bid) || submitting}
						onclick={() => handleClick(entry.bid)}
					>
						<span class="ai-bid-val">{entry.bid}</span>
						<span class="ai-bid-prob">{(entry.prob * 100).toFixed(0)}%</span>
					</button>
				{/each}
			</div>
			<div class="ai-hint">
				v={bidSuggestion.value_estimate.toFixed(2)} · press
				<kbd>Enter</kbd> for recommended
			</div>
		</div>
	{/if}

	{#if errorMessage}
		<p class="error">{errorMessage}</p>
	{/if}
</div>

<style>
	.keypad {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		padding: 0.75rem 1rem;
		height: 100%;
		box-sizing: border-box;
		overflow-y: auto;
	}

	.header {
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}

	.active-label {
		font-size: 1rem;
		color: #0f172a;
	}

	.dealer-tag {
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: #b45309;
		background: #fef3c7;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
	}

	.tally {
		margin: 0;
		font-size: 0.8rem;
		font-variant-numeric: tabular-nums;
	}

	.tally-green {
		color: #15803d;
	}

	.tally-amber {
		color: #b45309;
	}

	.buttons {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(2.5rem, 1fr));
		gap: 0.4rem;
	}

	.bid-btn {
		font-size: 1.05rem;
		font-weight: 600;
		padding: 0.6rem 0.4rem;
		border: 1px solid #cbd5e1;
		border-radius: 4px;
		background: #fff;
		color: #0f172a;
		cursor: pointer;
		font-variant-numeric: tabular-nums;
		transition: background-color 0.07s, border-color 0.07s;
	}

	.bid-btn:hover:not(:disabled) {
		background: #e2e8f0;
		border-color: #94a3b8;
	}

	.bid-illegal {
		opacity: 0.3;
		cursor: not-allowed;
	}

	.bid-forbidden {
		text-decoration: line-through;
		color: #b91c1c;
	}

	.bid-recommended {
		border-color: #15803d;
		box-shadow: inset 0 0 0 2px #15803d;
	}

	.ai-strip {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		padding: 0.5rem 0.6rem;
		border: 1px solid #bbf7d0;
		background: #f0fdf4;
		border-radius: 4px;
	}

	.ai-label {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		font-weight: 600;
		color: #166534;
	}

	.ai-bids {
		display: flex;
		gap: 0.35rem;
		flex-wrap: wrap;
	}

	.ai-pill {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.1rem;
		padding: 0.3rem 0.6rem;
		border-radius: 3px;
		border: 1px solid #86efac;
		background: #ecfdf5;
		cursor: pointer;
		font-variant-numeric: tabular-nums;
	}

	.ai-pill:hover:not(:disabled) {
		background: #dcfce7;
	}

	.ai-pill-recommended {
		background: #16a34a;
		color: #fff;
		border-color: #15803d;
		transform: translateY(-2px);
	}

	.ai-pill-recommended:hover:not(:disabled) {
		background: #15803d;
	}

	.ai-bid-val {
		font-size: 1rem;
		font-weight: 700;
	}

	.ai-bid-prob {
		font-size: 0.7rem;
	}

	.ai-hint {
		font-size: 0.7rem;
		color: #166534;
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
		font-size: 0.8rem;
		color: #b91c1c;
	}
</style>
