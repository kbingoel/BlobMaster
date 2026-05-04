<script lang="ts">
	import type { SessionSnapshot } from '$lib/api';
	import { isRed, rankLabel, suitGlyph, cardSuit, cardRank } from '$lib/cardUtils';

	interface Props {
		snapshot: SessionSnapshot;
	}

	let { snapshot }: Props = $props();

	type Slot = { card: number; trickIdx: number } | null;

	let players = $derived.by(() => {
		const np = snapshot.num_players;
		const ranking = rankFromScores(snapshot.cumulative_scores);
		return Array.from({ length: np }, (_, seat) => ({
			seat,
			name: snapshot.player_names[seat] ?? `P${seat}`,
			bid: snapshot.bids[seat] ?? null,
			tricksWon: snapshot.tricks_won[seat] ?? 0,
			score: snapshot.cumulative_scores[seat] ?? 0,
			rank: ranking[seat],
			isActive: seat === snapshot.current_player,
			isDealer: seat === snapshot.dealer,
			isHuman: seat === snapshot.human_seat,
			slots: slotsForSeat(seat)
		}));
	});

	function slotsForSeat(seat: number): Slot[] {
		const cardsDealt = snapshot.cards_dealt;
		const slots: Slot[] = Array(cardsDealt).fill(null);
		// Completed tricks for this round: each seat plays exactly one card per trick.
		// Slot index = trick index — i.e. the round's nth card per player goes into slot n.
		for (let t = 0; t < snapshot.trick_history.length; t++) {
			const rec = snapshot.trick_history[t];
			for (const play of rec.plays) {
				if (play.seat === seat) {
					slots[t] = { card: play.card, trickIdx: t };
				}
			}
		}
		// In-progress trick — fill the next slot for whichever seats have already played.
		const tIdx = snapshot.trick_history.length;
		if (tIdx < cardsDealt) {
			for (const play of snapshot.trick_in_progress) {
				if (play.seat === seat) {
					slots[tIdx] = { card: play.card, trickIdx: tIdx };
				}
			}
		}
		return slots;
	}

	function rankFromScores(scores: number[]): number[] {
		const indexed = scores.map((s, i) => ({ s, i }));
		indexed.sort((a, b) => b.s - a.s);
		const ranks = new Array(scores.length).fill(0);
		let lastScore = Number.NaN;
		let lastRank = 0;
		indexed.forEach((entry, i) => {
			const rank = entry.s === lastScore ? lastRank : i + 1;
			ranks[entry.i] = rank;
			lastScore = entry.s;
			lastRank = rank;
		});
		return ranks;
	}

	function makeStatus(bid: number | null, tricksWon: number, tricksLeft: number) {
		if (bid === null) return '' as const;
		if (tricksWon > bid) return 'over' as const;
		if (tricksWon + tricksLeft < bid) return 'under' as const;
		if (tricksWon === bid && tricksLeft > 0) return 'made' as const;
		return 'open' as const;
	}

	let cardsDealt = $derived(snapshot.cards_dealt);
	let tricksCompleted = $derived(snapshot.trick_history.length);
	let tricksLeft = $derived(cardsDealt - tricksCompleted);

	// Pulse the winner of the most recent completed trick. Keyed by trick
	// index so a 200ms one-shot animation re-fires whenever a new trick lands.
	let lastTrick = $derived(
		snapshot.trick_history.length > 0
			? snapshot.trick_history[snapshot.trick_history.length - 1]
			: null
	);
	let pulseSeat = $derived(lastTrick?.winner ?? -1);
	let pulseKey = $derived(snapshot.trick_history.length);
</script>

<div class="players">
	{#each players as p}
		{@const status = makeStatus(p.bid, p.tricksWon, tricksLeft)}
		<div
			class="row"
			class:row-active={p.isActive}
			class:row-human={p.isHuman}
		>
			{#if p.seat === pulseSeat}
				{#key pulseKey}
					<div class="pulse-overlay" aria-hidden="true"></div>
				{/key}
			{/if}
			<span class="rank-badge">#{p.rank}</span>

			<div class="name-col">
				<span class="seat">P{p.seat}</span>
				<span class="name">{p.name}</span>
				{#if p.isDealer}
					<span class="dealer-pill">D</span>
				{/if}
			</div>

			<span
				class="bid-chip"
				class:chip-over={status === 'over'}
				class:chip-made={status === 'made'}
				class:chip-under={status === 'under'}
			>
				{p.tricksWon} / {p.bid === null ? '—' : p.bid}
			</span>

			<span class="score">{p.score}</span>

			<div class="slots">
				{#each p.slots as slot, idx (idx)}
					{#if slot}
						{@const suit = cardSuit(slot.card)}
						{@const rank = cardRank(slot.card)}
						<span class="slot played" class:suit-red={isRed(suit)}>
							{rankLabel(rank)}{suitGlyph(suit)}
						</span>
					{:else}
						<span class="slot face-down">▒</span>
					{/if}
				{/each}
			</div>
		</div>
	{/each}
</div>

<style>
	.players {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding: 0.5rem;
		box-sizing: border-box;
		gap: 0.3rem;
		overflow-y: auto;
		position: relative;
	}

	.row {
		display: grid;
		grid-template-columns: auto 1fr auto auto 1fr;
		align-items: center;
		gap: 0.6rem;
		padding: 0.4rem 0.6rem;
		border-radius: 4px;
		border: 1px solid #e2e8f0;
		background: #fff;
		font-size: 0.85rem;
		position: relative;
	}

	.row-active {
		background: #fef9c3;
		border-color: #facc15;
		box-shadow: inset 3px 0 0 #ca8a04;
	}

	.row-human {
		font-weight: 500;
	}

	.rank-badge {
		font-size: 0.7rem;
		font-weight: 700;
		color: #475569;
		background: #f1f5f9;
		padding: 0.15rem 0.4rem;
		border-radius: 3px;
	}

	.name-col {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		min-width: 0;
	}

	.seat {
		font-size: 0.7rem;
		color: #94a3b8;
		font-variant-numeric: tabular-nums;
	}

	.name {
		font-weight: 600;
		color: #0f172a;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.dealer-pill {
		font-size: 0.7rem;
		font-weight: 700;
		color: #b45309;
		background: #fef3c7;
		padding: 0.05rem 0.3rem;
		border-radius: 3px;
	}

	.bid-chip {
		font-variant-numeric: tabular-nums;
		font-size: 0.85rem;
		font-weight: 600;
		padding: 0.1rem 0.5rem;
		border-radius: 999px;
		background: #f1f5f9;
		color: #334155;
	}

	.chip-made {
		background: #dcfce7;
		color: #15803d;
	}

	.chip-over,
	.chip-under {
		background: #fee2e2;
		color: #b91c1c;
	}

	.score {
		font-variant-numeric: tabular-nums;
		font-weight: 600;
		color: #0f172a;
		min-width: 2ch;
		text-align: right;
	}

	.slots {
		display: flex;
		gap: 0.15rem;
		justify-content: flex-end;
		font-size: 0.8rem;
	}

	.slot {
		font-family: ui-monospace, 'SFMono-Regular', monospace;
		min-width: 2.1rem;
		text-align: center;
		padding: 0.05rem 0.25rem;
		border-radius: 3px;
		font-variant-numeric: tabular-nums;
	}

	.face-down {
		color: #cbd5e1;
		background: #f8fafc;
		border: 1px solid #e2e8f0;
	}

	.played {
		color: #0f172a;
		background: #fff;
		border: 1px solid #cbd5e1;
		font-weight: 600;
	}

	.played.suit-red {
		color: #dc2626;
	}

	.pulse-overlay {
		position: absolute;
		inset: 0;
		pointer-events: none;
		border-radius: 4px;
		background: #facc15;
		opacity: 0;
		mix-blend-mode: multiply;
		animation: pulse-flash 200ms ease-out 1;
	}

	@keyframes pulse-flash {
		0% { opacity: 0; }
		40% { opacity: 0.45; }
		100% { opacity: 0; }
	}
</style>
