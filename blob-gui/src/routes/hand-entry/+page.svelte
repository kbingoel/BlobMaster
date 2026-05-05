<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { commands, type SessionSnapshot } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import { suitGlyph, isRed } from '$lib/cardUtils';
	import CardGrid from '$lib/components/CardGrid.svelte';
	import RoundProgressStrip from '$lib/components/RoundProgressStrip.svelte';

	let snapshot = $state<SessionSnapshot | null>(null);
	let selectedCards = $state<number[]>([]);
	let submitting = $state(false);
	let submitError = $state<string | null>(null);

	onMount(() => {
		// subscribe fires immediately with the current store value
		const unsub = sessionStore.subscribe((s) => {
			snapshot = s;
		});
		if (snapshot === null) {
			goto('/setup', { replaceState: true });
			return unsub;
		}
		// Resume routing: human already has a hand → straight to /play.
		// Phases beyond Bidding belong to other routes.
		if (snapshot.phase === 'scoring') goto('/round-summary', { replaceState: true });
		else if (snapshot.phase === 'complete') goto('/end', { replaceState: true });
		else if (snapshot.phase === 'playing' || snapshot.human_hand.length > 0) {
			goto('/play', { replaceState: true });
		}
		return unsub;
	});

	let cardsDealt = $derived(snapshot?.cards_dealt ?? 0);
	let roundNum = $derived((snapshot?.round_idx ?? 0) + 1);
	let totalRounds = $derived(snapshot?.total_rounds ?? 0);
	let trump = $derived(snapshot?.trump_suit ?? 0);
	let dealer = $derived(snapshot?.dealer ?? 0);
	let playerNames = $derived(snapshot?.player_names ?? []);
	let canSubmit = $derived(selectedCards.length === cardsDealt && cardsDealt > 0 && !submitting);
	let gameKey = $derived(snapshot ? `${snapshot.start_cards}-${snapshot.num_players}` : '');

	function handleCardclick(cardIdx: number) {
		if (selectedCards.includes(cardIdx)) {
			selectedCards = selectedCards.filter((c) => c !== cardIdx);
		} else if (selectedCards.length < cardsDealt) {
			selectedCards = [...selectedCards, cardIdx];
		}
	}

	async function confirmHand() {
		if (!canSubmit) return;
		submitting = true;
		submitError = null;
		const result = await commands.setHumanHand(selectedCards);
		submitting = false;
		if (result.status === 'ok') {
			sessionStore.set(result.data);
			goto('/play');
		} else {
			const err = result.error;
			submitError = 'message' in err ? err.message : err.kind;
		}
	}
</script>

{#if snapshot}
	<div class="hand-entry-layout">
		<RoundProgressStrip currentRound={snapshot.round_idx} {gameKey} />

		<!-- ── Sticky info strip ──────────────────────────────── -->
		<header class="info-strip">
			<span class="strip-item">
				Round <strong>{roundNum}</strong> / {totalRounds}
			</span>
			<span class="strip-item">
				Cards: <strong>{cardsDealt}</strong>
			</span>
			<span class="strip-item" class:trump-red={isRed(trump)} class:trump-dark={!isRed(trump)}>
				Trump: <strong>{suitGlyph(trump)}</strong>
			</span>
			<span class="strip-item">
				Dealer: <strong>{playerNames[dealer] ?? `P${dealer}`}</strong>
			</span>
		</header>

		<!-- ── Card grid fills remaining height ──────────────── -->
		<div class="grid-area">
			<CardGrid
				{snapshot}
				mode="hand-entry"
				{selectedCards}
				onCardclick={handleCardclick}
			/>
		</div>

		<!-- ── Action footer ─────────────────────────────────── -->
		<footer class="action-bar">
			<div class="counter" class:counter-ok={canSubmit} class:counter-warn={!canSubmit}>
				{selectedCards.length} / {cardsDealt} selected
			</div>

			{#if submitError}
				<p class="submit-error">{submitError}</p>
			{/if}

			<div class="action-buttons">
				<button
					type="button"
					onclick={() => goto('/setup')}
					class="btn-secondary"
				>
					Back to setup
				</button>
				<button
					type="button"
					onclick={confirmHand}
					disabled={!canSubmit}
					class="btn-primary"
				>
					{submitting ? 'Confirming…' : 'Confirm hand'}
				</button>
			</div>
		</footer>
	</div>
{:else}
	<div class="loading">Redirecting to setup…</div>
{/if}

<style>
	/* Full-screen column layout: header | grid | footer */
	.hand-entry-layout {
		display: flex;
		flex-direction: column;
		width: 100vw;
		height: 100vh;
		overflow: hidden;
		background: #f8fafc;
	}

	/* ── Info strip ──────────────────────────────────────────── */
	.info-strip {
		flex-shrink: 0;
		display: flex;
		align-items: center;
		gap: 1.5rem;
		padding: 0.5rem 1rem;
		background: #0f172a; /* slate-900 */
		color: #f1f5f9; /* slate-100 */
		font-size: 0.8rem;
	}

	.strip-item {
		white-space: nowrap;
	}

	.trump-red {
		color: #fca5a5; /* red-300 */
	}

	.trump-dark {
		color: #f1f5f9;
	}

	/* ── Grid area ───────────────────────────────────────────── */
	.grid-area {
		flex: 1;
		min-height: 0; /* crucial: lets the flex child shrink below content size */
		padding: 6px;
	}

	/* ── Action bar ──────────────────────────────────────────── */
	.action-bar {
		flex-shrink: 0;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.6rem 1rem;
		border-top: 1px solid #e2e8f0;
		background: #ffffff;
	}

	.counter {
		font-size: 0.875rem;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}

	.counter-ok {
		color: #16a34a; /* green-600 */
	}

	.counter-warn {
		color: #64748b; /* slate-500 */
	}

	.submit-error {
		font-size: 0.8rem;
		color: #dc2626;
		margin: 0;
	}

	.action-buttons {
		display: flex;
		gap: 0.5rem;
	}

	.btn-secondary {
		padding: 0.4rem 1rem;
		border: 1px solid #cbd5e1;
		border-radius: 4px;
		background: #fff;
		font-size: 0.875rem;
		color: #475569;
		cursor: pointer;
	}

	.btn-secondary:hover {
		background: #f8fafc;
	}

	.btn-primary {
		padding: 0.4rem 1.2rem;
		border-radius: 4px;
		background: #15803d; /* green-700 */
		color: #fff;
		font-size: 0.875rem;
		font-weight: 500;
		border: none;
		cursor: pointer;
	}

	.btn-primary:hover:not(:disabled) {
		background: #166534; /* green-800 */
	}

	.btn-primary:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}

	.loading {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100vh;
		color: #64748b;
		font-size: 0.875rem;
	}
</style>
