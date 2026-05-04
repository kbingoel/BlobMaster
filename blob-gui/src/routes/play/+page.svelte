<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { commands, type AiSuggestion, type SessionSnapshot } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import CardGrid from '$lib/components/CardGrid.svelte';
	import BiddingKeypad from '$lib/components/BiddingKeypad.svelte';
	import PlayersPanel from '$lib/components/PlayersPanel.svelte';
	import MyHandPanel from '$lib/components/MyHandPanel.svelte';

	let snapshot = $state<SessionSnapshot | null>(null);
	let aiSuggestion = $state<AiSuggestion | null>(null);
	let submitting = $state(false);
	let lastError = $state<string | null>(null);

	// Bumped on every snapshot replacement so we can ignore stale AI replies
	// (e.g. Player 3 plays before iter-229 finished thinking for Player 2).
	let suggestionRequestId = 0;

	onMount(() => {
		const unsub = sessionStore.subscribe((s) => {
			snapshot = s;
		});
		if (snapshot === null) goto('/setup', { replaceState: true });
		return unsub;
	});

	// Auto-fetch an AI suggestion whenever it's the human's turn in bidding
	// or playing. Session 9.7 streams partial updates via the `ai-thinking`
	// event channel; for 9.6 we just await the final reply.
	$effect(() => {
		if (!snapshot) return;
		const humanActive = snapshot.current_player === snapshot.human_seat;
		const phase = snapshot.phase;
		if (!humanActive || (phase !== 'bidding' && phase !== 'playing')) {
			aiSuggestion = null;
			return;
		}
		fetchSuggestion();
	});

	async function fetchSuggestion() {
		const id = ++suggestionRequestId;
		const result = await commands.requestAiSuggestion();
		if (id !== suggestionRequestId) return; // stale
		if (result.status === 'ok') {
			aiSuggestion = result.data;
		} else {
			aiSuggestion = null;
		}
	}

	async function submitBid(bid: number) {
		if (!snapshot || submitting) return;
		const seat = snapshot.current_player;
		submitting = true;
		lastError = null;
		// Drop the suggestion immediately — it's tied to the seat that just bid.
		aiSuggestion = null;
		const result = await commands.submitBid(seat, bid);
		submitting = false;
		if (result.status === 'ok') {
			sessionStore.set(result.data);
		} else {
			const err = result.error;
			lastError = 'message' in err ? err.message : err.kind;
		}
	}

	async function playCard(card: number) {
		if (!snapshot || submitting) return;
		if (snapshot.phase !== 'playing') return;
		const seat = snapshot.current_player;
		submitting = true;
		lastError = null;
		// Suggestion is tied to the active seat — invalidate immediately so
		// the bottom-left pane doesn't render stale eval against the next seat.
		aiSuggestion = null;
		const result = await commands.recordCardPlayed(seat, card);
		submitting = false;
		if (result.status === 'ok') {
			sessionStore.set(result.data);
		} else {
			const err = result.error;
			lastError = 'message' in err ? err.message : err.kind;
		}
	}

	function handleGridClick(card: number) {
		if (!snapshot) return;
		if (snapshot.phase === 'playing') {
			playCard(card);
		}
	}
</script>

{#if snapshot}
	<div class="play-layout">
		<!-- ── Left column: players (top) + phase-specific panel (bottom) -->
		<div class="left-col">
			<div class="top-left">
				<PlayersPanel {snapshot} />
			</div>
			<div class="bottom-left">
				{#if snapshot.phase === 'bidding'}
					<BiddingKeypad
						{snapshot}
						suggestion={aiSuggestion}
						{submitting}
						errorMessage={lastError}
						onSubmit={submitBid}
					/>
				{:else if snapshot.phase === 'playing'}
					<MyHandPanel
						{snapshot}
						suggestion={aiSuggestion}
						{submitting}
						errorMessage={lastError}
						onPlay={playCard}
					/>
				{:else}
					<div class="pane-placeholder">
						<p class="placeholder-label">Phase: {snapshot.phase}</p>
					</div>
				{/if}
			</div>
		</div>

		<!-- ── Right column: master CardGrid ─────────────────── -->
		<div class="right-col">
			<CardGrid {snapshot} mode="play" onCardclick={handleGridClick} />
		</div>
	</div>
{:else}
	<div class="loading">Redirecting to setup…</div>
{/if}

<style>
	.play-layout {
		display: flex;
		width: 100vw;
		height: 100vh;
		overflow: hidden;
	}

	.left-col {
		width: 50%;
		display: flex;
		flex-direction: column;
		border-right: 1px solid #e2e8f0;
	}

	.top-left {
		flex: 1;
		min-height: 0;
		border-bottom: 1px solid #e2e8f0;
		overflow: hidden;
	}

	.bottom-left {
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}

	.pane-placeholder {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		gap: 0.25rem;
	}

	.placeholder-label {
		font-size: 0.875rem;
		color: #94a3b8;
		margin: 0;
	}

	.right-col {
		width: 50%;
		height: 100%;
		padding: 6px;
		box-sizing: border-box;
		overflow: hidden;
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
