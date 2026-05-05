<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { commands, type AiSuggestion, type EngineSettings, type SessionSnapshot } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import CardGrid from '$lib/components/CardGrid.svelte';
	import BiddingKeypad from '$lib/components/BiddingKeypad.svelte';
	import PlayersPanel from '$lib/components/PlayersPanel.svelte';
	import MyHandPanel from '$lib/components/MyHandPanel.svelte';
	import RoundProgressStrip from '$lib/components/RoundProgressStrip.svelte';

	let snapshot = $state<SessionSnapshot | null>(null);
	let aiSuggestion = $state<AiSuggestion | null>(null);
	let submitting = $state(false);
	let lastError = $state<string | null>(null);

	// Live engine-settings state. Loaded once from disk so the GUI matches
	// what the user picked in /setup, then mutated locally and pushed
	// through `update_engine_settings` on every change.
	let engineSettings = $state<EngineSettings>({
		temperature: 1.0,
		mcts_simulations: 400,
		determinization_samples: 8,
		deterministic_seed: null,
		eval_display: 'win-rate',
		show_grid_eval: false
	});

	// Bumped on every snapshot replacement so we can ignore stale AI replies
	// (e.g. Player 3 plays before iter-229 finished thinking for Player 2).
	let suggestionRequestId = 0;

	onMount(() => {
		const unsub = sessionStore.subscribe((s) => {
			snapshot = s;
		});
		if (snapshot === null) goto('/setup', { replaceState: true });
		// Pull the persisted engine settings — user's setup-screen choices.
		commands.loadAppSettings().then((res) => {
			if (res.status === 'ok') {
				engineSettings = res.data.engine_settings;
			}
		});
		return unsub;
	});

	$effect(() => {
		if (!snapshot) return;
		// Phase transitions out of /play are handled here so any command that
		// flips the phase (the last apply_play, an advance_round on resume)
		// gets the user to the right screen automatically.
		if (snapshot.phase === 'scoring') {
			goto('/round-summary');
			return;
		}
		if (snapshot.phase === 'complete') {
			goto('/end');
			return;
		}
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

	async function updateEngineSettings(next: EngineSettings) {
		engineSettings = next;
		// Push through to the engine so the next AI call honors the new
		// values; refresh the suggestion if the human is still active.
		await commands.updateEngineSettings(next);
		if (snapshot && snapshot.current_player === snapshot.human_seat) {
			fetchSuggestion();
		}
	}

	let perCardEvals = $derived(
		aiSuggestion && aiSuggestion.phase === 'playing' ? aiSuggestion.per_card : []
	);
	let gameKey = $derived(snapshot ? `${snapshot.start_cards}-${snapshot.num_players}` : '');
</script>

{#if snapshot}
	<div class="play-layout">
		<RoundProgressStrip currentRound={snapshot.round_idx} {gameKey} />
		<div class="play-body">
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
							settings={engineSettings}
							{submitting}
							errorMessage={lastError}
							onPlay={playCard}
							onSettingsChange={updateEngineSettings}
						/>
					{:else}
						<div class="pane-placeholder">
							<p class="placeholder-label">Phase: {snapshot.phase}</p>
						</div>
					{/if}
				</div>
			</div>

			<div class="right-col">
				<CardGrid
					{snapshot}
					mode="play"
					onCardclick={handleGridClick}
					perCardEvals={engineSettings.show_grid_eval ? perCardEvals : []}
					evalMode={engineSettings.eval_display}
				/>
			</div>
		</div>
	</div>
{:else}
	<div class="loading">Redirecting to setup…</div>
{/if}

<style>
	.play-layout {
		display: flex;
		flex-direction: column;
		width: 100vw;
		height: 100vh;
		overflow: hidden;
	}

	.play-body {
		flex: 1;
		min-height: 0;
		display: flex;
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
