<script lang="ts">
	import { onMount } from 'svelte';
	import { get } from 'svelte/store';
	import { goto } from '$app/navigation';
	import { commands, type RoundSummary, type SessionSnapshot } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import { trumpEditingStore } from '$lib/stores/trumpEditing';
	import { pushToast } from '$lib/stores/toast';
	import { trumpLabel, isRed, NO_TRUMP } from '$lib/cardUtils';
	import RoundProgressStrip from '$lib/components/RoundProgressStrip.svelte';

	let snapshot = $state<SessionSnapshot | null>(null);
	let summary = $state<RoundSummary | null>(null);
	let confirming = $state(false);
	let advancing = $state(false);
	let saveStatus = $state<'idle' | 'saving' | 'saved' | 'error'>('idle');
	let saveError = $state<string | null>(null);
	let advanceError = $state<string | null>(null);

	onMount(() => {
		const unsub = sessionStore.subscribe((s) => {
			snapshot = s;
		});
		if (snapshot === null) {
			goto('/setup', { replaceState: true });
			return unsub;
		}
		// Phase guard: this route only makes sense at scoring time. Bidding /
		// playing means the user landed here via stale navigation.
		if (snapshot.phase !== 'scoring' && snapshot.phase !== 'complete') {
			goto('/play', { replaceState: true });
			return unsub;
		}
		fetchSummary();
		// Auto-save on round end (per the plan: "save … on every round end").
		autoSave();
		return unsub;
	});

	async function fetchSummary() {
		const res = await commands.roundSummary();
		if (res.status === 'ok') summary = res.data;
	}

	async function autoSave() {
		saveStatus = 'saving';
		const res = await commands.saveSession();
		if (res.status === 'ok') {
			saveStatus = 'saved';
		} else {
			saveStatus = 'error';
			saveError = 'message' in res.error ? res.error.message : res.error.kind;
			pushToast(`Save failed: ${saveError}`, 'error');
		}
	}

	async function continueToNext() {
		if (!summary) return;
		// Final round → end-of-game screen, no advance_round needed (engine
		// already transitioned to Complete on the last apply_play+score).
		if (summary.is_final_round) {
			advancing = true;
			const res = await commands.advanceRound();
			advancing = false;
			if (res.status === 'ok') {
				sessionStore.set(res.data);
				goto('/end');
			} else {
				advanceError = 'message' in res.error ? res.error.message : res.error.kind;
			}
			return;
		}
		advancing = true;
		advanceError = null;
		const res = await commands.advanceRound();
		advancing = false;
		if (res.status !== 'ok') {
			advanceError = 'message' in res.error ? res.error.message : res.error.kind;
			confirming = false;
			return;
		}
		sessionStore.set(res.data);
		goto('/hand-entry');
	}

	function requestContinue() {
		// "Undo across round boundaries: explicitly disabled — surface in the
		// UI with a confirmation dialog on Continue."
		confirming = true;
	}

	function cancelContinue() {
		confirming = false;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		if (get(trumpEditingStore)) return;
		if (e.key === 'n' || e.key === 'N') {
			e.preventDefault();
			if (advancing) return;
			if (confirming) continueToNext();
			else requestContinue();
		}
	}

	let trumpClass = $derived(
		summary ? (isRed(summary.trump_suit) ? 'trump-red' : summary.trump_suit === NO_TRUMP ? 'trump-nt' : 'trump-dark') : ''
	);
	let gameKey = $derived(snapshot ? `${snapshot.start_cards}-${snapshot.num_players}` : '');
	let currentRound = $derived(summary?.round_idx ?? 0);
</script>

<svelte:window onkeydown={handleKeydown} />

{#if snapshot && summary}
	<div class="layout">
		<RoundProgressStrip
			{currentRound}
			{gameKey}
			onTrumpsSaved={(s) => sessionStore.set(s)}
		/>

		<header class="header">
			<div>
				<h1 class="title">
					Round {summary.round_idx + 1} / {snapshot.total_rounds}
					{#if summary.is_final_round}
						<span class="final-badge">final</span>
					{/if}
				</h1>
				<p class="meta">
					<span>{summary.cards_dealt} cards</span>
					<span class="dot">·</span>
					<span class={trumpClass}>Trump <strong>{trumpLabel(summary.trump_suit)}</strong></span>
					<span class="dot">·</span>
					<span>Dealer <strong>{summary.player_names[summary.dealer] ?? `P${summary.dealer}`}</strong></span>
				</p>
			</div>
			<div class="save-status">
				{#if saveStatus === 'saving'}
					<span class="saving">Saving…</span>
				{:else if saveStatus === 'saved'}
					<span class="saved">✓ Saved</span>
				{:else if saveStatus === 'error'}
					<span class="error" title={saveError ?? ''}>Save failed</span>
				{/if}
			</div>
		</header>

		<main class="table-area">
			<table class="scoreboard">
				<thead>
					<tr>
						<th class="seat-col">Seat</th>
						<th>Player</th>
						<th class="num">Bid</th>
						<th class="num">Won</th>
						<th class="num">Round</th>
						<th class="num">Cumulative</th>
					</tr>
				</thead>
				<tbody>
					{#each summary.rows as row (row.seat)}
						<tr
							class:human={row.seat === snapshot.human_seat}
							class:made={row.bid === row.tricks_won}
							class:missed={row.bid !== row.tricks_won}
						>
							<td class="seat-col tabular-nums">{row.seat}</td>
							<td>
								{summary.player_names[row.seat] ?? `P${row.seat}`}
								{#if row.seat === summary.dealer}<span class="dealer-tag" title="Dealer">D</span>{/if}
							</td>
							<td class="num tabular-nums">{row.bid}</td>
							<td class="num tabular-nums">{row.tricks_won}</td>
							<td class="num tabular-nums">
								{#if row.round_score > 0}
									<span class="positive">+{row.round_score}</span>
								{:else}
									<span class="zero">0</span>
								{/if}
							</td>
							<td class="num tabular-nums cumulative">{row.cumulative_after}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</main>

		<footer class="footer">
			{#if advanceError}
				<p class="error-text">{advanceError}</p>
			{/if}
			{#if confirming}
				<div class="confirm">
					<p>
						{#if summary.is_final_round}
							This is the last round — finishing it ends the game.
						{:else}
							Continue to the next round? <strong>This round will be locked in</strong> — undo
							does not cross round boundaries.
						{/if}
					</p>
					<div class="confirm-buttons">
						<button type="button" class="btn-secondary" onclick={cancelContinue} disabled={advancing}>
							Cancel
						</button>
						<button type="button" class="btn-primary" onclick={continueToNext} disabled={advancing}>
							{advancing
								? 'Working…'
								: summary.is_final_round
									? 'Finish game'
									: 'Yes — continue'}
						</button>
					</div>
				</div>
			{:else}
				<button
					type="button"
					class="btn-primary big"
					onclick={requestContinue}
					disabled={advancing}
				>
					{summary.is_final_round ? 'Finish game' : 'Continue to next round'}
				</button>
			{/if}
		</footer>
	</div>
{:else if snapshot}
	<div class="loading">Computing round summary…</div>
{:else}
	<div class="loading">Redirecting to setup…</div>
{/if}

<style>
	.layout {
		display: flex;
		flex-direction: column;
		width: 100vw;
		height: 100vh;
		overflow: hidden;
		background: #f8fafc;
	}
	.header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 1rem;
		padding: 1rem 1.5rem 0.5rem;
	}
	.title {
		font-size: 1.5rem;
		font-weight: 600;
		margin: 0;
		color: #0f172a;
	}
	.final-badge {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 2px 8px;
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		background: #b45309;
		color: white;
		border-radius: 999px;
		vertical-align: middle;
	}
	.meta {
		margin: 0.25rem 0 0;
		font-size: 0.875rem;
		color: #64748b;
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
	}
	.dot {
		color: #cbd5e1;
	}
	.trump-red {
		color: #dc2626;
	}
	.trump-dark {
		color: #0f172a;
	}
	.trump-nt {
		color: #b45309;
		font-weight: 600;
	}
	.save-status {
		font-size: 0.75rem;
	}
	.saving {
		color: #64748b;
	}
	.saved {
		color: #15803d;
	}
	.error {
		color: #dc2626;
	}
	.table-area {
		flex: 1;
		min-height: 0;
		overflow: auto;
		padding: 1rem 1.5rem;
	}
	.scoreboard {
		width: 100%;
		max-width: 720px;
		margin: 0 auto;
		border-collapse: collapse;
		background: white;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		font-size: 0.95rem;
	}
	.scoreboard th,
	.scoreboard td {
		padding: 0.65rem 0.85rem;
		text-align: left;
		border-bottom: 1px solid #f1f5f9;
	}
	.scoreboard th {
		background: #f8fafc;
		font-weight: 600;
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #64748b;
	}
	.scoreboard tr:last-child td {
		border-bottom: none;
	}
	.scoreboard .num {
		text-align: right;
	}
	.tabular-nums {
		font-variant-numeric: tabular-nums;
	}
	.seat-col {
		width: 3rem;
		color: #94a3b8;
	}
	.cumulative {
		font-weight: 600;
		color: #0f172a;
	}
	tr.human {
		background: #ecfdf5;
	}
	tr.made td .positive {
		color: #15803d;
		font-weight: 600;
	}
	tr.missed td .zero {
		color: #94a3b8;
	}
	.dealer-tag {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 1px 6px;
		font-size: 0.65rem;
		background: #f59e0b;
		color: white;
		border-radius: 3px;
	}
	.footer {
		flex-shrink: 0;
		padding: 1rem 1.5rem;
		border-top: 1px solid #e2e8f0;
		background: white;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		align-items: flex-end;
	}
	.error-text {
		color: #dc2626;
		font-size: 0.85rem;
		margin: 0;
	}
	.confirm {
		display: flex;
		align-items: center;
		gap: 1rem;
		justify-content: flex-end;
		width: 100%;
	}
	.confirm p {
		margin: 0;
		font-size: 0.875rem;
		color: #475569;
		flex: 1;
	}
	.confirm-buttons {
		display: flex;
		gap: 0.5rem;
	}
	.btn-secondary {
		padding: 0.5rem 1rem;
		border: 1px solid #cbd5e1;
		border-radius: 4px;
		background: white;
		font-size: 0.875rem;
		color: #475569;
		cursor: pointer;
	}
	.btn-secondary:hover:not(:disabled) {
		background: #f8fafc;
	}
	.btn-primary {
		padding: 0.5rem 1.25rem;
		border-radius: 4px;
		background: #15803d;
		color: white;
		font-size: 0.875rem;
		font-weight: 500;
		border: none;
		cursor: pointer;
	}
	.btn-primary.big {
		padding: 0.65rem 1.75rem;
		font-size: 0.95rem;
	}
	.btn-primary:hover:not(:disabled) {
		background: #166534;
	}
	.btn-primary:disabled,
	.btn-secondary:disabled {
		opacity: 0.5;
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
