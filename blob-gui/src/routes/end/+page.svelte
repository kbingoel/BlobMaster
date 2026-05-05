<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { save as saveDialog } from '@tauri-apps/plugin-dialog';
	import { commands, type SessionSnapshot } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import RoundProgressStrip from '$lib/components/RoundProgressStrip.svelte';

	let snapshot = $state<SessionSnapshot | null>(null);
	let exporting = $state(false);
	let exportError = $state<string | null>(null);
	let exportNote = $state<string | null>(null);

	onMount(() => {
		const unsub = sessionStore.subscribe((s) => {
			snapshot = s;
		});
		if (snapshot === null) {
			goto('/setup', { replaceState: true });
			return unsub;
		}
		if (snapshot.phase !== 'complete') {
			// Race: end-of-game was navigated to before the engine got there.
			goto('/play', { replaceState: true });
		}
		return unsub;
	});

	type Row = { seat: number; name: string; score: number; rank: number };

	let standings = $derived<Row[]>(
		(() => {
			if (!snapshot) return [];
			const np = snapshot.num_players;
			const rows = Array.from({ length: np }, (_, i) => ({
				seat: i,
				name: snapshot!.player_names[i] ?? `P${i}`,
				score: snapshot!.cumulative_scores[i] ?? 0,
				rank: 0
			}));
			rows.sort((a, b) => b.score - a.score || a.seat - b.seat);
			let lastScore = -1;
			let lastRank = 0;
			rows.forEach((r, i) => {
				if (r.score !== lastScore) {
					lastRank = i + 1;
					lastScore = r.score;
				}
				r.rank = lastRank;
			});
			return rows;
		})()
	);

	let winner = $derived(standings[0] ?? null);
	let winnerIsHuman = $derived(snapshot && winner ? winner.seat === snapshot.human_seat : false);
	let isTie = $derived(standings.length > 1 && standings[0].score === standings[1].score);

	async function exportLog() {
		exporting = true;
		exportError = null;
		exportNote = null;
		const target = await saveDialog({
			defaultPath: `blobmaster-game-${Date.now()}.json`,
			filters: [{ name: 'Session log (JSON)', extensions: ['json'] }]
		});
		if (typeof target !== 'string') {
			exporting = false;
			return;
		}
		const res = await commands.exportSessionLog(target);
		if (res.status === 'ok') {
			exportNote = `Saved to ${res.data}`;
		} else {
			exportError = 'message' in res.error ? res.error.message : res.error.kind;
		}
		exporting = false;
	}

	function newGame() {
		sessionStore.set(null);
		goto('/setup');
	}

	let gameKey = $derived(snapshot ? `${snapshot.start_cards}-${snapshot.num_players}` : '');
	let lastRoundIdx = $derived(snapshot ? snapshot.total_rounds - 1 : 0);
</script>

{#if snapshot}
	<div class="layout">
		<RoundProgressStrip currentRound={lastRoundIdx} {gameKey} editable={false} />

		<main class="content">
			<header class="hero">
				<p class="eyebrow">Game complete</p>
				<h1 class="winner-line">
					{#if isTie}
						Tie at <span class="score">{winner?.score ?? 0}</span>
					{:else if winner}
						<strong>{winner.name}</strong> wins · <span class="score">{winner.score}</span>
						{#if winnerIsHuman}<span class="you-tag">You</span>{/if}
					{/if}
				</h1>
				<p class="meta">
					{snapshot.num_players} players · {snapshot.total_rounds} rounds · C={snapshot.start_cards}
				</p>
			</header>

			<table class="scoreboard">
				<thead>
					<tr>
						<th class="rank-col">#</th>
						<th>Player</th>
						<th class="num">Score</th>
					</tr>
				</thead>
				<tbody>
					{#each standings as row (row.seat)}
						<tr
							class:human={row.seat === snapshot.human_seat}
							class:winner={row.rank === 1 && !isTie}
						>
							<td class="rank-col tabular-nums">{row.rank}</td>
							<td>{row.name}{#if row.seat === snapshot.human_seat}<span class="you-tag inline">You</span>{/if}</td>
							<td class="num tabular-nums score-cell">{row.score}</td>
						</tr>
					{/each}
				</tbody>
			</table>

			{#if exportError}
				<p class="error-text">{exportError}</p>
			{/if}
			{#if exportNote}
				<p class="success-text">{exportNote}</p>
			{/if}

			<div class="actions">
				<button type="button" class="btn-secondary" onclick={exportLog} disabled={exporting}>
					{exporting ? 'Exporting…' : 'Export log'}
				</button>
				<button type="button" class="btn-primary" onclick={newGame}>Start new game</button>
			</div>
		</main>
	</div>
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
	.content {
		flex: 1;
		min-height: 0;
		overflow: auto;
		padding: 2rem 1.5rem;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 1.5rem;
	}
	.hero {
		text-align: center;
	}
	.eyebrow {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: #94a3b8;
		margin: 0;
	}
	.winner-line {
		font-size: 2rem;
		font-weight: 600;
		color: #0f172a;
		margin: 0.25rem 0 0;
	}
	.score {
		font-variant-numeric: tabular-nums;
		color: #15803d;
	}
	.you-tag {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 2px 8px;
		font-size: 0.7rem;
		background: #15803d;
		color: white;
		border-radius: 999px;
		vertical-align: middle;
	}
	.you-tag.inline {
		font-size: 0.6rem;
		padding: 1px 6px;
	}
	.meta {
		font-size: 0.85rem;
		color: #64748b;
		margin: 0.5rem 0 0;
	}
	.scoreboard {
		width: 100%;
		max-width: 480px;
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
	.rank-col {
		width: 3rem;
		color: #94a3b8;
	}
	.score-cell {
		font-weight: 600;
		color: #0f172a;
	}
	tr.human {
		background: #ecfdf5;
	}
	tr.winner td {
		background: #fef9c3;
		font-weight: 600;
	}
	tr.winner.human td {
		background: linear-gradient(to right, #ecfdf5, #fef9c3);
	}
	.error-text {
		color: #dc2626;
		font-size: 0.85rem;
		margin: 0;
	}
	.success-text {
		color: #15803d;
		font-size: 0.85rem;
		margin: 0;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 1rem;
	}
	.btn-secondary {
		padding: 0.55rem 1.25rem;
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
		padding: 0.55rem 1.5rem;
		border-radius: 4px;
		background: #15803d;
		color: white;
		font-size: 0.875rem;
		font-weight: 500;
		border: none;
		cursor: pointer;
	}
	.btn-primary:hover:not(:disabled) {
		background: #166534;
	}
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
