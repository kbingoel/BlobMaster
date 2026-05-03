<script lang="ts">
	import { commands, type SessionSnapshot } from '$lib/api';

	let versionResult = $state<string | null>(null);
	let snapshot = $state<SessionSnapshot | null>(null);
	let error = $state<string | null>(null);
	let pending = $state(false);

	async function callEngine() {
		pending = true;
		error = null;
		try {
			versionResult = await commands.engineVersion();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			pending = false;
		}
	}

	async function startSmokeGame() {
		pending = true;
		error = null;
		const result = await commands.newGame({
			num_players: 5,
			start_cards: 7,
			human_seat: 0,
			dealer: 0,
			player_names: ['You', 'P1', 'P2', 'P3', 'P4'],
			trump_mode: 'auto-rotate'
		});
		if (result.status === 'ok') {
			snapshot = result.data;
		} else {
			error = `${result.error.kind}: ${'message' in result.error ? result.error.message : ''}`;
			snapshot = null;
		}
		pending = false;
	}
</script>

<main class="mx-auto max-w-3xl px-6 py-12">
	<h1 class="mb-2 text-3xl font-semibold tracking-tight">BlobMaster</h1>
	<p class="mb-8 text-sm text-slate-500">
		Session 9.2 — typed IPC contract via tauri-specta.
	</p>

	<div class="flex gap-3">
		<button
			type="button"
			onclick={callEngine}
			disabled={pending}
			class="rounded bg-slate-900 px-4 py-2 text-white shadow-sm hover:bg-slate-700 disabled:opacity-50"
		>
			{pending ? 'Calling…' : 'engine_version'}
		</button>
		<button
			type="button"
			onclick={startSmokeGame}
			disabled={pending}
			class="rounded bg-emerald-700 px-4 py-2 text-white shadow-sm hover:bg-emerald-800 disabled:opacity-50"
		>
			{pending ? 'Calling…' : 'new_game (5p × 7c)'}
		</button>
	</div>

	{#if versionResult}
		<pre class="mt-6 rounded bg-slate-100 px-4 py-3 text-sm text-slate-800">{versionResult}</pre>
	{/if}

	{#if snapshot}
		<pre class="mt-6 max-h-96 overflow-auto rounded bg-slate-100 px-4 py-3 text-xs text-slate-800">{JSON.stringify(snapshot, null, 2)}</pre>
	{/if}

	{#if error}
		<pre class="mt-6 rounded bg-red-100 px-4 py-3 text-sm text-red-800">{error}</pre>
	{/if}
</main>
