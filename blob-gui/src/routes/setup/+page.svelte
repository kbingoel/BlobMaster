<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { open as openDialog } from '@tauri-apps/plugin-dialog';
	import {
		commands,
		type AppSettings,
		type EngineSettings,
		type EvalDisplay,
		type GuiError,
		type ModelInfo,
		type PlayerConfig,
		type SavedSessionInfo,
		type TrumpMode
	} from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import { pushToast } from '$lib/stores/toast';

	const MIN_PLAYERS = 4;
	const MAX_PLAYERS = 7;

	function defaultPlayers(): PlayerConfig[] {
		return [
			{ name: 'You', is_human: true },
			{ name: 'P1', is_human: false },
			{ name: 'P2', is_human: false },
			{ name: 'P3', is_human: false },
			{ name: 'P4', is_human: false }
		];
	}

	// Stable row IDs — prevents focus loss when player name changes.
	let nextId = 0;
	function makeId() { return nextId++; }
	let playerIds = $state<number[]>(defaultPlayers().map(() => makeId()));

	let form = $state<AppSettings>({
		start_cards: 7,
		trump_mode: 'auto-rotate',
		players: defaultPlayers(),
		engine_settings: {
			temperature: 1.0,
			mcts_simulations: 400,
			determinization_samples: 8,
			deterministic_seed: null,
			eval_display: 'win-rate',
			show_grid_eval: false
		},
		last_model_path: null
	});

	let discovered = $state<ModelInfo[]>([]);
	let recents = $state<ModelInfo[]>([]);
	let loadedModel = $state<ModelInfo | null>(null);
	let modelLoading = $state(false);
	let modelError = $state<string | null>(null);

	let advancedOpen = $state(false);
	let useDeterministicSeed = $state(false);
	let seedInput = $state('0');
	let mctsCustom = $state(false);
	let mctsCustomValue = $state(400);

	let starting = $state(false);
	let startError = $state<string | null>(null);

	let savedSessions = $state<SavedSessionInfo[]>([]);
	let resuming = $state(false);
	let resumeError = $state<string | null>(null);

	// Dealer seat tracked separately so clicking D never reorders the list.
	let dealerSeat = $state(defaultPlayers().length - 1);

	const TRUMP_MODES: { value: TrumpMode; label: string }[] = [
		{ value: 'auto-rotate', label: 'Auto-rotate' },
		{ value: 'spades', label: '♠ Spades' },
		{ value: 'hearts', label: '♥ Hearts' },
		{ value: 'clubs', label: '♣ Clubs' },
		{ value: 'diamonds', label: '♦ Diamonds' },
		{ value: 'no-trump', label: 'No trump' }
	];
	const MCTS_PRESETS: number[] = [0, 100, 400, 1600, 6400];
	const EVAL_DISPLAY_OPTIONS: { value: EvalDisplay; label: string }[] = [
		{ value: 'win-rate', label: 'Win-rate' },
		{ value: 'policy', label: 'Policy' },
		{ value: 'mcts-visits', label: 'MCTS visits' },
		{ value: 'value', label: 'Value' },
		{ value: 'off', label: 'Off' }
	];

	// --- derived ----------------------------------------------------------

	let numPlayers = $derived(form.players.length);
	let totalCards = $derived(numPlayers * form.start_cards);
	let maxStartCards = $derived(Math.min(13, Math.floor(52 / numPlayers)));
	let cardsValid = $derived(totalCards <= 52 && form.start_cards >= 1);
	let humanIndex = $derived(form.players.findIndex((p) => p.is_human));
	let hasHuman = $derived(humanIndex >= 0);

	let canStart = $derived(cardsValid && hasHuman);

	// --- mount ------------------------------------------------------------

	onMount(async () => {
		const [settings, disc, rec, sessions] = await Promise.all([
			commands.loadAppSettings(),
			commands.listModels(),
			commands.listRecentModels(),
			commands.listSessions()
		]);
		if (sessions.status === 'ok') savedSessions = sessions.data;
		if (settings.status === 'ok') {
			form = sanitize(settings.data);
			// Regenerate stable IDs to match the loaded player count.
			playerIds = form.players.map(() => makeId());
			// Default dealer to last seat after loading.
			dealerSeat = form.players.length - 1;
			if (settings.data.engine_settings.deterministic_seed !== null) {
				useDeterministicSeed = true;
				seedInput = String(settings.data.engine_settings.deterministic_seed);
			}
			if (!MCTS_PRESETS.includes(settings.data.engine_settings.mcts_simulations)) {
				mctsCustom = true;
				mctsCustomValue = settings.data.engine_settings.mcts_simulations;
			}
		}
		if (disc.status === 'ok') discovered = disc.data;
		if (rec.status === 'ok') recents = rec.data;

		// Auto-restore last model if the file still exists.
		if (form.last_model_path) {
			const stillThere =
				rec.status === 'ok' && rec.data.some((m) => m.path === form.last_model_path);
			if (stillThere) {
				await pickModel(form.last_model_path);
			}
		}
	});

	// --- helpers ----------------------------------------------------------

	/** Backstop against malformed/legacy persisted data. */
	function sanitize(s: AppSettings): AppSettings {
		const players = s.players?.length >= MIN_PLAYERS ? [...s.players] : defaultPlayers();
		if (players.length > MAX_PLAYERS) players.length = MAX_PLAYERS;
		// Ensure exactly one human.
		const firstHuman = players.findIndex((p) => p.is_human);
		players.forEach((p, i) => (p.is_human = i === (firstHuman >= 0 ? firstHuman : 0)));
		return { ...s, players };
	}

	function fmtSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
		return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
	}

	function fmtModified(ts: number | null): string {
		if (ts === null) return '';
		return new Date(ts * 1000).toLocaleString();
	}

	function errMessage(e: GuiError): string {
		if ('message' in e && e.message) return `${e.kind}: ${e.message}`;
		return e.kind;
	}

	// --- player ops -------------------------------------------------------

	function addPlayer() {
		if (form.players.length >= MAX_PLAYERS) return;
		const idx = form.players.length;
		form.players = [...form.players, { name: `P${idx}`, is_human: false }];
		playerIds = [...playerIds, makeId()];
	}

	function removePlayer(i: number) {
		if (form.players.length <= MIN_PLAYERS) return;
		const next = form.players.filter((_, idx) => idx !== i);
		if (!next.some((p) => p.is_human)) next[0].is_human = true;
		form.players = next;
		playerIds = playerIds.filter((_, idx) => idx !== i);
		// Adjust dealer seat.
		if (dealerSeat === i) dealerSeat = next.length - 1;
		else if (dealerSeat > i) dealerSeat--;
		if (form.start_cards > Math.floor(52 / next.length)) {
			form.start_cards = Math.floor(52 / next.length);
		}
	}

	function moveUp(i: number) {
		if (i <= 0) return;
		const next = [...form.players];
		[next[i - 1], next[i]] = [next[i], next[i - 1]];
		form.players = next;
		const nextIds = [...playerIds];
		[nextIds[i - 1], nextIds[i]] = [nextIds[i], nextIds[i - 1]];
		playerIds = nextIds;
		if (dealerSeat === i) dealerSeat = i - 1;
		else if (dealerSeat === i - 1) dealerSeat = i;
	}

	function moveDown(i: number) {
		if (i >= form.players.length - 1) return;
		const next = [...form.players];
		[next[i], next[i + 1]] = [next[i + 1], next[i]];
		form.players = next;
		const nextIds = [...playerIds];
		[nextIds[i], nextIds[i + 1]] = [nextIds[i + 1], nextIds[i]];
		playerIds = nextIds;
		if (dealerSeat === i) dealerSeat = i + 1;
		else if (dealerSeat === i + 1) dealerSeat = i;
	}

	function setDealer(i: number) {
		dealerSeat = i;
	}

	function setName(i: number, value: string) {
		form.players[i].name = value;
	}

	// --- model picker -----------------------------------------------------

	async function pickModelFromDialog() {
		const selected = await openDialog({
			multiple: false,
			directory: false,
			filters: [{ name: 'ONNX model', extensions: ['onnx'] }]
		});
		if (typeof selected !== 'string') return;
		await pickModel(selected);
	}

	async function pickModel(path: string) {
		modelLoading = true;
		modelError = null;
		const result = await commands.loadModel(path);
		modelLoading = false;
		if (result.status === 'ok') {
			loadedModel = result.data;
			form.last_model_path = result.data.path;
			const rec = await commands.listRecentModels();
			if (rec.status === 'ok') recents = rec.data;
		} else {
			loadedModel = null;
			modelError = errMessage(result.error);
			pushToast(`Model load failed: ${modelError}`, 'error');
		}
	}

	function clearModel() {
		loadedModel = null;
		form.last_model_path = null;
	}

	function applyEngineSettingsFromUi(): EngineSettings {
		const seed = useDeterministicSeed ? Number(seedInput) || 0 : null;
		const mcts = mctsCustom ? mctsCustomValue : form.engine_settings.mcts_simulations;
		return { ...form.engine_settings, deterministic_seed: seed, mcts_simulations: mcts };
	}

	function chooseMcts(v: number) {
		mctsCustom = false;
		form.engine_settings.mcts_simulations = v;
	}

	// --- resume saved session --------------------------------------------

	function fmtSessionAge(ts: number | null): string {
		if (ts === null) return '';
		return new Date(ts * 1000).toLocaleString();
	}

	async function resumeSession(info: SavedSessionInfo) {
		resuming = true;
		resumeError = null;
		const result = await commands.loadSession(info.path);
		if (result.status !== 'ok') {
			resumeError = errMessage(result.error);
			resuming = false;
			return;
		}
		// Re-load the model the user had picked previously, if any. If the
		// file's gone we silently skip — the heuristic evaluator still works.
		if (form.last_model_path) {
			await commands.loadModel(form.last_model_path).catch(() => {});
		}
		// Push the persisted engine settings through to the live session so
		// AI calls match what the saved game was using. (load_session only
		// restores the snapshot; the live engine settings come from the
		// settings.json the user picked in /setup.)
		await commands.updateEngineSettings(form.engine_settings);
		sessionStore.set(result.data);
		const phase = result.data.phase;
		resuming = false;
		if (phase === 'complete') goto('/end');
		else if (phase === 'scoring') goto('/round-summary');
		else if (phase === 'playing' || result.data.human_hand.length > 0) goto('/play');
		else goto('/hand-entry');
	}

	async function deleteSavedSession(info: SavedSessionInfo, ev: MouseEvent) {
		ev.stopPropagation();
		const res = await commands.deleteSession(info.path);
		if (res.status === 'ok') {
			savedSessions = savedSessions.filter((s) => s.path !== info.path);
		}
	}

	// --- start ------------------------------------------------------------

	async function startGame() {
		startError = null;
		if (!canStart) return;
		starting = true;

		const engine = applyEngineSettingsFromUi();
		form.engine_settings = engine;

		await commands.saveAppSettings(form);

		const result = await commands.newGame({
			num_players: numPlayers,
			start_cards: form.start_cards,
			human_seat: humanIndex,
			dealer: dealerSeat,
			player_names: form.players.map((p) => p.name),
			trump_mode: form.trump_mode
		});
		if (result.status !== 'ok') {
			starting = false;
			startError = errMessage(result.error);
			return;
		}
		const applied = await commands.updateEngineSettings(engine);
		starting = false;
		if (applied.status !== 'ok') {
			startError = errMessage(applied.error);
			return;
		}
		sessionStore.set(result.data);
		goto('/hand-entry');
	}
</script>

<main class="mx-auto max-w-3xl px-6 py-10">
	<header class="mb-8">
		<h1 class="text-3xl font-semibold tracking-tight">BlobMaster</h1>
		<p class="mt-1 text-sm text-slate-500">
			Arrange your table, then start. The engine handles rules and scoring; the model (when
			loaded) provides AI suggestions.
		</p>
	</header>

	{#if savedSessions.length > 0}
		<section class="mb-8">
			<div class="mb-3 flex items-baseline justify-between">
				<h2 class="text-sm font-medium tracking-wide text-slate-700 uppercase">
					Resume game ({savedSessions.length})
				</h2>
				<span class="text-xs text-slate-400">Saved at every round end &amp; on quit.</span>
			</div>
			{#if resumeError}
				<p class="mb-3 rounded bg-red-50 px-3 py-2 text-sm text-red-700">{resumeError}</p>
			{/if}
			<ul class="divide-y divide-slate-200 rounded border border-slate-200 bg-white">
				{#each savedSessions as session (session.path)}
					<li class="flex items-center gap-3 px-3 py-2">
						<button
							type="button"
							onclick={() => resumeSession(session)}
							disabled={resuming}
							class="flex flex-1 items-center justify-between text-left disabled:opacity-50"
						>
							<div>
								<div class="text-sm font-medium text-slate-800">
									{session.num_players} players · C={session.start_cards} ·
									Round {session.round_idx + 1}/{session.total_rounds}
									{#if session.phase === 'complete'}
										<span class="ml-1 rounded bg-amber-600 px-1.5 py-0.5 text-[10px] font-medium text-white">
											finished
										</span>
									{:else if session.phase === 'scoring'}
										<span class="ml-1 rounded bg-emerald-600 px-1.5 py-0.5 text-[10px] font-medium text-white">
											scoring
										</span>
									{/if}
								</div>
								<div class="text-xs text-slate-500">
									{fmtSessionAge(session.saved_unix_secs)}
									{#if session.leader_name}
										· Leader: <strong>{session.leader_name}</strong> ({session.leader_score})
									{/if}
								</div>
							</div>
							<span class="text-xs font-medium text-emerald-700">Resume →</span>
						</button>
						<button
							type="button"
							onclick={(e) => deleteSavedSession(session, e)}
							aria-label="Delete saved session"
							class="rounded px-2 py-1 text-xs text-slate-400 hover:bg-red-50 hover:text-red-600"
						>✕</button>
					</li>
				{/each}
			</ul>
		</section>
	{/if}

	<!-- Players -->
	<section class="mb-8">
		<div class="mb-3 flex items-baseline justify-between">
			<h2 class="text-sm font-medium tracking-wide text-slate-700 uppercase">
				Players ({numPlayers})
			</h2>
			<span class="text-xs text-slate-400">Top to bottom = play order</span>
		</div>

		<ul class="divide-y divide-slate-200 rounded border border-slate-200 bg-white">
			{#each form.players as player, i (playerIds[i])}
				<li class="flex items-center gap-2 px-3 py-2" class:bg-slate-50={i === dealerSeat}>
					<!-- reorder: two large side-by-side buttons -->
					<div class="flex gap-1">
						<button
							type="button"
							onclick={() => moveUp(i)}
							disabled={i === 0}
							aria-label="Move up"
							class="rounded bg-slate-100 px-3 py-2 text-sm font-bold leading-none text-slate-600 hover:bg-slate-200 disabled:opacity-30"
						>▲</button>
						<button
							type="button"
							onclick={() => moveDown(i)}
							disabled={i === form.players.length - 1}
							aria-label="Move down"
							class="rounded bg-slate-100 px-3 py-2 text-sm font-bold leading-none text-slate-600 hover:bg-slate-200 disabled:opacity-30"
						>▼</button>
					</div>

					<!-- seat number -->
					<span class="w-6 tabular-nums text-xs text-slate-400">{i}</span>

					<!-- name -->
					<input
						type="text"
						value={player.name}
						oninput={(e) => setName(i, e.currentTarget.value)}
						class="min-w-0 flex-1 rounded border border-slate-300 px-2 py-1 text-sm focus:border-slate-500 focus:outline-none"
					/>

					<!-- human indicator (no button — the "You" player is fixed) -->
					{#if player.is_human}
						<span
							class="rounded border border-emerald-600 bg-emerald-600 px-2 py-1 text-xs font-medium text-white"
						>You</span>
					{/if}

					<!-- dealer toggle (clicking just marks this seat as dealer, no reorder) -->
					<button
						type="button"
						onclick={() => setDealer(i)}
						title={i === dealerSeat
							? 'Dealer (bids and deals last)'
							: 'Mark as dealer'}
						aria-label="Mark as dealer"
						aria-pressed={i === dealerSeat}
						class="rounded border px-2 py-1 text-xs font-medium"
						class:border-amber-600={i === dealerSeat}
						class:bg-amber-600={i === dealerSeat}
						class:text-white={i === dealerSeat}
						class:border-slate-200={i !== dealerSeat}
						class:text-slate-400={i !== dealerSeat}
						class:hover:border-slate-400={i !== dealerSeat}
						class:hover:text-slate-600={i !== dealerSeat}
					>D</button>

					<!-- remove -->
					<button
						type="button"
						onclick={() => removePlayer(i)}
						disabled={form.players.length <= MIN_PLAYERS}
						aria-label="Remove player"
						class="rounded px-2 py-1 text-xs text-slate-400 hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-slate-400"
					>✕</button>
				</li>
			{/each}
		</ul>

		<div class="mt-2 flex items-center justify-between">
			<button
				type="button"
				onclick={addPlayer}
				disabled={form.players.length >= MAX_PLAYERS}
				class="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
			>+ Add player</button>
			<span class="text-xs text-slate-400">
				{MIN_PLAYERS}–{MAX_PLAYERS} players · seat 0 plays first · seat {dealerSeat} is dealer
			</span>
		</div>
		{#if !hasHuman}
			<p class="mt-2 text-xs text-red-600">No human seat found — check your settings.</p>
		{/if}
	</section>

	<!-- Rounds -->
	<section class="mb-8">
		<h2 class="mb-3 text-sm font-medium tracking-wide text-slate-700 uppercase">Rounds</h2>

		<label class="block">
			<span class="mb-1 flex items-center justify-between text-xs text-slate-500">
				<span>Starting cards (C)</span>
				<span class="tabular-nums text-slate-700">
					{form.start_cards} · {numPlayers}×{form.start_cards} = {totalCards} cards
				</span>
			</span>
			<input
				type="range"
				min="1"
				max="13"
				bind:value={form.start_cards}
				class="w-full"
			/>
			{#if !cardsValid}
				<p class="mt-1 text-xs text-red-600">
					num_players × C must be ≤ 52 (max C for {numPlayers} players is {maxStartCards}).
				</p>
			{/if}
		</label>

		<label class="mt-4 block">
			<span class="mb-1 block text-xs text-slate-500">Trump policy</span>
			<select
				bind:value={form.trump_mode}
				class="w-full rounded border border-slate-300 px-3 py-2 text-sm"
			>
				{#each TRUMP_MODES as mode (mode.value)}
					<option value={mode.value}>{mode.label}</option>
				{/each}
			</select>
			{#if form.trump_mode !== 'auto-rotate'}
				<p class="mt-1 text-xs text-amber-700">
					Engine override for fixed trump lands later — for now the game will still rotate.
				</p>
			{/if}
		</label>
	</section>

	<!-- Model picker -->
	<section class="mb-8">
		<div class="mb-3 flex items-baseline justify-between">
			<h2 class="text-sm font-medium tracking-wide text-slate-700 uppercase">Model</h2>
			<span class="text-xs text-slate-400">Optional — without one, AI suggestions stay off.</span>
		</div>

		<div class="mb-3 flex flex-wrap items-center gap-3">
			<button
				type="button"
				onclick={pickModelFromDialog}
				disabled={modelLoading}
				class="rounded bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700 disabled:opacity-50"
			>
				{modelLoading ? 'Loading…' : 'Pick .onnx file…'}
			</button>
			{#if loadedModel}
				<span class="text-sm text-emerald-700">✓ {loadedModel.file_name}</span>
				<button
					type="button"
					onclick={clearModel}
					class="text-xs text-slate-500 underline hover:text-slate-700"
				>Clear</button>
			{:else}
				<span class="text-sm text-slate-500">No model loaded.</span>
			{/if}
		</div>

		{#if modelError}
			<p class="mb-3 rounded bg-red-50 px-3 py-2 text-sm text-red-700">{modelError}</p>
		{/if}

		{#if loadedModel}
			<div class="mb-4 rounded border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs text-slate-700">
				<div class="font-medium text-slate-900">{loadedModel.file_name}</div>
				<div class="mt-1 font-mono text-[11px] break-all text-slate-500">{loadedModel.path}</div>
				<div class="mt-2 flex flex-wrap gap-4 tabular-nums">
					<span>{fmtSize(loadedModel.size_bytes)}</span>
					{#if loadedModel.modified_unix_secs !== null}
						<span>Modified {fmtModified(loadedModel.modified_unix_secs)}</span>
					{/if}
					{#if loadedModel.d_model !== null}
						<span>d_model={loadedModel.d_model}</span>
					{/if}
					{#if loadedModel.n_layers !== null}
						<span>n_layers={loadedModel.n_layers}</span>
					{/if}
				</div>
			</div>
		{/if}

		{#if recents.length > 0}
			<div class="mb-4">
				<h3 class="mb-2 text-xs font-medium tracking-wide text-slate-500 uppercase">Recents</h3>
				<ul class="divide-y divide-slate-200 rounded border border-slate-200 bg-white">
					{#each recents as m (m.path)}
						<li>
							<button
								type="button"
								onclick={() => pickModel(m.path)}
								class="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-slate-50"
								class:bg-emerald-50={loadedModel?.path === m.path}
							>
								<span class="font-mono text-xs text-slate-700">{m.file_name}</span>
								<span class="tabular-nums text-xs text-slate-400">{fmtSize(m.size_bytes)}</span>
							</button>
						</li>
					{/each}
				</ul>
			</div>
		{/if}

		{#if discovered.length > 0}
			<div>
				<h3 class="mb-2 text-xs font-medium tracking-wide text-slate-500 uppercase">
					In <code>checkpoints/</code>
				</h3>
				<ul class="divide-y divide-slate-200 rounded border border-slate-200 bg-white">
					{#each discovered as m (m.path)}
						<li>
							<button
								type="button"
								onclick={() => pickModel(m.path)}
								class="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-slate-50"
								class:bg-emerald-50={loadedModel?.path === m.path}
							>
								<span class="font-mono text-xs text-slate-700">{m.file_name}</span>
								<span class="tabular-nums text-xs text-slate-400">{fmtSize(m.size_bytes)}</span>
							</button>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	</section>

	<!-- Advanced -->
	<section class="mb-8">
		<button
			type="button"
			onclick={() => (advancedOpen = !advancedOpen)}
			class="flex w-full items-center justify-between border-b border-slate-200 pb-2 text-left text-sm font-medium tracking-wide text-slate-700 uppercase hover:text-slate-900"
		>
			<span>Advanced — engine settings</span>
			<span class="text-xs text-slate-400">{advancedOpen ? '▴' : '▾'}</span>
		</button>

		{#if advancedOpen}
			<div class="mt-4 space-y-5">
				<label class="block">
					<span class="mb-1 flex items-center justify-between text-xs text-slate-500">
						<span>Temperature</span>
						<span class="tabular-nums text-slate-700">{form.engine_settings.temperature.toFixed(2)}</span>
					</span>
					<input
						type="range"
						min="0"
						max="2"
						step="0.05"
						bind:value={form.engine_settings.temperature}
						class="w-full"
					/>
				</label>

				<div>
					<span class="mb-1 block text-xs text-slate-500">MCTS simulations</span>
					<div class="flex flex-wrap items-center gap-2">
						{#each MCTS_PRESETS as v (v)}
							<button
								type="button"
								onclick={() => chooseMcts(v)}
								class="rounded border px-3 py-1.5 text-sm tabular-nums"
								class:border-slate-900={!mctsCustom && form.engine_settings.mcts_simulations === v}
								class:bg-slate-900={!mctsCustom && form.engine_settings.mcts_simulations === v}
								class:text-white={!mctsCustom && form.engine_settings.mcts_simulations === v}
								class:border-slate-300={mctsCustom || form.engine_settings.mcts_simulations !== v}
								class:hover:bg-slate-100={mctsCustom || form.engine_settings.mcts_simulations !== v}
							>
								{v === 0 ? 'Pure policy' : v}
							</button>
						{/each}
						<label class="ml-2 flex items-center gap-2 text-sm text-slate-600">
							<input type="checkbox" bind:checked={mctsCustom} />
							Custom
							{#if mctsCustom}
								<input
									type="number"
									min="0"
									max="100000"
									bind:value={mctsCustomValue}
									class="w-24 rounded border border-slate-300 px-2 py-1 text-sm tabular-nums"
								/>
							{/if}
						</label>
					</div>
				</div>

				<label class="block">
					<span class="mb-1 flex items-center justify-between text-xs text-slate-500">
						<span>Determinization samples</span>
						<span class="tabular-nums text-slate-700">{form.engine_settings.determinization_samples}</span>
					</span>
					<input
						type="range"
						min="1"
						max="32"
						bind:value={form.engine_settings.determinization_samples}
						class="w-full"
					/>
				</label>

				<div>
					<label class="mb-1 flex items-center gap-2 text-sm text-slate-600">
						<input type="checkbox" bind:checked={useDeterministicSeed} />
						Use deterministic RNG seed
					</label>
					{#if useDeterministicSeed}
						<input
							type="number"
							min="0"
							bind:value={seedInput}
							class="w-48 rounded border border-slate-300 px-2 py-1 text-sm tabular-nums"
						/>
					{/if}
				</div>

				<label class="block">
					<span class="mb-1 block text-xs text-slate-500">Eval display (default)</span>
					<select
						bind:value={form.engine_settings.eval_display}
						class="w-full rounded border border-slate-300 px-3 py-2 text-sm"
					>
						{#each EVAL_DISPLAY_OPTIONS as opt (opt.value)}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				</label>
			</div>
		{/if}
	</section>

	<!-- Start -->
	<section class="border-t border-slate-200 pt-6">
		{#if startError}
			<p class="mb-3 rounded bg-red-50 px-3 py-2 text-sm text-red-700">{startError}</p>
		{/if}
		<div class="flex items-center justify-between">
			<p class="text-xs text-slate-500">
				{#if !hasHuman}
					No human seat configured.
				{:else if !cardsValid}
					Adjust starting cards.
				{:else if !loadedModel}
					Starting without a model — AI suggestions will be disabled.
				{:else}
					Settings persist to <code>~/.blobmaster/settings.json</code>.
				{/if}
			</p>
			<button
				type="button"
				onclick={startGame}
				disabled={!canStart || starting}
				class="rounded bg-emerald-700 px-6 py-2.5 text-sm font-medium text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{starting ? 'Starting…' : 'Start Game'}
			</button>
		</div>
	</section>
</main>
