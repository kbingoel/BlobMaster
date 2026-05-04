<script lang="ts">
	import type { AiSuggestion, CardEval, EngineSettings, EvalDisplay, SessionSnapshot } from '$lib/api';
	import { isRed, rankLabel, suitGlyph, cardSuit, cardRank } from '$lib/cardUtils';
	import {
		cycleEvalDisplay,
		evalDisplayLabel,
		primaryMetric,
		secondaryMetric,
		winRateTint
	} from '$lib/evalUtils';

	interface Props {
		snapshot: SessionSnapshot;
		suggestion?: AiSuggestion | null;
		settings: EngineSettings;
		submitting?: boolean;
		errorMessage?: string | null;
		onPlay: (card: number) => void;
		onSettingsChange: (settings: EngineSettings) => void;
	}

	let {
		snapshot,
		suggestion = null,
		settings,
		submitting = false,
		errorMessage = null,
		onPlay,
		onSettingsChange
	}: Props = $props();

	let isHumanTurn = $derived(snapshot.current_player === snapshot.human_seat);
	let humanSeat = $derived(snapshot.human_seat);

	let handCards = $derived([...snapshot.human_hand].sort((a, b) => a - b));

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

	let playSuggestion = $derived(
		suggestion && suggestion.phase === 'playing' ? suggestion : null
	);
	let recommendedCard = $derived(playSuggestion?.recommended_card ?? null);
	let valueEstimate = $derived(playSuggestion?.value_estimate ?? null);
	let simsCompleted = $derived(playSuggestion?.sims_completed ?? 0);
	let depth = $derived(playSuggestion?.depth ?? 0);

	let perCardEvals = $derived<CardEval[]>(playSuggestion?.per_card ?? []);
	let evalByCard = $derived(new Map(perCardEvals.map((e) => [e.card, e])));

	let myRoundScore = $derived(snapshot.tricks_won[humanSeat] ?? 0);
	let myCumulativeScore = $derived(snapshot.cumulative_scores[humanSeat] ?? 0);
	let myBid = $derived(snapshot.bids[humanSeat] ?? null);
	let roundDelta = $derived.by(() => {
		if (myBid === null) return null;
		return myRoundScore === myBid ? 10 + myBid : 0;
	});

	let evalMode = $derived<EvalDisplay>(settings.eval_display);
	let evalOff = $derived(evalMode === 'off');
	let showFooter = $state(false);

	function isLegal(card: number): boolean {
		if (!isHumanTurn) return false;
		return snapshot.legal_plays?.includes(card) ?? false;
	}

	function handleClick(card: number) {
		if (submitting) return;
		if (!isLegal(card)) return;
		onPlay(card);
	}

	function cycleDisplay() {
		onSettingsChange({ ...settings, eval_display: cycleEvalDisplay(settings.eval_display) });
	}

	function updateField<K extends keyof EngineSettings>(key: K, value: EngineSettings[K]) {
		onSettingsChange({ ...settings, [key]: value });
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
			return;
		}
		if (e.key === 'e' || e.key === 'E') {
			e.preventDefault();
			cycleDisplay();
		}
	}

	function statusValue(): string {
		if (valueEstimate === null) return '—';
		const sign = valueEstimate >= 0 ? '+' : '';
		return sign + valueEstimate.toFixed(2);
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="hand-panel">
	<header class="status">
		<span class="status-cell">MCTS: <strong>{simsCompleted}</strong></span>
		<span class="status-cell">depth <strong>{depth}</strong></span>
		<span class="status-cell">v=<strong>{statusValue()}</strong></span>
		<span class="status-cell"
			>round <strong>{myRoundScore}</strong>{myBid !== null ? ` / ${myBid}` : ''}{roundDelta !==
			null
				? ` (${roundDelta >= 0 ? '+' : ''}${roundDelta} round)`
				: ''}</span
		>
		<span class="status-cell">total <strong>{myCumulativeScore}</strong></span>
		<span class="status-cell mode-cell" title="Press E to cycle">
			eval: <strong>{evalDisplayLabel(evalMode)}</strong>
		</span>
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
			{@const entry = evalByCard.get(card)}
			{@const tint = entry && legal && !evalOff ? winRateTint(entry, perCardEvals) : 'transparent'}
			<button
				type="button"
				class="card"
				class:suit-red={isRed(suit)}
				class:legal
				class:illegal={isHumanTurn && !legal}
				class:recommended
				disabled={!legal || submitting}
				style:background={tint}
				onclick={() => handleClick(card)}
			>
				<span class="card-rank">{rankLabel(rank)}</span>
				<span class="card-suit">{suitGlyph(suit)}</span>
				<span class="card-eval">
					{#if entry && legal && !evalOff}
						<span class="eval-line">{primaryMetric(entry, evalMode)}</span>
						<span class="eval-line eval-line-2">{secondaryMetric(entry, evalMode)}</span>
					{:else}
						<span class="eval-line">—</span>
						<span class="eval-line eval-line-2">—</span>
					{/if}
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
			Waiting for P{snapshot.current_player} to play — click a card on the right grid to record their move.
		</p>
	{:else if recommendedCard !== null}
		<p class="hint">
			Press <kbd>Enter</kbd> to play the recommended card · <kbd>E</kbd> to cycle eval mode.
		</p>
	{/if}

	{#if errorMessage}
		<p class="error">{errorMessage}</p>
	{/if}

	<div class="settings-footer">
		<button type="button" class="footer-toggle" onclick={() => (showFooter = !showFooter)}>
			{showFooter ? '▾' : '▸'} engine settings
		</button>
		{#if showFooter}
			<div class="settings-grid">
				<label class="setting">
					<span>temperature</span>
					<input
						type="number"
						min="0"
						max="2"
						step="0.05"
						value={settings.temperature}
						oninput={(e) =>
							updateField('temperature', parseFloat((e.currentTarget as HTMLInputElement).value))}
					/>
				</label>
				<label class="setting">
					<span>MCTS sims</span>
					<input
						type="number"
						min="0"
						max="4000"
						step="50"
						value={settings.mcts_simulations}
						oninput={(e) =>
							updateField(
								'mcts_simulations',
								parseInt((e.currentTarget as HTMLInputElement).value, 10) || 0
							)}
					/>
				</label>
				<label class="setting">
					<span>determinizations</span>
					<input
						type="number"
						min="1"
						max="32"
						step="1"
						value={settings.determinization_samples}
						oninput={(e) =>
							updateField(
								'determinization_samples',
								parseInt((e.currentTarget as HTMLInputElement).value, 10) || 1
							)}
					/>
				</label>
				<label class="setting">
					<span>eval mode</span>
					<select
						value={settings.eval_display}
						onchange={(e) =>
							updateField(
								'eval_display',
								(e.currentTarget as HTMLSelectElement).value as EvalDisplay
							)}
					>
						<option value="win-rate">win-rate</option>
						<option value="policy">policy</option>
						<option value="mcts-visits">visits</option>
						<option value="value">value</option>
						<option value="off">off</option>
					</select>
				</label>
				<label class="setting checkbox">
					<input
						type="checkbox"
						checked={settings.show_grid_eval}
						onchange={(e) =>
							updateField('show_grid_eval', (e.currentTarget as HTMLInputElement).checked)}
					/>
					<span>show eval on master grid</span>
				</label>
			</div>
		{/if}
	</div>
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

	.mode-cell {
		margin-left: auto;
		color: #166534;
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
		min-height: 4.8rem;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: flex-start;
		gap: 0.05rem;
		padding: 0.35rem 0.2rem 0.25rem;
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
		align-items: center;
		gap: 0.05rem;
		font-size: 0.6rem;
		color: #1e293b;
		font-weight: 600;
	}

	.eval-line {
		line-height: 1;
	}

	.eval-line-2 {
		font-weight: 500;
		color: #475569;
		font-size: 0.55rem;
	}

	.card:hover:not(:disabled) {
		filter: brightness(0.96);
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

	.settings-footer {
		margin-top: auto;
		padding-top: 0.4rem;
		border-top: 1px solid #e2e8f0;
		font-size: 0.72rem;
	}

	.footer-toggle {
		background: none;
		border: 0;
		cursor: pointer;
		color: #475569;
		font-size: 0.72rem;
		padding: 0.1rem 0.2rem;
	}

	.footer-toggle:hover {
		color: #0f172a;
	}

	.settings-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr));
		gap: 0.4rem 0.7rem;
		padding: 0.4rem 0.2rem;
	}

	.setting {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		font-size: 0.7rem;
		color: #475569;
	}

	.setting input[type='number'],
	.setting select {
		font-size: 0.78rem;
		padding: 0.2rem 0.3rem;
		border: 1px solid #cbd5e1;
		border-radius: 3px;
		background: #fff;
	}

	.setting.checkbox {
		flex-direction: row;
		align-items: center;
		gap: 0.4rem;
	}
</style>
