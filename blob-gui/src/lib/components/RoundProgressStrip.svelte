<script lang="ts">
	import { commands, type RoundStructureEntry, type SessionSnapshot } from '$lib/api';
	import { trumpLabel, NO_TRUMP, isRed } from '$lib/cardUtils';
	import { trumpEditingStore } from '$lib/stores/trumpEditing';

	type Props = {
		currentRound: number;
		// Re-key so we refetch when start_cards / num_players change (i.e. on a new game).
		gameKey: string;
		/**
		 * `true` to allow toggling into trump-edit mode. Disabled on screens
		 * where editing isn't applicable (e.g. /end). Default `true`.
		 */
		editable?: boolean;
		/**
		 * Fires after a successful save with the fresh `SessionSnapshot` so
		 * the parent route can update its `sessionStore`. The snapshot's
		 * `trump_suit` reflects the current-round override if one was set.
		 */
		onTrumpsSaved?: (snapshot: SessionSnapshot) => void;
	};
	let { currentRound, gameKey, editable = true, onTrumpsSaved = () => {} }: Props = $props();

	let structure = $state<RoundStructureEntry[]>([]);

	// --- edit mode --------------------------------------------------------
	// Press 1-5 to set the trump for the cursor cell and step forward; click
	// a present/future cell to move the cursor; Save commits via
	// `set_trump_overrides`, Cancel discards. Past rounds stay locked.
	let editing = $state(false);
	let cursor = $state(0);
	let draft = $state<number[]>([]);
	let saving = $state(false);
	let saveError = $state<string | null>(null);

	const TRUMP_KEYS: Record<string, number> = {
		'1': 0, // ♠
		'2': 1, // ♥
		'3': 2, // ♣
		'4': 3, // ♦
		'5': NO_TRUMP
	};

	async function reload() {
		const res = await commands.roundStructure();
		if (res.status === 'ok') structure = res.data;
	}

	$effect(() => {
		// Read gameKey to register the dependency.
		void gameKey;
		reload();
	});

	function startEdit() {
		if (!editable || structure.length === 0) return;
		// Snapshot the currently-effective trumps as the starting draft so the
		// user can tweak relative to what's already in place.
		draft = structure.map((e) => e.trump_suit);
		// Place the cursor on the present round (first editable cell).
		cursor = Math.max(currentRound, 0);
		saveError = null;
		editing = true;
		trumpEditingStore.set(true);
	}

	function cancelEdit() {
		editing = false;
		draft = [];
		saveError = null;
		trumpEditingStore.set(false);
	}

	async function saveEdit() {
		// Only ship rounds that actually differ from the engine default *or*
		// that already had an override (so the user can clear by setting back
		// to the rotation). The backend rejects past-round entries, so filter.
		const overrides: { round_idx: number; trump: number }[] = [];
		for (let i = currentRound; i < structure.length; i++) {
			const draftValue = draft[i];
			if (draftValue === undefined) continue;
			overrides.push({ round_idx: i, trump: draftValue });
		}
		saving = true;
		saveError = null;
		const res = await commands.setTrumpOverrides(overrides);
		saving = false;
		if (res.status === 'ok') {
			editing = false;
			draft = [];
			trumpEditingStore.set(false);
			await reload();
			onTrumpsSaved(res.data);
		} else {
			saveError = 'message' in res.error ? res.error.message : res.error.kind;
		}
	}

	function clickCell(roundIdx: number) {
		if (!editing) return;
		if (roundIdx < currentRound) return; // locked
		cursor = roundIdx;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!editing) return;
		// Don't steal keystrokes while a form input has focus.
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

		if (e.key === 'Escape') {
			e.preventDefault();
			cancelEdit();
			return;
		}
		if (e.key === 'Enter') {
			e.preventDefault();
			saveEdit();
			return;
		}
		if (e.key === 'ArrowLeft') {
			e.preventDefault();
			cursor = Math.max(cursor - 1, currentRound);
			return;
		}
		if (e.key === 'ArrowRight') {
			e.preventDefault();
			cursor = Math.min(cursor + 1, structure.length - 1);
			return;
		}

		const trump = TRUMP_KEYS[e.key];
		if (trump !== undefined) {
			e.preventDefault();
			if (cursor < currentRound || cursor >= structure.length) return;
			const next = [...draft];
			next[cursor] = trump;
			draft = next;
			// Auto-advance to the next editable cell so the user can rip
			// through 1-5 presses.
			cursor = Math.min(cursor + 1, structure.length - 1);
		}
	}

	// What each cell should display: in edit mode the draft trump, otherwise
	// the persisted effective trump from the backend.
	function cellTrump(idx: number, fallback: number): number {
		if (editing && draft[idx] !== undefined) return draft[idx];
		return fallback;
	}

	let canEdit = $derived(editable && structure.length > 0);
	let dirty = $derived(
		editing &&
			draft.some((t, i) => i >= currentRound && t !== structure[i]?.trump_suit)
	);
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="strip-wrapper">
	<div class="strip" role="list">
		{#each structure as entry, i (entry.round_idx)}
			{@const trump = cellTrump(i, entry.trump_suit)}
			{@const past = entry.round_idx < currentRound}
			{@const draftChanged = editing && draft[i] !== entry.trump_suit}
			<button
				type="button"
				class="cell"
				class:current={entry.round_idx === currentRound}
				class:past
				class:trump-red={isRed(trump)}
				class:trump-nt={trump === NO_TRUMP}
				class:overridden={!editing && entry.trump_overridden}
				class:editing-cursor={editing && cursor === i}
				class:editing-draft={draftChanged}
				class:editing-locked={editing && past}
				disabled={!editing || past}
				title={editing
					? past
						? `Round ${entry.round_idx + 1}: locked (already played)`
						: `Round ${entry.round_idx + 1}: trump ${trumpLabel(trump)} — press 1-5 to change`
					: `Round ${entry.round_idx + 1}: ${entry.cards_dealt} cards · trump ${trumpLabel(trump)}${entry.trump_overridden ? ' (manual)' : ''}`}
				onclick={() => clickCell(i)}
			>
				<span class="cards tabular-nums">{entry.cards_dealt}</span>
				<span class="trump">{trumpLabel(trump)}</span>
				{#if !editing && entry.trump_overridden}
					<span class="override-dot" aria-hidden="true">●</span>
				{/if}
			</button>
		{/each}
	</div>

	{#if canEdit}
		<div class="controls">
			{#if editing}
				<span class="hint">
					<kbd>1</kbd>♠ <kbd>2</kbd>♥ <kbd>3</kbd>♣ <kbd>4</kbd>♦ <kbd>5</kbd>NT
					· <kbd>Enter</kbd> save · <kbd>Esc</kbd> cancel
				</span>
				<button type="button" class="btn-secondary" onclick={cancelEdit} disabled={saving}>
					Cancel
				</button>
				<button
					type="button"
					class="btn-primary"
					onclick={saveEdit}
					disabled={saving || !dirty}
				>{saving ? 'Saving…' : 'Save'}</button>
			{:else}
				<button type="button" class="btn-edit" onclick={startEdit}>Edit trumps</button>
			{/if}
		</div>
	{/if}
	{#if saveError}
		<p class="save-error">{saveError}</p>
	{/if}
</div>

<style>
	.strip-wrapper {
		display: flex;
		flex-direction: column;
		background: #0f172a; /* slate-900 */
	}
	.strip {
		display: flex;
		flex-wrap: nowrap;
		gap: 2px;
		padding: 4px 8px;
		overflow-x: auto;
	}
	.cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-width: 32px;
		padding: 2px 4px;
		border: 1px solid #334155; /* slate-700 */
		border-radius: 3px;
		background: #1e293b; /* slate-800 */
		color: #cbd5e1; /* slate-300 */
		font-size: 0.7rem;
		line-height: 1.05;
		flex-shrink: 0;
		position: relative;
		cursor: default;
		font-family: inherit;
	}
	.cell.past {
		opacity: 0.45;
	}
	.cell.current {
		background: #15803d; /* green-700 */
		border-color: #22c55e; /* green-500 */
		color: #f1f5f9;
		font-weight: 600;
	}
	.cards {
		font-size: 0.78rem;
		font-weight: 600;
	}
	.trump {
		font-size: 0.85rem;
	}
	.trump-red .trump {
		color: #fca5a5; /* red-300 */
	}
	.trump-nt .trump {
		font-size: 0.65rem;
		font-weight: 700;
		color: #fde68a; /* amber-200 */
	}
	.cell.overridden {
		border-color: #fbbf24; /* amber-400 */
	}
	.override-dot {
		position: absolute;
		top: 1px;
		right: 3px;
		font-size: 0.5rem;
		color: #fbbf24;
		line-height: 1;
	}

	/* ── Edit mode ─────────────────────────────────────────────── */
	.cell.editing-cursor {
		outline: 2px solid #fde68a;
		outline-offset: 1px;
		z-index: 1;
	}
	.cell.editing-draft {
		background: #7c2d12; /* amber-900-ish for draft */
		border-color: #f59e0b;
		color: #fef3c7;
	}
	.cell.editing-locked {
		cursor: not-allowed;
		opacity: 0.3;
	}
	.cell:not(:disabled):hover {
		background: #334155;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 4px 8px;
		border-top: 1px solid #1e293b;
		background: #0f172a;
	}
	.hint {
		font-size: 0.7rem;
		color: #94a3b8;
		margin-right: auto;
	}
	.hint kbd {
		font-family: ui-monospace, 'SFMono-Regular', monospace;
		font-size: 0.65rem;
		padding: 0 0.25rem;
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 3px;
		color: #f1f5f9;
		margin-left: 0.15rem;
	}
	.btn-edit {
		margin-left: auto;
		font-size: 0.7rem;
		padding: 2px 8px;
		background: transparent;
		border: 1px solid #334155;
		border-radius: 3px;
		color: #94a3b8;
		cursor: pointer;
	}
	.btn-edit:hover {
		background: #1e293b;
		color: #f1f5f9;
	}
	.btn-secondary {
		font-size: 0.7rem;
		padding: 2px 10px;
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 3px;
		color: #cbd5e1;
		cursor: pointer;
	}
	.btn-secondary:hover:not(:disabled) {
		background: #334155;
	}
	.btn-primary {
		font-size: 0.7rem;
		padding: 2px 10px;
		background: #15803d;
		border: 1px solid #22c55e;
		border-radius: 3px;
		color: #f1f5f9;
		font-weight: 600;
		cursor: pointer;
	}
	.btn-primary:hover:not(:disabled) {
		background: #166534;
	}
	.btn-primary:disabled,
	.btn-secondary:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.save-error {
		margin: 0;
		padding: 4px 8px;
		font-size: 0.7rem;
		color: #fca5a5;
		background: #450a0a;
	}
</style>
