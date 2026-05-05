<script lang="ts">
	import { commands, type RoundStructureEntry } from '$lib/api';
	import { trumpLabel, NO_TRUMP, isRed } from '$lib/cardUtils';

	type Props = {
		currentRound: number;
		// Re-key so we refetch when start_cards / num_players change (i.e. on a new game).
		gameKey: string;
	};
	let { currentRound, gameKey }: Props = $props();

	let structure = $state<RoundStructureEntry[]>([]);

	$effect(() => {
		// Read gameKey to register the dependency.
		void gameKey;
		commands.roundStructure().then((res) => {
			if (res.status === 'ok') structure = res.data;
		});
	});
</script>

<div class="strip">
	{#each structure as entry (entry.round_idx)}
		<div
			class="cell"
			class:current={entry.round_idx === currentRound}
			class:past={entry.round_idx < currentRound}
			class:trump-red={isRed(entry.trump_suit)}
			class:trump-nt={entry.trump_suit === NO_TRUMP}
			title={`Round ${entry.round_idx + 1}: ${entry.cards_dealt} cards · trump ${trumpLabel(entry.trump_suit)}`}
		>
			<span class="cards tabular-nums">{entry.cards_dealt}</span>
			<span class="trump">{trumpLabel(entry.trump_suit)}</span>
		</div>
	{/each}
</div>

<style>
	.strip {
		display: flex;
		flex-wrap: nowrap;
		gap: 2px;
		padding: 4px 8px;
		background: #0f172a; /* slate-900 */
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
</style>
