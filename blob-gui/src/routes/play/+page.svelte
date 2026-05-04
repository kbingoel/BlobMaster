<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { type SessionSnapshot } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import CardGrid from '$lib/components/CardGrid.svelte';

	let snapshot = $state<SessionSnapshot | null>(null);

	onMount(() => {
		const unsub = sessionStore.subscribe((s) => {
			snapshot = s;
		});
		if (snapshot === null) goto('/setup', { replaceState: true });
		return unsub;
	});
</script>

{#if snapshot}
	<div class="play-layout">
		<!-- ── Left column: two placeholder panes ─────────────── -->
		<div class="left-col">
			<div class="top-left pane-placeholder">
				<p class="placeholder-label">Players panel</p>
				<p class="placeholder-sub">Session 9.5</p>
			</div>
			<div class="bottom-left pane-placeholder">
				<p class="placeholder-label">Hand panel</p>
				<p class="placeholder-sub">Session 9.6</p>
			</div>
		</div>

		<!-- ── Right column: master CardGrid ─────────────────── -->
		<div class="right-col">
			<CardGrid {snapshot} mode="play" />
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
		border-bottom: 1px solid #e2e8f0;
	}

	.bottom-left {
		flex: 1;
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
		color: #94a3b8; /* slate-400 */
		margin: 0;
	}

	.placeholder-sub {
		font-size: 0.75rem;
		color: #cbd5e1; /* slate-300 */
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
