<script lang="ts">
	import { KEYMAP } from '$lib/keymap';

	let open = $state(false);

	function handleKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		if (e.key === '?') {
			e.preventDefault();
			open = !open;
			return;
		}
		if (open && e.key === 'Escape') {
			e.preventDefault();
			open = false;
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
	<div class="overlay" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
		<button
			type="button"
			class="backdrop"
			aria-label="Close keyboard help"
			onclick={() => (open = false)}
		></button>
		<div class="panel">
			<header class="panel-header">
				<h2 class="title">Keyboard shortcuts</h2>
				<button type="button" class="close-btn" onclick={() => (open = false)} aria-label="Close">✕</button>
			</header>
			<div class="sections">
				{#each KEYMAP as section}
					<section>
						<h3 class="section-title">{section.title}</h3>
						<ul class="binding-list">
							{#each section.bindings as binding}
								<li>
									<span class="keys">
										{#each binding.keys as k}
											<kbd>{k}</kbd>
										{/each}
									</span>
									<span class="desc">{binding.description}</span>
								</li>
							{/each}
						</ul>
					</section>
				{/each}
			</div>
			<footer class="hint">
				Press <kbd>?</kbd> or <kbd>Esc</kbd> to close.
			</footer>
		</div>
	</div>
{/if}

<style>
	.overlay {
		position: fixed;
		inset: 0;
		z-index: 1100;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.backdrop {
		position: absolute;
		inset: 0;
		background: rgba(15, 23, 42, 0.55);
		border: 0;
		cursor: pointer;
	}
	.panel {
		position: relative;
		background: #f8fafc;
		max-width: 720px;
		max-height: 86vh;
		width: 92vw;
		border-radius: 8px;
		box-shadow: 0 24px 48px -12px rgba(15, 23, 42, 0.6);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 12px 18px;
		border-bottom: 1px solid #e2e8f0;
	}
	.title {
		font-size: 1rem;
		margin: 0;
		font-weight: 600;
		color: #0f172a;
	}
	.close-btn {
		background: transparent;
		border: 0;
		font-size: 1rem;
		color: #64748b;
		cursor: pointer;
		padding: 4px 8px;
	}
	.close-btn:hover {
		color: #0f172a;
	}
	.sections {
		padding: 12px 18px;
		overflow-y: auto;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 18px 28px;
	}
	.section-title {
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: #64748b;
		margin: 0 0 6px;
	}
	.binding-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.binding-list li {
		display: flex;
		align-items: baseline;
		gap: 10px;
		font-size: 0.78rem;
		color: #1e293b;
	}
	.keys {
		flex: 0 0 auto;
		display: inline-flex;
		gap: 3px;
		align-items: center;
	}
	.desc {
		flex: 1;
		color: #475569;
	}
	kbd {
		font-family: ui-monospace, 'SFMono-Regular', monospace;
		font-size: 0.7rem;
		padding: 1px 6px;
		background: #fff;
		border: 1px solid #cbd5e1;
		border-bottom-width: 2px;
		border-radius: 3px;
		color: #0f172a;
	}
	.hint {
		padding: 10px 18px;
		border-top: 1px solid #e2e8f0;
		font-size: 0.72rem;
		color: #64748b;
	}
</style>
