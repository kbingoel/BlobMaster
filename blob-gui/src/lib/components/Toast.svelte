<script lang="ts">
	import { dismissToast, toastStore } from '$lib/stores/toast';
</script>

<div class="toast-stack" aria-live="polite" aria-atomic="false">
	{#each $toastStore as t (t.id)}
		<button
			type="button"
			class="toast toast-{t.kind}"
			onclick={() => dismissToast(t.id)}
			aria-label="Dismiss notification"
		>{t.message}</button>
	{/each}
</div>

<style>
	.toast-stack {
		position: fixed;
		bottom: 16px;
		right: 16px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		z-index: 1000;
		pointer-events: none;
	}
	.toast {
		pointer-events: auto;
		max-width: 380px;
		padding: 10px 14px;
		border-radius: 6px;
		font-size: 0.8rem;
		line-height: 1.3;
		text-align: left;
		border: 1px solid transparent;
		box-shadow: 0 4px 14px -4px rgba(15, 23, 42, 0.35);
		cursor: pointer;
		font-family: inherit;
		animation: slide-in 140ms ease-out;
	}
	.toast-info {
		background: #1e293b;
		color: #f1f5f9;
		border-color: #334155;
	}
	.toast-success {
		background: #052e1a;
		color: #bbf7d0;
		border-color: #166534;
	}
	.toast-warn {
		background: #422006;
		color: #fde68a;
		border-color: #b45309;
	}
	.toast-error {
		background: #450a0a;
		color: #fecaca;
		border-color: #b91c1c;
	}
	.toast:hover {
		filter: brightness(1.1);
	}
	@keyframes slide-in {
		from {
			transform: translateY(8px);
			opacity: 0;
		}
		to {
			transform: translateY(0);
			opacity: 1;
		}
	}
</style>
