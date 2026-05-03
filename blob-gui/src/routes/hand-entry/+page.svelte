<script lang="ts">
	import { goto } from '$app/navigation';
</script>

<main class="mx-auto max-w-3xl px-6 py-12">
	<h1 class="mb-2 text-3xl font-semibold tracking-tight">Hand entry</h1>
	<p class="mb-6 text-sm text-slate-500">
		Lands in Session 9.4. The setup flow already pushes you here; pick the cards in your
		hand and the play screen takes over.
	</p>
	<button
		type="button"
		onclick={() => goto('/setup')}
		class="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-50"
	>
		Back to setup
	</button>
</main>
