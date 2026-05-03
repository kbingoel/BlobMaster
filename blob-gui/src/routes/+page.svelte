<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	onMount(() => {
		goto('/setup', { replaceState: true });
	});
</script>

<main class="grid min-h-screen place-items-center text-sm text-slate-500">
	<p>Loading…</p>
</main>
