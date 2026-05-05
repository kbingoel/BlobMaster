<script lang="ts">
	import { onMount } from 'svelte';
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';
	import { commands } from '$lib/api';
	import { sessionStore } from '$lib/stores/session';
	import Toast from '$lib/components/Toast.svelte';
	import KeymapOverlay from '$lib/components/KeymapOverlay.svelte';

	let { children } = $props();

	onMount(() => {
		// Save-on-quit: best-effort. The Tauri webview fires beforeunload
		// when the window closes; we shoot a save_session over the bridge
		// (sync-ish — invoke is fire-and-forget from this hook's PoV).
		// Per Session 9.8: "save … on every round end and on app close."
		const handler = () => {
			let s: unknown = null;
			const unsub = sessionStore.subscribe((v) => (s = v));
			unsub();
			if (s) commands.saveSession().catch(() => {});
		};
		window.addEventListener('beforeunload', handler);
		return () => window.removeEventListener('beforeunload', handler);
	});
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>
{@render children()}
<Toast />
<KeymapOverlay />
