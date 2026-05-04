import { writable } from 'svelte/store';
import type { SessionSnapshot } from '$lib/api';

export const sessionStore = writable<SessionSnapshot | null>(null);
