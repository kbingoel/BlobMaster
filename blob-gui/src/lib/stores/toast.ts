import { writable } from 'svelte/store';

export type ToastKind = 'info' | 'success' | 'warn' | 'error';

export interface Toast {
	id: number;
	kind: ToastKind;
	message: string;
}

export const toastStore = writable<Toast[]>([]);

let nextId = 1;
const DEFAULT_TTL_MS: Record<ToastKind, number> = {
	info: 3000,
	success: 2500,
	warn: 4500,
	error: 6000
};

export function pushToast(message: string, kind: ToastKind = 'info', ttlMs?: number): number {
	const id = nextId++;
	toastStore.update((list) => [...list, { id, kind, message }]);
	const ttl = ttlMs ?? DEFAULT_TTL_MS[kind];
	if (ttl > 0) setTimeout(() => dismissToast(id), ttl);
	return id;
}

export function dismissToast(id: number) {
	toastStore.update((list) => list.filter((t) => t.id !== id));
}
