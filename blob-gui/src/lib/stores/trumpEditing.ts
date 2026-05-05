import { writable } from 'svelte/store';

/**
 * `true` while the RoundProgressStrip is in trump-edit mode. Other keyboard
 * consumers (BiddingKeypad's number keys, CardGrid's rank keys) read this so
 * the digit keys 1-5 and arrow keys are claimed exclusively by the editor.
 */
export const trumpEditingStore = writable(false);
