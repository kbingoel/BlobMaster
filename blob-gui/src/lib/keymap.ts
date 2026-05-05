/**
 * Single source of truth for keyboard shortcuts. The `KeymapOverlay`
 * component renders this list when the user presses `?`, and the README
 * sources its shortcut docs from this file. Bindings that don't appear here
 * shouldn't exist — anywhere a component intercepts a key it should be
 * documented in this list too.
 */

export interface KeyBinding {
	keys: string[]; // displayed glyphs, e.g. ["Enter"], ["Ctrl", "Z"]
	description: string;
}

export interface KeymapSection {
	title: string;
	bindings: KeyBinding[];
}

export const KEYMAP: KeymapSection[] = [
	{
		title: 'Global',
		bindings: [
			{ keys: ['?'], description: 'Toggle this help overlay' },
			{ keys: ['Esc'], description: 'Close help / cancel trump edit / clear pre-armed rank' }
		]
	},
	{
		title: 'Hand entry',
		bindings: [
			{ keys: ['2', '–', '9', ',', 'T', 'J', 'Q', 'K', 'A'], description: 'Pre-arm a rank' },
			{ keys: ['S', 'H', 'C', 'D'], description: 'Toggle the cell at the pre-armed rank' }
		]
	},
	{
		title: 'Bidding',
		bindings: [
			{ keys: ['0', '–', '9'], description: "Record the active bidder's bid" },
			{ keys: ['Enter'], description: "Accept the AI's recommended bid (your turn only)" }
		]
	},
	{
		title: 'Trick play',
		bindings: [
			{ keys: ['Enter'], description: "Play the AI's recommended card (your turn only)" },
			{ keys: ['E'], description: 'Cycle eval display: Win-rate / Policy / MCTS visits / Value / Off' }
		]
	},
	{
		title: 'Trump editor (round-progress strip)',
		bindings: [
			{ keys: ['Edit trumps'], description: 'Click to enter edit mode' },
			{ keys: ['1'], description: '♠ — set spades for the cursor round and advance' },
			{ keys: ['2'], description: '♥ — set hearts and advance' },
			{ keys: ['3'], description: '♣ — set clubs and advance' },
			{ keys: ['4'], description: '♦ — set diamonds and advance' },
			{ keys: ['5'], description: 'NT — set no-trump and advance' },
			{ keys: ['←', '→'], description: 'Move the cursor without changing trumps' },
			{ keys: ['Enter'], description: 'Save changes' },
			{ keys: ['Esc'], description: 'Cancel without saving' }
		]
	},
	{
		title: 'Round transitions',
		bindings: [{ keys: ['N'], description: 'Continue to the next round from the round-summary screen' }]
	}
];
