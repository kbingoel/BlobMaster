import type { CardEval, EvalDisplay } from '$lib/api';

/**
 * Cycle order for the `E` keyboard shortcut. `Off` is the last stop —
 * pressing `E` from `Off` returns to `WinRate`.
 */
export const EVAL_CYCLE: EvalDisplay[] = [
	'win-rate',
	'policy',
	'mcts-visits',
	'value',
	'off'
];

export function cycleEvalDisplay(current: EvalDisplay): EvalDisplay {
	const i = EVAL_CYCLE.indexOf(current);
	return EVAL_CYCLE[(i + 1) % EVAL_CYCLE.length];
}

export function evalDisplayLabel(d: EvalDisplay): string {
	switch (d) {
		case 'win-rate':
			return 'win-rate';
		case 'policy':
			return 'policy';
		case 'mcts-visits':
			return 'visits';
		case 'value':
			return 'value';
		case 'off':
			return 'off';
	}
}

/**
 * Renders the primary metric for an eval entry under the chosen display
 * mode. Returns `'—'` when the metric is unavailable (e.g. visits == 0
 * in pure-policy mode).
 */
export function primaryMetric(entry: CardEval, mode: EvalDisplay): string {
	switch (mode) {
		case 'win-rate':
			return `${Math.round(entry.win_rate * 100)}%`;
		case 'policy':
			return `${Math.round(entry.policy * 100)}%`;
		case 'mcts-visits':
			return entry.mcts_visits > 0 ? `${entry.mcts_visits}` : '—';
		case 'value': {
			const v = entry.mcts_value;
			return (v >= 0 ? '+' : '') + v.toFixed(2);
		}
		case 'off':
			return '';
	}
}

/**
 * Secondary metric — by default policy when the primary is win-rate,
 * win-rate otherwise. Prefixed with a glyph to disambiguate from the
 * primary line.
 */
export function secondaryMetric(entry: CardEval, mode: EvalDisplay): string {
	switch (mode) {
		case 'win-rate':
			return `π ${Math.round(entry.policy * 100)}%`;
		case 'policy':
			return `${Math.round(entry.win_rate * 100)}% W`;
		case 'mcts-visits':
			return `π ${Math.round(entry.policy * 100)}%`;
		case 'value':
			return entry.mcts_visits > 0 ? `${entry.mcts_visits}n` : '—';
		case 'off':
			return '';
	}
}

/**
 * Background tint for a hand card based on its win-rate percentile within
 * the legal subset. Uses a green→red gradient: highest win-rate goes
 * brightest green, lowest brightest red, intermediate values blend.
 */
export function winRateTint(entry: CardEval, all: CardEval[]): string {
	if (all.length <= 1) return 'transparent';
	const rates = all.map((e) => e.win_rate);
	const min = Math.min(...rates);
	const max = Math.max(...rates);
	if (max - min < 1e-4) return 'transparent';
	const t = (entry.win_rate - min) / (max - min); // 0 = worst, 1 = best
	// Red (#fecaca) → neutral (#fffbeb) → green (#bbf7d0).
	if (t < 0.5) {
		const u = t * 2;
		const r = Math.round(254 + (255 - 254) * u);
		const g = Math.round(202 + (251 - 202) * u);
		const b = Math.round(202 + (235 - 202) * u);
		return `rgb(${r}, ${g}, ${b})`;
	}
	const u = (t - 0.5) * 2;
	const r = Math.round(255 + (187 - 255) * u);
	const g = Math.round(251 + (247 - 251) * u);
	const b = Math.round(235 + (208 - 235) * u);
	return `rgb(${r}, ${g}, ${b})`;
}
