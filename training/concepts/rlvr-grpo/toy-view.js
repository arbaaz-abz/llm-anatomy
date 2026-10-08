// The toy's view model: (state, data) → every string the toy prints. Pure, unit-tested; toy.js only paints it.
import { evaluate } from './model.js';
import { fixed, fmt2, fmt3, fmt4, signed2, signed4, yesNo } from './format.js';
import { GROUP_SIZE, ZOOM_EPS_HIGH, EPS_HIGH_PPO } from './numbers.js';
import { advantageFill } from './stage.js';
import { lookupFact } from '@shared/claims.js';

export const NO_SIGNAL = 'No spread: every A = 0. Dynamic sampling would drop this group.';
export const STAND_IN = 'The eight answers and the ratios are hand-picked stand-ins; a real policy samples them and a real trainer measures the ratios.';
export const K_PRESETS = Object.freeze([0, 2, 4, 8]);

// The clip-higher chip loads from data/models.json glm-5.rl_clip_eps_high (0.28); without data it falls back to the storyboard's 0.28.
export const clipHigherEps = (data) => lookupFact(data?.models, 'glm-5', 'rl_clip_eps_high')?.value ?? ZOOM_EPS_HIGH;

export const epsOptions = (data) => [
  { value: EPS_HIGH_PPO, label: `PPO ${fmt2(EPS_HIGH_PPO)}` },
  { value: clipHigherEps(data), label: `clip-higher ${fmt2(clipHigherEps(data))}` },
];

// The selection survives a change of k: the row stays, the token clamps to that answer's last token.
export function clampSelection({ row, token }, group) {
  return { row, token: Math.min(token, group.rows[row].tokens.length - 1) };
}

export function toyView(state) {
  const group = evaluate(state);
  const selected = clampSelection(state, group);
  const row = group.rows[selected.row];
  const t = row.tokenRows[selected.token];
  const rows = group.rows.map((r, i) => ({
    index: i + 1,
    ok: r.reward === 1,
    reward: String(r.reward),
    advantage: signed2(r.advantage),
    push: signed4(r.pushPerAnswer),
    chips: r.tokenRows.map((c) => ({ text: c.text, fill: advantageFill(c.advantage), hatched: c.clipped })),
  }));
  return {
    group,
    selected,
    kText: `${state.k} / ${GROUP_SIZE}`,
    rows,
    stats: { mean: fixed(group.stats.mean, 3), std: fixed(group.stats.std, 3), totalPush: fmt2(group.totalPush) },
    signal: group.signal ? '' : NO_SIGNAL,
    inspector: [
      ['row', 'Row', String(selected.row + 1)],
      ['token', 'Token', t.text],
      ['adv', 'Advantage A', signed4(t.advantage)],
      ['weight', 'Weight w', fmt4(t.weight)],
      ['push', 'Push per token, A · w', signed4(t.push)],
      ['sampled', 'Probability when sampled', fmt3(t.sampled)],
      ['now', 'Probability now', fmt3(t.now)],
      ['ratio', 'Ratio r = now / when sampled', fmt3(t.ratio)],
      ['objective', 'Objective', fmt3(t.objective)],
      ['clipped', 'Clipped (gradient off)', yesNo(t.clipped)],
    ].map(([name, label, value]) => ({ name: `insp-${name}`, label, value })),
  };
}
