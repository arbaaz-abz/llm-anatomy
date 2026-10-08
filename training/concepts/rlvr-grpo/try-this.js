// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from model.js
// (math/grpo.js), formatted like the toy's own cells. Pure, no DOM. [prompt, insight, rest]; the page prints
// "prompt → Insight: insight" + rest (README lesson 34).
import { formatRatio } from '@math/core.js';
import { evaluate, divisionEffect } from './model.js';
import { INITIAL_STATE, signed2, signed4, fmt2, fmt3 } from './format.js';
import { ZOOM_EPS_HIGH } from './numbers.js';

const run = (patch) => evaluate({ ...INITIAL_STATE, ...patch });
const firstA = (patch) => run(patch).rows[0].advantage;
const rightAdvantages = (ks, patch = {}) => ks.map((k) => signed2(firstA({ ...patch, k }))).join(' → ');
const totals = (ks) => ks.map((k) => fmt2(run({ k }).totalPush)).join(' → ');
const lone = run({ k: 1 });
const loneOff = run({ k: 1, norm: false });
const effect = (k) => {
  const { withStd, withoutStd, ratio } = divisionEffect(k);
  return `${fmt2(withStd)} vs ${fmt2(withoutStd)} at k = ${k} (${formatRatio(ratio)})`;
};

function pushTrySecond() {
  const row = (state, i) => run(state).rows[i];
  const [long, short] = [3, 5];
  const perToken = (state, i) => row(state, i).tokenRows[0].push;
  const token = { agg: 'token' };
  return `With k = 2, select the 48 in row 4 (7 × 8 = 48 , so 48, ${row({}, long).tokens.length} tokens) and then the 63 in row 6 (${row({}, short).tokens.length} token). `
    + `Under sample aggregation the inspector shows a push per token of ${signed4(perToken({}, long))} for row 4 and ${signed4(perToken({}, short))} for row 6: both wrong, `
    + `but the long answer is punished ${formatRatio(perToken({}, short) / perToken({}, long))} less per token, and both rows print the same push per answer, ${signed4(row({}, long).pushPerAnswer)}. `
    + `Flip agg to token: every token now pays ${signed4(perToken(token, long))}, so the long wrong answer's push per answer becomes ${signed4(row(token, long).pushPerAnswer)} and the short one's ${signed4(row(token, short).pushPerAnswer)}.`;
}

function clipTryThird() {
  const hot = run({}).rows;
  const wide = run({ epsHigh: ZOOM_EPS_HIGH }).rows;
  const [one, five, four] = [hot[0].tokenRows[4], hot[4].tokenRows[4], hot[3].tokenRows[4]];
  return `With k = 2 and ε_high = ${fmt2(INITIAL_STATE.epsHigh)}, three chips are hatched: 56 in row 1 (r = ${fmt2(one.ratio)}), 56 in row 5 (r = ${fmt2(five.ratio)}), and the first 48 in row 4 (r = ${fmt2(four.ratio)}). `
    + `Select row 1's 56: objective ${fmt3(one.objective)}, clipped. Press the clip-higher ${fmt2(ZOOM_EPS_HIGH)} chip: row 1's 56 loses its hatch (objective ${fmt3(wide[0].tokenRows[4].objective)}, gradient on); `
    + `row 5's 56 stays hatched (${fmt2(five.ratio)} > ${fmt2(1 + ZOOM_EPS_HIGH)}); row 4's 48 stays hatched (${fmt2(four.ratio)} < 0.8, the lower bound did not move).`;
}

export function tryThis() {
  return [
    [`Predict first: if 4 of 8 answers are right instead of 2, does 7 × 8 = 56 get a bigger or smaller push? Slide k 2 → 4 → 8. A for a right answer goes ${rightAdvantages([2, 4, 8])}; Σ|A| goes ${totals([2, 4, 8])}. `
      + `Now slide to 1: ${signed2(lone.rows[0].advantage)} for the lone right answer, ${signed2(lone.rows[1].advantage)} for each of the seven wrong ones; and to 0: all zeros, the same banner as at 8. `
      + `Then switch norm off and repeat: ${rightAdvantages([2, 4, 8], { norm: false })}, and the lone right answer gets only ${signed4(loneOff.rows[0].advantage).replace(/0$/, '')}.`,
    'Advantages are relative, not absolute.',
    ` The rarer the outcome inside its group, the bigger its push; dividing by the std amplifies that unevenly. Compare Σ|A| with and without the division: ${[1, 2, 4].map(effect).join(', ')}. `
      + 'The total push still peaks at k = 4, but the division boosts near-impossible and near-solved prompts the most (the difficulty bias), which is Dr.GRPO\'s argument for dropping it, as DeepSeek-V3.2 does. '
      + 'A group with no spread teaches nothing, which is why dynamic sampling throws it away and why GLM-5 keeps only prompts its previous model solves rarely but can solve (see the facts below).'],
    [pushTrySecond(),
      'Sample-level averaging shields long wrong answers (and dilutes long right ones); token-level loss charges every token the same.',
      ' That is DAPO\'s token-level loss, part of the 2026 consensus recipe; Olmo 3 reports it (see the facts below).'],
    [clipTryThird(),
      'Clipping switches a token\'s gradient off, it does not shrink it, and clip-higher widens only the upward side',
      ' so a good token that was improbable when sampled can keep gaining probability (no entropy collapse), while the lower bound still stops collapse the other way.'],
  ];
}

