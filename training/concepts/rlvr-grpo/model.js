// rlvr-grpo's pure model: a group of answers, its rewards, advantages, per-token weights and clipped surrogates.
// Everything numeric comes from math/grpo.js; this file only wires it to the page's state. No DOM, inputs never mutated.
import { GROUP_TOY, buildGroup, verifyFinalAnswer, groupStats, groupAdvantages, hasSignal, totalPush, tokenWeights, clippedSurrogate } from '@math/grpo.js';
import { PI_OLD, EPS_LOW, ratioAt } from './numbers.js';

export const answersFor = (k) => buildGroup(k, GROUP_TOY);

// One token's numbers: advantage A, weight w, push A·w, the ratio pair, the clipped objective.
function tokenRow(answer, advantage, weights, epsHigh) {
  return answer.map((text, index) => {
    const ratio = ratioAt(answer, index);
    const { objective, clipped } = clippedSurrogate(ratio, advantage, { epsLow: EPS_LOW, epsHigh });
    return { text, advantage, weight: weights[index], push: advantage * weights[index], sampled: PI_OLD, now: PI_OLD * ratio, ratio, objective, clipped };
  });
}

// state: { k, norm, agg, epsHigh } → the whole group.
export function evaluate({ k, norm, agg, epsHigh }) {
  const answers = answersFor(k);
  const rewards = answers.map((tokens) => verifyFinalAnswer(tokens, GROUP_TOY.target));
  const stats = groupStats(rewards);
  const advantages = groupAdvantages(rewards, { normalizeStd: norm });
  const weights = tokenWeights(answers.map((tokens) => tokens.length), { aggregation: agg });
  const rows = answers.map((tokens, i) => {
    const row = tokenRow(tokens, advantages[i], weights[i], epsHigh);
    return { tokens, reward: rewards[i], advantage: advantages[i], tokenRows: row, pushPerAnswer: row.reduce((sum, t) => sum + t.push, 0) };
  });
  return { rows, rewards, advantages, stats, signal: hasSignal(rewards), totalPush: totalPush(advantages) };
}

// The Σ|A| of a group with and without the std division, and their ratio (try-this 1: equals 1 / std).
export function divisionEffect(k) {
  const rewards = answersFor(k).map((tokens) => verifyFinalAnswer(tokens, GROUP_TOY.target));
  const withStd = totalPush(groupAdvantages(rewards));
  const withoutStd = totalPush(groupAdvantages(rewards, { normalizeStd: false }));
  return { withStd, withoutStd, ratio: withStd / withoutStd };
}

// The std a trainer gets when it divides by G − 1 instead of G (math note a): the factor and the right answer's advantage.
export function unbiasedStd(k) {
  const { mean, std } = groupStats(answersFor(k).map((tokens) => verifyFinalAnswer(tokens, GROUP_TOY.target)));
  const G = GROUP_TOY.slotOrder.length;
  const factor = Math.sqrt(G / (G - 1));
  return { factor, advantage: (1 - mean) / (std * factor) };
}
