// Pure formatters for the sampling page: number text, the toy's distribution, the "Check my work" text (storyboard §6)
// and the math panel's worked line. No DOM; every number comes from math/sampling.js.
import { LOGITS, NAMED, VOCAB, applyTemperature, samplingDistribution, softmaxTerms } from '@math/sampling.js';

const MINUS = '−';
const NAME_WIDTH = 11; // "12 others" plus two spaces
const EXPR_WIDTH = 14; // "12 × e^−1.33" plus two spaces
const TERM_DIGITS = 3; // e^(z/T) terms
const SUM_DIGITS = 4; // their sum

// The toy's start: temperature 1, nothing cut, seed 1 (where the animation's frame 7 begins).
export const INITIAL_STATE = Object.freeze({ temperature: 1, greedy: false, topK: 0, topP: 1, seed: 1 });

export const realMinus = (s) => s.replace(/^-/, MINUS);

// The shortest exact form up to `digits` decimals ("−0.5", "0.75", "4"), real minus, never "−0".
export function trimNumber(v, digits = 2) {
  const s = v.toFixed(digits).replace(/\.?0+$/, '');
  return s === '-0' || s === '' ? '0' : realMinus(s);
}

// Table cells: three decimals, "0.000" for a cut token.
export const fmt3 = (v) => v.toFixed(3);

// Stage cells (NUMBER_CELL holds five characters): probabilities at 3 d.p., exact 0 and 1 as integers.
export const probText = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(3));
// Stage score cells: exact up to 2 decimals ("−0.5", "0.75"), at most five characters.
export const scoreText = (v) => trimNumber(v, 2);

// Significant digits; from 1,000 up the whole number prints with thousands separators ("2,981").
export function sig(v, digits) {
  if (v >= 1000) return Math.round(v).toLocaleString('en-US');
  return String(Number(v.toPrecision(digits)));
}

export const WORDS = Object.freeze([...NAMED.map((i) => VOCAB[i]), '12 others']);
export const quoted = (word) => (word === '.' ? '"."' : word);

// What the toy computes for a state: the temperature actually used (0 when greedy), the filters, the distribution.
export const effectiveTemperature = (state) => (state.greedy ? 0 : state.temperature);
export const filtersOf = (state) => ({ topK: state.topK > 0 ? state.topK : null, topP: state.topP < 1 ? state.topP : null });
export const distributionOf = (state) => samplingDistribution(LOGITS, { temperature: effectiveTemperature(state), ...filtersOf(state) });

// ---- "Check my work" ----
const OTHERS_COUNT = LOGITS.length - NAMED.length;
const OTHERS_INDEX = LOGITS.findIndex((_, i) => !NAMED.includes(i));
const exponent = (z, temperature) => `e^${trimNumber(z / temperature)}`;
const line = (name, expr, value) => `${name.padEnd(NAME_WIDTH)}${expr.padEnd(EXPR_WIDTH)}= ${value}`;

function filterLines(state, temperature, dist) {
  const { topK: k, topP: p } = filtersOf(state);
  if (k === null && p === null) return [];
  const names = [k === null ? null : `top-k ${k}`, p === null ? null : `top-p ${p}`].filter(Boolean).join(' and ');
  const afterTemperature = applyTemperature(LOGITS, temperature);
  return [
    `${names} kept ${dist.kept} ${dist.kept === 1 ? 'token' : 'tokens'}, holding ${fmt3(dist.mass)}`,
    `on = ${fmt3(afterTemperature[NAMED[0]])} ÷ ${fmt3(dist.mass)} = ${fmt3(dist.probs[NAMED[0]])}`,
  ];
}

export function checkWork(state) {
  const temperature = effectiveTemperature(state);
  if (temperature === 0) {
    return ['greedy (temperature 0): no division, no exponentials.', 'The largest score takes everything: on = 1, every other word 0.'].join('\n');
  }
  const { exps, sum } = softmaxTerms(LOGITS, temperature);
  const terms = NAMED.map((i, w) => line(quoted(WORDS[w]), exponent(LOGITS[i], temperature), sig(exps[i], TERM_DIGITS)));
  const others = line(WORDS[NAMED.length], `${OTHERS_COUNT} × ${exponent(LOGITS[OTHERS_INDEX], temperature)}`, sig(exps[OTHERS_INDEX] * OTHERS_COUNT, TERM_DIGITS));
  const probability = applyTemperature(LOGITS, temperature)[NAMED[0]];
  const answer = `on = ${sig(exps[NAMED[0]], TERM_DIGITS)} ÷ ${sig(sum, SUM_DIGITS)} = ${fmt3(probability)}`;
  return [...terms, others, line('sum', '', sig(sum, SUM_DIGITS)), answer, ...filterLines(state, temperature, distributionOf(state))].join('\n');
}

// ---- the math panel's worked line (storyboard §7), templated from the same terms ----
const texNumber = (v) => trimNumber(v).replace(MINUS, '-');

export function workedTex(temperature = 0.5) {
  const { exps, sum } = softmaxTerms(LOGITS, temperature);
  const expTex = (i) => `e^{${texNumber(LOGITS[i] / temperature)}}`;
  const denominator = [...NAMED.map(expTex), `${OTHERS_COUNT}\\,${expTex(OTHERS_INDEX)}`].join(' + ');
  return `\\text{worked } (T = ${texNumber(temperature)}):\\ \\frac{${expTex(NAMED[0])}}{${denominator}} = ${fmt3(exps[NAMED[0]] / sum)}`;
}
