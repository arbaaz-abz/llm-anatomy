// Picking the next token (docs/storyboards/sampling.md §6): temperature, top-k, top-p and the seeded draw over the
// course's 16-word toy vocabulary. Pure functions, no DOM; inputs are never mutated.
import { softmax, mulberry32, deepFreeze } from './core.js';

// The course's toy vocabulary. Owned here: speculative-decoding draws its tokens from it.
export const VOCAB = deepFreeze(['The', 'cat', 'sat', 'down', 'on', '.', 'and', 'the', 'mat', 'a', 'dog', 'ran', 'up', 'big', 'was', 'then']);
// decoder-anatomy frame 8: on 2.0 · "." 1.5 · and 0.5 · the 0.0 · the other 12 words −1.0.
export const LOGITS = deepFreeze([-1, -1, -1, -1, 2, 1.5, 0.5, 0, -1, -1, -1, -1, -1, -1, -1, -1]);
export const NAMED = deepFreeze([4, 5, 6, 7]); // on . and the

const TIE_TOLERANCE = 1e-12; // a running sum this close below p counts as having reached it

const sum = (list) => list.reduce((acc, x) => acc + x, 0);

// Indices sorted by probability, highest first; ties go to the lower vocabulary index. Every filter and the draw use it.
export function rankTokens(probs) {
  return probs.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value || a.index - b.index).map((o) => o.index);
}

// Temperature 0 is greedy: an exact one-hot on the top token. It is special-cased on purpose, because
// core.softmax throws on a temperature <= 0. Any other temperature is core.softmax(logits, temperature).
export function applyTemperature(logits, temperature) {
  if (temperature !== 0) return softmax(logits, temperature);
  if (logits.length === 0) throw new RangeError('applyTemperature: logits must not be empty');
  const top = rankTokens(logits)[0];
  return logits.map((_, i) => (i === top ? 1 : 0));
}

// Keep the tokens in `keep` (a Set of indices), rescale them to add up to 1. `kept` counts the survivors that can
// still be drawn (probability above 0); `mass` is the probability they held before rescaling.
function rescale(probs, keep) {
  const mass = sum([...keep].map((i) => probs[i]));
  const out = probs.map((value, i) => (keep.has(i) ? value / mass : 0));
  return { probs: out, kept: out.filter((v) => v > 0).length, mass };
}

// The k highest probabilities (ties by vocabulary order), rescaled. A k above the vocabulary size keeps everything.
export function topK(probs, k) {
  if (!Number.isInteger(k) || k < 1) throw new RangeError(`topK: k must be a whole number of at least 1, got ${k}`);
  return rescale(probs, new Set(rankTokens(probs).slice(0, k)));
}

// The shortest prefix of rankTokens whose probabilities reach p, rescaled.
export function topP(probs, p) {
  if (!(p > 0 && p <= 1)) throw new RangeError(`topP: p must be above 0 and at most 1, got ${p}`);
  const keep = new Set();
  let cumulative = 0;
  for (const i of rankTokens(probs)) {
    keep.add(i);
    cumulative += probs[i];
    if (cumulative >= p - TIE_TOLERANCE) break;
  }
  return rescale(probs, keep);
}

// Temperature, then top-k, then top-p, then rescale (a common order; libraries differ). `kept` and `mass` describe
// the final survivors against the distribution after temperature.
export function samplingDistribution(logits, { temperature = 1, topK: k = null, topP: p = null } = {}) {
  const afterTemperature = applyTemperature(logits, temperature);
  const afterK = k == null ? afterTemperature : topK(afterTemperature, k).probs;
  const probs = p == null ? afterK : topP(afterK, p).probs;
  const survivors = probs.map((v, i) => (v > 0 ? i : -1)).filter((i) => i >= 0);
  return { probs, kept: survivors.length, mass: sum(survivors.map((i) => afterTemperature[i])) };
}

// Walk rankTokens(probs) and return the first index whose cumulative probability exceeds u (0 <= u < 1).
// A running sum that falls short of u by rounding lands on the last token that can still be drawn.
export function sampleIndex(probs, u) {
  if (!(u >= 0 && u < 1)) throw new RangeError(`sampleIndex: u must be in [0, 1), got ${u}`);
  const ranked = rankTokens(probs);
  let cumulative = 0;
  for (const i of ranked) {
    cumulative += probs[i];
    if (u < cumulative) return i;
  }
  return ranked.filter((i) => probs[i] > 0).at(-1);
}

// n draws, each from the next number of mulberry32(seed): the same seed repeats the same draws.
export function drawSamples(probs, n, seed) {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`drawSamples: n must be a whole number of at least 0, got ${n}`);
  if (!Number.isInteger(seed)) throw new RangeError(`drawSamples: seed must be a whole number, got ${seed}`);
  const next = mulberry32(seed);
  return Array.from({ length: n }, () => sampleIndex(probs, next()));
}

// Display helper: the named cells plus one "others" cell. `each` is the largest probability among the others (all of
// them when they tie, as in this vocabulary), `together` their sum, `kept` how many can still be drawn, `of` how many.
export function collapseOthers(probs, named) {
  const rest = probs.filter((_, i) => !named.includes(i));
  return {
    named: named.map((i) => probs[i]),
    others: { each: rest.length ? Math.max(...rest) : 0, together: sum(rest), kept: rest.filter((v) => v > 0).length, of: rest.length },
  };
}
