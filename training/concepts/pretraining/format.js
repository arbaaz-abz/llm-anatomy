// pretraining pure helpers (no DOM): the toy's state and its transitions, number formats, and "Check my work".
// Every loss comes from math/lm.js; this file only arranges and formats them.
import { deepFreeze } from '@math/core.js';
import { tokenLoss, meanLoss, perplexity } from '@math/lm.js';
import { TOKENS, STAND_INS } from './numbers.js';

export const POSITIONS = Object.freeze([1, 2, 3, 4, 5, 6, 7]); // the predicting positions; position n predicts token n + 1
export const P_SLIDER = Object.freeze({ min: 0.01, max: 1, step: 0.01 });
export const UNIFORM_P = 1 / 16;
export const PRESETS = deepFreeze([
  { label: 'confident miss 0.01', p: 0.01 },
  { label: 'uniform 1/16', p: UNIFORM_P },
  { label: 'certain 1.00', p: 1 },
]);

// The toy opens on mat (position 6 predicts token 7) with the animation's stand-ins.
export const INITIAL_STATE = deepFreeze({ pos: 6, probs: STAND_INS });

export const targetOf = (pos) => pos + 1; // the 1-based chip position of the token that position `pos` predicts
export const wordOf = (pos) => TOKENS[pos]; // TOKENS is 0-based, so TOKENS[pos] is token pos + 1
const shownWord = (word) => (word === '.' ? '"."' : word);

function checkPos(fn, pos) {
  if (!POSITIONS.includes(pos)) throw new RangeError(`${fn}: pos must be an integer 1–7, got ${pos}`);
}

function checkP(fn, p) {
  if (typeof p !== 'number' || !(p > 0 && p <= 1)) throw new RangeError(`${fn}: p must be a probability with 0 < p ≤ 1, got ${p}`);
}

// ---- state transitions (pure: every one returns a new frozen state) ----
export function selectPosition(state, pos) {
  checkPos('selectPosition', pos);
  return Object.freeze({ ...state, pos });
}

export function setProb(state, p) {
  checkP('setProb', p);
  return Object.freeze({ ...state, probs: Object.freeze(state.probs.map((q, i) => (i === state.pos - 1 ? p : q))) });
}

export const allUniform = (state) => Object.freeze({ ...state, probs: Object.freeze(state.probs.map(() => UNIFORM_P)) });
export const resetProbs = (state) => Object.freeze({ ...state, probs: STAND_INS });

// ---- number formats (one per quantity) ----
const MAX_P_DECIMALS = 4;
const MIN_P_DECIMALS = 2;
// A probability at up to 4 decimals, trailing zeros dropped down to 2: 0.45, 0.10, 1.00, 0.0625, 0.3903.
export function formatProb(p) {
  checkP('formatProb', p);
  const [whole, frac] = p.toFixed(MAX_P_DECIMALS).split('.');
  return `${whole}.${frac.replace(/0+$/, '').padEnd(MIN_P_DECIMALS, '0')}`;
}
export const formatLoss = (loss) => loss.toFixed(3); // losses are ≥ 0, so no minus sign is ever printed
export const formatPerplexity = (ppl) => ppl.toFixed(2);

// ---- "Check my work" (storyboard §6): the selected line first, then the seven losses at 4 decimals ----
export function checkWork(state) {
  const losses = state.probs.map(tokenLoss);
  const p = state.probs[state.pos - 1];
  const sum = losses.reduce((s, l) => s + l, 0).toFixed(4);
  const mean = meanLoss(state.probs);
  const ppl = perplexity(mean);
  return [
    `selected: ${shownWord(wordOf(state.pos))}   −ln ${formatProb(p)} = ${tokenLoss(p).toFixed(4)}`,
    `sum   = ${losses.map((l) => l.toFixed(4)).join(' + ')} = ${sum}`,
    `mean  = ${sum} ÷ ${losses.length} = ${mean.toFixed(4)} → ${formatLoss(mean)}`,
    `perplexity = e^${mean.toFixed(4)} = ${ppl.toFixed(3)} → ${formatPerplexity(ppl)}`,
  ].join('\n');
}
