// speculative-decoding toy view model (pure, no DOM): state + data → every string and series the toy prints.
// Numbers come from specdec.js and serving.js; the largest batch that fits comes from the H200 entry in data/hardware.json.
import { formatBytes, formatDuration, formatInt } from '@math/core.js';
import { kvCacheBytes } from '@math/memory.js';
import { acceptanceRate, batchSpeedup, expectedTokens, outputDistribution, simpleSpeedup, verifyToken } from '@math/specdec.js';
import { hbmFor, maxUsersPerGpu, stepTime } from '@math/serving.js';
import { lookupFact } from '@shared/claims.js';
import { CONTEXT, MODEL, P, Q, ZOOM_WORDS } from './numbers.js';
import { cellText, chanceText, checkWork, probText, ratioText, tokensText } from './format.js';
import { batchSeries, speedupSeries, tokensSeries } from './sweeps.js';

export const INITIAL_STATE = Object.freeze({ alpha: 0.7, k: 3, c: 0.05, batch: 1, guess: 'down' });
export const FIXED_BATCHES = Object.freeze([1, 4, 16, 64, 128]); // the stops before the largest count that fits
export const C_OPTIONS = Object.freeze([0, 0.05, 0.1, 0.2]);
export const ALPHA_RANGE = Object.freeze({ min: 0.5, max: 0.95, step: 0.05 });
export const K_LIMITS = Object.freeze({ min: 1, max: 8 });
const LONGER_CONTEXT = 2048; // the storyboard's second example

function h200(data) {
  const entry = data?.hardware?.entries?.find((e) => e.id === 'h200');
  if (!entry) throw new RangeError('speculative-decoding toy: data/hardware.json has no h200 entry');
  return hbmFor(entry);
}

// Whole users whose caches fit beside the weights on one H200 at `context` tokens each.
export function usersFit(data, context = CONTEXT) {
  const free = h200(data).bytes - MODEL.weightBytesPerGpu;
  return maxUsersPerGpu(free, kvCacheBytes({ bytesPerToken: MODEL.kvBytesPerToken, tokens: context }));
}

export const batchStops = (data) => [...FIXED_BATCHES, usersFit(data)];

export function usersNote(data) {
  const hbm = h200(data);
  return `Each user has ${formatInt(CONTEXT)} tokens of context, so ${formatInt(usersFit(data))} users fit on one H200 (${formatBytes(hbm.bytes)} ${hbm.basis}, `
    + `${formatBytes(MODEL.weightBytesPerGpu)} of it weights); at ${formatInt(LONGER_CONTEXT)} tokens it would be ${formatInt(usersFit(data, LONGER_CONTEXT))}.`;
}

const listText = (values) => values.map(cellText).join(' · ');

function positionView(guess) {
  const index = ZOOM_WORDS.indexOf(guess);
  if (index < 0) throw new RangeError(`toyView: guess must be one of ${ZOOM_WORDS.join(', ')}, got ${guess}`);
  const check = verifyToken(P, Q, index);
  const result = outputDistribution(P, Q);
  return {
    guessIndex: index, keep: chanceText(check.accept), acceptance: chanceText(acceptanceRate(P, Q)),
    leftover: check.residual, leftoverText: listText(check.residual), result, resultText: listText(result),
  };
}

export function toyView(state, data) {
  const { alpha, k, c, batch, guess } = state;
  const round = batchSpeedup({ alpha, k, c, batch, model: MODEL });
  const verify = stepTime({ ...MODEL, tokens: batch * (k + 1), seqs: batch });
  return {
    tokens: tokensText(expectedTokens(alpha, k)),
    simple: ratioText(simpleSpeedup(alpha, k, c)),
    batchSpeedup: ratioText(round.speedup),
    batchLabel: `Speedup at ${formatInt(batch)} ${batch === 1 ? 'user' : 'users'}`,
    plain: formatDuration(round.plainMs / 1e3),
    verify: formatDuration(round.verifyMs / 1e3),
    verifyBound: `${verify.bound}-bound`,
    checkWork: checkWork(state),
    usersNote: usersNote(data),
    position: positionView(guess),
    kSeries: { tokens: tokensSeries(alpha), speedup: speedupSeries(alpha, c) },
    batchSeries: batchSeries({ alpha, k, c, max: usersFit(data) }),
    alphaText: probText(alpha),
  };
}

// Drafter cost chips and the MTP presets (acceptance from data/serving.json deepseek-v3-mtp.acceptance_pct, one guess).
export const cLabel = probText;
export function mtpPresets(data) {
  const pct = lookupFact(data?.serving, 'deepseek-v3-mtp', 'acceptance_pct')?.value;
  if (!Array.isArray(pct)) return [];
  return pct.map((v) => ({ value: v / 100, label: `MTP ${probText(v / 100)}` }));
}
