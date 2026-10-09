// serving-calculator pure helpers (no DOM): the toy's initial states, the number formats it prints, and the "Check my work" text.
// Every number comes from model.js (math/serving.js and friends); one formatter per quantity (README lesson 35, P4-R6):
// bytes → formatBytes, exact integers → formatInt, per-user rates → formatCount, durations → formatDuration, ratios → formatRatio.
import { formatBytes, formatCount, formatDuration, formatInt, formatRatio, formatUsd } from '@math/core.js';
import { V4_ID, LLAMA_ID } from './presets.js';

const SCENARIO_DEFAULTS = Object.freeze({ target: 27, price: 2.65, hit: 0, mtp: false });
const V4_DEFAULTS = Object.freeze({ model: V4_ID, weights: 'shipped', hw: 'gb300-nvl72', gpus: 16, context: 9216, kvEnd: 'low', kvFormat: 'bf16', users: 'target' });
const LLAMA_DEFAULTS = Object.freeze({ model: LLAMA_ID, weights: 'fp8', hw: 'h200', gpus: 1, context: 9216, kvEnd: 'low', kvFormat: 'bf16', users: 'target' });

// The model-dependent fields a model switch resets (storyboard §6, Review Focus 5): switching away and back restores them exactly.
export const modelDefaults = (model) => (model === V4_ID ? V4_DEFAULTS : LLAMA_DEFAULTS);
export const INITIAL_STATE = Object.freeze({ ...V4_DEFAULTS, ...SCENARIO_DEFAULTS });

// The users slider: numeric stops, then the two stops that move with the state. The slider holds the stop's index.
export const USER_STOPS = Object.freeze([1, 4, 16, 64, 256, 1024, 4096, 'fits', 'target']);
export const USER_STOP_INDEX = Object.freeze(Object.fromEntries(USER_STOPS.map((s, i) => [s, i])));
export const userStopLabel = (stop) => (stop === 'target' ? 'most at target' : stop === 'fits' ? 'max that fits' : formatInt(stop));

export const int = formatInt;
export const dur = formatDuration;
export const rate = (n) => formatCount(n);
export const ratio = formatRatio;
export const bytes = formatBytes;
// Bytes with a real minus for a shortfall ("−577 GB").
export const signedBytes = (n) => (n < 0 ? `−${formatBytes(-n)}` : formatBytes(n));
// Dollars through the shared formatter: three significant figures, at least cents ("$2.65", "$0.0144", "$0.119", "$87,072").
export const usd = formatUsd;
// 49e9 · 70e9 (the parameter count as the storyboard's check lines write it).
export const sci = (n) => `${Number((n / 1e9).toPrecision(3))}e9`;
const plain = (n) => String(Number(n.toPrecision(3)));
const gpuWord = (n) => `${int(n)} GPU${n === 1 ? '' : 's'}`;

// "1.73×", or "0.9 of the plain speed" for a slowdown (a ratio below 1 is never "N×", X-3).
export const speedupText = (x) => (x >= 1 ? formatRatio(x) : `${Number(x.toPrecision(2))} of the plain speed`);

// Why a speed number is missing: the weights do not fit, or they fit and not even one user does.
export const noUsersText = (c) => (c.fits ? 'no user fits' : 'does not fit');

function fitLine(c, state) {
  const perGpu = c.weights / state.gpus;
  const tail = c.fits ? '' : ': does not fit';
  return `Fit:    ${bytes(c.weights)} ÷ ${bytes(c.gpu.hbmBytes)} → ${gpuWord(c.minGpus)} for weights alone · ${bytes(c.weights)} ÷ ${state.gpus} = ${bytes(perGpu)} per GPU · ${bytes(c.gpu.hbmBytes)} − ${bytes(perGpu)} = ${signedBytes(c.free)} free${tail}`;
}

function kvLine(c, state) {
  const k = c.chosen;
  const head = `KV:     ${int(k.bytesPerToken)} B × ${int(state.context)} tokens = ${bytes(k.perUser)} per user`;
  return c.fits ? `${head} · ${bytes(c.free)} ÷ ${bytes(k.perUser)} = ${int(k.maxFit)} users fit` : `${head} · nothing is free: 0 users fit`;
}

function speedLine(c) {
  if (c.users < 1) return `Speed:  ${noUsersText(c)}, so there is no step to time`;
  const m = c.stepModel;
  const { step } = c;
  return `Speed:  t_math = 2 × ${sci(m.activeParamsPerGpu)} × ${int(c.users)} ÷ ${int(m.peakTflops)} TFLOP/s = ${dur(step.computeS)} · t_read = (${bytes(m.weightBytesPerGpu)} + ${bytes(step.actBytes)} + ${bytes(step.kvBytes)}) ÷ ${m.bandwidthTBps} TB/s = ${dur(step.memoryS)} · step = max = ${dur(step.timeS)}`;
}

// What limits the users at the target; a weights-don't-fit state says so instead of naming a cap.
export const limitText = (c) => (c.fits ? c.target.limit : 'does not fit');

function targetLine(c, state) {
  const t = c.target;
  const capacity = c.fits && t.limit === 'memory capacity' ? `, capacity ${int(t.maxUsers)}` : '';
  return `Target: 1 ÷ ${state.target} tok/s = ${dur(1 / state.target)} · compute allows ${int(t.byCompute)} users, memory ${int(t.byMem)}${capacity} · users = min = ${int(t.users)} (${limitText(c)})`;
}

const mtpLine = (c) => `MTP:    ${c.mtp.tokensPerRound.toFixed(2)} tokens per round × ${dur(c.mtp.plainMs / 1e3)} plain ÷ ${dur(c.mtp.roundMs / 1e3)} per round = ${speedupText(c.mtp.speedup)}`;

function costLine(c, state) {
  if (c.users < 1) return `Cost:   ${noUsersText(c)}, so there is no output to price`;
  return `Cost:   ${usd(state.price)} ÷ (${int(c.tokSGpu)} tok/s × 3,600 s) × 1,000,000 = ${usd(c.cost)} per M output tokens`;
}

function inputLine(c, state) {
  const m = c.stepModel;
  const hits = state.hit > 0 ? ` ÷ (1 − ${plain(state.hit)}) = ${int(c.inputTokS)} tok/s with cache hits` : '';
  const head = `Input:  ${int(m.peakTflops)} TFLOP/s ÷ (2 × ${sci(m.activeParamsPerGpu)}) = ${int(c.prefillTokS)} tok/s${hits}`;
  return c.users < 1 ? head : `${head} · output ÷ input cost = ${int(c.inputTokS)} ÷ ${int(c.tokSGpu)} = ${plain(c.costRatio)}`;
}

// The "Check my work" box (storyboard §6): one line per readout group, each the group's formula with this state's numbers.
// MTP off prints no MTP line. The default text is pinned by tests/serving-calculator-page.test.js.
export function checkWork(c, state) {
  return [fitLine(c, state), kvLine(c, state), speedLine(c), targetLine(c, state), ...(c.mtp ? [mtpLine(c)] : []), costLine(c, state), inputLine(c, state)].join('\n');
}
