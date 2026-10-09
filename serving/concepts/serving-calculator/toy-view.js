// serving-calculator toy view model (pure, no DOM): state + data → every string the toy prints and the specs of its figure.
// Numbers come from model.js (math/serving.js, math/specdec.js); formats from format.js (README lesson 35). A state whose
// weights do not fit, or in which not even one user fits, prints words ("does not fit", "no user fits"), never a negative count or NaN.
import { formatRatio } from '@math/core.js';
import { compute } from './model.js';
import {
  checkWork, bytes, int, dur, rate, ratio, usd, speedupText, noUsersText, limitText, signedBytes,
} from './format.js';
import { hbmText, WEIGHT_FORMATS } from './presets.js';

export const SPARSE_NOTE = 'Sparse attention reads less than the whole cache; this floor assumes it reads all of it, so per-user speed at this length is a lower bound.';
export const DENSE_NOTE = 'Llama-3.1-70B is dense: a real replica this size shares each layer between GPUs (tensor parallelism), so this floor, where each GPU runs its own users on its share of the weights, is optimistic beyond one GPU.';
export const RATIO_NOTE = 'V4-Pro\'s few-kB KV lets the ideal decode batch reach the ridge, so here the floor makes output as cheap as input; real decode never gets there (see step 8 of the animation).';
export const FLOOR_NOTE = 'Floors from bytes and FLOPs only; real systems are slower (see step 8 of the animation). V4-Pro\'s KV per token is an estimate range; both ends are shown.';
const LONG_CONTEXT = 500_000;
const CONTEXT_LABELS = Object.freeze({ 9216: '8K in / 1K out', 139264: '128K in / 8K out', 131072: 'context 131,072 (128K)', 1000000: 'context 1,000,000 (1M)' });
const RATIO_ONE = 1 + 1e-6;

export const contextLabel = (context) => CONTEXT_LABELS[context] ?? `context ${int(context)}`;

function kvConditions(c, state) {
  if (c.model.dense) return `KV ${bytes(c.chosen.bytesPerToken)}/token (${state.kvFormat === 'fp8' ? 'FP8' : 'BF16'} cache)`;
  return `KV ${bytes(c.model.kvBytes.low)}/token (low) – ${bytes(c.model.kvBytes.high)} (high), speed rows use ${state.kvEnd}`;
}

const parallelism = (c, state) => (c.model.dense ? `${state.gpus} GPU${state.gpus === 1 ? '' : 's'}` : `EP ${state.gpus}`);

// The visible footer line above the readouts; it names the HBM basis (nominal / usable) beside the memory number.
export function conditionsText(c, state) {
  return [`${c.model.label}`, `${c.gpu.label}, ${hbmText(c.gpu)} per GPU`, parallelism(c, state), contextLabel(state.context), kvConditions(c, state),
    `weights ${WEIGHT_FORMATS[state.weights].label}`, 'floor = bytes and FLOPs only, output tokens'].join(' · ');
}

const both = (c, pick) => (c.model.dense ? pick(c.kv[0]) : c.kv.map((k) => `${pick(k)} (${k.end})`).join(' · '));

function fitReadouts(c, state) {
  return {
    'weights-total': bytes(c.weights), 'gpus-min': int(c.minGpus), hbm: hbmText(c.gpu), 'weights-per-gpu': bytes(c.weights / state.gpus),
    'free-per-gpu': c.fits ? bytes(c.free) : 'does not fit',
  };
}

function speedReadouts(c) {
  const none = noUsersText(c);
  const has = c.users >= 1;
  return {
    users: int(c.users), 'step-time': has ? dur(c.step.timeS) : none, bound: has ? `${c.step.bound}-bound` : none,
    'tok-user': has ? `${rate(c.tokSUser)} tok/s` : none, 'tok-gpu': has ? `${int(c.tokSGpu)} tok/s` : none,
    'target-users': int(c.target.users), 'target-limit': limitText(c),
  };
}

function costReadouts(c) {
  const none = noUsersText(c);
  const has = c.users >= 1;
  return {
    'cost-floor': has ? usd(c.cost) : none, 'cost-gb300': usd(c.measured.gb300), 'cost-gb200': usd(c.measured.gb200),
    'prefill-tps': `${int(c.inputTokS)} tok/s`, 'input-cost': usd(c.inputCost), 'cost-ratio': has ? ratio(c.costRatio) : none,
  };
}

function mtpReadouts(c) {
  if (!c.mtp) return { 'mtp-speedup': '', 'mtp-tok-user': '', 'mtp-record': '' };
  const gain = c.scenario.lmsysGainPct;
  return {
    'mtp-speedup': speedupText(c.mtp.speedup), 'mtp-tok-user': `${rate(c.tokSUser * c.mtp.speedup)} tok/s`,
    'mtp-record': `LMSYS measured +${gain}% (${formatRatio(1 + gain / 100)}) per user for DeepSeek-R1 at 128K on GB300 NVL72, a different model; the floor above uses the low end of DeepSeek's reported acceptance.`,
  };
}

// The memory bar and the step bar of the toy's figure (seconds on one scale), or null when nothing fits to draw.
function figureSpec(c) {
  const hbm = c.gpu.hbmBytes;
  const weights = c.stepModel.weightBytesPerGpu;
  const memory = c.fits ? { total: hbm, weights, kv: c.users * c.chosen.perUser, free: Math.max(0, hbm - weights - c.users * c.chosen.perUser) } : null;
  const bw = c.stepModel.bandwidthTBps * 1e12;
  const step = c.users >= 1 ? {
    reading: [{ label: 'weights read', s: weights / bw }, { label: 'KV read', s: c.step.kvBytes / bw }, { label: 'activations', s: c.step.actBytes / bw }],
    mathS: c.step.computeS,
  } : null;
  return { memory, step };
}

function sizeFacts(c, state) {
  const short = c.fits ? '' : ` short by ${signedBytes(-c.free).replace('−', '')}`;
  return { fits: c.fits, short, clamped: c.clamped ? `capped from ${int(state.users)}: only ${int(c.maxFit)} fit` : '' };
}

// Everything the toy prints for one state; `compute` throws a RangeError for a state the GPU cannot run (the toy never offers one).
export function toyView(state, data) {
  const c = compute(state, data);
  const readouts = {
    ...fitReadouts(c, state), ...speedReadouts(c), ...costReadouts(c), ...mtpReadouts(c),
    'kv-per-user': both(c, (k) => bytes(k.perUser)), 'users-fit': both(c, (k) => int(k.maxFit)),
  };
  return {
    readouts, bound: c.users >= 1 ? c.step.bound : null, conditions: conditionsText(c, state), checkWork: checkWork(c, state),
    notes: {
      dense: c.model.dense && state.gpus > 1 ? DENSE_NOTE : '',
      sparse: state.context >= LONG_CONTEXT ? SPARSE_NOTE : '',
      ratio: c.users >= 1 && c.costRatio < RATIO_ONE ? RATIO_NOTE : '',
    },
    sub: sizeFacts(c, state), figure: figureSpec(c), showMtp: c.mtp !== null, compute: c,
  };
}
