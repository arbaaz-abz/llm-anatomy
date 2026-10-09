// disaggregation toy view model (pure, no DOM): state + data → every string the toy prints and the memory bar's spec.
// Bytes through formatBytes, durations through formatDuration, shares through sharePct (pct1), exact integers through formatInt,
// per-expert counts and intensities through formatCount (README lesson 35, P4-R6).
import { formatBytes, formatCount, formatDuration, formatInt } from '@math/core.js';
import { INITIAL_STATE, setupFrom, shipAnalysis, expertAnalysis, expertVerdict, checkWork, pct1, prefillCrossover } from './format.js';

export const TOY_NOTE = 'Floors from bytes, bandwidth and FLOPs; real transfers add start-up latency, and real all-to-all adds communication time not modeled here.';
export const NOT_FIT = 'does not fit';
export const SHORT_NOTE = (crossover) => `below ${formatInt(crossover)} tokens prefill is one weight read, so the ratio is smaller here; real transfers add a fixed start-up cost the toy does not model, which is why short prompts gain least`;

const BOUND_WORDS = Object.freeze({ 'memory-bound': 'memory-bound', 'compute-bound': 'compute-bound' });

function shipView(a, state) {
  return {
    kvBytes: formatBytes(a.kvBytes),
    transfer: formatDuration(a.transferS),
    prefill: formatDuration(a.prefillS),
    ratio: pct1(a.transferS, a.prefillS),
    mlaKv: formatBytes(a.mla.kvBytes),
    mlaTransfer: formatDuration(a.mla.transferS),
    mlaBytesPerToken: formatInt(a.mla.bytesPerToken),
    bytesPerToken: formatInt(a.bytesPerToken),
    linkLabel: a.link.label,
    shortNote: a.belowCrossover ? SHORT_NOTE(prefillCrossover()) : '',
    prompt: state.prompt,
  };
}

// The per-GPU memory bar: weights and free HBM, or no bar when the weights alone exceed the memory.
function memoryBar(e, setup) {
  const hbm = setup.gb300.hbmBytes;
  if (!e.fits) return null;
  return { parts: [{ name: `weights ${formatBytes(e.weightsBytes)}`, value: e.weightsBytes, hue: 1 }, { name: `free ${formatBytes(e.freeBytes)}`, value: e.freeBytes, hue: 2 }], total: hbm };
}

function expertView(e, setup) {
  const basis = setup.gb300.hbmBasis;
  return {
    expertsPerGpu: formatCount(e.expertsPerGpu),
    weights: formatBytes(e.weightsBytes),
    free: e.fits ? formatBytes(e.freeBytes) : NOT_FIT,
    hbmNote: `${formatBytes(setup.gb300.hbmBytes)} HBM per GPU (${basis}); weights ${formatBytes(setup.v4.checkpointBytes)} as shipped (reported)`,
    tokens: formatCount(e.tokens),
    intensity: formatCount(e.intensity),
    ridge: formatInt(e.ridge),
    tokensNeeded: formatInt(Math.ceil(e.tokensNeeded)),
    verdict: BOUND_WORDS[expertVerdict(e)],
    bar: memoryBar(e, setup),
  };
}

export function toyView(state, data) {
  const setup = setupFrom(data);
  return {
    setup,
    ship: shipView(shipAnalysis(state, setup), state),
    experts: expertView(expertAnalysis(state, setup), setup),
    checkWork: checkWork(state, setup),
  };
}

export { INITIAL_STATE };
