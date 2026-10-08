// cluster-topology toy view model (pure, no DOM): state + data → every string the toy prints.
// Peaks and links come from ctx.data (hardware.json keys named in the builder notes); ratios from math/topology.js.
import { formatBytes, formatRatio } from '@math/core.js';
import { epMinLinkGBps } from '@math/topology.js';
import { lookupFact } from '@shared/claims.js';
import { SYSTEMS, CAP_UNITS, LANES } from './numbers.js';
import {
  INITIAL_STATE, CUTS, DEGREE_STOPS, int, pct0, pct1, plural, ratioFor, checkWork, tensorLayerWork, fitsInside,
} from './format.js';

export { INITIAL_STATE };

export const SYSTEM_CHIPS = Object.freeze([
  { value: 'h100', label: 'H100 HGX' },
  { value: 'gb200', label: 'GB200 NVL72 (reported)' },
]);
export const CUT_OPTIONS = Object.freeze([
  { value: 'tensor', label: 'tensor' },
  { value: 'pipeline', label: 'pipeline' },
  { value: 'data', label: 'data' },
  { value: 'expert', label: 'expert' },
]);
export const NO_FIT_NOTE = 'this cut no longer fits in one NVLink domain';
export const BASIS_LINE = 'One full training step (forward and backward). GPT-3\'s shape (d_model 12,288, 96 blocks), compute at the chip\'s dense BF16 peak. Real kernels reach 35–55% of peak ([[scale-reliability]]), so real ratios are smaller, but their order is the same.';
export const TOKENS_NOTE = 'tokens per replica per step: stand-in';
export const HIDE_TEXT = Object.freeze({
  tensor: 'no: on the critical path',
  pipeline: 'mostly, in 1F1B\'s steady state',
  data: 'yes, during backward',
  expert: 'only below the hiding line',
});

function fact(dataset, id, key) {
  const found = lookupFact(dataset, id, key);
  if (!found) throw new RangeError(`cluster-topology toy: data has no ${id}.${key}`);
  return found.value;
}

// The chip's figures from data: dense BF16 peak, NVLink each way, and the scale-out port each way (Conventions).
export function systemFromData(id, data) {
  const stand = SYSTEMS[id];
  return {
    ...stand,
    peakTflops: fact(data?.hardware, stand.hwId, 'bf16_dense_tflops'),
    nvlinkGBps: fact(data?.hardware, stand.hwId, 'nvlink_gb_s_each_way'),
    networkGBps: fact(data?.hardware, stand.networkId, 'gb_s_each_way'),
  };
}

// The degrees a cut offers (the expert cut has none), and the snapped degree when the cut changes.
export const stopsFor = (cut) => DEGREE_STOPS[cut] ?? [];
export function snapDegree(cut, degree) {
  const stops = stopsFor(cut);
  if (stops.length === 0 || stops.includes(degree)) return degree;
  return stops.reduce((best, s) => (Math.abs(s - degree) < Math.abs(best - degree) ? s : best), stops[0]);
}

// The "runs on" choice: "inside" is disabled, with its note, once the degree passes the NVLink domain (P3-R12).
export function whereOptions(state, sys) {
  const fits = fitsInside(sys, state.cut, state.degree);
  return [
    { value: 'inside', label: 'inside', ...(fits ? {} : { disabled: true, note: NO_FIT_NOTE }) },
    { value: 'network', label: 'over the network' },
  ];
}

// The state the toy actually shows: "inside" becomes "network" once it is disabled.
export function effectiveState(state, sys) {
  return fitsInside(sys, state.cut, state.degree) || state.where !== 'inside' ? state : { ...state, where: 'network' };
}

// Lane geometry in time units: compute is 100; the comm lane is cut at the 520 px track with its percentage printed.
export function lanesFor(ratio) {
  const comm = ratio * LANES.computeUnits;
  const capped = comm > CAP_UNITS;
  return {
    compute: LANES.computeUnits,
    comm,
    cap: capped ? { at: CAP_UNITS, label: `continues: ${pct0(ratio)}` } : null,
    note: `both lanes in the same time units${capped ? `; comm lane cut at ${formatRatio(CAP_UNITS / LANES.computeUnits)} compute` : ''}`,
  };
}

const degreeLine = (state) => (state.cut === 'expert' ? 'all experts' : state.cut === 'data' ? `${plural(state.degree, 'replica')}, ${int(state.tokens)} tokens each` : plural(state.degree, 'GPU'));

export function toyView(requested, data) {
  const sys = systemFromData(requested.system, data);
  const state = effectiveState(requested, sys);
  const ratio = ratioFor(state, sys);
  const isExpert = state.cut === 'expert';
  return {
    sys,
    state,
    ratio,
    ratioText: pct1(ratio),
    lanes: lanesFor(ratio),
    hide: HIDE_TEXT[state.cut],
    linkNeeded: isExpert ? `${int(epMinLinkGBps({ peakTflops: sys.peakTflops }))} GB/s` : null,
    bytes: state.cut === 'tensor' ? formatBytes(tensorLayerWork(state.degree).bytes) : null,
    check: checkWork(state, sys),
    where: whereOptions(requested, sys),
    systemLine: `${sys.label}: NVLink ${int(sys.nvlinkGBps)} GB/s each way, network ${int(sys.networkGBps)} GB/s each way, ${int(sys.peakTflops)} TFLOPS dense BF16 per GPU.`,
    cutLine: `${state.cut} parallelism, ${degreeLine(state)}, over ${state.where === 'inside' ? 'NVLink' : 'the network'}`,
  };
}

// The try-this list (README lesson 34): prompts with their numbers computed from the same functions, then the insight.
export function tryThis(data) {
  const at = (system, cut, degree, where, tokens = INITIAL_STATE.tokens) => toyView({ system, cut, degree, where, tokens }, data);
  const r = (view) => view.ratioText;
  const h = (cut, degree, where, tokens) => at('h100', cut, degree, where, tokens);
  const g = (cut, degree, where) => at('gb200', cut, degree, where);
  return [
    {
      text: `Tensor, degree 8, H100 HGX: inside ${r(h('tensor', 8, 'inside'))}, network ${r(h('tensor', 8, 'network'))}. Raise the degree to 16: it no longer fits in a server, so it runs over the network: ${r(h('tensor', 16, 'network'))}. Switch to GB200 NVL72, degree 8, inside: ${r(g('tensor', 8, 'inside'))}; degree 16, inside: ${r(g('tensor', 16, 'inside'))}.`,
      insight: 'tensor parallelism belongs inside the fastest domain and stays small (around 8–16), and newer chips make it harder, because compute grew faster than NVLink.',
    },
    {
      text: `Data, degree 64, over the network: ${r(h('data', 64, 'network'))} at 262,144 tokens per replica; slide tokens down to 16,384: ${r(h('data', 64, 'network', 16384))}. Then pipeline, degree 16, network: ${r(h('pipeline', 16, 'network'))}.`,
      insight: 'data-parallel syncs are cheap only when each replica processes many tokens per step; since they also overlap with the backward pass, they can live on the slowest, outermost links.',
    },
    {
      text: `Expert on H100 HGX: inside ${r(h('expert', 8, 'inside'))}, network ${r(h('expert', 8, 'network'))}; the "link needed" readout says ${h('expert', 8, 'inside').linkNeeded}. Switch to GB200 NVL72: needed ${g('expert', 8, 'inside').linkNeeded}; inside ${r(g('expert', 8, 'inside'))}, network ${r(g('expert', 8, 'network'))}.`,
      insight: 'expert all-to-all hides only inside the NVLink domain, which is why expert groups stay inside a server or a 72-GPU rack (and why DeepSeek-V3 capped each token at 4 servers, frame 8).',
    },
  ];
}

export { CUTS };
