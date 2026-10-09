// disaggregation pure helpers (no DOM): the toy's two analyses (ship the KV; feed the experts), the formatters it prints
// through and the "Check my work" text. Every number comes from math/serving.js, math/roofline.js or math/memory.js.
import { formatBytes, formatCount, formatDuration, formatInt, formatRatio } from '@math/core.js';
import { kvCacheBytes, kvBytesPerTokenMla, sharePct } from '@math/memory.js';
import { matmulCost, arithmeticIntensity, ridgePoint, tokensToComputeBound, bytesPerElement } from '@math/roofline.js';
import { RUNNING_EXAMPLE, stepTime, kvTransferTime, tokensPerExpert, freeHbmPerGpu, hbmFor } from '@math/serving.js';
import { lookupFact } from '@shared/claims.js';
import { MLA, LINK_IDS } from './numbers.js';

export const INITIAL_STATE = Object.freeze({ prompt: 4096, link: 'net400', kv: 'bf16', ep: 16, users: 64 });
export const PROMPT_STOPS = Object.freeze([128, 512, 4096, 8192, 32768, 131072]);
export const EP_STOPS = Object.freeze([1, 8, 16, 32, 72]);
export const USER_PRESETS = Object.freeze([16, 64, 256]);
export const KV_FORMATS = Object.freeze({ bf16: { label: 'BF16', bytesPerElem: 2 }, fp8: { label: 'FP8', bytesPerElem: 1 } });
const BF16_BYTES = KV_FORMATS.bf16.bytesPerElem;
const GB = 1e9;
const CROSSING_STEPS = 60; // bisection steps: far below a token

export const pct1 = (part, whole) => `${sharePct(part, whole).toFixed(1)}%`;

const fact = (dataset, id, key) => {
  const f = lookupFact(dataset, id, key);
  if (f == null) throw new RangeError(`disaggregation: data is missing ${id}.${key}`);
  return f.value;
};

// The data the toy reads, from ctx.data: link speeds (each way), V4-Pro's expert counts and checkpoint, the GB300's HBM and FP4 peak.
export function setupFrom(data) {
  const hw = (id, key) => fact(data?.hardware, id, key);
  const model = (key) => fact(data?.models, 'deepseek-v4-pro', key);
  const net = (id) => ({ gbps: hw(id, 'gbps'), gbPerS: hw(id, 'gb_s_each_way') });
  const [n400, n800] = [net('network-400g'), net('network-800g')];
  const links = [
    { id: 'nvlink', label: `NVLink5 ${hw('b200', 'nvlink_gb_s_each_way')} GB/s`, gbPerS: hw('b200', 'nvlink_gb_s_each_way') },
    { id: 'net800', label: `network ${n800.gbps} Gb/s (${n800.gbPerS} GB/s)`, gbPerS: n800.gbPerS },
    { id: 'net400', label: `network ${n400.gbps} Gb/s (${n400.gbPerS} GB/s)`, gbPerS: n400.gbPerS },
  ].map((l) => ({ ...l, bytesPerSecond: l.gbPerS * GB }));
  const gb300 = data.hardware.entries.find((e) => e.id === 'gb300-nvl72');
  const hbm = hbmFor(gb300);
  return Object.freeze({
    links: Object.freeze(links),
    v4: Object.freeze({ expertsTotal: model('experts_total'), expertsActive: model('experts_active'), checkpointBytes: model('checkpoint_gb') * GB, dModel: model('d_model'), expertHidden: model('expert_hidden') }),
    gb300: Object.freeze({ hbmBytes: hbm.bytes, hbmBasis: hbm.basis, peakTflops: hw('gb300-nvl72', 'nvfp4_dense_tflops'), bandwidthTBps: hw('gb300-nvl72', 'hbm_tbps') }),
  });
}

const linkOf = (setup, id) => {
  if (!LINK_IDS.includes(id)) throw new RangeError(`disaggregation: link must be one of ${LINK_IDS.join(', ')}, got ${id}`);
  return setup.links.find((l) => l.id === id);
};
// The KV bytes per token of the running example (Llama-3.1-70B, BF16); FP8 stores half.
export const kvBytesPerToken = (kv) => (RUNNING_EXAMPLE.kvBytesPerToken * KV_FORMATS[kv].bytesPerElem) / BF16_BYTES;

// The prompt length at which a prefill's math takes as long as its weight read (the storyboard's 217 tokens): the crossing of
// stepTime's two times, found on real token counts (217.07) and printed whole.
export function prefillCrossover() {
  const at = (tokens) => stepTime({ ...RUNNING_EXAMPLE, tokens, seqs: 0, context: 0 });
  let [lo, hi] = [1, 4096];
  for (let i = 0; i < CROSSING_STEPS; i += 1) {
    const mid = (lo + hi) / 2;
    if (at(mid).computeS >= at(mid).memoryS) hi = mid; else lo = mid;
  }
  return Math.round(lo);
}

// Panel 1, "Ship the KV": what moves, how long it takes and what share of the prefill that is.
export function shipAnalysis({ prompt, link, kv }, setup) {
  if (!KV_FORMATS[kv]) throw new RangeError(`disaggregation: kv must be bf16 or fp8, got ${kv}`);
  const l = linkOf(setup, link);
  const bytesPerToken = kvBytesPerToken(kv);
  const prefillS = stepTime({ ...RUNNING_EXAMPLE, tokens: prompt, seqs: 0, context: 0 }).timeS;
  const transferS = kvTransferTime(prompt, bytesPerToken, l.bytesPerSecond);
  const mlaPerToken = kvBytesPerTokenMla({ ...MLA, bytesPerElem: KV_FORMATS[kv].bytesPerElem });
  return {
    link: l, bytesPerToken, prefillS, transferS,
    kvBytes: kvCacheBytes({ bytesPerToken, tokens: prompt }),
    mla: { bytesPerToken: mlaPerToken, kvBytes: kvCacheBytes({ bytesPerToken: mlaPerToken, tokens: prompt }), transferS: kvTransferTime(prompt, mlaPerToken, l.bytesPerSecond) },
    belowCrossover: prompt < prefillCrossover(),
  };
}

// Panel 2, "Feed the experts": V4-Pro's experts over `ep` GB300 GPUs with `users` per GPU.
export function expertAnalysis({ ep, users }, setup) {
  const { expertsTotal, expertsActive, checkpointBytes, dModel, expertHidden } = setup.v4;
  const { hbmBytes, peakTflops, bandwidthTBps } = setup.gb300;
  const tokens = tokensPerExpert({ usersPerGpu: users, epSize: ep, expertsActive, expertsTotal });
  const bytesPerElem = bytesPerElement('nvfp4');
  const roof = { peakTflops, bandwidthTBps };
  const freeBytes = freeHbmPerGpu({ hbmBytes, weightBytes: checkpointBytes, gpus: ep });
  return {
    expertsPerGpu: expertsTotal / ep, weightsBytes: checkpointBytes / ep, freeBytes, fits: freeBytes >= 0, tokens,
    intensity: arithmeticIntensity(matmulCost({ m: tokens, k: dModel, n: expertHidden, bytesPerElem })),
    ridge: ridgePoint(roof),
    tokensNeeded: tokensToComputeBound({ ...roof, bytesPerElem, k: dModel, n: expertHidden }),
  };
}

export const expertVerdict = (a) => (a.tokens >= a.tokensNeeded ? 'compute-bound' : 'memory-bound');

// The two formulas with this state's numbers substituted (storyboard §6; the default state is pinned by the page test).
export function checkWork(state, setup) {
  const a = shipAnalysis(state, setup);
  const e = expertAnalysis(state, setup);
  const { expertsActive, expertsTotal } = setup.v4;
  const gbPerS = formatCount(a.link.gbPerS);
  return [
    `transfer = prompt × KV bytes ÷ link = ${formatInt(state.prompt)} × ${formatInt(a.bytesPerToken)} B ÷ ${gbPerS} GB/s = ${formatBytes(a.kvBytes)} ÷ ${gbPerS} GB/s = ${formatDuration(a.transferS)}`,
    `tokens per expert = users × EP × k ÷ E = ${state.users} × ${state.ep} × ${expertsActive} ÷ ${expertsTotal} = ${formatCount(e.tokens)}`,
  ].join('\n');
}

export { formatRatio };
