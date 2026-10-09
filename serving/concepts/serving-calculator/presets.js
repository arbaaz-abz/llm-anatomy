// serving-calculator presets (pure, no DOM): data/*.json → the model, GPU, price, target and cache presets the toy reads.
// The one place the page reads model and chip numbers; the toy, the try-this list and the tests all go through it.
// Conventions (builder notes §2): dense peaks only; HBM capacity through hbmFor with its basis word; V4-Pro's KV per token
// is a reported range, low first.
import { lookupFact } from '@shared/claims.js';
import { hbmFor, weightBytes } from '@math/serving.js';
import { kvBytesPerToken } from '@math/memory.js';
import { bitsPerElement } from '@math/roofline.js';
import { formatBytes } from '@math/core.js';

export const V4_ID = 'deepseek-v4-pro';
export const LLAMA_ID = 'llama-3.1-70b';
export const MODEL_LABELS = Object.freeze({ [V4_ID]: 'DeepSeek-V4-Pro', [LLAMA_ID]: 'Llama-3.1-70B' });

// The weight-format control: each maps to an exact FORMATS key (never a bare 'fp8'); 'shipped' is V4-Pro's reported checkpoint.
export const WEIGHT_FORMATS = Object.freeze({
  shipped: Object.freeze({ label: 'As shipped (FP4 experts + FP8)', format: null, mathRate: 'fp8', actBytesPerElem: 1 }),
  bf16: Object.freeze({ label: 'BF16', format: 'bf16', mathRate: 'bf16', actBytesPerElem: 2 }),
  fp8: Object.freeze({ label: 'FP8', format: 'fp8_e4m3', mathRate: 'fp8', actBytesPerElem: 1 }),
  nvfp4: Object.freeze({ label: 'NVFP4', format: 'nvfp4', mathRate: 'nvfp4', actBytesPerElem: 1 }),
});
export const NO_FP4 = 'no FP4 figure in data';

// The GPUs, in the storyboard's order (§6 `hw`).
export const GPUS = Object.freeze([
  { id: 'h200', label: 'H200' },
  { id: 'b200', label: 'B200' },
  { id: 'gb200-nvl72', label: 'GB200 NVL72' },
  { id: 'gb300-nvl72', label: 'GB300 NVL72' },
].map(Object.freeze));

// The GPU counts per replica (§6 `gpus`).
export const GPU_COUNTS = Object.freeze([1, 2, 4, 8, 16, 32, 72]);

const value = (dataset, id, key) => lookupFact(dataset, id, key)?.value ?? null;
const positive = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;

function need(v, what) {
  if (!positive(v)) throw new RangeError(`presets: data has no ${what}`);
  return v;
}

// One model's numbers. V4-Pro: `kvBytes` is { low, high } (the reported range); Llama: bytes per token follow the cache format.
export function modelPreset(data, id) {
  if (!Object.hasOwn(MODEL_LABELS, id)) throw new RangeError(`modelPreset: model must be one of ${Object.keys(MODEL_LABELS).join(', ')}, got ${id}`);
  const get = (key) => value(data?.models, id, key);
  const base = { id, label: MODEL_LABELS[id], params: need(get('total_params'), `${id}.total_params`), dModel: need(get('d_model'), `${id}.d_model`) };
  if (id === V4_ID) {
    const range = get('kv_bytes_per_token');
    if (!Array.isArray(range) || range.length !== 2) throw new RangeError('presets: data has no deepseek-v4-pro.kv_bytes_per_token range');
    return Object.freeze({
      ...base, dense: false, activeParams: need(get('active_params'), `${id}.active_params`),
      shippedBytes: need(get('checkpoint_gb'), `${id}.checkpoint_gb`) * 1e9,
      kvBytes: Object.freeze({ low: range[0], high: range[1] }), contextLength: need(get('context_length'), `${id}.context_length`),
    });
  }
  const shape = { layers: need(get('layers'), `${id}.layers`), kvHeads: need(get('n_kv_heads'), `${id}.n_kv_heads`), headDim: need(get('head_dim'), `${id}.head_dim`) };
  return Object.freeze({
    ...base, dense: true, activeParams: base.params, shippedBytes: null, contextLength: need(get('context_length'), `${id}.context_length`),
    kvBytes: Object.freeze({ bf16: kvBytesPerToken({ ...shape, bytesPerElem: 2 }), fp8: kvBytesPerToken({ ...shape, bytesPerElem: 1 }) }),
  });
}

// KV bytes per token for a state's model: V4-Pro by the KV end (low / high), Llama by the cache format.
export const kvBytesFor = (model, { kvEnd, kvFormat }) => (model.dense ? model.kvBytes[kvFormat] : model.kvBytes[kvEnd]);

// Weights in bytes for a format: the reported checkpoint for "as shipped", else params · bits / 8 (bitsPerElement counts the block scales).
export function weightsFor(model, weights) {
  const w = WEIGHT_FORMATS[weights];
  if (!w) throw new RangeError(`weightsFor: weights must be one of ${Object.keys(WEIGHT_FORMATS).join(', ')}, got ${weights}`);
  if (w.format === null) {
    if (model.shippedBytes === null) throw new RangeError(`weightsFor: ${model.id} has no "as shipped" checkpoint`);
    return model.shippedBytes;
  }
  return weightBytes({ params: model.params, bitsPerParam: bitsPerElement(w.format) });
}

const hbmWords = { usable: 'usable', nominal: 'nominal' };

// One GPU's numbers: dense peaks per math rate (null where data has none), HBM bandwidth, HBM through hbmFor with its basis word.
export function gpuPreset(data, id) {
  const known = GPUS.find((g) => g.id === id);
  if (!known) throw new RangeError(`gpuPreset: gpu must be one of ${GPUS.map((g) => g.id).join(', ')}, got ${id}`);
  const entry = data?.hardware?.entries?.find((e) => e.id === id);
  if (!entry) throw new RangeError(`gpuPreset: data/hardware.json has no ${id}`);
  const get = (key) => value(data.hardware, id, key);
  const peak = { bf16: get('bf16_dense_tflops'), fp8: get('fp8_e4m3_dense_tflops'), nvfp4: get('nvfp4_dense_tflops') };
  const hbm = hbmFor(entry);
  const nominalGb = get('hbm_gb');
  return Object.freeze({
    id, label: known.label, peak: Object.freeze(Object.fromEntries(Object.entries(peak).map(([k, v]) => [k, positive(v) ? v : null]))),
    bandwidthTBps: need(get('hbm_tbps'), `${id}.hbm_tbps`), hbmBytes: hbm.bytes, hbmBasis: hbm.basis, nominalBytes: need(nominalGb, `${id}.hbm_gb`) * 1e9,
  });
}

// "288 GB nominal" · "180 GB usable (192 nominal)" · "186 GB nominal (the rack total over 72 GPUs)": never a bare "GB" for chip memory.
export function hbmText(gpu) {
  const base = `${formatBytes(gpu.hbmBytes)} ${hbmWords[gpu.hbmBasis]}`;
  if (gpu.hbmBasis === 'usable') return `${base} (${formatBytes(gpu.nominalBytes).replace(' GB', '')} nominal)`;
  return gpu.id === 'gb200-nvl72' ? `${base} (the rack total over 72 GPUs)` : base;
}

// Whether the weight format can run on the GPU (NVFP4 needs an FP4 peak): the toy disables it with NO_FP4 (P3-R12).
export const supportsWeights = (gpu, weights) => gpu.peak[WEIGHT_FORMATS[weights].mathRate] !== null;

export const mathPeak = (gpu, weights) => {
  const peak = gpu.peak[WEIGHT_FORMATS[weights].mathRate];
  if (peak === null) throw new RangeError(`mathPeak: ${gpu.id} has no ${WEIGHT_FORMATS[weights].mathRate} figure in data`);
  return peak;
};

// The scenario presets that are not model-specific: operating points, GPU-hour prices and the cache hit rate (serving.json).
export function scenarioPresets(data) {
  const sv = (id, key) => need(value(data?.serving, id, key), `${id}.${key}`);
  return Object.freeze({
    targets: Object.freeze([sv('inferencex-v4-pro-gb300', 'max_throughput_tok_s_user'), sv('inferencex-v4-pro-gb300', 'interactivity_tok_s_user')]),
    prices: Object.freeze([
      { usd: sv('inferencex-v4-pro-gb300', 'gpu_hour_usd'), name: 'GB300' },
      { usd: sv('inferencex-v4-pro-gb200', 'gpu_hour_usd'), name: 'GB200' },
      { usd: sv('deepseek-v3-production', 'gpu_hour_usd'), name: 'DeepSeek H800' },
    ].map(Object.freeze)),
    hitRate: sv('deepseek-v3-production', 'kv_hit_rate_pct') / 100,
    mtpAlpha: need(Math.min(...[].concat(value(data?.serving, 'deepseek-v3-mtp', 'acceptance_pct') ?? [])), 'deepseek-v3-mtp.acceptance_pct') / 100, // the low end of 85–90%
    mtpRange: Object.freeze([].concat(value(data?.serving, 'deepseek-v3-mtp', 'acceptance_pct') ?? [])),
    measured: Object.freeze({
      gb300: Object.freeze({ tokSGpu: sv('inferencex-v4-pro-gb300', 'throughput_tok_s_gpu'), usd: sv('inferencex-v4-pro-gb300', 'gpu_hour_usd'), publishedPerM: sv('inferencex-v4-pro-gb300', 'cost_per_m') }),
      gb200: Object.freeze({ tokSGpu: sv('inferencex-v4-pro-gb200', 'throughput_tok_s_gpu'), usd: sv('inferencex-v4-pro-gb200', 'gpu_hour_usd'), publishedPerM: sv('inferencex-v4-pro-gb200', 'cost_per_m') }),
    }),
    lmsysGainPct: sv('lmsys-gb300-longctx', 'mtp_per_user_gain_pct'),
  });
}
