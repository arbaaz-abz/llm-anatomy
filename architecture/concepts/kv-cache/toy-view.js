// kv-cache toy view model (pure, no DOM): state + data → every string the toy prints.
// Counts and bytes come from math/memory.js; the model shapes and GPU sizes from ctx.data.
import { formatBytes } from '@math/core.js';
import { kvBytesPerToken, kvBytesPerTokenMla, kvCacheBytes } from '@math/memory.js';
import { lookupFact } from '@shared/claims.js';
import {
  INITIAL_STATE, int, exactSpan, ratioText, compactSub, spanText, shareText, fitText, workFor, checkWorkA, checkWorkB,
} from './format.js';

export { INITIAL_STATE };

export const MODEL_CHIPS = Object.freeze([
  { value: 'toy', label: 'toy' },
  { value: 'gpt3', label: 'GPT-3' },
  { value: 'llama', label: 'Llama-3.1-70B' },
  { value: 'v3', label: 'DeepSeek-V3' },
  { value: 'v4pro', label: 'DeepSeek-V4-Pro (reported)' },
]);
const MODEL_IDS = Object.freeze({ gpt3: 'gpt-3', llama: 'llama-3.1-70b', v3: 'deepseek-v3', v4pro: 'deepseek-v4-pro' });
export const BYTES_PER_NUMBER = 2; // the cache is stored in 16-bit numbers (data note: "× 2 bytes")
export const GPUS = Object.freeze([
  { id: 'h100', key: 'h100', label: 'H100, 80 GB' },
  { id: 'b300', key: 'b300', label: 'B300, 288 GB (reported)' },
]);
export const V4_ESTIMATE_NOTE = 'formula-derived estimate; the layer mix is uncertain';
export const LATENT_NOTE = 'DeepSeek-V3 stores one latent per token instead of separate keys and values: how that works is in the MQA, GQA, MLA lesson.';

export function fact(data, id, key) {
  const found = lookupFact(data?.models, id, key);
  if (!found) throw new RangeError(`kv-cache toy: data/models.json has no ${id}.${key}`);
  return found.value;
}

// The shape a real chip snaps the shape sliders to (gpt3 and llama), or null when the chip does not use the sliders.
export function snapShape(model, data) {
  if (model !== 'gpt3' && model !== 'llama') return null;
  const id = MODEL_IDS[model];
  return { layers: fact(data, id, 'layers'), kvHeads: fact(data, id, 'n_kv_heads'), headDim: fact(data, id, 'head_dim'), bytes: BYTES_PER_NUMBER };
}

// The formula the active chip uses: 'mha' (any KV-head count), 'mla' (a latent) or 'range' (a reported estimate).
export function shapeOf(state, data) {
  if (state.model === 'toy') return { kind: 'mha', layers: state.layers, kvHeads: state.kvHeads, headDim: state.headDim, bytesPerElem: state.bytes };
  const id = MODEL_IDS[state.model];
  const context = fact(data, id, 'context_length');
  if (state.model === 'v4pro') return { kind: 'range', range: fact(data, id, 'kv_bytes_per_token'), bytesPerElem: BYTES_PER_NUMBER, context };
  if (state.model === 'v3') {
    return { kind: 'mla', layers: fact(data, id, 'layers'), dLatent: fact(data, id, 'mla_kv_rank'), dRope: fact(data, id, 'mla_rope_dim'), bytesPerElem: BYTES_PER_NUMBER, context };
  }
  const snap = snapShape(state.model, data);
  return { kind: 'mha', ...snap, bytesPerElem: snap.bytes, context };
}

// Bytes per token: a number, or [low, high] for the reported range.
export function bytesPerTokenOf(shape) {
  if (shape.kind === 'range') return shape.range;
  if (shape.kind === 'mla') return kvBytesPerTokenMla(shape);
  return kvBytesPerToken(shape);
}

const eachEnd = (value, fn) => (Array.isArray(value) ? value.map(fn) : fn(value));
const cacheOf = (bytesPerToken, tokens, sequences = 1) => eachEnd(bytesPerToken, (b) => kvCacheBytes({ bytesPerToken: b, tokens, sequences }));
const capacityOf = (data, key) => lookupFact(data?.hardware, key, 'hbm_gb')?.value * 1e9;

function gpuRow(totals, data) {
  return GPUS.map((gpu) => {
    const capacity = capacityOf(data, gpu.key);
    if (!(capacity > 0)) throw new RangeError(`kv-cache toy: data/hardware.json has no ${gpu.key}.hbm_gb`);
    const fills = [totals].flat().map((b) => Math.min(b / capacity, 1));
    return { ...gpu, capacity, share: spanText(totals, (b) => shareText(b, capacity)), fit: fitText(totals, capacity), fill: Math.min(...fills) };
  });
}

function contextNote(state, shape) {
  if (state.model === 'toy') return 'The toy has no context limit of its own.';
  const own = shape.context;
  return state.context > own
    ? `beyond this model's ${int(own)}-token context: a what-if at its shape`
    : `within this model's own ${int(own)}-token context`;
}

function workView(state) {
  const { current, flipped } = workFor(state);
  const off = state.cache ? flipped : current;
  const on = state.cache ? current : flipped;
  const cell = (n) => ({ value: int(n), sub: compactSub(n) });
  return {
    current: { positions: cell(current.positions), keyReads: cell(current.keyReads) },
    flipped: { positions: cell(flipped.positions), keyReads: cell(flipped.keyReads) },
    positionsRatio: ratioText(off.positions / on.positions),
    keyReadsRatio: ratioText(off.keyReads / on.keyReads),
  };
}

// The read-only shape lines for a real chip.
function shapeRows(shape) {
  if (shape.kind === 'range') return [];
  const rows = shape.kind === 'mla'
    ? [['Blocks', shape.layers], ['Latent size', shape.dLatent], ['RoPE key size', shape.dRope]]
    : [['Blocks', shape.layers], ['KV heads', shape.kvHeads], ['Head size', shape.headDim]];
  return [...rows.map(([label, value]) => ({ label, value: int(value) })), { label: 'Bytes per number', value: int(shape.bytesPerElem) }];
}

export function toyView(state, data) {
  const shape = shapeOf(state, data);
  const bytesPerToken = bytesPerTokenOf(shape);
  const one = cacheOf(bytesPerToken, state.context);
  const all = cacheOf(bytesPerToken, state.context, state.sequences);
  const gpus = gpuRow(all, data);
  return {
    shape,
    work: workView(state),
    checkA: checkWorkA(state),
    shapeRows: shapeRows(shape),
    bytesPerToken: { value: exactSpan(bytesPerToken), sub: Array.isArray(bytesPerToken) ? '' : (bytesPerToken >= 1000 ? formatBytes(bytesPerToken) : '') },
    cacheOne: { value: spanText(one, formatBytes), sub: exactSpan(one) },
    cacheAll: { value: spanText(all, formatBytes), sub: exactSpan(all) },
    readPerStep: { value: spanText(one, formatBytes), sub: 'the whole cache of one conversation' },
    gpus,
    contextNote: contextNote(state, shape),
    formulaNote: formulaNote(state.model),
    checkB: checkWorkB({ shape, bytesPerToken, tokens: state.context, sequences: state.sequences, total: all }),
  };
}

function formulaNote(model) {
  if (model === 'v3') return LATENT_NOTE;
  if (model === 'v4pro') return V4_ESTIMATE_NOTE;
  return '2 (K and V) × blocks × KV heads × head size × bytes per number';
}

// The three "try this" prompts, with every number computed from the same functions the toy uses.
export function tryThis(data) {
  const at = (model, extra = {}) => toyView({ ...INITIAL_STATE, model, context: 131_072, ...extra }, data);
  const [gpt3, llama, v3] = ['gpt3', 'llama', 'v3'].map((m) => at(m));
  const twoLlamas = at('llama', { sequences: 2 });
  const gpt3Eight = toyView({ ...INITIAL_STATE, model: 'gpt3', context: 2048, sequences: 8 }, data);
  const eightKv = toyView({ ...INITIAL_STATE, model: 'toy', ...snapShape('gpt3', data), kvHeads: 8, context: 131_072 }, data);
  const heavy = workFor({ prompt: 1000, reply: 1000, cache: true });
  const light = workFor({ prompt: 4, reply: 4, cache: true });
  const llamaShare = llama.gpus[0].share;
  return [
    {
      text: `Panel A, cache off, prompt 4, reply 4: ${int(light.flipped.positions)} positions. Cache on: ${int(light.current.positions)}. Now prompt 1,000 and reply 1,000: ${int(heavy.flipped.positions)} against ${int(heavy.current.positions)}, ${ratioText(heavy.flipped.positions / heavy.current.positions)} fewer. Read the second output: keys read are ${int(heavy.flipped.keyReads)} against ${int(heavy.current.keyReads)}, still almost two million.`,
      insight: 'the cache removes recomputation, not reading.',
      rest: ' Work per step stays flat; the read per step grows with every token in the conversation.',
    },
    {
      text: `Panel B, Llama-3.1-70B, tokens 131,072, conversations 1: ${llama.cacheOne.value}, ${llamaShare} of an H100. Set conversations to 2: ${twoLlamas.cacheAll.value} (${twoLlamas.gpus[0].share}), "${twoLlamas.gpus[0].fit}". Now GPT-3 at its own 2,048 tokens with 8 conversations (2,048 × 8 = 16,384 token-positions): ${gpt3Eight.cacheAll.value}, ${gpt3Eight.gpus[0].share}.`,
      insight: 'the cache, not the arithmetic, sets how many conversations one GPU can serve,',
      rest: ' and it grows with context × conversations.',
    },
    {
      text: `Panel B, tokens 131,072: tap GPT-3, Llama-3.1-70B, then DeepSeek-V3: ${formatBytes(bytesPerTokenOf(gpt3.shape))}, ${formatBytes(bytesPerTokenOf(llama.shape))}, then ${formatBytes(bytesPerTokenOf(v3.shape))} per token (${gpt3.cacheOne.value}, ${llama.cacheOne.value}, then ${v3.cacheOne.value} per conversation). GPT-3 could never hold 131,072 tokens; its ${gpt3.cacheOne.value} is its cache per token scaled to Llama's context, and the what-if label says so. Now tap GPT-3, then the toy chip (it keeps GPT-3's shape, now editable) and set KV heads from 96 to 8: ${formatBytes(bytesPerTokenOf(gpt3.shape))} becomes ${formatBytes(bytesPerTokenOf(eightKv.shape))}.`,
      insight: 'the formula multiplies KV heads, not query heads, so storing fewer key/value sets is the biggest lever.',
      rest: ' How models do that without losing quality: [[kv-compression]].',
    },
  ];
}

