// The four real shapes the page compares, read from data/models.json (pure). A missing fact makes a shape null,
// so a page test (and the toy's error card) names it instead of printing a made-up number.
import { lookupFact } from '@shared/claims.js';

export const SHAPE_SOURCES = Object.freeze([
  { key: 'gpt3', id: 'gpt-3', name: 'GPT-3', label: 'GPT-3' },
  { key: 'llama', id: 'llama-3.1-70b', name: 'Llama-3.1-70B', label: 'Llama-3.1-70B' },
  { key: 'minimax', id: 'minimax-m3', name: 'MiniMax-M3', label: 'MiniMax-M3' },
  { key: 'v3', id: 'deepseek-v3', name: 'DeepSeek-V3', label: 'DeepSeek-V3' },
]);

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const isCount = (v) => Number.isInteger(v) && v > 0;
const sum = (a, b) => (isCount(a) && isCount(b) ? a + b : null);

// V3's cached K and V per head are 128 wide (mla_nope_dim = mla_v_dim); its MHA baseline uses that width.
function readShape(data, { key, id, name, label }, mla) {
  const isV3 = key === 'v3';
  const kvHeads = isV3 ? null : fact(data, id, 'n_kv_heads');
  const shape = {
    key, id, name, label,
    layers: fact(data, id, 'layers'),
    queryHeads: fact(data, id, 'n_heads'),
    kvHeads,
    headDim: fact(data, id, isV3 ? 'mla_nope_dim' : 'head_dim'),
    contextLength: fact(data, id, 'context_length'),
    dLatent: isV3 ? fact(data, id, 'mla_kv_rank') : mla.dLatent,
    dRope: isV3 ? fact(data, id, 'mla_rope_dim') : mla.dRope,
    // V3 only: the real query/key head is nope + rope wide and its value head mla_v_dim wide (frame 8's 71× baseline).
    qkWidth: isV3 ? sum(fact(data, id, 'mla_nope_dim'), fact(data, id, 'mla_rope_dim')) : null,
    vWidth: isV3 ? fact(data, id, 'mla_v_dim') : null,
  };
  const ok = [shape.layers, shape.queryHeads, shape.headDim, shape.contextLength, shape.dLatent, shape.dRope].every(isCount) && (isV3 ? isCount(shape.qkWidth) && isCount(shape.vWidth) : isCount(kvHeads));
  if (!ok) return null;
  const scheme = isV3 ? 'mla' : kvHeads === shape.queryHeads ? 'mha' : kvHeads === 1 ? 'mqa' : 'gqa';
  return Object.freeze({ ...shape, scheme });
}

// → { gpt3, llama, minimax, v3 }, each a frozen shape or null.
export function modelShapes(data) {
  const mla = { dLatent: fact(data, 'deepseek-v3', 'mla_kv_rank'), dRope: fact(data, 'deepseek-v3', 'mla_rope_dim') };
  return Object.freeze(Object.fromEntries(SHAPE_SOURCES.map((source) => [source.key, readShape(data, source, mla)])));
}

// The same, or a RangeError naming the shapes whose facts are missing (a toy or a frame refuses to invent numbers).
export function requireShapes(data) {
  const shapes = modelShapes(data);
  const missing = SHAPE_SOURCES.filter((s) => shapes[s.key] === null).map((s) => s.id);
  if (missing.length) throw new RangeError(`kv-compression: data/models.json lacks the shape facts of ${missing.join(', ')}`);
  return shapes;
}
