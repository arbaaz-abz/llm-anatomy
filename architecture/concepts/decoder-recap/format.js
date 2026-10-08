// decoder-recap pure helpers (no DOM): the toy's six switches → a math/params.js config, and every string the toy prints.
// Counts come from paramBreakdown (math/params.js), cache sizes from kvBytesPerToken (math/memory.js).
import { paramBreakdown, publishedGap, PRESETS } from '@math/params.js';
import { kvBytesPerToken, sharePct } from '@math/memory.js';
import { formatCount, formatBytes } from '@math/core.js';

export const int = (n) => (n < 0 ? '−' : '') + Math.abs(n).toLocaleString('en-US');
// "+1,573,864", "−2,371,584", "0": a change with a real minus sign.
export const signedInt = (n) => (n === 0 ? '0' : `${n < 0 ? '−' : '+'}${int(Math.abs(n))}`);

const GPT3 = PRESETS.gpt3;
const GELU_HIDDEN = GPT3.mlp.hidden; // 4 × d_model
const SWIGLU_HIDDEN = (GELU_HIDDEN * 2) / 3; // 8/3 × d_model: three matrices cost what two did
export const EXPERT = Object.freeze({ routed: 64, shared: 0, topK: 8, hidden: SWIGLU_HIDDEN / 8, denseLayers: 0 }); // each an eighth of the SwiGLU width
const BYTES_PER_NUMBER = 2;

export const OPTIONS = Object.freeze({
  norm: Object.freeze([{ value: 'layernorm', label: 'LayerNorm' }, { value: 'rmsnorm', label: 'RMSNorm' }]),
  position: Object.freeze([{ value: 'learned', label: `learned table (${int(GPT3.maxPositions)})` }, { value: 'rope', label: 'RoPE' }]),
  mlp: Object.freeze([{ value: 'gelu', label: 'GELU, 4 × d' }, { value: 'swiglu', label: 'SwiGLU, 8/3 × d' }]),
  biases: Object.freeze([{ value: 'on', label: 'on' }, { value: 'off', label: 'off' }]),
  kvHeads: Object.freeze([{ value: 96, label: '96 (GPT-3)' }, { value: 8, label: '8' }, { value: 1, label: '1' }]),
  experts: Object.freeze([{ value: 'dense', label: 'dense' }, { value: 'moe', label: `${EXPERT.routed} experts, top-${EXPERT.topK}` }]),
});

export const INITIAL_STATE = Object.freeze({ norm: 'layernorm', position: 'learned', mlp: 'gelu', biases: 'on', kvHeads: 96, experts: 'dense' });
export const MODERN_STATE = Object.freeze({ norm: 'rmsnorm', position: 'rope', mlp: 'swiglu', biases: 'off', kvHeads: 8, experts: 'moe' });

// One switch flipped. Experts are SwiGLU experts, so turning them on flips the MLP, and a GELU MLP turns them off.
export function applySwitch(state, key, value) {
  if (!OPTIONS[key]) throw new RangeError(`applySwitch: unknown switch "${key}"`);
  if (!OPTIONS[key].some((o) => o.value === value)) throw new RangeError(`applySwitch: ${key} cannot be ${value}`);
  const next = { ...state, [key]: value };
  if (key === 'experts' && value === 'moe') next.mlp = 'swiglu';
  if (key === 'mlp' && value === 'gelu') next.experts = 'dense';
  return next;
}

export function configFor(state) {
  Object.keys(OPTIONS).forEach((key) => applySwitch(state, key, state[key]));
  if (state.experts === 'moe' && state.mlp !== 'swiglu') throw new RangeError('configFor: experts need a SwiGLU MLP');
  return {
    ...GPT3,
    norm: state.norm,
    positional: state.position,
    biases: state.biases === 'on',
    attention: { ...GPT3.attention, nKvHeads: state.kvHeads },
    mlp: { kind: state.mlp, hidden: state.mlp === 'gelu' ? GELU_HIDDEN : SWIGLU_HIDDEN },
    moe: state.experts === 'moe' ? { ...EXPERT } : null,
  };
}

export const breakdownFor = (state) => paramBreakdown(configFor(state));
export const cacheBytes = (state) => kvBytesPerToken({ layers: GPT3.layers, kvHeads: state.kvHeads, headDim: GPT3.attention.dHead, bytesPerElem: BYTES_PER_NUMBER });

// A signed relative gap as a percentage with a real minus: two decimals, four when the change is tiny ("−0.0014%").
export function signedPct(gap) {
  const decimals = Math.abs(gap) * 100 >= 0.01 ? 2 : 4;
  const pct = sharePct(gap, 1, { decimals });
  return pct === 0 ? '0.00%' : `${pct < 0 ? '−' : '+'}${Math.abs(pct).toFixed(decimals)}%`;
}

const PART_ROWS = Object.freeze([
  ['attention', 'attention'], ['mlp', 'MLP (dense)'], ['experts', 'experts'], ['router', 'router'], ['norms', 'norms'],
  ['positional', 'position table'], ['embedding', 'embedding (tied with the unembedding)'],
]);

export const ACTIVE_DEFINITION = 'Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication. (If the table is shared with the unembedding, as in GPT-3, it is counted once.)';

const LONGEST = Object.freeze({ learned: `${int(GPT3.maxPositions)} (table size)`, rope: 'set by training (see the RoPE lesson)' });
const ratioText = (n) => String(Number(n.toPrecision(3)));

// Every string the toy prints for one state (pure; the toy only paints it).
export function toyView(state) {
  const b = breakdownFor(state);
  const base = breakdownFor(INITIAL_STATE);
  const bytes = cacheBytes(state);
  const baseBytes = cacheBytes(INITIAL_STATE);
  const delta = b.total - base.total;
  return {
    total: int(b.total),
    totalSub: formatCount(b.total),
    active: int(b.active),
    activeSub: `${formatCount(b.active)}, ${sharePct(b.active, b.total)}% of total`,
    change: signedInt(delta),
    changeSub: signedPct(publishedGap(b.total, base.total)),
    cache: `${int(bytes)} B`,
    cacheSub: bytes === baseBytes ? formatBytes(bytes) : `${formatBytes(bytes)} (${ratioText(baseBytes / bytes)}× less than GPT-3)`,
    longest: LONGEST[state.position],
    parts: PART_ROWS.map(([part, label]) => ({ part, label, count: int(b.parts[part]), delta: signedInt(b.parts[part] - base.parts[part]) })),
  };
}

// Storyboard §6 "Try this": [prompt, insight, rest], every count read back from the same functions the toy uses.
const exact = (state) => breakdownFor(state);
const without = (state, key, value) => ({ ...state, [key]: value });
export function tryThis() {
  const start = exact(INITIAL_STATE);
  const rms = exact(without(INITIAL_STATE, 'norm', 'rmsnorm'));
  const glu = exact(without(INITIAL_STATE, 'mlp', 'swiglu'));
  const allButExperts = without(MODERN_STATE, 'experts', 'dense');
  const dense = exact(allButExperts);
  const moe = exact(MODERN_STATE);
  const rope = exact(without(INITIAL_STATE, 'position', 'rope'));
  const kvBefore = cacheBytes(INITIAL_STATE);
  const kvAfter = cacheBytes(without(INITIAL_STATE, 'kvHeads', 8));
  return [
    {
      prompt: `Flip norm: total ${int(start.total)} → ${int(rms.total)} (${int(start.total - rms.total)} fewer: the β vectors). Flip MLP: the MLP parts go ${int(start.parts.mlp)} → ${int(glu.parts.mlp)} (the bias vectors differ; without biases both are ${GPT3.layers} × ${int(GPT3.dModel * 2 * GELU_HIDDEN)}).`,
      insight: 'RMSNorm and SwiGLU change how the numbers flow, not how many there are; they won on stability and quality per FLOP, not size.',
      rest: '',
    },
    {
      prompt: `Flip position to RoPE: ${int(start.total - rope.total)} fewer parameters and "longest input: set by training". Set KV heads to 8: cache ${int(kvBefore)} → ${int(kvAfter)} B per token (${ratioText(kvBefore / kvAfter)}× less). With the other swaps on, the total falls to ${int(dense.total)} because W_K and W_V shrink.`,
      insight: `the big 2023–26 changes are about running cost: no position limit, a ${ratioText(kvBefore / kvAfter)}× smaller cache.`,
      rest: ' [[kv-cache]] lets you build this number by hand; [[kv-compression]] shows how the sharing works.',
    },
    {
      prompt: `With every other swap on, turn experts on: total ${int(dense.total)} → ${int(moe.total)}; active ${int(dense.active)} → ${int(moe.active)} (${signedInt(moe.active - dense.active)}: the router). Tap 2026-style, then GPT-3 (2020), and read the parts table.`,
      insight: "experts multiply what is stored, not what each token uses, and every other swap kept the block's shape.",
      rest: '',
    },
  ];
}

// ---- stage number formats (real minus, leading zero) ----
// Weights and scores at 3 d.p., zero included ("0.000"); a value that rounds to zero never prints a minus.
export const fixed3 = (v) => (Number(v.toFixed(3)) === 0 ? '0.000' : `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(3)}`);
// Normalized rows at 3 d.p., an exact zero as "0" (the zeros RMSNorm keeps at zero).
export const norm3 = (v) => (v === 0 ? '0' : fixed3(v));
// A short list as the storyboard prints it: [−0.5, 1.5, 0.25]. `digits` fixes the decimals; without it numbers print as they are.
export const listText = (values, digits) => `[${values.map((v) => (digits == null ? String(Number(v.toFixed(4))) : fixed3Digits(v, digits))).join(', ')}]`.replace(/-/g, '−');
const fixed3Digits = (v, digits) => (Number(v.toFixed(digits)) === 0 ? (0).toFixed(digits) : v.toFixed(digits));

// The shape every switch applies to, as the toy's intro prints it.
export const SHAPE_TEXT = `${GPT3.layers} blocks, d_model ${int(GPT3.dModel)}, ${GPT3.attention.nHeads} query heads × ${GPT3.attention.dHead}, vocabulary ${int(GPT3.vocab)}, tied embeddings`;
