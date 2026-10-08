// KV-cache and inference memory (kv-cache §6). Pure: no DOM, inputs never mutated.
// FROZEN in Plan 2 wave 0: kvBytesPerToken, kvBytesPerTokenMla and sharePct keep these exact signatures.
// Further functions are added only by a shared patch (Task 15). S1 added kvCacheBytes and decodeWork (kv-cache §6),
// kvGroups (kv-compression §6), stackKvBytes and linearStateBytes (long-context-attention §6).
// Training memory lives in math/training-memory.js (Training plan), never here.

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer`);
}

function requireSize(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) throw new RangeError(`${fn}: ${name} must be a positive finite number`);
}

// Bytes stored per token by every layer that keeps K and V per KV head: 2 · L · n_kv · d_head · b.
// MHA, GQA and MQA are the same formula with kvHeads = heads, groups, or 1.
export function kvBytesPerToken({ layers, kvHeads, headDim, bytesPerElem }) {
  requireCount('kvBytesPerToken', 'layers', layers);
  requireCount('kvBytesPerToken', 'kvHeads', kvHeads);
  requireCount('kvBytesPerToken', 'headDim', headDim);
  requireSize('kvBytesPerToken', 'bytesPerElem', bytesPerElem);
  return 2 * layers * kvHeads * headDim * bytesPerElem;
}

// Multi-head latent attention: one latent plus one small RoPE key per layer, L · (d_c + d_rope) · b.
// No factor of 2: K and V are both rebuilt from the same stored latent.
export function kvBytesPerTokenMla({ layers, dLatent, dRope, bytesPerElem }) {
  requireCount('kvBytesPerTokenMla', 'layers', layers);
  requireCount('kvBytesPerTokenMla', 'dLatent', dLatent);
  requireCount('kvBytesPerTokenMla', 'dRope', dRope);
  requireSize('kvBytesPerTokenMla', 'bytesPerElem', bytesPerElem);
  return layers * (dLatent + dRope) * bytesPerElem;
}

// Share of a whole as a percentage: the one definition of "share of a GPU", "share of a context" and a
// share bar's printed label (README lesson 16). One decimal unless asked for more.
export function sharePct(part, whole, { decimals = 1 } = {}) {
  if (typeof whole !== 'number' || !Number.isFinite(whole) || whole <= 0) throw new RangeError(`sharePct: whole must be a finite number > 0, got ${whole}`);
  if (typeof part !== 'number' || !Number.isFinite(part)) throw new RangeError(`sharePct: part must be a finite number, got ${part}`);
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 6) throw new RangeError(`sharePct: decimals must be an integer 0–6, got ${decimals}`);
  const scale = 10 ** decimals;
  return Math.round((part / whole) * 100 * scale) / scale;
}

// ---- Shared patch S1 ----

// Total cache for `sequences` conversations of `tokens` each (kv-cache §6): the one definition of "cache for N tokens".
export function kvCacheBytes({ bytesPerToken, tokens, sequences = 1 }) {
  requireSize('kvCacheBytes', 'bytesPerToken', bytesPerToken);
  requireCount('kvCacheBytes', 'tokens', tokens);
  requireCount('kvCacheBytes', 'sequences', sequences);
  return bytesPerToken * tokens * sequences;
}

// Work to generate `generated` tokens after a `prompt`-token prompt; the first pass is the prefill.
// positions: positions run through the blocks; keyReads: query–key scores per head per layer.
// cache = false reruns all t positions on every pass (t positions, t(t+1)/2 reads);
// cache = true runs the prompt once, then one position per pass (1 position, t reads at length t).
export function decodeWork({ prompt, generated, cache }) {
  requireCount('decodeWork', 'prompt', prompt);
  requireCount('decodeWork', 'generated', generated);
  if (typeof cache !== 'boolean') throw new RangeError(`decodeWork: cache must be true or false, got ${cache}`);
  const passes = Array.from({ length: generated }, (_, i) => prompt + i);
  return passes.reduce((acc, t, i) => (!cache || i === 0
    ? { positions: acc.positions + t, keyReads: acc.keyReads + (t * (t + 1)) / 2 }
    : { positions: acc.positions + 1, keyReads: acc.keyReads + t }), { positions: 0, keyReads: 0 });
}

// Which KV head each query head reads (0-based; drawn 1-based as Q1…Q8).
export function kvGroups({ queryHeads, kvHeads }) {
  requireCount('kvGroups', 'queryHeads', queryHeads);
  requireCount('kvGroups', 'kvHeads', kvHeads);
  if (queryHeads % kvHeads !== 0) throw new RangeError('kvGroups: kvHeads must divide queryHeads');
  const perGroup = queryHeads / kvHeads;
  return Array.from({ length: queryHeads }, (_, h) => Math.floor(h / perGroup));
}

const STACK_KINDS = ['full', 'window', 'compressed', 'linear'];

function checkGroup(group, i) {
  const fn = 'stackKvBytes';
  const name = (field) => `group ${i} ${field}`;
  if (!STACK_KINDS.includes(group?.kind)) throw new RangeError(`${fn}: group ${i} kind must be one of ${STACK_KINDS.join(', ')}, got ${group?.kind}`);
  requireCount(fn, name('layers'), group.layers);
  if (group.kind === 'full' || group.kind === 'window') requireSize(fn, name('bytesPerTokenPerLayer'), group.bytesPerTokenPerLayer);
  if (group.kind === 'window') requireCount(fn, name('window'), group.window);
  if (group.kind === 'compressed') {
    requireCount(fn, name('merge'), group.merge);
    requireSize(fn, name('bytesPerEntry'), group.bytesPerEntry);
  }
  if (group.kind === 'linear' && group.stateBytes !== null && !(typeof group.stateBytes === 'number' && Number.isFinite(group.stateBytes) && group.stateBytes > 0)) {
    throw new RangeError(`${fn}: group ${i} stateBytes must be a positive finite number or null, got ${group.stateBytes}`);
  }
}

// Per-token growth and fixed bytes of one group. A window stores min(tokens, window) entries (ruling S1-R2);
// a compressed layer stores tokens / merge entries (exact division, brief 04 §8.2); null = size not published.
function groupBytes(group, tokens) {
  if (group.kind === 'full') return { perToken: group.layers * group.bytesPerTokenPerLayer, fixed: 0 };
  if (group.kind === 'window') return { perToken: 0, fixed: group.layers * Math.min(tokens, group.window) * group.bytesPerTokenPerLayer };
  if (group.kind === 'compressed') return { perToken: (group.layers * group.bytesPerEntry) / group.merge, fixed: 0 };
  return { perToken: 0, fixed: group.stateBytes === null ? null : group.layers * group.stateBytes };
}

// Whole-model cache for a stack of layer groups: full, window, compressed and linear (long-context-attention §6).
// An unknown fixed part makes `fixed` and `total` null (ruling S1-R1); print the growing part with kvCacheBytes.
export function stackKvBytes({ groups, tokens }) {
  if (!Array.isArray(groups) || groups.length === 0) throw new RangeError('stackKvBytes: groups must be a non-empty array');
  groups.forEach(checkGroup);
  requireCount('stackKvBytes', 'tokens', tokens);
  const parts = groups.map((g) => groupBytes(g, tokens));
  const perToken = parts.reduce((acc, p) => acc + p.perToken, 0);
  const fixed = parts.some((p) => p.fixed === null) ? null : parts.reduce((acc, p) => acc + p.fixed, 0);
  return { perToken, fixed, total: fixed === null ? null : perToken * tokens + fixed };
}

// Fixed state of linear-attention layers: layers · heads · dKey · dValue · b.
export function linearStateBytes({ layers, heads, dKey, dValue, bytesPerElem }) {
  [['layers', layers], ['heads', heads], ['dKey', dKey], ['dValue', dValue]].forEach(([n, v]) => requireCount('linearStateBytes', n, v));
  requireSize('linearStateBytes', 'bytesPerElem', bytesPerElem);
  return layers * heads * dKey * dValue * bytesPerElem;
}
