// KV-cache and inference memory (kv-cache §6). Pure: no DOM, inputs never mutated.
// FROZEN in Plan 2 wave 0: kvBytesPerToken, kvBytesPerTokenMla and sharePct keep these exact signatures.
// Further functions (kvCacheBytes, decodeWork, kvGroups, …) are added only by a shared patch (Task 15).
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
