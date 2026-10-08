// The four ways to wire query heads to stored keys and values, as numbers (pure; every count comes from @math/memory.js).
import { kvBytesPerToken, kvBytesPerTokenMla, kvGroups } from '@math/memory.js';

export const SCHEMES = Object.freeze(['mha', 'gqa', 'mqa', 'mla']);
export const SCHEME_LABELS = Object.freeze({ mha: 'MHA', gqa: 'GQA', mqa: 'MQA', mla: 'MLA' });

// config: { scheme, queryHeads, kvHeads (GQA only), headDim, dLatent, dRope }
export function storedHeads({ scheme, queryHeads, kvHeads }) {
  if (scheme === 'mha') return queryHeads;
  if (scheme === 'mqa') return 1;
  if (scheme === 'gqa') return kvHeads;
  throw new RangeError(`storedHeads: ${scheme} stores a latent, not KV heads`);
}

// Bytes per token over `layers` layers at `bytesPerElem` bytes a number; layers 1 and bytesPerElem 1 count numbers per layer.
export function bytesPerToken(config, { layers, bytesPerElem }) {
  if (config.scheme === 'mla') return kvBytesPerTokenMla({ layers, dLatent: config.dLatent, dRope: config.dRope, bytesPerElem });
  return kvBytesPerToken({ layers, kvHeads: storedHeads(config), headDim: config.headDim, bytesPerElem });
}

export const numbersPerLayer = (config) => bytesPerToken(config, { layers: 1, bytesPerElem: 1 });

// "Times smaller than MHA": the MHA bytes at the same shape divided by this scheme's bytes (README lesson 16).
export function timesSmaller(config, { layers, bytesPerElem }) {
  const mha = bytesPerToken({ ...config, scheme: 'mha' }, { layers, bytesPerElem });
  return mha / bytesPerToken(config, { layers, bytesPerElem });
}

// For a drawing: the KV head each query head reads (0-based). MLA and MQA read one thing; MHA reads its own.
export function wiring(config) {
  const kvHeads = config.scheme === 'mla' ? 1 : storedHeads(config);
  return kvGroups({ queryHeads: config.queryHeads, kvHeads });
}

// The stored thing a scheme keeps, for a sentence: "8 KV heads", "1 KV head", "1 latent".
export function storedNoun(config) {
  if (config.scheme === 'mla') return '1 latent per token';
  const n = storedHeads(config);
  return `${n} KV ${n === 1 ? 'head' : 'heads'}`;
}
