// Training memory: bytes per parameter, ZeRO sharding, activations (training-memory §6). Units: bytes.
// Pure: no DOM, inputs never mutated. Owned by training-memory; parallelism imports zeroPerGpuBytes and TRAINING_RECIPES.
// Settled: math/memory.js stays KV and inference only (kvBytesPerToken lives there); this module imports nothing from it.
// FROZEN after Plan 3's shared prep (S3): change only through a shared patch.

import { deepFreeze } from './core.js';

// Bytes per parameter by part: BF16 weight and gradient, FP32 master copy, optimizer moments.
export const TRAINING_RECIPES = deepFreeze({
  adam: { weight: 2, grad: 2, master: 4, optimizer: 8 }, //            Adam m and v in FP32
  adamBf16Moments: { weight: 2, grad: 2, master: 4, optimizer: 4 }, // DeepSeek-V3-style BF16 moments
  muon: { weight: 2, grad: 2, master: 4, optimizer: 4 }, //            one FP32 momentum buffer
});

const PARTS = ['weight', 'grad', 'master', 'optimizer'];
const RECOMPUTE = ['none', 'selective', 'full'];

function requirePositive(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${fn}: ${name} must be a positive finite number, got ${value}`);
  }
}

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer, got ${value}`);
}

function requireRecipe(fn, recipe, prefix) {
  if (recipe === null || typeof recipe !== 'object') throw new RangeError(`${fn}: recipe must be an object of ${PARTS.join(', ')}`);
  for (const part of PARTS) {
    const v = recipe[part];
    if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) {
      throw new RangeError(`${fn}: ${prefix}${part} must be a finite number ≥ 0, got ${v}`);
    }
  }
}

// The one definition of "bytes per parameter": the parts plus their total.
export function trainingBytesPerParam(recipe) {
  requireRecipe('trainingBytesPerParam', recipe, '');
  const { weight, grad, master, optimizer } = recipe;
  return { weight, grad, master, optimizer, total: weight + grad + master + optimizer };
}

// Per-GPU state under ZeRO on a data-parallel group of `dp`. `recipe` is a TRAINING_RECIPES entry (an object, not its
// name); `optimizer` = master + moments. modelShards (= tp · pp) divides everything first. Stage 1 shards the optimizer
// state over dp, stage 2 also the gradients, stage 3 also the weights.
export function zeroPerGpuBytes({ params, recipe, stage, dp, modelShards = 1 }) {
  requirePositive('zeroPerGpuBytes', 'params', params);
  requireRecipe('zeroPerGpuBytes', recipe, 'recipe.');
  if (![0, 1, 2, 3].includes(stage)) throw new RangeError(`zeroPerGpuBytes: stage must be 0, 1, 2 or 3, got ${stage}`);
  requireCount('zeroPerGpuBytes', 'dp', dp);
  requireCount('zeroPerGpuBytes', 'modelShards', modelShards);
  const perShard = params / modelShards;
  const shard = (sharded) => (sharded ? perShard / dp : perShard);
  const weights = recipe.weight * shard(stage >= 3);
  const grads = recipe.grad * shard(stage >= 2);
  const optimizer = (recipe.master + recipe.optimizer) * shard(stage >= 1);
  return { weights, grads, optimizer, total: weights + grads + optimizer };
}

// Activations one transformer layer keeps for the backward pass (03 §3.2, Korthikanti et al.), BF16:
// 'none' s·b·h·(34 + 5·a·s/h) / tp · 'selective' 34·s·b·h / tp · 'full' 2·s·b·h (the block input only; no / tp, as specified).
export function activationBytesPerLayer({ seq, microBatch, hidden, heads, tp = 1, recompute = 'none' }) {
  requireCount('activationBytesPerLayer', 'seq', seq);
  requireCount('activationBytesPerLayer', 'microBatch', microBatch);
  requireCount('activationBytesPerLayer', 'hidden', hidden);
  requireCount('activationBytesPerLayer', 'heads', heads);
  requireCount('activationBytesPerLayer', 'tp', tp);
  if (!RECOMPUTE.includes(recompute)) throw new RangeError(`activationBytesPerLayer: recompute must be 'none', 'selective' or 'full', got ${recompute}`);
  const sbh = seq * microBatch * hidden;
  if (recompute === 'full') return 2 * sbh;
  if (recompute === 'selective') return (34 * sbh) / tp;
  return (sbh * (34 + (5 * heads * seq) / hidden)) / tp;
}

// layers × activationBytesPerLayer.
export function activationBytes({ layers, ...perLayerArgs }) {
  requireCount('activationBytes', 'layers', layers);
  return layers * activationBytesPerLayer(perLayerArgs);
}

// GPUs needed just to hold the state with any sharding (no activations): ceil(params · bytesPerParam / hbmBytes).
// hbmBytes is the chip's HBM: usable for B200, nominal otherwise (settled), labeled on the page.
export function gpusToHoldStates({ params, bytesPerParam, hbmBytes }) {
  requirePositive('gpusToHoldStates', 'params', params);
  requirePositive('gpusToHoldStates', 'bytesPerParam', bytesPerParam);
  requirePositive('gpusToHoldStates', 'hbmBytes', hbmBytes);
  return Math.ceil((params * bytesPerParam) / hbmBytes);
}
