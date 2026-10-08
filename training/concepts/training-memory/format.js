// training-memory formatters and "Check my work" (pure, no DOM; in the coverage gate). Memory in GB prints with thousands
// separators and up to two decimals, the storyboard's format; sizes below 1 GB and parameter counts use formatBytes / formatCount.
import { formatBytes, formatCount, formatRatio } from '@math/core.js';
import { TRAINING_RECIPES, trainingBytesPerParam, zeroPerGpuBytes, activationBytesPerLayer, activationBytes } from '@math/training-memory.js';
import { GPT3 } from './numbers.js';

const GB = 1e9;
const group = (n, decimals) => n.toLocaleString('en-US', { maximumFractionDigits: decimals });

export const gbText = (bytes) => `${group(bytes / GB, 2)} GB`;
// "4.4 H100s", "35 H100s": a fractional GPU count for the counter on frames 1–4 (one decimal, zeros dropped).
export const gpuCountText = (x) => group(x, 1);

export const RECIPES = Object.freeze([
  { value: 'adam', label: 'Adam, FP32 states (16 B)' },
  { value: 'adamBf16Moments', label: 'Adam, BF16 moments (12 B)' },
  { value: 'muon', label: 'Muon, one FP32 momentum (12 B)' },
]);
export const STAGES = Object.freeze([
  { value: 0, label: '0 (plain data parallel)' }, { value: 1, label: '1' }, { value: 2, label: '2' }, { value: 3, label: '3 (FSDP)' },
]);
export const RECOMPUTE = Object.freeze([
  { value: 'none', label: 'store everything' },
  { value: 'selective', label: 'skip attention scores (selective)' },
  { value: 'full', label: 'block inputs only (full)' },
]);
export const TOY_LIMITS = Object.freeze({
  dp: Object.freeze([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
  seq: Object.freeze([1024, 2048, 4096, 8192]),
});
// Model presets: parameter counts come from data/models.json (MoE presets use TOTAL parameters). Only GPT-3 has activations.
export const MODELS = Object.freeze([
  { value: 'zero-paper', label: 'ZeRO paper 7.5B', dataId: 'zero-paper-7.5b' },
  { value: 'gpt-3', label: 'GPT-3 175B', dataId: 'gpt-3' },
  { value: 'llama-3.1-405b', label: 'Llama 3.1 405B', dataId: 'llama-3.1-405b' },
  { value: 'deepseek-v3', label: 'DeepSeek-V3 671B', dataId: 'deepseek-v3' },
  { value: 'kimi-k2', label: 'Kimi K2 1.04T', dataId: 'kimi-k2' },
  { value: 'deepseek-v4-pro', label: 'DeepSeek-V4-Pro 1.6T', dataId: 'deepseek-v4-pro' },
]);
// Chips: the data key that holds the capacity and the label word that says which kind it is (conventions: B200 reads usable).
export const CHIPS = Object.freeze([
  { value: 'h100', label: 'H100', key: 'hbm_gb', basis: 'nominal' },
  { value: 'h200', label: 'H200', key: 'hbm_gb', basis: 'nominal' },
  { value: 'b200', label: 'B200', key: 'hbm_usable_gb', basis: 'usable' },
  { value: 'b300', label: 'B300', key: 'hbm_gb', basis: 'nominal' },
]);
export const INITIAL_STATE = Object.freeze({ model: 'gpt-3', recipe: 'adam', stage: 0, dp: 64, gpu: 'h100', seq: 2048, recompute: 'none' });

export const isGpt3 = (state) => state.model === 'gpt-3';

export function activationArgs(state) {
  return { layers: GPT3.layers, seq: state.seq, microBatch: GPT3.microBatch, hidden: GPT3.hidden, heads: GPT3.heads, recompute: state.recompute };
}

// The three formula lines of the activation row, per recompute mode (the sum is activationBytesPerLayer × layers).
function activationFormula(state) {
  const seq = group(state.seq, 0);
  const prefix = `${GPT3.layers} × ${seq} × ${group(GPT3.hidden, 0)}`;
  if (state.recompute === 'full') return `${GPT3.layers} × 2 × ${seq} × ${group(GPT3.hidden, 0)} B`;
  if (state.recompute === 'selective') return `${prefix} × 34 B`;
  return `${prefix} × (34 + 5 × ${GPT3.heads} × ${seq} ÷ ${group(GPT3.hidden, 0)}) B`;
}

// "Check my work": every line is a short sum from trainingBytesPerParam / zeroPerGpuBytes / activationBytes; a sharded
// part prints its "÷ dp". `chip` = { label, basis, bytes } (the toy view resolves it from data).
export function checkWork(state, { params, chip }) {
  const recipe = TRAINING_RECIPES[state.recipe];
  const per = trainingBytesPerParam(recipe);
  const z = zeroPerGpuBytes({ params, recipe, stage: state.stage, dp: state.dp });
  const p = formatCount(params);
  const div = (sharded) => (sharded ? ` ÷ ${state.dp}` : '');
  const states = [
    ['weights', `${p} × ${per.weight} B${div(state.stage >= 3)}`, z.weights],
    ['gradients', `${p} × ${per.grad} B${div(state.stage >= 2)}`, z.grads],
    ['optimizer', `${p} × (${per.master} + ${per.optimizer}) B${div(state.stage >= 1)}`, z.optimizer],
  ];
  const lines = states.map(([name, sum, bytes]) => `${name.padEnd(11)} = ${sum} = ${gbText(bytes)}`);
  if (state.stage === 0) lines[2] += '   (stage 0: nothing sharded)';
  const modeled = isGpt3(state);
  const act = modeled ? activationBytes(activationArgs(state)) : 0;
  lines.push(`${'activations'.padEnd(11)} = ${modeled ? `${activationFormula(state)} = ${gbText(act)}` : 'not modeled'}`);
  const total = z.total + act;
  const fits = total <= chip.bytes;
  lines.push(`${'total'.padEnd(11)} = ${gbText(total)} vs ${gbText(chip.bytes)} (${chip.label}, ${chip.basis}) → ${fits ? 'fits' : 'does not fit'}`);
  return lines.join('\n');
}

export const trafficText = (stage) => formatRatio(stage === 3 ? 1.5 : 1);

// What recomputation costs: full = one extra forward pass (forward 1 + backward 2 + rerun 1 = 4 units instead of 3).
export function extraComputeText(recompute) {
  if (recompute === 'full') return 'about +33%';
  if (recompute === 'selective') return 'small, not quantified here';
  return '0%';
}
export const SELECTIVE_HOVER = 'It recomputes only the attention-score grid.';

// Per-block activation split behind frame 6: the score grid is none − selective (80 of 114 parts at GPT-3's shape).
export function scoreSplit(args) {
  const none = activationBytesPerLayer({ ...args, recompute: 'none' });
  const selective = activationBytesPerLayer({ ...args, recompute: 'selective' });
  return { scores: none - selective, rest: selective, total: none };
}

// A size that may be below 1 GB: GB with two decimals from 0.1 GB up (the storyboard's "0.86 GB"), formatBytes below.
export const sizeText = (bytes) => (bytes >= 1e8 ? gbText(bytes) : formatBytes(bytes));
