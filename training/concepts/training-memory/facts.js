// training-memory's dated text (pure, no DOM): the §8 rows and framing, and the notes printed under the stage. Dated numbers
// are {entry.key|format} placeholders filled from data/models.json; sizes are computed with formatBytes from the same entries.
import { formatBytes } from '@math/core.js';
import { sharePct } from '@math/memory.js';
import { TRAINING_RECIPES, trainingBytesPerParam, zeroPerGpuBytes } from '@math/training-memory.js';
import { lookupFact } from '@shared/claims.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const ADAM = TRAINING_RECIPES.adam;
const ADAM_BYTES = trainingBytesPerParam(ADAM).total;
const MUON_BYTES = trainingBytesPerParam(TRAINING_RECIPES.muon).total;
const ZERO_DP = 64;

// "120 GB / 31.4 GB / 16.6 GB / 1.88 GB": the ZeRO paper's 7.5B table at 64 GPUs, stages 0–3, through formatBytes.
function zeroPaperTable(data) {
  const params = fact(data, 'zero-paper-7.5b', 'total_params');
  if (params === null) return '—';
  return [0, 1, 2, 3].map((stage) => formatBytes(zeroPerGpuBytes({ params, recipe: ADAM, stage, dp: ZERO_DP }).total)).join(' / ');
}

function k2PerGpu(data) {
  const [params, bytes, gpus] = ['total_params', 'resident_bytes_per_param', 'model_parallel_gpus'].map((k) => fact(data, 'kimi-k2', k));
  return params && bytes && gpus ? formatBytes((params * bytes) / gpus) : '—';
}

function v4Row(data) {
  const [total, active] = ['total_params', 'active_params'].map((k) => fact(data, 'deepseek-v4-pro', k));
  const share = total && active ? `${sharePct(active, total).toFixed(1)}%` : '—';
  const state = (bytes) => (total ? formatBytes(total * bytes) : '—');
  return `DeepSeek-V4-Pro: {deepseek-v4-pro.total_params|count} total, {deepseek-v4-pro.active_params|count} active (${share}). Its size at the ${ADAM_BYTES}-byte Adam recipe would be ${state(ADAM_BYTES)} of state `
    + `(a what-if: V4 trained with {deepseek-v4-pro.optimizer}); at Muon's ${MUON_BYTES} B, ${state(MUON_BYTES)}.`;
}

export const framing = () => 'Training recipes are decisions about these bytes. The rows below are what the published reports say about how each lab kept the state small enough to fit.';

// The 10 rows of storyboard §8, in order.
export function factRows(data) {
  return [
    { claim: `Mixed-precision Adam costs 2 + 2 + 12 = ${ADAM_BYTES} bytes per parameter (the ZeRO paper); 16 to 18 bytes is the safe number to teach. {zero-paper-7.5b.total_params|cite}` },
    { claim: `ZeRO paper example: {zero-paper-7.5b.total_params|count} parameters on ${ZERO_DP} GPUs → ${zeroPaperTable(data)} per GPU at stages 0 / 1 / 2 / 3.` },
    { claim: 'GPT-3 shape (a preset): {gpt-3.total_params|count} parameters, {gpt-3.layers} blocks, d_model {gpt-3.d_model}, {gpt-3.n_heads} heads, {gpt-3.context_length} context.' },
    { claim: 'DeepSeek-V3 ({deepseek-v3.release_date|year}) stored the optimizer moments in {deepseek-v3.optimizer_state_format} and trained with ZeRO-{deepseek-v3.zero_stage} only, because its pipeline and expert parallelism ({deepseek-v3.parallelism|raw}) already shrink per-GPU state.' },
    { claim: `Kimi K2 ({kimi-k2.release_date|year}): {kimi-k2.total_params|count} parameters; BF16 weights plus an FP32 gradient buffer are {kimi-k2.resident_bytes_per_param} bytes per parameter, about 6 TB over {kimi-k2.model_parallel_gpus} model-parallel GPUs (${k2PerGpu(data)} per GPU), leaving about 30 GB per GPU for its share of optimizer state, sharded across data-parallel ranks. The 6 B is weights and gradient buffer only, not comparable with ${ADAM_BYTES}.` },
    { claim: `Muon keeps one momentum buffer instead of Adam's two: about 4 bytes less per parameter. DeepSeek-V4-Pro uses {deepseek-v4-pro.optimizer}; Kimi K3 uses {kimi-k3.optimizer}.` },
    { claim: v4Row(data) },
    { claim: 'Llama 3.1 405B ({llama-3.1-405b.release_date|year}) used {llama-3.1-405b.sharding|raw}, to avoid a second all-gather.' },
    { claim: 'Modern activation tricks. DeepSeek-V3: {deepseek-v3.activation_tricks|raw}. Kimi K3: {kimi-k3.activation_tricks|raw}.' },
    { claim: 'Total parameters of the presets: Llama 3.1 405B {llama-3.1-405b.total_params|count}, DeepSeek-V3 {deepseek-v3.total_params|count} (mixture-of-experts presets use total, not active, parameters).' },
  ];
}

// Page text under the stage, one list per frame (frame → lesson hand-offs and the dated notes).
export const BELOW = Object.freeze([
  ["GPT-3's shape, today's recipe: GPT-3 itself ({gpt-3.release_date|year}) predates BF16 training. Where its {gpt-3.total_params|count} parameters sit inside the model: [[decoder-anatomy]]."],
  [],
  [],
  [],
  ['Activations grow with the tokens in flight, not with the parameters. Where they come from in a block: [[attention]].'],
  [],
  ['Real recipes recompute selectively and offload too; the "In today\'s models" list below names two.'],
  [],
  [],
  ["Llama 3.1 405B ({llama-3.1-405b.release_date|year}) shards optimizer states and gradients with FSDP but does not re-shard the weights after the forward pass. [[parallelism]] splits activations and state further."],
  ['What a mixture of experts is: [[moe]]. DeepSeek-V4-Pro ({deepseek-v4-pro.total_params|count} total, {deepseek-v4-pro.active_params|count} active) is the example.'],
]);
