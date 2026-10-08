// scale-reliability's dated text (pure, no DOM): the §8 rows, the framing and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/*.json, so each row keeps its source link and its
// "reported" chip; a number the formatters cannot print (3.8e25, a [low, high] band) sits beside a `|cite` placeholder and
// is pinned to the data by the page test (README lesson 29); a [low, high] band's `|cite` prints only its "–", so "38–43%" is
// written 38{…|cite}43%. Ratios are computed from the same entries.
import { formatRatio } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const hwFact = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;

export const FRAMING = 'Everything on this page is one chain of arithmetic: FLOPs, then GPU-hours at the utilization a run really got, then the hours lost to failures, then days and dollars. The runs are Llama 3.1 405B (the followed run) and DeepSeek-V3, because both published the numbers; the save and restart times in the toy are stand-ins.';

const megatronRatio = (data) => {
  const [gb300, gb200] = [hwFact(data, 'gb300-nvl72', 'megatron_dsv3_tflops_per_gpu'), hwFact(data, 'gb200-nvl72', 'megatron_dsv3_tflops_per_gpu')];
  return gb300 && gb200 ? formatRatio(gb300 / gb200) : '—';
};

const unexpectedPlusPlanned = (data) => {
  const [unexpected, planned] = [fact(data, 'llama-3.1-405b', 'interruptions_54d'), fact(data, 'llama-3.1-405b', 'interruptions_planned')];
  return unexpected != null && planned != null ? String(unexpected + planned) : '—';
};

function deepseekRow(data) {
  const [tokens, pretrain] = [fact(data, 'deepseek-v3', 'pretrain_tokens'), fact(data, 'deepseek-v3', 'pretrain_gpu_hours')];
  const perTrillion = tokens && pretrain ? `${Math.round(pretrain / (tokens / 1e12) / 1000)}K` : '—';
  return `DeepSeek-V3 ({deepseek-v3.release_date|year}): {deepseek-v3.training_gpu_hours|count4} H800-hours in all, {deepseek-v3.pretrain_gpu_hours|count4} for pre-training (${perTrillion} per trillion tokens on {deepseek-v3.training_gpus|int} GPUs), `
    + 'priced at $2 per GPU-hour: ${deepseek-v3.training_cost_usd_reported|count4}, excluding research and ablations.';
}

// The 13 rows of storyboard §8, in order.
export function factRows(data) {
  return [
    { claim: 'Llama 3.1 405B ({llama-3.1-405b.release_date|year}): {llama-3.1-405b.pretrain_tokens|count} tokens, 3.8 × 10²⁵ FLOPs{llama-3.1-405b.training_flops|cite}, up to {llama-3.1-405b.training_gpus|int} H100s.' },
    { claim: "Llama 3.1 405B's model card: {llama-3.1-405b.training_gpu_hours|count4} H100 GPU-hours, which covers more than the 15.6T-token pre-training." },
    { claim: 'Llama 3.1 405B BF16 MFU while training: {llama-3.1-405b.mfu_bf16|pct}% (43% with TP8/PP16/DP64 on 8,192 GPUs at 430 TFLOPS; 41% on 16,384 GPUs at 400; 38% in the long-context stage at 380).' },
    { claim: `Llama 3.1's 54-day window: ${unexpectedPlusPlanned(data)} interruptions, {llama-3.1-405b.interruptions_planned|int} planned and {llama-3.1-405b.interruptions_54d|int} unexpected; 78% hardware{llama-3.1-405b.interruptions_hardware_share|cite}, GPU issues 58.7% of the unexpected{llama-3.1-405b.interruptions_gpu_share|cite}; effective training time {llama-3.1-405b.effective_time}.` },
    { claim: 'Llama 3.1 405B run: {llama-3.1-405b.reliability_notes}.' },
    { claim: deepseekRow(data) },
    { claim: 'DeepSeek-V3 FP8 recipe: {deepseek-v3.fp8_recipe}; relative loss error against BF16 {deepseek-v3.fp8_loss_error}.' },
    { claim: 'DeepSeek-V3 reserves {deepseek-v3.comm_sms} SMs for communication; all-to-all and pipeline traffic "can be fully hidden".' },
    { claim: 'Llama 4 Behemoth: {llama-4-behemoth.training_gpus} GPUs, {llama-4-behemoth.pretrain_precision}, {llama-4-behemoth.achieved_tflops_per_gpu} TFLOPS per GPU; Meta gave no MFU.' },
    { claim: `NVIDIA Megatron Core on GB300 NVL72: {hw:gb300-nvl72.megatron_dsv3_tflops_per_gpu|int} TFLOPS per GPU on DeepSeek-V3 pre-training with 256 GPUs, ${megatronRatio(data)} GB200 NVL72's {hw:gb200-nvl72.megatron_dsv3_tflops_per_gpu|int} (precision not stated, so no MFU is derivable).` },
    { claim: 'Low-precision pre-training: MiMo-V2-Flash trained in {mimo-v2-flash.pretrain_precision} over {mimo-v2-flash.pretrain_tokens|count} tokens; Nemotron 3 Super and Ultra were both pre-trained in {nemotron-3-super.pretrain_precision}{nemotron-3-ultra.pretrain_precision|cite}; DeepSeek-V4 uses FP4 quantization-aware training only in post-training{deepseek-v4-pro.post_training_qat|cite}.' },
    { claim: 'DeepSeek-V4 warns that {deepseek-v4-pro.notes_power}.' },
    { claim: 'Kimi K2 ({kimi-k2.release_date|year}): {kimi-k2.loss_spikes} loss spikes over {kimi-k2.pretrain_tokens|count} tokens with MuonClip: training stability, not hardware reliability.' },
  ];
}

// Page text under the stage, one list per frame: hand-offs to other lessons and the dated notes.
export const BELOW = Object.freeze([
  ['How N and D are chosen: [[scaling-laws]]. Where the memory of a training run goes: [[training-memory]].'],
  [],
  [],
  ['Why communication takes time: [[cluster-topology]]. The idle gaps of pipeline parallelism: [[parallelism]].'],
  [],
  [],
  ['Llama 4 Behemoth trained on {llama-4-behemoth.training_gpus} GPUs. Each percentage names the peak it divides by, because FP8 doubles the peak.'],
  [],
  [],
  [],
  ['The same arithmetic prices inference: [[serving-calculator]].'],
]);
