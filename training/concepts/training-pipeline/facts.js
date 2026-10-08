// training-pipeline's dated text (pure, no DOM): the §8 rows and framing, and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; the 94.6% and 28.55T are computed
// from the stage counts with budgetShares. Rows keep their placeholders so the scaffold adds each row's source link.
import { budgetShares } from '@math/pipeline.js';
import { formatCount } from '@math/core.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { GLM5_PARTS } from './frames-budget.js';

export const FRAMING = 'This page is the course\'s synthesis of the 2026 open reports below: the canonical pipeline is pretraining, mid-training, SFT (a cold start), specialist RL, merging by on-policy distillation, and a final polish. '
  + 'Labs describe their own stages in their own units, so the rows give each figure as published.';

export const ROWS = Object.freeze([
  'Open frontier MoEs pretrain on {nemotron-3-super.pretrain_tokens|count}–{deepseek-v4-pro.pretrain_tokens|count} tokens: Nemotron 3 Super {nemotron-3-super.pretrain_tokens|count}, GLM-5 {glm-5.pretrain_tokens|count}, DeepSeek-V4-Pro {deepseek-v4-pro.pretrain_tokens|count}.',
  'GLM-5 pretrains on {glm-5.base_tokens|count} tokens at 4K context, then {glm-5.midtrain_tokens|count} of mid-training that stretches it: {glm-5.context_stages}.',
  'SFT sizes: DeepSeek-R1 ({deepseek-r1.release_date|year}) about {deepseek-r1.sft_samples|count} examples; Olmo 3 about {olmo-3.sft_traces|count} reasoning traces (reported).',
  'Specialist RL: Kimi K3 trains {kimi-k3.rl_experts} specialists; GLM-5 reports software-engineering environments: {glm-5.agentic_envs_swe}.',
  'Merging: DeepSeek-V4-Pro, {deepseek-v4-pro.opd}; Kimi K3, {kimi-k3.opd}; GLM-5, {glm-5.opd}.',
  'Nemotron 3 Super: RLVR over {nemotron-3-super.rl_environments} environments, a separate SWE-RL stage{nemotron-3-super.swe_rl_stage|cite}, then a separate RLHF stage with a judge model{nemotron-3-super.rlhf_stage|cite}.',
  'FP4-aware training in post-training: DeepSeek-V4-Pro, {deepseek-v4-pro.post_training_qat}; Kimi K3, {kimi-k3.post_training_qat}.',
  'DeepSeek-V3.2 post-training compute: {deepseek-v3.2.post_training_compute_share}.',
  'Mistral: one RL run at about {mistral-large-4.rl_gpus|count} GPUs produces about {mistral-large-4.rl_tokens_per_day|count} tokens a day, about {mistral-large-4.rl_trainable_tokens_per_day|count} of them trainable completion tokens.',
  'Olmo 3 (reported): {olmo-3.pretrain_tokens|count} pretraining tokens, {olmo-3.midtrain_tokens|count} mid-training, {olmo-3.dpo_pairs|count} DPO pairs, {olmo-3.rlvr_prompts|count} RLVR prompts.',
]);

const shares = () => budgetShares(GLM5_PARTS.map(({ name, value }) => ({ name, value })));
export const knownTotalText = () => formatCount(shares().knownTotal, { digits: 4 });
export const pretrainShare = () => formatShare(shares().parts[0].share);
export const midTrainShare = () => formatShare(shares().parts[1].share);

// Notes under each frame (index = frame - 1): dated figures as placeholders, stop lessons as [[slug]] (README lesson 33).
export function belowFor(index) {
  const notes = [
    [],
    ['Open frontier MoEs pretrain on {nemotron-3-super.pretrain_tokens|count}–{deepseek-v4-pro.pretrain_tokens|count} tokens: Nemotron 3 Super {nemotron-3-super.pretrain_tokens|count}, GLM-5 {glm-5.pretrain_tokens|count}, DeepSeek-V4-Pro {deepseek-v4-pro.pretrain_tokens|count}. Taught in [[pretraining]] and [[scaling-laws]].'],
    ['GLM-5: {glm-5.midtrain_tokens|count} tokens of mid-training, context 4K to 200K. Taught in [[midtraining]].'],
    ['DeepSeek-R1 ({deepseek-r1.release_date|year}): about {deepseek-r1.sft_samples|count} examples. Olmo 3: about {olmo-3.sft_traces|count} traces (reported). Taught in [[sft]].'],
    ['Kimi K3: {kimi-k3.rl_experts} specialists. GLM-5 software environments: {glm-5.agentic_envs_swe}. Taught in [[rlhf-dpo]], [[rlvr-grpo]] and [[agentic-rl]].'],
    ['DeepSeek-V4-Pro: {deepseek-v4-pro.opd}. Kimi K3: {kimi-k3.opd}. Taught in [[distillation]].'],
    ['Nemotron 3 Super ends with a separate RLHF stage that uses a judge model. DeepSeek-V4-Pro and Kimi K3 use FP4-aware training in post-training: {deepseek-v4-pro.post_training_qat}; {kimi-k3.post_training_qat}. See [[rlhf-dpo]] for the polish and [[quantization]] for serving.'],
    [`Shares are of the published stages only: {glm-5.base_tokens|count} + {glm-5.midtrain_tokens|count} = ${knownTotalText()}. Pretraining ${pretrainShare()}, mid-training ${midTrainShare()}.`],
    ['DeepSeek-V3.2: post-training is {deepseek-v3.2.post_training_compute_share}. Mistral: about {mistral-large-4.rl_tokens_per_day|count} RL tokens produced per day by one run on about {mistral-large-4.rl_gpus|count} GPUs, of which about {mistral-large-4.rl_trainable_tokens_per_day|count} are trainable completion tokens. See [[gpu-primer]] and [[scale-reliability]] for what the compute is spent on.'],
  ];
  return notes[index] ?? [];
}
