// agentic-rl's dated text (pure, no DOM): the §8 rows, the framing, and the two stage frames that print dated numbers.
// Dated numbers are {entry.key|format} placeholders filled from data/*.json; rows keep their placeholders so the scaffold adds
// each row's source link and "reported" chip. The stage text is filled with fillText (README lesson 33).
import { fillText } from '@shared/claims.js';

export function framing() {
  return 'These are the 2026 reports behind the page: how the biggest open models run agentic RL (environments, judges, asynchronous rollouts) and the patches they add around GRPO (importance-sampling corrections, little or no KL term). Rows marked reported are a report\'s own claim, not independently replicated.';
}

// The 10 rows of storyboard §8, in order.
export function factRows() {
  return [
    { claim: 'GLM-5: {glm-5.rl_infra|raw}; software-engineering environments: {glm-5.agentic_envs_swe|raw}; also {glm-5.agentic_envs_other|raw}.' },
    { claim: 'GLM-5 masks tokens whose trainer-to-engine ratio leaves the band ({glm-5.rl_mismatch_fix|raw}); its KL coefficient is {glm-5.rl_kl_coeff|raw}.' },
    { claim: 'DeepSeek-V3.2: {deepseek-v3.2.rl_patches|raw}; {deepseek-v3.2.rl_environments|int} synthetic environments with {deepseek-v3.2.rl_tasks|int} tasks; post-training compute {deepseek-v3.2.post_training_compute_share|raw}.' },
    { claim: 'DeepSeek-V4: {deepseek-v4-pro.rl_infra|raw}, because regenerating interrupted requests from scratch biases the data toward short responses; the reward model is {deepseek-v4-pro.reward_model|raw} (the policy itself is the judge).' },
    { claim: 'Kimi K3: {kimi-k3.rl_experts|int} specialist experts, {kimi-k3.rl_expert_grid}; {kimi-k3.rl_rollouts|raw}; reward model: {kimi-k3.reward_model|raw}.' },
    { claim: 'Nemotron 3 Super: {nemotron-3-super.rl_algorithm|raw} over {nemotron-3-super.rl_environments|int} environments, with a separate SWE-RL stage{nemotron-3-super.swe_rl_stage|cite} because software-engineering rollouts are slow and long.' },
    { claim: 'Olmo 3: {olmo-3.rl_patches|raw}.' },
    { claim: 'Mistral Large 4: asynchronous RL producing about {mistral-large-4.rl_tokens_per_day|count} tokens per day ({mistral-large-4.rl_trainable_tokens_per_day|count} of them trainable completion tokens) on about {mistral-large-4.rl_gpus|count} GPUs.' },
    { claim: 'MiniMax-M2: dense rewards beyond the task outcome ({minimax-m2.rl_rewards|raw}), such as language mixing and tool-format errors.' },
    { claim: 'Using FP16 instead of BF16 shrinks the rounding mismatch between the sampling engine and the trainer{paper:fp16-mismatch-2025.finding|cite}.' },
  ];
}

// Frame 9 and 10's dated stage lines, filled from the data (no data → the placeholders print "—", never on a shipped page).
const STAGE_LINES = Object.freeze({
  beta: 'β = {glm-5.rl_kl_coeff|raw} (GLM-5, Olmo 3) · "weak or zero for math" (DeepSeek-V3.2)',
  table: [
    'GLM-5: SWE environments {glm-5.agentic_envs_swe|raw}',
    'DeepSeek-V3.2: {deepseek-v3.2.rl_environments|int} environments, {deepseek-v3.2.rl_tasks|int} tasks',
    'Kimi K3: {kimi-k3.rl_experts|int} experts: {kimi-k3.rl_expert_grid}',
    'Nemotron 3 Super: {nemotron-3-super.rl_environments|int} environments',
    'Mistral: ~{mistral-large-4.rl_tokens_per_day|count} tokens produced per day by one run on ~{mistral-large-4.rl_gpus|count} GPUs, ~{mistral-large-4.rl_trainable_tokens_per_day|count} of them trainable completion tokens',
  ],
});

export function stageText(data) {
  return Object.freeze({ beta: fillText(STAGE_LINES.beta, data), table: STAGE_LINES.table.map((line) => fillText(line, data)) });
}
