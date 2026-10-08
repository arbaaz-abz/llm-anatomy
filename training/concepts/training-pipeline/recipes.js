// training-pipeline toy presets (storyboard §6): what each model's report says about the six stages, in its own unit.
// Texts are {entry.key|format} templates filled from data/models.json; null = the report does not describe the stage.
// Token parts: [name, key, hue]; key null = not published (budgetShares' null; hue only for known parts).
import { deepFreeze } from '@math/core.js';

export const RECIPES = deepFreeze({
  glm5: {
    id: 'glm-5',
    label: 'GLM-5',
    stages: [
      '{glm-5.base_tokens|count} tokens of base pretraining at 4K context',
      '{glm-5.midtrain_tokens|count} tokens at 32K, 128K and 200K context ({glm-5.stage_32k_tokens|count}, {glm-5.stage_128k_tokens|count}, {glm-5.stage_200k_tokens|count})',
      '{glm-5.sft_data}; {glm-5.sft_masking} (no example count published)',
      'reasoning RL, then agentic RL, then general RL; software-engineering environments: {glm-5.agentic_envs_swe}',
      'cross-stage distillation: {glm-5.opd}',
      'general RL with rewards: {glm-5.general_rl_rewards}',
    ],
    tokens: [['pretrain', 'base_tokens', 1], ['mid-train', 'midtrain_tokens', 2], ['post-training', null]],
  },
  deepseekV4: {
    id: 'deepseek-v4-pro',
    label: 'DeepSeek-V4',
    stages: [
      '{deepseek-v4-pro.pretrain_tokens|count} tokens, with the context schedule inside it ({deepseek-v4-pro.context_stages})',
      '{deepseek-v4-pro.midtrain_data}; its tokens are not counted separately',
      'SFT of each specialist before its RL (no example count published)',
      'specialist RL with {deepseek-v4-pro.rl_algorithm}',
      '{deepseek-v4-pro.opd}',
      'FP4-aware training: {deepseek-v4-pro.post_training_qat}',
    ],
    tokens: [['pretrain incl. mid-train', 'pretrain_tokens', 1], ['post-training', null]],
  },
  kimiK3: {
    id: 'kimi-k3',
    label: 'Kimi K3',
    stages: [
      'pretraining tokens: {kimi-k3.pretrain_tokens}',
      'long-context cooldown: {kimi-k3.context_stages}',
      'cold start: {kimi-k3.sft_data}',
      '{kimi-k3.rl_experts} specialists: 3 domains × 3 effort levels',
      '{kimi-k3.opd}',
      'FP4-aware training from SFT on: {kimi-k3.post_training_qat}',
    ],
    tokens: [['pretrain', null], ['mid-train', null], ['post-training', null]],
  },
  nemotron3Super: {
    id: 'nemotron-3-super',
    label: 'Nemotron 3 Super',
    stages: [
      '{nemotron-3-super.pretrain_tokens|count} tokens',
      null,
      null,
      'RLVR over {nemotron-3-super.rl_environments} environments, then a separate SWE-RL stage{nemotron-3-super.swe_rl_stage|cite}',
      null,
      'a separate RLHF stage with a judge model{nemotron-3-super.rlhf_stage|cite}',
    ],
    tokens: [['pretrain', 'pretrain_tokens', 1], ['post-training', null]],
  },
  olmo3: {
    id: 'olmo-3',
    label: 'Olmo 3 (reported)',
    stages: [
      '{olmo-3.pretrain_tokens|count} tokens (reported)',
      '{olmo-3.midtrain_tokens|count} tokens of mid-training (reported)',
      '{olmo-3.sft_traces|count} reasoning traces (reported)',
      '{olmo-3.dpo_pairs|count} preference pairs (DPO), then {olmo-3.rlvr_prompts|count} prompts (RLVR) (reported)',
      null,
      null,
    ],
    tokens: [['pretrain', 'pretrain_tokens', 1], ['mid-train', 'midtrain_tokens', 2], ['post-training', null]],
  },
});

export const RECIPE_ORDER = Object.freeze(['glm5', 'deepseekV4', 'kimiK3', 'nemotron3Super', 'olmo3']);
