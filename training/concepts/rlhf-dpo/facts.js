// rlhf-dpo "In today's models" rows (storyboard §8) and the frame 11 note under the stage. Dated numbers are
// {entry.key|format} placeholders; the scaffold fills the rows and adds each source link and "reported" chip.
export const FRAMING = 'In 2026, reasoning and agent gains come from RL on checkable rewards (see [[rlvr-grpo]] and [[agentic-rl]]). RLHF and DPO survive in narrower roles: a late polish stage with a judge model as the reward, and a cheap preference stage in smaller open pipelines.';

export const ROWS = Object.freeze([
  { claim: 'Nemotron 3 Super runs RLVR across {nemotron-3-super.rl_environments} environments, then SWE-RL, then a separate RLHF stage whose reward is a principle-following generative reward model: {nemotron-3-super.rlhf_stage}.' },
  { claim: 'GLM-5\'s general RL mixes rule rewards, outcome reward models and generative reward models, and anchors on human-written responses to avoid style drift: {glm-5.general_rl_rewards}.' },
  { claim: 'Mistral Large 4 verifies with {mistral-large-4.rl_verifiers}.' },
  { claim: 'Olmo 3: SFT, then DPO on about {olmo-3.dpo_pairs|count} "Delta Learning" pairs (chosen from a stronger model, rejected from a weaker one), then RLVR on about {olmo-3.rlvr_prompts|count} prompts.' },
  { claim: 'SmolLM3 used {smollm3.preference_method}, a DPO-family method.' },
  { claim: 'DPO is {deepseek-v4-pro.dpo_stage} as a main stage in the DeepSeek-V4-Pro, GLM-5{glm-5.dpo_stage|cite}, Kimi K3{kimi-k3.dpo_stage|cite}, MiniMax-M2{minimax-m2.dpo_stage|cite} and MiMo-V2-Flash{mimo-v2-flash.dpo_stage|cite} reports (absent from the reports read, not proof of non-use).' },
]);

// Mechanics, not a dated fact: a paragraph under the rows, so every row keeps a source link.
export const PROSE = Object.freeze([
  'PPO keeps four models in memory: policy, reference, reward model and critic (InstructGPT, arXiv 2203.02155). DPO\'s loss is derived from the same KL-anchored objective (arXiv 2305.18290).',
]);

// Frame 11's dated figures sit under the stage, filled from the data (decoder-anatomy's belowFor pattern).
export const FRAME_11_NOTE = 'Olmo 3: SFT → DPO (about {olmo-3.dpo_pairs|count} pairs) → RLVR (about {olmo-3.rlvr_prompts|count} prompts), reported. Nemotron 3 Super: RLVR → SWE-RL → RLHF with a principle-following judge, confirmed.';
