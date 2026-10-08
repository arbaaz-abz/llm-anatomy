// rlvr-grpo's dated text (pure, no DOM): the §8 rows and framing. Dated numbers are {entry.key|format} placeholders filled
// from data/models.json (raw prints a value as stored, so 0.2 stays "0.2"); rows keep their placeholders so the scaffold adds
// each row's source link and "reported" chip. Mechanics (GRPO, PPO, DAPO, Dr.GRPO, CISPO) are first principles, not dated facts.

export function framing() {
  return 'Most 2026 open-model reports ship some mix of the same patches around GRPO: a higher upper clip bound, a token-level loss, a KL term that is tiny or gone, filtering of groups with no spread, and corrections for the gap between the engine that sampled the answers and the trainer. Each row names the report it comes from; some are reports on a model series, and rows marked reported are not confirmed by a primary source.';
}

// The 13 rows of storyboard §8, in order.
export function factRows() {
  return [
    { claim: 'GLM-5 reasoning RL: {glm-5.rl_algorithm|raw}, ε_low {glm-5.rl_clip_eps_low|raw} and ε_high {glm-5.rl_clip_eps_high|raw}, KL coefficient {glm-5.rl_kl_coeff|raw} (the KL term is removed), group size {glm-5.rl_group_size}, fully on-policy.' },
    { claim: 'GLM-5 scores math, science, code and tool-integrated reasoning with a {glm-5.rl_reward_type|raw} reward, and filters prompts to ones the previous model solves rarely but stronger teachers can solve.' },
    { claim: 'DeepSeek-V3.2 defines the advantage as R minus mean(R), with no std division{deepseek-v3.2.rl_advantage_norm|cite}.' },
    { claim: 'DeepSeek-V3.2 reports post-training compute of {deepseek-v3.2.post_training_compute_share|raw} (the report\'s own accounting), over {deepseek-v3.2.rl_environments|int} synthetic environments.' },
    { claim: 'DeepSeek-V4 specialists are still trained with SFT then {deepseek-v4-pro.rl_algorithm|raw}; only the mixed-RL stage was replaced by on-policy distillation.' },
    { claim: 'Nemotron 3 Super trains with {nemotron-3-super.rl_algorithm|raw} over {nemotron-3-super.rl_environments} environments.' },
    { claim: 'The MiniMax-M2 series uses {minimax-m2.rl_algorithm|raw}: the importance weight is clipped to [0, 1 + ε_high] with a stop-gradient, so every token keeps a gradient.' },
    { claim: 'Length and effort are controlled through the reward: Kimi K3\'s judge gives over-long answers an automatic loss under a verbosity budget{kimi-k3.rl_length_control|cite}; MiniMax-M2 adds a completion-time reward{minimax-m2.rl_length_control|cite}.' },
    { claim: 'Olmo 3: {olmo-3.rl_patches|raw}.' },
    { claim: 'Magistral, Mistral\'s published RL recipe: KL-free, ε_high {magistral.rl_clip_eps_high|raw}, non-diverse group filtering. Mistral Large 4\'s own algorithm is not disclosed.' },
    { claim: 'The Qwen3 2507 models use {qwen3-2507.rl_algorithm|raw}, a sequence-level importance ratio and clip.' },
    { claim: 'Proprietary labs do not disclose their RL algorithm; the gpt-oss report says only "similar CoT RL techniques as OpenAI o3"{gpt-oss-120b.rl_algorithm|cite}.' },
    { claim: 'Why ratios drift: rollout engines and trainers compute different probabilities for the same tokens, and asynchronous rollouts make data stale. GLM-5\'s fix is {glm-5.rl_mismatch_fix|raw}; other fixes are covered in [[agentic-rl]].' },
  ];
}
