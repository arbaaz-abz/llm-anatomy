// training-pipeline: the long expected strings, exported once for the page test and the e2e spec.
// Storyboard §5, typed out again here on purpose: the page test fails if captions.js drifts from the storyboard.
export const CAPTIONS = Object.freeze([
  'The 2026 recipe has six stages. Each one starts from the weights the previous stage left, its checkpoint, and adds one thing.',
  'Pretraining predicts the next token over tens of trillions of tokens. The base model it produces knows a lot but only continues text.',
  'Mid-training is the end of that run: the best data, and a context window stretched from thousands to hundreds of thousands of tokens.',
  'Supervised fine-tuning shows it worked conversations. Now it answers in turns, thinks between tags, and writes tool calls.',
  'The model is copied into specialists, each trained with reinforcement learning on its own tasks. This is where most 2026 reasoning and agent skill comes from.',
  'On-policy distillation merges the specialists back into one model that keeps each one\'s best skills.',
  'A final polish tunes style and safety, often with RL against a judge model, and the weights are prepared for serving.',
  'By tokens, pretraining is nearly everything: 94.6% of GLM-5\'s published budget. Post-training token counts are mostly not published.',
  'By compute, post-training is no longer small: over a tenth of pretraining for DeepSeek-V3.2. Each RL token is generated and scored before it teaches anything.',
]);

// "Check my work" for each preset (storyboard §6; GLM-5 is the default).
export const CHECK_WORK = [
  'known total = 27T + 1.55T     = 28.55T',
  'pretrain    = 27T ÷ 28.55T    = 94.6%',
  'mid-train   = 1.55T ÷ 28.55T  = 5.4%',
  'post-training: not published, so not in the total',
].join('\n');
export const CHECK_WORK_KIMI = 'nothing published: no shares to compute';

export const SHARE_BAR_LABEL = '28.5T (its published stages sum to 28.55T)';
