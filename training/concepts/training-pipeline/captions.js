// training-pipeline storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
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
