// training-pipeline stand-ins: the six stages' names and block widths, and the few figures the stage prints that restate data
// (each equals its data/models.json entry; the page test pins that, P3-R13). Illustrative replies are labeled so on screen.
import { deepFreeze } from '@math/core.js';

// The blocks are as wide as their labels need; the row is 570 px (6 px gaps) inside the 580 px stage.
export const STAGES = deepFreeze([
  { n: 1, label: '1 pretrain', w: 96, name: 'pretraining' },
  { n: 2, label: '2 mid-train', w: 100, name: 'mid-training' },
  { n: 3, label: '3 SFT', w: 64, name: 'supervised fine-tuning' },
  { n: 4, label: '4 specialist RL', w: 128, name: 'specialist reinforcement learning' },
  { n: 5, label: '5 merge', w: 76, name: 'merging by on-policy distillation' },
  { n: 6, label: '6 polish', w: 76, name: 'final polish' },
]);

// GLM-5's published stage counts (models.glm-5.base_tokens, .midtrain_tokens) and headline (.pretrain_tokens).
export const GLM5_TOKENS = deepFreeze({ pretrain: 27e12, midTrain: 1.55e12, headline: 28.5e12 });

// Shown on the stage: each restates a data entry (tests/training-pipeline-page.test.js compares them).
export const STAGE_FIGURES = deepFreeze({
  pretrainRange: '25–33 trillion tokens of text', // nemotron-3-super 25T … deepseek-v4-pro 33T
  midTrainContext: '4K → 200K', // glm-5.context_stages
  v32Line: 'DeepSeek-V3.2: post-training > 10% of pretraining compute', // deepseek-v3.2.post_training_compute_share
});

// The prompt the RL lessons grade, and each stage's reply to it. All illustrative, not model outputs.
export const PROMPT = 'What is 7 × 8?';
export const REPLIES = deepFreeze({
  random: ['zq', 'mat', ',,', ',,'],
  base: ['What is 9 × 6?', 'What is 4 × 7?'],
  chat: ['<think>', '7 × 8 = 56', '</think>', '56'],
});

export const SPECIALISTS = deepFreeze([
  { name: 'math & code', under: 'checked answers' },
  { name: 'agents', under: 'sandbox tasks' },
  { name: 'chat', under: 'judge model' },
]);

// Stop lessons per stage (frames 2–9 and the toy's inspector): [[slug]] links in page text.
export const STOPS = deepFreeze([
  ['pretraining', 'scaling-laws'],
  ['midtraining'],
  ['sft'],
  ['rlhf-dpo', 'rlvr-grpo', 'agentic-rl'],
  ['distillation'],
  ['rlhf-dpo', 'quantization'],
]);
