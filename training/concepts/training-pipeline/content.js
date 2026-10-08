// training-pipeline: the complete lesson (storyboard docs/storyboards/training-pipeline.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FRAMING, ROWS, belowFor, knownTotalText, pretrainShare } from './facts.js';

const SLUG = 'training-pipeline';

const HOOK = 'What happens between a pile of 30 trillion tokens and a model that thinks before it answers, calls tools, and holds a conversation?';

const intuition = () => Object.freeze([
  'The canonical 2026 pipeline has six stages, each starting from the weights the last one left (smaller pipelines skip some; the toy shows which). Each stage needs data the one before could not use: text teaches knowledge but not conversation, conversations teach format but not which answer is right, and only a checker can teach that. Pretraining reads tens of trillions of tokens and learns to predict the next one; the result knows a great deal but only continues text. Mid-training is the end of that run, on the best data and with a longer context window. Supervised fine-tuning shows it worked conversations, so it learns to answer in turns and to reason between think tags.',
  'Then the model is copied. Each copy, a specialist, is trained with reinforcement learning on one kind of task: math and code with answers a program can check, agent tasks in sandboxes, conversation judged by another model. This is where most 2026 gains in reasoning and tool use come from. On-policy distillation then merges the specialists back into one model that keeps each one\'s peak. A final polish tunes style and safety, and the weights are prepared for serving.',
  `The stages are very unequal. Pretraining is nearly all of the tokens: ${pretrainShare()} of GLM-5's published budget (28.5T; its published stages sum to ${knownTotalText()}). But a post-training token is far more expensive than a pretraining one, since it has to be generated, scored and often run in a sandbox first, so post-training is no longer cheap: DeepSeek-V3.2's report puts it at {deepseek-v3.2.post_training_compute_share}. The rest of this section opens each stage in turn.`,
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{share}_i = \frac{T_i}{\sum_{j\,\in\,\text{published}} T_j}
\qquad
\text{GLM-5: } \frac{27}{27 + 1.55} = 94.6\%,\ \ \frac{1.55}{28.55} = 5.4\%` },
  { tex: tex`\text{training compute} \approx 6\,N_{\text{active}}\,D \quad (\text{taught in the scaling-laws lesson})` },
]);

const MATH_NOTES = Object.freeze([
  'Shares are of the published total only; an unpublished stage is shown in the neutral fill, never estimated.',
  'Why an RL token costs more than a pretraining token: it is first generated one token at a time (decode, see [[prefill-decode]]), then scored by a checker, sandbox or judge, and only then trained on. No formula is given because the ratio depends on the run.',
]);

const TAKEAWAYS = Object.freeze([
  'The canonical 2026 pipeline is six stages, each starting from the last one\'s weights: pretraining, mid-training, SFT, specialist RL, merging by on-policy distillation, and a final polish.',
  'Pretraining gives knowledge; SFT gives format; specialist RL gives most of the reasoning and agent skill; distillation merges the specialists into one model.',
  'Pretraining is nearly all the tokens, but post-training is no longer cheap in compute (over 10% of pretraining for DeepSeek-V3.2), because every RL token is generated and scored before it teaches.',
]);

const FURTHER = Object.freeze([
  { title: 'nanochat (Andrej Karpathy)', href: 'https://github.com/karpathy/nanochat', note: 'a runnable tokenizer, pretrain, midtrain, SFT and RL pipeline' },
  { title: 'The Smol Training Playbook (Hugging Face)', href: 'https://huggingfacetb-smol-training-playbook.hf.space/', note: 'what it takes to train a small model end to end' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: intuition().map(fill),
    animation: {
      label: 'The 2026 training pipeline: six stages, then how big each is, in nine steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The replies to "What is 7 × 8?" are illustrative, not real model outputs. Token counts are published figures.',
      belowFor: (index) => belowFor(index).map(fill),
    },
    toy: {
      title: 'Read a recipe',
      intro: 'Pick a 2026 model to see which of the six stages its report describes, what it says about each in its own units, and its token budget by stage. Select a stage block, with a click or the arrow keys, to read it.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: ROWS.map((claim) => ({ claim })) },
    takeaways: TAKEAWAYS,
    links: { next: ['pretraining'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
