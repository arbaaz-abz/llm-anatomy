// sft: the complete lesson (storyboard docs/storyboards/sft.md §3, §5, §7–§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FRAMING, FACT_ROWS, BELOW } from './facts.js';
import { mathBlocks } from './math-blocks.js';

export { CAPTIONS };

const SLUG = 'sft';

const HOOK = 'A pretrained model, asked "What is 7 × 8?", might just write another question. How do hundreds of thousands of examples teach it to answer, to think first, and to call tools?';

const INTUITION = Object.freeze([
  'A base model has read trillions of tokens, so it knows a great deal, but it only knows how to continue text. Supervised fine-tuning (SFT) shows it what the next part of a conversation looks like. Every example is wrapped in a chat template: special tokens that mark who is speaking (system, user, assistant, tool), so the model can tell a question it was asked from an answer it should write. Reasoning models add a `<think>` … `</think>` block before the answer. These are ordinary tokens; the model learns to use them because every example does. The knowledge is already in the weights; SFT only has to teach the shape of a conversation, which takes far fewer examples than learning the world did.',
  'The loss is the same next-token cross-entropy as in [[pretraining]], with one change: a loss mask. Only the assistant\'s tokens count. The prompt, the template tags and anything a tool returned are there as context but are never targets, so the model learns to write answers and tool calls, not to imitate the user or to invent tool results. GLM-5 adds a refinement for agent traces: when an example contains a mistake followed by a recovery, the mistake stays in the text but is masked, so the model sees how to recover without being trained to make the error.',
  'The data is the hard part. In 2026 it comes from earlier specialist models, from rejection sampling (keep only the samples that pass a check), and from runs in real tool environments, often with long reasoning traces. DeepSeek-R1 ({deepseek-r1.release_date|year}) used about {deepseek-r1.sft_samples|count} examples; Olmo 3 about {olmo-3.sft_traces|count} reasoning traces (reported). The result is a model that answers in the right format and sometimes reasons its way to the right answer. That "sometimes" is the point: it is the cold start that RL needs, because RL can only reinforce what the model already occasionally does. The cost is that SFT only imitates: it is bounded by its data, and wrong traces that are not filtered or masked are learned like right ones.',
]);

const STAND_IN = 'The template tags here are generic stand-ins; real ones differ per lab (see "In today\'s models" below). The base model\'s continuation in frame 1 is illustrative.';
const TOY_INTRO = 'Tags are generic stand-ins; each lab\'s template differs.';

const MATH_NOTES = Object.freeze([
  'Shapes: x [T] (T = 26 here); m [T] boolean; the denominator counts the trained tokens. Everything else is the loss of [[pretraining]]; the mask is the only change. Some labs average per example instead of per token; this page uses the token mean. Hover a highlighted term to outline the transcript on the stage.',
  'Thinking modes (DeepSeek-V4 {deepseek-v4-pro.thinking_modes}; Kimi K3 {kimi-k3.thinking_modes}; gpt-oss {gpt-oss-120b.thinking_modes}) are selected by template and trained with different length budgets later, in RL.',
]);

const TAKEAWAYS = Object.freeze([
  'SFT wraps conversations in a chat template (role tokens, `<think>` tags, tool-call tags) and trains with pretraining\'s own loss, counted only on the assistant\'s tokens.',
  'The loss mask is the whole trick: prompts, tags and tool outputs are context, never targets, and GLM-5 also masks a kept mistake so the model learns the recovery, not the error.',
  'SFT data mostly comes from other models and rejection sampling; the stage teaches format and gives RL its cold start, and it can only imitate what its data shows.',
]);

const FURTHER = Object.freeze([
  { title: 'Andrej Karpathy, nanochat', href: 'https://github.com/karpathy/nanochat', note: 'its SFT stage shows a real chat template and loss mask in code' },
  { title: 'Nathan Lambert, RLHF Book', href: 'https://rlhfbook.com/', note: 'the instruction-tuning chapter' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Supervised fine-tuning: a chat template, think tags, the loss mask on the prompt, the tool reply and a kept mistake, where the data comes from, and the cold start for RL',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: STAND_IN,
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: { title: 'Which tokens teach?', intro: TOY_INTRO, mount },
    math: { blocks: mathBlocks(), notes: MATH_NOTES.map(fill) },
    facts: { framing: FRAMING, rows: FACT_ROWS.map((row) => ({ ...row })) },
    takeaways: TAKEAWAYS.map(fill),
    links: { next: ['rlhf-dpo'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
