// serving-overview: the complete lesson (storyboard docs/storyboards/serving-overview.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { HOOK, INTUITION, INTUITION_NOTE, FRAMING, FACT_ROWS, BELOW, TAKEAWAYS } from './facts.js';

const SLUG = 'serving-overview';
const tex = String.raw;

const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-ttft}{\text{TTFT}} = t_{\text{queue}} + t_{\text{prefill}},\quad t_{\text{prefill}} = \max\!\left(\frac{2N\cdot\text{prompt}}{\text{peak FLOP/s}},\ \frac{\text{bytes read}}{\text{bandwidth}}\right) \qquad \htmlClass{hl-tpot}{\text{TPOT}} = \frac{1}{\text{decode speed per user}}` },
  { tex: tex`t_{\text{total}} = \htmlClass{hl-ttft}{\text{TTFT}} + (\text{answer tokens} - 1)\cdot\htmlClass{hl-tpot}{\text{TPOT}}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: none (scalars). N is the active parameters on the GPU; the bytes read are the weights plus the activations. One definition each across the serving track: TTFT = queue + prefill, ending when the first output token is picked; TPOT = the time between two consecutive output tokens of one request = one decode step; tokens/s per user = 1 ÷ TPOT.',
  'Hover TTFT or TPOT to outline its bracket on the stage (frames 5 and 7). Where the step time itself comes from, and why decode is limited by memory and prefill by math: [[prefill-decode]].',
]);

const FURTHER = Object.freeze([
  { title: 'LLM Inference Explained (Brendan Lynskey)', href: 'https://llm-inference-explained.vercel.app', note: 'the long-form companion for this whole track' },
  { title: 'Continuous batching from first principles (Hugging Face)', href: 'https://huggingface.co/blog/continuous_batching', note: 'from one request to a full batch' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: [...INTUITION],
    intuitionNote: INTUITION_NOTE,
    animation: {
      label: 'One request\'s journey: router, queue, prefill, decode, and the batch it shares',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'Your request\'s three-token prompt is a toy. Its times are a floor, not a measurement: Llama-3.1-70B in FP8 on one H200, from the step-time model in [[prefill-decode]]. Requests A–D are the course\'s toy batch.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Where does the time go?',
      intro: 'Every time here is a floor, not a measurement: one model on one GPU (Llama-3.1-70B in FP8 on an H200), from the step-time model in [[prefill-decode]]. Real servers are slower and vary with load.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: [...FACT_ROWS] },
    takeaways: [...TAKEAWAYS],
    links: { next: ['prefill-decode'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
