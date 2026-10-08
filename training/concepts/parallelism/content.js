// parallelism: the complete lesson (storyboard docs/storyboards/parallelism.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { formatBytes } from '@math/core.js';
import { BELOW, FRAMING, factRows } from './facts.js';
import { LLAMA, WEIGHT_BYTES_PER_PARAM } from './numbers.js';
import { int, formatStateGB, stateView } from './format.js';

const SLUG = 'parallelism';
const TOY_STATE = stateView({ ...LLAMA, zero: LLAMA.zeroStage });

const HOOK = `A {llama-3.1-405b.total_params|count} model needs ${formatBytes(LLAMA.params * WEIGHT_BYTES_PER_PARAM)} of training state and an H100 holds {hw:h100.hbm_gb} GB (nominal). When you spread it over ${int(LLAMA.gpus)} GPUs, what does each GPU actually hold, and what do they say to each other?`;

const INTUITION = Object.freeze([
  'There are five ways to cut a training step, and they cut different things. Data parallelism cuts the batch: every GPU runs the whole model on its own sequences and, once per step, they average gradients. Tensor parallelism cuts each weight matrix: two GPUs each hold half of every matrix and must add up their partial results inside every layer. Pipeline parallelism cuts the stack of blocks: each GPU runs a few consecutive blocks and hands activations to the next, like stations on an assembly line, which means stations sit idle while the line fills and drains. Context parallelism cuts one long sequence: later tokens need the keys and values of earlier ones, so those travel. Expert parallelism puts different experts on different GPUs, so tokens travel to their experts and back.',
  'Each cut buys memory and costs communication of a particular shape: one big all-reduce per step, small all-reduces inside every layer, point-to-point hand-offs, a ring of keys and values, or an all-to-all shuffle. Real runs stack several cuts, and the GPU count is the product of their degrees.',
  `Llama 3.1 405B ran ${LLAMA.tp}-way tensor, ${LLAMA.pp}-way pipeline and ${LLAMA.dp}-way data parallelism: ${LLAMA.tp} × ${LLAMA.pp} × ${LLAMA.dp} = ${int(LLAMA.gpus)} GPUs. That answers the hook: each of those GPUs holds one-eighth of every matrix in one-sixteenth of the blocks, which comes to ${formatStateGB(TOY_STATE.parts.total)} of state with optimizer states and gradients sharded (as Llama 3 did), and talks constantly to its ${LLAMA.tp - 1} tensor partners, now and then to its pipeline neighbors, and once per step to its ${LLAMA.dp - 1} data-parallel peers. How chatty each cut is decides where on the network it must live, which is the next lesson.`,
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{all-reduce: } \htmlClass{hl-comm}{2\,\tfrac{N-1}{N}\,S}
\qquad
\text{all-gather, reduce-scatter, all-to-all: } \htmlClass{hl-comm}{\tfrac{N-1}{N}\,S}
\qquad\text{bytes sent per GPU}` },
  { tex: tex`\text{TP (Megatron MLP): } Y = \text{act}(X A),\; A = [A_1 \mid A_2]
\quad\Rightarrow\quad Z = Y B = Y_1 B_1 + Y_2 B_2 \ \ (\text{one all-reduce})` },
  { tex: tex`\htmlClass{hl-bubble}{\text{bubble}} = \frac{p-1}{m+p-1}\ (\text{GPipe, 1F1B})
\qquad \frac{(p-1)/v}{m + (p-1)/v}\ (\text{interleaved, } v \text{ chunks per GPU})` },
  { tex: tex`\text{bubble time (DeepSeek-V3 Table 2): }\;
\text{1F1B } (p-1)(F+B) \quad \text{ZB1P } (p-1)(F+B-2W) \quad
\text{DualPipe } \big(\tfrac{p}{2}-1\big)(F\&B + B - 3W)` },
  { tex: tex`\#\text{GPUs} = \text{TP}\times\text{CP}\times\text{PP}\times\text{DP}
\qquad \text{(EP usually reuses the data- or tensor-parallel GPUs)}`, note: 'Worked: 8 × 1 × 16 × 64 = 8,192.' },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: X [tokens × d], A [d × h] split into A₁, A₂ [d × h/2]; B [h × d] split into B₁, B₂ [h/2 × d] (toy: d = 8, h = 16). p stages, m micro-batches, v virtual stages; F, B, W = time of a forward chunk, a backward-for-inputs chunk, a backward-for-weights chunk; F&B = an overlapped forward-and-backward pair. Hover a highlighted term to outline its glyph on the stage: the "sent per GPU" counter and the arrows are the communication, the hatched idle cells are the bubble.',
]);

const TAKEAWAYS = Object.freeze([
  'Five cuts, five kinds of traffic: data (one gradient all-reduce per step), tensor (an all-reduce of partial sums inside every layer), pipeline (activations handed between stages), context (keys and values along the sequence forward, their gradients back), expert (an all-to-all to the experts and back, where the busiest GPU sets the pace).',
  'Pipelines idle while they fill and drain: the bubble is (p − 1)/(m + p − 1). 1F1B caps activation memory at p micro-batches without changing the bubble; interleaving, zero-bubble and DualPipe shrink the bubble itself (formulas in the math panel; not demonstrated here); DualPipe\'s price is a second copy of the parameters.',
  `Real runs multiply cuts: Llama 3.1 405B's ${LLAMA.tp} × ${LLAMA.pp} × ${LLAMA.dp} = ${int(LLAMA.gpus)} H100s hold ${formatStateGB(TOY_STATE.parts.total)} of state each; DeepSeek-V3 skipped tensor parallelism entirely. Where each cut runs on the network is the next page.`,
]);

const FURTHER = Object.freeze([
  { title: 'Hugging Face, The Ultra-Scale Playbook', href: 'https://huggingface.co/spaces/nanotron/ultrascale-playbook', note: '5D parallelism, interactive' },
  { title: 'Lilian Weng, How to Train Really Large Models on Many GPUs', href: 'https://lilianweng.github.io/posts/2021-09-25-train-large/', note: 'the classic survey of the five cuts' },
  { title: 'DeepSeek, DualPipe', href: 'https://github.com/deepseek-ai/DualPipe', note: 'the bidirectional pipeline schedule, with code' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(HOOK),
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Splitting a model across GPUs: the five cuts of a training step, in eleven steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The toy model\'s sequences, partial sums and routes are small stand-ins; the byte counts and the schedules are exact.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Schedule a pipeline, then count the GPUs',
      intro: 'Panel A draws the pipeline schedule for any stage and micro-batch count and reads off its bubble. Panel B multiplies the degrees of a real run into a GPU count and the training state each GPU holds.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows() },
    takeaways: TAKEAWAYS,
    links: { next: ['cluster-topology', 'disaggregation'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
