// cluster-topology: the complete lesson (storyboard docs/storyboards/cluster-topology.md §3, §5, §7–§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { HOOK, FRAMING, intuition, factRows, belowFor } from './facts.js';
import { BASIS_LINE } from './toy-view.js';

const SLUG = 'cluster-topology';
const tex = String.raw;

const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{ratio} = \frac{\htmlClass{hl-comm}{\text{bytes sent per GPU}} / \htmlClass{hl-link}{\text{link}}}{\htmlClass{hl-flops}{\text{FLOPs per GPU}} / P_{\text{peak}}}` },
  {
    tex: tex`\text{TP: } \frac{16\frac{t-1}{t}\,sbh / L}{72\,sbh^2 / (t P)} = \frac{2(t-1)\,P}{9\,h\,L}
\qquad
\text{PP: } \frac{p\,P}{18\,h\,n_{\text{layers}}\,L}
\qquad
\text{DP: } \frac{2(N-1)\,P}{3\,N\,T\,L}
\qquad
\text{EP (DeepSeek-V4): hidden if } \frac{P}{L} \le 6144`,
  },
]);

const MATH_NOTES = Object.freeze([
  'Shapes and symbols: t, p, N = tensor, pipeline, data degrees; s, b, h = tokens per sequence, micro-batch, d_model (GPT-3: h = 12,288, 96 blocks); T = tokens per data-parallel replica per step; P = peak FLOP/s; L = link bytes/s per direction. Hover a highlighted term to outline its glyph on the stage.',
  'Assumptions: BF16 activations and gradients (2 bytes); ring collectives; basis: one full training step (forward 24·s·b·h² matmul FLOPs per layer, backward twice that; attention scores ignored); 6 FLOPs per parameter per token per step. The comm lane is hl-comm, the link label under the dot is hl-link, the compute lane is hl-flops.',
]);

const TAKEAWAYS = Object.freeze([
  'A cluster is fast islands (NVLink: 8 GPUs per server, 72 per NVL72 rack, 450–900 GB/s each way per GPU) on a slower network (50–100 GB/s per GPU), and big clusters thin the network further between pods (Llama 3: 1:7).',
  'Place by whether traffic blocks compute: tensor parallelism (every layer, on the critical path) inside the island; pipeline (tiny hand-offs) and data parallelism (one overlappable sync per step) on the network, data outermost. Faster chips make this harder, since compute outgrew links.',
  'Expert all-to-all hides only inside the fast island (DeepSeek-V4\'s rule: about 1 GB/s per 6 TFLOPS); DeepSeek exploits rail-optimized networks to route it, and NVIDIA\'s Nemotron 3 report keeps expert groups inside one rack.',
]);

const FURTHER = Object.freeze([
  { title: 'How to Scale Your Model, training chapter (Google)', href: 'https://jax-ml.github.io/scaling-book/training/', note: 'when each parallelism stays compute-bound' },
  { title: 'ML Engineering Open Book, networking (Stas Bekman)', href: 'https://github.com/stas00/ml-engineering', note: 'links, switches and collectives in practice' },
  { title: 'DeepEP (DeepSeek)', href: 'https://github.com/deepseek-ai/DeepEP', note: 'expert-parallel all-to-all kernels' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: intuition(data).map(fill),
    animation: {
      label: 'Two network layers, what each parallelism sends over them, and where each one runs',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'Frames 3 to 5 and 9 to 10 draw the page\'s own ratios for GPT-3\'s shape: one full training step, compute at the chip\'s dense BF16 peak, so real ratios are smaller. Tokens per replica (262,144) is a stand-in.',
      belowFor: (index) => belowFor(data, index).map(fill),
    },
    toy: {
      title: 'Put a cut on a link.',
      intro: `Pick a system, a parallelism and its degree, and whether it runs inside the NVLink domain or over the network; read the communication time as a share of the compute time. ${BASIS_LINE}`,
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['scale-reliability'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
