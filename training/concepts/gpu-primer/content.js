// gpu-primer: the complete lesson (storyboard docs/storyboards/gpu-primer.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/hardware.json and models.json; LESSON is the same lesson with no data.
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { hook, intuition, takeaways, framing, factRows, PROSE, BELOW } from './facts.js';

const SLUG = 'gpu-primer';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-int}{I} = \frac{\text{FLOPs}}{\text{bytes}} \qquad \text{matmul } X_{[m \times k]} W_{[k \times n]}:\quad I = \frac{2mkn}{b\,(mk + kn + mn)} \;\approx\; \frac{2m}{b} \ \ (m \ll k, n;\ \text{BF16: } I \approx m)` },
  { tex: tex`\htmlClass{hl-ridge}{I^{*}} = \frac{\htmlClass{hl-peak}{P_{\text{peak}}}}{\htmlClass{hl-bw}{\text{BW}}} \qquad \text{attainable} = \min\!\big(\htmlClass{hl-peak}{P_{\text{peak}}},\; \htmlClass{hl-int}{I}\cdot \htmlClass{hl-bw}{\text{BW}}\big) \qquad t = \max\!\Big(\tfrac{\text{FLOPs}}{P_{\text{peak}}},\; \tfrac{\text{bytes}}{\text{BW}}\Big)` },
  { tex: tex`m^{*} = \frac{I^{*} b\, k n}{2kn - I^{*} b\,(k + n)} \qquad \text{bits per number (block-scaled)} = \text{bits} + \frac{\text{scale bits}}{\text{block size}}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: X [m × k] (tokens × d), W [k × n], Y [m × n]; m tokens, b = bytes per number (BF16 → I ≈ m; FP8 → I ≈ 2m). Worked: H100 BF16, k = n = 8,192 → I* = 295.2, m* = 318.2.',
  'Hover a highlighted term to outline it on the stage: I is the followed dot and the intensity cell, P_peak the flat roof, BW the sloped roof and the HBM arrow, I* the bend\'s label. Real kernels add launch and latency overheads, so at a few hundred bytes (frame 4) the roofline is a bound, not a prediction.',
]);

const FURTHER = Object.freeze([
  { title: 'How to Scale Your Model: the roofline chapter (Google)', href: 'https://jax-ml.github.io/scaling-book/roofline/', note: 'the same roofline, worked for TPUs and GPUs' },
  { title: 'Making Deep Learning Go Brrrr (Horace He)', href: 'https://horace.io/brrr_intro.html', note: 'compute, memory bandwidth and overhead, with examples' },
  { title: 'GPU Glossary (Modal)', href: 'https://modal.com/gpu-glossary', note: 'every GPU term on this page, defined' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(hook),
    intuition: intuition(data).map(fill),
    intuitionNote: 'The multiply this page follows is the last one of [[attention]] (W_O); that lesson is optional background.',
    animation: {
      label: 'A GPU for LLM people: compute, memory and the roofline in eleven steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The 8 × 8 multiply and the 8,192 × 8,192 weight matrix are stand-ins for one real layer; the chips\' numbers are vendor dense peaks, not measured speeds.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Where does this multiply sit on the roof?',
      intro: 'One matmul, X [tokens × 8,192] · W [8,192 × 8,192], on a chip and in a format you pick. The 8,192 × 8,192 weight matrix is a stand-in for one real layer\'s weight; the chips\' numbers are vendor dense peaks, not measured speeds.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: factRows(data), prose: PROSE },
    takeaways: takeaways(data).map(fill),
    links: { next: ['training-memory', 'prefill-decode', 'quantization'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
