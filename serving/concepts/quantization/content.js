// quantization: the complete lesson (storyboard docs/storyboards/quantization.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data.
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { hook, intuition, takeaways, framing, factRows, BELOW } from './facts.js';

const SLUG = 'quantization';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-scale}{s} = \frac{\max_i |w_i|}{q_{\max}},\qquad \htmlClass{hl-code}{q_i} = \operatorname{round}_{\text{grid}}\!\left(\operatorname{clamp}\!\left(\frac{w_i}{\htmlClass{hl-scale}{s}}, -q_{\max}, q_{\max}\right)\right),\qquad \hat w_i = \htmlClass{hl-code}{q_i}\,\htmlClass{hl-scale}{s}` },
  { tex: tex`\text{INT4: } q_{\max}=7,\ \text{grid}=\mathbb{Z} \qquad \text{E2M1: grid} = \{0, 0.5, 1, 1.5, 2, 3, 4, 6\},\ q_{\max}=6 \qquad \text{MX: } \htmlClass{hl-scale}{s} = 2^{\lfloor \log_2 \max|w| \rfloor - 2}` },
  { tex: tex`\text{bits per weight} = b_{\text{elem}} + \frac{b_{\text{scale}}}{\text{block}} \quad(\text{NVFP4: } 4 + 8/16 = 4.5;\ \text{MXFP4: } 4 + 8/32 = 4.25), \qquad \text{bytes} = \text{params}\cdot\frac{\text{bits}}{8}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: a weight matrix [d_in × d_out] is cut into blocks of 16 or 32 consecutive values along one dimension, one scale each (DeepSeek-V4\'s FP8 weights use 128 × 128 tiles instead).',
  'Hover a highlighted term to outline it on the stage: s is the scale chips above each block and q the codes row; the restored row is ŵ. The MXFP4 rule is Algorithm 1 of the Microscaling paper; the NVFP4 scale in this toy (largest value over six, rounded to FP8) is a labeled stand-in.',
]);

const FURTHER = Object.freeze([
  { title: 'A Visual Guide to Quantization (Maarten Grootendorst)', href: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization', note: 'INT8, GPTQ and more, drawn' },
  { title: 'Introducing NVFP4 for efficient and accurate low-precision inference (NVIDIA)', href: 'https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference', note: 'E2M1 values, 16-value blocks, the E4M3 scale' },
  { title: 'FP8 KV cache (vLLM)', href: 'https://vllm.ai/blog/2026-04-22-fp8-kvcache', note: 'the needle test and the fix' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook,
    intuition: intuition(),
    animation: {
      label: 'Quantization: scale, round and multiply back, then what fewer bits do to a GPU, in eleven steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The eight weights are hand-picked, with one outlier on purpose; real blocks hold 16 or 32. The model-scale steps use Llama-3.1-70B on one GPU, a 2,048-token cache per user and vendor dense peaks, not measured speeds.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Round a block, then shrink a model',
      intro: 'The eight weights are hand-picked, with one outlier on purpose. Rounding is round-to-nearest; GPTQ and AWQ choose roundings more cleverly. The model panel uses Llama-3.1-70B on one GPU; the chips\' numbers are vendor dense peaks, not measured speeds.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: factRows() },
    takeaways: takeaways(),
    links: { next: ['serving-calculator'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
