// decoder-recap: the complete lesson (storyboard docs/storyboards/decoder-recap.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FACT_ROWS, framing, belowTexts } from './facts.js';
import { SHAPE_TEXT } from './format.js';

const SLUG = 'decoder-recap';

const HOOK = "GPT-3's block from 2020 and a 2026 block do the same job: normalize, attend, add; normalize, MLP, add. So what changed in each part, and why did each change win?";

const INTUITION = Object.freeze([
  "Lay GPT-3's block next to a 2026 one and almost every box has been replaced, but the wiring is the same. The changes come in three kinds. Some make training stable at depth and scale: normalizing before each half (GPT-3 already did), using the cheaper RMSNorm, and normalizing queries and keys so attention scores cannot blow up. Some make the model better per unit of compute: a gated MLP, and experts that let a model store far more than it uses per token. And many make it cheaper to run: queries share keys and values or read them from a small latent, position is a rotation instead of a fixed table, and some attention layers look only at a window or keep a fixed-size state.",
  "The pattern is that serving cost shaped the 2026 block. The parts that changed most are the ones that set what a model costs per token and per conversation: active parameters and the KV cache. Not every model takes every change: gpt-oss keeps biases, Gemma normalizes before and after, Kimi K3 drops positions entirely in some layers. And the newest changes are not in the block at all: models predict more than one token, train with a new optimizer and store weights in fewer bits.",
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\operatorname{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta` },
  { tex: tex`\htmlClass{hl-rms}{\operatorname{RMSNorm}(x)} = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_j x_j^2}}, \qquad \text{worked: } \operatorname{rms}(x_{\text{sat}}) = \sqrt{2.75/8} = 0.586` },
  { tex: tex`\operatorname{GELU\ MLP}(h) = \operatorname{GELU}(h W_{\text{in}})\, W_{\text{out}}` },
  { tex: tex`\htmlClass{hl-glu}{\operatorname{SwiGLU}(h)} = \big(\operatorname{SiLU}(h W_{\text{gate}}) \odot h W_{\text{in}}\big) W_{\text{out}}, \qquad 2 \cdot d \cdot 4d = 3 \cdot d \cdot \tfrac{8}{3}d = 8d^2` },
  { tex: tex`\htmlClass{hl-qk}{\text{QK-norm:}}\quad s_{ij} = \frac{\operatorname{RMSNorm}(q_i) \cdot \operatorname{RMSNorm}(k_j)}{\sqrt{d_{\text{head}}}}` },
  { tex: tex`|s_{ij}| \le \frac{d_{\text{head}}\,\gamma_q \gamma_k}{\sqrt{d_{\text{head}}}}` },
  { tex: tex`x \leftarrow x + \operatorname{Attn}(\operatorname{Norm}(x)),\quad x \leftarrow x + \operatorname{MLP}(\operatorname{Norm}(x))`, note: 'The pre-norm block: GPT-2 and GPT-3 onward; the original 2017 transformer normalized after the add.' },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: x [d_model] (8 here; 12,288 in GPT-3); q, k [d_head] (4); SwiGLU W_in, W_gate [d × h], W_out [h × d]. The QK-norm bound follows from each normalized vector having squared length d_head: with γ = 1 the dot product is at most d_head.',
  'Hover a highlighted term to outline its glyph on the stage: RMSNorm is frame 2\'s RMSNorm row, SwiGLU is frame 4\'s gated branches, QK-norm is frame 7\'s third row.',
]);

const TAKEAWAYS = Object.freeze([
  "The 2026 block has GPT-3's wiring: normalize, attend, add; normalize, MLP, add. Almost every box was swapped (frames 1 and 10).",
  'RMSNorm, SwiGLU and QK-norm are about stable, efficient training: same size, better behaved (frames 2, 4 and 7, try-this 1).',
  'RoPE, shared keys and values or latents, window and linear layers, and experts are about running cost: no position table, a smaller cache, less work per token (frames 3, 5, 6 and 9, try-this 2 and 3).',
]);

const FURTHER = Object.freeze([
  { title: 'Sebastian Raschka, LLM Architecture Gallery', href: 'https://sebastianraschka.com/llm-architecture-gallery/', note: 'a part-by-part comparison of 100+ models' },
  { title: 'Sebastian Raschka, The Big LLM Architecture Comparison', href: 'https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison', note: 'the same story in prose' },
  { title: 'Transformer Explainer', href: 'https://poloclub.github.io/transformer-explainer/', note: "see GPT-2's block, the baseline this page starts from" },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  const below = belowTexts();
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION,
    animation: {
      label: "From GPT-3's block to a 2026 block: ten swaps, one part per step",
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: "Rows are the earlier pages' hand-picked numbers; parameter and cache counts are exact for GPT-3's shape.",
      belowFor: (index) => (below[index] ?? []).map(fill),
    },
    toy: {
      title: 'Modernize GPT-3, one switch at a time.',
      intro: `Each switch applies to GPT-3's shape: ${SHAPE_TEXT}.`,
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: FACT_ROWS },
    takeaways: TAKEAWAYS,
    links: { next: ['moe', 'sampling', 'training-pipeline'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
