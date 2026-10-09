// multimodal: the complete lesson (storyboard docs/storyboards/multimodal.md §3, §5, §7–§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FRAMING, ROWS, BELOW } from './facts.js';
import { modelFacts } from './format.js';

const SLUG = 'multimodal';

const HOOK = 'A language model only ever multiplies vectors. How does a photo become something it can read, and why can one picture cost as many tokens as a short story?';

const INTUITION = Object.freeze([
  'An image is cut into a grid of small squares, patches, {kimi-k3.patch_size} pixels on a side in Kimi K3 and MiniMax-M3. Each patch\'s pixels are flattened and multiplied by a matrix into one vector, and a vision encoder (a transformer of its own, in which all patches of an image can see each other) turns those vectors into descriptions of what each patch shows in context. Position works differently than for text: an image has rows and columns, so instead of one running count, models typically give each patch its row and column (and its frame, for video) through a 2D or 3D version of RoPE. Neighboring patches are then merged, {kimi-k3.vision_merge} by {kimi-k3.vision_merge} patches into one in Kimi K3, and a small projector maps each merged vector to the language model\'s width. Those vectors join the residual stream exactly where words do.',
  'From there on nothing is special. Words after the image attend to the image tokens through the usual causal mask, and the blocks treat them like any other position. The cost is the count. Tokens grow with the area of the image divided by the patch area and the merge, so doubling an image\'s side quadruples its tokens. Models keep each image\'s own shape (dynamic resolution) so they don\'t waste tokens on padding, and video pays per frame, so encoders also pool over time.',
  'Two ways to build this are in use. The older adapter recipe bolts a pretrained encoder and projector onto a finished language model. The native recipe trains the encoder, projector and backbone on mixed text and images from the first step; it still has an encoder, it just learns everything together. And not every frontier model sees at all: several 2026 leaders are text-only.',
]);

const STAND_IN = 'The 16-pixel image, the 4-pixel patches and the width of 8 are toy sizes; real models use {kimi-k3.patch_size}-pixel patches on images up to thousands of pixels wide. Token counts are exact.';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{patches} = \frac{W}{p}\cdot\frac{H}{p},\qquad \htmlClass{hl-tok}{\text{tokens per image}} = \frac{W}{p}\cdot\frac{H}{p}\cdot\frac{1}{\htmlClass{hl-merge}{m}^2},\qquad \text{video: } \times\ \text{frames}` },
  { tex: tex`\text{worked: } \frac{1008}{14}\cdot\frac{1008}{14}\cdot\frac{1}{2^2} = 72 \cdot 72 \cdot \tfrac{1}{4} = 1{,}296` },
  { tex: tex`z_i = \operatorname{flatten}(\text{patch}_i)\, W_{\text{patch}},\quad Z = \operatorname{ViT}(z_1, \dots, z_N),\quad \htmlClass{hl-proj}{x^{\text{img}}_j} = \operatorname{MLP}\big([\,Z_a \,\|\, Z_b \,\|\, Z_c \,\|\, Z_d\,]\big)\in\mathbb{R}^{d_{\text{model}}}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes (toy in parentheses): patch [p × p × channels] (4 × 4 × 1 grey); W_patch [p²·channels × d_vit] (16 × 8); Z [N × d_vit] (16 × 8); a merged group [m²·d_vit] (32); projector output [d_model] (8). Real encoders are wider (MiniMax-M3\'s ViT: width {minimax-m3.vision_width}, {minimax-m3.vision_encoder_layers} layers). Patch positions are 2D (row, column; plus time for video); Qwen2-VL\'s M-RoPE and MiniMax-M3\'s 3D RoPE extend [[rope]]\'s rotation to those axes. Hover a highlighted term to outline its mark on the stage.',
]);

const TAKEAWAYS = Object.freeze([
  'An image becomes tokens through patches → patch embedding → vision encoder → merge → projector; after that the language model treats image tokens like words (frames 1–6).',
  'Tokens per image = (width ÷ patch) × (height ÷ patch) ÷ merge²: a 1,008-pixel square is 1,296 tokens in Kimi K3, its largest input 16,384, and video multiplies by frames (frames 7–9, try-this).',
  '"Native" means trained together from step 0, not "no encoder"; and several 2026 leaders are text-only (frame 10, the table above).',
]);

const FURTHER = Object.freeze([
  { title: 'ViT-Explainer (arXiv 2604.02182)', href: 'https://arxiv.org/abs/2604.02182', note: 'an interactive walkthrough of the vision transformer pipeline' },
  { title: 'Liu et al., LLaVA (Visual Instruction Tuning)', href: 'https://arxiv.org/abs/2304.08485', note: 'the adapter recipe' },
  { title: 'Wang et al., Qwen2-VL', href: 'https://arxiv.org/abs/2409.12191', note: 'dynamic resolution and M-RoPE' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  const facts = modelFacts(data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    intuitionNote: 'How an image looks inside the stream is drawn on [[decoder-anatomy]]; prefill and decode are covered in [[prefill-decode]].',
    animation: {
      label: 'An image becomes tokens: patches, vectors, merge, projector, then what it costs',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: (index, progress, stage) => render(index, progress, stage, facts),
      standIn: fill(STAND_IN),
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'What does a picture cost?',
      intro: 'It opens on Kimi K3\'s settings and the phone photo from the animation; one chip loads the toy image or a 3 × 3 merge.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: ROWS },
    takeaways: TAKEAWAYS,
    links: { next: ['model-card'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
