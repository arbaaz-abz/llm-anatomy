// moe: the complete lesson (storyboard docs/storyboards/moe.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { activeRange, belowFor, factRows, framing, hookFor } from './facts.js';

const SLUG = 'moe';

const intuitionFor = (data) => Object.freeze([
  'In a dense model, every token goes through the same MLP. A Mixture of Experts swaps that MLP for many small MLPs, the experts, plus a router: a small matrix that gives each expert a score for this token. The token goes only to the top few, and their outputs are blended by weights computed from the same scores. The rest of the experts sit idle for this token, though another token may choose them.',
  `That is how total and active parameters come apart. Add experts and the model can store more, because every expert is a full set of weights in memory; the work per token stays the same as long as the number chosen, and their size, stays the same. 2026 models push this hard: hundreds of small experts, a handful chosen, ${activeRange(data)}% of the parameters active per token. Cutting experts smaller and choosing more of them keeps the work fixed and gives the router many more combinations to pick from. One or two shared experts that every token uses hold what all tokens need, so the routed ones can differ.`,
  'The cost is in the routing. A router that learns to favor a few experts overloads them while others sit idle, and in serving and training the busiest expert sets the pace. DeepSeek\'s fix is a small per-expert bias added to the scores only when choosing: raise it for idle experts, lower it for busy ones, a little every step. The weights in memory, the communication between GPUs that hold different experts ([[parallelism]]), and the fact that a busy server ends up touching every expert anyway are the rest of the bill.',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-score}{s} = h\, W_{\text{router}} \in \mathbb{R}^{E},\qquad \mathcal{T} = \operatorname{top\text{-}k}\big(\htmlClass{hl-score}{s} + \htmlClass{hl-bias}{b}\big),\qquad \htmlClass{hl-gate}{g_e} = \frac{e^{s_e}}{\sum_{j \in \mathcal{T}} e^{s_j}}\ \ (e \in \mathcal{T})` },
  { tex: tex`\operatorname{MoE}(h) = \sum_{e \in \mathcal{T}} \htmlClass{hl-gate}{g_e}\, \operatorname{MLP}_e(h) \;+\; \sum_{\text{shared}} \operatorname{MLP}_{s}(h) \qquad \text{worked: } 0.622\, E_3(h) + 0.378\, E_6(h)` },
  { tex: tex`\text{per block: total} = (E + S)\cdot 3\,d\,h_e + d\,E,\qquad \text{active experts} = (k + S)\cdot 3\,d\,h_e \qquad \text{toy: } 8 \cdot 192 + 64 \ \text{vs}\ 2 \cdot 192 = 384` },
  { tex: tex`\htmlClass{hl-bias}{b_e} \leftarrow b_e + \gamma\, \operatorname{sign}\!\Big(\tfrac{T k}{E} - \text{load}_e\Big),\qquad \text{imbalance} = \frac{\max_e \text{load}_e}{T k / E}` },
  { tex: tex`\text{combinations} = \binom{E}{k}:\quad \binom{8}{2} = 28,\ \ \binom{16}{4} = 1{,}820` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes (toy): h [d_model] (8); W_router [d_model × E] (8 × 8); each expert W_in, W_gate [8 × h_e] and W_out [h_e × 8] with h_e = 8 (192 parameters). The gate here is a softmax over the chosen experts, as in Mixtral and Qwen3; DeepSeek-V3 uses sigmoid affinities normalized over the chosen ones, and DeepSeek-V4 a "sqrt-softplus" affinity.',
  'Hover a highlighted term to outline its glyph on the stage: scores and the router row, the bias row (frame 9), the gate cells (frame 3).',
]);

const TAKEAWAYS = Object.freeze([
  'A router scores every expert for each token, the top k run, and their outputs are blended by gate weights from the same scores; the rest are skipped (frames 1–4).',
  'Every expert is stored, but each token runs only k of them: total grows with the expert count while the work per token stays put. Fine-grained experts keep the work fixed and multiply the choices; shared experts run for every token (frames 5–7, try-this 1–2).',
  'Routers drift toward a few experts and the busiest sets the pace. A per-expert bias, used only for choosing and nudged each step, evens the load without an extra loss; too large a nudge overshoots (frames 8–10, try-this 3).',
]);

const FURTHER = Object.freeze([
  { title: 'A Visual Guide to Mixture of Experts (Maarten Grootendorst)', href: 'https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts', note: 'the clearest illustrated tour of routers and experts' },
  { title: 'DeepSeekMoE (Dai et al.)', href: 'https://arxiv.org/abs/2401.06066', note: 'fine-grained and shared experts' },
  { title: 'Auxiliary-Loss-Free Load Balancing (Wang et al.)', href: 'https://arxiv.org/abs/2408.15664', note: 'the selection-only bias of frames 9 and 10' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  const below = belowFor(data);
  return {
    slug: SLUG,
    hook: fill(hookFor(data)),
    intuition: intuitionFor(data).map(fill),
    animation: {
      label: 'Mixture of Experts: routing one token, counting parameters and balancing the load, in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'Router scores are hand-picked for the four words and seeded random numbers for the batch; the 8-number vectors are stand-ins too. Parameter counts are exact.',
      belowFor: (index) => (below[index] ?? []).map(fill),
    },
    toy: {
      title: 'Route, count, rebalance.',
      intro: 'Panel A counts parameters for the animation\'s toy and lets you change it; panel B replays the balancing batch.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(data), rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['model-card', 'parallelism'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
