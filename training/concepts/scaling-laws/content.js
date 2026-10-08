// scaling-laws: the complete lesson (storyboard docs/storyboards/scaling-laws.md §3, §5, §7-§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { factRows, belowFor } from './facts.js';

const SLUG = 'scaling-laws';

const HOOK = 'With a fixed compute budget, should you train a bigger model on fewer tokens or a smaller one on more? And why do 2026 labs pick "far more tokens" anyway?';

const INTUITION = Object.freeze([
  "Training compute is easy to estimate: each token passes through every active parameter once forward (about 2 FLOPs per parameter) and twice as much backward, so a run costs about 6 × parameters × tokens. DeepSeek-V4-Pro's {deepseek-v4-pro.active_params|count} active parameters on {deepseek-v4-pro.pretrain_tokens|count} tokens come to about 10²⁵ FLOPs. For a fixed budget you can spend it on a bigger model that sees fewer tokens or a smaller one that sees more.",
  'Scaling laws answer which split gives the lowest loss. Researchers trained hundreds of small models, fitted loss as a smooth function of parameters and tokens, and read off the minimum. The 2020 laws said "mostly make it bigger"; the 2022 Chinchilla work found that parameters and tokens should grow together, at about 20 tokens per parameter. Too big and the model is under-fed; too small and it cannot hold what the data offers.',
  'But compute-optimal only minimizes the cost of training. A deployed model also costs about 2 FLOPs per parameter for every token it serves, and a popular model serves far more tokens than it was trained on. So labs over-train: they pick a smaller model and feed it many more tokens to reach the same loss. They pay more for training and save much more on serving. That is why 2026 open models sit at hundreds to thousands of tokens per active parameter.',
  'The other lever is getting more out of each token. Muon, the optimizer behind DeepSeek-V4, GLM-5 and Kimi K3, replaces AdamW for most weight matrices. It adjusts each update so that every direction in it moves by about the same amount, rather than letting one dominant direction take the whole step. The cost is a few extra matrix multiplies per step.',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`C \approx 6\,\htmlClass{hl-n}{N}\,\htmlClass{hl-d}{D},\qquad L(N, D) = E + \frac{A}{\htmlClass{hl-n}{N}^{\alpha}} + \frac{B}{\htmlClass{hl-d}{D}^{\beta}} \quad (E = 1.8172,\ A = 482.01,\ B = 2085.43,\ \alpha = 0.3478,\ \beta = 0.3658)` },
  { tex: tex`N^*(C) = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}},\quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\quad D^* = \frac{C}{6N^*}` },
  { tex: tex`C_{\text{life}}(N) = 6\,N\,D(N) + 2\,N\,D_{\text{serve}},\qquad D(N) = \Big(\frac{B}{L^* - E - A N^{-\alpha}}\Big)^{1/\beta}` },
  { tex: tex`\text{Muon (per weight matrix):}\quad X_0 = \frac{M}{\lVert M \rVert_F},\quad X_{k+1} = a X_k + b\,(X_k X_k^\top) X_k + c\,(X_k X_k^\top)^2 X_k \;\Rightarrow\; \sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: N, D, C are scalars; M is one weight matrix\'s momentum [d_out × d_in]; the polynomial acts on each singular value σ separately, which is why the page can show it on two numbers. Hover a highlighted term to outline its glyph on the stage: hl-n is model size (the x axis and the Model size slider), hl-d is tokens (the readouts).',
  '(a) The fit is for dense models; using active parameters for a Mixture of Experts is this page\'s stated convention, not part of the fit. (b) 6ND ignores attention\'s extra cost at long context. (c) DeepSeek-V4 runs 10 iterations, 8 with (3.4445, −4.7750, 2.0315) then 2 with (2, −1.5, 0.5), with momentum 0.95, weight decay 0.1 and the update\'s RMS rescaled to 0.18; AdamW is kept for the embedding, the output head and the norm weights.',
]);

const TAKEAWAYS = Object.freeze([
  'Training compute ≈ 6 × parameters × tokens; for a fixed budget the loss is lowest near 20 tokens per parameter, with model and data scaled together.',
  'Serving costs about 2 FLOPs per active parameter per token, so a model that will be used heavily should be smaller and trained on far more tokens: 2026 open models sit at hundreds to thousands of tokens per active parameter.',
  'Muon replaces AdamW for most weight matrices at DeepSeek, GLM and Kimi: it equalizes each update\'s directions with a few cheap polynomial steps, at the cost of extra matrix multiplies per step.',
]);

const STAND_IN = "The loss values come from a published fit (Epoch AI's 2024 re-estimate of Chinchilla). Serving volumes are what-ifs, not any model's real traffic. Muon's update in frames 9–10 is a stand-in with two directions. In frame 2, block width is model size, printed beside each block.";
const TOY_INTRO = 'A fixed budget, a model-size slider along the valley, and a serving-volume what-if that moves the cheapest choice. Tokens per parameter always means pretraining tokens ÷ active parameters.';

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return Object.freeze({
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    intuitionNote: 'Learning-rate schedules, the other half of the recipe, live in [[midtraining]].',
    animation: {
      label: 'Compute, the valley, over-training and Muon',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      belowFor,
      standIn: STAND_IN,
    },
    toy: { title: 'Spend a compute budget.', intro: TOY_INTRO, mount },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: {
      framing: 'Compute is 6 × active parameters × tokens, so a model card\'s two numbers place it on the tokens-per-parameter ladder. Counted per active parameter, the 2026 open models sit far past Chinchilla\'s 20.',
      rows: factRows(data),
    },
    takeaways: TAKEAWAYS,
    links: {
      next: ['scale-reliability'],
      further: [
        { title: 'Google, How to Scale Your Model', href: 'https://jax-ml.github.io/scaling-book/', note: 'the systems side of the same budget' },
        { title: 'Hugging Face, The Smol Training Playbook', href: 'https://huggingfacetb-smol-training-playbook.hf.space/', note: 'what a real small-model run decided, and why' },
      ],
    },
  });
}

// For spec validation only; the page mounts lessonFor(ctx.data).
export const LESSON = lessonFor(null);
