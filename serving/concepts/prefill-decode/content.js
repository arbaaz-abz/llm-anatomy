// prefill-decode: the complete lesson (storyboard docs/storyboards/prefill-decode.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/hardware.json, models.json and serving.json; LESSON is the same lesson with no data.
import { fillText } from '@shared/claims.js';
import { FORWARD_FLOPS_PER_PARAM_TOKEN } from '@math/serving.js';
import { FLOPS_PER_PARAM_TOKEN } from '@math/scale.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { hook, intuition, intuitionNote, takeaways, framing, factRows, BELOW } from './facts.js';

const SLUG = 'prefill-decode';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-compute}{t_{\text{math}}} = \frac{2\,N_{\text{active}}\cdot \text{tokens}}{\text{peak FLOP/s}} \qquad \htmlClass{hl-memory}{t_{\text{read}}} = \frac{W_{\text{bytes}} + \tfrac{2 N b}{d_{\text{model}}}\cdot\text{tokens} + \text{users}\cdot \text{context}\cdot \text{KV bytes/token}}{\text{HBM bandwidth}}` },
  { tex: tex`t_{\text{step}} = \max\!\left(\htmlClass{hl-compute}{t_{\text{math}}},\ \htmlClass{hl-memory}{t_{\text{read}}}\right) \qquad m^{*} = \frac{I^{*}\, b\, d^{2}}{2d^{2} - 2 I^{*} b\, d}\quad(\text{tokens to compute-bound, } I^{*} = \text{ridge})` },
  { tex: tex`\text{TPOT} = t_{\text{step}}, \qquad \text{tokens/s per user} = \frac{1}{t_{\text{step}}}, \qquad \text{tokens/s per GPU} = \frac{\text{users}}{t_{\text{step}}}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: each multiply is [tokens × 8,192] · [8,192 × 8,192]; with one token it is a matrix-vector product, which is why decode reuses nothing. b = bytes per element (1 for FP8, 2 for BF16). Ignored on this page: attention FLOPs (about 1% of a 1,000-token prefill for this model), KV writes, communication. Hover a highlighted term to outline it on the stage: t_math is the arithmetic row of the step bar, t_read its reading row.',
  `Every step on the serving pages is a forward pass on one GPU (inference has no backward pass), and ratios across mechanisms are compared per token on that basis. The factor ${FORWARD_FLOPS_PER_PARAM_TOKEN} in t_math is the forward-pass share of training's ${FLOPS_PER_PARAM_TOKEN} FLOPs per parameter per token. Step time is one engine iteration on one GPU, a floor; TPOT is the decode step time; tokens/s per GPU counts output tokens only; cache per user is context × KV bytes per token, from [[kv-cache]].`,
]);

const FURTHER = Object.freeze([
  { title: 'How to Scale Your Model: the inference chapter (Google)', href: 'https://jax-ml.github.io/scaling-book/', note: 'prefill, decode and batching worked as rooflines' },
  { title: 'InferenceX dashboard (SemiAnalysis)', href: 'https://inferencex.semianalysis.com/about', note: 'live per-user vs per-GPU curves on real hardware' },
  { title: 'LLM Inference Explained (Lynskey)', href: 'https://llm-inference-explained.vercel.app', note: 'an interactive walk through the same two phases' },
]);

const TOY_INTRO = 'Pick a phase and a load, and watch which side of the roofline you land on. The model is fixed: Llama-3.1-70B (70,000,000,000 parameters, d_model 8,192, 327,680 KV bytes per token in BF16). These are floors from bytes and FLOPs alone. Real engines fall short of the bandwidth floor (no efficiency figure is sourced, so none is printed); attention math, communication and kernel overheads are not modeled.';

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook,
    intuition: intuition.map(fill),
    intuitionNote,
    animation: {
      label: 'Prefill vs decode: one step\'s reading and arithmetic, then batching, in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The model is Llama-3.1-70B at a rounded 70,000,000,000 parameters, with every weight matrix treated as 8,192 × 8,192; every time is a floor from bytes and FLOPs (attention math, KV writes and communication are not modeled).',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: { title: 'Step-time calculator', intro: TOY_INTRO, mount },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing, rows: factRows(data) },
    takeaways: [...takeaways],
    links: { next: ['batching', 'speculative-decoding', 'quantization'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
