// kv-cache: the complete lesson (storyboard docs/storyboards/kv-cache.md §3, §5, §7–§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { HOOK, FRAMING, BELOW, intuition, factRows } from './facts.js';

const SLUG = 'kv-cache';
const tex = String.raw;

const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{decode step } t:\quad \htmlClass{hl-k}{K_{1:t}} = \big[\,\htmlClass{hl-cache}{K_{1:t-1}} \;;\; x_t W_K\,\big],\qquad \htmlClass{hl-v}{V_{1:t}} = \big[\,\htmlClass{hl-cache}{V_{1:t-1}} \;;\; x_t W_V\,\big],\qquad o_t = \operatorname{softmax}\!\Big(\tfrac{q_t \htmlClass{hl-k}{K_{1:t}}^{\top}}{\sqrt{d_{\text{head}}}}\Big)\htmlClass{hl-v}{V_{1:t}}` },
  { tex: tex`\text{positions computed: } \underbrace{\textstyle\sum_{t=P}^{P+G-1} t}_{\text{no cache}} \quad\text{vs}\quad \underbrace{P + (G-1)}_{\text{cache}}, \qquad \text{worked: } 4+5+6+7 = 22 \ \text{vs}\ 4 + 3 = 7` },
  { tex: tex`\htmlClass{hl-bytes}{\text{bytes per token}} = 2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b, \qquad \text{worked (GPT-3): } 2 \cdot 96 \cdot 96 \cdot 128 \cdot 2 = 4{,}718{,}592\ \text{B}` },
  { tex: tex`\text{cache} = \htmlClass{hl-bytes}{\text{bytes per token}} \times \text{tokens} \times \text{conversations}, \qquad \text{worked: } 327{,}680 \times 131{,}072 = 42.9\ \text{GB}` },
  { tex: tex`\text{latent attention (see MQA, GQA, MLA): } \text{bytes per token} = L\,(d_c + d_{\text{rope}})\,b, \qquad 61 \cdot 576 \cdot 2 = 70{,}272\ \text{B}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: K, V per head per layer [t × d_head] (5 × 4 here); per position the cache holds 2 · n_kv · d_head numbers per layer (2 · 2 · 4 = 16 here). P = prompt length, G = reply length, L = attention layers that keep a growing cache, n_kv = KV heads, b = bytes per number. Hover a highlighted term to outline its glyph on the stage.',
  'Reading the whole cache on every step is why decode is limited by memory bandwidth, not arithmetic: the full argument is in [[prefill-decode]].',
]);

const TAKEAWAYS = Object.freeze([
  'Earlier positions\' keys and values never change, so a model stores them: each decode step computes one new position and reads the stored rest (frames 2–3). The prompt fills the cache in one prefill pass.',
  'The cache removes recomputation, not reading: every step still reads every stored key and value, so a longer conversation means a bigger read per token (frame 5, try-this 1).',
  'Bytes per token = 2 × layers × KV heads × head size × bytes, times tokens, times conversations. At long context the cache, not the weights, decides how many conversations fit on a GPU; it fell from 4.72 MB per token in GPT-3 to a few kB in 2026 models (frames 6–9).',
]);

const FURTHER = Object.freeze([
  { title: 'LLM Inference Explained (Brendan Lynskey)', href: 'https://llm-inference-explained.vercel.app', note: 'the generation loop and the KV cache in long form' },
  { title: 'LLM Architecture Gallery (Sebastian Raschka)', href: 'https://sebastianraschka.com/llm-architecture-gallery/', note: 'a memory calculator across 100+ models' },
  { title: 'Continuous batching from first principles (Hugging Face)', href: 'https://huggingface.co/blog/continuous_batching', note: 'derives serving from attention plus the cache' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: intuition(data).map(fill),
    animation: {
      label: 'Decoding with and without a KV cache, then what the cache costs',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'K and V rows 1–4 are the numbers from [[attention]]; the rows for the later tokens are hand-picked. Every count and byte size is exact.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Count the work, then weigh the memory.',
      intro: 'Panel A counts the positions a reply costs with and without the cache; panel B weighs the cache for any model, context and number of conversations. Both share one state and open on the toy model from the animation.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['kv-compression', 'serving-overview'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
