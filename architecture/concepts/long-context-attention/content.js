// The long-context-attention lesson (docs/storyboards/long-context-attention.md): every word on the page, plus the stage and the toy.
// No DOM at import time, so the node page test can import it. lessonFor(data) fills every dated number from data/models.json.
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { formatBytes } from '@math/core.js';
import { FRAMING, ROWS, belowFor } from './facts.js';
import { gptOssCache } from './real-scale.js';
import { CONTEXTS } from './numbers.js';
import { int } from './format.js';

export { CAPTIONS };

const HOOK = 'At a million tokens, full attention would make each new token read a million keys in every layer and keep a million entries per conversation. How do 2026 models get there, and what do they give up?';

const INTUITION = Object.freeze([
  'Full attention has two costs that grow with the conversation. Every new token\'s query reads every stored key, so the reading per token grows with the length. And the cache keeps one entry for every token, so the memory grows too. [[kv-compression]] made each entry smaller. This page changes how many entries are read, how many are kept, or both.',
  'The simplest move is a window: a layer only looks at its last few thousand tokens, or its last {gpt-oss-120b.window}, and forgets the rest, so both costs stop growing. To keep long-range ability, models interleave window layers with full ones. A window needs a "nothing here" option, because softmax must put all its weight somewhere; a learned sink logit gives it one. Sparse attention keeps everything but reads only the most promising entries, picked by a cheap scorer: less reading, the same memory. Compressed attention merges several tokens\' keys and values into one entry, which cuts both.',
  'The other route replaces most attention layers with linear attention: a fixed-size memory matrix that every token writes into and every query reads from, so it never grows. It is cheap and blurry, so models keep one full-attention layer in four for exact recall. Each trick costs something: windows forget, sparse reads can miss, compression blurs neighbors, and retrofitting windows onto a model trained with full attention went badly in GLM-5\'s tests.',
]);

const STAND_IN = 'Sizes are toy choices (16 tokens, window 4, top 4); the indexer scores are seeded random numbers and the window scores are hand-picked. Real models use windows of {gpt-oss-120b.window} to {gemma-3.window} and top-k of {deepseek-v4-pro.csa_top_k} to {glm-5.3.sparse_top_k}.';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  {
    tex: tex`\text{window } w:\quad \htmlClass{hl-read}{\text{reads}_t} = \min(t, w),\qquad \htmlClass{hl-store}{\text{stored}} = \min(t, w) \qquad \text{sparse top-}k:\quad \text{reads}_t = \min(t, k),\quad \text{stored} = t`,
    note: 'Token t reads at most w (window) or k (sparse) entries; a window keeps w entries, sparse attention keeps all t.',
  },
  {
    tex: tex`\text{compressed (merge } m\text{)}:\quad \text{stored} = \tfrac{t}{m} + w,\qquad \text{reads}_t = \min\!\big(\tfrac{t}{m}, k\big) + w`,
    note: 'm tokens per merged entry, w the window, k the merged entries read. The toy: t = 16, m = 4, w = 4, k = 1 stores 8 and reads 5.',
  },
  {
    tex: tex`\text{sink:}\quad a_j = \frac{e^{s_j}}{e^{\htmlClass{hl-sink}{s_{\varnothing}}} + \sum_{i} e^{s_i}},\qquad \text{worked: } \frac{e^{1}}{e^{1} + e^{-1} + e^{-0.5} + e^{-1} + e^{-0.75}} = 0.600`,
    note: 'Window scores −1, −0.5, −1, −0.75 and a sink logit of 1: the sink takes 0.600 and the four tokens share 0.400.',
  },
  {
    tex: tex`\text{linear (gated): } \htmlClass{hl-state}{S_t} = \alpha\, S_{t-1} + v_t k_t^{\top},\qquad o_t = S_t\, q_t \qquad \text{delta rule (Gated DeltaNet): } S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^{\top}\big) + \beta_t\, v_t k_t^{\top}`,
    note: 'S [d_v × d_k] (4 × 4); q, k, v [d_head] (4). The toy has no normalization and no softmax, so its output is the raw-score blend of the values; real layers add the delta rule (erase what a key already holds before writing), and Kimi\'s gate is per channel instead of one α.',
  },
]);

// The stack-cache formula, with gpt-oss-120b's worked line templated from the data (never typed).
function cacheBlock(data) {
  const texInt = (n) => int(n).replace(/,/g, '{,}');
  const g = gptOssCache(data, CONTEXTS[0]);
  const worked = g ? tex`,\qquad \text{gpt-oss: } ${g.fullLayers} \cdot ${texInt(g.bytesPerTokenPerLayer)} = ${texInt(g.perToken)}\ \text{B} + ${formatBytes(g.fixed)}\ \text{fixed}` : '';
  return {
    tex: tex`\text{cache per token (stack)} = \sum_{\text{full layers}} b_{\ell} \;+\; \frac{1}{\text{tokens}}\Big(\sum_{\text{window}} w\, b_{\ell} + \sum_{\text{linear}} \text{state}_{\ell}\Big)${worked}`,
    note: 'b_ℓ is the bytes one layer stores per token (2 · KV heads · head size · bytes per number). The worked line is the toy\'s real-scale readout for gpt-oss-120b: its full layers grow the cache, its window layers add a fixed amount.',
  };
}

const TAKEAWAYS = Object.freeze([
  'Full attention costs grow two ways with length: reads per token and entries stored. Windows cap both; sparse attention cuts only the reads; merging tokens cuts both (frames 1–7, try-this 1).',
  'A learned sink logit lets a row put its weight on "nothing", which keeps window layers stable; local and global layers are interleaved so the full layers still reach back (frames 3–4, try-this 2).',
  'Linear attention keeps a fixed state that blends and fades; hybrids keep one full layer in four for exact recall. The 2026 field is split between sparse or compressed softmax attention and linear hybrids, both on top of staged length training (and, on the softmax route, a large RoPE base) (frames 8–10, try-this 3).',
]);

const FURTHER = Object.freeze([
  { title: 'Sebastian Raschka, LLM Architecture Gallery', href: 'https://sebastianraschka.com/llm-architecture-gallery/', note: 'sliding-window, sparse and hybrid layers, model by model' },
  { title: 'Xiao et al., Efficient Streaming Language Models with Attention Sinks', href: 'https://arxiv.org/abs/2309.17453', note: 'why a window needs a sink' },
  { title: 'Yang et al., Gated Delta Networks', href: 'https://arxiv.org/abs/2412.06464', note: 'the delta rule behind the linear layers' },
]);

// The whole lesson. Pure; `data` fills the dated numbers (template rule 1).
export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: 'long-context-attention',
    hook: HOOK,
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Ten ways to read and keep less: window, sink, sparse, compressed, linear',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: fill(STAND_IN),
      belowFor: (index) => (belowFor(data)[index] ?? []).map(fill),
    },
    toy: {
      title: 'What does each trick read, and what does it keep?',
      intro: fill(STAND_IN),
      mount,
    },
    math: { blocks: [...MATH_BLOCKS, cacheBlock(data)] },
    facts: { framing: FRAMING, rows: ROWS },
    takeaways: TAKEAWAYS,
    links: { next: ['model-card'], further: FURTHER },
  };
}

// For spec validation only; the page mounts lessonFor(ctx.data).
export const LESSON = lessonFor(null);
