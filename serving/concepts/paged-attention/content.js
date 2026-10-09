// paged-attention: the complete lesson (storyboard docs/storyboards/paged-attention.md §3, §5, §7–§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { HOOK, FRAMING, BELOW, intuition, factRows } from './facts.js';

const SLUG = 'paged-attention';
const tex = String.raw;

const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{blocks}(n) = \left\lceil \frac{n}{B} \right\rceil,\qquad \htmlClass{hl-waste}{w_{\text{paged}}(n)} = B\left\lceil \frac{n}{B} \right\rceil - n \;\le\; B-1` },
  { tex: tex`\htmlClass{hl-waste}{w_{\text{contig}}(n)} = L_{\max} - n \qquad \text{wasted \%} = \frac{\sum_{\text{running}} w}{\text{pool slots}}` },
  { tex: tex`\text{token } t \ (t = 1, 2, \dots):\quad \text{block} = \htmlClass{hl-table}{\text{table}}\!\left[\left\lfloor \tfrac{t-1}{B} \right\rfloor\right],\qquad \text{slot} = (t-1) \bmod B` },
  { tex: tex`\text{bytes per block} = B \cdot \underbrace{2\, L\, n_{kv}\, d_{\text{head}}\, b}_{\text{bytes per token (GQA)}} \qquad \text{bytes per token (MLA)} = L\,(d_c + d_{\text{rope}})\, b` },
  { tex: tex`\text{fork of } s \text{ samples: } \text{physical} = \left\lfloor \tfrac{P}{B} \right\rfloor + s\left(\left\lceil \tfrac{P+g}{B} \right\rceil - \left\lfloor \tfrac{P}{B} \right\rfloor\right)\ \ (g \ge 1)` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: one slot is one K tile and one V tile of [n_kv × d_head] per layer, so a block of B tokens is [B × 2 L n_kv d_head] elements. Indexing: tokens t count from 1 on screen; blocks and slots count from 0. Color links: the w terms light the reserved-but-empty part of the usage bar; table lights the block table.',
  'P is the prompt length, g the tokens generated so far, B the block size, L_max the strip the old scheme reserved, s the number of samples. Hover a highlighted term to outline its glyph on the stage.',
]);

const TAKEAWAYS = Object.freeze([
  'KV memory per request grows one token at a time, and nobody knows the final length; reserving the maximum up front left {sv:pagedattention.waste_before_pct|raw}% of KV memory empty in 2023 systems and capped the batch (frames 2–3).',
  'Fixed-size blocks plus a per-request block table (an OS page table for KV) allocate on demand: waste is at most one partial block per request, freed blocks are reused at once, and the 2023 paper measured {sv:pagedattention.throughput_gain|raw} times more throughput at the same latency. Block size is the dial (try-this 2), and the price is over-commit: a dry pool means preemption ([[batching]]).',
  'A block is a unit of sharing: reference counts and copy-on-write let several sequences point at one prompt\'s blocks (frames 8–9); keeping full blocks after the request ends is prefix caching, which you will meet in [[prefix-caching]].',
]);

const FURTHER = Object.freeze([
  { title: 'Easy, Fast, and Cheap LLM Serving with PagedAttention (vLLM)', href: 'https://vllm.ai/blog/2023-06-20-vllm', note: 'the launch post' },
  { title: 'Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., SOSP \'23)', href: 'https://arxiv.org/abs/2309.06180', note: 'the paper' },
  { title: 'LLM Inference Explained (Brendan Lynskey)', href: 'https://llm-inference-explained.vercel.app', note: 'the long form' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(HOOK),
    intuition: intuition().map(fill),
    animation: {
      label: 'One KV pool, reserved per request and then paged in blocks',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The four requests, the 48-slot pool and the 16-token limit are hand-picked so every number can be counted; real prompts are thousands of tokens. Every step is one decode pass on one GPU, counted rather than timed; [[batching]] times the same steps.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Slide the block size and watch the waste.',
      intro: 'The four requests and the 16-token limit are hand-picked so you can count; real prompts are thousands of tokens. Both lanes always show the same requests: change a control and both redraw from the same functions.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows(data) },
    takeaways: TAKEAWAYS.map(fill),
    links: { next: ['prefix-caching'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
