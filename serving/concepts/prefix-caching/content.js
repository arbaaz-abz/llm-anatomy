// prefix-caching: the complete lesson (storyboard docs/storyboards/prefix-caching.md §3, §5, §7–§10). Pure, no DOM at import.
// lessonFor(data) fills the dated text from data/serving.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { renderFor } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { FRAMING, factRows, stageText } from './facts.js';

const SLUG = 'prefix-caching';

const HOOK = 'Why do API providers charge about a tenth as much (or less) for input tokens they have seen before, and why does putting them in the cache cost extra?';

export const INTUITION = Object.freeze([
  'In [[paged-attention]] a finished request handed its blocks back to the pool. Prefix caching keeps them instead. Each full block stays in GPU memory with a label: its own tokens plus the label of the block before it. That chain matters, because a token\'s keys and values depend on every earlier token. The same four words after a different start are a different block. When a new request arrives, the server walks its prompt block by block from the first token, and every block it finds already labeled is a hit: its keys and values are already there, so prefill skips those tokens entirely.',
  'System prompts, few-shot examples, documents, and above all multi-turn chats and agent loops resend the same beginning again and again, so hits are common. DeepSeek reported {sv:deepseek-v3-production.kv_hit_rate_pct|raw}% of its input tokens hitting the cache in production. That makes caching a router problem too: a hit is only possible on the replica that holds the blocks, so 2026 routers send each request where its prefix already lives.',
  'Memory still runs out. Unused labeled blocks wait in a free queue, oldest first, and are evicted only when someone needs space, so prefixes that keep getting reused (a shared system prompt) stay, and one-off tails go. Newer engines move evicted blocks to CPU memory or storage rather than dropping them, because reloading a long prefix is cheaper than recomputing it.',
  'That is the answer to the hook. A cache hit saves the provider the prefill math for those tokens, which for a long prompt is most of the request\'s input cost, so it can sell them for a tenth or less. But someone has to keep the KV around until the next request comes, in fast memory that could be serving other users. Writing to the cache is priced above plain input, and entries expire after minutes or an hour, depending on what you pay. The exact multipliers are business decisions that vendors do not explain.',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{key}(b_i) = \operatorname{hash}\big(\text{key}(b_{i-1}),\ \text{tokens}(b_i)\big),\qquad \text{key}(b_{-1}) = \text{root}` },
  { tex: tex`\htmlClass{hl-hit}{\text{hit tokens}} = B \cdot \max\{\,j : \text{key}(b_0), \dots, \text{key}(b_{j-1}) \in \text{cache}\,\}\qquad \text{hit rate} = \frac{\sum \htmlClass{hl-hit}{\text{hit tokens}}}{\sum \text{prompt tokens}}` },
  { tex: tex`\text{price per M input} = (1-h)\cdot p_{\text{in}}\cdot m_{\text{write}} + h \cdot p_{\text{in}}\cdot \htmlClass{hl-hit}{m_{\text{read}}}` },
]);

export const MATH_NOTES = Object.freeze([
  'Shapes: one block is B tokens of [n_kv × d_head] keys and values per layer, as in [[paged-attention]]. Block numbers count from 0; prompt positions on the stage count from 1. B is 4 on the stage and the pool is 8 blocks.',
  'What a hit saves (first principles, [[prefill-decode]]): 2 · N_active · (hit tokens) FLOPs of prefill. What it costs: (hit tokens) × (KV bytes per token), held until reuse.',
  'Hover a highlighted term to outline the hit blocks in a request\'s block table (frames 3 to 5).',
]);

const TAKEAWAYS = Object.freeze([
  'Prefix caching keeps finished requests\' full blocks, labeled by their whole prefix, and a new request reuses every block it matches from its first token; the same words after a different start do not match.',
  'Hits depend on the grain (block size), on eviction (least recently used first, so shared beginnings survive) and on routing (the hit is only on the replica that holds the blocks).',
  'A hit saves prefill math but the KV must be held somewhere, which is why cache reads are priced at a tenth or less of plain input and cache writes above it.',
]);

const FURTHER = Object.freeze([
  { title: 'Inside vLLM: anatomy of a high-throughput LLM inference system (Aleksa Gordić)', href: 'https://vllm.ai/blog/2025-09-05-anatomy-of-vllm' },
  { title: 'Tiered KV offloading (vLLM)', href: 'https://vllm.ai/blog/2026-09-10-tiered-kv-offloading' },
  { title: 'SGLang / RadixAttention (Zheng et al.)', href: 'https://arxiv.org/abs/2312.07104' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Prefix caching in eleven steps: the block tree, a hit, a miss, eviction, routing, offload and the price',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: renderFor(stageText(data)),
      standIn: fill('Hand-picked prompts, not real traffic: blocks of 4 where vLLM uses {sv:vllm.default_block_size}. These four requests are this page\'s own cast; only their letters and colors are the course\'s, from [[batching]] and [[paged-attention]].'),
    },
    toy: {
      title: 'Grow the prefix tree.',
      intro: 'Four hand-picked prompts; each finishes before the next arrives. Real system prompts run to thousands of tokens.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES.map(fill) },
    facts: { framing: FRAMING, rows: factRows() },
    takeaways: TAKEAWAYS.map(fill),
    links: { next: ['serving-calculator'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
