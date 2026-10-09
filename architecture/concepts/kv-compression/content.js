// kv-compression: the complete lesson (storyboard docs/storyboards/kv-compression.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { renderFor } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { hook, factRows, FRAMING, belowFor } from './facts.js';
import { mathBlocks, MATH_NOTES } from './math.js';

export { CAPTIONS };

const SLUG = 'kv-compression';

const INTUITION = Object.freeze([
  'In [[kv-cache]] the size of the cache came down to one product: 2 × layers × KV heads × head size × bytes. The layers and head size are the model\'s shape. The KV heads term is a choice. In the original design (multi-head attention, MHA) every query head has its own key head and value head, so the cache stores a full set per head. But a query head only needs some keys to score against; nothing says each query head needs its own.',
  'So let several query heads share one stored set. If all of them share one, that is multi-query attention (MQA): the smallest cache, with some loss in quality. If they share in groups, it is grouped-query attention (GQA), and {llama-3.1-70b.n_kv_heads} KV heads became the common middle setting. The heads keep their own queries, so they still ask different questions of the same keys and still produce different patterns.',
  'Multi-head latent attention (MLA) goes another way. It stores one short "latent" vector per token and learns how to rebuild every head\'s own key and value from it, so each head keeps its own keys and values while the cache stays close to a two-head GQA. The rebuild costs compute instead of memory, and most of it can be folded into the query and output matrices. Position is the exception: RoPE rotates each key by its token\'s position, which cannot be folded in that way, so MLA stores one small extra key per token that carries position ([[rope]]). Every 2026 design on this page trades a little quality or a little compute for a much smaller cache.',
]);

const TAKEAWAYS = Object.freeze([
  'The cache counts KV heads, not query heads. MQA shares one key/value set among all query heads, GQA shares within groups ({llama-3.1-70b.n_kv_heads} KV heads is common), and the cache shrinks in exact proportion.',
  'Sharing keys does not make heads alike: each head keeps its own queries and so its own pattern.',
  'MLA stores one short latent per token (plus a small position key) and rebuilds each head\'s keys and values from it: about the memory of two shared KV heads, paid for with extra compute at decode. It is the 2024–26 incumbent, not the endpoint.',
]);

const FURTHER = Object.freeze([
  { title: 'Sebastian Raschka, LLM Architecture Gallery', href: 'https://sebastianraschka.com/llm-architecture-gallery/', note: 'GQA and MLA explainers per model, with a memory calculator' },
  { title: 'Ainslie et al., GQA', href: 'https://arxiv.org/abs/2305.13245', note: 'the paper that grouped the heads' },
  { title: 'DeepSeek-AI, DeepSeek-V2', href: 'https://arxiv.org/pdf/2405.04434', note: 'the paper that introduced MLA' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  const below = belowFor(data).map((paragraphs) => paragraphs.map(fill));
  return {
    slug: SLUG,
    hook: fill(hook(data)),
    intuition: INTUITION.map(fill),
    animation: {
      label: 'How query heads share, group or compress the stored keys and values: MHA, MQA, GQA and MLA in nine steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: renderFor(data),
      standIn: 'This layer has 8 query heads so the sharing is visible; [[attention]]\'s toy had 2. Head counts and the latent size here are hand-picked toy sizes; frame 4 reuses the attention page\'s numbers; every byte count is exact.',
      belowFor: (index) => below[index] ?? [],
    },
    toy: {
      title: 'Share, group or compress.',
      intro: 'Head counts and the latent size here are hand-picked toy sizes; the pattern at the bottom uses the attention page\'s numbers; every byte count is exact.',
      mount,
    },
    math: { blocks: mathBlocks(data), notes: MATH_NOTES },
    facts: { framing: fill(FRAMING), rows: factRows(data) },
    takeaways: TAKEAWAYS.map(fill),
    links: { next: ['long-context-attention', 'paged-attention'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
