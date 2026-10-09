// model-card: the complete lesson (storyboard docs/storyboards/model-card.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FRAMING, factRows, belowFor } from './facts.js';
import { stageModel } from './stage-model.js';
import { mathBlocks, MATH_NOTES } from './math-blocks.js';

const SLUG = 'model-card';

const HOOK = '"{deepseek-v4-pro.total_params|count} total / {deepseek-v4-pro.active_params|count} active, {deepseek-v4-pro.layers} layers, {deepseek-v4-pro.experts_total} + {deepseek-v4-pro.experts_shared} experts, top-{deepseek-v4-pro.experts_active}, compressed attention with one {deepseek-v4-pro.head_dim}-wide KV head, {deepseek-v4-pro.context_length|count} context, {deepseek-v4-pro.modalities}-only." You have now seen every part this line describes. Can you read it, and can you tell which parts to trust?';

const INTUITION = Object.freeze([
  'A model card is the decoder you have been taking apart, written as numbers. Total and active parameters are all the experts versus the few each token uses ([[moe]]). Layers count blocks ([[decoder-anatomy]]). The experts line says how the MLP was split and how many pieces each token runs. The attention line says how many key/value sets are stored, whether they are compressed or windowed, and so how big the cache per token is ([[kv-compression]], [[long-context-attention]]); together with the context length, that decides how much memory a long conversation needs ([[kv-cache]]). The modalities field says whether there is a vision encoder in front at all ([[multimodal]]).',
  'Two habits make a card readable. First, turn each field into its cost: total parameters into weight memory (parameters × bytes per parameter: 70B parameters in 16-bit weights is 140 GB, in 8-bit 70 GB; [[quantization]] covers the formats), active parameters into compute per token, the attention line into bytes per token, the context into gigabytes per conversation. Second, read the source: a field from a config file is checkable, a field from a blog is not, and the same lab can count "active" two ways. When sources disagree, keep both numbers and say so. This page does exactly that.',
]);

const TAKEAWAYS = Object.freeze([
  'A model card is the decoder in numbers: total vs active is experts stored vs used, layers count blocks, the experts line is the router\'s menu, the attention line sets the cache per token, and modalities says what can enter.',
  'Turn fields into costs: total parameters into weight memory, active parameters into compute per token, the attention line times the context into memory per conversation.',
  'Read the source: config files beat blogs, labs count "active" differently, and where sources disagree keep both numbers.',
]);

const FURTHER = Object.freeze([
  { title: 'LLM Architecture Gallery (Sebastian Raschka)', href: 'https://sebastianraschka.com/llm-architecture-gallery/', note: '100+ model cards side by side, with a diff tool' },
  { title: 'DeepSeek-V4 (DeepSeek-AI)', href: 'https://arxiv.org/pdf/2606.19348', note: 'the card this page reads' },
  { title: 'The DeepSeek-V4-Pro config.json', href: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json', note: 'read the raw fields yourself' },
]);

const WHERE_NEXT = 'This page closes the Architecture track. To go on: how a model like this is trained ([[training-pipeline]]), what serving one costs ([[serving-calculator]]), and why active parameters and the cache set its speed ([[prefill-decode]]).';

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  const model = stageModel(data);
  return {
    slug: SLUG,
    hook: fill(HOOK),
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Reading DeepSeek-V4-Pro\'s model card, field by field',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: (index, progress, stage) => render(index, progress, stage, model),
      standIn: 'The card on the left is DeepSeek-V4-Pro\'s, read from the data file. The diagram on the right is schematic: a few tiles and boxes stand for hundreds, and nothing in it is drawn to scale.',
      belowFor: (index) => belowFor(index, data).map(fill),
    },
    toy: {
      title: 'Decode a card.',
      intro: 'Pick two cards. Every row shows the field, what it means, which lesson explains it, and where the figure came from; the cost rows turn fields into active share, cache per token and cache per conversation. A range stays a range.',
      mount,
    },
    math: { blocks: mathBlocks(data), notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows(data), prose: [WHERE_NEXT] },
    takeaways: TAKEAWAYS,
    links: { next: [], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
