// The three "Try this" prompts (storyboard §6). Every number comes from the same math the toy prints (pure; shapes from the data).
import { numbersPerLayer, bytesPerToken, timesSmaller } from './scheme.js';
import { BYTES_PER_NUMBER } from './numbers.js';
import { patternFor } from './pattern.js';
import { int, timesText } from './format.js';

const GQA_STOP = 8;
const basis = (shape) => ({ layers: shape.layers, bytesPerElem: BYTES_PER_NUMBER });
const at = (shape, scheme, kvHeads) => ({ ...shape, scheme, kvHeads });
const bytes = (shape, scheme, kvHeads) => bytesPerToken(at(shape, scheme, kvHeads), basis(shape));
const times = (shape, scheme, kvHeads) => timesText(timesSmaller(at(shape, scheme, kvHeads), basis(shape)));
const row = (weights, i, digits = 3) => `[${weights[i].slice(0, i + 1).map((w) => w.toFixed(digits)).join(', ')}]`;

export function tryThis(shapes) {
  const { gpt3, v3 } = shapes;
  const { shared, own } = patternFor();
  const mla = numbersPerLayer(at(v3, 'mla'));
  return [
    {
      prompt: `Pick GPT-3 with MHA: ${int(bytes(gpt3, 'mha'))} B per token. Switch to GQA with ${GQA_STOP} KV heads: ${int(bytes(gpt3, 'gqa', GQA_STOP))} B (${times(gpt3, 'gqa', GQA_STOP)} smaller); MQA: ${int(bytes(gpt3, 'mqa'))} B (${times(gpt3, 'mqa')}).`,
      insight: `the cache shrinks exactly in proportion to KV heads; the ${gpt3.queryHeads} query heads, and the ${gpt3.queryHeads} patterns, stay.`,
      rest: ' Try this 2 is the proof on two small heads.',
    },
    {
      prompt: `Switch the pattern: head B's row for "cat" goes from ${row(own.weights, 1)} to ${row(shared.weights, 1)} when it reads head A's keys, and its row for "down" from most weight on "sat" (${own.weights[3][2].toFixed(3)}) to ${shared.weights[3][2].toFixed(3)} on "sat" and ${shared.weights[3][3].toFixed(3)} on itself; head A's rows stay the same.`,
      insight: 'sharing keys changes what a head can find, not whether heads differ.',
      rest: ' Patterns come from the queries, and every head keeps its own.',
    },
    {
      prompt: `Pick DeepSeek-V3 (it opens on MLA): ${int(mla)} numbers per token per layer, ${int(bytes(v3, 'mla'))} B per token, ${times(v3, 'mla')} smaller than MHA at its ${v3.queryHeads} heads. Switch to GQA with 2 KV heads: ${int(numbersPerLayer(at(v3, 'gqa', 2)))} numbers; MQA: ${int(numbersPerLayer(at(v3, 'mqa')))}.`,
      insight: 'MLA costs about as much memory as two shared KV heads but keeps a key and value per head; it pays in compute at decode instead.',
      rest: ' GLM-5 enlarged its head size and cut its head count to tame that cost (see the facts below).',
    },
  ];
}
