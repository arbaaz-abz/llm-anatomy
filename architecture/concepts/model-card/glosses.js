// model-card learner glosses (pure, no DOM): one plain sentence per place where sources disagree or a figure is
// carried over. Every number comes from the data entry itself; the data file's maintainer notes are never printed
// (they hold raw URLs and sourcing chatter). The source link stays on the row, the sentence says what differs.
import { activeWithEmbeddings } from '@math/card.js';
import { formatBytes, formatCount } from '@math/core.js';
import { sharePct } from '@math/memory.js';
import { lookupFact } from '@shared/claims.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const entryOf = (data, id) => data?.models?.entries?.find((e) => e.id === id) ?? null;

const GLOSSES = Object.freeze({
  'glm-5.3.layers': (data) => {
    const [low, high] = fact(data, 'glm-5.3', 'layers');
    return `Its config says ${low} layers; the GLM-5 paper says ${high}.`;
  },
  'mistral-large-4.context_length': (data) => {
    const [low, high] = fact(data, 'mistral-large-4', 'context_length');
    return `Mistral's card claims ${formatCount(high)} tokens; independent evaluators reportedly measure about ${formatCount(low)}.`;
  },
  'deepseek-v4-pro.kv_bytes_per_token': (data) => {
    const [low, high] = fact(data, 'deepseek-v4-pro', 'kv_bytes_per_token');
    const ratio = sharePct(fact(data, 'deepseek-v4-pro', 'v32_kv_ratio_1m'), 1, { decimals: 0 });
    return `An estimate from the config: about ${formatBytes(low)} per token if the two kinds of compressed layer (CSA and HCA) alternate 1 : 1 (FP8), about ${formatBytes(high)} if the mix is 3 : 1 (BF16); the paper gives only a ratio (about ${ratio}% of V3.2's cache at 1M tokens).`;
  },
  'mistral-large-4.active_params': (data) => `${formatCount(fact(data, 'mistral-large-4', 'active_params'))} counts the routed experts only; ${formatCount(activeWithEmbeddings(entryOf(data, 'mistral-large-4')))} includes the embeddings.`,
  'glm-5.3.active_params': (data) => `GLM-5.3's card does not print its own active count; ${formatCount(fact(data, 'glm-5.3', 'active_params'))} is carried over from GLM-5.`,
  'deepseek-v4-pro.modalities': (data) => `DeepSeek-V4-Pro takes ${fact(data, 'deepseek-v4-pro', 'modalities')} only; its paper lists multimodality as future work, though some secondary sites call it multimodal.`,
});

// The sentence for one data key of one entry, or null when the page has nothing to add.
export function glossFor(data, id, key) {
  const make = GLOSSES[`${id}.${key}`];
  return make && fact(data, id, key) !== null ? make(data) : null;
}

// The "Labs differ" keys: where sources disagree or a figure is carried over (the modality sentence is frame 8's).
export const LABS_DIFFER_KEYS = Object.freeze(Object.keys(GLOSSES).filter((k) => !k.endsWith('.modalities')));
