// Hand-authored stand-ins of storyboard §4/§6: the toy's sizes, the seeded indexer scores, the hand-picked window scores
// and the real-scale assumptions. Everything here is a toy choice or a unit of this lesson, never a dated model fact.
import { randomMatrix, deepFreeze } from '@math/core.js';

export const TOKENS = 16; // positions in the toy; the followed query is the newest one
export const FOLLOWED = TOKENS - 1; // its 0-based row; drawn as "token 16"
export const WINDOW = 4;
export const TOP_K = 4;
export const MERGE = 4;
export const INDEXER_SEED = 7;
export const INDEXER = deepFreeze(randomMatrix(TOKENS, TOKENS, INDEXER_SEED)); // the sparse and compressed frames' scores

// Window scores for the followed query, by distance back from it (0 = itself): tokens 13–16 score −1, −0.5, −1, −0.75.
export const WINDOW_SCORES_BY_OFFSET = Object.freeze([-0.75, -1, -0.5, -1]);
export const SINK_LOGITS = Object.freeze([-2, 0, 1, 2]);
export const DEFAULT_SINK_LOGIT = 1;
export const GATES = Object.freeze([1, 0.5]);

export const ONE_M = 1_048_576; // "1M tokens" as the models in §8 count it
export const CONTEXTS = Object.freeze([131_072, ONE_M]);
export const BYTES_PER_NUMBER = 2; // BF16, the width behind every derived KV size on this page
export const STATE_SIZE = Object.freeze({ dKey: 4, dValue: 4 }); // the toy's linear-attention state

// DeepSeek-V4-Pro's cache estimate (storyboard §6, brief 04 §8.2): 60 attention layers, split CSA : HCA either 1 : 1 or 3 : 1,
// each entry 512 numbers wide at 1 or 2 bytes. Hand-authored assumptions, labeled "estimate" wherever they print.
export const V4_MIXES = Object.freeze([
  Object.freeze({ csaLayers: 30, hcaLayers: 30 }),
  Object.freeze({ csaLayers: 45, hcaLayers: 15 }),
]);
export const V4_ENTRY_BYTES = Object.freeze([1, 2]);
