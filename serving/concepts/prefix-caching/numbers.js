// prefix-caching's computed constants (storyboard §6 scale-up line): what one 10,000-token cached prefix saves and holds on the
// course's running example. Pure; the numbers come from math/serving.js and math/memory.js, nothing is typed in.
import { deepFreeze } from '@math/core.js';
import { RUNNING_EXAMPLE, stepTime } from '@math/serving.js';
import { kvCacheBytes } from '@math/memory.js';

export const SCALE_UP_TOKENS = 10000;

export const SCALE_UP = deepFreeze({
  tokens: SCALE_UP_TOKENS,
  skippedS: stepTime({ ...RUNNING_EXAMPLE, tokens: SCALE_UP_TOKENS, seqs: 0, context: 0 }).timeS, // the prefill a hit skips
  heldBytes: kvCacheBytes({ bytesPerToken: RUNNING_EXAMPLE.kvBytesPerToken, tokens: SCALE_UP_TOKENS }), // the KV held until reuse
});
