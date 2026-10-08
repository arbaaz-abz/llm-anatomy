// kv-cache's hand-authored numbers (storyboard §4, §5) and what follows from them. Pure, no DOM.
// K and V rows 1–4 are `attention`'s head A; every later row is a hand-picked stand-in on the same quarter grid
// (hover only on the stage). Model figures below are what the stage prints; the page test pins them to data/models.json.
import { deepFreeze } from '@math/core.js';
import { TOY } from '@math/attention.js';
import { kvBytesPerToken, kvBytesPerTokenMla, kvCacheBytes, decodeWork, sharePct } from '@math/memory.js';

export const TOKENS = TOY.tokens; // The cat sat down, shared with `attention` and `decoder-anatomy`
export const ROW_LABELS = Object.freeze([...TOKENS, 'on', 'the', 'mat', '.']); // 4 prompt rows, then the 4-token reply
export const PROMPT_ROWS = TOKENS.length;
export const ON_ROW = PROMPT_ROWS; // 0-based row of "on", the row followed in frames 1–5
export const MAX_ABS = 2;

export const K_ON = deepFreeze([0.5, 0, 0.5, 0]);
export const V_ON = deepFreeze([0, 0.5, 0.5, 0]);
export const K_ROWS = deepFreeze([...TOY.heads.A.K, K_ON, [0, 0.5, 0, 0.5], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]);
export const V_ROWS = deepFreeze([...TOY.heads.A.V, V_ON, [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0], [0.5, 0, 0.5, 0]]);

// Frame 6: the second head's stand-in rows for "on" (head A's are K_ON and V_ON).
export const HEAD_B_ON = deepFreeze({ k: [0, 0.5, 0, 0.5], v: [0.5, 0.5, 0, 0] });

export const TOY_SHAPE = deepFreeze({ layers: 2, kvHeads: 2, headDim: 4, bytesPerElem: 2 });
export const TOY_CONTEXT = 5; // frame 6: five tokens in the cache
export const REPLY = Object.freeze(['on', 'the', 'mat', '.']);
export const PROMPT_LEN = 4;

// What the stage prints for the real models (pinned to data/models.json in tests/kv-cache-page.test.js).
export const GPT3 = deepFreeze({ layers: 96, kvHeads: 96, headDim: 128, bytesPerElem: 2, context: 2048 });
export const LLAMA = deepFreeze({ layers: 80, kvHeads: 8, headDim: 128, bytesPerElem: 2, context: 131072 });
export const V3 = deepFreeze({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 });
export const V4_PRO_BYTES = deepFreeze([4000, 12000]); // reported estimate, bytes per token
export const GPU_BYTES = 80e9; // an H100's 80 GB, nominal (hardware.json h100.hbm_gb)
export const SLIDER_STOP = 1_048_576; // 2²⁰
export const LLAMA_TOKENS = 131_072;

export const BYTES_PER_TOKEN = Object.freeze({
  toy: kvBytesPerToken(TOY_SHAPE),
  gpt3: kvBytesPerToken(GPT3),
  llama: kvBytesPerToken(LLAMA),
  v3: kvBytesPerTokenMla(V3),
});

// The stage's byte figures, all from math/memory.js.
export const CACHE = Object.freeze({
  gpt3Context: kvCacheBytes({ bytesPerToken: BYTES_PER_TOKEN.gpt3, tokens: GPT3.context }),
  llamaOne: kvCacheBytes({ bytesPerToken: BYTES_PER_TOKEN.llama, tokens: LLAMA_TOKENS }),
  llamaTwo: kvCacheBytes({ bytesPerToken: BYTES_PER_TOKEN.llama, tokens: LLAMA_TOKENS, sequences: 2 }),
  toyFive: kvCacheBytes({ bytesPerToken: BYTES_PER_TOKEN.toy, tokens: TOY_CONTEXT }),
});
export const AT_STOP = Object.freeze({
  llama: kvCacheBytes({ bytesPerToken: BYTES_PER_TOKEN.llama, tokens: SLIDER_STOP }),
  v3: kvCacheBytes({ bytesPerToken: BYTES_PER_TOKEN.v3, tokens: SLIDER_STOP }),
  v4Low: kvCacheBytes({ bytesPerToken: V4_PRO_BYTES[0], tokens: SLIDER_STOP }),
  v4High: kvCacheBytes({ bytesPerToken: V4_PRO_BYTES[1], tokens: SLIDER_STOP }),
});

// Frame 5: the four passes of a 4-token reply (the first is the prefill) and their running totals.
export const passTotals = (pass, cache) => decodeWork({ prompt: PROMPT_LEN, generated: pass, cache });
export const gpuShare = (bytes) => sharePct(bytes, GPU_BYTES);
export const GPU_FILL = (bytes) => Math.min(bytes / GPU_BYTES, 1);
