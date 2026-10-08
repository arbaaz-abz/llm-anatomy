// The one long string that is both page text and a test fixture: frame 5's "Numbers shown", printed from math/params.js.
import { breakdownFor, MODERN_STATE, EXPERT, int } from './format.js';
import { formatCount } from '@math/core.js';

const modern = breakdownFor(MODERN_STATE);
const dense = breakdownFor({ ...MODERN_STATE, experts: 'dense' });
const swigluHidden = EXPERT.hidden * 8;

// "GPT-3's shape with SwiGLU, no biases, 8 KV heads, then 64 experts of hidden 4,096 (an eighth of 32,768), top-8: total … (960B), active … (148B) vs … dense; the difference is the router."
export const FRAME5_EXACT_TEXT = `GPT-3's shape with SwiGLU, no biases, 8 KV heads, then ${EXPERT.routed} experts of hidden ${int(EXPERT.hidden)} (an eighth of ${int(swigluHidden)}), top-${EXPERT.topK}: total ${int(modern.total)} (${formatCount(modern.total)}), active ${int(modern.active)} (${formatCount(modern.active)}) vs ${int(dense.total)} dense; the difference is the router.`;
