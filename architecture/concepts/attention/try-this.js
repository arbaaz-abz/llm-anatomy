// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from
// math/attention.js (never typed), formatted like the toy's own weight cells. No DOM.
import { TOY } from '@math/attention.js';
import { fmt3, runHead, INITIAL_STATE } from './format.js';

const SUBSCRIPTS = ['₁', '₂', '₃', '₄'];
const tok = (i) => `${TOY.tokens[i]}${SUBSCRIPTS[i]}`;
const SAT = 2;
const DOWN = 3;
const SMALL_DIVISOR = 0.5;
const LARGE_DIVISOR = 8;

const row = (state, query) => `[${runHead(state).weights[query].map(fmt3).join(', ')}]`;
const weightOf = (state, query, key) => fmt3(runHead(state).weights[query][key]);

// [prompt, insight, rest]; `**x**` is bold and `*x*` italic when the page renders them (see emphasis in toy.js).
export function tryThis() {
  const on = INITIAL_STATE;
  const off = { ...on, causal: false };
  const headB = { ...on, head: 'B' };
  return [
    [`Keep query = ${tok(SAT)} and switch the causal mask **off** → the row becomes ${row(off, SAT)}: ${weightOf(off, SAT, DOWN)} of the weight now lands on "${TOY.tokens[DOWN]}", a token that comes *later*, and ${TOY.tokens[1]}'s share drops from ${weightOf(on, SAT, 1)} to ${weightOf(off, SAT, 1)} because the row must still sum to 1`,
      'The mask is the only thing that makes attention causal.',
      ' The scores know nothing about order; only the mask stops a query from reading later keys.'],
    [`Mask back on. Drag the divisor to **${SMALL_DIVISOR}** → ${row({ ...on, divisor: SMALL_DIVISOR }, SAT)}; now to **${LARGE_DIVISOR}** → ${row({ ...on, divisor: LARGE_DIVISOR }, SAT)}`,
      '√d_head is a sharpness dial.',
      ' Small divisor: softmax collapses to "pick the max". Large divisor: it flattens to "average everything". Dot products are sums of d_head terms, so they grow with head size (see the mono line under the slider); dividing by √d_head holds them in the useful middle.'],
    [`Divisor back to ${on.divisor}. Set query = ${tok(DOWN)} and flip head **A → B** → head A gives ${row(on, DOWN)} (most weight on "${TOY.tokens[1]}"); head B gives ${row(headB, DOWN)} (most weight on "${TOY.tokens[SAT]}", the previous token). Choose **both** to see the two heatmaps side by side and the [1 × ${TOY.dHead * 2}] concat row`,
      'Same tokens, different pattern: multi-head is several attention patterns at once.',
      ' The tokens did not change; only the W matrices did.'],
  ];
}
