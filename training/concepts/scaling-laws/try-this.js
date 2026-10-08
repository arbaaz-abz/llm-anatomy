// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from math/scaling.js
// and printed with the toy's own formatters (never typed). No DOM.
import { computeOptimal, isoFlopLoss, inferenceAwareOptimum, lifetimeFlops } from '@math/scaling.js';
import { STAGE_BUDGET, BUDGETS } from './numbers.js';
import { perParam, lossText, sizeText, tokensText, savingText, powerText } from './format.js';

const ONE_B = 1e9;
const ONE_T = 1e12;
const SERVE_STEPS = Object.freeze([1e12, 1e13, 1e14, 1e15]);

function cheapest(budget, served) {
  const best = computeOptimal(budget);
  const pick = inferenceAwareOptimum({ targetLoss: best.loss, inferenceTokens: served });
  return { pick, saving: 1 - pick.total / lifetimeFlops(best.N, best.D, served) };
}

const SERVED_LABELS = Object.freeze({ 1e12: '1T', 1e13: '10T', 1e14: '100T', 1e15: '1,000T' });
const servedText = (served) => SERVED_LABELS[served];

// [prompt, insight, rest]; `**x**` is bold when the page renders it (see emphasis in toy.js).
export function tryThis() {
  const opt = computeOptimal(STAGE_BUDGET);
  const big = isoFlopLoss(STAGE_BUDGET, ONE_T);
  const small = isoFlopLoss(STAGE_BUDGET, ONE_B);
  const steps = SERVE_STEPS.map((served, i) => {
    const { pick, saving } = cheapest(STAGE_BUDGET, served);
    const lead = i === 0 ? 'the cheapest model is ' : '';
    const tail = i === 0 ? ' tokens per parameter' : '';
    return `**${servedText(served)}** → ${lead}${sizeText(pick.N)} at ${perParam(pick.tokensPerParam)}${tail} (${savingText(saving)}${i === 0 ? ' saving' : ''})`;
  });
  const low = computeOptimal(BUDGETS[0]);
  const high = computeOptimal(BUDGETS[BUDGETS.length - 1]);
  return [
    [`Predict first: at ${powerText(STAGE_BUDGET)} FLOPs, is a 1T-parameter model better than the compute-optimal ${sizeText(opt.N)} one? Drag **Model size** from ${sizeText(opt.N)} to 1T → loss ${lossText(opt.loss)} → ${lossText(big.loss)}, tokens ${tokensText(opt.D)} → ${tokensText(big.D)}. Drag down to 1B: ${lossText(small.loss)}.`,
      'there is a valley.',
      ` Too big and the model sees too few tokens; too small and it cannot use them. The bottom sits near ${perParam(opt.tokensPerParam)} tokens per parameter at this budget, the Chinchilla rule of about 20.`],
    [`Leave **Model size** at the optimum and step **Tokens the model will serve** at ${powerText(STAGE_BUDGET)} FLOPs: ${steps.join('; ')}`,
      'the more a model will be used, the smaller and longer-trained it should be.',
      ' At heavy use the best ratio lands in the hundreds to thousands of tokens per parameter, where 2026 models actually are (frame 8).'],
    [`Switch **Training budget** from ${powerText(BUDGETS[0])} to ${powerText(BUDGETS[BUDGETS.length - 1])} with **Tokens the model will serve** at 0: the optimum goes ${sizeText(low.N)} / ${tokensText(low.D)} tokens → ${sizeText(high.N)} / ${tokensText(high.D)}, and the ratio only drifts ${perParam(low.tokensPerParam)} → ${perParam(high.tokensPerParam)}`,
      'compute-optimal scales parameters and tokens together.',
      ' Ten thousand times the budget buys about a hundred times more of each.'],
  ];
}
