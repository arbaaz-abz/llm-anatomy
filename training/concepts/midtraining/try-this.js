// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from math/ or the
// data, formatted like the toy's own readouts. No DOM.
import { formatRatio } from '@math/core.js';
import { attentionCostRatio } from '@math/schedule.js';
import { INITIAL_STATE, fmt3, lrAtStop, tailShare, mainShare } from './format.js';
import { runData, glmBudget, labelK, countText } from './facts.js';

const pct = (v) => `${Number(v.toFixed(1))}%`;

// [prompt, insight, rest]; `**x**` is bold when the page renders it (see emphasis in toy-dom.js).
export function tryThis(data) {
  const { nemotron, minimax, stages, midtrain } = runData(data);
  const cosine = { ...INITIAL_STATE, schedule: 'cosine' };
  const budget = glmBudget({ stages, midtrain });
  const later = stages.slice(1);
  const tail = budget.tail.parts.map((q) => `${q.name} ${tailShare(q.share)}`);
  const last = stages[stages.length - 1];
  const wholeLast = mainShare(budget.wholeShare(last.name));
  const ratios = later.map((s) => formatRatio(attentionCostRatio(labelK(s.name), labelK(stages[0].name))));
  return [
    [`Predict first: you planned a full run but must stop at ${INITIAL_STATE.stopAt}%. Under **cosine** the learning rate there is ${fmt3(lrAtStop(cosine))} of peak and still falling, so the checkpoint never got its decay. Switch to **WSD**: ${fmt3(lrAtStop(INITIAL_STATE))}, on the plateau, ready to branch into a short decay of its own.`,
      'WSD lets you choose the end late.',
      ' You can branch anneals off one long run, with different mixes or lengths, instead of committing on day one. Kimi K3 still found cosine better when both were tuned separately, so this is a trade, not a rule.'],
    [`Press **Nemotron 3 Super ${pct(nemotron.decayPercent)}**, then **MiniMax-M2 ${pct(minimax.decayPercent)}**, and watch the decay band widen.`,
      'the decay window where the best data goes is a fifth to a third of these runs:',
      ` ${countText(nemotron.decayTokens)} of ${countText(nemotron.total)}, ${countText(minimax.decayTokens)} of ${countText(minimax.total)}.`],
    [`With **GLM-5**, read both bars: ${stages[0].name} ${mainShare(budget.wholeShare(stages[0].name))} of the run; zoomed, ${tail.join(', ')} of the last ${countText(budget.tail.knownTotal)} (${last.name} is ${wholeLast} of the run), and the attention-only readouts ${ratios.join(', ')}. Switch to **Kimi K3** or **DeepSeek-V4**: their per-stage token counts are not published, so the bar shows only neutral segments labeled "no published shares".`,
      'context is extended on a thin slice of tokens,',
      ' precisely because each long token costs much more attention work and long documents are scarce.'],
  ];
}
