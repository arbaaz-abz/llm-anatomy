// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from
// math/preference.js (never typed), printed like the toy's own outputs. No DOM.
import { dpoLoss } from '@math/preference.js';
import { formatRatio } from '@math/core.js';
import { fmt3, trim1, INITIAL_STATE } from './format.js';

const run = (dChosen, dRejected, beta = INITIAL_STATE.beta) => dpoLoss({ dChosen, dRejected, beta });
// Two settings with one margin (β × dChosen = 0.35), so one loss.
const SAME_LOSS = Object.freeze({ small: { beta: 0.1, dChosen: 3.5 }, large: { beta: 0.5, dChosen: 0.7 } });

// [prompt, insight, rest]; `**x**` is bold when the page renders it (see emphasis in toy.js).
export function tryThis() {
  const mid = run(5, -5);
  const far = run(10, -10);
  const both = run(-1, -3);
  const zero = run(0, 0);
  const start = run(INITIAL_STATE.dChosen, INITIAL_STATE.dRejected);
  const wide = run(INITIAL_STATE.dChosen, INITIAL_STATE.dRejected, 0.5);
  const small = run(SAME_LOSS.small.dChosen, 0, SAME_LOSS.small.beta);
  return [
    [`Predict first: if A keeps gaining and B keeps losing, does DPO keep pushing just as hard? Set A's change to **5** and B's to **−5**: loss ${fmt3(mid.loss)}, update weight ${fmt3(mid.weight)}. Keep going to **10** and **−10**: loss ${fmt3(far.loss)}, weight ${fmt3(far.weight)}`,
      'the update fades as a pair is learned.',
      ' DPO stops spending effort on pairs it already ranks correctly, the way a reward model\'s loss does (frame 2), but with no reward model and no sampling.'],
    [`Set A's change to **−1** and B's to **−3**. Both answers are now less likely than under the reference, yet the loss is ${fmt3(both.loss)}, lower than the ${fmt3(zero.loss)} at (0, 0)`,
      'DPO optimizes the gap, not the chosen answer.',
      ' The chosen answer can become less likely during DPO; this is a known side effect and one reason labs check likelihoods during preference training.'],
    [`At (${trim1(INITIAL_STATE.dChosen)}, ${trim1(INITIAL_STATE.dRejected)}), switch β from **${trim1(INITIAL_STATE.beta)}** to **0.5**: gap ${fmt3(start.margin)} → ${fmt3(wide.margin)}, loss ${fmt3(start.loss)} → ${fmt3(wide.loss)}. Then, at β = ${trim1(SAME_LOSS.small.beta)}, find the same loss: you need A's change = ${trim1(SAME_LOSS.small.dChosen)} with B's change = 0, against ${trim1(SAME_LOSS.large.dChosen)} at β = ${trim1(SAME_LOSS.large.beta)}`,
      'β is the leash.',
      ` A larger β reaches the same loss (${fmt3(small.loss)}) with ${formatRatio(SAME_LOSS.small.dChosen / SAME_LOSS.large.dChosen)} less drift from the reference (${trim1(SAME_LOSS.small.dChosen)} vs ${trim1(SAME_LOSS.large.dChosen)} of log-probability). It plays the role the KL term's β plays in RLHF (frame 8), because DPO is derived from that same KL-anchored objective (see the math).`],
  ];
}
