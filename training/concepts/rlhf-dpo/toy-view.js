// rlhf-dpo toy view model (pure, no DOM): state → every string the toy prints. All numbers come from math/preference.js.
import { dpoLoss, sigmoid } from '@math/preference.js';
import { fmt3, checkWork } from './format.js';

export const LOSS_REF = '0.693 = ln 2 (no preference yet)';

export function toyView(state) {
  const r = dpoLoss(state);
  return {
    rewards: [fmt3(r.rewardChosen), fmt3(r.rewardRejected)],
    rewardValues: [r.rewardChosen, r.rewardRejected],
    margin: fmt3(r.margin),
    pChosen: fmt3(sigmoid(r.margin)),
    loss: fmt3(r.loss),
    lossRef: LOSS_REF,
    weight: fmt3(r.weight),
    checkWork: checkWork(state),
  };
}
