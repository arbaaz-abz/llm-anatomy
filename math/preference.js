// Preference learning (rlhf-dpo §11): the Bradley–Terry reward-model loss, DPO's loss and RLHF's KL-penalized reward.
// Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('<fn>: <arg> must be …').
// Uses klDivergence (math/lm.js) and clippedSurrogate (math/grpo.js) on the page; neither is redefined here.

function requireFinite(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${fn}: ${name} must be a finite number, got ${value}`);
}

// Logistic function, written so a very negative or positive x never overflows.   0 → 0.5 · 1.5 → 0.8176 · −0.07 → 0.4825
export function sigmoid(x) {
  requireFinite('sigmoid', 'x', x);
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

// −ln σ(x) = ln(1 + e^−x), stable for large |x|.
const negLogSigmoid = (x) => (x >= 0 ? Math.log1p(Math.exp(-x)) : -x + Math.log1p(Math.exp(x)));

// Bradley–Terry pairwise reward-model loss: P(chosen preferred) = σ(r_chosen − r_rejected), loss = −ln of it.
//   (1.2, −0.3) → { pChosen: 0.8176, loss: 0.2014 } · (−0.3, 1.2) → loss 1.7014 · (0, 0) → { 0.5, 0.6931 }
export function bradleyTerry(rChosen, rRejected) {
  requireFinite('bradleyTerry', 'rChosen', rChosen);
  requireFinite('bradleyTerry', 'rRejected', rRejected);
  const gap = rChosen - rRejected;
  return { pChosen: sigmoid(gap), loss: negLogSigmoid(gap) };
}

// DPO loss from the summed log-probability changes versus the reference. weight = σ(−margin), the gradient scale.
//   (0.5, −0.2, 0.1) → { 0.05, −0.02, 0.07, 0.6588, 0.4825 } · (−1, −3, 0.1) → loss 0.5981 · (0.5, −0.2, 0.5) → loss 0.5334
export function dpoLoss({ dChosen, dRejected, beta = 0.1 } = {}) {
  requireFinite('dpoLoss', 'dChosen', dChosen);
  requireFinite('dpoLoss', 'dRejected', dRejected);
  if (typeof beta !== 'number' || !Number.isFinite(beta) || beta <= 0) throw new RangeError(`dpoLoss: beta must be a finite number > 0, got ${beta}`);
  const rewardChosen = beta * dChosen;
  const rewardRejected = beta * dRejected;
  const margin = rewardChosen - rewardRejected;
  return { rewardChosen, rewardRejected, margin, loss: negLogSigmoid(margin), weight: sigmoid(-margin) };
}

// RLHF's per-sample objective: reward minus beta times the KL from the reference.
//   (1.0, 0.1838, 0.5) → 0.9081 · (1.3, 2.1031, 0.5) → 0.2484 · (1.3, 2.1031, 0) → 1.3
export function klPenalizedReward(reward, kl, beta) {
  requireFinite('klPenalizedReward', 'reward', reward);
  if (typeof kl !== 'number' || !Number.isFinite(kl) || kl < 0) throw new RangeError(`klPenalizedReward: kl must be a finite number ≥ 0, got ${kl}`);
  if (typeof beta !== 'number' || !Number.isFinite(beta) || beta < 0) throw new RangeError(`klPenalizedReward: beta must be a finite number ≥ 0, got ${beta}`);
  return reward - beta * kl;
}
