// The eleven captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'A person reads two answers to the same prompt and picks the better one. That choice, a preference pair, is the only signal here: no program could check this.',
  'The reward model is a copy of the model that outputs one score. It is trained until the picked answer scores higher: 1.2 against −0.3 here.',
  'The policy, the model being trained, writes a fresh answer. The reward model scores it, and that score is what reinforcement learning tries to raise.',
  'A critic, a second trained network, predicts the score this prompt usually earns. Each token is pushed by how much the answer beat that prediction.',
  'PPO updates on each batch several times, so probabilities drift. Once a token is 20% more likely than when sampled, the clip range switches its update off.',
  'The reference model is a frozen copy of where the policy started. Comparing their probabilities shows how far training has moved the policy.',
  'Branch: with no anchor, the policy finds answers the reward model overrates, like opening with flattery. This is reward hacking: the score rises while real quality falls.',
  'The KL term subtracts a penalty for drifting from the reference. Flattery drifted far, so with the penalty the honest answer scores higher.',
  'Classic RLHF keeps four models in memory and generates fresh answers at every step. That cost is why simpler methods took over most of this job.',
  'DPO trains on the pairs directly: no reward model, no sampling. Measured against the reference, it widens the gap between the chosen and the rejected answer.',
  'In 2026, DPO is a cheap preference stage in smaller open pipelines. RLHF, now with a judge model as the reward, polishes style and safety at the end.',
]);
