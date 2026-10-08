// rlhf-dpo expected strings (storyboard §5 captions, §6 check box and try-this), written out once for the page test and the e2e spec.
export const CAPTIONS = [
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
];

export const CHECK_WORK = [
  'reward A = 0.1 × 0.5    = 0.050',
  'reward B = 0.1 × (−0.2) = −0.020',
  'gap      = 0.050 − (−0.020) = 0.070',
  'loss     = −ln σ(0.070) = −ln 0.517 = 0.659',
  'weight   = σ(−0.070) = 0.483',
].join('\n');

// [prompt, insight, rest] for the three try-this items, exactly as the page prints them.
export const TRY_THIS = [
  [
    'Predict first: if A keeps gaining and B keeps losing, does DPO keep pushing just as hard? Set A\'s change to **5** and B\'s to **−5**: loss 0.313, update weight 0.269. Keep going to **10** and **−10**: loss 0.127, weight 0.119',
    'the update fades as a pair is learned.',
    ' DPO stops spending effort on pairs it already ranks correctly, the way a reward model\'s loss does (frame 2), but with no reward model and no sampling.',
  ],
  [
    'Set A\'s change to **−1** and B\'s to **−3**. Both answers are now less likely than under the reference, yet the loss is 0.598, lower than the 0.693 at (0, 0)',
    'DPO optimizes the gap, not the chosen answer.',
    ' The chosen answer can become less likely during DPO; this is a known side effect and one reason labs check likelihoods during preference training.',
  ],
  [
    'At (0.5, −0.2), switch β from **0.1** to **0.5**: gap 0.070 → 0.350, loss 0.659 → 0.533. Then, at β = 0.1, find the same loss: you need A\'s change = 3.5 with B\'s change = 0, against 0.7 at β = 0.5',
    'β is the leash.',
    ' A larger β reaches the same loss (0.533) with 5× less drift from the reference (3.5 vs 0.7 of log-probability). It plays the role the KL term\'s β plays in RLHF (frame 8), because DPO is derived from that same KL-anchored objective (see the math).',
  ],
];
