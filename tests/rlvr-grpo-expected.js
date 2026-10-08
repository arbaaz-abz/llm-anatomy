// rlvr-grpo storyboard §5 captions and §6 try-this items, verbatim. Imported by the page test and the e2e spec.
// Two deviations from the storyboard's caption text, both forced by the lesson-spec rule that a caption holds no
// operator (README lesson 2): frames 4 and 5 print "1.73" where the storyboard prints "+1.73".
export const CAPTIONS = [
  'One prompt, eight sampled answers from the same model. Only the random sampling differs.',
  'A program checks each final answer: 1 if it is 56, 0 if not. Row 5 has wrong working and a right final token, and still gets a 1.',
  'The group\'s own mean is the baseline, the score an answer has to beat: 2 of 8 right gives 0.25. Right answers sit 0.75 above it, wrong ones 0.25 below.',
  'Divide by the group\'s spread, 0.43, and you have each answer\'s advantage: 1.73 for right, −0.58 for wrong. Within a group they always sum to zero.',
  'Every token in a right answer gets 1.73 and is pushed up; every token in a wrong one gets −0.58 and is pushed down, shared prefix included.',
  'All right, or all wrong: no spread, so every advantage is 0 and nothing moves. Dynamic sampling drops such groups and samples again until the batch has signal.',
  'PPO keeps four models in memory, including a critic network that estimates the baseline. GRPO takes the baseline from the group, and RLVR swaps the reward model for a checker.',
  'The trainer updates on the same answers several times, so 56 is already 25% more likely than when it was sampled. Past 1.2 the clip switches its gradient off.',
  'Clip-higher raises only the upper edge, to 1.28 (DAPO, GLM-5). A good token that started out unlikely can keep growing, so the model keeps exploring.',
];

// The toy's try-this list (README lesson 34): [prompt, insight, rest]; the page prints "prompt → Insight: insight" + rest.
export const TRY_THIS = [
  [
    'Predict first: if 4 of 8 answers are right instead of 2, does 7 × 8 = 56 get a bigger or smaller push? Slide k 2 → 4 → 8. A for a right answer goes +1.73 → +1.00 → 0.00; Σ|A| goes 6.93 → 8.00 → 0.00. Now slide to 1: +2.65 for the lone right answer, −0.38 for each of the seven wrong ones; and to 0: all zeros, the same banner as at 8. Then switch norm off and repeat: +0.75 → +0.50 → 0.00, and the lone right answer gets only +0.875.',
    'Advantages are relative, not absolute.',
    ' The rarer the outcome inside its group, the bigger its push; dividing by the std amplifies that unevenly. Compare Σ|A| with and without the division: 5.29 vs 1.75 at k = 1 (3.02×), 6.93 vs 3.00 at k = 2 (2.31×), 8.00 vs 4.00 at k = 4 (2×). The total push still peaks at k = 4, but the division boosts near-impossible and near-solved prompts the most (the difficulty bias), which is Dr.GRPO\'s argument for dropping it, as DeepSeek-V3.2 does. A group with no spread teaches nothing, which is why dynamic sampling throws it away and why GLM-5 keeps only prompts its previous model solves rarely but can solve (see the facts below).',
  ],
  [
    'With k = 2, select the 48 in row 4 (7 × 8 = 48 , so 48, 8 tokens) and then the 63 in row 6 (1 token). Under sample aggregation the inspector shows a push per token of −0.0090 for row 4 and −0.0722 for row 6: both wrong, but the long answer is punished 8× less per token, and both rows print the same push per answer, −0.0722. Flip agg to token: every token now pays −0.0148, so the long wrong answer\'s push per answer becomes −0.1184 and the short one\'s −0.0148.',
    'Sample-level averaging shields long wrong answers (and dilutes long right ones); token-level loss charges every token the same.',
    ' That is DAPO\'s token-level loss, part of the 2026 consensus recipe; Olmo 3 reports it (see the facts below).',
  ],
  [
    'With k = 2 and ε_high = 0.20, three chips are hatched: 56 in row 1 (r = 1.25), 56 in row 5 (r = 1.35), and the first 48 in row 4 (r = 0.75). Select row 1\'s 56: objective 2.078, clipped. Press the clip-higher 0.28 chip: row 1\'s 56 loses its hatch (objective 2.165, gradient on); row 5\'s 56 stays hatched (1.35 > 1.28); row 4\'s 48 stays hatched (0.75 < 0.8, the lower bound did not move).',
    'Clipping switches a token\'s gradient off, it does not shrink it, and clip-higher widens only the upward side',
    ' so a good token that was improbable when sampled can keep gaining probability (no entropy collapse), while the lower bound still stops collapse the other way.',
  ],
];
