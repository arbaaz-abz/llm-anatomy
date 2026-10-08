// distillation storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
export const CAPTIONS = Object.freeze([
  'A teacher, here the math specialist RL produced, and a student both give probabilities for the next token. The student is less sure.',
  'The oldest way: fine-tune the student on text the teacher wrote, exactly like SFT. Only the one token the teacher picked counts.',
  'Logit distillation uses the teacher\'s whole row as the target, so the student also learns how unlikely 54 and 48 are.',
  'Branch: both methods only trained on text the teacher wrote. When the student samples its own mistake, it is somewhere it never practised.',
  'On-policy distillation lets the student write and the teacher grade every token it wrote. 54 earns minus 1.79: the teacher finds it six times less likely.',
  'GRPO gave every token of this wrong answer the same minus 0.58. The teacher\'s grades are dense: the blame lands on 54, not on the correct steps before it.',
  'To merge RL specialists, each prompt the student answers is graded by the teacher for its domain. One student learns every specialist\'s skill.',
  'Training one skill after another tends to erase earlier ones: catastrophic forgetting. Distilling from all specialists at once keeps them, which is why 2026 pipelines end this way.',
  'Small models are made the same way: pretrain, fine-tune on a big teacher\'s traces, then distill on-policy. The cost: the signal runs out once the student matches the teacher.',
]);
