// rlvr-grpo storyboard §5 captions, one per frame. content.js puts them on the steps and frames.js labels each stage with them.
// Frames 4 and 5 print "1.73" where the storyboard has "+1.73": a caption holds no operator (lesson-spec, README lesson 2).
export const CAPTIONS = Object.freeze([
  'One prompt, eight sampled answers from the same model. Only the random sampling differs.',
  'A program checks each final answer: 1 if it is 56, 0 if not. Row 5 has wrong working and a right final token, and still gets a 1.',
  'The group\'s own mean is the baseline, the score an answer has to beat: 2 of 8 right gives 0.25. Right answers sit 0.75 above it, wrong ones 0.25 below.',
  'Divide by the group\'s spread, 0.43, and you have each answer\'s advantage: 1.73 for right, −0.58 for wrong. Within a group they always sum to zero.',
  'Every token in a right answer gets 1.73 and is pushed up; every token in a wrong one gets −0.58 and is pushed down, shared prefix included.',
  'All right, or all wrong: no spread, so every advantage is 0 and nothing moves. Dynamic sampling drops such groups and samples again until the batch has signal.',
  'PPO keeps four models in memory, including a critic network that estimates the baseline. GRPO takes the baseline from the group, and RLVR swaps the reward model for a checker.',
  'The trainer updates on the same answers several times, so 56 is already 25% more likely than when it was sampled. Past 1.2 the clip switches its gradient off.',
  'Clip-higher raises only the upper edge, to 1.28 (DAPO, GLM-5). A good token that started out unlikely can keep growing, so the model keeps exploring.',
]);
