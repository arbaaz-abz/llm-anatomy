// agentic-rl storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
export const CAPTIONS = Object.freeze([
  'In agentic RL the model acts inside an environment. It writes a tool call, a sandbox runs it, and the result comes back into its context.',
  'The sandbox\'s reply is an observation: the model did not write it, so it is masked out of the loss. Only the model\'s own 8 tokens will be trained.',
  'Episode or plain answer, each of the eight is scored and compared exactly as on the previous page. The update is still GRPO; everything around it changes.',
  'When nothing can be tested, a generative reward model writes a rubric for the task and scores each attempt against it. DeepSeek-V4 uses the policy itself as judge.',
  'Synchronous training waits for the slowest episode before it updates. Here one 16-minute episode leaves the other generators idle most of the time.',
  'Asynchronous training updates once most episodes are done; the rest finish under newer weights. Their tokens are off-policy: sampled by a slightly older model than the one being trained.',
  'Even with identical weights, the sampling engine and the trainer compute different probabilities for the same token. Here the trainer finds this 48 more than three times likelier.',
  'An importance-sampling correction reweights each token by that ratio, caps it, or drops it. IcePop, used by GLM-5, masks any token whose two probabilities differ more than twofold.',
  'Most 2026 open-model reports drop the KL term or make it tiny. Clip and IS masks already bound each step, checkers are hard to game, and the reference leaves memory.',
  'Agentic RL is now where most RL compute goes: thousands of sandboxes, long episodes, and separate specialist models. Distillation, next, merges those specialists into one.',
]);
