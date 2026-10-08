// agentic-rl storyboard §5 captions and §6 try-this wording, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "In agentic RL the model acts inside an environment. It writes a tool call, a sandbox runs it, and the result comes back into its context.",
  "The sandbox's reply is an observation: the model did not write it, so it is masked out of the loss. Only the model's own 8 tokens will be trained.",
  "Episode or plain answer, each of the eight is scored and compared exactly as on the previous page. The update is still GRPO; everything around it changes.",
  "When nothing can be tested, a generative reward model writes a rubric for the task and scores each attempt against it. DeepSeek-V4 uses the policy itself as judge.",
  "Synchronous training waits for the slowest episode before it updates. Here one 16-minute episode leaves the other generators idle most of the time.",
  "Asynchronous training updates once most episodes are done; the rest finish under newer weights. Their tokens are off-policy: sampled by a slightly older model than the one being trained.",
  "Even with identical weights, the sampling engine and the trainer compute different probabilities for the same token. Here the trainer finds this 48 more than three times likelier.",
  "An importance-sampling correction reweights each token by that ratio, caps it, or drops it. IcePop, used by GLM-5, masks any token whose two probabilities differ more than twofold.",
  "Most 2026 open-model reports drop the KL term or make it tiny. Clip and IS masks already bound each step, checkers are hard to game, and the reference leaves memory.",
  "Agentic RL is now where most RL compute goes: thousands of sandboxes, long episodes, and separate specialist models. Distillation, next, merges those specialists into one.",
];

// Storyboard §6 "Try this". The storyboard prints FP16 row 4's ρ as 1.157; mismatchRatio gives 1.1565 and the
// page prints what it computes at 3 decimals: 1.156 (a double rounding in the storyboard).
export const TRY_THIS = [
  {
    prompt: "Predict first: should the trainer fully trust the ratio? With row 4's 48 selected, step through “How the trainer treats the mismatch”: ignore → weight 1.000, push −0.577; full IS → weight 3.200, push −1.848; truncated IS → weight 2.000, push −1.155; IcePop → masked, push 0, and the readout says 3 of 39 masked.",
    insight: "corrections trade bias for variance.",
  },
  {
    prompt: "Keep IcePop and switch “Rollout and trainer number format” to FP16. Row 4's ρ falls 3.200 → 1.156, row 7's 2.400 → 1.116, row 6's 0.400 → 0.892, and the readout drops to 0 of 39 masked.",
    insight: "part of the mismatch is rounding,",
  },
  {
    prompt: "Step “Update when this share of episodes is done” from all 8 to 6 of 8 to 4 of 8: utilization 37.5% → 72.9% → 87.5%, while the rows that finish under newer weights go from none to rows 4 and 7 to rows 4, 5, 7 and 8.",
    insight: "async is a trade.",
  },
];
