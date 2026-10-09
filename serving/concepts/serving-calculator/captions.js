// serving-calculator storyboard §5 captions, verbatim (one per frame); content.js puts them on the steps, frames.js labels each stage.
// Wording changes against the storyboard, all course rulings: "Disaggregated serving" (the lesson's title, README lesson 33) for the
// backticked slug in caption 3; formatBytes outputs "54.1 GB" and "36.9" (P4-R6); ranges with an en dash, as in the storyboard.
export const CAPTIONS = Object.freeze([
  'The worked example: a 1.6-trillion-parameter MoE that uses 49 billion parameters per token. Shipped in 4- and 8-bit formats, it weighs about 865 GB.',
  'A GB300 holds 288 GB; three hold 864, a gigabyte short of the weights and with no room for anything else. The weights alone need four.',
  'In practice the replica spreads the experts over 16 GPUs, as in Disaggregated serving. Each GPU now holds 54.1 GB of weights and has 234 GB left for users.',
  'Each user\'s KV cache stays on the GPU for the whole answer. At 8K tokens a user costs 36.9–111 MB, so thousands fit beside the weights.',
  'At a million tokens a user costs 4–12 GB, so only 19–58 fit per GPU. Each step then re-reads a full GPU, capping users near 28 tokens per second.',
  'The fastest one user can go: each step must read the GPU\'s 54.1 GB of weights, which takes 6.76 ms. That is at most 148 tokens per second.',
  'A target of 27 tokens per second gives each step 37 ms. In the ideal, 1,889 users fit in that step before the math, not memory, runs out.',
  'Measured, it is far less: InferenceX got 6,182 tokens per second per GPU. Its 27 tokens per second per user implies 37 ms per token, 5.48 times the weight-read floor.',
  'A GPU-hour\'s price over the tokens it makes is the cost: about 12 cents per million input and output tokens on GB300. DeepSeek lists 66 cents in, $1.98 out.',
  'Output costs more because a GPU makes far fewer output tokens per second than it reads input. DeepSeek\'s nodes read five times more than they wrote.',
]);
