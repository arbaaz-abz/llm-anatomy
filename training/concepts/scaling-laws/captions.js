// The ten captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  "Training costs about six floating-point operations per parameter per token: two forward, four backward. DeepSeek-V4-Pro's run comes to about ten trillion trillion.",
  "The same budget can buy a small model reading many tokens or a big one reading few. Tokens per parameter measures the split.",
  "A scaling law is a smooth fit of loss against parameters and tokens, from hundreds of smaller runs (2022). Along a fixed budget, it is a valley.",
  "The bottom of the valley is the compute-optimal split: about 20 tokens per parameter, the 2022 Chinchilla rule. Earlier 2020 laws had favored bigger models.",
  "Bigger budgets move the optimum along both axes together: model and data grow at about the same rate, and the ratio stays near 20.",
  "Serving costs about two operations per parameter for every token generated. For a popular model, serving can dwarf training.",
  "Over-training: a smaller model fed more tokens reaches the same loss. Training costs more, serving much less, so over a model's life it is far cheaper.",
  "2026 open models train at hundreds to thousands of tokens per active parameter, far past Chinchilla. Counted per total parameter, the gap is much smaller.",
  "A weight update stretches some directions far more than others; the stretch factors are its singular values. Here one direction is ten times stronger than the other.",
  "Muon repeats a cheap polynomial step that pulls every singular value toward one, so every direction moves equally. DeepSeek-V4's last two steps settle them on one to four decimals.",
]);
