// scaling-laws storyboard §5 captions, §6 "Check my work" and try-this prompts, pinned literally. Imported by the page test and the e2e spec.
export const CAPTIONS = [
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
];
export const CHECK_WORK = [
  "N = 9.59 × 10¹⁰ parameters (active)",
  "D = 10²⁴ ÷ (6 × 9.59 × 10¹⁰) = 1.74 × 10¹² tokens",
  "tokens per parameter = D ÷ N = 18.1",
  "loss = 1.8172 + 482.01 ÷ N^0.3478 + 2085.43 ÷ D^0.3658",
  "     = 1.8172 + 0.07305 + 0.06946 = 1.960",
].join("\n");
export const TRY_THIS = [
  ["Predict first: at 10²⁴ FLOPs, is a 1T-parameter model better than the compute-optimal 95.9B one? Drag **Model size** from 95.9B to 1T → loss 1.960 → 2.013, tokens 1.74T → 167B. Drag down to 1B → 2.187","There is a valley."," Too big and the model sees too few tokens; too small and it cannot use them. The bottom sits near 18.1 tokens per parameter at this budget, the Chinchilla rule of about 20."],
  ["Leave **Model size** at the optimum and step **Tokens served** at 10²⁴ FLOPs: **1T** → the cheapest model is 79B at 27.1 tokens per parameter (1.6% saving); **10T** → 48.4B at 88.8 (24.0%); **100T** → 29.2B at 494.9 (58.5%); **1,000T** → 21B at 3,005.5 (74.1%)","The more a model will be used, the smaller and longer-trained it should be."," At heavy use the best ratio lands in the hundreds to thousands of tokens per parameter, where 2026 models actually are (frame 8)."],
  ["Switch **Budget** from 10²² to 10²⁶ with **Tokens served** at 0: the optimum goes 9.05B / 184B tokens → 1.02T / 16.4T, and the ratio only drifts 20.4 → 16.1","Compute-optimal scales parameters and tokens together."," Ten thousand times the budget buys about a hundred times more of each."],
];
