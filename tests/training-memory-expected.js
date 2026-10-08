// training-memory storyboard §5 captions and §6 "Check my work" text, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "Training starts with the weights: 2 bytes per parameter in BF16. GPT-3's 175 billion parameters already need 350 GB, more than four H100s.",
  "Backpropagation writes a gradient for every weight, the same shape and size. That is 2 more bytes per parameter, 700 GB so far.",
  "Adam keeps two running averages per parameter, called optimizer states, in FP32 so tiny updates stay precise. They add 8 bytes, the biggest share.",
  "The optimizer updates an FP32 master copy of each weight, then rounds it to BF16 for the next step. Total: 16 bytes per parameter, 2.8 TB for GPT-3.",
  "The backward pass needs numbers the forward pass computed, called activations, so each block saves them. For one 2,048-token sequence, GPT-3's 96 blocks save 275 GB.",
  "Most of it is the attention-score grid. Recomputing it during the backward pass, or never storing it as FlashAttention does, cuts the save to 82 GB.",
  "Full recomputation keeps only each block's input and reruns its forward pass during backward. Saved activations drop to 4.8 GB for about a third more compute.",
  "Sixty-four GPUs, each training on different data, is data parallelism. Every GPU keeps a full copy, so each one still needs all 2.8 TB.",
  "ZeRO shards the optimizer states: each GPU keeps one sixty-fourth and updates only that slice. Per-GPU memory falls from 2,800 GB to 733 GB.",
  "Shard gradients and weights too, and each GPU holds 44 GB of state. Before each layer runs, forward and backward, the GPUs gather its weights: that is ZeRO-3, or FSDP.",
  "A trillion parameters need 16 TB of training state: 200 H100s before any activations. A mixture of experts pays here for every expert, even the ones a token skips."
];

export const CHECK_WORK = [
  "weights     = 175B × 2 B = 350 GB",
  "gradients   = 175B × 2 B = 350 GB",
  "optimizer   = 175B × (4 + 8) B = 2,100 GB   (stage 0: nothing sharded)",
  "activations = 96 × 2,048 × 12,288 × (34 + 5 × 96 × 2,048 ÷ 12,288) B = 275.41 GB",
  "total       = 3,075.41 GB vs 80 GB (H100, nominal) → does not fit",
].join("\n");

// Storyboard §6 try-this numbers (per-GPU GB, GPU counts) as the toy must print them.
export const TRY_THIS = {
  zeroPaper: ["120", "31.41", "16.64", "1.88"],
  gpt3Zero3: { state: "43.75", none: "319.16", selective: "125.89", full: "48.58", stage0Full: "2,804.83" },
  v4Pro: { h100: 320, b200: 143, b300: 89, muonB300: 67, gpt3H100: 35 },
};
