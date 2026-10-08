// decoder-recap storyboard §5 captions and the page text it prints verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "This is GPT-3's block from 2020. A 2026 block keeps the same wiring, and almost every box has been swapped.",
  "RMSNorm skips subtracting the mean and just rescales the row. It is cheaper and trains as well, so most 2026 models use it.",
  "The position table is gone. RoPE turns each query and key by its position inside attention, so scores depend on distance and no table can run out.",
  "The MLP gains a gate: one branch decides how much of the other passes through. It uses three matrices, so the hidden width shrinks to keep the size the same.",
  "In most 2026 models the MLP becomes a Mixture of Experts. The model stores far more than each token uses (moe).",
  "Query heads now share their keys and values, or rebuild them from a small latent. GPT-3's shape with 8 shared sets stores 12 times less per token.",
  "QK-norm normalizes queries and keys before the dot product. However large they grow during training, the scores stay in a fixed range, so softmax stays soft.",
  "Some models add a learned sink score per head, so softmax can put its weight on \"nothing\". It matters most in window layers.",
  "Attention layers are no longer all alike. Many 2026 stacks mix full attention with window or linear layers that keep far less memory.",
  "Same wiring, new parts: RMSNorm, rotated positions, a gated MLP or experts, shared keys and values, and stabilized softmax. Most swaps cut what a token costs to run.",
];
// Frame 5's "Numbers shown", exactly (storyboard §5), printed under the stage.
export const FRAME5_EXACT = "GPT-3's shape with SwiGLU, no biases, 8 KV heads, then 64 experts of hidden 4,096 (an eighth of 32,768), top-8: total 959,815,311,360 (960B), active 148,066,492,416 (148B) vs 147,990,994,944 dense; the difference is the router.";
// Frame 7's visible line under the stage (README lesson 10).
export const FRAME7_LINE = "This row is [[attention]]'s hero row; that page computes it step by step.";
// The visible stand-in line under the stage (storyboard §4).
export const STAND_IN = "Rows are the earlier pages' hand-picked numbers; parameter and cache counts are exact for GPT-3's shape.";
// Visible note on the residual row (frame 10).
export const RESIDUAL_NOTE = "New in 2026 (DeepSeek-V4, Kimi K3); this course does not cover how they work.";
