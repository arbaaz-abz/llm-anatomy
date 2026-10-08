// decoder-recap storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
export const CAPTIONS = Object.freeze([
  "This is GPT-3's block from 2020. A 2026 block keeps the same wiring, and almost every box has been swapped.",
  'RMSNorm skips subtracting the mean and just rescales the row. It is cheaper and trains as well, so most 2026 models use it.',
  'The position table is gone. RoPE turns each query and key by its position inside attention, so scores depend on distance and no table can run out.',
  'The MLP gains a gate: one branch decides how much of the other passes through. It uses three matrices, so the hidden width shrinks to keep the size the same.',
  'In most 2026 models the MLP becomes a Mixture of Experts. The model stores far more than each token uses.',
  "Query heads now share their keys and values, or rebuild them from a small latent. GPT-3's shape with 8 shared sets stores 12 times less per token.",
  'QK-norm normalizes queries and keys before the dot product. However large they grow during training, the scores stay in a fixed range, so softmax stays soft.',
  'Some models add a learned sink score per head, so softmax can put its weight on "nothing". It matters most in window layers.',
  'Attention layers are no longer all alike. Many 2026 stacks mix full attention with window or linear layers that keep far less memory.',
  'Same wiring, new parts: RMSNorm, rotated positions, a gated MLP or experts, shared keys and values, and stabilized softmax. Most swaps cut what a token costs to run.',
]);
