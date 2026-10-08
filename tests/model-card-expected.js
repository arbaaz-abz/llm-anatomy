// model-card storyboard §5 captions, verbatim. Imported by the page test and the e2e spec.
// (The storyboard gives no "Check my work" text: the toy prints a table, not a worked sum.)
export const CAPTIONS = [
  "A model card lists a model's shape in numbers. Each field below points at one part of the decoder you have already taken apart.",
  "Total counts every expert; active counts what one token runs. This model stores 1.6 trillion parameters and uses 49 billion per token, about 3%.",
  "\"Layers\" on a card counts blocks: each one is attention plus an MLP or experts, added onto the stream. Here there are 61.",
  "Each token's router picks 6 of 384 small experts, plus one shared expert that every token uses.",
  "The attention line says how much each token stores: one shared 512-wide key/value set, merged across tokens, plus a short window.",
  "Turn the attention line into memory. Its paper gives only a ratio, so this is an estimate: about 4 to 12 kB per token.",
  "\"1M context\" is how many positions the model accepts. Reaching it took a stretched position encoding, staged training and cheap attention.",
  "The modalities field says what can enter the stream. DeepSeek-V4-Pro takes text only; its paper names images as future work.",
  "Some fields describe training or storage, not the block. They lead to the training and serving lessons.",
  "Sources disagree, and labs count differently. When they do, this course shows both numbers and where each came from.",
];
