// decoder-anatomy storyboard §5 captions and §6 "Check my work" text, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "Whatever comes in is cut into pieces: a text token, a small square of an image, a slice of audio. Each piece will get exactly one row below.",
  "A text token looks up its row in the embedding table. An image patch passes through a vision encoder and a projector, and arrives as the same kind of vector.",
  "These four rows are the residual stream. Each block (one repeated unit of the model) reads it and adds to it, so the embedding is still there at the end.",
  "First half of a block: normalize, let each row read the other rows (attention), add the result back. This is the one place where positions talk to each other.",
  "Second half: normalize, push each row alone through a small two-layer network, add back. Rows never see each other here, and this is where most parameters live.",
  "In most 2026 models the MLP is a Mixture of Experts: a router sends each row to 2 of 8 small MLPs. All 8 count as total, 2 as active.",
  "A model is this block repeated N times, each with its own weights: talk, think, talk, think. \"Layers: 61\" on a model card counts these blocks.",
  "After the last block, the last row is normalized and multiplied by the unembedding matrix: one score per vocabulary word. Softmax turns the 16 scores into probabilities: \"on\" 39%.",
  "Pick a token, append it, run again. Only the new row is computed; the earlier rows' keys and values were stored in the KV cache and are read, not recomputed.",
];
export const CHECK_WORK = [
  "per block",
  "  attention   W_Q, W_K, W_V, W_O        4 × (8 × 8)          =   256",
  "  MLP         W_in, W_gate, W_out       3 × (8 × 16)         =   384",
  "  norms       2 × 8                                          =    16",
  "  block                                                      =   656",
  "× 2 blocks                                                   = 1,312",
  "embedding table      16 × 8                                  =   128",
  "final norm                                                   =     8",
  "unembedding          16 × 8                                  =   128",
  "total                                                        = 1,576",
  "active = total − embedding lookup                            = 1,448",
].join('\n');
