// The ten captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'In a real model, three learned matrices turn each token\'s vector into a query, a key and a value. Here they are hand-picked so you can check every number.',
  'A score is the query dotted with a key. Against the four keys, "sat" scores −1.0, 3.0, 0.5 and −0.75; the biggest is "cat".',
  'Every query scores every key at once, in one matrix product. The grid has one row per query and one column per key.',
  'Divide every score by the square root of the vector length: 4 numbers, so divide by 2. Longer vectors give bigger scores; dividing keeps softmax soft.',
  'Causal mask: a token may only look at itself and earlier tokens. Future cells become −∞, so softmax will give them exactly 0.',
  'Softmax: exponentiate each score, then divide by the sum so the row adds to 1. "sat" now puts 70% of its attention on "cat".',
  'The output is the weighted sum of the values. "sat" leaves carrying mostly cat\'s value.',
  'Do the same for every row: one head is one attention pattern, the heatmap, and one output vector per token.',
  'A second head has its own W_Q, W_K and W_V, so it finds a different pattern: here, each token looks one step back.',
  'Heads run side by side. Their outputs are joined into one row of 8 numbers per token and mixed by one more matrix, W_O.',
]);
