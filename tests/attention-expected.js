// attention storyboard §5 captions and §6 "Check my work" text, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "In a real model, three learned matrices turn each token's vector into a query, a key and a value. Here they are hand-picked so you can check every number.",
  "A score is the query dotted with a key. Against the four keys, \"sat\" scores −1.0, 3.0, 0.5 and −0.75; the biggest is \"cat\".",
  "Every query scores every key at once, in one matrix product. The grid has one row per query and one column per key.",
  "Divide every score by the square root of the vector length: 4 numbers, so divide by 2. Longer vectors give bigger scores; dividing keeps softmax soft.",
  "Causal mask: a token may only look at itself and earlier tokens. Future cells become −∞, so softmax will give them exactly 0.",
  "Softmax: exponentiate each score, then divide by the sum so the row adds to 1. \"sat\" now puts 70% of its attention on \"cat\".",
  "The output is the weighted sum of the values. \"sat\" leaves carrying mostly cat's value.",
  "Do the same for every row: one head is one attention pattern, the heatmap, and one output vector per token.",
  "A second head has its own W_Q, W_K and W_V, so it finds a different pattern: here, each token looks one step back.",
  "Heads run side by side. Their outputs are joined into one row of 8 numbers per token and mixed by one more matrix, W_O.",
];
export const CHECK_WORK = [
  "q_sat · k_The  = 0·1    + 2·(−0.5) + 0.5·0   + 0·0.5    = −1.0",
  "q_sat · k_cat  = 0·0    + 2·1.5    + 0.5·0   + 0·(−0.5) =  3.0",
  "q_sat · k_sat  = 0·(−0.5) + 2·0    + 0.5·1   + 0·0.5    =  0.5",
  "q_sat · k_down = masked (−∞)",
  "÷ √4 = ÷ 2           →  −0.5     1.5      0.25",
  "exp (calculator)     →   0.607   4.482    1.284     sum 6.372",
  "÷ sum                →   0.095   0.703    0.202     (adds to 1.000)",
  "output = 0.095·v_The + 0.703·v_cat + 0.202·v_sat",
  "       = [−0.11, 1.41, 0.11, 0.80]",
].join('\n');
