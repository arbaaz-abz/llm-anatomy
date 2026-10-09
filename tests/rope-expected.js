// rope storyboard §5 captions, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "So far attention cannot see order: \"sat\" scores \"cat\" 3.0 whether \"cat\" is next to it or far away. Only the mask knows anything about position.",
  "RoPE cuts each query and key into pairs of numbers and treats each pair as a clock hand. Four numbers make two hands.",
  "Each hand turns by the token's position times its own speed. \"sat\" is token 3, so the fast hand turns 3 radians and the slow one 0.3.",
  "The score is still a dot product, now of the turned vectors. Each pair adds its own part, so the score drops from 3.0 to 1.596.",
  "Move both words ten tokens later: every hand turns further, but the angle between each pair stays the same. The score depends only on the offset.",
  "The fast hand finishes a turn every 6.3 tokens, so it tracks nearby order; the slow one takes 63. Tokens per full turn is the wavelength.",
  "A constant called the base sets how slow the slowest hand turns. A bigger base means a longer reach, so 2026 models raise it into the millions.",
  "Past its training length, a slow hand reaches angles the model never saw. Position interpolation slows every hand by the stretch factor, so all angles look familiar again.",
  "YaRN squeezes only the slow hands, which never finished a turn in training, and leaves the fast ones alone. Nearby words stay as distinct as before.",
  "Some models turn only part of each vector (partial RoPE), leaving the rest for content matching. Some layers turn none at all (NoPE); the causal mask still implies order.",
];
// The toy's "Check my work" text for its opening state (sat at 3, cat at 2, base 100): not in the storyboard, hand-checked against §5 frames 3-4.
export const CHECK_WORK = [
  "pair 1  q_sat (0, 2) turned 3 × 1 = 3 rad         → (−0.282, −1.980)",
  "        k_cat (0, 1.5) turned 2 × 1 = 2 rad       → (−1.364, −0.624)",
  "        dot (−0.282)(−1.364) + (−1.980)(−0.624) = 1.621",
  "pair 2  q_sat (0.5, 0) turned 3 × 0.1 = 0.3 rad   → (0.478, 0.148)",
  "        k_cat (0, −0.5) turned 2 × 0.1 = 0.2 rad  → (0.099, −0.490)",
  "        dot (0.478)(0.099) + (0.148)(−0.490) = −0.025",
  "score  1.621 + (−0.025) = 1.596   (unrotated 3.000)",
].join('\n');
