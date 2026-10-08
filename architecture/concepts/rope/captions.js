// The ten captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  "So far attention cannot see order: \"sat\" scores \"cat\" 3.0 whether \"cat\" is next to it or far away. Only the mask knows anything about position.",
  "RoPE cuts each query and key into pairs of numbers and treats each pair as a clock hand. Four numbers make two hands.",
  "Each hand turns by the token's position times its own speed. \"sat\" is token 3, so the fast hand turns 3 radians and the slow one 0.3.",
  "The score is still a dot product, now of the turned vectors. Each pair adds its own part, so the score drops from 3.0 to 1.596.",
  "Move both words ten tokens later: every hand turns further, but the angle between each pair stays the same. The score depends only on the offset.",
  "The fast hand finishes a turn every 6.3 tokens, so it tracks nearby order; the slow one takes 63. Tokens per full turn is the wavelength.",
  "A constant called the base sets how slow the slowest hand turns. A bigger base means a longer reach, so 2026 models raise it into the millions.",
  "Past its training length, a slow hand reaches angles the model never saw. Position interpolation slows every hand by the stretch factor, so all angles look familiar again.",
  "YaRN squeezes only the slow hands, which never finished a turn in training, and leaves the fast ones alone. Nearby words stay as distinct as before.",
  "Some models turn only part of each vector (partial RoPE), leaving the rest for content matching. Some layers turn none at all (NoPE); the causal mask still leaks order.",
]);
