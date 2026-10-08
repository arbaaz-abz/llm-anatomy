// sampling storyboard §5 captions, verbatim, and the toy's "Check my work" text (§6) for the default state (T = 1)
// and the storyboard's own T = 0.5 example. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "The model's last step gives one score per word in its vocabulary, called the logits. Softmax turns them into probabilities that add to 1.",
  "Greedy decoding always takes the most likely token. It is predictable, but the same prompt always gives the same text, and it can repeat itself.",
  "Sampling draws a random number and picks the token whose slice of the strip it lands in. Here 0.55 lands in the slice for the period.",
  "Temperature divides every logit before softmax. Below 1 the favorite takes more; above 1 the long tail takes more.",
  "Top-k keeps only the k most likely tokens and rescales them to add up to 1. With k equal to 3, \"on\" now gets 55%.",
  "Top-p keeps the fewest tokens whose probabilities reach p, then rescales. It keeps few tokens when the model is sure and many when it is not.",
  "Each draw uses a new random number, so the same prompt gives different text. A fixed seed repeats the same draws exactly.",
  "A multi-token-prediction head guesses the token after next in the same step. Here the model outputs \"on\" and drafts \"the\".",
  "The next step checks the draft while it computes anyway. If it agrees, the draft is kept and two tokens come out of one step.",
];

export const CHECK_WORK = [
  'on         e^2           = 7.39',
  '"."        e^1.5         = 4.48',
  'and        e^0.5         = 1.65',
  'the        e^0           = 1',
  '12 others  12 × e^−1     = 4.41',
  'sum                      = 18.93',
  'on = 7.39 ÷ 18.93 = 0.390',
].join('\n');

export const CHECK_WORK_HALF = [
  'on         e^4           = 54.6',
  '"."        e^3           = 20.1',
  'and        e^1           = 2.72',
  'the        e^0           = 1',
  '12 others  12 × e^−2     = 1.62',
  'sum                      = 80.03',
  'on = 54.6 ÷ 80.03 = 0.682',
].join('\n');

// The same T = 1 box after top-p 0.7 (kept 3 tokens holding 0.714): the rescale line is added.
export const CHECK_WORK_TOP_P = [
  ...CHECK_WORK.split('\n'),
  'top-p 0.7 kept 3 tokens, holding 0.714',
  'on = 0.390 ÷ 0.714 = 0.547',
].join('\n');

export const CHECK_WORK_GREEDY = [
  'greedy (temperature 0): no division, no exponentials.',
  'The largest score takes everything: on = 1, every other word 0.',
].join('\n');
