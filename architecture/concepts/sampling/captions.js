// The nine captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'The model\'s last step gives one score per word in its vocabulary, called the logits. Softmax turns them into probabilities that add to 1.',
  'Greedy decoding always takes the most likely token. It is predictable, but the same prompt always gives the same text, and it can repeat itself.',
  'Sampling draws a random number and picks the token whose slice of the strip it lands in. Here 0.55 lands in the slice for the period.',
  'Temperature divides every logit before softmax. Below 1 the favorite takes more; above 1 the long tail takes more.',
  'Top-k keeps only the k most likely tokens and rescales them to add up to 1. With k equal to 3, "on" now gets 55%.',
  'Top-p keeps the fewest tokens whose probabilities reach p, then rescales. It keeps few tokens when the model is sure and many when it is not.',
  'Each draw uses a new random number, so the same prompt gives different text. A fixed seed repeats the same draws exactly.',
  'A multi-token-prediction head guesses the token after next in the same step. Here the model outputs "on" and drafts "the".',
  'The next step checks the draft while it computes anyway. If it agrees, the draft is kept and two tokens come out of one step.',
]);
