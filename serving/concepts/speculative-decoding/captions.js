// speculative-decoding storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
export const CAPTIONS = Object.freeze([
  'Normal decoding: the big model, called the target here, makes one token per step. On the H200 example each step takes about 14.7 ms.',
  'A small drafter guesses the next three tokens: down, on, a. Each guess costs it about a twentieth of a target step.',
  'The target checks all three guesses in one pass, like a tiny prefill. Four positions cost about the same as one, because the step waits on memory.',
  'The target checks the guesses left to right. down and on pass; a fails, and anything guessed after the first failure is thrown away.',
  'At the failed position the target supplies its own token, the. This round made three tokens for about the price of one target step.',
  'A guess is kept with a chance set by both models, and a failure is redrawn from the probability the drafter left over. The result follows the target exactly.',
  'Average over many positions and you get an acceptance rate. At 0.7 with three guesses, a round yields 2.53 tokens; each extra guess adds less.',
  'Drafting is not free, so the speedup is smaller than the token count. With a drafter costing a twentieth of a step, three guesses give 2.2 times.',
  'With 128 users the verify pass checks 512 tokens, past the ridge, so it costs real math. The speedup falls from 2.2 to 1.53, and lower still as users grow.',
  'In 2026 open-model serving, the drafter is usually part of the model: an MTP head or an EAGLE-style head. Parallel drafters guess every token in one pass.',
]);
