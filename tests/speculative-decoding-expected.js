// speculative-decoding storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test
// and the e2e spec. Try-this speedups print through formatRatio ("2.2×", not "2.20"), README lesson 35 / X-3.

export const CAPTIONS = [
  'Normal decoding: the big model, called the target here, makes one token per step. On the H200 example each step takes about 14.7 ms.',
  'A small drafter guesses the next three tokens: down, on, a. Each guess costs it about a twentieth of a target step.',
  'The target checks all three guesses in one pass, like a tiny prefill. Four positions cost about the same as one, because the step waits on memory.',
  'The target checks the guesses left to right. down and on pass; a fails, and anything guessed after the first failure is thrown away.',
  'At the failed position the target supplies its own token, the. This round made three tokens for about the price of one target step.',
  'A guess is kept with a chance set by both models, and a failure is redrawn from the probability the drafter left over. The result follows the target exactly.',
  'Average over many positions and you get an acceptance rate. At 0.7 with three guesses, a round yields 2.53 tokens; each extra guess adds less.',
  'Drafting is not free, so the speedup is smaller than the token count. With a drafter costing a twentieth of a step, three guesses give 2.2 times.',
  'With 128 users the verify pass checks 512 tokens, past the ridge, so it costs real math. The speedup falls from 2.2 to 1.53, and lower still as users grow.',
  'In 2026 open-model serving, the drafter is usually part of the model: an MTP head or an EAGLE-style head. Parallel drafters guess every token in one pass.',];

export const CHECK_WORK = [
  'E = (1 − α^(k+1)) / (1 − α) = (1 − 0.7^4) / (1 − 0.7) = 2.53 tokens per round',
  'speedup = E / (1 + k·c) = 2.53 / (1 + 3 · 0.05) = 2.53 / 1.15 = 2.2× (rounded once, from the unrounded E)',
].join('\n');

// What the page prints at the 211-user stop and at 2,048 tokens of context (storyboard §6's visible line).
export const USERS_NOTE = 'Each user has 1,024 tokens of context, so 211 users fit on one H200 (141 GB nominal, 70 GB of it weights); at 2,048 tokens it would be 105.';

// "prompt → Insight: … rest"; the page prints the arrow between the prompt and the bold insight.
export const TRY_THIS = [
  'With Acceptance rate α at 0.7, Drafter cost per guess at 0.05 and Users in the batch at 1, slide Guesses per round 1 → 3 → 5 → 8: tokens per round 1.70 → 2.53 → 2.94 → 3.20; speedup 1.62× → 2.2× → 2.35× → 2.28×. → Insight: guesses have diminishing returns. A guess counts only if every earlier one survived, while each one costs drafting time, so the best number of guesses is small and depends on α and the drafter cost.',
  'With Guesses per round at 3, slide Users in the batch 1 → 64 → 128 → 211: speedup 2.2× → 2.14× → 1.53× → 1.19×. Now set Guesses per round to 5 at 211 users: 0.91×. → Insight: speculative decoding spends idle arithmetic, so it fades when the batch has none left. It is a latency tool for small batches, not a free throughput multiplier.',
  'Press MTP 0.85, which sets Guesses per round to 1: 1.85 tokens per round, 1.76× at 1 user and still 1.73× at 128 users. → Insight: one well-trained extra guess survives large batches, because the verify pass only doubles the tokens. That is close to DeepSeek-V3\'s reported 1.8× tokens per second and why MTP heads are popular for serving.',
  'In the position panel, switch Drafter\'s guess from down to on, then up: keep chance 0.857 → 1.000 → 1.000; the result stays 0.60, 0.25, 0.10, 0.05 every time. → Insight: only guesses the drafter overrates are ever rejected, and the leftover repays exactly what rejection removed, so the target\'s distribution is preserved whatever the drafter does.',
];
