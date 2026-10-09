// speculative-decoding: the complete lesson (storyboard docs/storyboards/speculative-decoding.md §3, §5, §7–§10). Pure, no DOM at import.
// lessonFor(data) fills the dated text from data/serving.json and models.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { expectedTokens, simpleSpeedup } from '@math/specdec.js';
import { renderFor } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { FRAMING, belowFor, factRows, stageText } from './facts.js';
import { ALPHA, C, K, GUESSES } from './numbers.js';
import { ratioText, tokensText } from './format.js';

const SLUG = 'speculative-decoding';

const HOOK = 'How can a small model\'s guesses make a big model faster without changing a single word the big model would have written?';

const GUESS_LIST = GUESSES.map((g) => `"${g}"`).join(' ');
const ROUND = tokensText(expectedTokens(ALPHA, K));
const SPEEDUP = ratioText(simpleSpeedup(ALPHA, K, C));

export const INTUITION = Object.freeze([
  `In [[prefill-decode]] you saw that a decode step mostly waits for the weights to arrive from memory: the arithmetic for one token is nearly free. Speculative decoding spends that free arithmetic. A small, fast drafter guesses the next few tokens, here ${GUESS_LIST}. Then the big model, the target, runs one pass over all of them at once: the guesses are already written down, so it can read them the way prefill reads a prompt. Because the step was waiting on memory anyway, checking four positions costs about the same as producing one.`,
  'The target then walks the guesses left to right. A guess survives with a probability that compares the two models: if the target likes the word at least as much as the drafter did, it is always kept; if the drafter was more confident than the target, it is kept only part of the time. At the first failure, the target picks a replacement from the probability the drafter left over, and everything after it is discarded. This rule, rejection sampling, makes the output follow the target\'s own probabilities exactly. With greedy decoding it reduces to "keep the guess if it is the target\'s top token". Either way the answer is the target\'s, only faster.',
  `How much faster depends on how often guesses survive. With a ${Math.round(ALPHA * 100)}% acceptance rate and ${K} guesses, a round yields ${ROUND} tokens on average, and after paying for the drafter that is about ${SPEEDUP} on the running example. The catch is the batch. With many users, checking four positions for each of them makes the verify pass large enough to become compute-bound, and the free arithmetic is gone. That is why speculative decoding is a per-user latency tool, strongest for interactive traffic at small batch. In 2026 open-model serving, the drafter is usually part of the model: multi-token prediction (MTP) heads or EAGLE-style heads that read the target's own hidden states, and newer parallel drafters guess all tokens in one pass. Many drafters also guess a small tree of alternatives instead of a single line, and the target checks every branch in the same pass. MTP heads are introduced in [[sampling]].`,
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`P(\text{keep } x) = \min\!\left(1,\ \frac{\htmlClass{hl-p}{p(x)}}{\htmlClass{hl-q}{q(x)}}\right), \qquad x' \sim \operatorname{norm}\big(\max(0,\ \htmlClass{hl-p}{p} - \htmlClass{hl-q}{q})\big) \text{ on reject}` },
  { tex: tex`\Pr[\text{output}=x] = q(x)\min\!\left(1,\tfrac{p(x)}{q(x)}\right) + \Big(1-\sum_y \min(p(y),q(y))\Big)\frac{\max(0,p(x)-q(x))}{\sum_y \max(0,p(y)-q(y))} = \htmlClass{hl-p}{p(x)}` },
  { tex: tex`\alpha = \sum_x \min(p(x), q(x)), \qquad \mathbb{E}[\text{tokens per round}] = \frac{1-\alpha^{k+1}}{1-\alpha}, \qquad \text{speedup} = \frac{\mathbb{E}\cdot t_{\text{step}}(B)}{k\,c\,t_{\text{step}}(B) + t_{\text{step}}\big(B(k+1)\big)}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: p, q ∈ ℝ^|V| (4 words in the zoom; the full vocabulary in a real model). t_step(n) is the step time of [[prefill-decode]] with n tokens and B sequences\' KV. The constant-α formula assumes every guess is accepted independently with the same α; real acceptance falls with depth. Hover a highlighted term to outline its glyph on the stage: p is the target\'s row and the result, q the drafter\'s row.',
]);

const TAKEAWAYS = Object.freeze([
  'A cheap drafter guesses k tokens and the target checks them all in one pass, which costs about one decode step while decode is memory-bound.',
  'Rejection sampling keeps a guess with chance min(1, p/q) and redraws failures from the leftover, so the output is exactly the target\'s; tokens per round grow with α but each extra guess adds less.',
  'The gain is per-user latency: as the batch grows the verify pass becomes compute-bound and the speedup falls, even below 1× with many guesses; one high-acceptance MTP guess holds up best.',
]);

const FURTHER = Object.freeze([
  { title: 'A Hitchhiker\'s Guide to Speculative Decoding (PyTorch)', href: 'https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/' },
  { title: 'EAGLE-3 (Li et al.)', href: 'https://arxiv.org/abs/2503.01840', note: 'the paper behind the 2025 head' },
  { title: 'P-EAGLE (vLLM)', href: 'https://vllm.ai/blog/2026-03-13-p-eagle', note: 'parallel drafting in one pass' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Speculative decoding in ten steps: draft, verify, accept or reject, and where the gain fades',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: renderFor(stageText(data)),
      standIn: 'The guesses, the probabilities and the drafter\'s cost are hand-picked stand-ins; the step times and the speedups are exact for them. Each user holds 1,024 tokens of context.',
      belowFor: belowFor(data),
    },
    toy: {
      title: 'Guess and check',
      intro: 'It opens on the animation\'s numbers: acceptance 0.7, three guesses, a drafter costing a twentieth of a step, one user.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows() },
    takeaways: TAKEAWAYS,
    links: { next: ['serving-calculator'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
