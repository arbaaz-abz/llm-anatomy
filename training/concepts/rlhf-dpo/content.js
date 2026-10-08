// rlhf-dpo: the complete lesson (storyboard docs/storyboards/rlhf-dpo.md §3, §5–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { FRAMING, ROWS, PROSE, FRAME_11_NOTE } from './facts.js';

export { CAPTIONS };

const SLUG = 'rlhf-dpo';
const LAST_FRAME = CAPTIONS.length - 1;

const HOOK = 'How do you train a model toward the answers people prefer, when "better" is something no program can check?';

const INTUITION = Object.freeze([
  'For a question like "why is the sky blue?" asked by a child, there is no unit test. What you can get is a judgment: show a person two answers and ask which is better. RLHF (reinforcement learning from human feedback) turns many such judgments into a training signal in two steps. First it trains a reward model: a copy of the language model with a score head, adjusted until it scores the picked answer of each pair above the other. Then it runs reinforcement learning against that score.',
  'The model being trained is called the policy. In PPO, the classic algorithm, the policy writes fresh answers, the reward model scores them, and a fourth network, the critic, predicts what score this prompt usually earns. Each answer\'s tokens are pushed up or down by how much the answer beat that prediction. Two brakes keep the steps safe. The clip range ε stops pushing a token once its probability has already moved more than 20%, in the direction the update was pushing it, from where it was when the answer was sampled. The KL term subtracts a penalty for drifting away from a frozen reference model, the SFT model the run started from. The KL term matters because the reward model is only an approximation: left alone, the policy finds answers it overrates, such as flattery, and the score rises while real quality falls.',
  'That is four models in memory (policy, reference, reward model, critic) and a sampling loop on every step. DPO (direct preference optimization) skips both the reward model and the sampling. It works straight from the pairs: for each pair it measures how much more likely the policy made each answer than the reference does, and widens the gap between the chosen and rejected answers. The price is that it only ever learns from the fixed pairs it was given, and, since it optimizes a gap, the chosen answer itself can get less likely (try-this 2).',
  'In 2026 both still exist, but in narrower roles. Reasoning comes from RL with checkable rewards. RLHF, now usually with a judge model as the reward, polishes helpfulness, style and safety at the end of the pipeline (Nemotron 3 Super runs it as a separate final stage). DPO is the cheap preference stage of smaller, fully open pipelines (Olmo 3 runs it between SFT and RL).',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{Reward model (Bradley–Terry):}\quad P(\htmlClass{hl-a}{A} \succ B) = \sigma\big(r_\phi(x, \htmlClass{hl-a}{A}) - r_\phi(x, B)\big), \qquad \mathcal{L}_{\text{RM}} = -\ln \sigma\big(r_\phi(x, A) - r_\phi(x, B)\big)`, note: 'r_φ outputs one scalar per (prompt, answer): here r(A) = 1.2 and r(B) = −0.3, so P = σ(1.5) = 0.818 and the loss is 0.201.' },
  { tex: tex`\text{RLHF objective:}\quad \max_\theta\; \mathbb{E}_{y \sim \htmlClass{hl-pol}{\pi_\theta}}\big[\, r_\phi(x, y) \,\big] - \htmlClass{hl-beta}{\beta}\, \mathbb{D}_{\text{KL}}\big(\htmlClass{hl-pol}{\pi_\theta}(\cdot \mid x) \,\|\, \htmlClass{hl-ref}{\pi_{\text{ref}}}(\cdot \mid x)\big), \qquad \mathbb{D}_{\text{KL}}(p \,\|\, q) = \sum_v p_v \ln \frac{p_v}{q_v}`, note: 'The probability rows on screen are [4] slices of a [vocab] distribution at one position.' },
  { tex: tex`\text{PPO step:}\quad \hat A_t = r - V_\psi(x)\ \ (\text{GAE in practice}),\qquad \min\!\Big( r_t \hat A_t,\ \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\, \hat A_t \Big),\quad r_t = \frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{old}}(y_t \mid \cdot)}`, note: 'At r = 1.25 with advantage +0.2: min(1.25 · 0.2, 1.2 · 0.2) = 0.24, clipped.' },
  { tex: tex`\text{DPO:}\quad \mathcal{L}_{\text{DPO}} = -\ln \sigma\Big( \htmlClass{hl-beta}{\beta}\big[ \underbrace{\ln \tfrac{\pi_\theta(A)}{\pi_{\text{ref}}(A)}}_{d_A} - \underbrace{\ln \tfrac{\pi_\theta(B)}{\pi_{\text{ref}}(B)}}_{d_B} \big] \Big)`, note: 'd_A and d_B are sums of per-token log-probability changes over each whole answer (scalars).' },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: r_φ gives one score per (prompt, answer); the probability rows on the stage are [4] slices of a [vocab] distribution at one position; d_A and d_B are sums of per-token log-probability changes over each whole answer (scalars). Hover a highlighted term to outline its glyph: A is the chosen answer (frames 1, 2, 10), π_θ the policy row, π_ref the reference row (frames 6–8), β the β label (frames 8, 10).',
  'Notes: (a) DPO is derived by solving the RLHF objective above in closed form (the optimal policy is the reference reweighted by exp(r/β)) and substituting it into the Bradley–Terry loss, which is why the same β appears in both. (b) Real PPO computes per-token advantages with GAE and a per-token KL; this page uses one advantage and one position. (c) Variants: SimPO (no reference), KTO (single good/bad labels), ORPO (SFT and preference in one stage), APO.',
]);

const TAKEAWAYS = Object.freeze([
  'RLHF learns a reward model from preference pairs, then runs PPO against it: four models in memory (policy, reference, reward model, critic) and fresh samples every step.',
  'The reward model is an approximation the policy can game; the KL term against the frozen reference model is the anchor that stops it, and the clip range switches a token\'s update off once it has moved far enough.',
  'DPO gets the preference signal straight from the pairs (no reward model, no sampling) by widening the chosen-vs-rejected gap relative to the reference; in 2026 it is a cheap stage for open pipelines, while RLHF with a judge model polishes style and safety at the end.',
]);

const FURTHER = Object.freeze([
  { title: 'Nathan Lambert, RLHF Book', href: 'https://rlhfbook.com/', note: 'reward modeling, policy gradients, direct alignment' },
  { title: 'UNIPO', href: 'https://poloclub.github.io/unipo/', note: 'interactive comparison of REINFORCE, PPO and GRPO-family objectives with per-token coloring' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    animation: {
      label: 'RLHF and DPO: a reward model, a PPO update, the KL anchor and the DPO shortcut, in eleven steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The scores and probabilities are hand-picked stand-ins; the arithmetic on them is exact. Real KL terms sum over every token of an answer; this page shows one position.',
      belowFor: (index) => (index === LAST_FRAME ? [fill(FRAME_11_NOTE)] : []),
    },
    toy: {
      title: 'Tune a DPO pair.',
      intro: 'The pair from frame 1, its two log-probability changes, and β, with live DPO numbers. The log-probability changes are what training would produce; here you set them by hand.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: ROWS, prose: PROSE },
    takeaways: TAKEAWAYS,
    links: { next: ['rlvr-grpo'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
