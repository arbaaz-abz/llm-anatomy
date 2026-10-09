// rlvr-grpo: the complete lesson (storyboard docs/storyboards/rlvr-grpo.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { formatRatio } from '@math/core.js';
import { framing, factRows } from './facts.js';
import { evaluate, unbiasedStd } from './model.js';
import { INITIAL_STATE, fixed, signed, signed2 } from './format.js';

const SLUG = 'rlvr-grpo';

const HOOK = 'How can a model get better at reasoning when the only feedback is "right" or "wrong" on its final answer?';

const intuition = () => {
  const { rows } = evaluate(INITIAL_STATE);
  const [up, down] = [rows[0].advantage, rows[1].advantage];
  return [
    'Reinforcement learning with verifiable rewards (RLVR) replaces the learned reward model with a program: a math checker, a unit test, a compiler, a format check. The program reads an answer and returns a reward, usually just 1 or 0. That is cheap, precise, and much harder to game than a learned reward model, and it is available for exactly the tasks where 2026 models made the biggest gains: math, code, science, tool use. RLVR needs an automatic checker, so it works where an answer can be verified (math, code, a required format); open-ended writing still relies on a reward model or preference pairs ([[rlhf-dpo]]).',
    'A 0/1 reward alone is not enough, because "right" means nothing without knowing how hard the prompt was. GRPO solves this with a group. For one prompt the policy (the model being trained) samples several answers (8 on this page; GLM-5 uses {glm-5.rl_group_size}). The checker scores each one. Each answer\'s advantage, how much better it did than its siblings, is its reward minus the group\'s mean, divided by the group\'s spread. The mean is the baseline: the score an answer has to beat. It is grading on a curve: a right answer in a group where everyone is right earns nothing; a right answer in a group of failures earns a lot. That baseline is what PPO had to learn with a separate critic network. The price is sampling: every prompt costs G full generations (forward passes only), so generation, not the update step (a forward and a backward pass), is where RL time goes; long rollouts leave GPUs idle unless they run asynchronously, and the total is no longer small (DeepSeek-V3.2 reports its post-training compute at {deepseek-v3.2.post_training_compute_share|raw}, in the report\'s own accounting).',
    `The advantage then flows back onto every token of its answer as a push: how strongly the update raises or lowers that token's probability. Each token in a right answer becomes more likely, each token in a wrong answer less likely, through the same kind of clipped ratio PPO uses. The trainer usually takes several update steps on one batch of answers, and with asynchronous rollouts the answers may come from a slightly older model. So by the time a token is used, its probability has already moved; the clip limits how far. Under an outcome reward, every token in an answer shares that answer's fate, including the shared prefix \`7 × 8 =\`. Across many groups, a token that doesn't change the odds of a right ending is pushed up about as often as down, so its net push is near zero; tokens that do change the odds keep a net push. In this group, \`7 × 8 =\` starts one right answer and three wrong ones: ${signed(up, 3)} − 3 × ${fixed(-down, 3)} ≈ 0 (exactly zero before rounding).`,
    'Because the reward only checks the final answer, any token sequence that ends in the right answer is rewarded. RL can only reinforce what the model already sometimes samples, so training starts from an SFT model that sometimes reasons its way to the answer. From there, decomposing the problem, checking, and backtracking all raise the chance of a right ending, so the policy learns to spend more tokens on them: that is why a right/wrong signal on the final answer is enough to grow reasoning. The original 2024 recipe was unstable at scale, and most 2026 open-model reports ship some mix of the same patches: a higher upper clip bound, a token-level loss, a KL term (a penalty for drifting from the starting model) that is tiny or gone, filtering of groups with no spread, and corrections for the gap between the engine that sampled the answers and the trainer that updates on them.',
  ];
};

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-r}{R_i} = \text{checker}(o_i) \in \{0, 1\}, \quad i = 1..G` },
  { tex: tex`\htmlClass{hl-a}{A_i} = \frac{\htmlClass{hl-r}{R_i} - \operatorname{mean}(R)}{\operatorname{std}(R)} \qquad A_i = 0 \text{ when } \operatorname{std}(R) = 0 \qquad \text{(Dr.GRPO / DeepSeek-V3.2: } A_i = R_i - \operatorname{mean}(R)\text{)}` },
  { tex: tex`\htmlClass{hl-ratio}{r_{i,t}} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}` },
  { tex: tex`\mathcal{J}_{\text{sample}} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\Big( \htmlClass{hl-ratio}{r_{i,t}}\,\htmlClass{hl-a}{A_i},\; \operatorname{clip}(\htmlClass{hl-ratio}{r_{i,t}},\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}})\,\htmlClass{hl-a}{A_i} \Big) - \beta\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})` },
  { tex: tex`\mathcal{J}_{\text{token}} = \frac{1}{\sum_i |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min(\cdots) \qquad \text{(DAPO token-level; 2026 consensus: } \beta \approx 0,\; \varepsilon_{\text{high}} > \varepsilon_{\text{low}}\text{)}` },
]);

function mathNotes() {
  const { factor, advantage } = unbiasedStd(INITIAL_STATE.k);
  const { advantages, rows } = evaluate(INITIAL_STATE);
  const tokens = rows.reduce((sum, row) => sum + row.tokens.length, 0);
  return [
    `Shapes: R, A ∈ ℝ^G (G = 8 on this page); r_{i,t} is one scalar per token, Σ|o_i| = ${tokens} tokens in the default group; the policy's logits are [|o_i| × vocab] per answer, never shown. Population std (divides by G): std = √(p(1−p)) for 0/1 rewards with pass rate p, so Σ|A_i| = 2G·√(p(1−p)), which peaks at p = 0.5 and is 0 at p ∈ {0, 1}.`,
    `Notes: (a) Trainers such as TRL and verl divide by G − 1, which makes the std ${formatRatio(factor)} larger and every advantage ${formatRatio(factor)} smaller (${signed2(advantage)} instead of ${signed2(advantages[0])}), and add a small ε instead of returning zeros. (b) Dr.GRPO also drops the 1/|oᵢ| length term (compare the agg toggle); the norm toggle shows only the std change. (c) π_old is the policy that produced the sample; it differs from π_θ because the trainer takes several steps per batch and rollouts may be asynchronous (frame 8).`,
    'Hover a highlighted term to outline its glyph on the stage: R is the reward column, A the advantage column and the chip fills, r the marker on the clip line.',
  ];
}

const TAKEAWAYS = Object.freeze([
  'RLVR scores an answer with a program, not a learned reward model, and the score is usually just 1 or 0 on the final answer; every token in the answer shares that score, and reasoning grows because it raises the odds of a right ending in a model that already sometimes gets there.',
  'GRPO\'s advantage is the reward standardized within its own group of samples: relative, zero-mean, and zero when all answers agree. That group mean is the baseline PPO needed a critic for; the price is G full generations per prompt.',
  'The 2026 recipe is GRPO plus patches: clip-higher (no entropy collapse: unlikely good tokens can keep growing), token-level loss (long wrong answers pay per token), dynamic sampling (no rollouts wasted on groups with no spread), and two covered in [[agentic-rl]]: little or no KL penalty, and importance-sampling corrections for the sampler-vs-trainer gap.',
]);

const FURTHER = Object.freeze([
  { title: 'UNIPO (Polo Club)', href: 'https://poloclub.github.io/unipo/', note: 'an interactive GRPO-family explorer with per-token coloring' },
  { title: 'The State of Reinforcement Learning for LLM Reasoning (Sebastian Raschka)', href: 'https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training', note: 'the 2025 recipes compared' },
  { title: 'The Illustrated DeepSeek-R1 (Jay Alammar)', href: 'https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1', note: 'the R1 pipeline in pictures' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: intuition().map(fill),
    intuitionNote: 'The policy, reward model, critic, KL term and the ratio r are defined in [[rlhf-dpo]]; this page uses them.',
    animation: {
      label: 'RLVR and GRPO: from a group of sampled answers to a push on every token, in nine steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The eight answers and the ratios are hand-picked stand-ins; a real policy samples them and a real trainer measures the ratios.',
    },
    toy: {
      title: 'Grade a group',
      intro: 'One prompt, eight seeded answers: slide how many are right, switch the two loss choices, and click a token to see every step behind its push.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: mathNotes() },
    facts: { framing: framing(), rows: factRows() },
    takeaways: TAKEAWAYS,
    links: { next: ['agentic-rl'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
