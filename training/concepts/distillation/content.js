// distillation: the complete lesson (storyboard docs/storyboards/distillation.md §3, §5–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FORWARD_KL, FORWARD_KL_TERMS, DEFAULT_STUDENT, DEFAULT_TEACHER } from './numbers.js';
import { fixed } from './format.js';
import { BELOW, FRAMING, factRows } from './facts.js';

const SLUG = 'distillation';

const HOOK = 'A lab has trained a math expert, a coding agent and a chat model with RL. How does it ship one model that is as good as each of them at its own job?';

const INTUITION = Object.freeze([
  'Distillation trains a student to match a teacher. The oldest way is to let the teacher write answers and fine-tune the student on that text, exactly like SFT: for each position, only the one token the teacher wrote counts. A richer way, logit distillation, uses the teacher\'s whole probability row at each position: not just "56" but "56 at 0.90, 54 at 0.05". Both have the same blind spot. The student only ever practises on text the teacher wrote, so the first time it samples its own mistake it is somewhere it has never been, and nothing taught it what to do next.',
  'On-policy distillation flips who writes. The student samples its own answer, mistakes included, and the teacher scores every token the student wrote: the reward is how much likelier the teacher finds that token than the student did. A good token earns a positive reward, the student\'s own 54 a strongly negative one. That is an RL loop, with two differences that make it efficient: the reward is dense (one per token, not one per answer, so the blame lands on 54 rather than on all of 7 × 8 = 54), and it needs no checker, so it works for any task a teacher can do.',
  'That is why it became the 2026 way to merge specialists. Labs train separate experts with RL (math and code, agents, chat, sometimes at several effort levels), then distill all of them into one student: each prompt is graded by the teacher for its domain, or by a weighted mix. DeepSeek-V4 uses a whole panel of teachers, Kimi K3 uses {kimi-k3.rl_experts}, and MiMo-V2-Flash reports that the merged model keeps each teacher\'s peak. The cost: every teacher must be run (or cached) during training, and the distillation reward stops pushing once the student matches its teacher; to go past the teacher, MiMo-V2-Flash adds an outcome advantage (see the math panel).',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{trace (SFT) distillation:}\quad \mathcal{L} = -\ln \pi_S(y^{T}_t \mid y^{T}_{<t}),\quad y^{T} \sim \htmlClass{hl-t}{\pi_T}` },
  { tex: tex`\text{logit distillation:}\quad \mathcal{L} = \mathbb{D}_{\text{KL}}\big(\htmlClass{hl-t}{\pi_T}(\cdot \mid y_{<t}) \,\|\, \htmlClass{hl-s}{\pi_S}(\cdot \mid y_{<t})\big) \quad\text{on fixed text}` },
  { tex: tex`\text{on-policy:}\quad y \sim \htmlClass{hl-s}{\pi_S},\qquad \htmlClass{hl-r}{r_t} = \operatorname{sg}\big(\ln \htmlClass{hl-t}{\pi_T}(y_t \mid y_{<t}) - \ln \htmlClass{hl-s}{\pi_S}(y_t \mid y_{<t})\big),\qquad \mathbb{E}_{y_t \sim \pi_S}[r_t] = -\mathbb{D}_{\text{KL}}(\pi_S \,\|\, \pi_T)` },
  { tex: tex`\text{DeepSeek-V4:}\ \mathcal{L} = \sum_i w_i\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{E_i})\ (\text{full vocabulary}) \qquad \text{Kimi K3:}\ r_t = \operatorname{clip}\big(\operatorname{sg}\ln \tfrac{\pi_T}{\pi_\theta}, \pm R_{\max}\big) \qquad \text{MiMo-V2-Flash:}\ A = \operatorname{sg}\ln \tfrac{\pi_T}{\pi_\theta} + \alpha A_{\text{ORM}}` },
]);

// Note (d): the four terms of KL(teacher ‖ student) for the default pair, computed from the same numbers the stage prints.
const klTerms = () => DEFAULT_TEACHER.map((t, i) => `${t.toFixed(2)} ln ${(t / DEFAULT_STUDENT[i]).toFixed(2)} = ${fixed(FORWARD_KL_TERMS[i], 3)}`).join(' · ');

const MATH_NOTES = Object.freeze([
  'Shapes: π_T and π_S are [|V|] rows at each position; the page shows a [4] slice (four candidates), so its KLs are over those four entries only. sg is the stop-gradient: the reward is a number, not something to differentiate through. Color links: the teacher row, the student row and sampled chip, and the reward cells.',
  '(a) Reverse KL (student ‖ teacher) is "mode-seeking": it punishes the student for putting probability where the teacher puts little; forward KL is "mean-covering". (b) DeepSeek chose full-vocabulary KL because sampled-token estimates had high variance, and caches each teacher\'s last-layer hidden states to rebuild logits on the fly. (c) Kimi K3 found top-k logit variants gave no gain.',
  `(d) Frame 3's four terms of KL(teacher ‖ student): ${klTerms()}; sum ${fixed(FORWARD_KL, 3)}.`,
]);

const TAKEAWAYS = Object.freeze([
  'A student can learn from a teacher\'s text (one token per position), from its full probabilities on fixed text, or on-policy: the student writes and the teacher grades every token it wrote.',
  'On-policy distillation is RL with a dense reward (the log of the teacher\'s probability over the student\'s): it trains the student on its own mistakes and puts the blame on the token that caused them.',
  '2026 labs train RL specialists, then merge them with multi-teacher on-policy distillation (DeepSeek-V4, Kimi K3, MiMo-V2-Flash, GLM-5 across its own stages); the cost is running every teacher, and the reward stops pushing once the student matches its teacher.',
]);

const FURTHER = Object.freeze([
  { title: 'On-Policy Distillation (Thinking Machines)', href: 'https://thinkingmachines.ai/blog/on-policy-distillation/', note: 'the blog that popularized it, with the compute comparison against RL' },
  { title: 'RLHF Book (Nathan Lambert)', href: 'https://rlhfbook.com/', note: 'the chapter on on-policy distillation' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    intuitionNote: 'The specialists come from [[agentic-rl]]; the whole map of the stages is [[training-pipeline]].',
    animation: {
      label: 'Distillation in nine steps: learning from the teacher\'s text, its probabilities, and its grades on the student\'s own samples',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The teacher and student probabilities are hand-picked stand-ins; the losses and rewards computed from them are exact. Only four candidate tokens are shown, so every KL here sums over those four.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Grade the student',
      intro: 'The 7 × 8 = position with a teacher row and a student row. Switch how the student learns, which token it sampled, and which teacher grades it.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows() },
    takeaways: TAKEAWAYS,
    links: { next: [], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
