// agentic-rl: the complete lesson (storyboard docs/storyboards/agentic-rl.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { renderFor } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { factRows, framing } from './facts.js';
import { CONTENT_TEXT } from './content-text.js';

const SLUG = 'agentic-rl';
const tex = String.raw;

const MATH_BLOCKS = Object.freeze([
  {
    tex: tex`\htmlClass{hl-rho}{\rho_{i,t}} = \frac{\pi^{\text{train}}_{\text{old}}(o_{i,t} \mid \cdot)}{\pi^{\text{engine}}_{\text{old}}(o_{i,t} \mid \cdot)} \qquad w_{i,t} = \begin{cases} 1 & \text{ignore} \\ \rho_{i,t} & \text{full IS} \\ \min(\rho_{i,t}, C) & \text{truncated IS (TIS)} \\ \mathbb{1}\big[\tfrac{1}{\beta} \le \rho_{i,t} \le \beta\big] & \text{IcePop } (\beta = 2) \end{cases}`,
  },
  {
    tex: tex`\mathcal{J} = \frac{1}{\sum_i |o_i|_{\text{policy}}} \sum_{i=1}^{G} \sum_{t \in \text{policy tokens}} w_{i,t}\, \min\!\big( r_{i,t} \htmlClass{hl-a}{A_i},\ \operatorname{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\, \htmlClass{hl-a}{A_i} \big) \;-\; \underbrace{\beta_{\text{KL}}}_{\approx\, 0}\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})`,
  },
  {
    tex: tex`\text{utilization} = \frac{\sum_i \min(\ell_i, T)}{G \cdot T},\qquad T = \ell_{(\lceil \lambda G \rceil)} \;(\text{the } \lceil \lambda G\rceil\text{-th shortest episode})`,
  },
]);

export { CAPTIONS };

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return Object.freeze({
    slug: SLUG,
    hook: CONTENT_TEXT.hook,
    intuition: CONTENT_TEXT.intuition.map(fill),
    intuitionNote: CONTENT_TEXT.intuitionNote,
    animation: {
      label: 'Agentic RL: an episode in a sandbox, asynchronous rollouts, and the importance-sampling correction, in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: renderFor(data),
      standIn: CONTENT_TEXT.standIn,
    },
    toy: { title: 'Fix the mismatch.', intro: CONTENT_TEXT.toyIntro, mount },
    math: { blocks: MATH_BLOCKS, notes: CONTENT_TEXT.mathNotes },
    facts: { framing: framing(), rows: factRows() },
    takeaways: CONTENT_TEXT.takeaways,
    links: { next: ['distillation'], further: CONTENT_TEXT.further },
  });
}

// For spec validation only; the page mounts lessonFor(ctx.data).
export const LESSON = lessonFor(null);
