// sampling: the complete lesson (storyboard docs/storyboards/sampling.md §3, §5, §7–§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FRAMING, FACT_ROWS, BELOW, tokensPerStepText, mtpTex } from './facts.js';
import { workedTex } from './format.js';

export { CAPTIONS };

const SLUG = 'sampling';

const HOOK = 'The model ends every step with {kimi-k3.vocab_size|int} scores in Kimi K3, one per possible token. Who decides which one becomes the next word, and why does asking twice give two different answers?';

const INTUITION = Object.freeze([
  'The last step of [[decoder-anatomy]] gave one score per vocabulary word, the logits, and softmax turned them into probabilities. That is where the model\'s job ends. Choosing a token is a separate step with no learned parameters. The simplest choice is to always take the most likely token, greedy decoding. It is predictable, and it tends to fall into loops. Either way, the chosen token is appended to the input and the model runs again for the next one. Generation stops when the model picks a special end-of-sequence token, which it learned to emit where a reply ends, or when the reply reaches a length limit set by whoever runs the model.',
  'Most chat systems draw instead: they pick a token at random in proportion to its probability, so "on" comes out 39% of the time and "." 24% (over many draws). That is why the same prompt gives different replies. Three knobs shape the draw. Temperature divides the scores before softmax: below 1 it sharpens toward the favorite, above 1 it flattens toward the long tail. Top-k throws away everything but the k most likely tokens. Top-p keeps the smallest set of tokens whose probabilities add up to p, so it keeps few tokens when the model is confident and many when it is not. After any cut, the survivors are rescaled to add up to 1.',
  'One more 2026 twist: some models predict more than one token per step. A small extra head, trained with the model, guesses the token after next. On the following step the model checks that guess while computing anyway, and if it agrees, two tokens come out of one step. The cost is the extra head and some wasted work when the guess is wrong; [[speculative-decoding]] turns this into a full speed-up method.',
]);

const STAND_IN = 'The 16 scores are [[decoder-anatomy]]\'s hand-picked stand-ins; the random number in frame 3 is picked by hand, and the toy\'s draws are seeded.';
const TOY_INTRO = 'Same 16 hand-picked scores as the animation; the toy\'s draws are seeded.';

const tex = String.raw;
const mathBlocks = (data) => [
  { tex: tex`\htmlClass{hl-p}{p_v} = \frac{e^{\htmlClass{hl-z}{z_v} / \htmlClass{hl-t}{T}}}{\sum_u e^{z_u / T}},\qquad ${workedTex(0.5)}`, note: 'Worked with the T = 0.5 terms of the toy\'s "Check my work".' },
  { tex: tex`\text{top-}k:\ \mathcal{K} = \text{the } k \text{ largest } p_v,\qquad \text{top-}p:\ \mathcal{K} = \text{smallest prefix of the sorted } p \text{ with } \textstyle\sum_{v \in \mathcal{K}} p_v \ge p,\qquad \tilde p_v = \frac{p_v}{\sum_{u \in \mathcal{K}} p_u}\ (v \in \mathcal{K})` },
  { tex: tex`\text{draw: } v = \min\Big\{\, v_j : \textstyle\sum_{i \le j} \tilde p_{v_i} > u \Big\},\quad u \sim \mathcal{U}[0, 1)\qquad \text{worked: } u = 0.55 \in [0.390,\ 0.627) \Rightarrow \text{"."}` },
  { tex: mtpTex(data) },
];

const MATH_NOTES = Object.freeze([
  'Shapes: z, p [|V|] (16 here; {kimi-k3.vocab_size|int} in Kimi K3). The MTP head reads the last block\'s output for position t and predicts position t + 2. Greedy is the limit T → 0. Hover a highlighted term to outline its glyph on the stage.',
]);

const TAKEAWAYS = Object.freeze([
  'The model ends with logits, one score per vocabulary entry; a parameter-free sampler turns them into a token: greedy takes the top, sampling draws by probability.',
  'Temperature rescales the logits (sharper below 1, flatter above); top-k keeps a fixed number of tokens; top-p keeps as many as it takes to reach p, so it adapts to how sure the model is. A seed makes the draws repeatable.',
  'Some 2026 models carry a multi-token-prediction head that drafts the token after next; when the next step agrees, one step yields two tokens ({sv:deepseek-v3-mtp.acceptance_pct}% of the time in DeepSeek-V3). The full method is [[speculative-decoding]].',
]);

const FURTHER = Object.freeze([
  { title: 'Transformer Explainer', href: 'https://poloclub.github.io/transformer-explainer/', note: 'its temperature slider runs on real GPT-2' },
  { title: 'Gloeckle et al., Better & Faster Large Language Models via Multi-token Prediction', href: 'https://arxiv.org/abs/2404.19737', note: 'the paper behind the MTP head' },
  { title: 'DeepSeek-AI, DeepSeek-V3 Technical Report', href: 'https://arxiv.org/pdf/2412.19437', note: 'see its multi-token prediction section' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data).replace('{tokensPerStep}', tokensPerStepText(data));
  return {
    slug: SLUG,
    hook: fill(HOOK),
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Picking the next token: greedy, a random draw, temperature, top-k, top-p, seeds, and a second token per step',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: STAND_IN,
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: { title: 'Shape the draw.', intro: TOY_INTRO, mount },
    math: { blocks: mathBlocks(data), notes: MATH_NOTES.map(fill) },
    facts: { framing: FRAMING, rows: FACT_ROWS.map((row) => ({ ...row })) },
    takeaways: TAKEAWAYS.map(fill),
    links: { next: ['speculative-decoding'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
