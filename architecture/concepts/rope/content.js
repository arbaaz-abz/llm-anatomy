// rope: the complete lesson (storyboard docs/storyboards/rope.md §3, §5, §7-§10). Pure, no DOM at import time.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { belowNotes, factRows, framing } from './facts.js';

export { CAPTIONS };

const SLUG = 'rope';

const HOOK = 'In [[attention]], "sat" scored "cat" 3.0 no matter where "cat" stood. How does a model learn that "cat" came just before "sat", and why can it then not simply read ten times further than it was trained on?';

const INTUITION = Object.freeze([
  'The five steps on [[attention]] never use positions. Swap two tokens and every score stays the same; only the causal mask knows about order, and it only says "earlier or not". GPT-3 fixed this by learning one extra vector per position and adding it to each token\'s embedding: a table with {gpt-3.context_length|int} rows, and nothing for the next position ([[decoder-recap]]).',
  'RoPE puts position into the query and the key instead. Cut each vector into pairs of numbers and draw each pair as a clock hand. Before the dot product, turn every hand by the token\'s position times a fixed speed: the first pair turns fast, the last pair very slowly. A dot product of two hands only cares about the angle between them, so after turning, the score depends on how far apart the two tokens are, not on where they sit. That is exactly the information language needs: "the word just before me" means the same thing on page 1 and page 300.',
  'The slowest hand sets how far the model can tell positions apart, and a constant called the base sets how slow it is: 2026 models raise it from {deepseek-v4-pro.rope_theta|int} to as much as {qwen3.8.rope_theta|int}. Training fixes which angles the model has seen. Feed it a longer text and the slow hands swing into angles it has never seen, so the scores stop meaning anything. Context extension squeezes the positions back into the seen range, and YaRN squeezes only the slow hands, which keeps the fast ones sharp. The price of squeezing is resolution: neighbors end up closer together on the dial.',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  {
    tex: tex`\theta_i = \htmlClass{hl-base}{\beta}^{-2i/d},\quad i = 0,\dots,\tfrac{d}{2}-1 \qquad R(a) = \begin{pmatrix}\cos a & -\sin a\\ \sin a & \cos a\end{pmatrix}`,
    note: 'Shapes: q, k [d_head] (4), d_head / 2 pairs (2), one speed θ per pair; β is the base (100 here).',
  },
  { tex: tex`\tilde q^{(i)}_m = R(m\,\theta_i)\, q^{(i)},\qquad \tilde k^{(i)}_n = R(n\,\theta_i)\, k^{(i)} \qquad (q^{(i)} = \text{pair } i)` },
  {
    tex: tex`\htmlClass{hl-score}{\tilde q_m \cdot \tilde k_n} = \sum_i q^{(i)\top} R\big((n-m)\,\theta_i\big)\, k^{(i)} \quad\text{depends only on } \htmlClass{hl-off}{n - m}`,
    note: 'Turning both hands by the same angle leaves the angle between them alone, so shifting both words by the same amount changes nothing.',
  },
  {
    tex: tex`\text{worked } (n - m = -1): \underbrace{3\cos(-1)}_{\text{pair 1}} + \underbrace{0.25\,\sin(-0.1)}_{\text{pair 2}} = 1.621 - 0.025 = 1.596`,
    note: 'Pair 2 uses the 2D identity (a, b)ᵀR(φ)(c, d) = (ac + bd) cos φ + (bc − ad) sin φ with q pair (0.5, 0), k pair (0, −0.5) and φ = (n − m) θ₂ = −0.1 (the key is one token before the query): ac + bd = 0 and bc − ad = 0.25, so 0.25 · sin(−0.1) = −0.025. Pair 1 is unchanged: 3 cos(−1) = 3 cos(1) = 1.621.',
  },
  {
    tex: tex`\text{wavelength}_i = \frac{2\pi}{\theta_i};\qquad \text{PI: } \theta_i \to \theta_i / s;\qquad \text{YaRN-style: } \theta_i \to \begin{cases}\theta_i & 2\pi/\theta_i \le L_{\text{train}}\\ \theta_i / s & \text{otherwise}\end{cases}`,
    note: 'Real YaRN ramps smoothly between "keep" and "squeeze" by each pair\'s turns per trained length, and also rescales attention by a temperature. NTK-aware scaling raises the base instead of slowing positions. Pairing dimension i with i + d/2, as many codebases do, is the same rotation on reordered dimensions.',
  },
]);

const MATH_NOTES = Object.freeze(['Hover a highlighted term to outline its glyph on the stage.']);

const TAKEAWAYS = Object.freeze([
  'RoPE turns each pair of query and key numbers by position × the pair\'s speed, inside every attention layer; nothing is added to the token\'s vector.',
  'Because a dot product only sees the angle between two hands, the score depends only on how far apart two tokens are. Fast pairs see nearby order, slow pairs far order, and the base sets how slow the slowest is.',
  'Past the trained length the slow pairs hit unseen angles. Position interpolation squeezes every pair and blurs neighbors; YaRN squeezes only the slow pairs. Some heads rotate only part of each vector, and some layers none.',
]);

const FURTHER = Object.freeze([
  { title: 'RoFormer (Su et al.)', href: 'https://arxiv.org/abs/2104.09864', note: 'the RoPE paper' },
  { title: 'YaRN (Peng et al.)', href: 'https://arxiv.org/abs/2309.00071', note: 'stretching a trained context with a smooth per-pair ramp' },
  { title: 'Position Interpolation (Chen et al.)', href: 'https://arxiv.org/abs/2306.15595', note: 'the simplest stretch: slow every pair by the same factor' },
]);

const STAND_IN = 'Two pairs and base 100 are toy choices so both hands visibly move; real heads have 32 to 256 pairs and a base of 10,000 or more. Vectors are the attention page\'s numbers; cos and sin need a calculator, like exp did there.';

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  const below = belowNotes(data);
  return Object.freeze({
    slug: SLUG,
    hook: fill(HOOK),
    intuition: INTUITION.map(fill),
    intuitionNote: 'Every angle on this page is in radians: a full turn is about 6.28.',
    animation: {
      label: 'RoPE: turning the query and key by position, then stretching the context, in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: STAND_IN,
      belowFor: (index) => (below[index] ?? []).map(fill),
    },
    toy: { title: 'Turn the hands.', intro: STAND_IN, mount },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['long-context-attention', 'midtraining'], further: FURTHER },
  });
}

export const LESSON = lessonFor(null);
