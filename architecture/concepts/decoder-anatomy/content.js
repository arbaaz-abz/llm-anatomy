// decoder-anatomy: the complete lesson (storyboard docs/storyboards/decoder-anatomy.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { BELOW, factRows, framing, fillText } from './facts.js';

const SLUG = 'decoder-anatomy';

const HOOK = 'Between "The cat sat down" and the next word, what does the model actually do, and where in that machine do {deepseek-v4-pro.total_params|count} parameters sit?';

const INTUITION = Object.freeze([
  'The input is cut into pieces: words or word fragments for text, small squares of pixels for an image, short slices for audio. Each piece becomes one vector of d_model numbers (8 on this page, {deepseek-v4-pro.d_model} in DeepSeek-V4-Pro and Kimi K3). A text token gets its vector by looking up a row of a table; an image patch gets there through a small vision encoder and a projector. From then on the model does not care where a vector came from.',
  'The vectors, one per position, form the residual stream. Think of it as a highway with on-ramps: each block reads the stream, computes something, and adds it back. Nothing is overwritten, so the embedding is still in there at the end, with two deltas per block piled on top. A block has two halves. First, attention lets each position read the other positions and pull in what it needs: this is the only place positions talk to each other. Second, an MLP (in 2026, usually a Mixture of Experts) transforms each position on its own: this is where most of the parameters live, because the MLP is two or three wide matrices (d_model × hidden, with hidden two to four times d_model) against attention\'s four matrices of about d_model × d_model (smaller when heads share keys and values), and a Mixture of Experts splits the MLP into dozens or hundreds of smaller MLPs and keeps them all. Each half is wrapped in "normalize, compute, add"; the normalize step keeps the numbers in a stable range as the stack gets deep.',
  'A model is that block repeated N times ({deepseek-v4-pro.layers} in DeepSeek-V4-Pro, {kimi-k3.layers} in Kimi K3), each with its own weights: talk, think, talk, think. After the last block, the last position\'s vector is normalized and multiplied by one more matrix, the unembedding, giving one score per vocabulary entry. Softmax turns the scores into probabilities, a sampler picks a token, the token is appended, and the whole thing runs again for the next one. With a causal mask, earlier positions\' keys and values do not change, so most models keep them in a KV cache and read them rather than recompute them.',
  'The price of this picture, which the rest of the course is about: each generated token multiplies by roughly all the active parameters in one forward pass (so active parameters set compute), carries the whole residual stream through N blocks one after another (depth means N steps in sequence, which sets a floor on latency), and reads the KV cache of every earlier position (so context length sets memory traffic). That is why 2026 models shrink active parameters with experts and shrink the cache with new attention designs.',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-x}{x^{(0)}_i} = E[t_i] \quad (\text{text}),\qquad \htmlClass{hl-x}{x^{(0)}_i} = \operatorname{Proj}\big(\operatorname{ViT}(\text{patch}_i)\big) \quad (\text{image}), \qquad x^{(0)} \in \mathbb{R}^{n \times d_{\text{model}}}` },
  { tex: tex`\htmlClass{hl-a}{a^{(\ell)}} = \operatorname{Attn}\big(\operatorname{RMSNorm}(x^{(\ell)})\big),\quad \tilde{x}^{(\ell)} = x^{(\ell)} + \htmlClass{hl-a}{a^{(\ell)}};\qquad \htmlClass{hl-m}{m^{(\ell)}} = \operatorname{MLP}\big(\operatorname{RMSNorm}(\tilde{x}^{(\ell)})\big),\quad x^{(\ell+1)} = \tilde{x}^{(\ell)} + \htmlClass{hl-m}{m^{(\ell)}}` },
  { tex: tex`\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_j x_j^2}},\qquad \text{worked: } \operatorname{rms}(x_{\text{sat}}) = \sqrt{2.75/8} = 0.586` },
  { tex: tex`\operatorname{MLP}(h) = \big(\sigma(h W_{\text{gate}}) \odot h W_{\text{in}}\big) W_{\text{out}},\qquad \operatorname{MoE}(h) = \sum_{e \in \operatorname{top\text{-}k}(h W_{\text{router}})} g_e\, \operatorname{MLP}_e(h)` },
  { tex: tex`\htmlClass{hl-z}{z} = \operatorname{RMSNorm}\big(x^{(N)}_n\big)\, W_U \in \mathbb{R}^{|V|},\qquad \htmlClass{hl-p}{p_v} = \frac{e^{z_v}}{\sum_{u} e^{z_u}},\qquad \text{worked: } p_{\text{on}} = \frac{e^{2}}{18.93} = 0.390` },
  { tex: tex`\text{params per block} = \underbrace{d\,(n_H d_h) + 2d\,(n_{KV} d_h) + (n_H d_h)\,d}_{\text{attention}} + \underbrace{3\, d\, h}_{\text{SwiGLU MLP}} \quad\text{or}\quad \underbrace{N_e \cdot 3\, d\, h_e + d\, N_e}_{\text{experts + router}}` },
  { tex: tex`\text{total} = |V|\,d \;(+\,|V|\,d \text{ if untied}) + N \cdot \text{params per block} + \text{norms},\qquad \text{active} = \text{total} - (N_e - k)\, N\, 3\, d\, h_e - |V|\, d\ \text{(the lookup table, untied models only)}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes (toy in parentheses): x [n × d_model] (4 × 8); E [|V| × d_model] (16 × 8) and W_U [d_model × |V|] (8 × 16); W_in, W_gate [d_model × h] (8 × 16); W_out [h × d_model] (16 × 8); z, p [|V|] (16). Each toy expert has h_e = 8, so two active experts (2 · 192) equal the dense MLP (384).',
  'Simplifications: γ = 1 everywhere; the toy\'s MLP hidden is 2·d (real SwiGLU models use about 8/3·d and GPT-3 used 4·d), so the toy\'s MLP : attention ratio is 1.5 : 1 against GPT-3\'s 2 : 1. One "active" convention throughout: every multiplied block parameter plus the unembedding, the lookup table left out unless it is the unembedding. Hover a highlighted term to outline its glyph on the stage.',
]);

const TAKEAWAYS = Object.freeze([
  'One forward pass: pieces of input become one vector per position; N blocks each add an attention result (positions talk) and an MLP/MoE result (each position thinks) onto a single residual stream; a final norm and the unembedding give one score per vocabulary entry; a sampler picks, the token is appended, and it runs again with earlier positions\' K and V read from the cache.',
  'Parameters live mostly in the MLP: two thirds of a dense model, over 90% of a Mixture of Experts. Attention is a third at most; the embedding table only matters in small models.',
  'A model card is this picture in numbers: "layers" counts blocks, d_model is the row width, "total / active" is all experts versus the few each token uses, and the KV cache holds keys and values for every attention block and earlier position (newer designs store far less: [[kv-cache]]).',
]);

const FURTHER = Object.freeze([
  { title: 'LLM Visualization (Brendan Bycroft)', href: 'https://bbycroft.net/llm', note: 'a 3D walk through every tensor of a tiny GPT, the best way to see the shapes' },
  { title: 'Transformer Explainer', href: 'https://poloclub.github.io/transformer-explainer/', note: 'type your own sentence into GPT-2 in the browser' },
  { title: 'The Illustrated Transformer (Jay Alammar)', href: 'https://jalammar.github.io/illustrated-transformer/', note: 'the canonical static diagrams' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(HOOK),
    intuition: INTUITION.map(fill),
    intuitionNote: 'How text is cut into tokens is its own topic: see [[pretraining]].',
    animation: {
      label: 'The whole model, end to end: a decoder\'s forward pass in nine steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The eight-number vectors and the 16 scores are hand-picked stand-ins; the parameter counts in the toy below are exact.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Where do the parameters live?',
      intro: 'It opens on the toy model from the animation, so the counts match its 656-per-block readout; one chip jumps to a real model.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(data), rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['decoder-recap', 'attention'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
