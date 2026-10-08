// pretraining: the complete lesson (storyboard docs/storyboards/pretraining.md §3, §5, §7–§10). Pure, no DOM at import.
// lessonFor(data) fills the dated text from data/models.json; LESSON is the same lesson with no data (validation, tests).
import { fillText } from '@shared/claims.js';
import { renderFor } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { BELOW, FRAMING, factRows, stageText, trillionRange } from './facts.js';

const SLUG = 'pretraining';
const RANGE = '%RANGE%'; // the 2026 budget range in trillions, computed from the data (facts.js)

const HOOK = 'A pretrained model is only ever asked to guess the next token. Why does that, repeated over 30 trillion tokens, produce something that knows chemistry, grammar and Python?';

export const INTUITION = Object.freeze([
  'Pretraining shows the model text and, at every position, asks for a probability for every possible next token. The grade is simple: take the probability it gave the token that actually came next, and score minus its logarithm. A confident right guess costs almost nothing, an unsure one costs a little, and a confident wrong one costs a lot. The average of that score over all positions is the cross-entropy loss, and training nudges every weight to lower it.',
  'Nothing in that grade mentions facts or grammar. But text is full of them, and the cheapest way to assign high probability to the next token of a chemistry paper, a legal contract or a Python file is to have absorbed how chemistry, law and Python work. Across tens of trillions of tokens there is no shortcut left: lowering the loss further requires knowing more. That is why the objective is still plain next-token cross-entropy in 2026, with one common addition: several labs also predict a second token ahead (multi-token prediction) as an extra loss.',
  `Two things make this affordable. Every position in a sequence is graded in the same forward pass, so one 8-token sentence is 7 training examples. And the text is chosen with great care: raw crawl is filtered by rules and by quality classifiers, near-duplicates are removed, domains like code and math are upsampled, and some knowledge and math text is rephrased by other models. The open frontier MoEs reported in 2026 read ${RANGE} trillion tokens this way (Kimi K3 did not say; smaller open models read far less, Olmo 3 about {olmo-3.pretrain_tokens|count}, reported). The cost is that this stage is by far the largest in tokens and usually in compute; how to split that compute between model size and tokens is the next page, [[scaling-laws]].`,
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\mathcal{L} = -\frac{1}{T-1}\sum_{t=2}^{T} \ln \htmlClass{hl-p}{p_\theta(x_t \mid x_{<t})}, \qquad \text{PPL} = e^{\mathcal{L}}` },
  { tex: tex`\htmlClass{hl-p}{p_\theta(\cdot \mid x_{<t})} = \operatorname{softmax}(z_t),\quad z_t = W_U\, \operatorname{norm}(h_t) \in \mathbb{R}^{|V|} \qquad \text{uniform: } p = \tfrac{1}{|V|} \Rightarrow \mathcal{L} = \ln |V|` },
  { tex: tex`\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda_{\text{MTP}}\, \mathcal{L}_{\text{MTP}} \qquad (\text{DeepSeek-V4: depth 1},\ \lambda = 0.3 \to 0.1 \text{ when LR decay starts})` },
]);

export const MATH_NOTES = Object.freeze([
  'Shapes (toy in parentheses): x [T] token ids (8); h_t [d_model]; W_U [d_model × |V|] (|V| = 16 here, {kimi-k3.vocab_size} in Kimi K3); z_t, p [|V|]; the loss strip has T − 1 = 7 entries. The causal mask lets all T − 1 predictions come from one forward pass.',
  '(a) The mean is over the T − 1 positions that have a target: every position but the first, so 7 here. (b) DeepSeek-V4 packs documents into sequences and masks attention between them, so the loss never asks a document to predict its neighbor.',
  'Hover a highlighted term to outline its glyph on the stage: p is the probability row (frames 2–3) and the loss strip (frames 4–5).',
]);

const TAKEAWAYS = Object.freeze([
  'Pretraining grades one thing: minus the log of the probability given to the true next token, averaged over every position. Confident mistakes cost far more than certainty earns, and perplexity (its exponential) is the effective number of choices.',
  'Every position is graded in the same pass, and the text is curated hard: filters, a quality classifier, deduplication, an upsampled code-and-math mixture, and rephrased knowledge.',
  `The open frontier MoEs reported in 2026 read ${RANGE} trillion tokens; it is the largest stage, and the next page asks how to split that compute between model size and tokens.`,
]);

const FURTHER = Object.freeze([
  { title: 'The Smol Training Playbook (Hugging Face)', href: 'https://huggingfacetb-smol-training-playbook.hf.space/' },
  { title: 'nanochat (Andrej Karpathy)', href: 'https://github.com/karpathy/nanochat', note: 'a runnable tokenizer → pretrain → midtrain → SFT pipeline' },
  { title: 'Tiktokenizer', href: 'https://tiktokenizer.vercel.app/', note: 'type text and see the tokens' },
]);

export function lessonFor(data) {
  const range = trillionRange(data);
  const fill = (text) => fillText(text, data).replace(RANGE, range);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Pretraining in ten steps: the objective, the data pipeline and the token budget',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: renderFor(stageText(data)),
      standIn: 'Position 4\'s probabilities are the exact softmax worked in [[decoder-anatomy]]; the other six probabilities are hand-picked stand-ins. The losses are exact for those numbers.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Grade a sentence.',
      intro: 'The animation\'s sentence with its seven probabilities. Select a target chip, then set the probability the model gave it: every loss, the mean and the perplexity update live.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES.map(fill) },
    facts: { framing: FRAMING, rows: factRows(data) },
    takeaways: TAKEAWAYS.map(fill),
    links: { next: ['scaling-laws', 'midtraining', 'sft'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
