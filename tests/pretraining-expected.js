// pretraining storyboard §5 captions, §6 "Check my work" and §6 try-this text, verbatim (numbers as the data fills them).
// Imported by the page test and the e2e spec; page code never imports from tests/.
export const CAPTIONS = [
  'A tokenizer cuts text into pieces from a fixed vocabulary: whole common words, fragments of rare ones. Kimi K3\'s vocabulary has 163,840 pieces.',
  'At each position the model gives a probability to every possible next token. Here, after "down", the true next token "on" got 0.390.',
  'The grade is the cross-entropy loss: minus the log of the probability given to the true token. Unsure costs a little; confident and wrong costs a lot.',
  'One pass grades every position at once: the causal mask lets each position see only the past. Eight tokens give seven graded predictions.',
  'Training lowers the mean loss over all positions. Its exponential, perplexity, is how many equally likely tokens the model is effectively choosing between: about 2.9 here.',
  'Raw crawl is mostly not worth training on. Rules remove the obvious junk, and a quality classifier, itself a small model, scores the rest.',
  'Deduplication removes exact and near-copies, so the model does not spend its budget reading the same page many times.',
  'The surviving data is mixed by domain, with code and math upsampled. Kimi K3 set its per-domain rates with small-model experiments.',
  'Some knowledge and math text is rephrased by another model, checked against its source, and added back. Mass-produced templated text is filtered out instead.',
  '2026 open frontier models pretrain on 25 to 33 trillion tokens. It is the biggest stage by far, and how to split that budget is the next page\'s question.',
];

export const CHECK_WORK = [
  'selected: mat   −ln 0.45 = 0.7985',
  'sum   = 2.3026 + 1.3863 + 1.2040 + 0.9410 + 0.5108 + 0.7985 + 0.2231 = 7.3663',
  'mean  = 7.3663 ÷ 7 = 1.0523 → 1.052',
  'perplexity = e^1.0523 = 2.864 → 2.86',
].join('\n');

// The toy's default readouts (storyboard §6 and §11).
export const DEFAULT_LOSSES = ['2.303', '1.386', '1.204', '0.941', '0.511', '0.799', '0.223'];

// Storyboard §6 "Try this": { prompt, insight, rest }; `x` marks code text. Kimi K3's vocabulary is filled from the data.
export const TRY_THIS = [
  {
    prompt: 'Predict first: select `mat` (0.45, loss 0.799). Which moves the mean more: making it certain, or making it a confident miss? Slide to 1.00: mean 1.052 → 0.938 (down 0.114). Slide to 0.01: its loss is 4.605 and the mean jumps to 1.596; perplexity 2.86 → 4.93.',
    insight: 'the log punishes confident mistakes far more than it rewards certainty.',
    rest: ' One badly wrong token moves the average almost five times as much (+0.544) as one perfect one does (−0.114).',
  },
  {
    prompt: 'Select `on`, the model\'s top guess at 0.390 (against 0.237 for the runner-up `.`). Its loss is 0.941, not 0.',
    insight: 'there is no credit for being top-1, only for probability.',
    rest: ' The model keeps learning even on tokens it already ranks first, by making them more likely.',
  },
  {
    prompt: 'Press `uniform guess over 16`: every loss becomes 2.773, the mean is 2.773 and perplexity is exactly 16. Now read the reference mark: a model that knows nothing about Kimi K3\'s vocabulary of 163,840 pieces would start near 12.01.',
    insight: 'perplexity is the effective number of choices.',
    rest: ' Training a real model is the long walk from about ln(vocabulary size) down toward the loss of the text itself.',
  },
];
