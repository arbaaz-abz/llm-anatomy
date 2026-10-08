// pretraining storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps; frames.js labels each
// stage with them. Kept apart so the two never import each other. Frame 1's and 10's numbers are pinned to the data by the page test.
export const CAPTIONS = Object.freeze([
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
]);
