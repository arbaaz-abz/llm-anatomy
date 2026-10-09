// prefill-decode storyboard §5 captions, verbatim (one per frame); content.js puts them on the steps, frames.js labels each stage.
// Caption 4 names "A GPU for LLM people" where the storyboard prints the slug: captions are plain text (README lesson 33).
export const CAPTIONS = Object.freeze([
  'The model\'s 70 billion weights sit in GPU memory as 70 GB. Every token that passes through does about two operations per weight.',
  'One decode token: the GPU must read all 70 GB to do its math. Reading takes 14.6 ms, the math 70.7 µs, so the step waits on memory.',
  'Prefill, 1,000 tokens: the same 70 GB is read once and reused for every token. Now the math takes 70.7 ms and the reading only 18.1.',
  'That ratio, operations per byte read, is the arithmetic intensity from A GPU for LLM people. One decode token scores 2; a 1,000-token prefill scores about 1,600.',
  'On the H200\'s roofline the ridge sits at 412 operations per byte. Decode lands far left, waiting on memory; prefill lands on the flat part, waiting on math.',
  'At 217 prompt tokens, math and reading both take 15.4 ms, barely longer than one decode step. A prompt reuses each weight hundreds of times; an answer uses it once.',
  'Batch eight users and they share one read of the weights. The step grows from 14.7 to 15.7 ms while the GPU makes 7.49 times more tokens per second.',
  'Each user\'s cached keys and values are read every step too. At 64 users that adds 9 ms, so each user\'s stream slows while the GPU\'s total keeps rising.',
  'At 105 users the GPU\'s memory is full, and the batch is still far left of the ridge. At this context, KV memory, not arithmetic, caps the batch.',
  'Real servers choose a point on this curve. Measured on DeepSeek-V4-Pro and GB300, asking for twice the speed per user roughly halved the tokens per GPU.',
]);
