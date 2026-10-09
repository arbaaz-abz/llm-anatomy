// prefill-decode storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test
// and the e2e spec. Caption 4 names "A GPU for LLM people" where the storyboard prints the slug `gpu-primer`: captions are
// plain text and README lesson 33 names a lesson by its title on the stage.

export const CAPTIONS = [
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
];

// Default state: decode, H200, FP8, 8 users, 2,048 tokens of context (builder notes, prefill-decode).
export const CHECK_WORK = [
  't_math = 2 × 70e9 × 8 ÷ 1,979 TFLOP/s = 566 µs',
  't_read = (70 GB weights + 137 MB activations + 5.37 GB KV) ÷ 4.8 TB/s = 75.5 GB ÷ 4.8 TB/s = 15.7 ms',
  't_step = max(566 µs, 15.7 ms) = 15.7 ms, memory-bound',
].join('\n');

// BF16 weights on an H100 (Review Focus 2): the weights alone do not fit, so every output is replaced.
export const DOES_NOT_FIT = 'does not fit on one GPU: see Serving a 1T model';
export const CHECK_WORK_DOES_NOT_FIT = [
  'weights    = 70e9 × 2 bytes = 140 GB',
  'GPU memory = 80 GB nominal',
  '140 GB of weights > 80 GB of memory → does not fit on one GPU',
].join('\n');

// Storyboard §6 "Try this", as the page prints it ("prompt → Insight: … rest"; [[slug]] prints as the lesson title).
export const TRY_THIS = [
  'Set Phase to prefill (GPU H200, Weight format FP8) and slide Prompt tokens from 1 to 8,192, watching the bound: memory-bound up to 217 tokens, compute-bound from there (15.4 ms at 217; 70.7 ms at 1,000; 580 ms at 8,192). Switch Weight format to BF16: the flip stays at 217. Switch GPU to B200: it moves to 302, in BF16 as in FP8. → Insight: the crossover, counted in tokens per weight read, is set by the GPU\'s ratio of math to bandwidth. FP8 halves the bytes and doubles the math rate, so on an H200 it moves the ridge but not the token crossover (the same point made for the H100\'s 318 in A GPU for LLM people).',
  'Set Phase to decode with Context per user at 2,048 and slide Users in the batch 1 → 8 → 64 → max that fits: per user 67.9 → 63.6 → 42.1 → 33.7 tok/s; per GPU 68 → 509 → 2,694 → 3,543 tok/s. → Insight: batching trades each user\'s speed for the GPU\'s total, cheaply at first (one shared read of the weights), then more steeply as KV reads grow.',
  'Keep Users in the batch at max that fits and slide Context per user 2,048 → 8,192 → 32,768 → 131,072: max users 105 → 26 → 6 → 1; tokens per GPU 3,543 → 890 → 214 → 42. → Insight: at long context the KV cache, not the weights, decides how many users a GPU can hold, and with them its throughput. That is the problem PagedAttention, MQA, GQA, MLA and FP8 KV caches (see Quantization) attack.',
  'Switch Weight format to BF16 on the H200 at Context per user 2,048: 1 GB is left and only 1 user fits (34 tok/s per GPU). → Insight: the weight format decides whether there is room for users at all. You\'ll see what FP8 and FP4 cost in quality in Quantization.',
];
