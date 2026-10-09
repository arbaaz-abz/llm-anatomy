// batching storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps and frames.js labels each stage with them.
export const CAPTIONS = Object.freeze([
  'Four requests and a GPU that runs at most three at a time. Each running request holds a seat, and every step each seated request gets one more token.',
  'Static batching: A, B and C start together, and this batch runs until its longest member, C, is done.',
  'B is done after step 2 and A after step 4, but their seats stay held until C finishes. D arrived at step 1 and waits outside.',
  'D finally starts at step 7, alone, and is done at step 10. Over the whole run, seats were busy only 57.6% of the time.',
  'Continuous batching runs the scheduler before every step. B leaves after step 2, so D takes its seat at step 3 instead of waiting for the whole batch.',
  'Same four requests: all done by step 6 instead of 10, with seats busy 90.5% of the time. The GPU did the same work in fewer steps.',
  'At step 3 the GPU runs D\'s six-token prefill and A\'s and C\'s next tokens in one pass. Prefill and decode can share a step.',
  'What if D\'s prompt were 4,096 tokens? Step 3 now takes 290 ms, so A and C wait 19.9 times longer than usual for their next token.',
  'Chunked prefill caps each step at 512 tokens and feeds D\'s prompt in slices. A and C never wait over 36 ms; D\'s first token comes 14 ms later.',
  'In 2023 engines each seat reserved memory for the longest possible answer, which kept seats few. PagedAttention makes room for a fourth, and D starts at step 1.',
]);
