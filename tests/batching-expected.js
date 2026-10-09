// batching: the strings the page must print, typed here once (the page test and the e2e spec both import them).
// CAPTIONS are storyboard §5's captions, verbatim; TRY_THIS is §6's list with its numbers filled in from the scenarios.
export const CAPTIONS = [
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
];

export const TRY_THIS = [
  'Defaults (3 seats). Compare the lanes: static ends at step 10 with seats busy 57.6% of the time; continuous ends at step 6 with 90.5%. → Insight: refilling a seat the step after it frees is the whole trick. Same work, 4 fewer steps, and D waits 2 steps instead of 6.',
  'Predict first: what happens to D if C\'s answer grows from 6 to 10 tokens? Set C\'s answer length to 10. Static: D now waits until step 11 and is done at step 14 (seats busy 51.1%). Continuous: D still runs steps 3–6. → Insight: in a static batch the longest answer sets everyone\'s schedule; in a continuous one it only sets its own.',
  'Set the Seats slider to 4. Continuous: D starts at step 1 and is done at step 4, but seat utilization drops to 67.9%, because nobody else is waiting for that fourth seat. Static with 4 seats: D still waits until step 7. → Insight: a seat helps only if someone is waiting, and only continuous batching can hand it out mid-run. In 2023 engines each seat also cost a full reserved strip of KV memory, which is why seats were scarce (see PagedAttention).',
  'Set D\'s prompt to 4,096 tokens. With the token budget off: one 290 ms step, A done at 348 ms. Budget 2,048: longest step 145 ms, A done at 334 ms. Budget 512: longest step 36.2 ms, A done at 116 ms, C at 189 ms, while D\'s first token moves only from 334 ms to 348 ms. → Insight: chunked prefill trades a little of the long prompt\'s TTFT for a bounded TPOT for everyone else. The smaller the budget, the smoother the streams, until steps become too small to keep the GPU busy (not modeled here; the 217-token crossover in Prefill vs decode is the floor).',
];

// Frame texts printed on the stage (the "Numbers shown" column), pinned by the page test.
export const STAGE_TEXT = {
  frame2Done: 'A done step 4 · B done step 2 · C done step 6',
  frame3Idle: 'idle seat-steps: B 4, A 2 · D waiting steps 1–6',
  frame4Busy: 'busy 19 of 33 seat-steps = 57.6% · 15 tokens in 11 steps',
  frame5Admitted: 'D admitted step 3 (B\'s seat) · done step 6',
  frame6Static: 'last step 10 · busy 57.6% · 1.36 tokens per step',
  frame6Continuous: 'last step 6 · busy 90.5% · 2.14 tokens per step',
  frame7Mixed: '6 + 2 = 8 tokens · 14.6 ms',
  frame8Step: 'step 3: 4,098 tokens, 290 ms',
  frame8Times: 'A done at 348 ms · C done at 378 ms · D\'s first token at 334 ms',
  frame8Note: 'step 0 (23 prompt tokens) 14.7 ms, other short steps 14.6 ms',
  frame9Budget: 'budget 512 tokens per step: longest step 36.2 ms',
  frame9Done: 'A done 116 ms (was 348 ms) · C done 189 ms (was 378 ms)',
  frame9First: 'D\'s first token 348 ms (was 334 ms)',
  frame10Four: 'last step 6 · busy 67.9% · 2.14 tokens per step',
};
