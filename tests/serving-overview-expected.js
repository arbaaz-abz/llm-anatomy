// serving-overview storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test and the e2e spec.
// The try-this text is what the page prints: [[slug]] links read as the lesson's title.

export const CAPTIONS = [
  'You press Enter. Your text is cut into tokens, three here, and sent to the provider as one request.',
  'A router sends the request to one of several identical copies of the model, called replicas. Which one it picks matters later, when some copies remember earlier prompts.',
  'Inside the replica, a scheduler decides which requests join the GPU\'s next step. When the GPU is busy, a request waits in a queue.',
  'Prefill: the model reads all three prompt tokens in one pass and stores their keys and values. At the end of that pass it picks the first answer token.',
  'The first token streams back right away. The wait so far is the time to first token, TTFT: time in the queue plus prefill.',
  'Decode: the newest token goes back in, adds its key and value to the cache, and the model picks one more. Every step re-reads all the weights.',
  'Tokens arrive at a steady beat, one per decode step. The gap between two of them is the time per output token, TPOT.',
  'An end token, or a length limit, stops the request. Its KV cache is freed, unless the server keeps it in case the next prompt starts the same way.',
  'Zoom out: the GPU never serves you alone. Each step advances every running request by one token, so one read of the weights serves all of them, the batch.',
  'Each lesson in this track speeds up one stop on this journey. Next: why prefill and decode behave so differently on a GPU.',
];

export const CHECK_WORK = [
  'TTFT = queue + prefill = 0 s + 141 ms = 141 ms',
  'total = TTFT + (n − 1) · TPOT = 141 ms + 499 · 14.7 ms (7.35 s) = 7.49 s   (each value rounded once, from the unrounded function output)',
].join('\n');

// Storyboard §6 "Try this", as the page prints each item: "prompt → Insight: insight rest".
export const TRY_THIS = [
  'Predict first: which costs more time, a 10× longer prompt or a 10× longer answer? Set Prompt length from 2,000 to 20,000: TTFT 141 ms → 1.41 s, total 7.49 → 8.76 s. Set Prompt length back to 2,000, then Answer length from 500 to 50: total 7.49 s → 863 ms. → Insight: the prompt sets the wait for the first token; the answer\'s length sets the total. Decode is 98.1% of the default request\'s time.',
  'Switch Decode speed per user from "alone on the GPU" (67.9 tok/s) to "sharing with 104 others" (33.7 tok/s): TTFT does not move (141 ms), total goes 7.49 → 14.9 s. → Insight: TTFT and TPOT are separate dials. Sharing the GPU with more users slows each user\'s stream, not the first token, which is why servers track both, and why Continuous batching and Disaggregated serving treat them as two targets.',
  'On "sharing with 104 others", set Time in queue to 2 s: TTFT 141 ms → 2.14 s, total 14.9 → 16.9 s, while TPOT stays 29.6 ms. → Insight: a queue only delays the start. Once your request is in the batch, its pace depends on the step time, not on how long it waited.',
  'Slide Prompt length down to 3 tokens: TTFT is still 14.6 ms, not near zero. → Insight: even a tiny prompt costs one full read of the weights, the same floor a decode step pays (see Prefill vs decode).',
];
