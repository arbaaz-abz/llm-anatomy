// The ten captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
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
]);
