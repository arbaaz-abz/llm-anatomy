// serving-overview's text (pure, no DOM): the hook and intuition (storyboard §3), the §8 rows and framing, the notes under the
// stage and the takeaways (§9). Dated numbers are {sv:entry.key} placeholders filled from data/serving.json; nothing is typed.

export const HOOK = 'What happens between pressing Enter and the first word appearing, and why does the rest of the answer arrive at a steady beat instead of all at once?';

export const INTUITION = Object.freeze([
  'Your text is cut into tokens and sent as one request. A router picks one of many identical copies of the model (replicas), and inside that replica a scheduler decides when your request joins the GPU\'s work. Then the model reads your whole prompt in a single pass. This is prefill: every prompt token flows through every layer together, and each one leaves its key and value in the KV cache. At the end of that pass the model picks the first word of the answer. The time until that word reaches you is TTFT, time to first token.',
  'After that, the model can only move one token at a time. The next token depends on the one just picked, so it cannot be computed until that pick is known. It feeds the newest token back in, adds one key and value to the cache, and picks the next token. This is decode, and it is why the answer streams: each step makes exactly one more token for you, and steps come at a steady pace, the time per output token (TPOT). A 500-token answer needs about 500 steps, so for most chats the answer, not the prompt, decides how long you wait in total.',
  'The GPU is not working for you alone. Every decode step re-reads all of the model\'s weights, and the same read serves every request in the running batch, so a server keeps many requests in flight at once. That is the first of many tricks; the rest of this track takes the journey one stop at a time.',
]);

export const INTUITION_NOTE = 'Tokens, keys and values, and why decoding re-reads the past are taught in [[kv-cache]].';

export const FRAMING = 'Measured 2026 servers and the software inside them. These are other models on other hardware: the times in the animation and the toy are a floor for one model on one GPU, not a measurement.';

// The 5 rows of storyboard §8, in order. Rows keep their placeholders, so the scaffold adds each source link and "reported" chip.
export const FACT_ROWS = Object.freeze([
  { claim: 'A busy 2026 server, measured: DeepSeek-V4-Pro on GB300 NVL72 at {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user} output tokens/s per user (ISL 8K / OSL 1K, FP4, disaggregated Dynamo + vLLM; InferenceX, measured {sv:inferencex-v4-pro-gb300.date}).' },
  { claim: 'A 128K-token prompt took {sv:lmsys-gb300-longctx.ttft_128k_s|raw} s to its first token: DeepSeek-R1 on GB300 NVL72 with chunked pipeline-parallel prefill (LMSYS, {sv:lmsys-gb300-longctx.date}).' },
  { claim: 'Prompts often start the same way, which is why routers try to send a request to the replica already holding its prefix: {sv:deepseek-v3-production.kv_hit_rate_pct|raw}% of DeepSeek\'s input tokens hit its KV cache (V3/R1 production, {sv:deepseek-v3-production.date|date}).' },
  { claim: 'The engines inside a replica: {sv:engines.names}; orchestration by NVIDIA Dynamo (1.0, GA {sv:dynamo.ga_date}) or llm-d (CNCF Sandbox {sv:llm-d.cncf_sandbox_date}).' },
  { claim: 'Closed providers do not publish their serving stacks; this track uses open engines and published measurements.', derived: true },
]);

// Page text under the stage, one list per frame (0-based): hand-offs to the lessons that go deeper.
export const BELOW = Object.freeze([
  [],
  ['Why the router\'s choice matters: see [[prefix-caching]].'],
  [],
  ['How a pick is made from the model\'s probabilities: see [[sampling]].'],
  [],
  [],
  [],
  ['Keeping a finished request\'s cache for the next prompt that starts the same way: see [[prefix-caching]].'],
  ['Requests A–D are the course\'s toy batch: [[batching]] follows these same four step by step, and [[paged-attention]] stores their caches. Steps count from 0; token positions count from 1.'],
  [],
]);

export const TAKEAWAYS = Object.freeze([
  'A request goes router → queue → scheduler → GPU; prefill reads the whole prompt in one pass and ends with the first token (TTFT), then decode adds one token per step at a steady TPOT (frames 2–7).',
  'For chat-sized prompts, the answer\'s length sets most of the total time; the prompt mostly sets TTFT (try-this 1).',
  'Each decode step advances every request in the batch at once, which is why servers share GPUs and why the rest of this track is about keeping that step fast and full (frames 9–10).',
]);
