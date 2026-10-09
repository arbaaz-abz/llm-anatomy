// disaggregation storyboard §5 captions, verbatim (one per frame); content.js puts them on the steps, frames.js labels each stage.
export const CAPTIONS = Object.freeze([
  'On shared GPUs, prefill and decode take turns. D\'s long prompt stalled A and C for 290 ms, and even with chunks their steps stretched to 36 ms.',
  'Users judge two numbers: time to first token and time per token. Goodput counts only the requests that meet both targets, and on shared GPUs they fight.',
  'Disaggregation gives prefill and decode their own GPUs. D\'s long prompt runs on the prefill GPU, while A and C keep their 14.6 ms steps.',
  'Then D\'s keys and values must move to the decode GPU: 1.34 GB, about 26.8 ms over a 400 Gb/s network port and 1.49 ms over NVLink.',
  'Each pool gets its own size and parallelism. vLLM\'s best DeepSeek-R1 setup on GB200 was four 2-GPU prefill groups feeding one 8-GPU decode group.',
  'With 384 experts and 6 per token, 64 users give each expert about one token per step. Each expert is read from memory to do almost no math.',
  'Wide expert parallelism spreads the experts over 16 GPUs and sends each token to its expert\'s GPU. Pooling all their users gives each expert 16 tokens per step.',
  'DeepSeek\'s 2025 production did exactly this: prefill spread over 32 GPUs, decode over 144, each decode GPU holding just two routed experts and one shared.',
  'Wide EP exchanges tokens at every MoE layer, so it wants every GPU on the fastest link. A 72-GPU NVLink rack keeps the whole exchange inside one domain.',
  'The bill: two pools to balance, KV crossing a link for every request, and constant expert traffic. With short prompts, small models or slow links it may not pay.',
]);
