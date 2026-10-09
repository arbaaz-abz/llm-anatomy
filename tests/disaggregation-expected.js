// disaggregation storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test and the e2e spec.
// The page code never imports from tests/; captions.js is the page's copy, and the page test asserts the two are equal.

export const CAPTIONS = [
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
];

export const CHECK_WORK = [
  'transfer = prompt × KV bytes ÷ link = 4,096 × 327,680 B ÷ 50 GB/s = 1.34 GB ÷ 50 GB/s = 26.8 ms',
  'tokens per expert = users × EP × k ÷ E = 64 × 16 × 6 ÷ 384 = 16',
].join('\n');

// Storyboard §6 "Try this", as the page prints it ("prompt → Insight: … rest"; [[slug]] prints as the lesson title).
export const TRY_THIS = [
  'Set Link between pools to network 400 Gb/s and KV cache to BF16. Slide Prompt length 512 → 4,096 → 131,072: transfer 3.36 ms → 26.8 ms → 859 ms, prefill 36.2 ms → 290 ms → 9.27 s, ratio 9.3% every time. Now slide down to 128: transfer 839 µs, prefill 15 ms, ratio 5.6%, and a note beside the slider says why: below 217 tokens a prefill is one weight read, and real transfers add a fixed start-up cost the toy does not model. → Insight: past the crossover, the transfer-to-prefill ratio is set by the model and the link, not the prompt. Long prompts do not make disaggregation worse in this model; slow links do.',
  'Keep Prompt length at 4,096 tokens and switch Link between pools: network 400 Gb/s 9.3% · network 800 Gb/s 4.6% · NVLink5 0.5%. Then set KV cache to FP8, which halves each: 4.6%, 2.3%, 0.3%. The MLA readout, at 400 Gb/s with BF16 KV: 288 MB, 5.76 ms. → Insight: a fast link or a smaller KV makes the split nearly free; this is one more reason MLA and FP8 KV caches matter: see MQA, GQA, MLA and Quantization.',
  'Set Users per GPU to 64 and slide GPUs sharing the experts (EP size) from 1 to 72. At 1 the free memory reads "does not fit" (865 GB of weights against 288 GB of memory) and each expert gets 1 token; EP 8 gives 8 tokens per expert and 108 GB of weights per GPU; EP 16, 16 and 54.1 GB; EP 72, 72, 12 GB and 276 GB free. → Insight: wide EP both shrinks each GPU\'s share of the weights and pools users\' tokens at each expert, so every expert does more math per byte read. Even at EP 72 with 256 users per GPU (288 tokens per expert) the intensity is 903, still under the FP4 ridge of 1,875, which an expert reaches only at 699 tokens.',
];
