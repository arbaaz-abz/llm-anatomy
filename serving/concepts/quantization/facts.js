// quantization's dated text (pure, no DOM): the §8 rows and framing, the hook, intuition and takeaways, and the notes printed
// under the stage. Dated numbers are {entry.key} placeholders filled from data/*.json; numbers derived from them (users per GPU,
// bits per weight) are computed with math/ functions from the stage constants, which the page test pins to the same data.
import { bitsPerElement } from '@math/roofline.js';
import { memoryPlan, kvBytesFor } from './format.js';
import { stagePreset } from './hardware.js';

const h200 = stagePreset('h200');
const usersAtBits = (bits) => memoryPlan({ bits, hbmBytes: h200.hbmBytes, kvBytesPerToken: kvBytesFor('bf16') }).users;
const bits = (format) => bitsPerElement(format);

export const hook = 'How can you throw away three-quarters of every weight\'s bits and still get nearly the same answers, and why bother?';

export const intuition = () => [
  `Why bother is the easy half. In [[prefill-decode]] a decode step was mostly the time to read the weights. Store each weight in 4 bits instead of 16 and there are about a quarter as many bytes to read, so steps get faster, and the freed memory holds more users' KV caches. A 70B model in BF16 fills an H200 and leaves room for ${usersAtBits(bits('bf16')) === 1 ? 'one user' : `${usersAtBits(bits('bf16'))} users`} at 2,048 tokens; in a 4-bit format it leaves room for ${usersAtBits(bits('nvfp4'))}.`,
  'The how is a scale and a rounding. Take a small block of weights, divide them by a scale so the largest lands at the top of a tiny grid, round each to the nearest grid point, and store the grid points plus the scale. To use a weight, multiply back. Each weight comes back slightly off, but the errors are small and point in random directions, so their effect on a layer\'s output mostly cancels. Two things break this. One large weight in a block stretches the scale and crushes its neighbors to zero, which is why modern formats give every 16 or 32 weights their own scale. And errors that are tiny per token can pile up where a model attends over hundreds of thousands of tokens.',
  `The 2026 formats differ in their grids and scales. FP8 keeps 8-bit floating-point numbers and roughly halves bytes and math time compared with BF16. MXFP4 and NVFP4 store 4-bit floats; MXFP4 shares a power-of-two scale across 32 weights, NVFP4 an 8-bit float scale across 16, which costs a little more memory (${bits('nvfp4')} bits per weight instead of ${bits('mxfp4')}) and rounds more finely. The safest 4-bit models are trained to expect their format: gpt-oss and DeepSeek-V4 shipped their weights in 4-bit formats, and Kimi K2.x is reported to ship INT4. The formats themselves are taught in [[gpu-primer]].`,
];

export const takeaways = () => [
  'Quantizing is scale, round, multiply back; giving every small block its own scale keeps one outlier from crushing its neighbors (frames 2–5, try-this 1).',
  `FP4 formats round to an uneven grid; NVFP4's 8-bit scale per 16 weights fits more closely than MXFP4's power-of-two scale per 32, at ${bits('nvfp4')} against ${bits('mxfp4')} bits per weight (frames 6–8, try-this 2).`,
  'Fewer weight bytes speed up decode and free memory for users; only low-precision math speeds up prefill; and long-context accuracy is where careless quantization fails (frames 8–10, try-this 3–4).',
];

export const framing = () => 'Every format and model below is stored with fewer than 16 bits per weight; ratios and sizes are the vendors\' or labs\' own, and a size or a recovery figure not confirmed by a primary source is marked reported.';

// The sourced rows of storyboard §8 (nine). A `|cite` placeholder prints nothing and attaches the fact's source link and its "reported" flag.
export function factRows() {
  return [
    { claim: 'Weight-only INT4 (GPTQ, AWQ) stores weights in 4 bits and keeps the math in 16-bit{sv:quant-weight-only.note|cite}: about a quarter of the bytes are read in decode, but the math is unchanged, so it helps less at large compute-bound batches.' },
    { claim: 'DeepSeek-V4-Pro ships {deepseek-v4-pro.weight_formats}: FP8 is half of BF16\'s bytes and, on hardware with FP8 tensor cores, half its math time.' },
    { claim: 'MXFP4: 4-bit E2M1 numbers, {hw:formats.mxfp4_block_size} per block, one {hw:formats.mxfp4_scale_bits}-bit power-of-two (E8M0) scale. NVFP4: {hw:formats.nvfp4_block_size} per block, an {hw:formats.nvfp4_scale_format} scale plus a per-tensor FP32 scale, {hw:formats.nvfp4_bits_per_value|raw} bits per value, native on Blackwell.' },
    { claim: 'gpt-oss-120b shipped post-trained with MXFP4 MoE weights{gpt-oss-120b.weight_format|cite}: {gpt-oss-120b.total_params|count5} total and {gpt-oss-120b.active_params|count} active parameters, and it fits one {gpt-oss-120b.fits_gpu_gb} GB GPU.' },
    { claim: 'DeepSeek-V4-Pro: experts in FP4, other weights in FP8, trained with quantization-aware training{deepseek-v4-pro.post_training_qat|cite}; the checkpoint is about {deepseek-v4-pro.checkpoint_gb|raw} GB (size reported).' },
    { claim: 'Kimi K2.5 ships {kimi-k2.5.weight_format} MoE weights (quantization-aware training); the other Kimi K2 models are reported to do the same.' },
    { claim: 'Precision plus hardware: Kimi K2.5 on B200 in NVFP4 against H200 in INT4 gave {sv:inferencex-kimi-k2.5-b200.cost_gain_vs_h200|raw}× lower dollars per million tokens, between 30 and 90 tokens per second per user (InferenceX; its tokens per second may include input tokens, the source does not say).' },
    { claim: 'FP8 to NVFP4 on DeepSeek-R1-0528 cost {sv:nvfp4.r1_accuracy_loss_pct_max}% or less accuracy on most benchmarks (NVIDIA\'s own measurement).' },
    { claim: 'vLLM\'s FP8 KV cache ({sv:vllm-fp8-kv.date|date}) halves KV bytes ({sv:vllm-fp8-kv.bytes_factor|raw} of BF16\'s). A naive kernel dropped 128K needle-in-a-haystack accuracy from {sv:vllm-fp8-kv.baseline_niah_128k_pct}% to {sv:vllm-fp8-kv.naive_niah_128k_pct}%; the final configuration recovers {sv:vllm-fp8-kv.auc_recovery_128k_pct}% of the baseline AUC at 128K for Llama-3.3-70B (the post\'s words; it does not tie the figure to one fix). Throughput on Llama rose {sv:vllm-fp8-kv.throughput_gain_llama_pct|raw}%; below about {sv:vllm-fp8-kv.breakeven_tokens} tokens it is not worth it.' },
  ];
}

// Page text under the stage, one list per frame: dated notes (filled from data) and frame → lesson hand-offs.
export const BELOW = Object.freeze([
  [],
  [],
  [],
  [],
  [],
  [],
  ['The toy\'s NVFP4 scale rule is a stand-in. Real MXFP4 rounds as in the Microscaling paper (a power-of-two scale from the largest value); NVIDIA\'s per-block rule is not in our sources.'],
  ['H200: {hw:h200.hbm_gb} GB, nominal. B200: {hw:b200.hbm_usable_gb} GB usable ({hw:b200.hbm_gb} nominal), so its user counts use 180. How a user\'s cache is counted: [[prefill-decode]].'],
  ['Dense peaks, reported: B200 BF16 {hw:b200.bf16_dense_tflops}, FP8 {hw:b200.fp8_e4m3_dense_tflops}, NVFP4 {hw:b200.nvfp4_dense_tflops} TFLOPS; the H200 has no FP4 figure. Why prefill is compute-bound: [[prefill-decode]].'],
  ['The test: vLLM\'s FP8 KV-cache post ({sv:vllm-fp8-kv.date|date}), Llama-3.3-70B-Instruct, 128K-token needle-in-a-haystack; the final configuration recovers {sv:vllm-fp8-kv.auc_recovery_128k_pct}% of the baseline AUC at 128K. FP8 KV pays off above about {sv:vllm-fp8-kv.breakeven_tokens} tokens. More on the cache: [[paged-attention]].'],
  ['gpt-oss-120b stores {gpt-oss-120b.weight_format}. DeepSeek-V4-Pro ships {deepseek-v4-pro.weight_formats}. Kimi K2.5: {kimi-k2.5.weight_format}, reported.'],
]);
