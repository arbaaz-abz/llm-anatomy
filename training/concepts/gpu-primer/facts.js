// gpu-primer's dated text (pure, no DOM): the §8 rows and framing, the hook, intuition and takeaways, and the notes printed
// under the stage. Dated numbers are {hw:…}/{entry.key} placeholders filled from data/*.json; numbers derived from them
// (ridge points, the H100's idle share, NVLink both directions) are computed from the same entries, never typed.
import { bitsPerElement } from '@math/roofline.js';
import { sharePct } from '@math/memory.js';
import { formatRatio } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';
import { chipPreset } from './hardware.js';
import { analyze, int } from './format.js';
import { SPARSE_FACTOR } from './numbers.js';

const DASH = '—';
const hw = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;

// A derived number, or a dash when the data is absent (LESSON = lessonFor(null) is for validation only).
function derived(data, compute) {
  try {
    return data?.hardware ? compute() : DASH;
  } catch {
    return DASH;
  }
}

const bf16Ridge = (data, chip) => derived(data, () => int(analyze({ chip, fmt: 'bf16', tokens: 4 }, chipPreset(data, chip)).ends[0].ridge));
// Tensor cores idle while HBM delivers: 1 − compute ÷ memory for the 4-token real-size multiply on an H100, in BF16.
export const h100Idle = (data) => derived(data, () => {
  const { time } = analyze({ chip: 'h100', fmt: 'bf16', tokens: 4 }, chipPreset(data, 'h100')).ends[0];
  return `${sharePct(time.memoryS - time.computeS, time.memoryS).toFixed(1)}%`;
});
const h100PeakShare = (data) => derived(data, () => {
  const a = analyze({ chip: 'h100', fmt: 'bf16', tokens: 4 }, chipPreset(data, 'h100'));
  return `${sharePct(a.ends[0].attainable, a.peakTflops, { decimals: 2 }).toFixed(2)}%`;
});
const bothWays = (data) => derived(data, () => int(2 * hw(data, 'h100', 'nvlink_gb_s_each_way')));

export const hook = 'Why does an H100, rated at {hw:h100.bf16_dense_tflops} trillion operations a second, run a four-token matrix multiply at about 1% of that speed?';

export function intuition(data) {
  const ridge = bf16Ridge(data, 'h100');
  return [
    `A GPU is two machines glued together. One is compute: {hw:h100.sm_count} small processors on an H100, each with tensor cores that do nothing but multiply matrices, fast. The other is memory: stacks of HBM beside the chip that hold the weights and everything else the GPU works on. Every operation pays twice, once to move its numbers between HBM and the processors and once to do the math, and on a modern GPU the math is the cheap part. An H100 can do ${ridge} operations in the time it takes HBM to deliver one byte.`,
    `So the question for any operation is how much math it does per byte it moves: its arithmetic intensity. A matrix multiply reads each weight once and uses it once for every token in the batch. In BF16 (2 bytes per number, 2 operations per weight per token) that makes the intensity about the number of tokens: with four tokens, each weight byte buys about four operations, far below ${ridge}, and the tensor cores sit idle ${h100Idle(data)} of the time, waiting on memory. That is the hook's answer: the chip isn't slow, the multiply is starved. With 4,096 tokens the same weights buy 2,048 operations per byte and the tensor cores become the limit. Training pushes millions of tokens through each weight per step, so it lives on the compute side; generating text one token at a time for a few users lives on the memory side (see [[prefill-decode]]).`,
    'Smaller number formats move both sides. FP8 halves the bytes and, on an H100, doubles the tensor-core rate, so most multiplies simply run twice as fast. The B300\'s FP4 is the twist: its tensor cores grew faster than its byte savings, so a B300 needs roughly twice as many tokens per weight read before FP4 math, rather than memory, sets the pace (on a B200 the shift is small). The price of low precision is range and resolution: FP4 has 16 values, so blocks of numbers share a scale, and accuracy is a recipe question (see [[quantization]] and [[scale-reliability]]).',
  ];
}

export function takeaways(data) {
  return [
    `Every operation pays for bytes moved and FLOPs done; its arithmetic intensity (FLOPs per byte) against the chip's ridge point (peak ÷ bandwidth, ${bf16Ridge(data, 'h100')} on an H100) says which one sets its speed.`,
    `A matrix multiply's intensity is about 2 × tokens ÷ bytes per number, so in BF16 it is roughly the number of tokens sharing each weight read: a few tokens are memory-bound (${h100PeakShare(data)} of peak at 4), a few hundred cross the ridge, and training batches sit far on the compute side.`,
    `Lower precision shrinks bytes and speeds up tensor cores: FP8 makes an H100 multiply about 2× faster without changing the batch needed; FP4 needs a shared block scale (${bitsPerElement('nvfp4')} bits in NVFP4) and, on a B300, about twice the tokens to stay compute-bound (a B200 barely shifts). The BF16 ridge is flat across NVIDIA's flagship generations; the H200's bandwidth bump is the exception and lowers it.`,
  ];
}

export function framing() {
  return `The 2026 hardware table: dense peaks only (vendor sparse figures are ${formatRatio(SPARSE_FACTOR)} and not used), HBM capacities nominal in decimal GB, link speeds with their direction. Each ridge point is that chip\'s dense BF16 peak ÷ its HBM bandwidth, computed from the same entries.`;
}

// The 11 sourced rows of storyboard §8 (its FlashAttention row is the prose paragraph below: no data entry to cite yet).
export function factRows(data) {
  return [
    { claim: `H100 SXM ({hw:h100.release_date|year}): {hw:h100.hbm_gb} GB HBM3 (nominal), {hw:h100.hbm_tbps|raw} TB/s, {hw:h100.bf16_dense_tflops} TFLOPS BF16, {hw:h100.fp8_e4m3_dense_tflops} FP8 (2 × BF16), NVLink ${bothWays(data)} GB/s both directions ({hw:h100.nvlink_gb_s_each_way} each way, the convention in [[cluster-topology]]); ridge ${bf16Ridge(data, 'h100')}.` },
    { claim: 'An H100 has {hw:h100.sm_count} SMs (streaming multiprocessors).' },
    { claim: `H200: {hw:h200.hbm_gb} GB (nominal), {hw:h200.hbm_tbps|raw} TB/s, the same compute as the H100 ({hw:h200.bf16_dense_tflops} TFLOPS BF16); ridge ${bf16Ridge(data, 'h200')}: the extra bandwidth makes it easier to be compute-bound.` },
    { claim: 'B200 (HGX): {hw:b200.hbm_gb} GB nominal ({hw:b200.hbm_usable_gb} usable), {hw:b200.hbm_tbps|raw} TB/s, {hw:b200.bf16_dense_tflops} / {hw:b200.fp8_e4m3_dense_tflops} / {hw:b200.nvfp4_dense_tflops} TFLOPS BF16 / FP8 / NVFP4.' },
    { claim: 'B300: {hw:b300.hbm_gb} GB (nominal), {hw:b300.hbm_tbps|raw} TB/s, {hw:b300.bf16_dense_tflops} / {hw:b300.fp8_e4m3_dense_tflops} / {hw:b300.nvfp4_dense_tflops} TFLOPS BF16 / FP8 / NVFP4.' },
    { claim: 'MI355X: {hw:mi355x.hbm_gb} GB (nominal), {hw:mi355x.hbm_tbps|raw} TB/s, {hw:mi355x.bf16_dense_tflops} / {hw:mi355x.fp8_e4m3_dense_tflops} / {hw:mi355x.mxfp4_dense_tflops} TFLOPS BF16 / FP8 / MXFP4.' },
    { claim: 'TPU v7 Ironwood: {hw:tpu-v7.hbm_gb} GB (nominal), {hw:tpu-v7.hbm_tbps|raw} TB/s, {hw:tpu-v7.bf16_dense_tflops} TFLOPS BF16, {hw:tpu-v7.fp8_e4m3_dense_tflops} FP8.' },
    { claim: 'Rubin: {hw:rubin.hbm_gb} GB HBM4, {hw:rubin.nvfp4_dense_tflops} TFLOPS dense NVFP4 for training; HBM bandwidth {hw:rubin.hbm_tbps|raw} TB/s, because NVIDIA\'s own pages disagree, so it is shown as a range everywhere; no settled BF16 or FP8 figure, so neither is shown.' },
    { claim: 'Number formats (sign/exponent/mantissa bits): FP32 {hw:formats.fp32_layout}, BF16 {hw:formats.bf16_layout}, FP16 {hw:formats.fp16_layout}, FP8 E4M3 {hw:formats.fp8_e4m3_layout} or E5M2 {hw:formats.fp8_e5m2_layout}; MXFP4 shares one {hw:formats.mxfp4_scale_bits}-bit power-of-two scale per {hw:formats.mxfp4_block_size} numbers; NVFP4 one {hw:formats.nvfp4_scale_format} scale per {hw:formats.nvfp4_block_size}, plus a per-tensor FP32 scale.' },
    { claim: 'gpt-oss-120b stores its MoE weights as {gpt-oss-120b.weight_format}, so the {gpt-oss-120b.total_params|count} model fits one {hw:h100.hbm_gb} GB GPU (post-training quantization, not native FP4 training).' },
    { claim: 'NVIDIA trained a {nvidia-nvfp4-12b.total_params|count} model on {nvidia-nvfp4-12b.pretrain_tokens|count} tokens in {nvidia-nvfp4-12b.pretrain_precision} and matched its FP8 loss ({nvidia-nvfp4-12b.release_date|year}); Nemotron 3 Super and Ultra were pretrained in {nemotron-3-super.pretrain_precision}{nemotron-3-ultra.pretrain_precision|cite} too.' },
  ];
}

export const PROSE = Object.freeze([
  'FlashAttention tiles attention so the score matrix stays in on-chip SRAM instead of HBM: the same math, far fewer HBM bytes (arXiv 2205.14135).',
]);

// Page text under the stage, one list per frame: dated notes (filled from data) and frame → lesson hand-offs.
export const BELOW = Object.freeze([
  ['H100 SXM ({hw:h100.release_date|year}): {hw:h100.hbm_gb} GB of HBM3, nominal. What fills it during training is [[training-memory]].'],
  [`Vendor sheets also print "with sparsity" figures, ${formatRatio(SPARSE_FACTOR)} these; every peak on this page is dense.`],
  ['SRAM per SM is a rough figure ({hw:memory-hierarchy.sram_per_sm}, reported); HBM is {hw:h100.hbm_gb} GB on an H100.'],
  ['The multiply followed here is the last one of [[attention]]: four token rows times W_O. That lesson is optional background.'],
  [],
  ['The roofline is a bound, not a prediction: real kernels add launch and latency overheads, which matter most at a few hundred bytes like this toy multiply.'],
  [],
  ['Decoding for a few users sits on the memory side of this roof; prefill sits on the compute side: [[prefill-decode]].'],
  ['The H100\'s FP8 figure ({hw:h100.fp8_e4m3_dense_tflops} TFLOPS) is reported: half of NVIDIA\'s with-sparsity figure, 2 × BF16.'],
  ['gpt-oss-120b stores its MoE weights as {gpt-oss-120b.weight_format}. What fewer bits cost in quality: [[quantization]].'],
  ['Rubin\'s HBM bandwidth is published as {hw:rubin.hbm_tbps|raw} TB/s (NVIDIA\'s own pages disagree), so its ridge and its crossing print as ranges.'],
]);
