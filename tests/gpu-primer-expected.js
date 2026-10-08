// gpu-primer storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test and the e2e spec.
// Caption 11 says "6-fold" / "3.56-fold" where the storyboard prints "6×" / "3.56×": "×" is a formula operator, which
// README lesson 2 and validateLessonSpec forbid in captions (reported as a storyboard finding in the task report).
import { formatDuration } from '../math/core.js';

export const CAPTIONS = [
  'A GPU is compute next to stacks of memory called HBM. On an H100, HBM holds 80 GB: the weights and everything else the GPU works on.',
  'Tensor cores inside the compute grid do matrix multiplies at 989 trillion operations a second. HBM feeds them only 3.35 trillion bytes a second.',
  'Each SM also has a tiny on-chip memory, SRAM, much faster than HBM. Fast kernels such as FlashAttention keep their working numbers there between steps.',
  'Take attention\'s last multiply: four tokens times W_O. It moves 256 bytes and does 512 floating-point operations, or FLOPs: 16 for each output number.',
  'Arithmetic intensity is FLOPs per byte moved. This multiply does 2 FLOPs for every byte it reads or writes.',
  'Plot speed against intensity and the roof bends once. The bend, the ridge point, is where moving a byte takes as long as 295 FLOPs on an H100.',
  'At real size the weights are nearly all the bytes, so four tokens get only 4 FLOPs per byte. This multiply is memory-bound: HBM sets the pace.',
  'Each weight read from HBM is used once per token. At 4,096 tokens the multiply does 2,048 FLOPs per byte and is compute-bound: the tensor cores set its pace.',
  'FP8 stores each number in 1 byte and, on an H100, runs the tensor cores twice as fast. Both sides double, so most multiplies just get twice as fast.',
  'A 4-bit number has only 16 possible values, so each block of numbers shares one scale. NVFP4 shares one per 16 numbers: 4.5 bits each, not 4.',
  'From H100 to B300 the BF16 ridge barely moved. B300\'s FP4 tensor cores grew 6-fold while bytes shrank 3.56-fold, so it needs about 600 tokens per weight read.',
];

export const CHECK_WORK = [
  'FLOPs     = 2 × 4 × 8,192 × 8,192 = 536,870,912',
  'bytes     = 2 × (4 × 8,192 + 8,192 × 8,192 + 4 × 8,192) = 134,348,800',
  'intensity = 536,870,912 ÷ 134,348,800 = 4.0 FLOPs/byte',
  'ridge     = 989 ÷ 3.35 = 295.2 FLOPs/byte → memory-bound',
  'crossing  = the token count whose intensity reaches 295.2 = 318.2 tokens',
].join('\n');

// The H100 BF16 4-token compute time, through the one duration formatter (README lesson 35). The storyboard prints
// "0.543 µs"; formatDuration prints "0.54 µs" until the shared fix lands (shared request; the page test pins the
// storyboard text as a todo test that passes once it does).
export const COMPUTE_4 = formatDuration((2 * 4 * 8192 * 8192) / 989e12);

// Storyboard §6 "Try this", as the page prints it ("prompt → Insight: … rest"; [[slug]] prints as the lesson title).
export const TRY_THIS = [
  `H100, BF16, 4 tokens: 1.35% of peak, memory-bound, 40.1 µs of memory time for ${COMPUTE_4} of math. Predict how many tokens it takes to become compute-bound, then slide: 64 (21.3%), 256 (81.6%, still memory-bound), 512 (compute-bound). The readout says 318.2. → Insight: in BF16 the intensity is about the number of tokens sharing each weight read (in general, 2 × tokens ÷ bytes per number), so batch size is the lever. Training pushes thousands of tokens through every weight and is compute-bound; decoding for a few users is not (see Prefill vs decode).`,
  'H100 at 256 tokens: switch BF16 → FP8. Intensity 241 → 482, ridge 295 → 591, still memory-bound; memory time 42.6 µs → 21.3 µs; "tokens needed" stays 318.2 → 318.3. → Insight: on an H100, FP8 doubles both sides of the roof, so a multiply runs about twice as fast on either side of the ridge, but the batch needed to cross does not change.',
  'B300 at 512 tokens: BF16 is compute-bound (intensity 455 vs ridge 313). Switch to FP4: intensity 1,618 vs ridge 1,875, now memory-bound; tokens needed 338 → 605. Then step H100 → B200 → B300 in BF16: the ridge reads 295, 281, 313; the H200 reads 206. Last, B200 in FP4: tokens needed 302 → 343. → Insight: the BF16 ridge is flat across NVIDIA\'s flagship generations (the H200\'s bandwidth bump is the exception, and it lowers the crossing); on the B300 and Rubin, FP4 compute outran memory, so FP4 needs bigger batches there, while on a B200 the crossing barely moves (302 → 343). ("Lower precision pushes the ridge right" is true in FLOPs per byte; in tokens it depends on the chip.)',
];
