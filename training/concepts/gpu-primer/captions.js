// gpu-primer storyboard §5 captions, verbatim (one per frame); content.js puts them on the steps, frames.js labels each stage.
// Caption 11 says "6-fold" / "3.56-fold" for the storyboard's "6×" / "3.56×" (no operators in captions: README lesson 2).
export const CAPTIONS = Object.freeze([
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
]);
