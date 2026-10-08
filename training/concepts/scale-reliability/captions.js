// scale-reliability storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
export const CAPTIONS = Object.freeze([
  'Training costs about 6 FLOPs per parameter per token: 2 in the forward pass, 4 in the backward. For Llama 3.1 405B that matches the paper\'s own total.',
  'Divide by the H100\'s peak and the run needs 10.6 million GPU-hours, 27 days on 16,384 GPUs. Meta\'s model card reports 30.8 million GPU-hours.',
  'Model FLOPs utilization, MFU, is the share of peak a run actually delivers. Llama 3.1 ran at 38 to 43% while training, so most of the gap is utilization.',
  'Exposed communication is time GPUs spend waiting on other GPUs. A step that computes for 100 ms, then waits 40 ms, keeps its tensor cores busy 71% of the time.',
  'Overlap hides it: send one layer\'s gradients while computing the next. DeepSeek-V3 set aside 20 of each GPU\'s 132 SMs to run communication alongside the math.',
  'DeepSeek-V3 ran its big matrix multiplies in FP8, with one scale per 128 numbers so an outlier spoils only its own tile. Its loss stayed within 0.25% of BF16.',
  'With FP8, utilization depends on which peak you divide by. Behemoth\'s 390 TFLOPS per GPU is 39% of an H100\'s BF16 peak, or 20% of its FP8 peak.',
  'Llama 3.1\'s 16,384 GPUs hit 419 unexpected interruptions in 54 days, one every 3.1 hours. The cluster\'s mean time between failures, MTBF, is one GPU\'s divided by the GPU count.',
  'Here the run saves a checkpoint every few minutes. A failure loses the work since the last save plus the restart; saving too often loses time to the saves.',
  'With 100,000 of the same GPUs, failures come every 30 minutes and checkpointing loses 28% of the run. Faster saves and restarts now matter as much as faster chips.',
  'The bill is GPU-hours times the price of an hour. DeepSeek priced V3\'s 2.79 million H800-hours at $2 each: $5.6 million, for the final run alone.',
]);
