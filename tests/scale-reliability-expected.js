// Expected strings for scale-reliability: the storyboard's captions (§5, verbatim) and the toy's fixed texts.
// The page test and the e2e spec import this; page code never does.
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

// The toy's default state (Llama 3.1 405B preset, MFU 40%, best interval): storyboard §5 frame 10 and §6 try-this 1.
export const DEFAULT_READOUTS = Object.freeze({
  flops: '3.79 × 10²⁵', 'useful-hours': '26.62M', mtbf: '3.09 h', interval: '13.6 min', loss: '9.0%',
  'gpu-hours': '29.24M', days: '74.4 days', cost: '$58.5M', 'run-average-mfu': '36.4% of the H100 BF16 peak',
});
export const LOSS_SPLIT = 'save 3.7% · lost work 3.7% · restarts 1.6%';
export const RECORD_LLAMA = 'Model card: 30.84M GPU-hours. This run: 29.24M (−5.2%). The card covers more stages than 6ND counts.';
export const RECORD_LLAMA_MFU_38 = 'Model card: 30.84M GPU-hours. This run: 30.78M (−0.2%). The card covers more stages than 6ND counts.';
export const RECORD_DEEPSEEK = 'Reproduces 2.664M by construction (MFU while training set from the record): useful 2.589M GPU-hours plus a 2.8% failure tax is 2.664M.';
export const NO_RECORD = 'No record to compare: this is a hypothetical run.';
export const INVALID = "outside the formula's range: interval plus restart exceed half the cluster MTBF";
export const DEEPSEEK_NOTE = 'H100 peaks stand in for the H800 (its dense peaks are not in the data). DeepSeek ran its matmuls in FP8; against the FP8 peak the run-average is 17.3%.';

// Storyboard §6 "Try this", with the figures its reproducer prints (durations through formatDuration).
export const TRY_THIS = Object.freeze([
  "Llama 3.1 preset at MFU 40%: 29.24M GPU-hours, 74.4 days, against the model card's 30.84M. Slide MFU to 38%: 30.78M.",
  'Keep everything and raise GPUs 16,384 → 32,768 → 100,000: 74.4 days → 39.2 days → 15.4 days, but the loss grows 9.0% → 13.6% → 28.0% and GPU-hours 29.24M → 30.81M → 36.97M.',
  'At 16,384 GPUs set the interval to 5 min (loss 13.0%), 60 min (18.6%), then "best" (13.6 min, 9.0%). Halve the save time to 15 s: best becomes 9.63 min and the loss 6.8%; at 100,000 GPUs the same change takes 28.0% to 22.7%.',
]);
