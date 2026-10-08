// parallelism storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "A training step can be cut by data, inside each matrix, between blocks, along the sequence or between experts. A cut's degree is how many GPUs share it.",
  "Data parallelism gives each GPU a full copy and different sequences. Once per step, an all-reduce averages their gradients so every copy takes the same update.",
  "Tensor parallelism cuts each weight matrix between GPUs. Cut the MLP's first matrices by columns and each GPU computes half of the hidden numbers, without talking.",
  "Cut the second matrix by rows and each GPU holds a partial sum of the output. An all-reduce adds them inside every layer before anything downstream can start.",
  "Pipeline parallelism gives each GPU a run of consecutive blocks, called a stage. In the forward pass only activations cross a stage boundary, from one GPU to the next.",
  "Split the batch into micro-batches so stages can work at the same time. Stage 2 starts micro-batch 1 while stage 1 moves on to micro-batch 2.",
  "Stages still wait while the pipeline fills and drains. That idle time is the bubble: with 3 stages and 4 micro-batches, a third of every GPU's time.",
  "The 1F1B schedule starts each backward as soon as it can. The bubble stays the same, but stage 1 never holds more than 3 micro-batches of activations.",
  "Context parallelism splits one long sequence between GPUs. In the forward pass, later tokens need the keys and values of earlier ones, so those travel one way.",
  "Expert parallelism puts different experts on different GPUs. Each token is sent to its experts and back, an all-to-all twice per MoE layer, and the busiest GPU sets the pace.",
  "Real runs stack these cuts, and the GPU count is the product of their degrees. Llama 3.1 405B used 8-way tensor, 16-way pipeline and 64-way data parallelism: 8,192 GPUs.",
];

export const CHECK_WORK = [
  "bubble  = (4 − 1) ÷ (4 + 4 − 1) = 3 ÷ 7 = 42.9%",
  "GPUs    = 8 × 1 × 16 × 64 = 8,192",
  "per GPU (ZeRO-2): 405B ÷ (8 × 16) = 3.164B parameters",
  "  weights 3.164B × 2 B = 6.33 GB · gradients 3.164B × 2 B ÷ 64 = 0.10 GB",
  "  optimizer 3.164B × 12 B ÷ 64 = 0.59 GB · total 7.02 GB",
].join('\n');

export const TRY_THIS = [
  "GPipe, 4 stages: set micro-batches 4 → 8 → 16 → 32 and read the bubble: 42.9% → 27.3% → 15.8% → 8.6%. Then 8 stages with 32 micro-batches: 17.9%. → Insight: the bubble is (p − 1)/(m + p − 1), so a deep pipeline needs many more micro-batches than stages.",
  "4 stages, 16 micro-batches: GPipe holds 16 micro-batches of activations at stage 1; switch to 1F1B: 4, while the bubble stays 15.8%. → Insight: 1F1B doesn't shrink the bubble; it caps activation memory at p micro-batches, which is what makes large m affordable. Shrinking the bubble itself takes interleaved, zero-bubble or DualPipe schedules, which this page does not draw.",
  "Load the Llama 3.1 \"8K GPUs\" preset: 8,192 GPUs, 7.02 GB of state each with ZeRO-2 (13.25 GB if only optimizer states were sharded). Set all four degrees to 1: one GPU, 6,480 GB. Now set tp = pp = 1 and dp = 8,192 with ZeRO-3: 0.79 GB, the same as tp 8 × pp 16 × dp 64 with ZeRO-3. → Insight: for state alone, ZeRO-3 over all GPUs would do; tensor and pipeline splits are there for what ZeRO cannot touch: activations, and the traffic of gathering every weight from thousands of GPUs every layer.",
];

export const SCHEDULE_NOTE = "Forward and backward each take one time unit here; real backward passes take about twice as long, which changes the bubble's size a little but not its shape.";
