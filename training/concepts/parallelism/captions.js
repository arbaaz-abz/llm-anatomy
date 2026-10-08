// parallelism storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps;
// frames.js labels each stage with them. Kept apart so the two never import each other.
export const CAPTIONS = Object.freeze([
  'A training step can be cut by data, inside each matrix, between blocks, along the sequence or between experts. A cut\'s degree is how many GPUs share it.',
  'Data parallelism gives each GPU a full copy and different sequences. Once per step, an all-reduce averages their gradients so every copy takes the same update.',
  'Tensor parallelism cuts each weight matrix between GPUs. Cut the MLP\'s first matrices by columns and each GPU computes half of the hidden numbers, without talking.',
  'Cut the second matrix by rows and each GPU holds a partial sum of the output. An all-reduce adds them inside every layer before anything downstream can start.',
  'Pipeline parallelism gives each GPU a run of consecutive blocks, called a stage. In the forward pass only activations cross a stage boundary, from one GPU to the next.',
  'Split the batch into micro-batches so stages can work at the same time. Stage 2 starts micro-batch 1 while stage 1 moves on to micro-batch 2.',
  'Stages still wait while the pipeline fills and drains. That idle time is the bubble: with 3 stages and 4 micro-batches, a third of every GPU\'s time.',
  'The 1F1B schedule starts each backward as soon as it can. The bubble stays the same, but stage 1 never holds more than 3 micro-batches of activations.',
  'Context parallelism splits one long sequence between GPUs. In the forward pass, later tokens need the keys and values of earlier ones, so those travel one way.',
  'Expert parallelism puts different experts on different GPUs. Each token is sent to its experts and back, an all-to-all twice per MoE layer, and the busiest GPU sets the pace.',
  'Real runs stack these cuts, and the GPU count is the product of their degrees. Llama 3.1 405B used 8-way tensor, 16-way pipeline and 64-way data parallelism: 8,192 GPUs.',
]);
