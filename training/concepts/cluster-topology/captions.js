// The ten captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'Inside a server, eight GPUs talk through NVLink switches at 450 GB/s each way on H100s. That fast island is called the scale-up domain.',
  'Between servers, each GPU gets one network port: 50 GB/s each way, nine times slower than NVLink. That second layer is the scale-out network.',
  'Tensor parallelism all-reduces inside every layer, and the next layer waits. Over NVLink that takes 28% as long as the math; over the network, two and a half times longer.',
  'Data parallelism syncs gradients once per step, and the sync can overlap with the backward pass that produces them. Over the network it costs 5% of the compute time here.',
  'Pipeline stages hand over one activation per micro-batch at each boundary, and its gradient comes back. Even over the network that is about 1.5% of the compute.',
  'So the cut that blocks compute gets the fastest link. Llama 3 kept tensor parallelism inside each 8-GPU server and put pipeline, then data parallelism, on the network.',
  'Meta\'s 24,576-GPU Llama 3 cluster has full bandwidth only within pods of 3,072 GPUs. Above them, links are oversubscribed seven to one, so chatty traffic stays inside a pod.',
  'In a rail-optimized network, GPU 3 of every server shares one switch. DeepSeek sends each token over the network to the same-numbered GPU, then over NVLink to its expert.',
  'DeepSeek-V4 states when expert traffic can hide behind compute: roughly 1 GB/s of link for every 6 TFLOPS. An H100 needs 161 GB/s; the network gives 50.',
  'A GB200 NVL72 rack puts 72 GPUs in one NVLink domain, nine servers\' worth. Expert groups bigger than 8 can now stay off the network.',
]);
