// Long expected strings for cluster-topology (storyboard §5, §6): the page test and the e2e spec import them once.
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

// storyboard §6 "Check my work" (default state: tensor, degree 8, H100 HGX, inside)
export const CHECK_WORK = [
  'bytes per GPU, one layer, full step = 4 × ring all-reduce of 2,048 × 12,288 × 2 B over 8 GPUs = 352.32 MB',
  'FLOPs per GPU, one layer, full step = 72 × 2,048 × 12,288² ÷ 8 = 2,783.1 GFLOP',
  'comm ÷ compute = (352.32 MB ÷ 450 GB/s) ÷ (2,783.1 GFLOP ÷ 989 TFLOPS) = 27.8%',
].join('\n');

// storyboard §6 "Try this": each list item as printed (prompt → Insight: …)
export const TRY_THIS = Object.freeze([
  'Tensor, degree 8, H100 HGX: inside 27.8%, network 250.4%. Raise the degree to 16: it no longer fits in a server, so it runs over the network: 536.6%. Switch to GB200 NVL72, degree 8, inside: 35.2%; degree 16, inside: 75.4%. → Insight: tensor parallelism belongs inside the fastest domain and stays small (around 8–16), and newer chips make it harder, because compute grew faster than NVLink.',
  'Data, degree 64, over the network: 5.0% at 262,144 tokens per replica; slide tokens down to 16,384: 79.2%. Then pipeline, degree 16, network: 1.5%. → Insight: data-parallel syncs are cheap only when each replica processes many tokens per step; since they also overlap with the backward pass, they can live on the slowest, outermost links.',
  'Expert on H100 HGX: inside 35.8%, network 321.9%; the "link needed" readout says 161 GB/s. Switch to GB200 NVL72: needed 407 GB/s; inside 45.2%, network 406.9%. → Insight: expert all-to-all hides only inside the NVLink domain, which is why expert groups stay inside a server or a 72-GPU rack (and why DeepSeek-V3 capped each token at 4 servers, frame 8).',
]);

// the line above the controls (storyboard §6)
export const BASIS_LINE = 'One full training step (forward and backward). GPT-3\'s shape (d_model 12,288, 96 blocks), compute at the chip\'s dense BF16 peak. Real kernels reach 35–55% of peak (scale-reliability), so real ratios are smaller, but their order is the same.';
