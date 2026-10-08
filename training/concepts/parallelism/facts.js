// parallelism's dated text (pure, no DOM): the §8 rows, the framing paragraph and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; rows keep their placeholders so the
// scaffold adds each row's source link and "reported" chip.

export const FRAMING = 'Real runs combine several cuts, and the published configurations below show which.';

// The 8 rows of storyboard §8, in order.
export function factRows() {
  return [
    { claim: 'Llama 3.1 405B ({llama-3.1-405b.release_date|year} paper): {llama-3.1-405b.parallelism}.' },
    { claim: 'Llama 3 places parallelism innermost to outermost as {llama-3.1-405b.parallelism_order}: "innermost parallelism requires the highest network bandwidth" (details in [[cluster-topology]]).' },
    { claim: 'Llama 3 used all-gather context parallelism ({llama-3.1-405b.cp_long_context}-way) only in the final long-context stage, since attention FLOPs dwarf the all-gather and GQA keeps K and V small.' },
    { claim: 'DeepSeek-V3 ({deepseek-v3.release_date|year}): {deepseek-v3.training_gpus} H800s, {deepseek-v3.parallelism}; {deepseek-v3.pipeline_schedule} schedule; communication kernels use only {deepseek-v3.comm_sms} SMs.' },
    { claim: 'DualPipe feeds micro-batches from both ends of the pipeline and overlaps all-to-all and pipeline traffic with compute; it keeps two copies of the parameters. {deepseek-v3.pipeline_schedule|cite}' },
    { claim: 'Kimi K2 ({kimi-k2.release_date|year}): {kimi-k2.parallelism}; trainable on any multiple of 32 nodes, minimum 256 GPUs.' },
    { claim: 'Kimi K3: pipeline with virtual stages, MoonEP expert parallelism (dynamic redundant experts for perfect load balance), ZeRO-1 plus "Pipeline ZeRO-2", context parallelism that passes the linear-attention state ({kimi-k3.parallelism}).' },
    { claim: 'DeepSeek-V4 ({deepseek-v4-pro.release_date|year}) fuses dispatch, expert GEMMs and combine into one mega-kernel so expert traffic hides behind compute: {deepseek-v4-pro.ep_kernel} (the condition is in [[cluster-topology]]).' },
  ];
}

// Page text under the stage, one list per frame: the dated notes (filled from data) and the frame → lesson hand-offs.
export const BELOW = Object.freeze([
  ['Where each cut runs on the network is the next lesson: [[cluster-topology]].'],
  ['ZeRO shards this state instead of copying it: [[training-memory]].'],
  ['The matrices cut here are the MLP\'s, from [[decoder-anatomy]]; the course toy has d_model 8 and hidden size 16.'],
  [],
  ['Llama 3.1 405B ({llama-3.1-405b.release_date|year} paper) cut its blocks into {llama-3.1-405b.parallelism_order} order from the inside out; DeepSeek-V3 ({deepseek-v3.release_date|year}) used {deepseek-v3.pipeline_schedule}, which the table below describes.'],
  [],
  [],
  ['Zero-bubble and DualPipe schedules shrink the bubble itself; their formulas are in the math panel and their sources in the table.'],
  ['Keys, values and the causal mask are from [[attention]].'],
  ['The router and its toy routes are from [[moe]]. The same expert parallelism serves requests in [[disaggregation]].'],
  ['Llama 3.1 405B ({llama-3.1-405b.release_date|year} paper): {llama-3.1-405b.parallelism}. DeepSeek-V3 ({deepseek-v3.release_date|year}): {deepseek-v3.parallelism}, {deepseek-v3.training_gpus} H800s. Where each cut goes on the network: [[cluster-topology]].'],
]);
