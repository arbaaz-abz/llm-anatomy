// cluster-topology's dated text (pure, no DOM): the hook, the intuition, the §8 rows and framing, and the notes under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/*.json; the few figures computed from them (a doubled
// both-directions total, a ratio) are built here from math/ and ride inside the claim, so the scaffold still adds each row's source.
import { formatRatio } from '@math/core.js';
import { sharePct } from '@math/memory.js';
import { tpCommRatio, ppCommRatio, dpCommRatio, epCommRatio, epMinLinkGBps } from '@math/topology.js';
import { lookupFact } from '@shared/claims.js';
import { GPT3, META, SYSTEMS, DEGREES, TOKENS_PER_REPLICA } from './numbers.js';
import { systemFromData } from './toy-view.js';
import { int, pct0 } from './format.js';

const MISSING = '—';
const BOTH_DIRECTIONS = 2; // NVIDIA quotes NVLink counting both directions: twice the each-way figure

const fact = (data, dataset, id, key) => lookupFact(data?.[dataset], id, key)?.value ?? null;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
// "nine times" for a whole ratio up to ten (the storyboard's wording), else the 3-significant-figure ratio.
const timesWord = (x) => (Number.isInteger(x) && x <= 10 ? `${WORDS[x]} times` : formatRatio(x).replace('×', ' times'));
const ready = (...values) => values.every((v) => v != null);
const doubled = (v) => (Array.isArray(v) ? v.map((x) => int(x * BOTH_DIRECTIONS)).join('–') : int(v * BOTH_DIRECTIONS));

function ratios(data) {
  const h100 = ready(fact(data, 'hardware', 'h100', 'bf16_dense_tflops'), fact(data, 'hardware', 'h100', 'nvlink_gb_s_each_way'), fact(data, 'hardware', 'network-400g', 'gb_s_each_way'))
    ? systemFromData('h100', data) : null;
  if (!h100) return null;
  const tp = (link) => tpCommRatio({ tp: DEGREES.tensor, hidden: GPT3.hidden, peakTflops: h100.peakTflops, linkGBps: link });
  return {
    tpNvlink: tp(h100.nvlinkGBps),
    tpNetwork: tp(h100.networkGBps),
    pp: ppCommRatio({ pp: DEGREES.pipeline, hidden: GPT3.hidden, layers: GPT3.layers, peakTflops: h100.peakTflops, linkGBps: h100.networkGBps }),
    dp: (tokens) => dpCommRatio({ dp: DEGREES.data, tokensPerReplica: tokens, peakTflops: h100.peakTflops, linkGBps: h100.networkGBps }),
    nineTimes: h100.nvlinkGBps / h100.networkGBps,
    h100,
  };
}

export const HOOK = 'Llama 3.1 405B\'s 16,384 GPUs could have been wired as one flat network. Why does it matter which GPUs share a server, and which parallelism runs where?';

export function intuition(data) {
  const r = ratios(data);
  const both = r ? doubled(r.h100.nvlinkGBps) : MISSING;
  const nine = r ? timesWord(r.nineTimes) : MISSING;
  const tpNvlink = r ? pct0(r.tpNvlink) : MISSING;
  const pp = r ? `${sharePct(r.pp, 1).toFixed(1)}%` : MISSING;
  return [
    `A GPU cluster is not one network but two. Inside a server, or inside a rack-scale system like NVL72, GPUs connect through NVLink switches: on H100s, ${both} GB/s per GPU counting both directions. `
      + `Between servers, each GPU gets one network port: {hw:network-400g.gbps} Gb/s on H100-era clusters, which is {hw:network-400g.gb_s_each_way} GB/s each way, ${nine} less than NVLink's {hw:h100.nvlink_gb_s_each_way}. Every byte of communication runs on one of those two layers.`,
    `The parallelisms from [[parallelism]] send very different traffic. Tensor parallelism all-reduces partial sums inside every layer, and the next layer can't start until they arrive. For GPT-3's shape that traffic costs ${tpNvlink} of the compute time over NVLink and two and a half times the compute time over the network (counted over a full forward-and-backward step). `
      + `Data parallelism sends a big gradient sync, but once per step, while the backward pass is still running, so it can hide. Pipeline parallelism hands one activation across each stage boundary per micro-batch (and its gradient back), about ${pp} of the compute even over the network. `
      + 'So the rule is: the chattiest cut, the one that blocks compute, gets the fastest link. That is the hook\'s answer: Llama 3 kept tensor parallelism inside each 8-GPU server and put pipeline and data parallelism on the network, data outermost.',
    'Rack-scale systems widen the fast island. A GB200 NVL72 puts 72 GPUs on one NVLink domain, nine servers\' worth, which matters most for mixture-of-experts training: its all-to-all can hide behind compute inside the rack but not across the network. The cost is that everything outside the island still runs at network speed, and big clusters thin the network further between pods.',
  ];
}

export const FRAMING = 'Link speeds here are per GPU and each way, as the data stores them; NVIDIA\'s own figures usually count both directions, so they look twice as large. Rows marked reported are derived from vendor pages rather than stated by the vendor, and the network thins further above the pod.';

// The 11 rows of storyboard §8, in order. Rows keep their placeholders (source link, "reported" chip); computed figures are literal.
export function factRows(data) {
  const nvlink = (id) => fact(data, 'hardware', id, 'nvlink_gb_s_each_way');
  const totals = [nvlink('h100'), nvlink('b200'), nvlink('rubin')].every((v) => v != null)
    ? `${doubled(nvlink('h100'))}, ${doubled(nvlink('b200'))} and ${doubled(nvlink('rubin'))}` : MISSING;
  const v3Ratio = fact(data, 'models', 'deepseek-v3', 'nvlink_effective_gb_s') != null
    ? formatRatio(fact(data, 'models', 'deepseek-v3', 'nvlink_effective_gb_s') / fact(data, 'models', 'deepseek-v3', 'ib_gb_s')) : MISSING;
  const perTflops = fact(data, 'models', 'deepseek-v4-pro', 'ep_hiding_flops_per_byte');
  const hides = perTflops == null ? MISSING : (perTflops / 1000).toFixed(1);
  return [
    { claim: `NVLink per GPU, each way: H100 {hw:h100.nvlink_gb_s_each_way} GB/s, B200 and B300 {hw:b200.nvlink_gb_s_each_way} GB/s, Rubin {hw:rubin.nvlink_gb_s_each_way} GB/s (the Rubin sources conflict). NVIDIA publishes the both-directions totals, ${totals} GB/s.` },
    { claim: 'Scale-out per GPU, each way: {hw:network-400g.gbps} Gb/s = {hw:network-400g.gb_s_each_way} GB/s in H100-era clusters; {hw:network-800g.gbps} Gb/s = {hw:network-800g.gb_s_each_way} GB/s (ConnectX-8) on Blackwell, which is reported in general and confirmed for GB300 NVL72.' },
    { claim: 'GB200 NVL72: {hw:gb200-nvl72.scale_up_domain}. GB300 NVL72 has the same domain, {hw:gb300-nvl72.rack_nvlink_tbps} TB/s in total.' },
    { claim: 'Vera Rubin NVL72: {hw:rubin.scale_up_domain}. Rubin Ultra\'s NVL{hw:rubin-ultra.scale_up_gpus} rack is planned for H2 2027 (reported).' },
    { claim: `Meta's Llama 3 cluster ({hw:meta-llama3-cluster.release_date|year} paper): {hw:meta-llama3-cluster.gpus_per_rack} GPUs per rack, {hw:meta-llama3-cluster.racks_per_pod} racks per pod = {hw:meta-llama3-cluster.gpus_per_pod} GPUs with full bisection bandwidth, {hw:meta-llama3-cluster.pods} pods = 24K GPUs, {hw:meta-llama3-cluster.oversubscription} oversubscribed above the pods, {hw:meta-llama3-cluster.per_gpu_gbps} Gb/s per GPU, topology-aware scheduling.` },
    { claim: 'Llama 3 order, innermost to outermost: {llama-3.1-405b.parallelism_order} ("innermost parallelism requires the highest network bandwidth and lowest latency"); the paper calls the model Llama 3, the checkpoint is Llama 3.1 405B.' },
    { claim: `DeepSeek-V3 ({deepseek-v3.release_date|year}): H800s with NVLink {deepseek-v3.nvlink_effective_gb_s} GB/s vs InfiniBand {deepseek-v3.ib_gb_s} GB/s per GPU (${v3Ratio}), DeepSeek's stated effective rates (direction not given; {deepseek-v3.ib_gb_s} GB/s matches a 400 Gb/s port each way). The H800's NVLink is reduced vs the H100 ({hw:h800.nvlink_gb_s_each_way} vs {hw:h100.nvlink_gb_s_each_way} GB/s each way). All-to-all goes over InfiniBand to the same-index GPU, then NVLink; each token reaches at most {deepseek-v3.max_nodes_per_token} nodes.` },
    { claim: `DeepSeek-V4 ({deepseek-v4-pro.release_date|year}): expert traffic hides behind compute when compute ÷ bandwidth is at most {deepseek-v4-pro.ep_hiding_flops_per_byte|int} FLOPs per byte (each GB/s hides about ${hides} TFLOPS).` },
    { claim: 'Kimi K2 ({kimi-k2.release_date|year}) servers: {kimi-k2.node}.' },
    { claim: 'NVIDIA\'s Nemotron 3 RL report: {nemotron-3-ultra.ep_colocation}.' },
  ];
}

// A paragraph under the rows: the claim has no data entry to cite, so it points to the Go deeper link instead of a row.
export const PROSE = Object.freeze([
  'Tensor parallelism stays compute-bound only up to about 8–16-way (Google\'s How to Scale Your Model, listed under Go deeper).',
]);

const pctOne = (ratio) => `${sharePct(ratio, 1).toFixed(1)}%`;

// GB200 NVL72's expert-traffic line (frame 10): the link it needs, and the share on NVLink and on the network port.
function gb200Line(data) {
  const ok = ['bf16_dense_tflops', 'nvlink_gb_s_each_way'].every((k) => fact(data, 'hardware', 'gb200-nvl72', k) != null) && fact(data, 'hardware', 'network-800g', 'gb_s_each_way') != null;
  if (!ok) return MISSING;
  const g = systemFromData('gb200', data);
  const tp16 = tpCommRatio({ tp: 16, hidden: GPT3.hidden, peakTflops: g.peakTflops, linkGBps: g.nvlinkGBps });
  return `On GB200 (${int(g.peakTflops)} TFLOPS BF16) expert traffic needs ${int(epMinLinkGBps({ peakTflops: g.peakTflops }))} GB/s: NVLink at ${int(g.nvlinkGBps)} GB/s each way gives ${pctOne(epCommRatio({ peakTflops: g.peakTflops, linkGBps: g.nvlinkGBps }))}, a network port at ${int(g.networkGBps)} GB/s each way (800 Gb/s, reported) gives ${pctOne(epCommRatio({ peakTflops: g.peakTflops, linkGBps: g.networkGBps }))}. Tensor parallelism of 16 inside the rack spends ${pctOne(tp16)}.`;
}

// Page text under the stage, one list per frame (0-based): dated notes filled from data and the arithmetic behind a frame.
export function belowFor(data, index) {
  const r = ratios(data);
  const both = r ? doubled(r.h100.nvlinkGBps) : MISSING;
  const need = r ? int(epMinLinkGBps({ peakTflops: r.h100.peakTflops })) : MISSING;
  const dpLow = r ? pctOne(r.dp(TOKENS_PER_REPLICA.low)) : MISSING;
  const dpHigh = r ? pctOne(r.dp(TOKENS_PER_REPLICA.default)) : MISSING;
  const domains = SYSTEMS.gb200.domain / SYSTEMS.h100.domain;
  const notes = [
    [`NVIDIA quotes ${both} GB/s per GPU for H100 NVLink, counting both directions; this page prints each way.`],
    ['Each GPU\'s port: {hw:network-400g.gbps} Gb/s ÷ 8 bits per byte = {hw:network-400g.gb_s_each_way} GB/s each way. HBM bandwidth is a separate total and is not on this ladder (see [[gpu-primer]]).'],
    ['Per layer per step, forward and backward: the stage shows the bytes sent and the FLOPs done per GPU for GPT-3\'s shape; the all-reduces are the ones from [[parallelism]].'],
    [`Tokens per replica per step is a stand-in (${int(TOKENS_PER_REPLICA.default)}); the ratio rises to ${dpLow} at ${int(TOKENS_PER_REPLICA.low)} tokens.`],
    [`${int(DEGREES.pipeline)} stages, one per server; the ratio is the same at every boundary.`],
    [`Data parallelism sends more per unit of compute than pipeline here (${dpHigh} against ${r ? pctOne(r.pp) : MISSING}), but its sync overlaps with the backward pass, while pipeline hand-offs sit between stages.`],
    [`Per pod: {hw:meta-llama3-cluster.gpus_per_rack} GPUs per rack × {hw:meta-llama3-cluster.racks_per_pod} racks = {hw:meta-llama3-cluster.gpus_per_pod} GPUs; {hw:meta-llama3-cluster.pods} pods = ${int(META.gpusPerPod * META.pods)} GPUs; {hw:meta-llama3-cluster.per_gpu_gbps} Gb/s per GPU; {hw:meta-llama3-cluster.oversubscription} above the pods.`],
    ['DeepSeek-V3 ({deepseek-v3.release_date|year}, H800): NVLink {deepseek-v3.nvlink_effective_gb_s} GB/s vs network {deepseek-v3.ib_gb_s} GB/s, DeepSeek\'s stated effective rates; the paper gives no direction, and {deepseek-v3.ib_gb_s} GB/s matches a 400 Gb/s port each way. The H800\'s NVLink is reduced vs the H100. Each token reaches at most {deepseek-v3.max_nodes_per_token} servers.'],
    [`${r ? int(r.h100.peakTflops) : MISSING} TFLOPS ÷ {deepseek-v4-pro.ep_hiding_flops_per_byte|int} FLOPs per byte = ${need} GB/s of link needed. Expert traffic is the all-to-all from [[parallelism]].`],
    [`${int(SYSTEMS.gb200.domain)} ÷ ${int(SYSTEMS.h100.domain)} = ${int(domains)} servers' worth of GPUs in one domain. ${gb200Line(data)}`],
  ];
  return notes[index] ?? [];
}
