// cluster-topology's hand-authored stand-in numbers (storyboard §4–§6): the two systems, GPT-3's shape and the degrees the
// frames draw. The page test pins every system figure to data/hardware.json and data/models.json (P3-R13 style).
// Ratios, bytes and FLOPs are never typed: they come from math/topology.js, math/parallel.js and math/scale.js.
import { deepFreeze } from '@math/core.js';

// Peak in dense BF16 TFLOPS; links per GPU, each way, in GB/s. hwId / networkId name the data entries.
export const SYSTEMS = deepFreeze({
  h100: { id: 'h100', label: 'H100 HGX', hwId: 'h100', networkId: 'network-400g', peakTflops: 989, nvlinkGBps: 450, networkGBps: 50, domain: 8 },
  gb200: { id: 'gb200', label: 'GB200 NVL72', hwId: 'gb200-nvl72', networkId: 'network-800g', peakTflops: 2500, nvlinkGBps: 900, networkGBps: 100, domain: 72 },
});

// GPT-3's shape (the decoder-anatomy / training-memory preset) with the toy's tokens per micro-batch.
export const GPT3 = deepFreeze({ hidden: 12288, layers: 96, seq: 2048, microBatch: 1, bytesPerNumber: 2 });

export const DEGREES = deepFreeze({ tensor: 8, pipeline: 16, data: 64 });
export const TOKENS_PER_REPLICA = deepFreeze({ default: 262144, low: 16384 });

// Llama 3.1 405B's 8,192-GPU layout (frame 6) and Meta's 24K cluster (frame 7).
export const LLAMA3_LAYOUT = deepFreeze({ tp: 8, pp: 16, dp: 64, order: ['tensor', 'context', 'pipeline', 'data'] });
export const META = deepFreeze({ gpusPerRack: 16, racksPerPod: 192, gpusPerPod: 3072, pods: 8, oversubscription: '1:7', perGpuGbps: 400 });

// DeepSeek-V3 (frame 8): stated effective rates, direction not given.
export const DEEPSEEK_V3 = deepFreeze({ nvlinkGBps: 160, ibGBps: 50, maxNodesPerToken: 4 });
export const RAIL = deepFreeze({ servers: 4, gpuIndex: 3, expertGpuIndex: 6 }); // 1-based on screen

// The lanes (frame 3 and the toy): compute is 100 units drawn 120 px wide; a longer comm lane is cut at the 520 px track.
export const LANES = deepFreeze({ computeUnits: 100, pxPerUnit: 1.2, trackPx: 520 });
export const CAP_UNITS = LANES.trackPx / LANES.pxPerUnit; // 433.33…, i.e. 4.33× compute

// GB200 NVL72 (frame 10): GPUs and Grace CPUs in the one NVLink domain, and the rack's NVLink total in TB/s
// (the page test pins rackNvlinkTbps to hardware.json gb200-nvl72.rack_nvlink_tbps).
export const NVL72 = deepFreeze({ gpus: 72, graceCpus: 36, rackNvlinkTbps: 130 });
