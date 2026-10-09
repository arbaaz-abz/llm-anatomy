// disaggregation's hand-authored stage constants (storyboard §4–§8). The stage renders without ctx.data, so the figures it
// draws are restated here; tests/disaggregation-page.test.js asserts each equals data/*.json (the P3-R13 pattern).
import { deepFreeze } from '@math/core.js';

export const BRANCH_PROMPT = 4096; // D's prompt in batching's "What if?" branch
export const CHUNK_BUDGET = 512; // the chunked version in frames 1–2
export const BRANCH_CONTEXT = 16; // the step-time context batching uses for its branch timeline
export const GOODPUT_BUDGETS = Object.freeze([Infinity, 512, 256, 128]); // frame 2's token budgets per step, in order
// Illustrative service targets for frame 2 (the page says so on screen): not a published SLO.
export const TTFT_TARGET_MS = 400;
export const TPOT_TARGET_MS = 15;

export const EXPERT_COUNT = deepFreeze({ total: 384, active: 6, columns: 24, rows: 16 }); // DeepSeek-V4-Pro (models.json)
export const EP_SHOWN = 16; // frames 6–7: the EP size and the users per GPU the stage follows
export const USERS_SHOWN = 64;
export const ROUTING_SEED = 20261009; // frame 6's stand-in routing (an even spread, so every expert gets one token)

// V4-Pro on a GB300 NVL72 GPU (the toy's second panel): restated for the stage, read from data by the toy.
export const V4_PRO = deepFreeze({ expertsTotal: 384, expertsActive: 6, checkpointBytes: 865e9, dModel: 7168, expertHidden: 3072 });
export const GB300 = deepFreeze({ hbmBytes: 288e9, peakTflops: 15000, bandwidthTBps: 8, gpus: 72, rackNvlinkTbps: 130 });
export const MLA = deepFreeze({ layers: 61, dLatent: 512, dRope: 64 }); // DeepSeek-V3 (models.json: 70,272 B/token at 2 bytes)
export const LINK_IDS = Object.freeze(['nvlink', 'net800', 'net400']);
export const STAGE_LINKS = deepFreeze({
  nvlink: { label: 'NVLink5', gbPerS: 900 },
  net800: { label: 'network 800 Gb/s', gbPerS: 100 },
  net400: { label: 'network 400 Gb/s', gbPerS: 50 },
});

// Published layouts and gains the stage prints (serving.json; confidence confirmed for every one).
export const VLLM_GB200 = deepFreeze({ prefillGroups: 4, prefillGpusEach: 2, decodeGpus: 8, prefillTokSGpu: 26200, decodeTokSGpu: 10100 });
export const DEEPSEEK = deepFreeze({ prefillEp: 32, decodeEp: 144, routedExperts: 256, redundant: 32, activePerToken: 8, prefillRoutedPerGpu: 9, prefillSharedPerGpu: 1, gpusPerNode: 8 });
export const GB300_GAIN = 2.83; // InferenceX: per-GPU throughput over GB200 at 27 tok/s/user
export const GB300_TOK_S_USER = 27;
// Measurement dates the stage prints (serving.json `date` keys, printed as the `|date` claim format does): pinned by the page test.
export const DATES = deepFreeze({ vllmGb200: '2026-02-03', deepseekProduction: 'Feb 2025', inferencexGb300: '2026-05-22' });
export const PAPERS = deepFreeze({ distserve: { year: '2024', goodputGain: 7.4, sloGain: 12.6 }, splitwise: { year: '2023', throughputGain: 1.4, costCutPct: 20 } });
