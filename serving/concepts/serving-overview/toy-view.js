// serving-overview toy view model (pure, no DOM): state + data → every string the toy prints and the bar's geometry.
// Times come from requestTimeline (math/serving.js) on RUNNING_EXAMPLE; the H200's memory from hbmFor, printed with its basis.
import { formatBytes, formatCount, formatDuration, formatInt } from '@math/core.js';
import { requestTimeline, freeHbmPerGpu, maxUsersPerGpu, hbmFor, stepTime, RUNNING_EXAMPLE } from '@math/serving.js';
import { kvCacheBytes } from '@math/memory.js';
import { formatShare } from '@shared/glyphs.js';
import { decodeRate, DECODE_CONTEXT } from './numbers.js';
import { INITIAL_STATE, checkWork, prefillRule, timelineGeometry } from './format.js';

export { INITIAL_STATE };
export const FIGURE_WIDTH = 440; // px for the whole request: the bar is drawn to scale, full width = total time
export const GPU_ID = 'h200';

export function hardwareEntry(data, id = GPU_ID) {
  const entry = data?.hardware?.entries?.find((e) => e.id === id);
  if (!entry) throw new RangeError(`serving-overview toy: data/hardware.json has no ${id}`);
  return entry;
}

// The "sharing" chip: as many DECODE_CONTEXT-token caches as fit beside the weights on one H200 (Prefill vs decode).
export function sharedUsers(hbmBytes) {
  const free = freeHbmPerGpu({ hbmBytes, weightBytes: RUNNING_EXAMPLE.weightBytesPerGpu, gpus: 1 });
  return maxUsersPerGpu(free, kvCacheBytes({ bytesPerToken: RUNNING_EXAMPLE.kvBytesPerToken, tokens: DECODE_CONTEXT }));
}

// The two decode-speed chips, their unrounded rates and the note that says where they come from.
export function speeds(data) {
  const hbm = hbmFor(hardwareEntry(data));
  const users = sharedUsers(hbm.bytes);
  const rates = Object.freeze({ alone: decodeRate(1), shared: decodeRate(users) });
  return {
    users,
    rates,
    options: [
      { value: 'alone', label: `${formatCount(rates.alone)} tok/s, alone on the GPU` },
      { value: 'shared', label: `${formatCount(rates.shared)} tok/s, sharing with ${formatInt(users - 1)} others` },
    ],
    note: `Output tokens per second for one user: one token per decode step, so 1 ÷ the step time from [[prefill-decode]] at ${formatInt(DECODE_CONTEXT)} tokens of context. `
      + `Sharing means ${formatInt(users)} users, as many ${formatInt(DECODE_CONTEXT)}-token caches as fit beside the ${formatBytes(RUNNING_EXAMPLE.weightBytesPerGpu)} of weights in an H200's ${formatBytes(hbm.bytes)} (${hbm.basis}).`,
  };
}

export const timelineFor = (state, rates) => requestTimeline({
  queueS: state.queue, promptTokens: state.prompt, model: RUNNING_EXAMPLE, outputTokens: state.output, decodeTokPerS: rates[state.decodeRate],
});

function readouts(state, tl) {
  const { bound } = stepTime({ ...RUNNING_EXAMPLE, tokens: state.prompt, seqs: 0, context: 0 });
  return {
    prefill: { value: formatDuration(tl.prefillS), sub: prefillRule(bound) },
    ttft: { value: formatDuration(tl.ttftS), sub: 'queue + prefill' },
    tpot: { value: formatDuration(tl.tpotS), sub: '1 ÷ decode speed' },
    total: { value: formatDuration(tl.e2eS), sub: 'TTFT + (answer − 1) × TPOT' },
    decodeShare: { value: formatShare(tl.decodeShare), sub: 'after the first token ÷ total' },
  };
}

function figure(state, tl) {
  const geo = timelineGeometry(tl, state.output, FIGURE_WIDTH);
  return {
    ...geo,
    ttftLabel: `TTFT ${formatDuration(tl.ttftS)}`,
    decodeLabel: state.output > 1 ? `after the first token ${formatDuration(tl.e2eS - tl.ttftS)}` : '',
  };
}

export function toyView(state, data) {
  const { rates } = speeds(data);
  const tl = timelineFor(state, rates);
  const fig = figure(state, tl);
  return {
    timeline: tl,
    readouts: readouts(state, tl),
    figure: fig,
    widthNote: `full width = ${formatDuration(tl.e2eS)} (the whole request)`,
    tickNote: fig.thinned ? 'one tick per 10 answer tokens' : 'one tick per answer token',
    checkWork: checkWork(tl, state.output),
  };
}
