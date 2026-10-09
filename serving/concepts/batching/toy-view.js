// batching toy view model (pure, no DOM): state → the scenario's lanes and every string the toy prints. Unit-tested.
import { formatCount, formatInt } from '@math/core.js';
import { tokensToComputeBound } from '@math/roofline.js';
import { RUNNING_EXAMPLE } from '@math/serving.js';
import {
  requestsFor, stepLane, msLane, SEATS, LONG_PROMPT, SHORT_PROMPT, longestStepMs, timing,
} from './model.js';
import { ms, share, laneTitle } from './format.js';
import { TITLES } from './numbers.js';

export const INITIAL_STATE = Object.freeze({ seats: SEATS, cOutput: 6, dPrompt: SHORT_PROMPT, budget: 0 }); // budget 0 = off
export const TOY_LIMITS = Object.freeze({ seats: [2, 3, 4], cOutput: [2, 10], prompts: [SHORT_PROMPT, LONG_PROMPT], budgets: [0, 2048, 512] });
export const PROMPT_OPTIONS = Object.freeze([{ value: SHORT_PROMPT, label: `${SHORT_PROMPT} tokens` }, { value: LONG_PROMPT, label: `${formatInt(LONG_PROMPT)} tokens` }]);
export const BUDGET_OPTIONS = Object.freeze([{ value: 0, label: 'off' }, { value: 2048, label: formatInt(2048) }, { value: 512, label: '512' }]);
export const TOY_INTRO = `Four hand-picked requests (the same four as in [[paged-attention]]). Steps are timed with Llama-3.1-70B on one H200; memory is not modeled here.`;

export const SUMMARY_NOTE_MS = 'Steps differ in length once D\'s long prompt runs, so the lanes are compared in milliseconds.';
export const REQUESTS_TITLE = 'Per request, in steps';

function checkState({ seats, cOutput, dPrompt, budget }) {
  if (!TOY_LIMITS.seats.includes(seats)) throw new RangeError(`toyView: seats must be one of ${TOY_LIMITS.seats.join(', ')}, got ${seats}`);
  if (!Number.isInteger(cOutput) || cOutput < TOY_LIMITS.cOutput[0] || cOutput > TOY_LIMITS.cOutput[1]) throw new RangeError(`toyView: cOutput must be an integer ${TOY_LIMITS.cOutput.join('–')}, got ${cOutput}`);
  if (!TOY_LIMITS.prompts.includes(dPrompt)) throw new RangeError(`toyView: dPrompt must be ${TOY_LIMITS.prompts.join(' or ')}, got ${dPrompt}`);
  if (!TOY_LIMITS.budgets.includes(budget)) throw new RangeError(`toyView: budget must be one of ${TOY_LIMITS.budgets.join(', ')}, got ${budget}`);
}

// The lanes of a state. With D's short prompt the toy draws steps; with the long prompt it draws milliseconds, and the
// budget (continuous lane only) applies. The timing lane is always in milliseconds.
export function scenario(state) {
  checkState(state);
  const { seats, cOutput, dPrompt } = state;
  const requests = requestsFor({ cOutput, dPrompt });
  const long = dPrompt === LONG_PROMPT;
  const budget = long && state.budget > 0 ? state.budget : null;
  const timed = msLane({ kind: 'continuous', requests, seats, budget });
  const lanes = long
    ? { static: msLane({ kind: 'static', requests, seats }), continuous: timed }
    : { static: stepLane({ kind: 'static', requests, seats }), continuous: stepLane({ kind: 'continuous', requests, seats }) };
  return { requests, mode: long ? 'ms' : 'steps', budget, lanes, timed };
}

// The summary of a lane from its rows: last step, seat utilization, tokens per step (equal to the simulators' for unchunked runs).
export function laneSummary(lane) {
  const lastStep = Math.max(...lane.rows.map((r) => r.done));
  const steps = lastStep + 1;
  const seatSteps = lane.seats * steps;
  const busy = lane.rows.reduce((sum, r) => sum + (r.done - r.admitted + 1), 0);
  const outputs = lane.rows.reduce((sum, r) => sum + r.ticks.length, 0);
  return { lastStep, steps, seatSteps, busy, outputs, tokensPerStep: outputs / steps };
}

const rowOf = (lane, id) => lane.rows.find((r) => r.id === id);

function requestRows(lanes) {
  return ['A', 'B', 'C', 'D'].map((id) => ({
    id,
    cells: ['static', 'continuous'].flatMap((kind) => {
      const row = rowOf(lanes[kind], id);
      return [['admitted', row.admitted], ['done', row.done], ['waited', row.waited]].map(([field, value]) => ({ name: `${kind}-${id}-${field}`, value: String(value) }));
    }),
  }));
}

const KINDS = ['static', 'continuous'];

// With D's long prompt the steps differ in length, so the lanes are compared in time, never in step counts (lesson 25).
function summaryRowsMs(lanes) {
  const cells = (name, at) => KINDS.map((kind) => ({ name: `${kind}-${name}`, value: ms(at(lanes[kind])) }));
  return [
    { label: 'Run ends at', sub: 'end of the last step, on the running example', cells: cells('run-ends', (lane) => lane.edges.at(-1)) },
    { label: 'D done at', cells: cells('d-done', (lane) => timing(lane, 'D').doneMs) },
  ];
}

function summaryRows(lanes, mode) {
  if (mode === 'ms') return summaryRowsMs(lanes);
  const sums = { static: laneSummary(lanes.static), continuous: laneSummary(lanes.continuous) };
  const cell = (field, text) => ['static', 'continuous'].map((kind) => ({ name: `${kind}-${field}`, value: text(sums[kind]) }));
  return [
    { label: 'Last step', cells: cell('last-step', (s) => String(s.lastStep)) },
    { label: 'Seat utilization', sub: 'busy seat-steps ÷ all seat-steps', cells: cell('utilization', (s) => share(s.busy, s.seatSteps)) },
    { label: 'Tokens per step', cells: cell('tokens-per-step', (s) => formatCount(s.tokensPerStep)) },
  ];
}

function timingRows(timed) {
  const a = timing(timed, 'A');
  return [
    { label: 'Longest step', name: 'longest-step', value: ms(longestStepMs(timed)) },
    { label: 'Longest wait between A\'s tokens', name: 'a-gap', value: ms(a.longestGapMs) },
    { label: 'A done', name: 'a-done', value: ms(a.doneMs) },
    { label: 'C done', name: 'c-done', value: ms(timing(timed, 'C').doneMs) },
    { label: 'D\'s first token', name: 'd-first-token', value: ms(timing(timed, 'D').firstTokenMs) },
  ];
}

export function toyView(state) {
  const sc = scenario(state);
  return {
    mode: sc.mode,
    budget: sc.budget,
    showBudget: state.dPrompt === LONG_PROMPT,
    lanes: sc.lanes,
    titles: { static: laneTitle('static', state.seats), continuous: laneTitle('continuous', state.seats) },
    requestRows: requestRows(sc.lanes),
    summaryRows: summaryRows(sc.lanes, sc.mode),
    summaryNote: sc.mode === 'ms' ? SUMMARY_NOTE_MS : '',
    requestsTitle: sc.mode === 'ms' ? `${REQUESTS_TITLE} (steps differ in length)` : REQUESTS_TITLE,
    timingRows: timingRows(sc.timed),
    axisMs: sc.mode === 'ms' ? Math.ceil(Math.max(sc.lanes.static.edges.at(-1), sc.lanes.continuous.edges.at(-1)) / 50) * 50 : null,
    axisSteps: Math.max(sc.lanes.static.edges.length - 1, sc.lanes.continuous.edges.length - 1),
  };
}

// ---- Try this (storyboard §6): numbers computed from the same scenarios the toy shows ----
const at = (state) => scenario({ ...INITIAL_STATE, ...state });
const summaryOf = (lane) => laneSummary(lane);
const dOf = (lane) => rowOf(lane, 'D');
const FLOOR = Math.round(tokensToComputeBound({ peakTflops: RUNNING_EXAMPLE.peakTflops, bandwidthTBps: RUNNING_EXAMPLE.bandwidthTBps, bytesPerElem: RUNNING_EXAMPLE.actBytesPerElem, k: RUNNING_EXAMPLE.dModel, n: RUNNING_EXAMPLE.dModel }));

export function tryThis() {
  const base = at({});
  const [s0, c0] = [base.lanes.static, base.lanes.continuous];
  const longC = at({ cOutput: 10 }).lanes;
  const four = at({ seats: 4 }).lanes;
  const long = at({ dPrompt: LONG_PROMPT });
  const mid = at({ dPrompt: LONG_PROMPT, budget: 2048 });
  const small = at({ dPrompt: LONG_PROMPT, budget: 512 });
  const sum = (lane) => summaryOf(lane);
  const util = (lane) => share(sum(lane).busy, sum(lane).seatSteps);
  return [
    {
      prompt: `Defaults (3 seats). Compare the lanes: static ends at step ${s0.sim.lastStep} with seats busy ${util(s0)} of the time; continuous ends at step ${c0.sim.lastStep} with ${util(c0)}.`,
      insight: 'refilling a seat the step after it frees is the whole trick.',
      rest: ` Same work, ${s0.sim.lastStep - c0.sim.lastStep} fewer steps, and D waits ${dOf(c0).waited} steps instead of ${dOf(s0).waited}.`,
    },
    {
      prompt: `Predict first: what happens to D if C's answer grows from 6 to 10 tokens? Set C's answer length to 10. Static: D now waits until step ${dOf(longC.static).admitted} and is done at step ${dOf(longC.static).done} (seats busy ${util(longC.static)}). Continuous: D still runs steps ${dOf(longC.continuous).admitted}–${dOf(longC.continuous).done}.`,
      insight: 'in a static batch the longest answer sets everyone\'s schedule; in a continuous one it only sets its own.',
      rest: '',
    },
    {
      prompt: `Set the Seats slider to 4. Continuous: D starts at step ${dOf(four.continuous).admitted} and is done at step ${dOf(four.continuous).done}, but seat utilization drops to ${util(four.continuous)}, because nobody else is waiting for that fourth seat. Static with 4 seats: D still waits until step ${dOf(four.static).admitted}.`,
      insight: 'a seat helps only if someone is waiting, and only continuous batching can hand it out mid-run.',
      rest: ` In 2023 engines each seat also cost a full reserved strip of KV memory, which is why seats were scarce (see ${TITLES.pagedAttention}).`,
    },
    {
      prompt: `Set D's prompt to 4,096 tokens. With the token budget off: one ${ms(longestStepMs(long.timed))} step, A done at ${ms(timing(long.timed, 'A').doneMs)}. Budget 2,048: longest step ${ms(longestStepMs(mid.timed))}, A done at ${ms(timing(mid.timed, 'A').doneMs)}. Budget 512: longest step ${ms(longestStepMs(small.timed))}, A done at ${ms(timing(small.timed, 'A').doneMs)}, C at ${ms(timing(small.timed, 'C').doneMs)}, while D's first token moves only from ${ms(timing(long.timed, 'D').firstTokenMs)} to ${ms(timing(small.timed, 'D').firstTokenMs)}.`,
      insight: 'chunked prefill trades a little of the long prompt\'s TTFT for a bounded TPOT for everyone else.',
      rest: ` The smaller the budget, the smoother the streams, until steps become too small to keep the GPU busy (not modeled here; the ${formatInt(FLOOR)}-token crossover in ${TITLES.prefillDecode} is the floor).`,
    },
  ];
}
