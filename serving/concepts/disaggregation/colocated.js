// disaggregation frames 1–3: batching's "What if D's prompt were 4,096 tokens?" branch, timed in milliseconds (pure, no DOM).
// scheduleTokens (math/batching.js) decides what each step carries; stepTime (math/serving.js) says how long it takes.
import { scheduleTokens } from '@math/batching.js';
import { stepTime, RUNNING_EXAMPLE, TOY_REQUESTS } from '@math/serving.js';
import { BRANCH_PROMPT, BRANCH_CONTEXT, TTFT_TARGET_MS, TPOT_TARGET_MS } from './numbers.js';

const MS = 1000;
const SEATS = 3;
export const BRANCH_REQUESTS = Object.freeze(TOY_REQUESTS.map((r) => (r.id === 'D' ? { ...r, prompt: BRANCH_PROMPT } : { ...r })));
export const SHOWN = Object.freeze(['A', 'C', 'D']); // B has finished before the branch (step 2)

// Every step with its start and length in ms: [{ step, decode: [ids], prefill: [{ id, tokens }], tokens, fromMs, ms }].
export function branchSchedule(budget = Infinity) {
  let clock = 0;
  return scheduleTokens({ requests: BRANCH_REQUESTS, seats: SEATS, budget }).map((s) => {
    const ms = stepTime({ ...RUNNING_EXAMPLE, tokens: s.tokens, seqs: s.decode.length, context: BRANCH_CONTEXT }).timeS * MS;
    const timed = { ...s, fromMs: clock, ms };
    clock += ms;
    return timed;
  });
}

export const arrivalMs = (schedule, id) => schedule.find((s) => s.step === BRANCH_REQUESTS.find((r) => r.id === id).arrives)?.fromMs ?? 0;

// One request's lane as { arrivesMs, steps: [{ fromMs, toMs, kind }] }: queue until its first prefill step, then prefill, then decode.
export function laneOf(schedule, id) {
  const arrives = arrivalMs(schedule, id);
  const mine = schedule.flatMap((s) => {
    if (s.prefill.some((p) => p.id === id)) return [{ fromMs: s.fromMs, toMs: s.fromMs + s.ms, kind: 'prefill' }];
    return s.decode.includes(id) ? [{ fromMs: s.fromMs, toMs: s.fromMs + s.ms, kind: 'decode' }] : [];
  });
  const queued = mine[0].fromMs > arrives ? [{ fromMs: arrives, toMs: mine[0].fromMs, kind: 'queue' }] : [];
  return { arrivesMs: arrives, steps: [...queued, ...mine] };
}

export const prefillEndMs = (schedule, id) => laneOf(schedule, id).steps.filter((s) => s.kind === 'prefill').at(-1).toMs;
export const decodeSteps = (schedule, id) => laneOf(schedule, id).steps.filter((s) => s.kind === 'decode');

// Frame 2: time to first token (arrival → the end of the request's prefill) and time per token (its mean decode step).
export function colocatedMetrics(budget) {
  const schedule = branchSchedule(budget);
  const rows = SHOWN.map((id) => {
    const decode = decodeSteps(schedule, id);
    const ttftMs = prefillEndMs(schedule, id) - arrivalMs(schedule, id);
    const tpotMs = decode.reduce((sum, s) => sum + (s.toMs - s.fromMs), 0) / decode.length;
    return { id, ttftMs, tpotMs, okTtft: ttftMs <= TTFT_TARGET_MS, okTpot: tpotMs <= TPOT_TARGET_MS };
  });
  const meeting = rows.filter((r) => r.okTtft && r.okTpot).length;
  return { budget, rows, meeting, total: rows.length, worstTtftMs: Math.max(...rows.map((r) => r.ttftMs)), worstTpotMs: Math.max(...rows.map((r) => r.tpotMs)) };
}

// Frame 3: D's prefill on its own GPU is one 4,096-token forward pass; A and C keep the decode step of the running example.
export function separatedTimes() {
  const prefillMs = stepTime({ ...RUNNING_EXAMPLE, tokens: BRANCH_PROMPT, seqs: 0, context: 0 }).timeS * MS;
  const decodeMs = stepTime({ ...RUNNING_EXAMPLE, tokens: 2, seqs: 2, context: BRANCH_CONTEXT }).timeS * MS;
  return { prefillMs, decodeMs, stepsLong: prefillMs / decodeMs };
}
