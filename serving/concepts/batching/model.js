// batching lane model (pure, no DOM): the scenario's requests, per-step schedules and durations, and the lane rows
// (seat, prefill span, decode ticks, idle hold, queue wait) every frame and the toy draw. Step counts come from
// math/batching.js; step times from math/serving.js stepTime. Inputs are never mutated; the same input gives a deep-equal output.
import { simulateStatic, simulateContinuous, scheduleTokens } from '@math/batching.js';
import { stepTime, RUNNING_EXAMPLE, TOY_REQUESTS } from '@math/serving.js';

export const SEATS = 3;
export const CONTEXT = 16; // batching §6: step times on the running example at a 16-token context
export const LONG_PROMPT = 4096;
export const SHORT_PROMPT = 6; // D's prompt in TOY_REQUESTS
export const BUDGETS = Object.freeze([null, 2048, 512]); // null = chunked prefill off
const MS = 1e3;

// The four toy requests with the toy's two editable lengths: C's answer and D's prompt.
export function requestsFor({ cOutput = 6, dPrompt = SHORT_PROMPT } = {}) {
  return TOY_REQUESTS.map((r) => {
    if (r.id === 'C') return { ...r, output: cOutput };
    if (r.id === 'D') return { ...r, prompt: dPrompt };
    return { ...r };
  });
}

export const simulate = (kind, requests, seats) => (kind === 'static' ? simulateStatic({ requests, seats }) : simulateContinuous({ requests, seats }));

// The per-step work of a simulated lane, one entry per step 0..lastStep: ids decoding (one token each) and prompts prefilled
// whole in their admission step. For continuous with budget Infinity this equals scheduleTokens (tested).
export function scheduleFromSim(sim, requests) {
  const prompts = new Map(requests.map((r) => [r.id, r.prompt]));
  return Array.from({ length: sim.steps }, (_, step) => {
    const decode = sim.live.filter((r) => r.admitted < step && step <= r.finishes).map((r) => r.id);
    const prefill = sim.live.filter((r) => r.admitted === step).map((r) => ({ id: r.id, tokens: prompts.get(r.id) }));
    return { step, decode, prefill, tokens: decode.length + prefill.reduce((sum, p) => sum + p.tokens, 0) };
  });
}

// A step's time in milliseconds on the running example (prefill-only steps have seqs 0).
export function stepMs(entry) {
  if (entry.tokens === 0) return 0;
  return stepTime({ ...RUNNING_EXAMPLE, tokens: entry.tokens, seqs: entry.decode.length, context: CONTEXT }).timeS * MS;
}

export const durationsMs = (schedule) => schedule.map(stepMs);

// Step boundaries in axis units: [0, d0, d0 + d1, ...]. In step mode every duration is 1.
export function edgesFrom(durations) {
  return durations.reduce((edges, d) => [...edges, edges.at(-1) + d], [0]);
}

export const unitDurations = (schedule) => schedule.map(() => 1);

// Seats, first free wins. A request holds its seat from `admitted` up to `release` (an exclusive step index).
function assignSeats(rows, seats) {
  const taken = [];
  return [...rows].sort((a, b) => a.admitted - b.admitted || a.arrivalOrder - b.arrivalOrder).map((row) => {
    const seat = Array.from({ length: seats }, (_, s) => s).find((s) => taken.every((t) => t.seat !== s || t.release <= row.admitted));
    if (seat === undefined) throw new RangeError(`assignSeats: no free seat for ${row.id} at step ${row.admitted}`);
    taken.push({ seat, release: row.release });
    return { id: row.id, seat };
  });
}

function stepsOf(schedule, id) {
  const prefill = schedule.filter((e) => e.prefill.some((p) => p.id === id)).map((e) => e.step);
  const decode = schedule.filter((e) => e.decode.includes(id)).map((e) => e.step);
  return { prefill, decode };
}

// One row per request in axis units (edges maps a step index to its start; edges[steps] is the end of the run).
// kind 'static' holds every seat of a batch until the batch's longest member is done (the idle hold, hatched).
export function laneRows({ schedule, requests, kind, seats, edges }) {
  const base = requests.map((r, arrivalOrder) => {
    const { prefill, decode } = stepsOf(schedule, r.id);
    if (prefill.length === 0 || decode.length !== r.output) throw new RangeError(`laneRows: request ${r.id} is not scheduled (prefill ${prefill.length} steps, ${decode.length} of ${r.output} tokens)`);
    return { id: r.id, arrives: r.arrives, arrivalOrder, admitted: prefill[0], waited: prefill[0] - r.arrives, prefillEnd: prefill.at(-1) + 1, decode, done: decode.at(-1) };
  });
  const batchEnd = (row) => Math.max(...base.filter((o) => o.admitted === row.admitted).map((o) => o.done)) + 1;
  const held = base.map((row) => ({ ...row, release: kind === 'static' ? batchEnd(row) : row.done + 1 }));
  const seatOf = new Map(assignSeats(held, seats).map((s) => [s.id, s.seat]));
  return held.map((row) => ({
    id: row.id,
    seat: seatOf.get(row.id),
    arrives: row.arrives,
    admitted: row.admitted,
    done: row.done,
    release: row.release,
    waited: row.waited,
    prefill: { from: edges[row.admitted], to: edges[row.prefillEnd] },
    ticks: row.decode.map((s) => edges[s + 1]),
    idle: row.release > row.done + 1 ? { from: edges[row.done + 1], to: edges[row.release] } : null,
    queue: row.waited === 0 ? null : { from: edges[row.arrives], to: edges[row.admitted] },
  }));
}

// A lane in step units: the sim, its schedule and rows (edges 0, 1, 2, ...).
export function stepLane({ kind, requests, seats = SEATS }) {
  const sim = simulate(kind, requests, seats);
  const schedule = scheduleFromSim(sim, requests);
  const edges = edgesFrom(unitDurations(schedule));
  return { kind, seats, sim, schedule, edges, rows: laneRows({ schedule, requests, kind, seats, edges }) };
}

// A lane timed in milliseconds. A budget (continuous only) chunks the prefill through scheduleTokens.
// `blend` (0..1) interpolates every step's time from the same schedule with 1-step-short prompts (frame 8's widening step).
export function msLane({ kind, requests, seats = SEATS, budget = null }) {
  const sim = simulate(kind, requests, seats);
  const chunked = kind === 'continuous' && budget !== null;
  const schedule = chunked ? scheduleTokens({ requests, seats, budget }) : scheduleFromSim(sim, requests);
  const durations = durationsMs(schedule);
  const edges = edgesFrom(durations);
  return { kind, seats, sim, schedule, durations, edges, rows: laneRows({ schedule, requests, kind, seats, edges }), totalMs: edges.at(-1) };
}

// Frame 8: the long-prompt schedule (budget off) with every step's time between the short-prompt run (blend 0) and the long one (1).
export function widenedLane({ requests, shortRequests, blend, seats = SEATS }) {
  const long = msLane({ kind: 'continuous', requests, seats });
  const short = msLane({ kind: 'continuous', requests: shortRequests, seats });
  if (short.durations.length !== long.durations.length) throw new RangeError('widenedLane: the short and long runs must have the same steps');
  const durations = long.durations.map((d, i) => short.durations[i] + (d - short.durations[i]) * blend);
  const edges = edgesFrom(durations);
  return { ...long, durations, edges, rows: laneRows({ schedule: long.schedule, requests, kind: 'continuous', seats, edges }), totalMs: edges.at(-1) };
}

// Busy seat-steps completed by step boundary `r` (a number of steps): the step counter of frame 4.
export const busyAt = (sim, r) => sim.live.reduce((sum, l) => sum + Math.max(0, Math.min(l.finishes + 1, Math.floor(r)) - l.admitted), 0);

// Idle seat-steps of a static row, up to boundary `r`.
export const idleAt = (row, r) => (row.idle ? Math.max(0, Math.min(row.idle.to, Math.floor(r)) - row.idle.from) : 0);

// The request that held `row`'s seat just before it (frame 5: "B's seat").
export function previousHolder(rows, row) {
  return rows.filter((o) => o.seat === row.seat && o.release <= row.admitted).sort((a, b) => b.release - a.release)[0]?.id ?? null;
}

// Per-step times of a lane's schedule, the longest step and the request-level times of storyboard §6 (milliseconds).
export function timing(lane, id) {
  const row = lane.rows.find((r) => r.id === id);
  const stepsOfId = lane.schedule.filter((e) => e.decode.includes(id)).map((e) => e.step);
  return {
    doneMs: lane.edges[row.done + 1],
    firstTokenMs: row.prefill.to,
    longestGapMs: Math.max(...stepsOfId.map((s) => lane.durations[s])),
  };
}

export const longestStepMs = (lane) => Math.max(...lane.durations);
