// batching pure helpers (no DOM): every string the stage and the toy print. Counts are exact integers (formatInt), shares go
// through formatShare from the unrounded value (R3), durations through formatDuration, tokens per step through formatCount.
import { formatCount, formatDuration, formatInt, formatRatio } from '@math/core.js';
import { formatShare } from '@shared/glyphs.js';

const MS = 1e-3; // milliseconds → seconds for formatDuration
export const ms = (value) => formatDuration(value * MS);
export const share = (part, whole) => formatShare(part / whole);
export const steps = (n) => `${n} step${n === 1 ? '' : 's'}`;

export const laneTitle = (kind, seats) => ({ static: `static batching (${seats} seats)`, continuous: `continuous batching (${seats} seats)`, continuousN: `continuous, ${seats} seats` })[kind];

// "A done step 4 · B done step 2": the requests whose last token is in by boundary t (frame 2).
export function doneLine(rows, t) {
  return rows.filter((r) => r.done + 1 <= t).map((r) => `${r.id} done step ${r.done}`).join(' · ');
}

// "idle seat-steps: B 4, A 2 · D waiting steps 1–6" with the idle counts of frame 3 counting up to boundary r.
export function idleLine(rows, r, idleOf) {
  const held = rows.filter((row) => row.idle).sort((a, b) => (b.idle.to - b.idle.from) - (a.idle.to - a.idle.from) || a.id.localeCompare(b.id));
  const idle = held.map((row) => `${row.id} ${idleOf(row, r)}`).join(', ');
  const waiting = rows.filter((row) => row.waited > 0).map((row) => `${row.id} waiting ${row.waited === 1 ? 'step' : 'steps'} ${row.waited === 1 ? row.arrives : `${row.arrives}–${row.admitted - 1}`}`).join(' · ');
  return [idle ? `idle seat-steps: ${idle}` : '', waiting].filter(Boolean).join(' · ');
}

// Frame 4's counter: "busy 15 of 33 seat-steps", and the share and token count once the lane is complete.
export function busyLine(sim, busy, complete) {
  const base = `busy ${formatInt(busy)} of ${formatInt(sim.seatSteps)} seat-steps`;
  return complete ? `${base} = ${share(sim.busySeatSteps, sim.seatSteps)} · ${formatInt(sim.outputTokens)} tokens in ${formatInt(sim.steps)} steps` : base;
}

// Frame 6 and the 4-seat lane: "last step 10 · busy 57.6% · 1.36 tokens per step".
export const summaryLine = (sim) => `last step ${sim.lastStep} · busy ${share(sim.busySeatSteps, sim.seatSteps)} · ${formatCount(sim.tokensPerStep)} tokens per step`;

// Frame 5: "D admitted step 3 (B's seat)" and, once D is done, " · done step 6".
export function admittedLine(row, holder, t) {
  if (t < row.admitted) return '';
  const seat = holder ? ` (${holder}'s seat)` : '';
  const admitted = `${row.id} admitted step ${row.admitted}${seat}`;
  return row.done + 1 <= t ? `${admitted} · done step ${row.done}` : admitted;
}

// "D 3 → 6" beside the followed bar once it is complete.
export const spanLabel = (row) => `${row.id} ${row.admitted} → ${row.done}`;

// Frame 7: the mixed step.
export function mixedLine({ prefillTokens, decoders, tokens, timeS }) {
  return `${prefillTokens} + ${decoders} = ${tokens} tokens · ${formatDuration(timeS)}`;
}

// Frames 8 and 9: request-level times in milliseconds.
export const slowdown = (longMs, shortMs) => formatRatio(longMs / shortMs).replace('×', '');
export const stepLine = (lane, step) => `step ${step}: ${formatInt(lane.schedule[step].tokens)} tokens, ${ms(lane.durations[step])}`;
export const timesLine = ({ a, c, d }) => `A done at ${ms(a)} · C done at ${ms(c)} · D's first token at ${ms(d)}`;
export const budgetLine = (budget, longestMs) => `budget ${formatInt(budget)} tokens per step: longest step ${ms(longestMs)}`;
export const changedLine = (id, now, was) => `${id} done ${ms(now)} (was ${ms(was)})`;
export const firstTokenLine = (now, was) => `D's first token ${ms(now)} (was ${ms(was)})`;
export const shortNote = (promptTokens, firstMs, otherMs) => `step 0 (${formatInt(promptTokens)} prompt tokens) ${ms(firstMs)}, other short steps ${ms(otherMs)}`;
export const slicesLine = (slices) => `D's prompt in slices: ${slices.map((n) => formatInt(n)).join(', ')}`;
