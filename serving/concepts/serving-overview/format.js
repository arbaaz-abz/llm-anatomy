// Pure formatters for the serving-overview toy (storyboard §6): its state, slider stops, "Check my work", the
// "a → b" text of the try-this list and the to-scale geometry of the timeline bar. No DOM. Durations print through
// formatDuration, shares through formatShare, integers through formatInt (README lesson 35); each rounds once.
import { formatDuration, formatInt } from '@math/core.js';

export const INITIAL_STATE = Object.freeze({ prompt: 2000, output: 500, queue: 0, decodeRate: 'alone' });
export const PROMPT_STOPS = Object.freeze([3, 100, 500, 2000, 8000, 20_000, 128_000]);
export const OUTPUT_STOPS = Object.freeze([2, 50, 200, 500, 1000, 4000]);
export const QUEUE = Object.freeze({ min: 0, max: 5, step: 0.5 });
export const TICKS_EVERY_TOKEN_UP_TO = 200; // past 200 decode steps the bar draws one tick per 10 tokens
export const TOKENS_PER_THIN_TICK = 10;

const unitOf = (text) => text.split(' ').at(-1);
const numberOf = (text) => text.split(' ').slice(0, -1).join(' ');

// "7.49 → 8.76 s" when both ends share a unit, "141 ms → 1.41 s" when they do not (the storyboard's try-this form).
export function arrowText(fromS, toS) {
  const [a, b] = [formatDuration(fromS), formatDuration(toS)];
  return unitOf(a) === unitOf(b) ? `${numberOf(a)} → ${b}` : `${a} → ${b}`;
}

// The two "Check my work" lines, templated from requestTimeline's unrounded output (each value rounded once).
export function checkWork(timeline, outputTokens) {
  const { queueS, prefillS, ttftS, tpotS, e2eS } = timeline;
  const steps = outputTokens - 1;
  const ttft = formatDuration(ttftS);
  return [
    `TTFT = queue + prefill = ${formatDuration(queueS)} + ${formatDuration(prefillS)} = ${ttft}`,
    `total = TTFT + (n − 1) · TPOT = ${ttft} + ${formatInt(steps)} · ${formatDuration(tpotS)} (${formatDuration(steps * tpotS)}) = ${formatDuration(e2eS)}   (each value rounded once, from the unrounded function output)`,
  ].join('\n');
}

// The fixed prefill rule (storyboard §6 `prefill` readout) and which side of it sets this prompt's time.
export const prefillRule = (bound) => `one read of the weights or the math, whichever is longer (here: ${bound === 'compute' ? 'the math' : 'the weight read'})`;

function decodeSpans(decodeSteps) {
  const per = decodeSteps > TICKS_EVERY_TOKEN_UP_TO ? TOKENS_PER_THIN_TICK : 1;
  return Array.from({ length: Math.ceil(decodeSteps / per) }, (_, k) => [k * per, Math.min((k + 1) * per, decodeSteps)]);
}

// The timeline bar drawn to scale: `width` px is the whole request (e2eS). Steps are G.request `steps` (px from the bar's
// left end): an optional queue, the prefill, then one decode step per token (one per 10 tokens past 200 steps).
export function timelineGeometry(timeline, outputTokens, width) {
  if (!(width > 0)) throw new RangeError(`timelineGeometry: width must be > 0, got ${width}`);
  if (!Number.isInteger(outputTokens) || outputTokens < 1) throw new RangeError(`timelineGeometry: outputTokens must be a positive integer, got ${outputTokens}`);
  const px = width / timeline.e2eS;
  const queueX = timeline.queueS * px;
  const ttftX = timeline.ttftS * px;
  const decodeSteps = outputTokens - 1;
  const spans = decodeSpans(decodeSteps);
  const at = (step) => (step === decodeSteps ? width : ttftX + step * timeline.tpotS * px);
  const steps = [
    ...(queueX > 0 ? [{ from: 0, to: queueX, kind: 'queue' }] : []),
    { from: queueX, to: ttftX, kind: 'prefill' },
    ...spans.map(([a, b]) => ({ from: at(a), to: at(b), kind: 'decode' })),
  ];
  return { steps, queueX, ttftX, width, ticks: spans.length, thinned: decodeSteps > TICKS_EVERY_TOKEN_UP_TO };
}
