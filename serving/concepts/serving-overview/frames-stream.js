// Frames 5–8: the first token streams back (TTFT), decode steps add one token each (TPOT), the end token stops the
// request (storyboard §5). Pure functions of p; each frame starts where the previous one ended.
import { formatCount, formatDuration, formatInt } from '@math/core.js';
import { PROMPT, ANSWER, PREFILL_S, TPOT_S, RATE_ALONE, CHAT_TIMELINE, CHAT, DECODE_CONTEXT, TITLES } from './numbers.js';
import {
  CHOSEN, GPU, KV, seg, ease, leaving, center, outX, layer,
  pipeline, cacheTiles, yourBar, barSteps, ttftBracket, tpotBracket, streamBack, answerRow, answerChip, intoGpu, outOfGpu, note, label,
} from './stage.js';

const PICK_NOTE = 'picks = samples from the model\'s probabilities (Picking the next token)';
const PREFILL_NOTE = `${PROMPT.length} tokens in, ${PROMPT.length} K/V tiles written, 1 token out`;
export const TTFT_NOTE = `this ${PROMPT.length}-token prompt: TTFT ${formatDuration(PREFILL_S)}, one full read of the weights`;
const CHAT_NOTE = `(a ${formatInt(CHAT.prompt)}-token prompt: ${formatDuration(CHAT_TIMELINE.prefillS)})`;
const STEP_NOTE = `step 1: 1 token in, 1 tile, 1 token out · cache ${PROMPT.length + 1} tokens`;
export const TPOT_NOTE = `TPOT ${formatDuration(TPOT_S)} (${formatCount(RATE_ALONE)} tokens/s; one user, ${formatInt(DECODE_CONTEXT)} tokens of context)`;
export const END_NOTE = `your request: ${PROMPT.length} prompt + ${ANSWER.length - 1} generated tokens in the cache at the end`;
const readsNote = (cached) => `reads all weights + ${cached} cached tokens`;
const GPU_NOTE_Y = 100;
const gpuNote = (svg, cached, opacity) => label(svg, center(GPU), GPU_NOTE_Y, readsNote(cached), { anchor: 'middle', opacity });

// The out slot's token rising back into the GPU for the next decode step.
const feedBack = (svg, i, t) => intoGpu(svg, outX(ANSWER[i]), t, ({ x, y, opacity }) => answerChip(svg, i, { x, y, opacity }));

// ---- frame 5: "down" streams back to you; TTFT spans arrival to the end of prefill ----
export function drawFrame5(svg, p) {
  pipeline(svg, CHOSEN);
  cacheTiles(svg, PROMPT.length);
  yourBar(svg, barSteps(1, 0));
  answerChip(svg, 0, { x: outX(ANSWER[0]) }); // kept in the out slot: it goes back in at the next step
  const back = seg(p, 0.1, 0.6);
  if (back < 1) streamBack(svg, 0, back);
  else answerRow(svg, 1);
  ttftBracket(svg, ease(seg(p, 0.55, 0.85)));
  note(svg, 0, PICK_NOTE, leaving(p));
  note(svg, 1, PREFILL_NOTE, leaving(p));
  note(svg, 0, TTFT_NOTE, seg(p, 0.85, 1));
  note(svg, 1, CHAT_NOTE, seg(p, 0.85, 1));
}

// ---- frame 6: one decode step: "down" goes back in, the cache gains a tile, "." comes out ----
export function drawFrame6(svg, p) {
  pipeline(svg, CHOSEN);
  answerRow(svg, 1);
  cacheTiles(svg, PROMPT.length + (p >= 0.4 ? 1 : 0));
  yourBar(svg, barSteps(1, p >= 0.75 ? 1 : 0));
  ttftBracket(svg, 1);
  feedBack(svg, 0, seg(p, 0.1, 0.35));
  outOfGpu(svg, 1, seg(p, 0.45, 0.75));
  gpuNote(svg, PROMPT.length + 1, seg(p, 0.1, 0.25));
  note(svg, 0, TTFT_NOTE);
  note(svg, 1, CHAT_NOTE, leaving(p));
  note(svg, 1, STEP_NOTE, seg(p, 0.75, 0.9));
}

// ---- frame 7: steps repeat at a steady beat; TPOT spans two adjacent ticks ----
export function drawFrame7(svg, p) {
  pipeline(svg, CHOSEN);
  const back = seg(p, 0.05, 0.35);
  if (back < 1) {
    answerRow(svg, 1);
    answerChip(svg, 1, { x: outX(ANSWER[1]) });
    streamBack(svg, 1, back);
  } else {
    answerRow(svg, 2);
    feedBack(svg, 1, seg(p, 0.35, 0.5));
  }
  cacheTiles(svg, PROMPT.length + (p >= 0.5 ? 2 : 1));
  outOfGpu(svg, 2, seg(p, 0.5, 0.7));
  yourBar(svg, barSteps(1, p >= 0.7 ? 2 : 1));
  ttftBracket(svg, 1);
  tpotBracket(svg, ease(seg(p, 0.72, 0.95)));
  gpuNote(svg, PROMPT.length + (p >= 0.35 ? 2 : 1), 1);
  note(svg, 0, TTFT_NOTE);
  note(svg, 1, STEP_NOTE, leaving(p));
  note(svg, 1, TPOT_NOTE, seg(p, 0.85, 1));
}

// ---- frame 8: the end token reaches you; the request's cache is freed (or kept for reuse) ----
export function drawFrame8(svg, p) {
  pipeline(svg, CHOSEN);
  const back = seg(p, 0.05, 0.4);
  if (back < 1) {
    answerRow(svg, 2);
    if (back <= 0) answerChip(svg, 2, { x: outX(ANSWER[2]) });
    else streamBack(svg, 2, back);
  } else answerRow(svg, 3);
  cacheTiles(svg, PROMPT.length + ANSWER.length - 1, { faint: seg(p, 0.45, 0.7) });
  yourBar(svg, barSteps(1, 2));
  ttftBracket(svg, 1);
  tpotBracket(svg, 1);
  gpuNote(svg, PROMPT.length + 2, leaving(p));
  const reuse = seg(p, 0.6, 0.75);
  label(svg, KV.x + 100, KV.y + 46, 'kept for reuse?', { anchor: 'end', cls: '', opacity: reuse });
  label(svg, KV.x + 100, KV.y + 60, `see ${TITLES.prefixCaching}`, { anchor: 'end', cls: '', opacity: reuse });
  note(svg, 0, TTFT_NOTE);
  note(svg, 1, TPOT_NOTE);
  note(svg, 2, END_NOTE, seg(p, 0.7, 0.85));
}

// What frame 8 ended on (everything of your request), fading out while frame 9 zooms out.
export function frame8End(svg, opacity) {
  if (opacity <= 0) return;
  cacheTiles(svg, PROMPT.length + ANSWER.length - 1, { faint: 1, tiles: opacity });
  const g = layer(svg, opacity);
  answerRow(g, 3);
  yourBar(g, barSteps(1, 2));
  ttftBracket(g, 1);
  tpotBracket(g, 1);
  label(g, KV.x + 100, KV.y + 46, 'kept for reuse?', { anchor: 'end', cls: '' });
  label(g, KV.x + 100, KV.y + 60, `see ${TITLES.prefixCaching}`, { anchor: 'end', cls: '' });
  [TTFT_NOTE, TPOT_NOTE, END_NOTE].forEach((text, row) => note(g, row, text));
}
