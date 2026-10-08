// Frames 1–3: MHA, MQA, GQA on the 8-head toy layer (storyboard §5). Each frame is a pure function of its progress p (0 → 1);
// the end of frame n is the start of frame n + 1. Stored heads are drawn from their 8 MHA places and slide to their group's.
import {
  QUERY_HEADS, HEAD_CENTERS, GROUP_CENTERS, SINGLE_CENTER, HEADER, READOUT, LINES, STACK, qx, Q,
  seg, lerp, ease, key, swapKey, bracket, queryRow, kvHead, wire,
} from './stage.js';
import { TOY_SHAPE } from './numbers.js';

const heads = Array.from({ length: QUERY_HEADS }, (_, h) => h);
const FORMULA = Object.freeze({
  mha: '8 KV heads × 4 numbers × 2 (K and V) = 64 numbers per token per layer',
  mqa: '1 × 4 × 2 = 8 numbers · 64 ÷ 8 = 8× smaller',
  gqa: '2 × 4 × 2 = 16 numbers · 64 ÷ 16 = 4× smaller',
});
const readout = (svg, n, opacity = 1) => key(svg, READOUT.x, READOUT.y, `stored per token per layer: ${n} numbers`, { anchor: 'end', opacity });
const formula = (svg, text, opacity) => key(svg, LINES.x, LINES.share, text, { opacity });
const numbersAt = (kvHeads) => 2 * kvHeads * TOY_SHAPE.headDim; // numbers per token per layer, as storyboard §5 writes it

export function drawFrame1(svg, p) {
  const stackT = (h) => ease(seg(p, 0.07 * h + 0.1, 0.07 * h + 0.3)); // stack h fills in; its wire draws just before
  queryRow(svg);
  key(svg, HEADER.x, HEADER.y, 'MHA');
  heads.forEach((h) => {
    const draw = seg(p, 0.07 * h, 0.07 * h + 0.2);
    wire(svg, { fromX: HEAD_CENTERS[h], head: h, opacity: draw, progress: 0.62 * draw });
    if (stackT(h) > 0) kvHead(svg, HEAD_CENTERS[h], stackT(h));
  });
  readout(svg, Math.round(numbersAt(1) * heads.reduce((sum, h) => sum + stackT(h), 0)));
  formula(svg, FORMULA.mha, seg(p, 0.8, 1));
}

// Seven stacks slide into the first and fade; every wire follows its own stack, so all eight land on one.
export function drawFrame2(svg, p) {
  const slide = ease(seg(p, 0.1, 0.75));
  const gone = seg(p, 0.35, 0.8);
  queryRow(svg);
  swapKey(svg, HEADER.x, HEADER.y, 'MHA', 'MQA', p, [0, 0.2, 0.4]);
  const x = (h) => lerp(HEAD_CENTERS[h], SINGLE_CENTER, slide);
  heads.forEach((h) => wire(svg, { fromX: x(h), head: h }));
  [...heads].reverse().forEach((h) => kvHead(svg, x(h), h === 0 ? 1 : 1 - gone));
  readout(svg, Math.round(lerp(numbersAt(8), numbersAt(1), slide)));
  key(svg, SINGLE_CENTER, STACK.y + STACK.h + 22, 'some quality loss', { anchor: 'middle', opacity: seg(p, 0.7, 0.95) });
  swapKey(svg, LINES.x, LINES.share, FORMULA.mha, FORMULA.mqa, p, [0, 0.2, 0.85]);
}

// The single stack splits in two; wires for Q5–Q8 swing to the new one. Brackets name the groups.
export function drawFrame3(svg, p) {
  const split = ease(seg(p, 0.1, 0.7));
  const groupOf = (h) => (h < 4 ? 0 : 1);
  const x = (g) => lerp(SINGLE_CENTER, GROUP_CENTERS[g], split);
  queryRow(svg);
  swapKey(svg, HEADER.x, HEADER.y, 'MQA', 'GQA', p, [0, 0.2, 0.4]);
  key(svg, SINGLE_CENTER, STACK.y + STACK.h + 22, 'some quality loss', { anchor: 'middle', opacity: 1 - seg(p, 0, 0.2) });
  heads.forEach((h) => wire(svg, { fromX: x(groupOf(h)), head: h }));
  kvHead(svg, x(1), seg(p, 0, 0.15));
  kvHead(svg, x(0));
  readout(svg, Math.round(lerp(numbersAt(1), numbersAt(2), split)));
  [[0, 3], [4, 7]].forEach(([a, b]) => {
    const opacity = seg(p, 0.7, 0.95);
    bracket(svg, { x1: qx(a), x2: qx(b) + Q.w, y: STACK.y + STACK.h + 14, opacity });
    key(svg, (qx(a) + qx(b) + Q.w) / 2, STACK.y + STACK.h + 30, 'group of 4', { anchor: 'middle', opacity });
  });
  swapKey(svg, LINES.x, LINES.share, FORMULA.mqa, FORMULA.gqa, p, [0, 0.2, 0.85]);
}
