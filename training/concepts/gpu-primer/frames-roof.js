// gpu-primer frames 6–9: the roofline, the multiply at real size (memory-bound), more tokens (compute-bound), then FP8.
// The followed dot (a roofline point with followed: true, which draws its own selection mark) is on the plot in every frame.
import { attainableTflops, ridgePoint, tokensToComputeBound, bytesPerElement } from '@math/roofline.js';
import { formatDuration, formatRatio } from '@math/core.js';
import { sharePct } from '@math/memory.js';
import { STAGE_CHIPS, TOKENS, FRAME8_TOKENS, D_REAL } from './numbers.js';
import { LEFT, LINE, seg, lerp, arriving, leaving, layer, note, lines, chips, shape, roof, roofFade, realCost, lanes, laneScale, laneNote } from './stage.js';
import { restScene, xMatrix, readCells, FRAME4_END, INTENSITY, INTENSITY_LINE } from './frames-gpu.js';
import { int, fixed1 } from './format.js';

const H100 = STAGE_CHIPS.h100;
const BW = H100.bandwidths[0];
const BF16 = Object.freeze({ peakTflops: H100.peak.bf16, bandwidthTBps: BW, title: 'H100 · BF16' });
const FP8 = Object.freeze({ peakTflops: H100.peak.fp8, bandwidthTBps: BW, title: 'H100 · FP8 (E4M3)' });
const RIDGE = ridgePoint(BF16);
const FP8_B = bytesPerElement('fp8_e4m3');
const tokensLabel = (m) => `${int(m)} token${m === 1 ? '' : 's'}`;
const N4 = TOKENS.length;
const REAL4 = realCost(N4, BF16.peakTflops, BW);
const REAL4_FP8 = realCost(N4, FP8.peakTflops, BW, FP8_B);
const REAL4096 = realCost(FRAME8_TOKENS.at(-1), BF16.peakTflops, BW);
const REAL4096_FP8 = realCost(FRAME8_TOKENS.at(-1), FP8.peakTflops, BW, FP8_B);
const CROSS = tokensToComputeBound({ ...BF16, bytesPerElem: 2, k: D_REAL, n: D_REAL });
const CROSS_FP8 = tokensToComputeBound({ ...FP8, bytesPerElem: FP8_B, k: D_REAL, n: D_REAL });
const FIXED_SCALE = laneScale(REAL4.time.memoryS * 1e6); // frames 7 and 9: full width = the 4-token BF16 memory time

const TOY_DOT = Object.freeze({ ...BF16, points: [{ intensity: INTENSITY, label: 'toy multiply', followed: true }] });
const dot4 = (spec, cost) => ({ ...spec, points: [{ intensity: cost.intensity, label: tokensLabel(N4), followed: true }] });
const ROOF7 = Object.freeze(dot4(BF16, REAL4));

function frame6Notes(parent) {
  const attainable = attainableTflops({ intensity: INTENSITY, ...BF16 });
  note(parent, LEFT, 312, `ridge = ${int(BF16.peakTflops)} ÷ ${BW} = ${fixed1(RIDGE)} FLOPs per byte`);
  note(parent, LEFT, 328, `dot: intensity ${INTENSITY.toFixed(2)} → ${fixed1(attainable)} TFLOPS (${sharePct(attainable, BF16.peakTflops, { decimals: 2 }).toFixed(2)}% of peak)`);
}

// Frame 6: W_O, Y and the GPU step away; the roofline fades in; the toy multiply's dot appears at intensity 2 on the slope.
export function drawFrame6(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    restScene(g, FRAME4_END);
    note(g, LEFT, 344, INTENSITY_LINE);
  }
  frame6Kept(svg);
  roof(svg, { ...BF16, opacity: seg(p, 0.15, 0.5) * (1 - seg(p, 0.55, 0.8)) });
  roof(svg, { ...TOY_DOT, opacity: seg(p, 0.55, 0.8) });
  frame6Notes(layer(svg, seg(p, 0.5, 0.7)));
}

// What frames 6 and 7 share at the left: the chips, X and the three cells (frame 7 lets X and the cells go).
function frame6Kept(svg, { x = 1 } = {}) {
  chips(svg);
  if (x <= 0) return;
  const g = layer(svg, x);
  xMatrix(g);
  readCells(g, { ...FRAME4_END, intensity: INTENSITY });
}

const REAL_SHAPES = Object.freeze({ xY: 52, wY: 82, w: 196, wSide: 90 });
function realShapes(svg, m = N4, xHeight = 6) {
  shape(svg, { x: LEFT, y: REAL_SHAPES.xY, w: REAL_SHAPES.w, h: xHeight, label: `X [${int(m)} × ${int(D_REAL)}]` });
  shape(svg, { x: LEFT, y: REAL_SHAPES.wY + xHeight - 6, w: REAL_SHAPES.wSide, h: REAL_SHAPES.wSide, label: `W [${int(D_REAL)} × ${int(D_REAL)}]` });
}

const toySizeMark = (a) => `toy size: inputs and outputs are half the bytes; at real size the weights are ${sharePct(D_REAL * D_REAL * 2, a.cost.bytes).toFixed(1)}%`;
function frame7Readouts(svg) {
  const att = attainableTflops({ intensity: REAL4.intensity, ...BF16 });
  lines(svg, LEFT, 196, [
    `FLOPs ${int(REAL4.cost.flops)}`, `bytes ${int(REAL4.cost.bytes)}`, `intensity ${REAL4.intensity.toFixed(2)}`,
    `${fixed1(att)} TFLOPS = ${sharePct(att, BF16.peakTflops, { decimals: 2 }).toFixed(2)}% of peak`,
  ]);
}
function frame7Marks(svg) {
  const { memoryS, computeS } = REAL4.time;
  note(svg, LEFT, 344, toySizeMark(REAL4));
  note(svg, LEFT, 360, `tensor cores idle ${sharePct(memoryS - computeS, memoryS).toFixed(1)}% of the time; memory takes ${formatRatio(memoryS / computeS)} as long as compute`);
}

// Frame 7: real size; the dot moves to intensity 4; the memory lane runs to 40.1 µs while compute stops early and idles.
export function drawFrame7(svg, p) {
  const out = leaving(p);
  frame6Kept(svg, { x: out });
  if (out > 0) frame6Notes(layer(svg, out));
  realShapes(layer(svg, arriving(p)));
  roofFade(svg, TOY_DOT, ROOF7, seg(p, 0.1, 0.4));
  const grow = seg(p, 0.35, 0.85);
  laneNote(svg, REAL4.time.memoryS * 1e6, seg(p, 0.3, 0.4));
  lanes(svg, { time: REAL4.time, scale: FIXED_SCALE, grow });
  if (grow >= 1) {
    frame7Readouts(svg);
    frame7Marks(svg);
  }
}

// ---- frame 8: the token counter 4 → 64 → 256 → 4,096 ----
const STOPS = FRAME8_TOKENS;
const STEP = Object.freeze({ start: 0.12, len: 0.28 });
// Where the counter is: the stop index it has reached, and a tweened (log-space) token count for the dot and lanes.
function counter(p) {
  const k = (p - STEP.start) / STEP.len;
  if (k <= 0) return { m: STOPS[0], shown: 0 };
  const i = Math.min(Math.floor(k), STOPS.length - 2);
  const t = Math.min(k - i, 1);
  const tween = Math.min(t / 0.7, 1);
  const m = tween >= 1 ? STOPS[i + 1] : Math.exp(lerp(Math.log(STOPS[i]), Math.log(STOPS[i + 1]), tween));
  return { m, shown: t >= 0.35 ? i + 1 : i };
}

function frame8Left(svg, c) {
  note(svg, LEFT, 16, tokensLabel(STOPS[c.shown]), { cls: '' });
  realShapes(svg, STOPS[c.shown], lerp(6, 18, c.shown / (STOPS.length - 1)));
  const intensities = STOPS.map((m) => int(realCost(m, BF16.peakTflops, BW).intensity));
  lines(svg, LEFT, 214, [
    `tokens ${STOPS.slice(0, c.shown + 1).map(int).join(' → ')}`,
    `intensity ${intensities.slice(0, c.shown + 1).join(' → ')}`,
    `crossing at ${fixed1(CROSS)} tokens`,
  ]);
}

// Frame 8: the counter ticks; the dot slides along the roof past the bend; the lanes swap which is longer.
export function drawFrame8(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    chips(g);
    realShapes(g);
    frame7Readouts(g);
    frame7Marks(g);
  }
  const c = counter(p);
  const now = realCost(c.m, BF16.peakTflops, BW);
  frame8Left(layer(svg, arriving(p)), c);
  // The plot and lanes show the tweened count itself (its dot label, lane lengths and full width all agree);
  // the counter at the left prints only the stops.
  roof(svg, { ...BF16, points: [{ intensity: now.intensity, label: tokensLabel(Math.round(c.m)), followed: true }] });
  laneNote(svg, Math.max(now.time.memoryS, now.time.computeS) * 1e6);
  lanes(svg, { time: now.time, labels: STOPS.includes(c.m) });
}

const ROOF9_BF16 = Object.freeze({ ...BF16, points: [{ intensity: REAL4.intensity, label: tokensLabel(N4), followed: true }, { intensity: REAL4096.intensity, label: tokensLabel(4096) }] });
const ROOF9_FP8 = Object.freeze({ ...FP8, points: [{ intensity: REAL4_FP8.intensity, label: tokensLabel(N4), followed: true }, { intensity: REAL4096_FP8.intensity, label: tokensLabel(4096) }] });
const blendTime = (a, b, t) => ({ memoryS: lerp(a.memoryS, b.memoryS, t), computeS: lerp(a.computeS, b.computeS, t) });

function frame9Left(svg) {
  lines(svg, LEFT, 16, [`FP8: ${FP8_B} byte per number,`, `tensor cores ${formatRatio(FP8.peakTflops / BF16.peakTflops)} faster`]);
  lines(svg, LEFT, 64, [
    `ridge ${fixed1(RIDGE)} → ${fixed1(ridgePoint(FP8))}`,
    `${tokensLabel(N4)}: intensity ${REAL4.intensity.toFixed(2)} → ${REAL4_FP8.intensity.toFixed(2)}`,
    `memory ${formatDuration(REAL4.time.memoryS)} → ${formatDuration(REAL4_FP8.time.memoryS)}`,
    `crossing ${fixed1(CROSS)} → ${fixed1(CROSS_FP8)} tokens`,
  ]);
  lines(svg, LEFT, 64 + 5 * LINE, [`H100 FP8 ${int(FP8.peakTflops)} TFLOPS`, '(reported: 2 × BF16)'], { cls: 'g-label' });
  lines(svg, LEFT, 64 + 8 * LINE, ['inputs and outputs in the same', 'format as the weights'], { cls: 'g-label' });
}

// Frame 9: back to the four tokens (outlined) with 4,096 plain; the flat roof doubles, the bend and both dots move right ×2,
// and the 4-token lanes halve on a held axis.
export function drawFrame9(svg, p) {
  const out = leaving(p);
  if (out > 0) frame8Left(layer(svg, out), { m: STOPS.at(-1), shown: STOPS.length - 1 });
  const switchT = seg(p, 0.3, 0.75);
  if (p < 0.3) roofFade(svg, { ...BF16, points: [{ intensity: REAL4096.intensity, label: tokensLabel(4096), followed: true }] }, ROOF9_BF16, arriving(p));
  else roofFade(svg, ROOF9_BF16, ROOF9_FP8, switchT);
  const t9 = blendTime(REAL4.time, REAL4_FP8.time, switchT);
  if (out > 0) {
    laneNote(svg, REAL4096.time.computeS * 1e6, out);
    lanes(svg, { time: REAL4096.time, opacity: out });
  }
  laneNote(svg, REAL4.time.memoryS * 1e6, arriving(p));
  lanes(svg, { time: t9, scale: FIXED_SCALE, labels: switchT === 0 || switchT === 1, opacity: arriving(p) });
  if (switchT >= 1) frame9Left(svg);
  else note(layer(svg, arriving(p)), LEFT, 16, `dots: ${int(N4)} and ${tokensLabel(4096)}`, { cls: '' });
}

export const FRAME9_END = Object.freeze({ roof: ROOF9_FP8, time: REAL4_FP8.time, scale: FIXED_SCALE });
export { frame9Left };
