// Frames 1–5: decoding "on" with and without a KV cache (storyboard §5). Each frame is a pure function of its progress
// p (0 → 1); the end of frame n is the start of frame n + 1, and what leaves or arrives fades during [0, HANDOFF].
import * as G from '@shared/glyphs.js';
import { decodeWork } from '@math/memory.js';
import { int } from './format.js';
import { REPLY, PROMPT_LEN, ON_ROW, passTotals } from './numbers.js';
import {
  GRID, MAT_Y, LANE, BRANCH, RIGHT, NOTE_Y, W_BLOCK, TOKEN_Y, HANDOFF,
  seg, ease, arriving, leaving, layer, label, textBlock, promptChips, onChip, wBlock, chipFlows, chipCenters,
  kvMatrices, strips, cacheStack, counter,
} from './stage.js';

const BLANK_NOTE = 'blank cells: not computed yet';
const ROW_CENTER = (row) => MAT_Y + row * GRID + GRID / 2;
const BRANCH_LABEL = (parent, text, opacity = 1) => label(parent, BRANCH.x, BRANCH.y, text, { cls: '', opacity });
const blankNote = (parent, opacity = 1) => label(parent, 50, NOTE_Y, BLANK_NOTE, { opacity });

// ---- frame 1: "on" arrives; one more position to look at ----
export function drawFrame1(svg, p) {
  const slide = ease(seg(p, 0, 0.55));
  promptChips(svg);
  onChip(svg, { slide });
  wBlock(svg);
  kvMatrices(svg, { rows: 5, solid: 4 });
  textBlock(svg, RIGHT.x, RIGHT.y, [slide >= 1 ? '4 → 5 positions' : '4 positions'], { cls: '' });
  textBlock(svg, RIGHT.x, RIGHT.y + 22, ['this toy: 2 blocks,', '2 heads in each block;', 'head A is drawn']);
  blankNote(svg);
}

// ---- frame 2: without a cache, every position is computed again ----
const REFILL = Object.freeze({ blankAt: 0.15, from: 0.2, step: 0.13 }); // rows 1–5 refill one per step, top to bottom
const rowEnd = (i) => REFILL.from + REFILL.step * (i + 1);
const refilled = (p) => [0, 1, 2, 3, 4].filter((i) => p >= rowEnd(i)).length;
const FILLS_ROW_5 = Object.freeze([rowEnd(3), rowEnd(4)]); // k_on and v_on type in while row 5 is being computed

export function drawFrame2(svg, p) {
  promptChips(svg);
  onChip(svg);
  wBlock(svg);
  const solid = p < REFILL.blankAt ? 4 : refilled(p);
  kvMatrices(svg, { rows: 5, solid });
  BRANCH_LABEL(svg, 'Without a cache', arriving(p));
  chipFlows(svg, 5, seg(p, 0, REFILL.from), 1 - seg(p, REFILL.from, REFILL.from + 0.05));
  for (let i = 0; i < Math.min(solid, 4); i += 1) label(svg, LANE.x + 2, ROW_CENTER(i), '= same as before');
  if (p >= FILLS_ROW_5[0]) strips(svg, { fill: seg(p, FILLS_ROW_5[0], FILLS_ROW_5[1]) });
  if (p >= REFILL.from) counter(svg, RIGHT.y, 'positions computed', solid, 'this step');
  blankNote(svg, 1 - seg(p, 0.8, 1));
}

// ---- frame 3: with a cache, only the new token is computed and the rest is read ----
const READS = Object.freeze({ from: 0.7, stagger: 0.04, span: 0.12 }); // five keys (and five values) read, one after another
const readProgress = (p, j) => seg(p, READS.from + READS.stagger * j, READS.from + READS.stagger * j + READS.span);
const readsDone = (p) => [0, 1, 2, 3, 4].filter((j) => readProgress(p, j) >= 1).length;
const FRAME3 = Object.freeze({ flow: [0.15, 0.35], type: [0.35, 0.5], slide: [0.5, 0.7] });

// What frame 2 ended on, fading away while frame 3 begins.
function withoutCacheEnd(parent, opacity) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  BRANCH_LABEL(g, 'Without a cache');
  for (let i = 0; i < 4; i += 1) label(g, LANE.x + 2, ROW_CENTER(i), '= same as before');
  strips(g, {});
  counter(g, RIGHT.y, 'positions computed', 5, 'this step');
}

function readArrows(svg, p) {
  [0, 1, 2, 3, 4].forEach((j) => {
    const t = readProgress(p, j);
    if (t <= 0) return;
    const y = ROW_CENTER(j);
    G.flow(svg, { from: [LANE.x, y], to: [LANE.keyEnd, y], carry: 'kv', progress: t });
    G.flow(svg, { from: [LANE.valueStart, y], to: [LANE.valueEnd, y], carry: 'kv', progress: t });
  });
  if (p >= READS.from) label(svg, LANE.center, ROW_CENTER(2), 'q_on', { anchor: 'middle', cls: '' });
}

export function drawFrame3(svg, p) {
  const lose = leaving(p);
  promptChips(svg);
  onChip(svg);
  wBlock(svg);
  const appended = p >= FRAME3.slide[1];
  kvMatrices(svg, { rows: 5, solid: appended ? 5 : 4, fading: lose > 0 ? [{ row: 4, opacity: lose }] : [] });
  withoutCacheEnd(svg, lose);
  BRANCH_LABEL(svg, 'With a cache', arriving(p));
  cacheStack(svg, { count: appended ? 5 : 4, highlight: appended ? [4] : [], opacity: arriving(p) });
  if (p > FRAME3.flow[0] && p < FRAME3.type[0]) G.flow(svg, { from: [chipCenters[4], TOKEN_Y + 26], to: [chipCenters[4], W_BLOCK.y - 2], carry: 'activation', progress: seg(p, ...FRAME3.flow) });
  if (p >= FRAME3.type[0] && !appended) strips(svg, { fill: seg(p, ...FRAME3.type), lift: ease(seg(p, ...FRAME3.slide)) });
  if (p >= FRAME3.type[0]) counter(svg, RIGHT.y, 'positions computed', 1, 'this step');
  readArrows(svg, p);
  if (p >= READS.from) counter(svg, RIGHT.y + 44, 'keys read', readsDone(p), 'per head, per layer');
}

// ---- frame 4: prefill reads the whole prompt in one pass, then each decode step adds one row ----
const PREFILL = decodeWork({ prompt: PROMPT_LEN, generated: 1, cache: true });
const DECODE_STEP = decodeWork({ prompt: PROMPT_LEN, generated: 2, cache: true });
const FRAME4 = Object.freeze({ flow: [0.15, 0.35], filled: 0.35, chip: [0.55, 0.8], decodeAt: 0.9 });

// What frame 3 ended on, fading away while frame 4 rewinds.
function withCacheEnd(parent, opacity) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  BRANCH_LABEL(g, 'With a cache');
  counter(g, RIGHT.y, 'positions computed', 1, 'this step');
  counter(g, RIGHT.y + 44, 'keys read', 5, 'per head, per layer');
  [0, 1, 2, 3, 4].forEach((j) => {
    const y = ROW_CENTER(j);
    G.flow(g, { from: [LANE.x, y], to: [LANE.keyEnd, y], carry: 'kv', progress: 1 });
    G.flow(g, { from: [LANE.valueStart, y], to: [LANE.valueEnd, y], carry: 'kv', progress: 1 });
  });
  label(g, LANE.center, ROW_CENTER(2), 'q_on', { anchor: 'middle', cls: '' });
}

const prefillText = (parent, opacity) => {
  textBlock(parent, RIGHT.x, RIGHT.y, ['prefill: 4 positions', 'in one pass', `${PREFILL.keyReads} key reads`, '(1 + 2 + 3 + 4)'], { cls: '', opacity });
};
const decodeText = (parent, opacity) => {
  textBlock(parent, RIGHT.x, RIGHT.y + 82, ['decode: 1 position', 'per step', `${DECODE_STEP.keyReads - PREFILL.keyReads} key reads for "on"`], { cls: '', opacity });
};

export function drawFrame4(svg, p) {
  const lose = leaving(p);
  const chipSlide = ease(seg(p, ...FRAME4.chip));
  const rowsBlank = p >= FRAME4.flow[0] && p < FRAME4.filled;
  const decoded = p >= FRAME4.decodeAt;
  promptChips(svg);
  if (lose > 0) onChip(svg, { opacity: lose });
  else if (chipSlide > 0) onChip(svg, { slide: chipSlide });
  wBlock(svg);
  kvMatrices(svg, {
    rows: 5, solid: rowsBlank ? 0 : (decoded ? 5 : 4), follow: lose > 0 ? lose : chipSlide,
    fading: lose > 0 ? [{ row: 4, opacity: lose }] : [],
  });
  withCacheEnd(svg, lose);
  label(svg, BRANCH.x, BRANCH.y, 'Before and after "on"', { cls: '', opacity: arriving(p) });
  cacheStack(svg, { count: p < FRAME4.flow[0] ? 5 : rowsBlank ? 0 : decoded ? 5 : 4, highlight: decoded || p < FRAME4.flow[0] ? [4] : [] });
  chipFlows(svg, 4, seg(p, ...FRAME4.flow), p < FRAME4.flow[1] ? 1 : 0);
  if (p >= FRAME4.filled) prefillText(svg, seg(p, FRAME4.filled, FRAME4.filled + 0.1));
  if (decoded) decodeText(svg, seg(p, FRAME4.decodeAt, FRAME4.decodeAt + 0.1));
  blankNote(svg, rowsBlank ? 1 : 0);
}

// ---- frame 5: a whole four-token reply, counted pass by pass ----
const PASSES = 4; // prefill, then three decode steps: on, the, mat
const PASS_SPAN = (1 - HANDOFF) / PASSES;
const passEnd = (k) => HANDOFF + PASS_SPAN * k - 0.04;
const passesDone = (p) => Array.from({ length: PASSES }, (_, i) => i + 1).filter((k) => p >= passEnd(k)).length;
const TABLE = Object.freeze({ noCacheX: 530, cacheX: RIGHT.end });
const LONG_REPLY = Object.freeze({ off: decodeWork({ prompt: 1000, generated: 1000, cache: false }).positions, on: decodeWork({ prompt: 1000, generated: 1000, cache: true }).positions });

function table(parent, done, opacity) {
  const none = { positions: 0, keyReads: 0 };
  const [off, on] = done === 0 ? [none, none] : [passTotals(done, false), passTotals(done, true)];
  label(parent, TABLE.noCacheX, RIGHT.y, 'no cache', { anchor: 'end', opacity });
  label(parent, TABLE.cacheX, RIGHT.y, 'cache', { anchor: 'end', opacity });
  [['positions', 'positions'], ['key reads', 'keyReads']].forEach(([name, key], i) => {
    const y = RIGHT.y + 22 + i * 18;
    label(parent, RIGHT.x, y, name, { opacity });
    label(parent, TABLE.noCacheX, y, int(off[key]), { anchor: 'end', cls: '', opacity });
    label(parent, TABLE.cacheX, y, int(on[key]), { anchor: 'end', cls: '', opacity });
  });
}

export function drawFrame5(svg, p) {
  const lose = leaving(p);
  const done = passesDone(p);
  const stored = done === 0 ? 0 : PROMPT_LEN - 1 + done; // rows computed so far
  promptChips(svg);
  onChip(svg);
  wBlock(svg);
  kvMatrices(svg, { rows: 8, solid: stored, fading: lose > 0 ? [0, 1, 2, 3, 4].map((row) => ({ row, opacity: lose })) : [] });
  if (lose > 0) {
    label(svg, BRANCH.x, BRANCH.y, 'Before and after "on"', { cls: '', opacity: lose });
    prefillText(svg, lose);
    decodeText(svg, lose);
  }
  label(svg, BRANCH.x, BRANCH.y, 'Over a four-token reply', { cls: '', opacity: arriving(p) });
  const count = p < HANDOFF ? 5 : stored;
  cacheStack(svg, { count, highlight: count > ON_ROW ? [ON_ROW] : [] }); // the followed token keeps its mark: tile 5, "on"
  table(svg, done, seg(p, 0, HANDOFF));
  const pass = Math.min(done + 1, PASSES);
  label(svg, RIGHT.x, RIGHT.y + 66, `pass ${pass} of ${PASSES} (${pass === 1 ? 'prefill' : 'decode'})`, { opacity: arriving(p) });
  label(svg, RIGHT.x, RIGHT.y + 88, `reply: ${REPLY.slice(0, done).join(' ')}`, { cls: '', opacity: arriving(p) });
  textBlock(svg, RIGHT.x, RIGHT.y + 122, ['1,000-token prompt and', `reply: ${int(LONG_REPLY.off)} vs`, `${int(LONG_REPLY.on)} positions`], { opacity: arriving(p) });
  blankNote(svg);
}
