// multimodal frames 4–6 (storyboard §5): merge neighboring patches, project to the stream's width, join the sequence.
import * as G from '@shared/glyphs.js';
import { Z1, MERGED, IMAGE_TOKENS, SEQ_PATTERN, SEQ_VISIBLE, SEQUENCE, TOKEN_WORDS, SCALE, FOLLOWED, GRID_N, groupOf, slotOf } from './numbers.js';
import * as S from './stage.js';
import { encoderScene } from './frames-patches.js';

const { IMAGE_LABEL_Y, Z, ROW, SMALL, NOTE_Y, seg, lerp, arriving, leaving, layer, note, noteLines, mark, linked, patchGrid, groupLines, zMatrix } = S;

const MERGED_ROW_PAD = (SMALL - S.MERGED.cell) / 2; // merged rows sit centered on their sequence row
const mergedY = (g) => S.rowY(g) + MERGED_ROW_PAD;
const MERGED_W = 4 * S.MERGED.slotW;
const SLIDE = S.mergedX(0) - S.MERGED_LEFT; // frame 5: how far the merged vectors travel left

// ---- frame 4: 16 vectors → 4 merged vectors ----
// Row i of Z1 at merge progress t: from its place in Z (cell 16) to its slot in its group's merged row (cell 6).
function mergingRow(parent, i, t) {
  const slot = slotOf(i);
  const x = lerp(Z.x, S.mergedX(slot), t);
  const y = lerp(Z.y + i * Z.cell, mergedY(groupOf(i)), t);
  G.vector(parent, { x, y, values: Z1[i], cell: lerp(Z.cell, S.MERGED.cell, t), orient: 'row', maxAbs: SCALE.z1 });
}

const MERGE_NOTE = '2 × 2 merge: 16 vectors → 4';
function mergeLabel(parent, opacity) {
  const box = { x: 152, y: 193, w: MERGE_NOTE.length * 6.6, h: 14 };
  note(linked(layer(parent, opacity), 'merge', box), box.x, 200, MERGE_NOTE);
}

// The followed row's outline: around row 6 of Z, then around the merged vector it ends up in.
function followedMark(parent, t) {
  const from = { x: Z.x, y: Z.y + FOLLOWED * Z.cell, w: 8 * Z.cell, h: Z.cell };
  const to = { x: S.mergedX(0), y: mergedY(0), w: MERGED_W, h: S.MERGED.cell };
  mark(parent, lerp(from.x, to.x, t), lerp(from.y, to.y, t), lerp(from.w, to.w, t), lerp(from.h, to.h, t));
}

const FRAME4_NOTES = ['16 → 4 vectors of 4 × 8 = 32 numbers', 'merge 2 × 2 divides the tokens by 4'];
const mergedLabel = (parent, x, opacity) => noteLines(parent, x, 14, ['4 merged vectors,', 'each 4 × 8 = 32 numbers'], opacity);

export function drawFrame4(svg, p) {
  note(svg, S.GRID.x, IMAGE_LABEL_Y, 'image: 16 × 16 px');
  patchGrid(svg);
  groupLines(svg, seg(p, 0.1, 0.4));
  const gone = leaving(p);
  if (gone > 0) encoderScene(svg, gone);
  const t = seg(p, 0.4, 0.9);
  if (t === 0) zMatrix(svg, Z1, { scale: SCALE.z1, label: 'patch vectors Z' });
  else Z1.forEach((_, i) => mergingRow(svg, i, t));
  note(svg, Z.x, Z.y - 10, 'patch vectors Z', { opacity: t === 0 ? 0 : 1 - t });
  followedMark(svg, t);
  mergeLabel(svg, seg(p, 0.2, 0.4));
  mergedLabel(svg, S.mergedX(0), seg(p, 0.85, 1));
  noteLines(svg, S.GRID.x, NOTE_Y[0], FRAME4_NOTES, seg(p, 0.85, 1));
}

// ---- frame 5: the projector ----
const PROJ = Object.freeze({ x: S.PROJECTOR.x, y: S.PROJECTOR.y, w: S.PROJECTOR.w, h: 3 * ROW.stride + ROW.cell });
const rowMid = (k) => S.rowY(k) + ROW.cell / 2;
const isShown = (p, k) => p >= 0.68 + 0.07 * k + 0.1 - 1e-9;

function mergedRows(parent, shift) {
  MERGED.forEach((values, g) => {
    for (let s = 0; s < 4; s += 1) {
      G.vector(parent, { x: S.mergedX(s) - SLIDE * shift, y: mergedY(g), values: values.slice(s * 8, s * 8 + 8), cell: S.MERGED.cell, orient: 'row', maxAbs: SCALE.merged });
    }
  });
  mark(parent, S.mergedX(0) - SLIDE * shift, mergedY(0), MERGED_W, S.MERGED.cell);
}

function projectorBlock(parent, opacity) {
  const box = linked(layer(parent, opacity), 'proj', PROJ);
  G.block(box, { ...PROJ, label: 'projector [32 → 8]', state: 'active' });
}

function imageRows(parent, p) {
  IMAGE_TOKENS.forEach((values, k) => {
    const t = seg(p, 0.68 + 0.07 * k, 0.78 + 0.07 * k);
    if (t <= 0) return;
    const g = layer(parent, t);
    G.vector(g, { x: S.STREAM_X, y: S.rowY(k), values, cell: SMALL, orient: 'row', maxAbs: SCALE.tokens });
    note(g, S.STREAM_X + 8 * SMALL + 8, rowMid(k), SEQUENCE[k]);
  });
  if (isShown(p, 0)) mark(parent, S.STREAM_X, S.rowY(0), 8 * SMALL, SMALL);
}

function projectorFlows(parent, p) {
  IMAGE_TOKENS.forEach((_, k) => {
    const into = seg(p, 0.5 + 0.07 * k, 0.6 + 0.07 * k);
    const out = seg(p, 0.6 + 0.07 * k, 0.7 + 0.07 * k);
    if (into > 0 && into < 1) G.flow(parent, { from: [S.MERGED_LEFT + MERGED_W + 2, rowMid(k)], to: [PROJ.x - 3, rowMid(k)], carry: 'activation', progress: into });
    if (out > 0 && out < 1) G.flow(parent, { from: [PROJ.x + PROJ.w + 2, rowMid(k)], to: [S.STREAM_X - 3, rowMid(k)], carry: 'activation', progress: out });
  });
}

const FRAME5_NOTES = ['32 → 8 · 4 image tokens of d_model 8', 'the projector is a small MLP'];

export function drawFrame5(svg, p) {
  const gone = leaving(p);
  if (gone > 0) {
    note(svg, S.GRID.x, IMAGE_LABEL_Y, 'image: 16 × 16 px', { opacity: gone });
    patchGrid(svg, { opacity: gone });
    groupLines(svg, 1, gone);
    mergeLabel(svg, gone);
  }
  const shift = seg(p, 0.15, 0.4);
  mergedRows(svg, shift);
  mergedLabel(svg, S.mergedX(0) - SLIDE * shift, 1);
  projectorBlock(svg, seg(p, 0.4, 0.5));
  projectorFlows(svg, p);
  imageRows(svg, p);
  noteLines(svg, S.GRID.x, NOTE_Y[0], FRAME4_NOTES, gone);
  noteLines(svg, S.GRID.x, NOTE_Y[0], FRAME5_NOTES, seg(p, 0.85, 1));
}

// ---- frame 6: image tokens and words in one sequence ----
const HEAT = Object.freeze({ x: 70, y: 88, cell: SMALL });

function words(parent, p) {
  TOKEN_WORDS.forEach((text, i) => {
    const t = seg(p, 0.15 + 0.07 * i, 0.4 + 0.07 * i);
    if (t <= 0) return;
    const g = layer(parent, t);
    G.token(g, { x: S.STREAM_X + 40 * (1 - t), y: S.rowY(4 + i), text, index: 5 + i });
  });
}

function sequencePattern(parent, p) {
  const filled = Math.floor(seg(p, 0.35, 1) * 8 + 1e-9);
  const values = SEQ_PATTERN.map((row, i) => (i < filled ? row : row.map(() => Number.NaN)));
  const visible = SEQ_VISIBLE;
  G.heatmap(layer(parent, seg(p, 0.3, 0.4)), {
    x: HEAT.x, y: HEAT.y, values, mask: visible, cell: HEAT.cell, maxAbs: SCALE.seq, label: 'attention pattern',
    rowLabels: SEQUENCE, colLabels: SEQUENCE.map((_, i) => String(i + 1)),
  });
}

export function drawFrame6(svg, p) {
  const gone = leaving(p);
  if (gone > 0) {
    mergedRows(layer(svg, gone), 1);
    mergedLabel(svg, S.MERGED_LEFT, gone);
    projectorBlock(svg, gone);
    noteLines(svg, S.GRID.x, NOTE_Y[0], FRAME5_NOTES, gone);
  }
  imageRows(svg, 1);
  words(svg, p);
  sequencePattern(svg, p);
  const rest = seg(p, 0.5, 0.8);
  noteLines(svg, S.STREAM_X, S.rowY(7) + 44, ['in this sequence the words', 'sit at positions 5–8'], rest);
  note(svg, HEAT.x + 8 * HEAT.cell + 8, HEAT.y + 6 * HEAT.cell + HEAT.cell / 2, '← sat (7) reads 7 keys', { opacity: rest });
  noteLines(svg, 40, HEAT.y + 8 * HEAT.cell + 16, ['columns are the keys, positions 1 to 8', 'each word row reads the 4 image tokens', 'and the words before it'], rest);
}
