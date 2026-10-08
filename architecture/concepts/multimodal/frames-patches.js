// multimodal frames 1–3 (storyboard §5): cut the image into patches, embed each patch, let the patches see each other.
// Each drawFrameN(svg, p) is a pure function of its progress p (0 → 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { PATCHES, FOLLOWED, Z0, Z1, PATTERN, SCALE, GRID_N } from './numbers.js';
import {
  IMAGE_LABEL_Y, GRID, Z, CENTER, ENCODER, SMALL, NOTE_Y, seg, lerp, arriving, leaving, layer, note, noteLines, mark, guide,
  patchGrid, patchX, patchY, zMatrix,
} from './stage.js';

// ---- patch 6, magnified (end of frame 1) and flattened (frame 2) ----
const BIG = Object.freeze({ x: 176, y: 56, size: 90 }); // the magnified patch: 4 × 4 pixels of 21 px, 3 px inset
const BIG_PX = (BIG.size - 6) / 4;
const FLAT = Object.freeze({ x: 160, y: 56, cell: 10 }); // the same 16 greys as one row
const EMBED = Object.freeze({ x: CENTER.x, y: 96, w: CENTER.w, h: 34 });
const EMBED_VEC = Object.freeze({ x: 168, y: 150 });
const bigPixel = (k) => ({ x: BIG.x + 3 + (k % 4) * BIG_PX, y: BIG.y + 3 + Math.floor(k / 4) * BIG_PX, size: BIG_PX });
const flatPixel = (k) => ({ x: FLAT.x + k * FLAT.cell, y: FLAT.y, size: FLAT.cell - 1 });

// Patch 6's pixels moving from the 4 × 4 square (t = 0) to one row (t = 1). Image content: greys, not the value scale.
function pixelCells(parent, t, opacity = 1) {
  const g = G.svgEl('g', { class: 'glyph g-patch g-patch--idle', role: 'img', 'aria-label': 'patch 6: 16 grey values', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  if (t < 1) G.svgEl('rect', { class: 'g-frame', x: BIG.x, y: BIG.y, width: BIG.size, height: BIG.size, rx: 5, opacity: (1 - t).toFixed(3) }, g);
  PATCHES[FOLLOWED].forEach((grey, k) => {
    const [from, to] = [bigPixel(k), flatPixel(k)];
    const r = G.svgEl('rect', { class: 'g-pixel', x: lerp(from.x, to.x, t), y: lerp(from.y, to.y, t), width: lerp(from.size, to.size, t), height: lerp(from.size, to.size, t) }, g);
    r.style.fill = G.pixelFill(grey);
  });
  return g;
}

// ---- frame 1 ----
function tileLines(svg, drawn, opacity) {
  const [x0, y0] = [patchX(0, 0), patchY(0, 0)];
  const span = GRID_N * GRID.size;
  for (let k = 1; k < GRID_N; k += 1) {
    guide(svg, x0 + k * GRID.size, y0, x0 + k * GRID.size, y0 + span * drawn, 0.7 * opacity);
    guide(svg, x0, y0 + k * GRID.size, x0 + span * drawn, y0 + k * GRID.size, 0.7 * opacity);
  }
}

export function drawFrame1(svg, p) {
  const slide = seg(p, 0.4, 0.8);
  note(svg, GRID.x, IMAGE_LABEL_Y, 'image: 16 × 16 px');
  patchGrid(svg, { gap: slide, numbers: seg(p, 0.7, 0.85) });
  tileLines(svg, seg(p, 0, 0.35), 1 - slide);
  const big = seg(p, 0.8, 1);
  if (big > 0) {
    const g = layer(svg, big);
    pixelCells(g, 0);
    mark(g, BIG.x, BIG.y, BIG.size, BIG.size);
    note(g, BIG.x, BIG.y - 10, 'patch 6, enlarged');
    note(g, BIG.x, BIG.y + BIG.size + 14, '16 grey values');
  }
  noteLines(svg, GRID.x, NOTE_Y[0], ['16 × 16 pixels ÷ 4 × 4 pixels = 16 patches', 'each patch holds 16 grey values']);
}

// ---- frame 2 ----
const embedBlock = (parent, opacity) => {
  const g = layer(parent, opacity);
  G.block(g, { ...EMBED, label: 'patch embedding [16 → 8]', state: 'active' });
  return g;
};

// The lasting marks of frame 2: the flattened row, the embedding block and the notes (they leave in frame 3).
function embedScene(parent, opacity, notesOpacity = opacity) {
  const g = layer(parent, opacity);
  pixelCells(g, 1);
  mark(g, FLAT.x - 2, FLAT.y - 2, 16 * FLAT.cell + 3, FLAT.cell + 3);
  embedBlock(g, 1);
  note(g, FLAT.x, FLAT.y - 12, 'patch 6 flattened: 16 greys');
  noteLines(parent, GRID.x, NOTE_Y[0], ['16 greys → 8 numbers · matrix [16 × 8] = 128 parameters', '16 patches → 16 patch vectors'], notesOpacity);
}

// The vector patch 6 became, sliding into its row of Z.
function embedVector(svg, p, slide) {
  const x = lerp(EMBED_VEC.x, Z.x, slide);
  const y = lerp(EMBED_VEC.y, Z.y + FOLLOWED * Z.cell, slide);
  const cell = lerp(SMALL, Z.cell, slide);
  if (slide >= 1) return;
  const g = layer(svg, seg(p, 0.5, 0.6));
  G.vector(g, { x, y, values: Z0[FOLLOWED], cell, orient: 'row', maxAbs: SCALE.z0 });
  mark(g, x, y, 8 * cell, cell);
}

export function drawFrame2(svg, p) {
  const flat = seg(p, 0.12, 0.4);
  note(svg, GRID.x, IMAGE_LABEL_Y, 'image: 16 × 16 px', { opacity: 1 });
  patchGrid(svg);
  const ghost = leaving(p) * (1 - flat);
  if (ghost > 0) { note(svg, BIG.x, BIG.y + BIG.size + 14, '16 grey values', { opacity: ghost }); note(svg, BIG.x, BIG.y - 10, 'patch 6, enlarged', { opacity: ghost }); }
  embedBlock(svg, arriving(p));
  pixelCells(svg, flat);
  if (flat >= 1) mark(svg, FLAT.x - 2, FLAT.y - 2, 16 * FLAT.cell + 3, FLAT.cell + 3);
  else mark(svg, BIG.x, BIG.y, BIG.size, BIG.size, 1 - flat);
  note(svg, FLAT.x, FLAT.y - 12, 'patch 6 flattened: 16 greys', { opacity: flat });
  const into = seg(p, 0.4, 0.55);
  if (into > 0 && into < 1) G.flow(svg, { from: [FLAT.x + 80, FLAT.y + 12], to: [FLAT.x + 80, EMBED.y - 3], carry: 'activation', progress: into });
  const out = seg(p, 0.55, 0.66);
  if (out > 0 && out < 1) G.flow(svg, { from: [FLAT.x + 80, EMBED.y + EMBED.h + 1], to: [FLAT.x + 80, EMBED_VEC.y - 3], carry: 'activation', progress: out });
  const slide = seg(p, 0.66, 0.82);
  embedVector(svg, p, slide);
  const rows = seg(p, 0.82, 0.95);
  zMatrix(svg, Z0, { scale: SCALE.z0, visible: (i) => slide >= 1 && i === FOLLOWED, opacity: arriving(p) });
  if (rows > 0) zMatrix(svg, Z0, { scale: SCALE.z0, opacity: rows, label: '' });
  if (slide >= 1) mark(svg, Z.x, Z.y + FOLLOWED * Z.cell, 8 * Z.cell, Z.cell);
  noteLines(svg, GRID.x, NOTE_Y[0], ['16 greys → 8 numbers · matrix [16 × 8] = 128 parameters', '16 patches → 16 patch vectors'], seg(p, 0.85, 1));
}

// ---- frame 3 ----
const ENC_HEAD = Object.freeze({ x: ENCODER.x, y: 22, w: ENCODER.w, h: 28 });
const HEAT = Object.freeze({ x: 180, y: 84, cell: 11 });
const HEAT_SIDE = 16 * HEAT.cell;

// Z moves from the embedded vectors to the encoded ones, cell by cell between two real states.
const blend = (t) => Z0.map((row, i) => row.map((v, j) => lerp(v, Z1[i][j], t)));

// The encoder column: header block, the [16 × 16] pattern (`rowsRead` rows filled) and its notes. Frame 3's marks.
function encoderParts(parent, { enter, rowsRead, text }) {
  G.block(layer(parent, enter), { ...ENC_HEAD, label: 'vision encoder', state: 'active' });
  const values = PATTERN.map((row, i) => (i < rowsRead ? row : row.map(() => Number.NaN)));
  G.heatmap(layer(parent, enter), { x: HEAT.x, y: HEAT.y, values, cell: HEAT.cell, maxAbs: SCALE.pattern, label: 'patch pairs' });
  noteLines(parent, ENCODER.x + 12, HEAT.y + HEAT_SIDE + 16, ['within one image,', 'every patch sees every patch', '16 × 16 = 256 patch pairs read'], text);
  noteLines(parent, GRID.x, NOTE_Y[0], ['no causal mask: every cell of the pattern is read'], text);
}

// Frame 3's end state without Z, for frame 4 to fade out.
export const encoderScene = (parent, opacity) => encoderParts(parent, { enter: opacity, rowsRead: 16, text: opacity });

export function drawFrame3(svg, p) {
  note(svg, GRID.x, IMAGE_LABEL_Y, 'image: 16 × 16 px');
  patchGrid(svg);
  const gone = leaving(p);
  if (gone > 0) embedScene(svg, gone);
  encoderParts(svg, { enter: arriving(p), rowsRead: Math.floor(seg(p, 0.15, 0.5) * 16 + 1e-9), text: seg(p, 0.5, 0.8) });
  const dots = seg(p, 0.12, 0.45);
  if (dots > 0 && dots < 1) G.flow(svg, { from: [Z.x - 4, Z.y + 8 * Z.cell], to: [HEAT.x + HEAT_SIDE + 3, HEAT.y + HEAT_SIDE / 2], carry: 'activation', progress: dots });
  zMatrix(svg, blend(seg(p, 0.5, 0.95)), { scale: SCALE.z1, label: 'patch vectors Z' });
  mark(svg, Z.x, Z.y + FOLLOWED * Z.cell, 8 * Z.cell, Z.cell);
}
