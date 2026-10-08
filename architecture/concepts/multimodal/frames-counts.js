// multimodal frames 7–9 (storyboard §5): what an image and a video cost, with real sizes. `ctx` carries the model facts
// (patch size, merge, largest side) the page reads from data/models.json; the stage never types them.
import * as G from '@shared/glyphs.js';
import { visionTokens, patchGrid } from '@math/vision.js';
import * as S from './stage.js';
import { drawFrame6 } from './frames-merge.js';
import { CONTEXT_1M, PHONE, WIDE, VIDEO_FRAME, int, shareText, imagesThatFit, framesOf } from './format.js';

const { seg, layer, note, noteLines, NOTE_Y, leaving, linked } = S;
const LINE = 18;
const COARSE = 9; // coarse sketch cells: they encode nothing, the counts are printed

// A grid of identical neutral cells standing for a patch grid (drawn coarsely; the counts are exact).
function coarseGrid(parent, x, y, cols, rows, cell = COARSE, opacity = 1) {
  G.matrix(layer(parent, opacity), { x, y, values: Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0)), cell, maxAbs: 1 });
}

// A text line that a math term names: wrapped in a linked group so the math panel outlines it on hover.
function linkedNote(parent, letter, x, y, text, opacity) {
  if (opacity <= 0) return;
  const g = linked(layer(parent, opacity), letter, { x, y: y - 8, w: text.length * 6.6, h: 16 });
  note(g, x, y, text);
}

// The counts printed under a photo: sides, patches, merge, tokens, share (and how many fit).
function photoLines({ width, height, patch, merge }, context = CONTEXT_1M) {
  const grid = patchGrid({ width, height, patch });
  const v = visionTokens({ width, height, patch, merge });
  const lines = [
    { text: `${width} ÷ ${patch} = ${grid.cols}` },
    { text: `${grid.cols} × ${grid.rows} = ${int(grid.patches)} patches` },
    { text: `÷ ${merge * merge} (${merge} × ${merge} merge)`, link: 'merge' },
    { text: `→ ${int(v.tokensPerFrame)} tokens`, link: 'tok' },
    { text: `${shareText(v.tokensPerFrame, context)} of ${int(context)}` },
  ];
  return { lines, v };
}

function paintLines(parent, x, y, lines, start, opacityOf) {
  lines.forEach((line, j) => {
    const opacity = opacityOf(start + j);
    if (line.link) linkedNote(parent, line.link, x, y + j * LINE, line.text, opacity);
    else note(parent, x, y + j * LINE, line.text, { opacity });
  });
}

// ---- frame 7: tokens grow with area ----
const PANEL = Object.freeze({ left: 24, right: 300, titleY: 80, gridY: 96 });
const BIG_GRID = 14; // the largest input is drawn about 3.5 times the phone photo's side

export function drawFrame7(svg, p, ctx) {
  const gone = leaving(p);
  if (gone > 0) drawFrame6(layer(svg, gone), 1);
  const { patch, merge, maxSide } = ctx;
  const phone = photoLines({ ...PHONE, patch, merge });
  const largest = photoLines({ width: maxSide, height: maxSide, patch, merge });
  note(svg, 16, 24, 'How many tokens is a phone photo?', { opacity: seg(p, 0.1, 0.3) });
  note(svg, 16, 42, `real sizes now, not the toy image: patch ${patch}, ${merge} × ${merge} merge`, { opacity: seg(p, 0.1, 0.3) });
  const left = (i) => seg(p, 0.45 + 0.05 * i, 0.55 + 0.05 * i);
  note(svg, PANEL.left, PANEL.titleY, `${int(PHONE.width)} × ${int(PHONE.height)} photo`, { opacity: left(0) });
  coarseGrid(svg, PANEL.left, PANEL.gridY, 4, 4, COARSE, left(0));
  paintLines(svg, PANEL.left, PANEL.gridY + 36 + 22, phone.lines, 0, left);
  const right = (i) => seg(p, 0.7 + 0.05 * i, 0.8 + 0.05 * i);
  note(svg, PANEL.right, PANEL.titleY, `${int(maxSide)} × ${int(maxSide)}, the largest input`, { opacity: right(-1) });
  coarseGrid(svg, PANEL.right, PANEL.gridY, BIG_GRID, BIG_GRID, COARSE, right(-1));
  const textY = PANEL.gridY + BIG_GRID * COARSE + 20;
  paintLines(svg, PANEL.right, textY, largest.lines, 0, right);
  note(svg, PANEL.right, textY + 5 * LINE, `${imagesThatFit(CONTEXT_1M, largest.v.tokensPerFrame)} of these fill a 1M window`, { opacity: right(5) });
  note(svg, 16, NOTE_Y[1], 'the grids are coarse sketches; the counts are exact', { opacity: left(0) });
}

// ---- frame 8: dynamic resolution ----
const WIDE_LEFT = 40;
const WIDE_RIGHT = 300;
const WIDE_GRID_Y = 100;
const SQUASH_CELL = 10;

// The two lines frame 8 compares: the patch count and the token count of a photo.
const summary = (parent, x, y, lines, opacity) => [lines[1], lines[3]].forEach((line, j) => note(parent, x, y + j * LINE, line.text, { opacity }));

export function drawFrame8(svg, p, ctx) {
  const gone = leaving(p);
  if (gone > 0) drawFrame7(layer(svg, gone), 1, ctx);
  const { patch, merge } = ctx;
  const squashed = photoLines({ ...PHONE, patch, merge });
  const kept = photoLines({ ...WIDE, patch, merge });
  const head = seg(p, 0.12, 0.3);
  note(svg, 16, 24, `a wide photo: ${int(WIDE.width)} × ${int(WIDE.height)} pixels`, { opacity: head });
  note(svg, WIDE_LEFT, 62, 'stretched to a square', { opacity: head });
  coarseGrid(svg, WIDE_LEFT, WIDE_GRID_Y, 8, 8, SQUASH_CELL, head);
  const textY = WIDE_GRID_Y + 8 * SQUASH_CELL + 22;
  summary(svg, WIDE_LEFT, textY, squashed.lines, head);
  // The copy on the right starts as the square and deflates into the wide grid: its lower half fades out.
  const copy = seg(p, 0.3, 0.5);
  const deflate = seg(p, 0.55, 0.85);
  note(svg, WIDE_RIGHT, 62, 'kept at its shape', { opacity: seg(p, 0.8, 1) });
  coarseGrid(svg, WIDE_RIGHT, WIDE_GRID_Y, 8, 4, SQUASH_CELL, copy);
  coarseGrid(svg, WIDE_RIGHT, WIDE_GRID_Y + 4 * SQUASH_CELL, 8, 4, SQUASH_CELL, copy * (1 - deflate));
  const before = copy * (1 - deflate);
  summary(svg, WIDE_RIGHT, textY, squashed.lines, before);
  summary(svg, WIDE_RIGHT, textY, kept.lines, deflate);
  noteLines(svg, 16, NOTE_Y[0], [`${int(WIDE.width)} × ${int(WIDE.height)} at patch ${patch}: ${kept.lines[1].text} → ${int(kept.v.tokensPerFrame)} tokens`, `(against ${int(squashed.v.tokensPerFrame)} as a square)`], seg(p, 0.85, 1));
}

// ---- frame 9: video pays for every frame ----
const FILM = Object.freeze({ x: 16, y: 46, thumb: 24, stride: 36, count: 12 });
const BARS = Object.freeze({ x: 40, y: 168, w: 200, h: 120 });

export function drawFrame9(svg, p, ctx) {
  const gone = leaving(p);
  if (gone > 0) drawFrame8(layer(svg, gone), 1, ctx);
  const { patch, merge } = ctx;
  const frame = visionTokens({ ...VIDEO_FRAME, patch, merge });
  const minute = visionTokens({ ...VIDEO_FRAME, patch, merge, frames: framesOf({ media: 'video', seconds: 60, fps: 2 }) });
  const hour = visionTokens({ ...VIDEO_FRAME, patch, merge, frames: framesOf({ media: 'video', seconds: 3600, fps: 2 }) });
  note(svg, 16, 24, `video: ${VIDEO_FRAME.width} × ${VIDEO_FRAME.height} frames, 2 per second`, { opacity: seg(p, 0.1, 0.25) });
  for (let i = 0; i < FILM.count; i += 1) {
    const grid = layer(svg, seg(p, 0.15 + 0.03 * i, 0.25 + 0.03 * i));
    G.matrix(grid, { x: FILM.x + i * FILM.stride, y: FILM.y, values: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], cell: 8, maxAbs: 1 });
  }
  note(svg, FILM.x, FILM.y + 44, `${int(frame.tokensPerFrame)} tokens per frame`, { opacity: seg(p, 0.5, 0.65) });
  const bars = layer(svg, seg(p, 0.6, 0.8));
  G.bars(bars, { ...BARS, values: [minute.tokens, hour.tokens], labels: ['1 minute', '1 hour'], max: hour.tokens, reference: { value: CONTEXT_1M, label: '1M context' }, format: int, label: 'video tokens' });
  const right = seg(p, 0.75, 0.95);
  noteLines(svg, 330, BARS.y + 10, [
    `1 minute: ${int(minute.tokens / frame.tokensPerFrame)} frames`, `= ${int(minute.tokens)} tokens (${shareText(minute.tokens, CONTEXT_1M)})`,
    `1 hour: ${int(hour.tokens / frame.tokensPerFrame)} frames`, `= ${int(hour.tokens)} tokens (${shareText(hour.tokens, CONTEXT_1M)})`,
  ], right);
  noteLines(svg, 16, NOTE_Y[0], [`toy settings (${VIDEO_FRAME.width} × ${VIDEO_FRAME.height}, 2 fps, ${merge} × ${merge}); real encoders also pool over time`], seg(p, 0.85, 1));
}
