// pretraining frames 6–9: the data pipeline's fixed layout and drawing helpers. The seven stages stand in a column (the
// labels are too long for one row at 580 px); documents sit in a row beside the stage that last passed them.
import * as G from '@shared/glyphs.js';
import { DIM, lerp, layer, note } from './stage.js';

export const BLOCKS = Object.freeze(['crawl', 'rules', 'quality classifier', 'dedup', 'mixture', 'rephrase', 'packed sequences']);
export const STAGES = Object.freeze({ crawl: 0, rules: 1, classifier: 2, dedup: 3, mixture: 4, rephrase: 5, packed: 6 });
const PIPE = Object.freeze({ x: 16, w: 140, h: 26, y0: 20, pitch: 38 });
const ROW = Object.freeze({ x: 176, step: 34 });
const FLOW_X = 166; // the stream's arrow, between the stages and their document rows
const FACT_Y = Object.freeze([306, 326]);
export const LANES = Object.freeze(['web', 'code', 'math', 'knowledge']);
const LANE = Object.freeze({ labelX: ROW.x, x: 246, y0: 134, pitch: 26 });

export const blockY = (k) => PIPE.y0 + k * PIPE.pitch;
export const rowY = (k) => blockY(k) + 1; // a 24 px chip centered on a 26 px stage
export const docX = (slot) => ROW.x + slot * ROW.step;
export const laneY = (i) => LANE.y0 + i * LANE.pitch;
export const laneX = (slot) => LANE.x + slot * ROW.step;
export const rowEnd = (slots) => docX(slots) + 4; // where a row's note starts

// One state per stage: 'active' (this frame's stage), 'idle' (passed), 'dim' (waiting). `from` cross-fades the change.
export function pipeline(parent, states, { from = states, t = 1 } = {}) {
  BLOCKS.forEach((label, k) => {
    const box = { x: PIPE.x, y: blockY(k), w: PIPE.w, h: PIPE.h, label };
    if (from[k] === states[k] || t >= 1) G.block(parent, { ...box, state: states[k] });
    else {
      G.block(layer(parent, 1 - t), { ...box, state: from[k] });
      if (t > 0) G.block(layer(parent, t), { ...box, state: states[k] });
    }
  });
}

// The stream down the pipeline; its dot sits at height y (the documents' current row).
export function stream(parent, y) {
  const top = blockY(0);
  const bottom = blockY(BLOCKS.length - 1) + PIPE.h;
  G.flow(parent, { from: [FLOW_X, top], to: [FLOW_X, bottom], carry: 'token', progress: (y - top) / (bottom - top - 7) });
}

// A document chip (the token glyph) at (x, y); dim = dropped.
export function doc(parent, { x, y, label, dim = 0, opacity = 1 }) {
  if (opacity <= 0) return;
  G.token(layer(parent, opacity * lerp(1, DIM, dim)), { x, y, text: label });
}

export function factLines(parent, lines) {
  lines.forEach((text, i) => note(parent, PIPE.x, FACT_Y[i], text, { cls: '' }));
}

export function laneLabels(parent) {
  LANES.forEach((name, i) => note(parent, LANE.labelX, laneY(i) + 16, name));
}
