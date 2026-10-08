// Frames 9 and 10: one update's two stretch factors (singular values), then Muon's Newton-Schulz steps pull them to one.
import * as G from '@shared/glyphs.js';
import { newtonSchulzSingular } from '@math/scaling.js';
import { SINGULAR_VALUES, MUON_SCHEDULE, MUON_COEFFICIENTS } from './numbers.js';
import { coefficientText } from './format.js';
import { seg, lerp, ease, label, ink, select, linked } from './stage.js';

const TRACE = Object.freeze(newtonSchulzSingular(SINGULAR_VALUES, MUON_SCHEDULE).map((row) => Object.freeze(row))); // 11 rows: step 0 is the normalized input
const STEPS = MUON_SCHEDULE.length;
const SCALE = 3; // value-scale maximum for both frames (the largest raw singular value)
const CELL = G.NUMBER_CELL;
const FIRST = Object.freeze({ norm2: SINGULAR_VALUES.reduce((sum, s) => sum + s * s, 0) });
const VECTOR = Object.freeze({ x: 250, y: 92 });
const THREE = (v) => v.toFixed(3);
const TWO = (v) => v.toFixed(2);

function stretchVector(svg, values, { format, y = VECTOR.y, x = VECTOR.x }) {
  const parent = linked(svg, 'n', { x, y, w: 2 * CELL, h: CELL });
  G.vector(parent, { x, y, values, cell: CELL, orient: 'row', maxAbs: SCALE, format });
  select(svg, x, y, CELL, CELL); // the larger singular value is the followed item
}

// ---- frame 9: the update's stretch factors, then normalized together ----
export function drawFrame9(svg, p) {
  const fill = seg(p, 0, 0.3);
  const norm = ease(seg(p, 0.45, 0.85));
  const [a, b] = SINGULAR_VALUES;
  const norm0 = TRACE[0];
  G.block(svg, { x: 16, y: 20, w: 240, h: 34, label: 'update for one weight matrix' });
  label(svg, VECTOR.x, VECTOR.y - 14, 'stretch per direction (singular values)', { opacity: fill });
  if (fill > 0) {
    const values = [lerp(a, norm0[0], norm), lerp(b, norm0[1], norm)];
    stretchVector(svg, values, { format: norm > 0 ? THREE : TWO });
  }
  label(svg, 16, 168, `[${TWO(a)}, ${TWO(b)}] ÷ √${TWO(FIRST.norm2)} = [${THREE(norm0[0])}, ${THREE(norm0[1])}] (÷ the update's overall size)`, { opacity: norm });
  ink(svg, 16, 206, 'momentum step: the 10 : 1 stretch stays', { opacity: seg(p, 0.6, 0.9) });
  label(svg, 16, 224, '(AdamW rescales elements, not directions)', { opacity: seg(p, 0.7, 1) });
}

// ---- frame 10: ten Newton-Schulz steps ----
const TABLE = Object.freeze({ headY: 150, firstY: 168, rowH: 15, stepX: 16, largerX: 130, smallerX: 210, coefX: 290 });
const FINISH_FROM = MUON_SCHEDULE.length - MUON_COEFFICIENTS.finishSteps + 1; // the first step that uses the finishing coefficients
const rowText = (row, step) => row.map((v) => (step === STEPS ? v.toFixed(4) : THREE(v)));
const stepLabel = (step) => (step === 0 ? 'input' : `step ${step}`);
const coefsFor = (step) => (step >= FINISH_FROM ? coefficientText(MUON_COEFFICIENTS.finish) : coefficientText(MUON_COEFFICIENTS.bulk, 4));

export function drawFrame10(svg, p) {
  const pos = p * STEPS;
  const k = Math.min(Math.floor(pos), STEPS - 1);
  const t = ease(pos - k);
  const values = TRACE[k].map((v, i) => lerp(v, TRACE[k + 1][i], t));
  const reached = t >= 0.5 ? k + 1 : k;
  label(svg, 16, 28, 'Muon: Newton–Schulz iterations');
  ink(svg, 16, 56, `step ${reached} of ${STEPS}`);
  stretchVector(svg, values, { format: THREE, y: 44, x: 250 });
  label(svg, TABLE.stepX, TABLE.headY, 'step');
  label(svg, TABLE.largerX, TABLE.headY, 'larger', { anchor: 'end' });
  label(svg, TABLE.smallerX, TABLE.headY, 'smaller', { anchor: 'end' });
  label(svg, TABLE.coefX, TABLE.headY, 'coefficients (a, b, c)');
  TRACE.forEach((row, step) => {
    const opacity = step <= k ? 1 : step === k + 1 ? t : 0;
    if (opacity <= 0) return;
    const y = TABLE.firstY + step * TABLE.rowH;
    const [larger, smaller] = rowText(row, step);
    label(svg, TABLE.stepX, y, stepLabel(step), { opacity });
    ink(svg, TABLE.largerX, y, larger, { anchor: 'end', opacity });
    ink(svg, TABLE.smallerX, y, smaller, { anchor: 'end', opacity });
    if (step > 0) label(svg, TABLE.coefX, y, coefsFor(step), { opacity });
  });
  const last = TRACE[STEPS];
  label(svg, 16, TABLE.firstY + (STEPS + 1) * TABLE.rowH + 6, `step ${STEPS} is exactly ${last[0].toFixed(6)} and ${last[1].toFixed(6)}: within 1e-4 of 1`, { opacity: seg(p, 0.96, 1) });
}
