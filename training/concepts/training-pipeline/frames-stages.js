// training-pipeline frames 1–7: the six stages left to right (storyboard §5). Each drawFrameN(svg, p) is a pure function of
// p; its end state is the next frame's start.
import * as G from '@shared/glyphs.js';
import { STAGES, REPLIES, SPECIALISTS, STAGE_FIGURES } from './numbers.js';
import {
  ROW, MERGED, MERGED_X, SPEC, seg, lerp, arriving, leaving, layer, note, typed, stageRow, selectStage, checkpointChip, replyRow, blockCenter, chipRow,
} from './stage.js';

// Stage i is active; earlier stages are done (idle), later ones wait (dim).
const statesFor = (active) => STAGES.map((_, i) => (i < active ? 'idle' : i === active ? 'active' : 'dim'));
const fade = (words, t) => (i) => (i < words.length ? t : 0);
const SPEC_X0 = (580 - (3 * SPEC.w + 2 * SPEC.gap)) / 2;
const specX = (i) => SPEC_X0 + i * (SPEC.w + SPEC.gap);
const specCenter = (i) => specX(i) + SPEC.w / 2;
const SPEC_LABEL_Y = SPEC.y + SPEC.h + 14;

// Reply words cross-fade: `from` fades out over [a, b], `to` types in over [b, c].
function replies(svg, from, to, [a, b, c], p) {
  const out = 1 - seg(p, a, b);
  if (out > 0) replyRow(layer(svg, out), { words: from });
  const into = to.length ? to.map((_, i) => seg(p, b + ((c - b) * i) / to.length, b + ((c - b) * (i + 1)) / to.length)) : [];
  if (into.some((o) => o > 0)) replyRow(svg, { words: to, opacity: (i) => into[i] });
}

export function drawFrame1(svg, p) {
  const t = (i) => seg(p, i * 0.08, i * 0.08 + 0.25);
  stageRow(svg, statesFor(-1), { opacity: t });
  note(layer(svg, seg(p, 0.5, 0.8)), 6, ROW.y + ROW.h + 16, 'not every lab runs all six (see the toy)');
  const model = layer(svg, seg(p, 0.35, 0.75));
  G.blockStack(model, { x: 14, y: 120, w: 170, count: 2, shown: 2, lastLabel: 'block N' });
  G.selectionMark(model, { x: 14, y: 120, w: 170, h: 176 });
  note(layer(svg, seg(p, 0.5, 0.85)), 200, 190, 'the decoder');
  note(layer(svg, seg(p, 0.5, 0.85)), 200, 206, '(random weights)');
  note(layer(svg, seg(p, 0.5, 0.85)), 200, 150, '6 stages', { cls: 'g-label' });
  replyRow(layer(svg, seg(p, 0.6, 1)), { words: REPLIES.random, label: '(random weights, illustrative)' });
}

export function drawFrame2(svg, p) {
  stageRow(svg, statesFor(0));
  const lost = leaving(p);
  if (lost > 0) { // the decoder fades as the stream arrives
    const model = layer(svg, lost);
    G.blockStack(model, { x: 14, y: 120, w: 170, count: 2, shown: 2, lastLabel: 'block N' });
    G.selectionMark(model, { x: 14, y: 120, w: 170, h: 176 });
  }
  selectStage(svg, 0, arriving(p));
  checkpointChip(svg, 0, seg(p, 0.55, 0.85));
  const stream = layer(svg, arriving(p));
  chipRow(stream, ['web', 'code', 'books', 'papers'], { x: 14, y: 196 });
  G.flow(stream, { from: [blockCenter(0), 190], to: [blockCenter(0), ROW.y + ROW.h + 6], carry: 'token', progress: seg(p, 0.1, 0.7) });
  note(stream, 76, 150, STAGE_FIGURES.pretrainRange);
  note(stream, 76, 166, '(2026 open frontier MoEs)');
  note(stream, 14, 246, 'adds: knowledge, language, code');
  replies(svg, REPLIES.random, REPLIES.base, [0.3, 0.5, 0.9], p);
}

export function drawFrame3(svg, p) {
  stageRow(svg, statesFor(1));
  selectStage(svg, 0, leaving(p));
  selectStage(svg, 1, arriving(p));
  checkpointChip(svg, lerp(0, 1, seg(p, 0.05, 0.4)));
  const grow = seg(p, 0.25, 0.9);
  note(svg, 14, 136, `best data · longer context: ${STAGE_FIGURES.midTrainContext} (GLM-5)`);
  G.block(svg, { x: 14, y: 150, w: 260, h: 18, label: '', state: 'dim' });
  G.block(svg, { x: 14, y: 150, w: lerp(12, 260, grow), h: 18, label: '', state: 'active' });
  note(svg, 14, 186, '4K');
  note(svg, 274, 186, '200K', { anchor: 'end' });
  note(svg, 14, 232, 'adds: long context, reasoning-heavy data');
  replyRow(svg, { words: REPLIES.base });
}

export function drawFrame4(svg, p) {
  stageRow(svg, statesFor(2));
  selectStage(svg, 1, leaving(p));
  selectStage(svg, 2, arriving(p));
  checkpointChip(svg, lerp(1, 2, seg(p, 0.05, 0.4)));
  note(svg, 14, 136, 'worked conversations, in a chat template');
  note(svg, 14, 232, 'adds: format, readable reasoning');
  const into = REPLIES.chat.map((_, i) => seg(p, 0.35 + i * 0.15, 0.5 + i * 0.15));
  const out = 1 - seg(p, 0.1, 0.35);
  if (out > 0) replyRow(layer(svg, out), { words: REPLIES.base });
  replyRow(svg, { words: REPLIES.chat, opacity: (i) => into[i] });
}

function specialists(svg, { opacity = 1, lit = -1, flows = 0 }) {
  SPECIALISTS.forEach((s, i) => {
    const g = layer(svg, opacity);
    G.block(g, { x: specX(i), y: SPEC.y, w: SPEC.w, h: SPEC.h, label: s.name, state: lit === i ? 'active' : 'idle' });
    note(g, specCenter(i), SPEC_LABEL_Y, s.under, { anchor: 'middle' });
    if (flows > 0) G.flow(g, { from: [blockCenter(3), ROW.y + ROW.h + 4], to: [specCenter(i), SPEC.y - 3], carry: 'activation', progress: flows });
  });
}

export function drawFrame5(svg, p) {
  stageRow(svg, statesFor(3));
  selectStage(svg, 2, leaving(p));
  selectStage(svg, 3, arriving(p));
  checkpointChip(svg, lerp(2, 3, seg(p, 0.05, 0.3)));
  const cycle = seg(p, 0.3, 0.97);
  specialists(svg, { opacity: seg(p, 0.1, 0.35), lit: cycle > 0 && cycle < 1 ? Math.floor(cycle * 3) : -1, flows: seg(p, 0.1, 0.35) });
  note(layer(svg, seg(p, 0.35, 0.6)), SPEC_X0, 224, 'adds: reasoning and agent skill from checked tasks');
  replyRow(svg, { words: REPLIES.chat });
}

export function mergedBlock(svg, { opacity, state, flows = 1, withSpecialists = 1 }) {
  if (withSpecialists > 0) {
    specialists(layer(svg, withSpecialists), {});
    SPECIALISTS.forEach((_, i) => {
      G.flow(layer(svg, withSpecialists), { from: [specCenter(i), SPEC_LABEL_Y + 6], to: [MERGED_X + (i + 1) * (MERGED.w / 4), MERGED.y - 3], carry: 'activation', progress: flows });
    });
  }
  if (opacity > 0) G.block(layer(svg, opacity), { x: MERGED_X, y: MERGED.y, w: MERGED.w, h: MERGED.h, label: 'one model', state });
}

export function drawFrame6(svg, p) {
  stageRow(svg, statesFor(4));
  selectStage(svg, 3, leaving(p));
  selectStage(svg, 4, arriving(p));
  checkpointChip(svg, lerp(3, 4, seg(p, 0.05, 0.3)));
  mergedBlock(svg, { opacity: seg(p, 0.3, 0.75), state: 'active', flows: seg(p, 0.1, 0.7) });
  note(layer(svg, seg(p, 0.5, 0.85)), 6, 296, 'adds: one model with each specialist\'s best skills');
  replyRow(svg, { words: REPLIES.chat });
}

// The polish stage's labels beside the merged block (frame 7; frame 8 fades them out).
export function polishLabels(svg, t, opacity = 1) {
  const g = layer(svg, opacity);
  const x = MERGED_X + MERGED.w + 12;
  note(g, x, MERGED.y + 10, typed('style and safety', seg(t, 0, 0.45)));
  note(g, x, MERGED.y + 24, typed('(judge-model RL)', seg(t, 0, 0.45)));
  note(g, x, MERGED.y + 38, typed('prepared for serving', seg(t, 0.5, 1)));
}

export function drawFrame7(svg, p) {
  stageRow(svg, statesFor(5));
  selectStage(svg, 4, leaving(p));
  selectStage(svg, 5, arriving(p));
  checkpointChip(svg, lerp(4, 5, seg(p, 0.05, 0.3)));
  mergedBlock(svg, { opacity: 1, state: 'idle', withSpecialists: leaving(p) });
  polishLabels(svg, seg(p, 0.2, 0.9));
  replyRow(svg, { words: REPLIES.chat });
}
