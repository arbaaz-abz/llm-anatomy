// Frames 8–9: where SFT data comes from, and what the stage is for (storyboard §5).
import * as G from '@shared/glyphs.js';
import { lossMask } from '@math/sft.js';
import { CHIPS, NOTE_Y, HANDOFF, seg, layer, note, roleLabels, drawChips, arriving, leaving } from './stage.js';
import { SEGMENTS, CANDIDATES, STAGES, RL_STAGE } from './numbers.js';

// ---- frame 8: an earlier model proposes traces, a checker keeps the ones that pass ----
const SPECIALIST = Object.freeze({ x: 16, y: 126, w: 124, h: 44 });
const TRACE = Object.freeze({ x: 190, w: 96, h: 30, y0: 40, pitch: 62 });
const VERDICT_X = 318;
const PILE = Object.freeze({ x: 456, y: 126, w: 108, h: 44 });
const traceY = (i) => TRACE.y0 + i * TRACE.pitch;
const mid = (box) => box.y + box.h / 2;

export function dataBody(svg, p, opacity = 1) {
  const g = layer(svg, opacity);
  G.block(g, { ...SPECIALIST, label: 'earlier specialist', state: 'idle' });
  G.block(g, { ...PILE, label: 'SFT data', state: 'idle' });
  note(g, VERDICT_X, TRACE.y0 - 22, 'checker', { anchor: 'middle' });
  CANDIDATES.forEach((c, i) => {
    const appear = seg(p, 0.05 + i * 0.1, 0.2 + i * 0.1);
    const stamp = seg(p, 0.5 + i * 0.07, 0.6 + i * 0.07);
    const fade = c.ok ? 1 : 1 - 0.7 * seg(p, 0.75, 0.9);
    const row = layer(g, appear * fade);
    G.block(row, { x: TRACE.x, y: traceY(i), w: TRACE.w, h: TRACE.h, label: c.label, state: 'idle' });
    if (appear > 0) G.flow(row, { from: [SPECIALIST.x + SPECIALIST.w + 2, mid(SPECIALIST)], to: [TRACE.x - 3, traceY(i) + TRACE.h / 2], carry: 'token', progress: appear });
    if (stamp > 0) G.verdict(layer(g, stamp * fade), { x: VERDICT_X, y: traceY(i) + TRACE.h / 2, ok: c.ok });
    const into = c.ok ? seg(p, 0.7 + i * 0.03, 0.9) : 0;
    if (into > 0) G.flow(g, { from: [VERDICT_X + 16, traceY(i) + TRACE.h / 2], to: [PILE.x - 3, mid(PILE) + (i - 1) * 6], carry: 'token', progress: into });
  });
  const kept = CANDIDATES.filter((c) => c.ok).length;
  note(g, 76, NOTE_Y[1] + 20, `${kept} of ${CANDIDATES.length} kept (illustrative)`, { opacity: seg(p, 0.9, 1) });
}

export function drawFrame8(svg, p) {
  dataBody(svg, p, 1);
}

// ---- frame 9: the key frame, SFT gives RL its cold start ----
const MODEL_BOX = Object.freeze({ x: 76, y: 244, w: 110, h: 30 });
const STRIP = Object.freeze({ x: 4, y: 304, w: 92, h: 30, gap: 4 });
const stageX = (i) => STRIP.x + i * (STRIP.w + STRIP.gap);
const MASKED = lossMask(SEGMENTS, { maskError: true }).map((trained) => !trained);

export function drawFrame9(svg, p) {
  if (p < HANDOFF) dataBody(svg, 1, leaving(p));
  const fade = arriving(p);
  const scene = layer(svg, fade);
  roleLabels(scene);
  drawChips(scene, CHIPS, { hatchedOf: (chip) => MASKED[chip.flat] });
  G.block(scene, { ...MODEL_BOX, label: 'SFT model', state: 'idle' });
  const lit = seg(p, 0.5, 0.7) > 0.5;
  STAGES.forEach((label, i) => G.block(scene, { x: stageX(i), y: STRIP.y, w: STRIP.w, h: STRIP.h, label, state: i === RL_STAGE && lit ? 'active' : 'dim' }));
  const reach = seg(p, 0.3, 0.6);
  const target = { x: stageX(RL_STAGE) + STRIP.w / 2, y: STRIP.y - 2 };
  if (reach > 0) G.flow(scene, { from: [MODEL_BOX.x + MODEL_BOX.w + 4, MODEL_BOX.y + MODEL_BOX.h / 2 + 4], to: [target.x, target.y], carry: 'token', progress: reach });
  note(scene, MODEL_BOX.x + MODEL_BOX.w + 24, NOTE_Y[0] + 2, 'sometimes right → RL has something to reinforce', { opacity: seg(p, 0.7, 0.9) });
}

