// Frames 8–9: a multi-token-prediction head drafts the token after next, and the next step checks the draft
// (storyboard §5). Each frame is a pure function of its progress p (0 → 1); frame 9 starts where frame 8 ends.
import * as G from '@shared/glyphs.js';
import { HANDOFF, CHIP_Y, SLOT_X, seg, lerp, ease, arriving, leaving, layer, note, chips, slotChip, selectChip } from './stage.js';
import { drawsBody } from './frames-shape.js';

const NOTE_ROW = 62;
const ROW_X_LABEL = 36; // the "in some 2026 models" label lines up with the stack
const STACK = Object.freeze({ x: 36, y: 90, w: 130, h: 82 }); // the last block, drawn by blockStack (count 1)
const HEADS = Object.freeze({
  unembedding: { x: 226, y: 86, w: 112, h: 34, label: 'unembedding' },
  mtp: { x: 226, y: 146, w: 112, h: 34, label: 'MTP head' },
});
const OUT_X = 392; // the output chips, right of the heads
const OUT_Y = Object.freeze({ pick: 91, draft: 151, meet: 121 });
const LABEL_X = OUT_X + 46;
const CENTER = (box) => box.y + box.h / 2;
const STACK_MID = STACK.y + STACK.h / 2;
const FLOWS = Object.freeze({
  toUnembedding: { from: [STACK.x + STACK.w + 2, STACK_MID - 12], to: [HEADS.unembedding.x - 2, CENTER(HEADS.unembedding)] },
  toMtp: { from: [STACK.x + STACK.w + 2, STACK_MID + 12], to: [HEADS.mtp.x - 2, CENTER(HEADS.mtp)] },
  toPick: { from: [HEADS.unembedding.x + HEADS.unembedding.w + 2, CENTER(HEADS.unembedding)], to: [OUT_X - 4, CENTER(HEADS.unembedding)] },
  toDraft: { from: [HEADS.mtp.x + HEADS.mtp.w + 2, CENTER(HEADS.mtp)], to: [OUT_X - 4, CENTER(HEADS.mtp)] },
});
const VERDICT = Object.freeze({ x: 506, y: (OUT_Y.meet + OUT_Y.draft + 24) / 2 });

const flow = (svg, key, progress) => G.flow(svg, { ...FLOWS[key], carry: 'activation', progress });
const head = (svg, box, opacity) => G.block(layer(svg, opacity), { ...box, state: 'idle' });
const lastBlock = (svg) => G.blockStack(svg, { x: STACK.x, y: STACK.y, w: STACK.w, count: 1, shown: 1, lastLabel: 'last block' });

// ---- frame 8: one pass, two heads: the model's output and a draft of the token after next ----
export function mtpBody(svg, p, opacity = 1) {
  const inner = layer(svg, opacity);
  note(inner, ROW_X_LABEL, NOTE_ROW, 'in some 2026 models');
  lastBlock(inner);
  head(inner, HEADS.unembedding, 1);
  head(inner, HEADS.mtp, 1);
  const into = seg(p, 0.15, 0.4);
  const out = seg(p, 0.4, 0.65);
  if (into > 0) { flow(inner, 'toUnembedding', into); flow(inner, 'toMtp', into); }
  if (out > 0) { flow(inner, 'toPick', out); flow(inner, 'toDraft', out); }
  const landed = seg(p, 0.6, 0.7);
  slotChip(inner, 'five', 'on', { at: { x: OUT_X, y: OUT_Y.pick }, opacity: landed });
  slotChip(inner, 'six', 'the', { state: 'draft', at: { x: OUT_X, y: OUT_Y.draft }, opacity: landed });
  note(inner, LABEL_X, OUT_Y.draft + 12, 'draft', { opacity: seg(p, 0.7, 0.8) });
  note(inner, HEADS.mtp.x, HEADS.mtp.y + HEADS.mtp.h + 18, '1 extra head', { opacity: seg(p, 0.8, 0.9) });
}

export function drawFrame8(svg, p) {
  chips(svg, [{ slot: 'five', text: '?' }]);
  if (p < HANDOFF) drawsBody(svg, 1, leaving(p));
  mtpBody(svg, p, arriving(p));
}

// ---- frame 9: the next step checks the draft ----
const MOVE = Object.freeze({ up: [0, 0.25], rerun: [0.25, 0.55], meet: [0.65, 0.85], verdict: [0.85, 0.95] });

// What the right-hand side shows while the next pass runs: its own pick slides to the draft and a verdict appears.
function checkBody(svg, p) {
  const meet = ease(seg(p, ...MOVE.meet));
  const pickY = lerp(OUT_Y.pick, OUT_Y.meet, meet);
  const pickIn = seg(p, 0.55, 0.65);
  const swap = seg(p, 0.85, 0.95);
  const pick = slotChip(svg, 'six', 'the', { at: { x: OUT_X, y: pickY }, opacity: pickIn });
  selectChip(svg, { x: pick.x, y: pick.y, text: 'the', opacity: pickIn });
  slotChip(svg, 'six', 'the', { state: 'draft', at: { x: OUT_X, y: OUT_Y.draft }, opacity: 1 - swap });
  slotChip(svg, 'six', 'the', { at: { x: OUT_X, y: OUT_Y.draft }, opacity: swap });
  note(svg, LABEL_X, OUT_Y.meet + 12 + (pickY - OUT_Y.meet), 'pick', { opacity: pickIn });
  note(svg, LABEL_X, OUT_Y.draft + 12, swap < 0.5 ? 'draft' : 'kept');
  const shown = seg(p, ...MOVE.verdict);
  if (shown > 0) G.verdict(layer(svg, shown), { ...VERDICT, ok: true });
}

export function drawFrame9(svg, p) {
  const up = ease(seg(p, ...MOVE.up));
  const landed = seg(p, 0.9, 1);
  const first = 1 - seg(p, 0.1, 0.25); // the first pass's MTP head fades; the pass that checks the draft stays
  chips(svg, [
    { slot: 'five', text: p >= MOVE.up[1] ? 'on' : '?', mark: 1 - seg(p, 0.05, 0.25) },
    { slot: 'six', text: '?', opacity: 1 - landed, mark: seg(p, 0.1, 0.25) },
    { slot: 'six', text: 'the', opacity: landed },
  ]);
  note(svg, ROW_X_LABEL, NOTE_ROW, 'in some 2026 models');
  lastBlock(svg);
  note(svg, STACK.x, STACK.y + STACK.h + 14, 'next step', { opacity: seg(p, 0.2, 0.3) });
  head(svg, HEADS.unembedding, 1);
  const rerun = p >= MOVE.rerun[0];
  flow(svg, 'toUnembedding', rerun ? seg(p, 0.25, 0.4) : 1);
  flow(svg, 'toPick', rerun ? seg(p, 0.4, 0.55) : 1);
  if (first > 0) {
    const old = layer(svg, first);
    head(old, HEADS.mtp, 1);
    flow(old, 'toMtp', 1);
    flow(old, 'toDraft', 1);
    note(old, HEADS.mtp.x, HEADS.mtp.y + HEADS.mtp.h + 18, '1 extra head');
  }
  if (p < MOVE.up[1]) slotChip(svg, 'five', 'on', { at: { x: lerp(OUT_X, SLOT_X.five, up), y: lerp(OUT_Y.pick, CHIP_Y, up) } });
  checkBody(svg, p);
  note(svg, HEADS.mtp.x, 232, '2 tokens from 1 step', { opacity: landed });
}
