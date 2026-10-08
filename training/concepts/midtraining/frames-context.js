// Frames 6-9: context extension (the stage bar and its zoomed tail, the cost per token, RoPE's dials) and the 2026 table.
import * as G from '@shared/glyphs.js';
import { attentionCostRatio } from '@math/schedule.js';
import { formatRatio } from '@math/core.js';
import { DIAL_STAND_IN } from './numbers.js';
import { countText, glmBudget, labelK } from './facts.js';
import { trimNumber } from './format.js';
import { drawContextBar, lastStageBox } from './context-bar.js';
import { drawFrame5 } from './frames-curve.js';
import { seg, lerp, ease, label, select, fade, outgoing, wrap } from './stage.js';

const BAR = Object.freeze({ x: 10, y: 26 });
const LINE_Y = Object.freeze([204, 220, 236]);
const CHIP_Y = 262;
const CHIP_GAP = 10;
const CHIP_TOP_7 = 206; // frame 7 lifts the chips to make room for the readouts under them

// The three document chips: they get longer with the stage; the 200K chip is the followed item.
function chipRow(svg, run, y, opacity = 1) {
  let x = BAR.x;
  const chips = run.stages.slice(1).map((s) => {
    const text = `${s.name} doc`;
    const g = fade(G.token(svg, { x, y, text, state: s.name === '200K' ? 'active' : 'idle' }), opacity);
    const box = { x, y, w: G.tokenWidth(text), h: 24 };
    if (s.name === '200K') select(svg, box, opacity);
    x += box.w + CHIP_GAP;
    return { name: s.name, box, g };
  });
  return chips;
}

function stageBar(svg, run, opacity) {
  const g = drawContextBar(svg, run.stages, { ...BAR, label: 'GLM-5 context stages' });
  fade(g, opacity);
  const box = lastStageBox(run.stages, BAR);
  if (box) select(svg, box, opacity);
}

function numberLines(svg, run, p) {
  const { whole, tail } = glmBudget(run);
  const [base, ...later] = run.stages;
  const per = run.stages.map((s) => `${countText(s.value)} @ ${s.name}`).join(' · ');
  const zoom = later.map((s) => `${trimNumber(s.value / 1e12)} / ${trimNumber(tail.knownTotal / 1e12)} = ${(100 * (s.value / tail.knownTotal)).toFixed(1)}%`).join(' · ');
  const last = run.stages[run.stages.length - 1];
  const lastShare = (100 * whole.parts.find((q) => q.name === last.name).share).toFixed(2);
  label(svg, BAR.x, LINE_Y[0], per, { opacity: seg(p, 0.55, 0.75) });
  label(svg, BAR.x, LINE_Y[1], `zoomed: ${zoom}`, { opacity: seg(p, 0.65, 0.85) });
  label(svg, BAR.x, LINE_Y[2], `${last.name} is ${lastShare}% of the whole run (${base.name} is the base)`, { opacity: seg(p, 0.75, 0.95) });
}

// Frame 6 (key frame): GLM-5's whole run by context length, and the last 5% zoomed.
export function drawFrame6(svg, p, ctx) {
  const { run } = ctx;
  outgoing(svg, (g) => drawFrame5(g, 1, ctx), 1 - seg(p, 0, 0.2));
  label(svg, BAR.x, 12, `GLM-5: ${countText(run.reportedTotal)} (its published stages sum to ${countText(glmBudget(run).whole.knownTotal, 4)})`, { opacity: seg(p, 0.15, 0.4) });
  stageBar(svg, run, seg(p, 0.15, 0.6));
  numberLines(svg, run, p);
  chipRow(svg, run, CHIP_Y, seg(p, 0.6, 0.9));
}

// Frame 7: the cost per new token, relative to 4K, under each zoomed stage.
export function drawFrame7(svg, p, { run }) {
  const lift = ease(seg(p, 0.05, 0.4));
  label(svg, BAR.x, 12, `GLM-5: ${countText(run.reportedTotal)} (its published stages sum to ${countText(glmBudget(run).whole.knownTotal, 4)})`);
  stageBar(svg, run, 1);
  outgoing(svg, (g) => numberLines(g, run, 1), 1 - seg(p, 0, 0.3));
  const base = labelK(run.stages[0].name);
  const y = lerp(CHIP_Y, CHIP_TOP_7, lift);
  const chips = chipRow(svg, run, y);
  chips.forEach((c, i) => {
    const ratio = attentionCostRatio(labelK(c.name), base);
    label(svg, c.box.x + c.box.w / 2, y + 40, formatRatio(ratio), { anchor: 'middle', opacity: seg(p, 0.45 + i * 0.12, 0.6 + i * 0.12) });
  });
  const sums = chips.map((c) => `${c.name} / ${run.stages[0].name} = ${attentionCostRatio(labelK(c.name), base)}`).join(' · ');
  label(svg, BAR.x, CHIP_TOP_7 + 66, sums, { opacity: seg(p, 0.7, 0.9) });
  label(svg, BAR.x, CHIP_TOP_7 + 84, 'long documents are scarce', { opacity: seg(p, 0.75, 0.95) });
  wrap('attention only: the rest of the forward pass costs the same per token (long-context-attention shows how 2026 models cut this)', 80)
    .forEach((line, i) => label(svg, BAR.x, CHIP_TOP_7 + 102 + i * 16, line, { opacity: seg(p, 0.8, 1) }));
}

const DIAL = Object.freeze({ r: 46, y: 118, xs: [150, 430] });
const TURN = 2 * Math.PI;
const SEEN = Object.freeze([0, DIAL_STAND_IN.turnsAt4K * TURN]);
const turnText = (turns) => (turns === 0.75 ? '¾ turn' : `${trimNumber(turns)} turn`);

// Frame 8: the slowest RoPE pair at offset 4K (inside what training saw) and at 200K (far outside it).
export function drawFrame8(svg, p, ctx) {
  const { text } = ctx;
  outgoing(svg, (g) => drawFrame7(g, 1, ctx), 1 - seg(p, 0, 0.2));
  const sweep = ease(seg(p, 0.2, 0.7));
  const turns = [DIAL_STAND_IN.turnsAt4K, lerp(DIAL_STAND_IN.turnsAt4K, DIAL_STAND_IN.turnsAt200K, sweep)];
  ['offset 4K', 'offset 200K'].forEach((name, i) => {
    G.dial(svg, { x: DIAL.xs[i], y: DIAL.y, r: DIAL.r, angle: turns[i] * TURN, seen: SEEN, label: name });
    label(svg, DIAL.xs[i], DIAL.y + DIAL.r + 18, i === 0 ? turnText(DIAL_STAND_IN.turnsAt4K) : turnText(Number(turns[1].toFixed(3))), { anchor: 'middle' });
  });
  select(svg, { x: DIAL.xs[1] - DIAL.r, y: DIAL.y - DIAL.r, w: 2 * DIAL.r, h: 2 * DIAL.r }, seg(p, 0.1, 0.3));
  label(svg, 290, 200, 'RoPE: rescale (see rope)', { anchor: 'middle', opacity: seg(p, 0.7, 0.9) });
  label(svg, 290, 220, text.kimiLayers, { anchor: 'middle', opacity: seg(p, 0.75, 0.95) });
  label(svg, 290, 240, 'pair speed chosen so 200K is ¾ turn', { anchor: 'middle', opacity: seg(p, 0.8, 1) });
  label(svg, BAR.x, 268, `slowest pair, stand-in speed: ¾ turn × 4K / 200K = ${trimNumber(DIAL_STAND_IN.turnsAt4K)} turn at 4K · ¾ turn at 200K`, { opacity: seg(p, 0.85, 1) });
  label(svg, BAR.x, 288, 'trained to 4K · extended to 200K (GLM-5) · Kimi K3 256K → 1M in its cooldown', { opacity: seg(p, 0.9, 1) });
}

// Frame 9: the 2026 table of staged context.
export function drawFrame9(svg, p, ctx) {
  const { text, run } = ctx;
  outgoing(svg, (g) => drawFrame8(g, 1, ctx), 1 - seg(p, 0, 0.2));
  const rows = [
    ['GLM-5', run.runs['glm-5'].stages.map((s) => s.name).join(' → ')],
    ['MiniMax-M2', text.minimaxRow],
    ['DeepSeek-V4', text.deepseekRow],
    ['Kimi K3', text.kimiRow],
  ];
  rows.forEach(([name, stages], i) => {
    const o = seg(p, 0.25 + i * 0.15, 0.4 + i * 0.15);
    label(svg, BAR.x, 60 + i * 44, name, { cls: 'g-label', opacity: o });
    label(svg, 140, 60 + i * 44, stages, { cls: '', opacity: o });
  });
  select(svg, { x: BAR.x, y: 60 - 14, w: 560, h: 28 }, seg(p, 0.25, 0.4));
  wrap(text.kimiNote, 84).forEach((line, i) => label(svg, BAR.x, 262 + i * 18, line, { opacity: seg(p, 0.85, 1) }));
}
