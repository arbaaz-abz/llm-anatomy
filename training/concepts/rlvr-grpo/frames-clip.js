// Frames 7–9: the PPO shelf, then one token's clip (storyboard §5). Pure functions of p; frame 8 ends where frame 9 starts.
import * as G from '@shared/glyphs.js';
import { fixed, signed2 } from './format.js';
import { clippedSurrogate } from '@math/grpo.js';
import { PI_OLD, EPS_LOW, EPS_HIGH_PPO, ZOOM_RATIO, ZOOM_EPS_HIGH } from './numbers.js';
import { GROUP, CELL, advantageFill, seg, lerp, ease, label, select } from './stage.js';

const A_ROW1 = GROUP.rows[0].advantage; // +1.73
const CHIP_TEXT = '56';

// ---- frame 7: PPO keeps four models, GRPO + RLVR keeps one (and a checker) ----
const SHELF = Object.freeze({
  left: { x: 24, title: 'GRPO + RLVR', policy: { x: 24, y: 88 }, checker: { x: 24, y: 148 } },
  right: { x: 230, title: 'PPO', policy: { x: 230, y: 88 }, reference: { x: 390, y: 88 }, reward: { x: 230, y: 148 }, critic: { x: 390, y: 148 } },
  block: { w: 144, h: 40 },
});

export function drawFrame7(svg, p) {
  const swap = ease(seg(p, 0.15, 0.55));
  const state = swap > 0.5 ? 'dim' : 'active';
  const { left, right, block } = SHELF;
  label(svg, left.x, 60, left.title);
  label(svg, right.x, 60, right.title);
  G.block(svg, { ...left.policy, ...block, label: 'policy', state: 'active' });
  G.block(svg, { ...left.checker, ...block, label: 'checker', state: 'active' });
  G.block(svg, { ...right.policy, ...block, label: 'policy', state: 'active' });
  G.block(svg, { ...right.reference, ...block, label: 'reference', state: 'active' });
  G.block(svg, { ...right.reward, ...block, label: 'reward model', state });
  G.block(svg, { ...right.critic, ...block, label: 'critic', state });
  label(svg, right.reward.x, 204, 'replaced by the checker', { opacity: swap });
  label(svg, right.critic.x, 204, 'replaced by the group mean', { opacity: swap });
  label(svg, 24, 290, 'PPO: 4 models in memory');
  label(svg, 24, 312, 'GRPO + RLVR: 1 (+ reference only if a KL term is kept)', { opacity: ease(seg(p, 0.4, 0.8)) });
}

// ---- frames 8–9: one token's ratio against the clip band ----
const ZOOM = Object.freeze({
  chip: { x: 40, y: 34 },
  cells: { sampled: { x: 40, y: 98 }, now: { x: 176, y: 98 } },
  readouts: { x: 312, y: 78 },
  line: { x: 40, y: 262, w: 500, lo: 0.6, hi: 1.6, band: [1 - EPS_LOW, 1 + EPS_HIGH_PPO] },
});

function zoomScene(svg, { ratio, epsHigh, bandHigh, objectiveText, note }) {
  const clipped = clippedSurrogate(ratio, A_ROW1, { epsLow: EPS_LOW, epsHigh }).clipped;
  const { chip, cells, readouts, line } = ZOOM;
  G.token(svg, { ...chip, text: CHIP_TEXT, hatched: clipped, fill: advantageFill(A_ROW1) });
  select(svg, chip.x, chip.y, G.tokenWidth(CHIP_TEXT), 24);
  label(svg, chip.x + G.tokenWidth(CHIP_TEXT) + 12, chip.y + 12, 'row 1, last token');
  const probability = (v) => fixed(v, 2);
  label(svg, cells.sampled.x, cells.sampled.y - 14, 'when sampled');
  G.cell(svg, { ...cells.sampled, size: CELL, v: PI_OLD, maxAbs: 1, format: probability });
  label(svg, cells.now.x, cells.now.y - 14, 'now');
  G.cell(svg, { ...cells.now, size: CELL, v: PI_OLD * ratio, maxAbs: 1, format: probability });
  [`r = ${probability(PI_OLD * ratio)} / ${probability(PI_OLD)} = ${fixed(ratio, 2)}`, `A = ${signed2(A_ROW1)}`, `objective ${objectiveText}`].forEach((text, i) => label(svg, readouts.x, readouts.y + i * 22, text));
  G.clipLine(svg, { ...line, band: [line.band[0], bandHigh], marker: ratio, label: 'ratio r = now / when sampled' });
  if (note) note();
}

const objectiveFor = (ratio, epsHigh) => {
  const { objective, clipped } = clippedSurrogate(ratio, A_ROW1, { epsLow: EPS_LOW, epsHigh });
  return `${fixed(objective, 2)}${clipped ? ' (clipped)' : ''}`;
};

export function drawFrame8(svg, p) {
  const ratio = lerp(1, ZOOM_RATIO, ease(seg(p, 0.1, 0.9)));
  zoomScene(svg, { ratio, epsHigh: EPS_HIGH_PPO, bandHigh: 1 + EPS_HIGH_PPO, objectiveText: objectiveFor(ratio, EPS_HIGH_PPO) });
}

export function drawFrame9(svg, p) {
  const eps = lerp(EPS_HIGH_PPO, ZOOM_EPS_HIGH, ease(seg(p, 0.1, 0.9)));
  label(svg, ZOOM.line.x, 298, `clip-higher: ε_high ${fixed(EPS_HIGH_PPO, 2)} → ${fixed(ZOOM_EPS_HIGH, 2)}`, { opacity: ease(seg(p, 0, 0.3)) });
  label(svg, ZOOM.line.x, 322, 'prevents entropy collapse: unlikely good tokens can keep growing', { opacity: ease(seg(p, 0, 0.3)) });
  zoomScene(svg, { ratio: ZOOM_RATIO, epsHigh: eps, bandHigh: 1 + eps, objectiveText: objectiveFor(ZOOM_RATIO, eps) });
}
