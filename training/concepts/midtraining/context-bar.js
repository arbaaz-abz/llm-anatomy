// The context-stage share bar of frames 6-7 and the toy: one shareBar with its zoomed tail (P3-R7), and the box of the
// last (longest-context) stage's segment, which the page outlines itself (shareBar draws no selection mark).
import * as G from '@shared/glyphs.js';
import { mainShare, tailShare } from './format.js';

export const BAR_W = 560;
export const BAR_H = 14;
export const TAIL_OFFSET = 54; // main bar top → zoomed bar top: shareBar's TAIL_GAP (40) + the bar height
export const TAIL_LABEL = 'last 5%';
export const MIN_SEGMENT = 18;

// Stage parts for shareBar: a published count is a hue 1-5 segment, an unpublished one is { value: null } (P3-R6).
export const barParts = (stages) => stages.map((s, i) => (s.value === null ? { name: s.name, value: null } : { name: s.name, value: s.value, hue: i + 1 }));

const barOptions = (stages) => ({ w: BAR_W, parts: barParts(stages), minSegment: MIN_SEGMENT, tailBasis: 'tail' });

export function drawContextBar(parent, stages, { x, y, label = 'context stages' }) {
  return G.shareBar(parent, { ...barOptions(stages), x, y, label, format: mainShare, tailFormat: tailShare, tail: 'zoom', tailLabel: TAIL_LABEL });
}

// The box (stage coordinates) around the last stage's segment, on the main bar or on the zoomed tail; null when unpublished.
export function lastStageBox(stages, { x, y }) {
  const last = stages[stages.length - 1];
  if (last.value === null) return null;
  const layout = G.shareBarLayout(barOptions(stages).parts, { w: BAR_W, minSegment: MIN_SEGMENT, tailBasis: 'tail' });
  const inTail = layout.tail.find((s) => s.name === last.name);
  const seg = inTail ?? layout.main.find((s) => s.name === last.name);
  return { x: x + seg.x, y: y + (inTail ? TAIL_OFFSET : 0), w: seg.width, h: BAR_H };
}
