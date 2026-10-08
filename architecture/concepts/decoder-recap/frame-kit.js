// decoder-recap frame composition: the two stacks, the selection outlines and the evidence panel that every frame
// assembles the same way (so each frame's start equals the previous frame's end). Pure functions of (frame, progress).
import { LEFT_X, RIGHT_X, layer, seg, arriving, leaving } from './stage.js';
import { drawStack, drawHybrid, partsAfter, GPT3_PARTS, SELECTED, selectParts, selectHybrid, pulseParts } from './stacks.js';
import { GPT3_FACTS } from './numbers.js';

// When (in a frame's progress) the right stack's swapped part changes: [from, to].
const SWAP = Object.freeze({ 2: [0.8, 1], 3: [0.2, 0.6], 4: [0.45, 0.85], 5: [0.5, 0.8], 6: [0.5, 0.85], 7: [0.6, 0.8], 8: [0.4, 0.7] });
const LAYERS = GPT3_FACTS.layers;

// Frame k's stacks (k = 1…10). Frame 2 fades the left stack out and frame 3 brings it back (a stage-clipping-safe stand-in for sliding off).
export function drawStacks(svg, k, p) {
  const left = k === 2 ? 1 - seg(p, 0, 0.3) : (k === 3 ? seg(p, 0, 0.3) : 1);
  const from = partsAfter(Math.max(k - 1, 1));
  const to = partsAfter(Math.min(k, 8));
  const t = SWAP[k] ? seg(p, SWAP[k][0], SWAP[k][1]) : 1;
  const detail = k === 9 ? 1 - seg(p, 0.05, 0.35) : (k === 10 ? seg(p, 0, 0.3) : 1);
  drawStack(svg, { x: LEFT_X, title: 'GPT-3 (2020)', from: GPT3_PARTS, layers: LAYERS, opacity: k === 1 ? seg(p, 0, 0.5) : left });
  drawStack(svg, { x: RIGHT_X, title: '2026', from, to, t, layers: LAYERS, opacity: (k === 1 ? seg(p, 0.4, 0.9) : 1) * detail });
  if (k === 9) drawHybrid(svg, { opacity: seg(p, 0.05, 0.35), relabel: seg(p, 0.5, 0.9) });
  if (k === 10) drawHybrid(svg, { opacity: 1 - seg(p, 0, 0.3), relabel: 1 });
}

// Selection outlines: this frame's swapped parts fade in, the previous frame's fade out (a part in both stays).
export function drawSelection(svg, k, p) {
  if (k === 10) {
    selectHybrid(svg, leaving(p));
    pulseParts(svg, p);
    return;
  }
  const now = k === 9 ? [] : (SELECTED[k] ?? []);
  const before = k === 10 ? [] : (SELECTED[k - 1] ?? []);
  selectParts(svg, now.filter((n) => before.includes(n)), 1);
  selectParts(svg, now.filter((n) => !before.includes(n)), arriving(p));
  selectParts(svg, before.filter((n) => !now.includes(n)), leaving(p));
  if (k === 9) selectHybrid(svg, arriving(p));
}

// The evidence panel: the previous frame's panel (at its end state) fades out while this frame's panel fades in.
export function drawPanels(svg, k, p, panels) {
  const previous = panels[k - 2];
  if (previous && leaving(p) > 0) previous(layer(svg, leaving(p)), 1);
  panels[k - 1](layer(svg, arriving(p)), p);
}

// A whole frame: stacks, selection, panel.
export function drawFrame(svg, k, p, panels) {
  drawStacks(svg, k, p);
  drawPanels(svg, k, p, panels);
  drawSelection(svg, k, p);
}
