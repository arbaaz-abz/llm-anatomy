// The selectable stage strip (P3-R17): six focusable <g role="button" tabindex="0" aria-label="stage n: name"> blocks.
// Enter / Space selects, ArrowLeft / ArrowRight move to the neighbor and select it, Home / End jump; focus stays on the
// selected block after a repaint. The selection is drawn with G.selectionMark.
import * as G from '@shared/glyphs.js';
import { STAGES } from './numbers.js';

const GAP = 6;
const TOP = 8;
const BLOCK_H = 40;
const NOTE_Y = TOP + BLOCK_H + 16;
export const STRIP = Object.freeze({ w: 580, h: NOTE_Y + 8 });

const blockX = (i) => STAGES.slice(0, i).reduce((sum, s) => sum + s.w + GAP, 0) + 4;

export function nextStage(current, key) {
  const last = STAGES.length;
  const moves = { ArrowRight: Math.min(current + 1, last), ArrowDown: Math.min(current + 1, last), ArrowLeft: Math.max(current - 1, 1), ArrowUp: Math.max(current - 1, 1), Home: 1, End: last };
  return key in moves ? moves[key] : null;
}

// Draws the strip into `svg` (cleared first). `strip` is toyView(...).strip; `selected` is 1–6; onSelect(n) picks a stage.
export function paintStrip(svg, strip, selected, onSelect) {
  const hadFocus = svg.contains(document.activeElement);
  svg.replaceChildren();
  const buttons = strip.map((s, i) => {
    const g = G.svgEl('g', {
      role: 'button', tabindex: 0, 'aria-label': `stage ${s.n}: ${s.name}${s.described ? '' : ' (not described)'}`, 'aria-pressed': String(s.n === selected), 'data-stage': s.n,
    }, svg);
    G.block(g, { x: blockX(i), y: TOP, w: STAGES[i].w, h: BLOCK_H, label: s.label, state: s.described ? 'idle' : 'dim' });
    if (!s.described) {
      const t = G.svgEl('text', { x: blockX(i) + STAGES[i].w / 2, y: NOTE_Y, class: 'g-label', 'text-anchor': 'middle' }, g);
      t.textContent = 'not described';
    }
    if (s.n === selected) G.selectionMark(g, { x: blockX(i), y: TOP, w: STAGES[i].w, h: BLOCK_H });
    g.addEventListener('click', () => onSelect(s.n));
    g.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(s.n); return; }
      const next = nextStage(s.n, event.key);
      if (next !== null) { event.preventDefault(); onSelect(next); }
    });
    return g;
  });
  if (hadFocus) buttons[selected - 1].focus();
  return buttons;
}
