// The selectable stage strip (P3-R17): six focusable stage items from the shared stage-select helper. Enter / Space
// selects, arrows move to the neighbor and select it, Home / End jump; focus stays on the selected block after a
// repaint (the item groups persist; only their children are redrawn). The selection is drawn with G.selectionMark.
import * as G from '@shared/glyphs.js';
import { mountStageSelect } from '@shared/ui/stage-select.js';
import { STAGES } from './numbers.js';

const GAP = 6;
const TOP = 8;
const BLOCK_H = 40;
const NOTE_Y = TOP + BLOCK_H + 16;
export const STRIP = Object.freeze({ w: 580, h: NOTE_Y + 8 });

const blockX = (i) => STAGES.slice(0, i).reduce((sum, s) => sum + s.w + GAP, 0) + 4;
const labelOf = (s) => `stage ${s.n}: ${s.name}${s.described ? '' : ' (not described)'}`;

// Mounts the strip into `svg`. onSelect(n) picks a stage (1–6). → { paint(strip, selected), destroy }
// `strip` is toyView(...).strip; paint() redraws every block and the selection mark.
export function mountStrip(svg, strip, selected, onSelect) {
  const select = mountStageSelect(svg, { items: strip.map((s) => ({ value: s.n, label: labelOf(s) })), value: selected, onSelect });
  return {
    paint(nextStrip, current) {
      nextStrip.forEach((s, i) => {
        const g = select.node(s.n);
        g.replaceChildren();
        g.setAttribute('aria-label', labelOf(s));
        G.block(g, { x: blockX(i), y: TOP, w: STAGES[i].w, h: BLOCK_H, label: s.label, state: s.described ? 'idle' : 'dim' });
        if (!s.described) {
          const t = G.svgEl('text', { x: blockX(i) + STAGES[i].w / 2, y: NOTE_Y, class: 'g-label', 'text-anchor': 'middle' }, g);
          t.textContent = 'not described';
        }
        if (s.n === current) G.selectionMark(g, { x: blockX(i), y: TOP, w: STAGES[i].w, h: BLOCK_H });
      });
      select.sync(current);
    },
    destroy: select.destroy,
  };
}
