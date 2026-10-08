// rlhf-dpo toy DOM helpers: the implicit-reward cells (two NUMBER_CELLs on the value scale) and the try-this list.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { appendRich } from '@shared/lesson-page.js';
import { fmt3 } from './format.js';

const CELL = G.NUMBER_CELL;
const GAP = 8;
const PAD = 4;
const SIZE = Object.freeze({ w: 2 * CELL + GAP + 2 * PAD, h: CELL + 22 + PAD });
const VALUE_MAX_ABS = 1;

// Two cells, A then B, each printed to three decimals; paint(values) redraws them.
export function mountRewardCells(host, { labelledBy }) {
  const svg = G.svgEl('svg', { width: SIZE.w, height: SIZE.h, viewBox: `0 0 ${SIZE.w} ${SIZE.h}`, role: 'group', 'aria-labelledby': labelledBy });
  const group = G.svgEl('g', {}, svg);
  group.dataset.readout = 'rewards';
  host.append(svg);
  return {
    paint(values) {
      group.replaceChildren();
      ['A', 'B'].forEach((letter, i) => {
        const x = PAD + i * (CELL + GAP);
        const t = G.svgEl('text', { x: x + CELL / 2, y: 12, class: 'g-label', 'text-anchor': 'middle' }, group);
        t.textContent = letter;
        G.vector(group, { x, y: 20, values: [values[i]], cell: CELL, orient: 'row', maxAbs: VALUE_MAX_ABS, format: fmt3 });
      });
    },
    destroy: () => svg.remove(),
  };
}

// "**5**" is bold; everything else goes through the lesson's rich text.
function emphasized(node, text, ctx) {
  text.split('**').forEach((part, i) => {
    if (i % 2 === 1) node.append(el('strong', { textContent: part }));
    else appendRich(node, part, ctx);
  });
  return node;
}

// The list at the end of the toy: [prompt, insight, rest] per item, each leading to a named insight.
export function tryThisList(items, ctx) {
  const lis = items.map(([prompt, insight, rest]) => {
    const li = emphasized(el('li'), prompt, ctx);
    li.append(' → ', el('strong', { textContent: `Insight: ${insight}` }));
    return emphasized(li, rest, ctx);
  });
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, lis)];
}
