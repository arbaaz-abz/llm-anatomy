// Selectable items on an SVG figure (ruling P3-R17): each item is a focusable <g role="button" tabindex aria-label aria-pressed>.
// Click, Enter or Space selects; arrow keys (and Home / End) move the selection and the focus together, so one outline
// (G.selectionMark, drawn by the caller) marks both. One roving tab stop: Tab lands on the selected item.
// The item groups persist across repaints, so focus survives; the caller redraws each group's children on every paint.
import * as G from '../glyphs.js';

const NEXT = new Set(['ArrowRight', 'ArrowDown']);
const PREV = new Set(['ArrowLeft', 'ArrowUp']);
const SELECT = new Set(['Enter', ' ']);

// The index a navigation key moves to from `index` among `count` items (the ends hold); null for any other key.
export function keyStep(key, index, count) {
  if (NEXT.has(key)) return Math.min(index + 1, count - 1);
  if (PREV.has(key)) return Math.max(index - 1, 0);
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  return null;
}

// items: [{ value, label }]; value: the selected item's value; onSelect(value) on a click or key (the caller then sync()s).
// → { node(value) → <g>, sync(value), destroy() }
export function mountStageSelect(parent, { items, value, onSelect }) {
  if (!Array.isArray(items) || items.length === 0) throw new RangeError('mountStageSelect: items must be a non-empty array');
  const nodes = items.map(({ value: v, label }) => {
    const g = G.svgEl('g', { class: 'stage-item', role: 'button', 'aria-label': label }, parent);
    g.dataset.value = String(v);
    return g;
  });
  const indexOf = (target) => nodes.indexOf(target.closest?.('.stage-item'));
  const onClick = (event) => {
    const i = indexOf(event.target);
    if (i >= 0) onSelect(items[i].value);
  };
  const onKey = (event) => {
    const i = indexOf(event.target);
    if (i < 0) return;
    if (SELECT.has(event.key)) {
      event.preventDefault();
      onSelect(items[i].value);
      return;
    }
    const next = keyStep(event.key, i, items.length);
    if (next == null) return;
    event.preventDefault();
    nodes[next].focus();
    onSelect(items[next].value);
  };
  const sync = (selected) => nodes.forEach((g, i) => {
    const on = items[i].value === selected;
    g.setAttribute('aria-pressed', String(on));
    g.setAttribute('tabindex', on ? '0' : '-1');
  });
  parent.addEventListener('click', onClick);
  parent.addEventListener('keydown', onKey);
  sync(value);
  return {
    node: (v) => nodes[items.findIndex((it) => it.value === v)],
    sync,
    destroy() {
      parent.removeEventListener('click', onClick);
      parent.removeEventListener('keydown', onKey);
      nodes.forEach((g) => g.remove());
    },
  };
}
