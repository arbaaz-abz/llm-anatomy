// prefill-decode toy DOM helpers: readout tables, the verdict colour, and the three figures (step bar, memory bar,
// per-user vs per-GPU curve), each repainted from toyView and scrolling inside its own box at 400 px.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { formatBytes } from '@math/core.js';
import { TOY_STEP_W, TOY_MEMORY_W } from './toy-view.js';

const CURVE = Object.freeze({ w: 420, h: 220 });
const BOUND_CLASS = Object.freeze({ 'memory-bound': 'sem-text--memory', 'compute-bound': 'sem-text--compute' });

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

// The bound keeps its words (they carry the meaning); a class only colours them, memory teal and compute amber.
function colourBound(table) {
  const out = table.querySelector('[data-readout="bound"]');
  if (!out || !BOUND_CLASS[out.textContent]) return;
  out.replaceChildren(el('span', { className: BOUND_CLASS[out.textContent], textContent: out.textContent }));
}

export function readoutTables(specs) {
  return specs.map((spec) => {
    const table = readoutTable(spec);
    colourBound(table);
    return table;
  });
}

const svgFigure = (label) => G.svgEl('svg', { role: 'img', 'aria-label': label });

export function paintStepBar(svg, spec) {
  svg.replaceChildren();
  if (!spec) return;
  G.stepBar(svg, { x: 0, y: 0, w: TOY_STEP_W, scaleS: spec.scaleS, reading: spec.reading, mathS: spec.mathS, label: 'this step' });
  G.fitViewBox(svg, 4);
}

export function paintMemoryBar(svg, spec) {
  svg.replaceChildren();
  if (!spec) return;
  G.shareBar(svg, { x: 0, y: 0, w: TOY_MEMORY_W, tail: 'none', minSegment: 0, label: spec.label, parts: spec.parts, format: (share) => formatBytes(share * spec.drawnTotal) });
  G.fitViewBox(svg, 4);
}

export function paintCurve(svg, spec) {
  svg.replaceChildren();
  if (!spec) return;
  G.curvePlot(svg, { x: 0, y: 0, w: CURVE.w, h: CURVE.h, ...spec, label: 'tokens per second per user against per GPU, one dot per users stop' });
  G.fitViewBox(svg, 6);
}

// The figures block: a caption line and a scrolling box per figure; `hidden` when the state has no such figure.
export function figureBlocks() {
  const make = (label) => {
    const caption = el('p', { className: 'toy-note' });
    const svg = svgFigure(label);
    return { root: el('div', {}, [caption, el('div', { className: 'scroll-x' }, [svg])]), caption, svg };
  };
  return { step: make('this step: reading against arithmetic'), memory: make('GPU memory'), curve: make('tokens per second per user against per GPU') };
}
