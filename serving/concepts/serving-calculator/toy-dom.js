// serving-calculator toy DOM helpers: the grouped readout tables and the figure (memory bar + step bar), repainted from toyView.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { formatBytes } from '@math/core.js';

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

const BOUND_CLASS = Object.freeze({ 'memory-bound': 'sem-text--memory', 'compute-bound': 'sem-text--compute' });

// The verdict words keep their text (they carry the meaning); a class only colours them, memory teal and compute amber.
function colourBound(table) {
  const out = table.querySelector('[data-readout="bound"]');
  if (!out) return;
  const parts = out.textContent.split(/(memory-bound|compute-bound)/);
  out.replaceChildren(...parts.filter(Boolean).map((t) => (BOUND_CLASS[t] ? el('span', { className: BOUND_CLASS[t], textContent: t }) : document.createTextNode(t))));
}

const cell = (r, name, sub) => ({ value: r[name], name, ...(sub ? { sub } : {}) });

function fitTable(r, v) {
  return readoutTable({
    head: ['Fit', ''], name: 'fit',
    rows: [
      { label: 'Weights, whole model', cells: [cell(r, 'weights-total')] },
      { label: 'GPUs for the weights alone', sub: 'weights ÷ one GPU\'s memory, rounded up', cells: [cell(r, 'gpus-min')] },
      { label: 'Memory per GPU', cells: [cell(r, 'hbm')] },
      { label: 'Weights per GPU', sub: 'weights ÷ GPUs per replica', cells: [cell(r, 'weights-per-gpu')] },
      { label: 'Free per GPU', sub: v.sub.fits ? 'upper bound: no activation reserve' : '', cells: [cell(r, 'free-per-gpu')] },
    ],
  });
}

function kvTable(r) {
  return readoutTable({
    head: ['Users', ''], name: 'kv',
    rows: [
      { label: 'KV per user', sub: 'KV bytes per token × tokens', cells: [cell(r, 'kv-per-user')] },
      { label: 'Users that fit per GPU', cells: [cell(r, 'users-fit')] },
    ],
  });
}

function speedTable(r, v) {
  return readoutTable({
    head: ['Speed floor', ''], name: 'speed',
    rows: [
      { label: 'Users per GPU', sub: v.sub.clamped, cells: [cell(r, 'users')] },
      { label: 'Step time', cells: [cell(r, 'step-time')] },
      { label: 'Bound by', cells: [cell(r, 'bound')] },
      { label: 'Tokens/s per user', cells: [cell(r, 'tok-user')] },
      { label: 'Tokens/s per GPU', sub: 'output tokens only', cells: [cell(r, 'tok-gpu')] },
      { label: 'Most users at the target', cells: [cell(r, 'target-users')] },
      { label: 'What limits them', cells: [cell(r, 'target-limit')] },
    ],
  });
}

function mtpTable(r) {
  return readoutTable({
    head: ['MTP', ''], name: 'mtp',
    rows: [
      { label: 'Speed per user, relative', cells: [cell(r, 'mtp-speedup')] },
      { label: 'Tokens/s per user with MTP', cells: [cell(r, 'mtp-tok-user')] },
    ],
  });
}

function costTable(r, v) {
  return readoutTable({
    head: ['Cost and input side', ''], name: 'cost',
    rows: [
      { label: 'Floor cost per M output tokens', sub: 'this page\'s definition', cells: [cell(r, 'cost-floor')] },
      { label: 'Measured, GB300 per M tokens', sub: v.sub.gb300, cells: [cell(r, 'cost-gb300')] },
      { label: 'Measured, GB200 per M tokens', sub: v.sub.gb200, cells: [cell(r, 'cost-gb200')] },
      { label: 'Input tokens/s per GPU', sub: 'prefill ceiling, with cache hits', cells: [cell(r, 'prefill-tps')] },
      { label: 'Input cost per M tokens', cells: [cell(r, 'input-cost')] },
      { label: 'Output cost ÷ input cost', cells: [cell(r, 'cost-ratio')] },
    ],
  });
}

// The grouped tables; the MTP table is built only when MTP is on.
export function readoutTables(v) {
  const r = v.readouts;
  const tables = [fitTable(r, v), kvTable(r), speedTable(r, v), ...(v.showMtp ? [mtpTable(r)] : []), costTable(r, v)];
  tables.forEach(colourBound);
  return tables;
}

const FIGURE_W = 520;

// The followed GPU's memory (weights · KV · free, bytes printed) and the step as a step bar. Nothing to draw when nothing fits.
export function paintFigure(svg, figure) {
  svg.replaceChildren();
  if (!figure.memory) {
    const t = G.svgEl('text', { x: 0, y: 14, class: 'g-label' }, svg);
    t.textContent = 'does not fit: no memory bar to draw';
    G.fitViewBox(svg, 4);
    return;
  }
  const { total, weights, kv, free } = figure.memory;
  const parts = [{ name: 'weights', value: weights, hue: 1 }, { name: 'KV', value: kv, hue: 2 }, { name: 'free', value: free, hue: 3 }];
  G.shareBar(svg, { x: 0, y: 18, w: FIGURE_W, h: 14, parts, format: (share) => formatBytes(share * total), tail: 'none', label: 'followed GPU memory' });
  if (figure.step) {
    const scaleS = Math.max(figure.step.reading.reduce((a, b) => a + b.s, 0), figure.step.mathS);
    G.stepBar(svg, { x: 0, y: 110, w: 440, scaleS, reading: figure.step.reading, mathS: figure.step.mathS, label: 'one decode step' });
  }
  G.fitViewBox(svg, 6);
}
