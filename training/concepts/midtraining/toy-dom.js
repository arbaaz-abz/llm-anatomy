// DOM helpers of the midtraining toy: the schedule plot, the stage bar, the readout tables and the try-this list.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { plotSpec, PLOT } from './stage.js';
import { drawContextBar, lastStageBox } from './context-bar.js';
import { runData } from './facts.js';
import { curves } from './toy-view.js';

const BAR_HEIGHT = 214;
const BAR_ORIGIN = Object.freeze({ x: 10, y: 22 });

// The schedule plot: the chosen curve solid, the other muted, the stop as the one followed marker.
export function plotView(state, scheduleLabels) {
  const c = curves(state);
  const bands = c.decayStart === null ? [] : [{ from: 0, to: c.decayStart, label: 'stable' }, { from: c.decayStart, to: 1, label: 'decay' }];
  const series = [
    { points: c.other, label: scheduleLabels.other, style: 'muted', labelAt: 'mid' },
    { points: c.chosen, label: scheduleLabels.chosen },
  ];
  const svg = G.svgEl('svg', { width: 580, height: PLOT.h + 8, viewBox: `0 0 580 ${PLOT.h + 8}`, role: 'img', 'aria-label': 'Learning rate over the run for the chosen schedule, the other schedule, and your stop point' });
  G.hatchFill(svg);
  G.curvePlot(svg, { x: 4, y: 4, ...plotSpec({ series, markers: [c.marker], bands, label: 'learning rate over the run' }) });
  return svg;
}

// GLM-5's (or another run's) context stages as one share bar with its zoomed tail; the longest stage is outlined.
export function barView(state, data) {
  const run = runData(data).runs[state.run];
  const svg = G.svgEl('svg', { width: 580, height: BAR_HEIGHT, viewBox: `0 0 580 ${BAR_HEIGHT}`, role: 'img', 'aria-label': `${run.name} context stages` });
  G.hatchFill(svg);
  const text = G.svgEl('text', { x: BAR_ORIGIN.x, y: 10, class: 'g-label', 'dominant-baseline': 'central' }, svg);
  text.textContent = `${run.name}: tokens per context length`;
  drawContextBar(svg, run.stages, { ...BAR_ORIGIN, label: `${run.name} context stages` });
  const box = lastStageBox(run.stages, BAR_ORIGIN);
  if (box) G.selectionMark(svg, box);
  return svg;
}

export function lrTable(view) {
  const rows = [
    { label: view.chosen.name, sub: 'chosen', cells: [{ value: view.chosen.lr, name: 'lr-chosen' }] },
    { label: view.other.name, sub: 'the other schedule', cells: [{ value: view.other.lr, name: 'lr-other' }] },
    ...(view.nemotron === null ? [] : [{ label: 'Nemotron 3 Super', sub: 'absolute rate at the stop', cells: [{ value: view.nemotron, name: 'lr-nemotron' }] }]),
  ];
  return readoutTable({ head: ['Schedule', 'Learning rate at the stop, as a fraction of peak'], name: 'lr', rows });
}

export function stageTable(view) {
  const rows = view.rows.map((r, i) => ({
    label: r.name,
    cells: [{ value: r.tokens, name: `tokens-${i}` }, { value: r.share, name: `share-${i}` }, { value: r.attention, name: `attn-${i}` }],
  }));
  return readoutTable({ head: ['Context', 'Tokens', 'Share of the run', 'Attention work per new token, vs the first stage'], caption: 'Attention only: the rest of the forward pass costs the same per token.', name: 'stages', rows });
}

// "**x**" in the storyboard's wording becomes <strong>.
export function emphasis(text) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) => (part.startsWith('**') ? el('strong', { textContent: part.slice(2, -2) }) : part));
}

export function tryThisList(items) {
  const lis = items.map(([prompt, insight, rest]) => el('li', {}, [...emphasis(prompt), ' → ', el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, lis)];
}
