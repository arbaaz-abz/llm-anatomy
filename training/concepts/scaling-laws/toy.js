// "Spend a compute budget" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import * as G from '@shared/glyphs.js';
import { computeOptimal } from '@math/scaling.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { BUDGETS } from './numbers.js';
import { INITIAL_STATE, sliderSizes, sizeText, powerText } from './format.js';
import { toyView, SERVED_OPTIONS, VALIDITY_NOTE } from './toy-view.js';
import { tryThis } from './try-this.js';

const PLOT_SIZE = Object.freeze({ w: 720, h: 300 });
const PLOT_PAD = 4; // the figure sits inside the svg so its outermost text keeps clear of the edge
const TABLE_HEAD = Object.freeze(['', 'This size', 'Compute-optimal', 'Cheapest for the same loss']);
const COLUMNS = Object.freeze(['this', 'opt', 'cheap']);
const ROWS = Object.freeze([
  { name: 'n', label: 'Model size N', sub: 'active parameters', key: 'N' },
  { name: 'd', label: 'Tokens D', sub: 'D = C ÷ 6N', key: 'D' },
  { name: 'ratio', label: 'Tokens per parameter', sub: 'D ÷ N', key: 'ratio' },
  { name: 'loss', label: 'Fitted loss', key: 'loss' },
  { name: 'life', label: 'Lifetime compute', sub: 'FLOPs, training + serving', key: 'life' },
]);
const EXTRAPOLATED = 'extrapolated';

// "**bold**" in the storyboard's wording becomes <strong>.
function emphasis(text) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) => (part.startsWith('**') ? el('strong', { textContent: part.slice(2, -2) }) : part));
}

function tryThisList() {
  const items = tryThis().map(([prompt, insight, rest]) => el('li', {}, [...emphasis(prompt), ' → ', el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function tableSpec(view) {
  return {
    head: TABLE_HEAD,
    name: 'scaling-table',
    rows: ROWS.map((row) => ({
      label: row.label,
      ...(row.sub ? { sub: row.sub } : {}),
      cells: COLUMNS.map((col) => ({ value: view.cols[col][row.key], name: `${row.name}-${col}`, ...(row.key === 'ratio' && view.cols[col].extrapolated ? { sub: EXTRAPOLATED } : {}) })),
    })),
  };
}

function plotSvg(plot) {
  const w = PLOT_SIZE.w + 2 * PLOT_PAD;
  const h = PLOT_SIZE.h + 2 * PLOT_PAD;
  const svg = G.svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img' });
  G.curvePlot(svg, { x: PLOT_PAD, y: PLOT_PAD, w: PLOT_SIZE.w, h: PLOT_SIZE.h, ...plot, label: 'Fitted loss against model size at this budget' });
  return svg;
}

function mountControls(host, onChange) {
  const parts = ['C', 'logN', 'Dinf'].map(() => el('div'));
  const reset = el('button', { type: 'button', id: 'reset', className: 'choice-option', textContent: 'Reset' });
  const validity = el('p', { className: 'toy-note', textContent: VALIDITY_NOTE });
  host.append(parts[0], parts[1], parts[2], validity, el('div', {}, [reset]));
  let slider = null;
  const mountSize = (C, N) => {
    slider?.destroy();
    slider = mountSlider(parts[1], { id: 'logN', label: 'Model size (active parameters)', values: sliderSizes(C), value: N, format: sizeText, onInput: (n) => onChange({ N: n }) });
  };
  const budget = mountChoice(parts[0], {
    id: 'C', label: 'Training budget', variant: 'chips', value: INITIAL_STATE.C,
    options: BUDGETS.map((c) => ({ value: c, label: `${powerText(c)} FLOPs` })),
    onChange: (C) => { const N = computeOptimal(C).N; mountSize(C, N); onChange({ C, N }); },
  });
  const served = mountChoice(parts[2], { id: 'Dinf', label: 'Tokens the model will serve over its life (what-if)', variant: 'chips', value: INITIAL_STATE.Dinf, options: SERVED_OPTIONS, onChange: (Dinf) => onChange({ Dinf }) });
  mountSize(INITIAL_STATE.C, INITIAL_STATE.N);
  const onReset = () => {
    budget.set(INITIAL_STATE.C);
    served.set(INITIAL_STATE.Dinf);
    mountSize(INITIAL_STATE.C, INITIAL_STATE.N);
    onChange({ ...INITIAL_STATE });
  };
  reset.addEventListener('click', onReset);
  return () => { reset.removeEventListener('click', onReset); slider.destroy(); budget.destroy(); served.destroy(); };
}

function mountReadouts(host) {
  const plot = el('div', { className: 'scroll-x' });
  const table = el('div', { className: 'scroll-x' });
  const saving = el('output', { className: 'readout' });
  saving.dataset.readout = 'saving';
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  host.append(plot, table, el('p', { className: 'toy-note' }, [saving]), el('h4', { textContent: 'Check my work' }), check, ...tryThisList());
  return { plot, table, saving, check };
}

function paint(refs, view) {
  refs.plot.replaceChildren(plotSvg(view.plot));
  refs.table.replaceChildren(readoutTable(tableSpec(view)));
  refs.saving.textContent = view.savingLine;
  refs.check.textContent = view.check;
}

export function mount(host) {
  let toy = null; // set below; controls only fire on user input, after mount
  const unmountControls = mountControls(host, (patch) => toy?.set(patch));
  const refs = mountReadouts(host);
  toy = createToyState(INITIAL_STATE, (state) => paint(refs, toyView(state)));
  return () => {
    toy.destroy();
    unmountControls();
    host.replaceChildren();
  };
}
