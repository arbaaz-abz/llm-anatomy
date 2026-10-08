// "What does each trick read, and what does it keep?" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { CONTEXTS, SINK_LOGITS, GATES, TOKENS } from './numbers.js';
import { int } from './format.js';
import { INITIAL_STATE, PATTERN_OPTIONS, realOptions, toyView } from './toy-view.js';
import { numberRowView, newFigure, drawPattern, drawState } from './toy-dom.js';

const WEIGHT_SCALE = 1;
const STATE_SCALE = 6;

// A slider lives in a plain wrapper, because .slider is a grid and a hidden attribute would not hide it.
function sliderBox() {
  return el('div');
}

function buildDom(host) {
  const refs = {
    pattern: el('div'), real: el('div'),
    boxes: Object.fromEntries(['window', 'topK', 'merge', 'query', 'sinkLogit', 'gate', 'context'].map((n) => [n, sliderBox()])),
    figure: newFigure(), table: el('div'), sink: numberRowView('sink-weights', 'sink'), linear: numberRowView('linear-output', 'o_sat'),
    realTable: el('div'), cacheTable: el('div'), note: el('p', { className: 'toy-note' }),
  };
  refs.sinkNote = el('p', { className: 'toy-note' });
  refs.linearNote = el('p', { className: 'toy-note', textContent: 'The state has 16 numbers however long the text; the gate halves it before each new token.' });
  const b = refs.boxes;
  host.append(
    refs.pattern, b.window, b.topK, b.merge, b.sinkLogit, b.gate, b.query,
    el('div', { className: 'scroll-x' }, [refs.figure]), refs.table, refs.sink.node, refs.sinkNote, refs.linear.node, refs.linearNote,
    el('h4', { textContent: 'At real scale' }), refs.real, b.context, refs.realTable, refs.cacheTable, refs.note,
  );
  return refs;
}

function mountControls(refs, set, data) {
  const b = refs.boxes;
  const sliders = [
    mountSlider(b.window, { id: 'window', label: 'Window', values: [2, 4, 8], value: INITIAL_STATE.window, unit: 'tokens', onInput: (v) => set({ window: v }) }),
    mountSlider(b.topK, { id: 'topK', label: 'Top-k (sparse, compressed)', values: [1, 2, 4, 8], value: INITIAL_STATE.topK, onInput: (v) => set({ topK: v }) }),
    mountSlider(b.merge, { id: 'merge', label: 'Tokens per entry (compressed)', values: [2, 4, 8], value: INITIAL_STATE.merge, unit: 'tokens', onInput: (v) => set({ merge: v }) }),
    mountSlider(b.query, { id: 'query', label: 'Follow token', min: 1, max: TOKENS, step: 1, value: INITIAL_STATE.query, format: (v) => `token ${v}`, onInput: (v) => set({ query: v }) }),
    mountSlider(b.sinkLogit, { id: 'sinkLogit', label: 'Sink logit', values: SINK_LOGITS, value: INITIAL_STATE.sinkLogit, format: int, onInput: (v) => set({ sinkLogit: v }) }),
    mountSlider(b.gate, { id: 'gate', label: 'Gate (how much of the state survives each token)', values: GATES, value: INITIAL_STATE.gate, onInput: (v) => set({ gate: v }) }),
    mountSlider(b.context, { id: 'context', label: 'Context', values: CONTEXTS, value: INITIAL_STATE.context, format: int, unit: 'tokens', onInput: (v) => set({ context: v }) }),
  ];
  const choices = [
    mountChoice(refs.pattern, { id: 'pattern', label: 'Layer type', options: PATTERN_OPTIONS, value: INITIAL_STATE.pattern, onChange: (v) => set({ pattern: v }) }),
    mountChoice(refs.real, { id: 'real', label: 'Model', variant: 'chips', options: realOptions(data), value: INITIAL_STATE.real, onChange: (v) => set({ real: v }) }),
  ];
  return [...sliders, ...choices];
}

const patternTable = (view) => readoutTable({
  head: ['The followed token\'s layer', 'Count'],
  rows: [
    { label: `Entries read by token ${view.row + 1}`, cells: [{ value: view.reads, sub: view.readsSub, name: 'reads' }] },
    { label: 'Cells read, all rows', cells: [{ value: view.cells, sub: view.cellsSub, name: 'cells' }] },
    { label: 'Entries stored in this layer', cells: [{ value: view.stored, sub: view.storedSub, name: 'stored' }] },
  ],
});

const realTables = (real) => ({
  layers: real.rows.length ? readoutTable({ head: real.head, rows: real.rows.map((r) => ({ ...r, cells: r.cells })) }) : el('p', { className: 'toy-note', textContent: real.note }),
  cache: readoutTable({ head: ['Whole model', 'Cache per conversation'], rows: [{ label: 'Cache per conversation', cells: [{ value: real.cache.value, sub: real.cache.sub, name: 'real-cache' }] }] }),
});

function paint(refs, state, data) {
  const view = toyView(state, data);
  Object.entries(view.controls).forEach(([name, visible]) => { refs.boxes[name].hidden = !visible; });
  const { pattern } = view;
  refs.table.hidden = !pattern;
  refs.figure.hidden = false;
  if (pattern) {
    drawPattern(refs.figure, pattern);
    refs.table.replaceChildren(patternTable(pattern));
  } else {
    drawState(refs.figure, view.linear);
    refs.table.replaceChildren();
  }
  refs.sink.node.hidden = refs.sinkNote.hidden = !view.sink;
  if (view.sink) {
    refs.sink.set(view.sink.values, 'weight', WEIGHT_SCALE, view.sink.label);
    refs.sinkNote.textContent = `${view.sink.sumText}: the sink takes the share the window does not.`;
  }
  refs.linear.node.hidden = refs.linearNote.hidden = !view.linear;
  if (view.linear) refs.linear.set(view.linear.output, 'output', STATE_SCALE);
  const tables = realTables(view.real);
  refs.realTable.replaceChildren(tables.layers);
  refs.cacheTable.replaceChildren(tables.cache);
  refs.note.textContent = view.real.note;
  refs.note.hidden = !view.real.note || !view.real.rows.length;
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch), data);
  toy = createToyState(INITIAL_STATE, (state) => paint(refs, state, data));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
