// kv-cache toy, "Count the work, then weigh the memory" (storyboard §6): mount(el, ctx) → destroy.
// One state object, one paint; every string it prints comes from toyView (toy-view.js), so the page shows what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { appendRich } from '@shared/lesson-page.js';
import { INITIAL_STATE, SLIDER_VALUES, SHAPE_LIMITS, int, plural } from './format.js';
import { MODEL_CHIPS, GPUS, toyView, tryThis, snapShape } from './toy-view.js';

const GPU_SVG = Object.freeze({ w: 440, h: 100 });
const GPU_BOX = Object.freeze({ w: 96, h: 72 });
const GPU_X = Object.freeze([10, 240]);
const STOP_LINE = '1,048,576 (2²⁰) is a slider stop; each model\'s own context comes from its data entry.';
const SIZE_NOTE = 'Sizes are decimal (kB, MB, GB), as GPU memory is sold; per-token sizes also print their exact byte count.';

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};
const checkBox = (name) => {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = name;
  return pre;
};
const note = (children) => el('p', { className: 'toy-note' }, children);

function buildDom(host, ctx) {
  const refs = {
    toggle: el('div'), sliders: Object.fromEntries(['prompt', 'reply', 'context', 'sequences', 'layers', 'kvHeads', 'headDim', 'bytes'].map((k) => [k, el('div')])),
    chips: el('div'), work: el('div'), shapeBox: el('div', { className: 'toy-controls' }), shapeTable: el('div'),
    tables: el('div', { className: 'toy-tables' }), gpus: G.svgEl('svg', { width: GPU_SVG.w, height: GPU_SVG.h, viewBox: `0 0 ${GPU_SVG.w} ${GPU_SVG.h}`, role: 'group', 'aria-label': 'GPU memory fill' }),
    checkA: checkBox('check-work-a'), checkB: checkBox('check-work-b'),
    contextNote: output('context-note'), formulaNote: output('formula-note'),
  };
  refs.shapeBox.append(...['layers', 'kvHeads', 'headDim', 'bytes'].map((k) => refs.sliders[k]));
  const items = tryThis(ctx?.data).map(({ text, insight, rest }) => el('li', {}, [`${text} → `, el('strong', { textContent: `Insight: ${insight}` }), appendRich(el('span'), rest, ctx)]));
  host.append(
    el('h4', { textContent: 'A. Count the work' }),
    el('div', { className: 'toy-controls' }, [refs.toggle, refs.sliders.prompt, refs.sliders.reply]),
    refs.work, el('h4', { textContent: 'Check my work' }), refs.checkA,
    el('h4', { textContent: 'B. Weigh the memory' }),
    refs.chips, note([refs.formulaNote]), refs.shapeBox, el('div', { className: 'scroll-x' }, [refs.shapeTable]),
    el('div', { className: 'toy-controls' }, [refs.sliders.context, refs.sliders.sequences]),
    note([refs.contextNote]), note([STOP_LINE]), refs.tables, el('div', { className: 'scroll-x' }, [refs.gpus]), note([SIZE_NOTE]),
    el('h4', { textContent: 'Check my work' }), refs.checkB,
    el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items),
  );
  return refs;
}

const cellOf = (c, name) => ({ value: c.value, sub: c.sub, name });

function workTable(view, state) {
  const { work } = view;
  const [now, other] = state.cache ? ['cache on', 'cache off'] : ['cache off', 'cache on'];
  return readoutTable({
    head: ['', 'Positions computed', 'Keys read (per head, per layer)'],
    rows: [
      { label: 'Your setting', sub: now, cells: [cellOf(work.current.positions, 'positions'), cellOf(work.current.keyReads, 'key-reads')] },
      { label: 'Flipped', sub: other, cells: [cellOf(work.flipped.positions, 'positions-flipped'), cellOf(work.flipped.keyReads, 'key-reads-flipped')] },
      { label: 'The cache saves', cells: [{ value: work.positionsRatio, sub: 'fewer', name: 'positions-ratio' }, { value: work.keyReadsRatio, sub: 'fewer', name: 'key-reads-ratio' }] },
    ],
  });
}

function memoryTables(view) {
  const bytes = readoutTable({
    head: ['Memory', 'Size'],
    rows: [
      { label: 'Bytes per token', cells: [cellOf(view.bytesPerToken, 'bytes-per-token')] },
      { label: 'Cache for one conversation', cells: [cellOf(view.cacheOne, 'cache-one')] },
      { label: 'Cache for all conversations', cells: [cellOf(view.cacheAll, 'cache-all')] },
      { label: 'Read per decode step', cells: [cellOf(view.readPerStep, 'read-per-step')] },
    ],
  });
  const gpus = readoutTable({
    head: ['GPU', 'Share of its memory', 'Fits?'],
    rows: view.gpus.map((g) => ({ label: g.label, cells: [{ value: g.share, name: `share-${g.id}` }, { value: g.fit, name: `fit-${g.id}` }] })),
  });
  return [bytes, gpus].map((t) => el('div', { className: 'scroll-x' }, [t]));
}

function paintGpus(svg, view) {
  svg.replaceChildren();
  view.gpus.forEach((g, i) => G.gpu(svg, { x: GPU_X[i], y: 4, ...GPU_BOX, memFill: g.fill, label: GPUS[i].label }));
}

function paint(refs, view, state) {
  refs.work.replaceChildren(el('div', { className: 'scroll-x' }, [workTable(view, state)]));
  refs.checkA.textContent = view.checkA;
  refs.checkB.textContent = view.checkB;
  refs.tables.replaceChildren(...memoryTables(view));
  refs.contextNote.textContent = view.contextNote;
  refs.formulaNote.textContent = view.formulaNote;
  const toy = state.model === 'toy';
  refs.shapeBox.hidden = !toy;
  refs.shapeTable.hidden = toy || view.shapeRows.length === 0;
  refs.shapeTable.replaceChildren(...(view.shapeRows.length ? [readoutTable({ head: ['Shape', 'Value'], rows: view.shapeRows.map((r) => ({ label: r.label, cells: [{ value: r.value }] })) })] : []));
  paintGpus(refs.gpus, view);
}

function mountControls(refs, set, data) {
  const slider = (key, props) => mountSlider(refs.sliders[key], { id: key, value: INITIAL_STATE[key], onInput: (v) => set({ [key]: v }), ...props });
  const stops = (key, label, unit, format = int) => slider(key, { label, values: SLIDER_VALUES[key], unit, format });
  const shape = [
    slider('layers', { label: 'Blocks (layers)', min: SHAPE_LIMITS.layers[0], max: SHAPE_LIMITS.layers[1], step: 1, format: (v) => plural(v, 'block') }),
    slider('kvHeads', { label: 'KV heads', min: SHAPE_LIMITS.kvHeads[0], max: SHAPE_LIMITS.kvHeads[1], step: 1, format: (v) => plural(v, 'head') }),
    stops('headDim', 'Head size', 'numbers'),
    stops('bytes', 'Bytes per number', 'bytes'),
  ];
  const chooseModel = (model) => {
    const snap = snapShape(model, data);
    set({ model, ...(snap ?? {}) });
    if (snap) shape.forEach((s, i) => s.set([snap.layers, snap.kvHeads, snap.headDim, snap.bytes][i]));
  };
  return [
    mountToggle(refs.toggle, { id: 'cache', label: 'KV cache', value: INITIAL_STATE.cache, onChange: (cache) => set({ cache }) }),
    stops('prompt', 'Prompt length', 'tokens'), stops('reply', 'Reply length', 'tokens'),
    mountChoice(refs.chips, { id: 'model', label: 'Model', variant: 'chips', value: INITIAL_STATE.model, options: MODEL_CHIPS, onChange: chooseModel }),
    stops('context', 'Tokens in the cache', 'tokens', (v) => (v === 1_048_576 ? '1,048,576 (2²⁰)' : int(v))),
    stops('sequences', 'Conversations at once', '', (v) => plural(v, 'conversation')),
    ...shape,
  ];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, ctx);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch), data);
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
