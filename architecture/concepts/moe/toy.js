// moe toy, "Route, count, rebalance.": mount(el, ctx) → destroy. One state object, one render; every string it prints comes
// from toyView (toy-view.js), so the page shows exactly what the tests check. Panel A counts parameters, panel B balances a batch.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { appendRich } from '@shared/lesson-page.js';
import { TOY_LIMITS, BATCH, expertName } from './format.js';
import { INITIAL_STATE, REAL_MODELS, toyView, tryThis } from './toy-view.js';
import { routingGrid, loadBars } from './figures.js';
import { MAX_LOAD } from './numbers.js';

const BALANCE_BARS = Object.freeze({ x: 4, y: 20, w: 344, h: 130 });
const FAIR_SHARE = (BATCH.tokens * BATCH.k) / BATCH.experts;
const SPLIT_OPTIONS = Object.freeze(TOY_LIMITS.split.map((v) => ({ value: v, label: String(v) })));
const REAL_OPTIONS = Object.freeze([{ value: 'toy', label: 'toy' }, ...REAL_MODELS]);
const EXPERT_NAMES = Object.freeze(Array.from({ length: BATCH.experts }, (_, e) => expertName(e)));

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};
const note = (...children) => el('p', { className: 'toy-note' }, children);
const scroll = (child) => el('div', { className: 'scroll-x' }, [child]);

// ---- tables (rebuilt on every render; cells carry data-readout names for the tests) ----
const toyTotals = (a) => readoutTable({
  head: ['Whole toy, 2 blocks', 'Parameters'],
  rows: [
    { label: 'Total', cells: [{ value: a.total, name: 'total' }] },
    { label: 'Active', sub: 'per token', cells: [{ value: a.active, name: 'active' }] },
    { label: 'Active share', sub: 'of the total', cells: [{ value: a.activeShare, name: 'active-share' }] },
  ],
});

const toyPerBlock = (a) => readoutTable({
  head: ['Per block', ''],
  rows: [
    { label: 'Expert parameters per token', cells: [{ value: a.expertActive, name: 'expert-active' }] },
    { label: 'Router parameters', cells: [{ value: a.router, name: 'router' }] },
    { label: 'Possible expert combinations', cells: [{ value: a.combos, name: 'combos' }] },
  ],
});

const realTable = (a) => readoutTable({
  head: [a.label, 'Published'],
  rows: [
    { label: 'Total parameters', cells: [{ value: a.total, name: 'real-total' }] },
    { label: 'Active parameters', sub: a.activeReported ? 'reported' : 'per token', cells: [{ value: a.active, name: 'real-active' }] },
    { label: 'Active share', cells: [{ value: a.share, name: 'real-share' }] },
    { label: 'Routed experts per token', cells: [{ value: a.routed, name: 'real-routed' }] },
    { label: 'Share of routed experts used', cells: [{ value: a.routedShare, name: 'real-routed-share' }] },
    { label: 'Shared experts', cells: [{ value: a.shared, name: 'real-shared' }] },
    { label: 'Ways to choose them', sub: 'per token', cells: [{ value: a.combos, name: 'real-combos' }] },
  ],
});

const routingTable = (routing) => readoutTable({
  head: ['Word', 'Picks', 'Gate weights'],
  rows: routing.rows.map((r, t) => ({ label: r.token, cells: [{ value: r.picks, name: `picks-${t + 1}` }, { value: r.gates, name: `gates-${t + 1}` }] })),
});

const balanceTable = (b) => readoutTable({
  head: ['', ...EXPERT_NAMES],
  rows: [
    { label: 'Load', sub: 'tokens', cells: b.loadTexts.map((value, e) => ({ value, name: `load-${e + 1}` })) },
    { label: 'Bias', cells: b.biasTexts.map((value, e) => ({ value, name: `bias-${e + 1}` })) },
  ],
});

const imbalanceTable = (b) => readoutTable({
  head: ['Busiest ÷ fair share', ''],
  rows: [{ label: `${b.busiest} ÷ ${b.fairShare}`, sub: 'busiest load ÷ fair share', cells: [{ value: b.imbalance, name: 'imbalance' }] }],
});

// ---- figures ----
function paintRouting(svg, routing) {
  svg.replaceChildren();
  if (!routing) return;
  routingGrid(svg, { picks: routing.rows.map((r) => r.pickIndices) });
  loadBars(svg, { values: routing.loads });
  G.fitViewBox(svg, 8);
}

function paintBalance(svg, b) {
  svg.replaceChildren();
  G.bars(svg, { ...BALANCE_BARS, values: b.loads, labels: EXPERT_NAMES, max: MAX_LOAD, reference: { value: FAIR_SHARE, label: `fair share ${FAIR_SHARE}` }, format: String, label: 'tokens per expert' });
  G.fitViewBox(svg, 8);
}

// ---- DOM ----
function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => {
    const li = el('li');
    appendRich(li, `${prompt} → `, ctx);
    li.append(el('strong', { textContent: `Insight: ${insight}` }));
    appendRich(li, rest, ctx);
    return li;
  });
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, ctx) {
  const refs = {
    real: el('div'), routed: el('div'), split: el('div'), shared: el('div'), gamma: el('div'), step: el('div'),
    totals: el('div'), perBlock: el('div'), realBox: el('div'), routingTable: el('div'), balanceTable: el('div'), imbalance: el('div'),
    routingSvg: G.svgEl('svg', { role: 'img', 'aria-label': 'router scores for the four words and the load on each expert' }),
    balanceSvg: G.svgEl('svg', { role: 'img', 'aria-label': 'tokens per expert in one batch, against the fair share' }),
    spec: output('spec'), edge: output('edge-note'), routingNote: output('routing-note'), wobble: output('wobble-note'),
  };
  refs.splitControls = el('div', { className: 'toy-controls' }, [refs.split, refs.shared]);
  refs.toyControls = el('div', { className: 'toy-controls' }, [refs.routed, refs.splitControls]);
  refs.toyBox = el('div', {}, [refs.totals, refs.perBlock, note(refs.edge), note(refs.routingNote)]);
  refs.routingBox = el('div', { className: 'scroll-x' }, [refs.routingSvg, refs.routingTable]);
  refs.definition = note();
  host.append(
    el('h4', { textContent: 'Panel A: count the parameters' }), refs.real, refs.toyControls, note(refs.spec),
    el('div', { className: 'toy-tables' }, [refs.toyBox, refs.realBox]), refs.definition, refs.routingBox,
    el('h4', { textContent: 'Panel B: balance a batch' }), el('div', { className: 'toy-controls' }, [refs.gamma, refs.step]),
    scroll(refs.balanceSvg), scroll(refs.balanceTable), refs.imbalance, note(refs.wobble),
    ...tryThisList(ctx?.data, ctx),
  );
  return refs;
}

function paintPanelA(refs, view) {
  const { a, real } = view;
  refs.toyControls.hidden = real;
  refs.toyBox.hidden = real;
  refs.realBox.hidden = !real;
  refs.routingBox.hidden = real || !a.routing;
  refs.definition.textContent = view.definition;
  refs.definition.hidden = real;
  if (real) {
    refs.realBox.replaceChildren(realTable(a));
    refs.spec.textContent = `${a.label}: published figures from data/models.json.`;
    return;
  }
  refs.spec.textContent = a.spec;
  refs.totals.replaceChildren(toyTotals(a));
  refs.perBlock.replaceChildren(toyPerBlock(a));
  refs.edge.textContent = a.edgeNote;
  refs.routingNote.textContent = a.routingNote;
  refs.routingTable.replaceChildren(...(a.routing ? [routingTable(a.routing)] : []));
  paintRouting(refs.routingSvg, a.routing);
}

function paint(refs, view, state) {
  paintPanelA(refs, view);
  refs.splitControls.hidden = state.routed === 0;
  refs.balanceTable.replaceChildren(balanceTable(view.b));
  refs.imbalance.replaceChildren(imbalanceTable(view.b));
  refs.wobble.textContent = view.b.note;
  paintBalance(refs.balanceSvg, view.b);
}

function mountControls(refs, set) {
  return [
    mountChoice(refs.real, { id: 'real', label: 'Compare a real model', variant: 'chips', value: INITIAL_STATE.real, options: REAL_OPTIONS, onChange: (v) => set({ real: v }) }),
    mountSlider(refs.routed, { id: 'routed', label: 'Routed experts', values: TOY_LIMITS.routed, value: INITIAL_STATE.routed, format: (v) => (v === 0 ? '0 (dense MLP)' : `${v} experts`), onInput: (v) => set({ routed: v }) }),
    mountChoice(refs.split, { id: 'split', label: 'Split each expert into (hidden 8, 4, 2)', value: INITIAL_STATE.split, options: SPLIT_OPTIONS, onChange: (v) => set({ split: v }) }),
    mountToggle(refs.shared, { id: 'shared', label: 'Shared expert', value: INITIAL_STATE.shared, onChange: (on) => set({ shared: on }) }),
    mountSlider(refs.gamma, { id: 'gamma', label: 'Balancing step', values: TOY_LIMITS.gamma, value: INITIAL_STATE.gamma, format: (v) => (v === 0 ? '0 (off)' : String(v)), onInput: (v) => set({ gamma: v }) }),
    mountSlider(refs.step, { id: 'step', label: 'Batch', min: TOY_LIMITS.step[0], max: TOY_LIMITS.step[1], step: 1, value: INITIAL_STATE.step, format: (v) => `step ${v}`, onInput: (v) => set({ step: v }) }),
  ];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, ctx);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
