// decoder-anatomy toy, "Where do the parameters live?": mount(el, ctx) → destroy. One state object, one render;
// every string it prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { formatCount } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';
import { TOY_LIMITS } from './format.js';
import { PRESET_CHIPS, INITIAL_STATE, toyView, kimiLine, v3GapLine, tryThis } from './toy-view.js';

const BAR_W = 264; // wide enough that the toy keeps the storyboard's folds (only "other" folds); scrolls in its box at 400 px
// The experts slider's stops as chips: "dense" and "8, like the animation" are the storyboard's two named ones.
const EXPERT_OPTIONS = Object.freeze(TOY_LIMITS.experts.map((v) => ({ value: v, label: v === 0 ? 'dense' : v === 8 ? '8, like the animation' : String(v) })));
const READOUTS = Object.freeze([['total', 'Total'], ['active', 'Active'], ['active-with-embedding', 'Active, counting the lookup'], ['per-block', 'Per block']]);

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

function tryThisList(data) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items), el('p', {}, [output('kimi-line', { textContent: kimiLine(data) })])];
}

function convention(data, inline) {
  const mistral = lookupFact(data?.models, 'mistral-large-4', 'active_params')?.value;
  const labs = mistral ? `Mistral Large 4 quotes ${formatCount(mistral)} routed-active and a higher figure with the embedding counted` : 'their published counts differ by the embedding table';
  return el('p', { className: 'toy-note' }, [
    'Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication. (If the table is shared with the unembedding, as in GPT-3, it is counted once.) Counting the lookup too gives ',
    inline, `; labs differ: ${labs}.`,
  ]);
}

function partsTable() {
  const head = el('thead', {}, [el('tr', {}, ['part', 'count', '% of total', '% of active'].map((t) => el('th', { scope: 'col', textContent: t })))]);
  const body = el('tbody');
  const table = el('table', { className: 'parts-table' }, [el('caption', { className: 'muted', textContent: 'Where the parameters are' }), head, body]);
  table.dataset.readout = 'parts';
  return { table, body };
}

function paintRows(body, rows) {
  body.replaceChildren(...rows.map((r) => {
    const cells = [['count', r.count], ['share', r.share], ['active-share', r.activeShare]].map(([col, text]) => {
      const td = el('td', { textContent: text });
      td.dataset.col = col;
      return td;
    });
    const tr = el('tr', {}, [el('th', { scope: 'row', textContent: r.label }), ...cells]);
    tr.dataset.part = r.part;
    return tr;
  }));
}

function readoutGrid(out) {
  return el('div', { className: 'readouts' }, READOUTS.map(([name, label]) => el('div', { className: 'readout' }, [
    el('span', { textContent: label }), out(name), ...(name === 'active' ? [out('active-pct')] : []),
  ])));
}

function buildDom(host, data) {
  const outputs = {};
  const out = (name) => (outputs[name] = output(name));
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  outputs['check-work'] = pre;
  const refs = {
    outputs, chips: el('div'), sliders: [el('div'), el('div'), el('div')], expertChips: el('div'), inline: output('active-with-embedding-inline'),
    v3: output('v3-gap', { className: 'toy-note', textContent: v3GapLine(data) }),
    bar: G.svgEl('svg', { role: 'group', 'aria-label': 'parameter shares' }), parts: partsTable(),
  };
  refs.controls = el('div', { className: 'toy-controls' }, [...refs.sliders, refs.expertChips, el('p', { className: 'toy-note' }, [out('experts-note')])]);
  host.append(
    refs.chips, el('p', {}, [refs.v3]), refs.controls, el('p', { className: 'toy-note' }, [out('spec')]),
    readoutGrid(out), convention(data, refs.inline), el('p', {}, [out('active-note')]),
    el('p', {}, [out('published-gap')]), el('p', {}, [out('published-active-gap')]),
    el('div', { className: 'scroll-x' }, [refs.bar]), el('div', { className: 'scroll-x' }, [refs.parts.table]),
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data),
  );
  return refs;
}

function paintBar(svg, parts) {
  svg.replaceChildren();
  G.shareBar(svg, { x: 4, y: 6, w: BAR_W, parts, label: 'parameter shares' });
  G.fitViewBox(svg, 8);
}

function paint(refs, view, state) {
  const text = { total: view.total, active: view.active, 'active-pct': view.activePct, 'active-with-embedding': view.activeWithEmbedding, 'per-block': view.perBlock,
    'active-note': view.activeNote, 'published-gap': view.gap, 'published-active-gap': view.activeGap, 'experts-note': view.expertsNote, spec: view.spec, 'check-work': view.checkWork };
  Object.entries(text).forEach(([name, value]) => { refs.outputs[name].textContent = value; });
  refs.inline.textContent = view.activeWithEmbedding;
  refs.v3.hidden = state.preset !== 'deepseekV3';
  paintRows(refs.parts.body, view.rows);
  paintBar(refs.bar, view.bar);
  // Real presets: the spec line replaces the toy's controls, which are hidden (and disabled) rather than left showing toy values.
  const real = state.preset !== 'toy';
  refs.controls.hidden = real;
  refs.controls.querySelectorAll('input, button').forEach((c) => { c.disabled = real; });
}

function mountControls(refs, set) {
  const [layersHost, dModelHost, expertsHost] = refs.sliders;
  let chips = null;
  const layers = mountSlider(layersHost, { id: 'layers', label: 'Blocks (N)', min: TOY_LIMITS.layers[0], max: TOY_LIMITS.layers[1], step: 1, value: INITIAL_STATE.layers, unit: 'blocks', onInput: (v) => set({ layers: v }) });
  const dModel = mountSlider(dModelHost, { id: 'dModel', label: 'd_model', values: TOY_LIMITS.dModel, value: INITIAL_STATE.dModel, unit: 'numbers per row', onInput: (v) => set({ dModel: v }) });
  const experts = mountSlider(expertsHost, {
    id: 'experts', label: 'Routed experts (top-2, each half the dense hidden)', values: TOY_LIMITS.experts, value: INITIAL_STATE.experts,
    format: (v) => (v === 0 ? '0 (dense MLP)' : `${v} experts`), onInput: (v) => { chips?.set(v); set({ experts: v }); },
  });
  chips = mountChoice(refs.expertChips, { id: 'experts-preset', label: 'Experts', variant: 'chips', value: INITIAL_STATE.experts, options: EXPERT_OPTIONS, onChange: (v) => experts.set(v) });
  const preset = mountChoice(refs.chips, { id: 'preset', label: 'Model', variant: 'chips', value: INITIAL_STATE.preset, options: PRESET_CHIPS.map(({ value, label }) => ({ value, label })), onChange: (v) => set({ preset: v }) });
  return [layers, dModel, experts, chips, preset];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
