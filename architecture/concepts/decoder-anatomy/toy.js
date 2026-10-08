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
import { PRESET_CHIPS, INITIAL_STATE, toyView, kimiLine, v3GapLine } from './toy-view.js';

const BAR_W = 264; // wide enough that the toy keeps the storyboard's folds (only "other" folds), narrow enough for 400 px
const EXPERT_CHIPS = Object.freeze([{ value: 0, label: 'dense' }, { value: 8, label: '8, like the animation' }]);
const READOUTS = Object.freeze([['total', 'Total'], ['active', 'Active'], ['active-with-embedding', 'Active, counting the lookup'], ['per-block', 'Per block']]);

const output = (name, extra = {}) => {
  const o = el('output', extra);
  o.dataset.readout = name;
  return o;
};

const TRY_THIS = Object.freeze([
  ['Toy preset, read the shares: MLP 48.7%, attention 32.5%, embedding + head 16.2%, norms 2.5%. Tap GPT-3: MLP 66.4%, attention 33.2%, embedding 0.35%. Tap gpt-oss-120b: experts 98.2%, attention 0.8%.',
    'Most parameters think per token; they don\'t talk between tokens.',
    ' A dense block\'s MLP is 8·d² against attention\'s 4·d² (1.5× in the toy, whose hidden is 2·d); turn the MLP into experts and attention shrinks to a rounding error.'],
  ['Toy preset, set Blocks = 1 and d_model = 8: embedding + head = 256 of 920, 27.8%. Now Blocks = 8, d_model = 64: 0.6%. The far end is in the line below this list.',
    'The embedding table is vocab × d_model, the blocks are about 10–12 · N · d_model² (10 in this toy, 12 in GPT-3); the table only matters in small models.',
    ' Read gpt-oss-120b\'s "% of active" column: its unembedding is 579M of 5.13B active, 11%, so in a sparse model the head is a visible slice of what each token actually multiplies.'],
  ['Toy preset, Blocks = 2, slide Routed experts 0 → 8 → 16: total 1,576 → 4,008 → 7,208; active 1,448 → 1,576 → 1,704. Then tap DeepSeek-V4-Pro: 1.6T total, 49B active, 3.1%.',
    'MoE grows total with the number of experts; active barely moves.',
    ' Two active experts of hidden 8 (2 × 192 = 384) are exactly one dense MLP of hidden 16, so the only active growth is the router: one row of d_model per expert per block (64 → 128 here). Real 2026 experts are fine-grained in the same way.'],
]);

function tryThis(data) {
  const items = TRY_THIS.map(([prompt, insight, rest]) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items), el('p', {}, [output('kimi-line', { textContent: kimiLine(data) })])];
}

function convention(data) {
  const mistral = lookupFact(data?.models, 'mistral-large-4', 'active_params')?.value;
  const labs = mistral ? `Mistral Large 4 quotes ${formatCount(mistral)} routed-active and a higher figure with the embedding counted` : 'their published counts differ by the embedding table';
  return el('p', { className: 'toy-note' }, [
    'Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication. (If the table is shared with the unembedding, as in GPT-3, it is counted once.) Counting the lookup too gives ',
    output('active-with-embedding-inline'), `; labs differ: ${labs}.`,
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
    const tr = el('tr', {}, [el('th', { scope: 'row', textContent: r.label }), ...[['count', r.count], ['share', r.share], ['active-share', r.activeShare]].map(([col, text]) => {
      const td = el('td', { textContent: text });
      td.dataset.col = col;
      return td;
    })]);
    tr.dataset.part = r.part;
    return tr;
  }));
}

function buildDom(host, data) {
  const refs = { chips: el('div'), sliders: [el('div'), el('div'), el('div')], expertChips: el('div', { className: 'choice choice--chips' }), outputs: {} };
  const out = (name) => (refs.outputs[name] = output(name));
  refs.v3 = output('v3-gap', { className: 'toy-note', textContent: v3GapLine(data) });
  refs.bar = G.svgEl('svg', { role: 'group', 'aria-label': 'parameter shares', style: 'max-width:100%;height:auto' });
  refs.parts = partsTable();
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  refs.outputs['check-work'] = pre;
  const readouts = el('div', { className: 'readouts' }, READOUTS.map(([name, label]) => el('div', { className: 'readout' }, [el('span', { textContent: label }), out(name), ...(name === 'active' ? [out('active-pct')] : [])])));
  const conv = convention(data);
  refs.inline = conv.querySelector('[data-readout="active-with-embedding-inline"]');
  host.append(
    refs.chips, el('p', {}, [refs.v3]),
    el('div', { className: 'toy-controls' }, [...refs.sliders, refs.expertChips, el('p', { className: 'toy-note' }, [out('experts-note')])]),
    el('p', { className: 'toy-note' }, [out('spec')]),
    readouts, conv, el('p', {}, [out('active-note')]),
    el('p', {}, [out('published-gap')]), el('p', {}, [out('published-active-gap')]),
    el('div', { className: 'scroll-x' }, [refs.bar]), el('div', { className: 'scroll-x' }, [refs.parts.table]),
    el('h4', { textContent: 'Check my work' }), pre,
    ...tryThis(data),
  );
  return refs;
}

function paintBar(svg, parts) {
  svg.replaceChildren();
  G.shareBar(svg, { x: 4, y: 6, w: BAR_W, parts, label: 'parameter shares' });
  G.fitViewBox(svg, 8);
}

function paint(refs, view, state) {
  const o = refs.outputs;
  const text = { total: view.total, active: view.active, 'active-pct': view.activePct, 'active-with-embedding': view.activeWithEmbedding, 'per-block': view.perBlock,
    'active-note': view.activeNote, 'published-gap': view.gap, 'published-active-gap': view.activeGap, 'experts-note': view.expertsNote, spec: view.spec, 'check-work': view.checkWork };
  Object.entries(text).forEach(([name, value]) => { o[name].textContent = value; });
  refs.inline.textContent = view.activeWithEmbedding;
  refs.v3.hidden = state.preset !== 'deepseekV3';
  paintRows(refs.parts.body, view.rows);
  paintBar(refs.bar, view.bar);
  const real = state.preset !== 'toy';
  refs.sliders.forEach((h) => { h.querySelector('input').disabled = real; });
  refs.expertChips.querySelectorAll('button').forEach((b) => {
    b.disabled = real;
    b.setAttribute('aria-pressed', String(!real && Number(b.dataset.expertsChip) === state.experts));
  });
}

function expertChipButtons(host, onPick) {
  host.setAttribute('role', 'group');
  host.setAttribute('aria-label', 'Experts presets');
  const buttons = EXPERT_CHIPS.map((c) => {
    const b = el('button', { type: 'button', className: 'choice-option', textContent: c.label });
    b.dataset.expertsChip = String(c.value);
    return b;
  });
  const onClick = (event) => {
    const b = event.target.closest('[data-experts-chip]');
    if (b && !b.disabled) onPick(Number(b.dataset.expertsChip));
  };
  host.replaceChildren(...buttons);
  host.addEventListener('click', onClick);
  return () => { host.removeEventListener('click', onClick); host.replaceChildren(); };
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data);
  let toy = null;
  const set = (patch) => toy?.set(patch);
  const [layersHost, dModelHost, expertsHost] = refs.sliders;
  const layers = mountSlider(layersHost, { id: 'layers', label: 'Blocks (N)', min: TOY_LIMITS.layers[0], max: TOY_LIMITS.layers[1], step: 1, value: INITIAL_STATE.layers, unit: 'blocks', onInput: (v) => set({ layers: v }) });
  const dModel = mountSlider(dModelHost, { id: 'dModel', label: 'd_model', values: TOY_LIMITS.dModel, value: INITIAL_STATE.dModel, unit: 'numbers per row', onInput: (v) => set({ dModel: v }) });
  const experts = mountSlider(expertsHost, { id: 'experts', label: 'Routed experts (top-2, each half the dense hidden)', values: TOY_LIMITS.experts, value: INITIAL_STATE.experts, format: (v) => (v === 0 ? '0 (dense MLP)' : `${v} experts`), onInput: (v) => set({ experts: v }) });
  const unchip = expertChipButtons(refs.expertChips, (v) => experts.set(v));
  const preset = mountChoice(refs.chips, { id: 'preset', label: 'Model', variant: 'chips', value: INITIAL_STATE.preset, options: PRESET_CHIPS.map(({ value, label }) => ({ value, label })), onChange: (v) => set({ preset: v }) });
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    toy.destroy();
    [layers, dModel, experts, preset].forEach((c) => c.destroy());
    unchip();
    host.replaceChildren();
  };
}
