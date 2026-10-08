// "Share, group or compress" (storyboard §6): mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { requireShapes } from './shapes.js';
import { CONTEXTS } from './numbers.js';
import { int } from './format.js';
import { tryThis } from './try-this.js';
import { INITIAL_STATE, MODEL_CHIPS, SCHEME_OPTIONS, stopsFor, presetPatch, toyView } from './toy-view.js';
import { wiringFigure, patternMap } from './toy-dom.js';

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};
const note = (children) => el('p', { className: 'toy-note' }, children);

// A slider host is a grid (theme.css), which beats the hidden attribute, so a hidden slider hides its wrapper.
const wrap = (box, child) => { box.append(child); return box; };

function buildDom(host, shapes) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    chips: el('div'), scheme: el('div'), kv: el('div'), latent: el('div'), context: el('div'), pattern: el('div'), kvBox: el('div'), latentBox: el('div'),
    wiring: el('div', { className: 'scroll-x' }), maps: el('div', { className: 'toy-maps' }), table: el('div'),
    spec: output('spec'), groups: output('groups'), whatIf: output('what-if'), check: pre,
  };
  const items = tryThis(shapes).map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  host.append(
    refs.chips, note([refs.spec]),
    el('div', { className: 'toy-controls' }, [refs.scheme, wrap(refs.kvBox, refs.kv), wrap(refs.latentBox, refs.latent), refs.context]),
    refs.wiring, note([refs.groups]), refs.table, note([refs.whatIf]),
    el('h4', { textContent: 'Check my work' }), pre,
    el('h4', { textContent: 'Same keys, different questions' }), refs.pattern, refs.maps,
    el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items),
  );
  return refs;
}

function memoryTable(view) {
  return readoutTable({
    head: ['Per token', 'Value'], name: 'memory',
    rows: [
      { label: 'Stored per layer', sub: 'numbers', cells: [{ value: view.perLayer, name: 'per-layer' }] },
      { label: 'Bytes, all layers', sub: '2 bytes a number', cells: [{ value: view.bytes, sub: view.bytesSub, name: 'bytes-per-token' }] },
      { label: 'Times smaller than MHA', sub: 'same shape', cells: [{ value: view.ratio, name: 'times-smaller' }] },
      { label: view.cacheLabel, cells: [{ value: view.cache, name: 'cache' }] },
    ],
  });
}

function paint(refs, view, state) {
  refs.spec.textContent = view.spec;
  refs.groups.textContent = view.groups;
  refs.whatIf.textContent = view.whatIf;
  refs.check.textContent = view.checkWork;
  refs.kvBox.hidden = state.scheme !== 'gqa';
  refs.latentBox.hidden = state.scheme !== 'mla';
  refs.wiring.hidden = view.wiring === null;
  refs.wiring.replaceChildren(...(view.wiring ? [wiringFigure(view.wiring)] : []));
  refs.table.replaceChildren(memoryTable(view));
  refs.maps.replaceChildren(
    patternMap({ name: 'pattern-a', title: 'head A', ...view.headA }),
    patternMap({ name: 'pattern-b', title: view.headBTitle, ...view.headB }),
  );
}

// The two sliders whose stops depend on the model are mounted again when the model changes.
function mountModelSliders(refs, model, values, shapes, set) {
  const stops = stopsFor(model, shapes);
  return [
    mountSlider(refs.kv, { id: 'kvHeads', label: 'KV heads (GQA)', values: stops.kvHeads, value: values.kvHeads, unit: 'KV heads', format: int, onInput: (v) => set({ kvHeads: v }) }),
    mountSlider(refs.latent, { id: 'latent', label: 'Latent size (MLA)', values: stops.latent, value: values.latent, unit: 'numbers', format: int, onInput: (v) => set({ latent: v }) }),
  ];
}

function mountControls(refs, shapes, set) {
  let sliders = mountModelSliders(refs, INITIAL_STATE.model, INITIAL_STATE, shapes, set);
  const onModel = (model) => {
    const patch = presetPatch(model, shapes);
    sliders.forEach((s) => s.destroy());
    sliders = mountModelSliders(refs, model, patch, shapes, set);
    scheme.set(patch.scheme);
    set(patch);
  };
  const chips = mountChoice(refs.chips, { id: 'model', label: 'Shape', variant: 'chips', value: INITIAL_STATE.model, options: MODEL_CHIPS, onChange: onModel });
  const scheme = mountChoice(refs.scheme, { id: 'scheme', label: 'Keys and values', value: INITIAL_STATE.scheme, options: SCHEME_OPTIONS, onChange: (v) => set({ scheme: v }) });
  const context = mountSlider(refs.context, { id: 'context', label: 'Tokens in the cache', values: CONTEXTS, value: INITIAL_STATE.context, unit: 'tokens', format: int, onInput: (v) => set({ context: v }) });
  const toggle = mountToggle(refs.pattern, { id: 'pattern', label: 'Head B reads head A\'s keys and values', value: INITIAL_STATE.shareKeys, onChange: (on) => set({ shareKeys: on }) });
  return () => [chips, scheme, context, toggle, ...sliders].forEach((c) => c.destroy());
}

export function mount(host, ctx) {
  const shapes = requireShapes(ctx?.data);
  const refs = buildDom(host, shapes);
  let toy = null;
  const set = (patch) => toy?.set(patch);
  const destroyControls = mountControls(refs, shapes, set);
  toy = createToyState(INITIAL_STATE, (state) => paint(refs, toyView(state, shapes), state));
  return () => {
    toy.destroy();
    destroyControls();
    host.replaceChildren();
  };
}
