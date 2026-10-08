// training-memory toy, "Will it fit?": mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { appendRich } from '@shared/lesson-page.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { formatBytes } from '@math/core.js';
import { MODELS, CHIPS, RECIPES, STAGES, RECOMPUTE, TOY_LIMITS } from './format.js';
import { INITIAL_STATE, SCOPE_NOTE, toyView, tryThis, chipLabel } from './toy-view.js';

const BAR_W = 264; // a 2 B slice of the 16 B bar is 33 px, so every part prints inside its segment

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), appendRich(el('span'), rest, ctx)]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    chips: { model: el('div'), gpu: el('div') }, recipe: el('div'), stage: el('div'), dp: el('div'),
    shape: el('div', { className: 'toy-shape scroll-x' }, [el('div'), el('div')]), shapeNote: output('shape-note'),
    perParam: G.svgEl('svg', { role: 'group', 'aria-label': 'bytes per parameter' }), composition: G.svgEl('svg', { role: 'group', 'aria-label': 'per-GPU composition' }),
    table: el('div'), verdict: output('verdict'), holders: output('gpus-to-hold'), traffic: output('traffic'), compute: output('extra-compute'), pre,
  };
  const line = (label, node) => el('p', { className: 'toy-line' }, [`${label}: `, node]);
  host.append(
    el('p', { className: 'toy-note', textContent: SCOPE_NOTE }),
    refs.chips.model, refs.chips.gpu, el('div', { className: 'scroll-x' }, [refs.recipe]), el('div', { className: 'scroll-x' }, [refs.stage]), refs.dp, refs.shape, el('p', { className: 'toy-note' }, [refs.shapeNote]),
    el('div', { className: 'scroll-x' }, [refs.perParam]), el('div', { className: 'toy-tables' }, [refs.table]),
    el('p', {}, [refs.verdict]), el('div', { className: 'scroll-x' }, [refs.composition]),
    line('GPUs just to hold the state, any sharding, no activations', refs.holders), line('Traffic vs plain data parallel', refs.traffic), line('Extra compute from recomputation', refs.compute),
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data, ctx),
  );
  return refs;
}

function perGpuTable(view) {
  return readoutTable({
    head: ['Per GPU', 'Size'], name: 'per-gpu',
    rows: [
      { label: 'Weights', cells: [{ value: view.state.weights, name: 'weights' }] },
      { label: 'Gradients', cells: [{ value: view.state.grads, name: 'grads' }] },
      { label: 'Optimizer (master + moments)', cells: [{ value: view.state.optimizer, name: 'optimizer' }] },
      { label: 'State total', cells: [{ value: view.state.total, name: 'state-total' }] },
      { label: 'Saved activations', cells: [{ value: view.activations, name: 'activations' }] },
      { label: 'Total', sub: view.chipWords, cells: [{ value: view.total, name: 'total' }] },
    ],
  });
}

function paintBars(refs, view) {
  refs.perParam.replaceChildren();
  G.shareBar(refs.perParam, { x: 4, y: 6, w: BAR_W, tail: 'none', label: `bytes per parameter: ${view.perParam.total}`, parts: view.perParam.parts, format: (share) => formatBytes(share * view.perParam.totalBytes) });
  G.fitViewBox(refs.perParam, 8);
  refs.composition.replaceChildren();
  G.shareBar(refs.composition, { x: 4, y: 6, w: 420, label: 'per-GPU composition', parts: view.composition });
  G.fitViewBox(refs.composition, 8);
}

function paint(refs, view, state) {
  paintBars(refs, view);
  refs.table.replaceChildren(perGpuTable(view));
  const modeled = state.model === 'gpt-3';
  refs.shape.hidden = !modeled;
  refs.shapeNote.textContent = modeled ? '' : view.activations;
  refs.shapeNote.parentElement.hidden = modeled;
  refs.verdict.textContent = view.verdictLine;
  refs.verdict.dataset.verdict = view.fits ? 'fits' : 'does-not-fit';
  refs.holders.textContent = view.gpusToHold;
  refs.traffic.textContent = view.traffic;
  refs.compute.textContent = view.extraCompute.text;
  refs.compute.title = view.extraCompute.hover;
  refs.pre.textContent = view.checkWork;
}

function mountControls(refs, set) {
  const choice = (root, id, label, options, value, key, variant) => mountChoice(root, { id, label, options, value, variant, onChange: (v) => set({ [key]: v }) });
  return [
    choice(refs.chips.model, 'model', 'Model', MODELS.map(({ value, label }) => ({ value, label })), INITIAL_STATE.model, 'model', 'chips'),
    choice(refs.chips.gpu, 'gpu', 'GPU', CHIPS.map(({ value }) => ({ value, label: chipLabel(value, refs.data) })), INITIAL_STATE.gpu, 'gpu', 'chips'),
    choice(refs.recipe, 'recipe', 'Optimizer recipe', RECIPES, INITIAL_STATE.recipe, 'recipe', 'segmented'),
    choice(refs.stage, 'stage', 'ZeRO stage', STAGES, INITIAL_STATE.stage, 'stage', 'segmented'),
    mountSlider(refs.dp, { id: 'dp', label: 'Data-parallel GPUs', values: TOY_LIMITS.dp, value: INITIAL_STATE.dp, unit: 'GPUs', onInput: (v) => set({ dp: v }) }),
    mountSlider(refs.shape.children[0], { id: 'seq', label: 'Sequence length (GPT-3 only)', values: TOY_LIMITS.seq, value: INITIAL_STATE.seq, unit: 'tokens', format: (v) => v.toLocaleString('en-US'), onInput: (v) => set({ seq: v }) }),
    choice(refs.shape.children[1], 'recompute', 'Saved activations (GPT-3 only)', RECOMPUTE, INITIAL_STATE.recompute, 'recompute', 'segmented'),
  ];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = { ...buildDom(host, data, ctx), data };
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
