// serving-calculator toy, "Size a deployment": mount(host, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check. A model change resets the model-dependent
// controls to that model's defaults (Review Focus 5); the weight format and context chips are rebuilt per model.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { formatInt, formatBytes } from '@math/core.js';
import {
  GPUS, GPU_COUNTS, WEIGHT_FORMATS, NO_FP4, V4_ID, LLAMA_ID, MODEL_LABELS, gpuPreset, modelPreset, hbmText, supportsWeights, scenarioPresets,
} from './presets.js';
import { INITIAL_STATE, USER_STOPS, USER_STOP_INDEX, userStopLabel, modelDefaults, usd } from './format.js';
import { toyView, FLOOR_NOTE } from './toy-view.js';
import { tryThis } from './try-this.js';
import { output, readoutTables, paintFigure } from './toy-dom.js';

const rich = (text, ctx, props = {}) => appendRich(el('p', { className: 'toy-note', ...props }), text, ctx);

const CONTEXT_OPTIONS = Object.freeze({
  [V4_ID]: [{ value: 9216, label: '8K + 1K' }, { value: 139264, label: '128K + 8K' }, { value: 1_000_000, label: '1M = 1,000,000' }],
  [LLAMA_ID]: [{ value: 9216, label: '8K + 1K' }, { value: 131072, label: '128K = 131,072' }],
});
const WEIGHT_OPTIONS = Object.freeze({ [V4_ID]: ['shipped', 'bf16', 'fp8', 'nvfp4'], [LLAMA_ID]: ['bf16', 'fp8', 'nvfp4'] });
const WEIGHTS_NOTE = 'As shipped uses the FP8 math rate (conservative: the experts run FP4); NVFP4 uses the GPU\'s FP4 rate.';

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx) {
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  const names = ['model', 'weights', 'kvEnd', 'kvFormat', 'hw', 'gpus', 'context', 'users', 'target', 'price', 'hit', 'mtp'];
  const refs = Object.fromEntries(names.map((n) => [n, el('div')]));
  refs.weightsNote = rich(WEIGHTS_NOTE, ctx);
  refs.conditions = output('conditions', { className: 'toy-note' });
  refs.tables = el('div', { className: 'toy-readouts' });
  refs.notes = Object.fromEntries(['dense', 'sparse', 'ratio'].map((n) => [n, output(`note-${n}`, { className: 'toy-note' })]));
  refs.record = output('mtp-record', { className: 'toy-note' });
  refs.figure = G.svgEl('svg', { role: 'img', 'aria-label': 'followed GPU memory and decode step' });
  refs.check = check;
  host.append(
    rich(FLOOR_NOTE, ctx),
    el('div', { className: 'toy-controls' }, [refs.model, refs.weights, refs.weightsNote, refs.kvEnd, refs.kvFormat, refs.hw, refs.gpus, refs.notes.dense, refs.context, refs.users, refs.target, refs.price, refs.hit, refs.mtp]),
    el('p', {}, [refs.conditions]),
    refs.tables, refs.record, el('div', { className: 'scroll-x' }, [refs.figure]), ...Object.values(refs.notes).slice(1),
    el('h4', { textContent: 'Check my work' }), check, ...tryThisList(data, ctx),
  );
  return refs;
}

function paint(refs, view) {
  refs.conditions.textContent = view.conditions;
  refs.tables.replaceChildren(...readoutTables(view));
  Object.entries(refs.notes).forEach(([name, node]) => { node.textContent = view.notes[name]; node.hidden = view.notes[name] === ''; });
  refs.record.textContent = view.readouts['mtp-record'];
  refs.record.hidden = !view.showMtp;
  paintFigure(refs.figure, view.figure);
  refs.check.textContent = view.checkWork;
}

// The choice options that depend on the model or the GPU, rebuilt from the data.
const contextOptions = (model) => CONTEXT_OPTIONS[model];
const weightOptions = (model, gpu) => WEIGHT_OPTIONS[model].map((id) => ({
  value: id, label: WEIGHT_FORMATS[id].label, ...(supportsWeights(gpu, id) ? {} : { disabled: true, note: NO_FP4 }),
}));

// The state a control change leads to: a model change resets the model-dependent fields; a GPU change that cannot run the
// weight format moves it to FP8 first (P3-R12), so no state is ever unrunnable.
export function nextState(state, patch, data) {
  let next = { ...state, ...patch };
  if (patch.model !== undefined && patch.model !== state.model) next = { ...next, ...modelDefaults(patch.model) };
  if (patch.hw !== undefined && !supportsWeights(gpuPreset(data, next.hw), next.weights)) next = { ...next, weights: 'fp8' };
  return next;
}

function mountStatic(refs, data, apply) {
  const presets = scenarioPresets(data);
  const gpus = GPUS.map((g) => gpuPreset(data, g.id));
  const v4 = modelPreset(data, V4_ID);
  const pick = (key) => (v) => apply({ [key]: v });
  return {
    model: mountChoice(refs.model, { id: 'model', label: 'Model', variant: 'chips', value: INITIAL_STATE.model, options: Object.entries(MODEL_LABELS).map(([value, label]) => ({ value, label })), onChange: pick('model') }),
    hw: mountChoice(refs.hw, { id: 'hw', label: 'GPU', variant: 'chips', value: INITIAL_STATE.hw, options: gpus.map((g) => ({ value: g.id, label: `${g.label} · ${hbmText(g)}` })), onChange: pick('hw') }),
    kvEnd: mountChoice(refs.kvEnd, { id: 'kvEnd', label: 'V4 KV estimate', value: INITIAL_STATE.kvEnd, options: [{ value: 'low', label: `low (${formatBytes(v4.kvBytes.low)}/token)` }, { value: 'high', label: `high (${formatBytes(v4.kvBytes.high)}/token)` }], onChange: pick('kvEnd') }),
    kvFormat: mountChoice(refs.kvFormat, { id: 'kvFormat', label: 'KV cache (Llama)', value: INITIAL_STATE.kvFormat, options: [{ value: 'bf16', label: 'BF16' }, { value: 'fp8', label: 'FP8' }], onChange: pick('kvFormat') }),
    gpus: mountSlider(refs.gpus, { id: 'gpus', label: 'GPUs per replica', values: GPU_COUNTS, value: INITIAL_STATE.gpus, unit: 'GPUs', format: formatInt, onInput: pick('gpus') }),
    users: mountSlider(refs.users, { id: 'users', label: 'Users per GPU', values: USER_STOPS.map((_, i) => i), value: USER_STOP_INDEX[INITIAL_STATE.users], format: (i) => userStopLabel(USER_STOPS[i]), onInput: (i) => apply({ users: USER_STOPS[i] }) }),
    target: mountChoice(refs.target, { id: 'target', label: 'Target speed per user', variant: 'chips', value: INITIAL_STATE.target, options: presets.targets.map((t) => ({ value: t, label: `${t} tok/s` })), onChange: pick('target') }),
    price: mountChoice(refs.price, { id: 'price', label: '$ per GPU-hour', variant: 'chips', value: INITIAL_STATE.price, options: presets.prices.map((p) => ({ value: p.usd, label: `${usd(p.usd)} · ${p.name}` })), onChange: pick('price') }),
    hit: mountChoice(refs.hit, { id: 'hit', label: 'Prompt cache hit rate', variant: 'chips', value: INITIAL_STATE.hit, options: [{ value: 0, label: '0%' }, { value: presets.hitRate, label: `${Number((presets.hitRate * 100).toFixed(1))}% (DeepSeek 2025)` }], onChange: pick('hit') }),
    mtp: mountToggle(refs.mtp, { id: 'mtp', label: `MTP speculation (k = 1, α = ${presets.mtpAlpha})`, value: INITIAL_STATE.mtp, onChange: pick('mtp') }),
  };
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data, ctx);
  let toy = null;
  let syncing = false;
  const dynamic = { weights: null, context: null, model: null };
  const controls = {};
  const mountDynamic = (state) => {
    dynamic.weights?.destroy();
    dynamic.context?.destroy();
    dynamic.weights = mountChoice(refs.weights, { id: 'weights', label: 'Weight format', variant: 'chips', value: state.weights, options: weightOptions(state.model, gpuPreset(data, state.hw)), onChange: (v) => apply({ weights: v }) });
    dynamic.context = mountChoice(refs.context, { id: 'context', label: 'Tokens per user (in + out)', variant: 'chips', value: state.context, options: contextOptions(state.model), onChange: (v) => apply({ context: v }) });
    dynamic.model = state.model;
  };
  const sync = (state) => {
    syncing = true;
    try {
      if (dynamic.model !== state.model) mountDynamic(state);
      else {
        dynamic.weights.update(weightOptions(state.model, gpuPreset(data, state.hw)));
        if (dynamic.weights.value !== state.weights) dynamic.weights.set(state.weights);
        if (dynamic.context.value !== state.context) dynamic.context.set(state.context);
      }
      ['model', 'hw', 'kvEnd', 'kvFormat', 'target', 'price', 'hit'].forEach((k) => { if (controls[k].value !== state[k]) controls[k].set(state[k]); });
      if (controls.gpus.value !== state.gpus) controls.gpus.set(state.gpus);
      if (controls.users.value !== USER_STOP_INDEX[state.users]) controls.users.set(USER_STOP_INDEX[state.users]);
      const dense = state.model === LLAMA_ID;
      refs.kvEnd.hidden = dense;
      refs.kvFormat.hidden = !dense;
      refs.weightsNote.hidden = dense;
    } finally {
      syncing = false;
    }
  };
  function apply(patch) {
    if (syncing) return;
    const next = nextState(toy.get(), patch, data);
    toy.set(next);
    sync(toy.get());
  }
  mountDynamic(INITIAL_STATE);
  Object.assign(controls, mountStatic(refs, data, apply));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data)));
  sync(toy.get());
  return () => {
    toy.destroy();
    [...Object.values(controls), dynamic.weights, dynamic.context].forEach((c) => c?.destroy());
    host.replaceChildren();
  };
}
