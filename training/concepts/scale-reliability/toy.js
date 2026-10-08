// scale-reliability toy, "Plan a training run": mount(el, ctx) → destroy. One state object, one render; every string it
// prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { formatCount, formatDuration } from '@math/core.js';
import { STOPS, MFU_RANGE, PRICE_RANGE } from './numbers.js';
import { int } from './format.js';
import { CHIPS, PRESETS, PRESET_FIELDS, INITIAL_STATE, STAND_IN_NOTE, PRICE_NOTE, toyView, tryThis, peakFor } from './toy-view.js';

const BAR_W = 300;
const PERCENT = 100;
const SECONDS_PER_MINUTE = 60;

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

function slider(host, { id, label, stops, value, format, unit, onInput }) {
  return mountSlider(host, { id, label, values: stops, value, format, unit, onInput });
}

function tryThisList(data) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data) {
  const refs = {
    presets: el('div'), chips: el('div'), sliderHosts: Array.from({ length: 9 }, () => el('div')), auto: el('div'),
    note: output('preset-note', { className: 'toy-note' }), invalid: output('invalid', { className: 'toy-note' }), record: output('record'),
    table: el('div'), bar: G.svgEl('svg', { role: 'group', 'aria-label': 'GPU-hours of the planned run' }),
  };
  const [params, tokens, gpus, mfu, mtbf, save, restart, interval, price] = refs.sliderHosts;
  refs.intervalHost = el('div', {}, [interval]); // the slider's own box sets its display, so the wrapper carries `hidden`
  host.append(
    el('p', { className: 'toy-note', textContent: STAND_IN_NOTE }), refs.presets, refs.chips,
    el('div', { className: 'toy-controls' }, [params, tokens, gpus, mfu, mtbf, save, restart, refs.auto, refs.intervalHost, price]),
    el('p', { className: 'toy-note', textContent: PRICE_NOTE }), el('p', { className: 'toy-note' }, [refs.note]),
    el('div', { className: 'scroll-x' }, [refs.table]), el('p', { className: 'toy-note' }, [refs.invalid]), el('div', { className: 'scroll-x' }, [refs.bar]),
    el('h4', { textContent: 'Check against the record' }), el('p', {}, [refs.record]), ...tryThisList(data),
  );
  return refs;
}

function tableOf(view) {
  const row = (label, name, value, sub) => ({ label, cells: [{ value, sub, name }] });
  return readoutTable({
    head: ['Quantity', 'Value'], name: 'plan',
    rows: [
      row('Training FLOPs', 'flops', view.flops), row('Useful GPU-hours, at the MFU while training', 'useful-hours', view.usefulHours),
      row('Cluster MTBF', 'mtbf', view.mtbf), row('Checkpoint interval', 'interval', view.interval, view.intervalSub),
      row('Lost to checkpointing and failures', 'loss', view.loss, view.lossSplit),
      row('GPU-hours', 'gpu-hours', view.gpuHours), row('Days', 'days', view.days), row('Cost', 'cost', view.cost),
      row('Run-average MFU', 'run-average-mfu', view.runAverageMfu),
    ],
  });
}

function paint(refs, view) {
  refs.table.replaceChildren(tableOf(view));
  refs.invalid.textContent = view.invalid;
  refs.invalid.hidden = view.invalid === '';
  refs.record.textContent = view.record;
  refs.note.textContent = view.presetNote;
  refs.note.hidden = view.presetNote === '';
  refs.bar.replaceChildren();
  refs.bar.hidden = !view.bar;
  if (!view.bar) return;
  G.shareBar(refs.bar, { x: 4, y: 6, w: BAR_W, parts: view.bar, label: 'GPU-hours of the planned run', tail: 'none' });
  G.fitViewBox(refs.bar, 8);
}

function mountControls(refs, data, state) {
  let syncing = false;
  const guarded = (fn) => (v) => { if (!syncing) fn(v); };
  const set = (patch) => state.toy?.set(patch);
  const [params, tokens, gpus, mfu, mtbf, save, restart, interval, price] = refs.sliderHosts;
  const sliders = {
    params: slider(params, { id: 'params', label: 'Active parameters N', stops: STOPS.params, value: INITIAL_STATE.params, format: (v) => formatCount(v), onInput: guarded((v) => set({ params: v })) }),
    tokens: slider(tokens, { id: 'tokens', label: 'Tokens D', stops: STOPS.tokens, value: INITIAL_STATE.tokens, format: (v) => formatCount(v), onInput: guarded((v) => set({ tokens: v })) }),
    gpus: slider(gpus, { id: 'gpus', label: 'GPUs', stops: STOPS.gpus, value: INITIAL_STATE.gpus, format: int, onInput: guarded((v) => set({ gpus: v })) }),
    mfu: mountSlider(mfu, { id: 'mfu', label: 'MFU while training', min: MFU_RANGE.min, max: MFU_RANGE.max, step: MFU_RANGE.step, value: INITIAL_STATE.mfu * PERCENT, format: (v) => `${v.toFixed(1)}%`, onInput: guarded((v) => set({ mfu: v / PERCENT })) }),
    mtbf: slider(mtbf, { id: 'mtbf', label: 'MTBF per GPU', stops: STOPS.perGpuMtbfH, value: INITIAL_STATE.perGpuMtbfH, format: int, unit: 'h', onInput: (v) => set({ perGpuMtbfH: v }) }),
    save: slider(save, { id: 'save', label: 'Checkpoint save time', stops: STOPS.saveS, value: INITIAL_STATE.saveS, format: (v) => formatDuration(v), onInput: (v) => set({ saveS: v }) }),
    restart: slider(restart, { id: 'restart', label: 'Restart time', stops: STOPS.restartMin, value: INITIAL_STATE.restartMin, format: (v) => formatDuration(v * SECONDS_PER_MINUTE), onInput: (v) => set({ restartMin: v }) }),
    interval: slider(interval, { id: 'interval', label: 'Checkpoint interval', stops: STOPS.intervalMin, value: INITIAL_STATE.intervalMin, format: (v) => formatDuration(v * SECONDS_PER_MINUTE), onInput: (v) => set({ intervalMin: v }) }),
    price: mountSlider(price, { id: 'price', label: 'Price per GPU-hour', min: PRICE_RANGE.min, max: PRICE_RANGE.max, step: PRICE_RANGE.step, value: INITIAL_STATE.price, format: (v) => `$${v}`, onInput: (v) => set({ price: v }) }),
  };
  const auto = mountToggle(refs.auto, { id: 'interval-auto', label: 'Best interval', value: INITIAL_STATE.auto, onChange: (on) => set({ auto: on }) });
  const peak = mountChoice(refs.chips, {
    id: 'peak', label: 'Chip and precision', variant: 'chips', value: INITIAL_STATE.chip,
    options: CHIPS.map((c) => ({ value: c.value, label: `${c.label} ${int(peakFor(c.value, data))}` })), onChange: (v) => set({ chip: v }),
  });
  const preset = mountChoice(refs.presets, {
    id: 'preset', label: 'Run', variant: 'chips', value: INITIAL_STATE.preset, options: Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label })),
    onChange: (value) => {
      const chosen = PRESETS[value];
      syncing = true;
      sliders.params.set(chosen.params); sliders.tokens.set(chosen.tokens); sliders.gpus.set(chosen.gpus); sliders.mfu.set(chosen.mfu * PERCENT); peak.set(chosen.chip);
      syncing = false;
      set({ preset: value, ...Object.fromEntries(PRESET_FIELDS.map((k) => [k, chosen[k]])) });
    },
  });
  return { all: [...Object.values(sliders), auto, peak, preset] };
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data);
  const state = { toy: null };
  const controls = mountControls(refs, data, state);
  state.toy = createToyState(INITIAL_STATE, (s) => {
    paint(refs, toyView(s, data));
    refs.intervalHost.hidden = s.auto;
  });
  return () => {
    state.toy.destroy();
    controls.all.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
