// "Grade a group" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { INITIAL_STATE } from './format.js';
import { GROUP_SIZE } from './numbers.js';
import { toyView, epsOptions, K_PRESETS, STAND_IN } from './toy-view.js';
import { buildTable } from './toy-dom.js';
import { tryThis } from './try-this.js';

const AGG_OPTIONS = [{ value: 'sample', label: 'sample' }, { value: 'token', label: 'token' }];

function tryThisList() {
  const items = tryThis().map(([prompt, insight, rest]) => el('li', {}, [prompt, ' → ', el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function controls(host, data, set) {
  const parts = ['k', 'preset', 'norm', 'agg', 'eps'].map(() => el('div'));
  const [kBox, presetBox, normBox, aggBox, epsBox] = parts;
  host.prepend(kBox, presetBox, normBox, aggBox, epsBox); // controls above the readouts
  const slider = mountSlider(kBox, { id: 'k', label: 'Right answers out of 8 (how hard the prompt is)', min: 0, max: GROUP_SIZE, step: 1, value: INITIAL_STATE.k, format: (k) => `${k} / ${GROUP_SIZE}`, onInput: (k) => set({ k }) });
  // The chips set the slider; a delegated click also covers a chip whose value the choice already holds (the slider moved on).
  const preset = mountChoice(presetBox, { id: 'k-preset', label: 'Presets', options: K_PRESETS.map((k) => ({ value: k, label: String(k) })), value: INITIAL_STATE.k, variant: 'chips' });
  const onPreset = (event) => {
    const chip = event.target.closest('[data-value]');
    if (chip) slider.set(Number(chip.dataset.value));
  };
  presetBox.addEventListener('click', onPreset);
  const norm = mountToggle(normBox, { id: 'norm', label: 'Divide by the group\'s std', value: INITIAL_STATE.norm, onChange: (on) => set({ norm: on }) });
  const agg = mountChoice(aggBox, { id: 'agg', label: 'Loss aggregation', options: AGG_OPTIONS, value: INITIAL_STATE.agg, onChange: (a) => set({ agg: a }) });
  const eps = mountChoice(epsBox, { id: 'eps', label: 'Upper clip bound ε_high (ε_low fixed at 0.2)', options: epsOptions(data), value: INITIAL_STATE.epsHigh, variant: 'chips', onChange: (e) => set({ epsHigh: e }) });
  return {
    presetButtons: () => [...presetBox.querySelectorAll('[data-value]')],
    destroy() { presetBox.removeEventListener('click', onPreset); [slider, preset, norm, agg, eps].forEach((c) => c.destroy()); },
  };
}

function readouts(host) {
  const stats = readoutTable({ name: 'stats', rows: [['mean', 'Mean reward'], ['std', 'Std (population)'], ['total-push', 'Total push Σ|A|']].map(([name, label]) => ({ label, cells: [{ value: '', name }] })) });
  const signal = el('output', { className: 'toy-note' });
  signal.dataset.readout = 'signal';
  const table = el('div', { className: 'toy-rows' });
  const inspector = el('div');
  host.append(stats, el('p', { className: 'toy-note' }, [signal]), table, inspector,
    el('p', { className: 'toy-note', textContent: STAND_IN }), ...tryThisList());
  return { stats, signal, table, inspector };
}

// Repaints the whole toy from one view; keyboard focus stays on the selected chip when it was inside the table.
function paint(view, state, set, controlsApi) {
  const v = toyView(state);
  const { selected } = v;
  const focused = view.table.contains(document.activeElement);
  view.table.replaceChildren(buildTable(v, (row, token) => set({ row, token })));
  if (focused) view.table.querySelector(`[data-row="${selected.row + 1}"] [data-tok="${selected.token + 1}"]`)?.focus();
  Object.entries({ mean: v.stats.mean, std: v.stats.std, 'total-push': v.stats.totalPush }).forEach(([name, text]) => { view.stats.querySelector(`[data-readout="${name}"]`).textContent = text; });
  view.signal.textContent = v.signal;
  view.signal.parentElement.hidden = v.signal === '';
  view.signal.hidden = v.signal === '';
  view.inspector.replaceChildren(readoutTable({ caption: 'Token inspector', name: 'inspector', rows: v.inspector.map(({ name, label, value }) => ({ label, cells: [{ value, name }] })) }));
  controlsApi.presetButtons().forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.value) === state.k)));
}

export function mount(host, ctx) {
  const view = readouts(host);
  let controlsApi = null;
  const set = (patch) => toy.set(patch);
  const toy = createToyState(INITIAL_STATE, (state) => { if (controlsApi) paint(view, state, set, controlsApi); });
  controlsApi = controls(host, ctx?.data, set);
  paint(view, toy.get(), set, controlsApi);
  return () => {
    toy.destroy();
    controlsApi.destroy();
    host.replaceChildren();
  };
}
