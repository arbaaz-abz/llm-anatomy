// "Plan the end of a run" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { mountPresetButtons } from '@shared/ui/preset-buttons.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { INITIAL_STATE, SCHEDULES, DECAY_VALUES, STOP_RANGE, RUN_IDS, percentText } from './format.js';
import { runData } from './facts.js';
import { FALLBACK } from './numbers.js';
import { toyView } from './toy-view.js';
import { tryThis } from './try-this.js';
import { plotView, barView, lrTable, stageTable, tryThisList } from './toy-dom.js';

// Preset buttons for the decay share; each sets the schedule it describes (Nemotron's minus-sqrt, MiniMax's linear).
const presetOptions = (run) => [
  { value: 'nemotron', label: `Nemotron 3 Super ${percentText(run.nemotron.decayPercent)}` },
  { value: 'minimax', label: `MiniMax-M2 ${percentText(run.minimax.decayPercent)}` },
];
const presetSetting = (run) => ({
  nemotron: { schedule: 'wsd-minus-sqrt', decayFrac: run.nemotron.decayPercent },
  minimax: { schedule: 'wsd', decayFrac: run.minimax.decayPercent },
});

function layout(host) {
  const parts = Object.fromEntries(['schedule', 'preset', 'decay', 'stop', 'run'].map((k) => [k, el('div')]));
  const decayBox = el('div', { className: 'toy-decay' }, [parts.preset, parts.decay]);
  const wide = new Set(['plot', 'bar']); // 580 px figures scroll inside their own container (spec §5.6)
  const out = Object.fromEntries(['plot', 'lr', 'bar', 'stages'].map((k) => [k, el('div', { className: wide.has(k) ? `toy-${k} scroll-x` : `toy-${k}` })]));
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  host.append(parts.schedule, decayBox, parts.stop, out.plot, out.lr, parts.run, out.bar, out.stages, el('h4', { textContent: 'Check my work' }), check);
  return { parts, decayBox, out, check };
}

function paint(ui, state, data) {
  const view = toyView(state, data);
  ui.decayBox.hidden = state.schedule === 'cosine';
  ui.out.plot.replaceChildren(plotView(state, { chosen: view.chosen.name, other: view.other.name }));
  ui.out.lr.replaceChildren(lrTable(view));
  ui.out.bar.replaceChildren(barView(state, data));
  ui.out.stages.replaceChildren(stageTable(view));
  ui.check.textContent = view.check;
}

function controls(ui, data, set) {
  const run = runData(data);
  const settings = presetSetting(run);
  let applying = false; // a preset button sets the other controls; that must not count as a manual change
  const manual = (patch) => {
    if (!applying) set({ ...patch, preset: 'custom' });
  };
  const schedule = mountChoice(ui.parts.schedule, { id: 'schedule', label: 'Learning-rate schedule', value: INITIAL_STATE.schedule, variant: 'chips', options: SCHEDULES, onChange: (s) => manual({ schedule: s }) });
  const decay = mountSlider(ui.parts.decay, { id: 'decayFrac', label: 'Share of the run spent decaying (WSD)', values: DECAY_VALUES, value: INITIAL_STATE.decayFrac, format: percentText, onInput: (v) => manual({ decayFrac: v }) });
  const preset = mountPresetButtons(ui.parts.preset, { id: 'decayPreset', label: 'Presets', options: presetOptions(run), onPick: (key) => {
    applying = true;
    schedule.set(settings[key].schedule);
    decay.set(settings[key].decayFrac);
    applying = false;
    set({ ...settings[key], preset: key });
  } });
  const stop = mountSlider(ui.parts.stop, { id: 'stopAt', label: 'Where you decide to stop or branch', ...STOP_RANGE, value: INITIAL_STATE.stopAt, format: percentText, onInput: (v) => set({ stopAt: v }) });
  const runs = mountChoice(ui.parts.run, { id: 'run', label: 'Context stages', value: INITIAL_STATE.run, variant: 'chips', options: RUN_IDS.map((id) => ({ value: id, label: FALLBACK.runs[id].name })), onChange: (id) => set({ run: id }) });
  return () => [schedule, decay, preset, stop, runs].forEach((c) => c.destroy());
}

export function mount(host, ctx) {
  const data = ctx?.data ?? null;
  const ui = layout(host);
  const toy = createToyState(INITIAL_STATE, (state) => paint(ui, state, data));
  const unmountControls = controls(ui, data, (patch) => toy.set(patch));
  host.append(...tryThisList(tryThis(data)));
  return () => {
    toy.destroy();
    unmountControls();
    host.replaceChildren();
  };
}
