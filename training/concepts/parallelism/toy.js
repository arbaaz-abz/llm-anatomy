// "Schedule a pipeline, then count the GPUs" (storyboard §6): one state object, one paint(state); destroy tears it down.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { mountPresetButtons } from '@shared/ui/preset-buttons.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { DEGREE_STOPS, MICRO_CHIPS, MICRO_RANGE, SCHEDULES, SCHEDULE_LABELS, STAGE_STOPS, ZERO_STAGES } from './format.js';
import { LLAMA_PRESETS } from './numbers.js';
import { INITIAL_STATE, PRESET_CHIPS, SCHEDULE_NOTE, STATE_NOTE, toyView, tryThis } from './toy-view.js';

const LANES_W = 568;
const LANES_PAD = 4;
const DEGREE_LABELS = Object.freeze({ tp: 'Tensor degree', cp: 'Context degree', pp: 'Pipeline degree', dp: 'Data degree' });

function tryThisList() {
  const items = tryThis().map((t) => el('li', {}, [t.prompt, ' → ', el('strong', { textContent: `Insight: ${t.insight}` }), t.rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function pipelinePanel(host, set) {
  const parts = Array.from({ length: 4 }, () => el('div'));
  const chipsHost = el('div');
  host.append(el('h4', { textContent: 'A. Schedule a pipeline' }), parts[0], parts[1], parts[2], chipsHost);
  const schedule = mountChoice(parts[0], { id: 'schedule', label: 'Schedule', value: INITIAL_STATE.schedule, onChange: (v) => set({ schedule: v }),
    options: SCHEDULES.map((s) => ({ value: s, label: SCHEDULE_LABELS[s] })) });
  const stages = mountChoice(parts[1], { id: 'stages', label: 'Pipeline stages p', value: INITIAL_STATE.stages, onChange: (v) => set({ stages: v }),
    options: STAGE_STOPS.map((s) => ({ value: s, label: String(s) })) });
  const micro = mountSlider(parts[2], { id: 'micro', label: 'Micro-batches m', ...MICRO_RANGE, step: 1, value: INITIAL_STATE.micro, onInput: (v) => set({ micro: v }) });
  const chips = mountPresetButtons(chipsHost, { id: 'micro-chips', label: 'Micro-batch presets', options: MICRO_CHIPS.map((m) => ({ value: m, label: String(m) })), onPick: (m) => micro.set(m) });
  return { micro, chips, destroy: () => [schedule, stages, micro, chips].forEach((c) => c.destroy()) };
}

function countPanel(host, set) {
  const sliders = Object.fromEntries(Object.keys(DEGREE_LABELS).map((key) => [key, el('div')]));
  const presetHost = el('div');
  const zeroHost = el('div');
  host.append(el('h4', { textContent: 'B. Count the GPUs' }), presetHost, ...Object.values(sliders), zeroHost);
  let syncing = false;
  const preset = mountPresetButtons(presetHost, { id: 'preset', label: 'Llama 3.1 405B', options: PRESET_CHIPS, onPick: (key) => set({ ...LLAMA_PRESETS[key] }) });
  const degrees = Object.entries(DEGREE_LABELS).map(([key, text]) => mountSlider(sliders[key], {
    id: key, label: text, values: DEGREE_STOPS[key], value: INITIAL_STATE[key], onInput: (v) => { if (!syncing) set({ [key]: v }); },
  }));
  const zero = mountChoice(zeroHost, { id: 'zero', label: 'ZeRO stage on the data-parallel group', value: INITIAL_STATE.zero, onChange: (v) => set({ zero: v }),
    options: ZERO_STAGES.map((s) => ({ value: s, label: String(s) })) });
  const keys = Object.keys(DEGREE_LABELS);
  const show = (state) => {
    syncing = true;
    keys.forEach((k, i) => { if (degrees[i].value !== state[k]) degrees[i].set(state[k]); });
    syncing = false;
  };
  return { show, degrees, destroy: () => [preset, zero, ...degrees].forEach((c) => c.destroy()) };
}

function readouts() {
  const a = readoutTable({ caption: 'Pipeline', name: 'pipeline', rows: [
    { label: 'Bubble', sub: '(p − 1) / (m + p − 1), equal to the idle cells', cells: [{ value: '', name: 'bubble' }] },
    { label: 'Peak micro-batches in flight, stage 1', sub: 'activation memory ∝ this', cells: [{ value: '', name: 'peak' }] },
    { label: 'Step length', sub: 'time units', cells: [{ value: '', name: 'step-length' }] },
  ] });
  const b = readoutTable({ caption: 'GPUs and state', name: 'gpus-state', rows: [
    { label: 'GPU count', sub: 'tensor × context × pipeline × data', cells: [{ value: '', name: 'gpus' }] },
    { label: 'State per GPU, 405B with Adam', sub: 'weights, gradients and optimizer states', cells: [{ value: '', name: 'state-per-gpu' }] },
  ] });
  const lanes = el('div', { className: 'toy-lanes' });
  lanes.dataset.readout = 'lanes';
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  return { a, b, lanes, check };
}

function paintLanes(host, view) {
  const lanes = view.lanes;
  const layout = G.laneTimelineLayout({ w: LANES_W, lanes, labels: view.labelled });
  const w = LANES_W + 2 * LANES_PAD;
  const h = layout.height + 2 * LANES_PAD;
  const svg = G.svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': `${lanes.length}-stage pipeline schedule, ${view.columns} time units` });
  svg.style.maxWidth = '100%';
  svg.style.height = 'auto';
  G.hatchFill(svg);
  G.laneTimeline(svg, { x: LANES_PAD, y: LANES_PAD, w: LANES_W, lanes, labels: view.labelled, label: 'pipeline schedule' });
  host.replaceChildren(svg);
}

function paint(view, panels, state) {
  const v = toyView(state);
  const out = (name) => view.root.querySelector(`[data-readout="${name}"]`);
  paintLanes(view.lanes, v);
  out('bubble').textContent = v.bubble;
  out('peak').textContent = v.peak;
  out('step-length').textContent = v.stepLength;
  out('gpus').textContent = v.gpus;
  out('state-per-gpu').textContent = v.statePerGpu;
  view.check.textContent = v.checkWork;
  panels.count.show(state);
}

export function mount(host) {
  const read = readouts();
  const note = (text, cls = 'toy-note') => el('p', { className: cls, textContent: text });
  const pipeHost = el('div', { className: 'toy-panel' });
  const countHost = el('div', { className: 'toy-panel' });
  const view = { root: host, lanes: read.lanes, check: read.check };
  let toy = null;
  const set = (patch) => toy?.set(patch);
  host.append(note(SCHEDULE_NOTE, 'toy-intro'), pipeHost, read.lanes, read.a, countHost, read.b, note(STATE_NOTE),
    el('h4', { textContent: 'Check my work' }), read.check, ...tryThisList());
  const panels = { pipe: pipelinePanel(pipeHost, set), count: countPanel(countHost, set) };
  toy = createToyState(INITIAL_STATE, (state) => paint(view, panels, state));
  return () => { toy.destroy(); panels.pipe.destroy(); panels.count.destroy(); host.replaceChildren(); };
}
