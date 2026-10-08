// "What does a picture cost?" (storyboard §6): one state object, one paint(state); destroy tears everything down.
// Every string it prints comes from toyView (toy-view.js); every count from math/vision.js.
import { el } from '@shared/ui/dom.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { syncedChoice, syncedSlider } from './toy-dom.js';
import {
  modelFacts, initialState, applyPreset, applySetting, sideValues, int, CONTEXTS, SECONDS, FPS,
} from './format.js';
import { PRESET_CHIPS, MERGE_OPTIONS, toyView, tryThis } from './toy-view.js';

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

function buildDom(host) {
  const spec = output('spec');
  const slots = Object.fromEntries(['preset', 'width', 'height', 'patch', 'merge', 'media', 'seconds', 'fps', 'context'].map((name) => [name, el('div')]));
  const videoBox = el('div', { className: 'toy-controls' }, [slots.seconds, slots.fps]);
  const sizeBox = el('div', { className: 'toy-controls' }, [slots.width, slots.height, slots.patch, slots.merge]);
  const table = el('div', { className: 'toy-tables' });
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  host.append(
    slots.preset, el('p', { className: 'toy-note' }, [spec]),
    sizeBox, slots.media, videoBox, slots.context,
    table, el('p', { className: 'toy-note', textContent: 'Counts are exact for the patch grid and the merge; no pooling over time is applied to video.' }),
    el('h4', { textContent: 'Check my work' }), check,
  );
  return { spec, slots, videoBox, sizeBox, table, check };
}

function tableFor(view) {
  const rows = [
    { label: 'Patch grid', sub: 'columns × rows = patches', cells: [{ value: view.grid, name: 'grid' }] },
    { label: view.perFrameLabel, cells: [{ value: view.perFrame, name: 'tokens-per-frame' }] },
    ...(view.showVideo ? [{ label: 'Frames', cells: [{ value: view.frames, name: 'frames' }] }] : []),
    { label: 'Total tokens', cells: [{ value: view.tokens, sub: view.tokensApprox, name: 'tokens' }] },
    { label: 'Share of the context window', cells: [{ value: view.share, sub: view.shareNote, name: 'share' }] },
    { label: view.fitLabel, cells: [{ value: view.fit, name: 'fit' }] },
  ];
  return readoutTable({ head: ['Count', 'Value'], rows });
}

function paint(refs, controls, state, facts) {
  const view = toyView(state, facts);
  // The toy image has its own fixed size, patch and merge: those controls are hidden and left as they were.
  const hiddenWithToy = new Set(['width', 'height', 'patch', 'merge']);
  Object.entries(controls).forEach(([name, c]) => { if (view.showSize || !hiddenWithToy.has(name)) c.sync(state); });
  refs.spec.textContent = view.spec;
  refs.table.replaceChildren(tableFor(view));
  refs.check.textContent = view.checkWork;
  refs.videoBox.hidden = !view.showVideo;
  refs.sizeBox.hidden = !view.showSize;
}

function mountControls(refs, update, facts) {
  const change = (patch) => update((state) => applySetting(state, patch, facts));
  const sizeValues = (state) => sideValues({ patch: state.patch, merge: state.merge, maxSide: facts.maxSide });
  const slider = (name, label) => syncedSlider(refs.slots[name], { id: name, label, unit: 'px', valuesOf: sizeValues, read: (s) => s[name], format: int, onInput: (v) => change({ [name]: v }) });
  const choice = (name, label, options, variant, onChange) => syncedChoice(refs.slots[name], { id: name, label, options, variant, read: (s) => s[name], onChange });
  const plain = (name, label, options) => choice(name, label, options, 'segmented', (v) => change({ [name]: v }));
  return {
    preset: choice('preset', 'Model settings', PRESET_CHIPS, 'chips', (v) => update((state) => applyPreset(state, v, facts))),
    width: slider('width', 'Width'),
    height: slider('height', 'Height'),
    patch: plain('patch', 'Patch size', [{ value: 14, label: '14 px' }, { value: 16, label: '16 px' }]),
    merge: plain('merge', 'Merge neighbors', MERGE_OPTIONS),
    media: choice('media', 'Input', [{ value: 'image', label: 'image' }, { value: 'video', label: 'video' }], 'segmented', (v) => update((state) => ({ ...state, media: v }))),
    seconds: syncedSlider(refs.slots.seconds, { id: 'seconds', label: 'Video length', unit: 's', valuesOf: () => SECONDS, read: (s) => s.seconds, format: int, onInput: (v) => update((state) => ({ ...state, seconds: v })) }),
    fps: choice('fps', 'Frames per second', FPS.map((v) => ({ value: v, label: String(v) })), 'segmented', (v) => update((state) => ({ ...state, fps: v }))),
    context: choice('context', 'Context window', CONTEXTS.map((v) => ({ value: v, label: `${int(v)} tokens` })), 'segmented', (v) => update((state) => ({ ...state, context: v }))),
  };
}

function tryThisList(facts) {
  const items = tryThis(facts).map(({ prompt, insight }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

export function mount(host, ctx) {
  const facts = modelFacts(ctx?.data);
  const refs = buildDom(host);
  let toy = null;
  const update = (next) => toy?.set(next);
  const controls = mountControls(refs, update, facts);
  toy = createToyState(initialState(facts), (state) => paint(refs, controls, state, facts));
  host.append(...tryThisList(facts));
  return () => {
    toy.destroy();
    Object.values(controls).forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
