// paged-attention toy, "Slide the block size and watch the waste" (storyboard §6): mount(el, ctx) → destroy.
// One state object, one paint; every string it prints comes from toyView (toy-view.js), so the page shows what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { appendRich } from '@shared/lesson-page.js';
import { INITIAL_STATE, SLIDER_VALUES, STEP_RANGE } from './format.js';
import {
  MODEL_CHIPS, FOLLOW_CHIPS, SHARED_LABEL, KERNEL_NOTE, COMPARE_NOTE, toyView, tryThis,
} from './toy-view.js';
import { paintLanes, TOY_SVG } from './toy-lanes.js';
import { LAST_STEP } from './numbers.js';

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};
const note = (children, name) => {
  const p = el('p', { className: 'toy-note' }, children);
  if (name) p.dataset.readout = name;
  return p;
};
const cell = (c, name) => ({ value: c.value, sub: c.sub, name });

function buildDom(host, ctx) {
  const refs = {
    sliders: { blockSize: el('div'), step: el('div') }, toggle: el('div'), follow: el('div'), model: el('div'),
    svg: G.svgEl('svg', { width: TOY_SVG.w, height: TOY_SVG.h, viewBox: `0 0 ${TOY_SVG.w} ${TOY_SVG.h}`, role: 'img', 'aria-label': 'Both lanes of the KV pool' }),
    lanes: el('div', { className: 'scroll-x' }), requests: el('div', { className: 'scroll-x' }), table: el('div', { className: 'scroll-x' }), saved: el('div', { className: 'scroll-x' }), scale: el('div', { className: 'scroll-x' }),
  };
  const items = tryThis().map(({ text, insight, rest }) => el('li', {}, [`${text} → `, el('strong', { textContent: `Insight: ${insight}` }), appendRich(el('span'), rest, ctx)]));
  host.append(
    el('div', { className: 'toy-controls' }, [refs.sliders.blockSize, refs.sliders.step]), refs.toggle,
    el('div', { className: 'scroll-x' }, [refs.svg]),
    refs.lanes, refs.requests,
    el('h4', { textContent: 'Follow one request' }), refs.follow, refs.table,
    refs.saved, note([KERNEL_NOTE]),
    el('h4', { textContent: 'Scale it up' }), refs.model, refs.scale,
    el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items),
  );
  return refs;
}

function lanesTables(view) {
  const lane = (label, key, sub) => ({ label, sub, view: view[key], key });
  const lanes = [lane('Before', 'before', 'one strip each'), lane('After', 'after', 'blocks on demand')];
  const memory = readoutTable({
    head: ['Lane', 'Useful', 'Reserved but empty (wasted)', 'Free'],
    rows: lanes.map((l) => ({ label: l.label, sub: l.sub, cells: [cell(l.view.useful, `${l.key}-useful`), cell(l.view.wasted, `${l.key}-wasted`), cell(l.view.free, `${l.key}-free`)] })),
  });
  const schedule = readoutTable({
    head: ['Lane', 'In use', 'D starts', 'D finishes', `Average wasted, steps 0 to ${LAST_STEP}`],
    rows: lanes.map((l) => ({ label: l.label, cells: [{ value: l.view.blocks, name: `${l.key}-blocks` }, { value: l.view.dStart, name: `${l.key}-d-start` }, { value: l.view.dFinish, name: `${l.key}-d-finish` }, { value: l.view.average, name: `${l.key}-average` }] })),
  });
  return [memory, schedule];
}

function requestsTable(view) {
  return readoutTable({
    caption: `Each request at step ${view.step}`,
    head: ['Request', 'Before: status', 'tokens', 'wasted slots', 'After: status', 'tokens', 'blocks', 'wasted slots'],
    rows: view.requests.map((r) => ({
      label: r.id,
      cells: [
        { value: r.before.status, name: `before-${r.id}-status` }, { value: r.before.tokens, name: `before-${r.id}-tokens` }, { value: r.before.wasted, name: `before-${r.id}-wasted` },
        { value: r.after.status, name: `after-${r.id}-status` }, { value: r.after.tokens, name: `after-${r.id}-tokens` }, { value: r.after.blocks, name: `after-${r.id}-blocks` }, { value: r.after.wasted, name: `after-${r.id}-wasted` },
      ],
    })),
  });
}

function tableNodes(view) {
  const { follow } = view;
  if (follow.empty) return [note([follow.empty], 'follow-empty')];
  return [readoutTable({
    caption: `Block table of ${follow.id}`,
    head: ['Logical block', 'Physical block'],
    rows: follow.rows.map((r) => ({ label: `logical ${r.logical}`, cells: [{ value: r.physical, name: `table-${r.logical}` }] })),
  })];
}

function scaleNodes(view) {
  const s = view.scale;
  if (!s) return [note(['Pick a model to see what one block and one reservation weigh in bytes.'], 'scale-empty')];
  return [
    readoutTable({
      caption: `${s.name}, per request`,
      head: ['Quantity', 'Size'],
      rows: [
        { label: 'Per token', cells: [cell(s.perToken, 'scale-per-token')] },
        { label: 'One block', cells: [cell(s.block, 'scale-block')] },
        { label: 'One request\'s whole context, reserved up front', cells: [cell(s.reserve, 'scale-reserve')] },
        { label: 'Share of one GPU', cells: [cell(s.share, 'scale-share')] },
      ],
    }),
    note([COMPARE_NOTE]),
  ];
}

function paint(refs, view, state) {
  paintLanes(refs.svg, { sims: view.sims, blockSize: state.blockSize, follow: state.follow });
  refs.lanes.replaceChildren(...lanesTables(view));
  refs.requests.replaceChildren(requestsTable(view));
  refs.table.replaceChildren(...tableNodes(view));
  refs.saved.replaceChildren(readoutTable({ rows: [{ label: 'Blocks saved by the shared prompt', cells: [cell(view.saved, 'blocks-saved')] }] }));
  refs.scale.replaceChildren(...scaleNodes(view));
}

function mountControls(refs, set) {
  const controls = [
    mountSlider(refs.sliders.blockSize, { id: 'blockSize', label: 'Block size', values: SLIDER_VALUES.blockSize, unit: 'tokens', value: INITIAL_STATE.blockSize, onInput: (blockSize) => set({ blockSize }) }),
    mountSlider(refs.sliders.step, { id: 'step', label: 'Time', min: STEP_RANGE[0], max: STEP_RANGE[1], step: 1, value: INITIAL_STATE.step, format: (v) => `step ${v}`, onInput: (step) => set({ step }) }),
    mountToggle(refs.toggle, { id: 'sharedPrefix', label: SHARED_LABEL, value: INITIAL_STATE.sharedPrefix, onChange: (sharedPrefix) => set({ sharedPrefix }) }),
    mountChoice(refs.follow, { id: 'follow', label: 'Request to follow', variant: 'chips', value: INITIAL_STATE.follow, options: FOLLOW_CHIPS, onChange: (follow) => set({ follow }) }),
    mountChoice(refs.model, { id: 'model', label: 'Scale it up', variant: 'chips', value: INITIAL_STATE.model, options: MODEL_CHIPS, onChange: (model) => set({ model }) }),
  ];
  return controls;
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, ctx);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
