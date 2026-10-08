// "Shape the draw" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { INITIAL_STATE } from './format.js';
import { view, tryThis, DRAWS, SEED_RANGE } from './toy-view.js';

const TEMPERATURES = [0.25, 0.5, 0.75, 1, 1.5, 2];
const TOP_KS = [0, 1, 2, 3, 5, 8]; // 0 = off
const TOP_PS = [1, 0.95, 0.9, 0.75, 0.7, 0.5]; // 1 = off
const BARS = Object.freeze({ x: 12, y: 24, h: 120, w: 5 * 52, labels: ['on', '.', 'and', 'the', '12 others'] });

const kText = (k) => (k === 0 ? 'off' : `${k} ${k === 1 ? 'token' : 'tokens'}`);
const pText = (p) => (p === 1 ? 'off' : `${p} of the probability`);

// The control boxes first (paint needs them to hide the temperature slider), the controls themselves once the state exists.
function controlBoxes(host) {
  const box = () => el('div', {});
  const boxes = { temperature: box(), greedy: box(), topK: box(), topP: box(), seed: box() };
  const temperatureWrap = el('div', {}, [boxes.temperature]); // the wrapper is what hides: .slider sets its own display
  const newSeed = el('button', { type: 'button', id: 'new-seed', className: 'choice-option', textContent: 'new seed' });
  const greedyNote = el('p', { className: 'toy-note', hidden: true, textContent: 'Temperature 0: greedy, always the top token.' });
  const order = el('p', { className: 'toy-note', textContent: 'Order of operations: temperature → top-k → top-p → rescale → draw (a common order; libraries differ).' });
  host.append(temperatureWrap, greedyNote, boxes.greedy, boxes.topK, boxes.topP, boxes.seed, newSeed, order);
  return { boxes, newSeed, temperatureWrap, greedyNote };
}

function mountControls({ boxes, newSeed }, set) {
  const seed = mountSlider(boxes.seed, { id: 'seed', label: 'Seed', ...SEED_RANGE, step: 1, value: INITIAL_STATE.seed, onInput: (s) => set({ seed: s }) });
  const parts = [
    mountSlider(boxes.temperature, { id: 'temperature', label: 'Temperature', values: TEMPERATURES, value: INITIAL_STATE.temperature, onInput: (t) => set({ temperature: t }) }),
    mountToggle(boxes.greedy, { id: 'greedy', label: 'Greedy (temperature → 0)', value: INITIAL_STATE.greedy, onChange: (on) => set({ greedy: on }) }),
    mountSlider(boxes.topK, { id: 'topK', label: 'Top-k', values: TOP_KS, value: INITIAL_STATE.topK, format: kText, onInput: (k) => set({ topK: k }) }),
    mountSlider(boxes.topP, { id: 'topP', label: 'Top-p', values: TOP_PS, value: INITIAL_STATE.topP, format: pText, onInput: (p) => set({ topP: p }) }),
    seed,
  ];
  const onNewSeed = () => seed.set(seed.value >= SEED_RANGE.max ? SEED_RANGE.min : seed.value + 1);
  newSeed.addEventListener('click', onNewSeed);
  return () => { newSeed.removeEventListener('click', onNewSeed); parts.forEach((c) => c.destroy()); };
}

function readouts(host) {
  const tables = el('div', { className: 'toy-tables' });
  const drawsTitle = el('p', { className: 'choice-label' });
  const bars = el('div', {});
  const firstEight = el('output', { className: 'readout' });
  firstEight.dataset.readout = 'first-eight';
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  const tries = tryThis().map(({ prompt, insight }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]));
  host.append(tables, drawsTitle, bars, el('p', { className: 'toy-note', textContent: 'first eight draws:' }), firstEight,
    el('h4', { textContent: 'Check my work' }), check, el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, tries));
  return { tables, drawsTitle, bars, firstEight, check };
}

const distributionTable = (rows) => readoutTable({
  head: ['Token', 'After temperature', 'After filters, rescaled'],
  name: 'distribution',
  rows: rows.map((r) => ({
    label: r.label,
    sub: r.labelSub ?? undefined,
    cells: [{ value: r.pre, sub: r.preSub ?? undefined, name: `pre-${r.name}` }, { value: r.post, sub: r.postSub ?? undefined, name: `p-${r.name}` }],
  })),
});

const keptTable = (v) => readoutTable({
  name: 'kept-table',
  rows: [
    { label: 'Tokens that can still be drawn', cells: [{ value: v.kept, name: 'kept' }] },
    { label: 'Probability kept before rescaling', cells: [{ value: v.mass, name: 'mass' }] },
  ],
});

function drawBars(counts) {
  const svg = G.svgEl('svg', { width: BARS.w + 2 * BARS.x, height: BARS.y + BARS.h + 30, viewBox: `0 0 ${BARS.w + 2 * BARS.x} ${BARS.y + BARS.h + 30}`, role: 'img', 'aria-label': `${DRAWS} draws per category` });
  G.bars(svg, { x: BARS.x, y: BARS.y, w: BARS.w, h: BARS.h, values: counts, labels: BARS.labels, max: DRAWS, label: `${DRAWS} draws` });
  return svg;
}

function paint(v, refs, boxes) {
  refs.tables.replaceChildren(distributionTable(v.rows), keptTable(v));
  refs.drawsTitle.textContent = v.drawsTitle;
  refs.bars.replaceChildren(drawBars(v.counts));
  refs.firstEight.textContent = v.firstEight;
  refs.check.textContent = v.checkWork;
  boxes.temperatureWrap.hidden = v.greedy;
  boxes.greedyNote.hidden = !v.greedy;
}

export function mount(host) {
  const controlHost = el('div', { className: 'toy-controls' });
  const readoutHost = el('div', { className: 'toy-readouts' });
  host.append(controlHost, readoutHost);
  const refs = readouts(readoutHost);
  const boxes = controlBoxes(controlHost);
  const toy = createToyState(INITIAL_STATE, (state) => paint(view(state), refs, boxes));
  const unmountControls = mountControls(boxes, (patch) => toy.set(patch));
  return () => {
    toy.destroy();
    unmountControls();
    host.replaceChildren();
  };
}
