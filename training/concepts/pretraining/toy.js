// pretraining toy, "Grade a sentence": mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { appendRich } from '@shared/lesson-page.js';
import { INITIAL_STATE, P_SLIDER, PRESETS, formatProb, selectPosition, setProb, allUniform, resetProbs } from './format.js';
import { toyView, tryThis } from './toy-view.js';
import { mountStrip } from './toy-dom.js';

const STRIP_LABEL_ID = 'pretraining-pos-label';
const onGrid = (p) => Math.min(Math.max(Math.round(p * 100) / 100, P_SLIDER.min), P_SLIDER.max); // where the slider can sit

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => {
    const li = appendRich(el('li'), prompt, ctx);
    li.append(' → ', el('strong', { textContent: `Insight: ${insight}` }), rest);
    return li;
  });
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

// Action buttons (they set values; they are not modes, so no pressed state): plain buttons styled as chips.
function buttonGroup(caption, specs) {
  const buttons = specs.map(({ text, onClick }) => {
    const b = el('button', { type: 'button', className: 'choice-option', textContent: text });
    b.addEventListener('click', onClick);
    return [b, onClick];
  });
  const node = el('div', { className: 'choice choice--chips', role: 'group' }, [el('span', { className: 'choice-label', textContent: caption }), ...buttons.map(([b]) => b)]);
  return { node, destroy: () => buttons.forEach(([b, fn]) => b.removeEventListener('click', fn)) };
}

function readouts(view) {
  return [
    readoutTable({
      head: ['This sentence', 'Value'], name: 'readouts',
      rows: [
        { label: `Loss at ${view.selected.word === '.' ? '"."' : view.selected.word}`, sub: `−ln ${view.selected.pText}`, cells: [{ value: view.selected.lossText, name: 'selected-loss' }] },
        { label: 'Mean loss', sub: `over the ${view.strip.length} predictions`, cells: [{ value: view.mean, name: 'mean-loss' }] },
        { label: 'Perplexity', sub: 'e to the mean loss', cells: [{ value: view.perplexity, name: 'perplexity' }] },
      ],
    }),
    readoutTable({
      head: ['Reference marks', 'Loss'],
      rows: [
        { label: 'Uniform guess over 16 words', sub: 'perplexity 16', cells: [{ value: view.ref16, name: 'ref-16' }] },
        { label: `Uniform guess over Kimi K3's ${view.kimiVocab} pieces`, sub: 'where a model that knows nothing starts', cells: [{ value: view.refKimi, name: 'ref-kimi' }] },
      ],
    }),
  ];
}

function buildDom(host, data, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = { stripHost: el('div', { className: 'scroll-x' }), slider: el('div'), presets: el('div'), all: el('div'), tables: el('div', { className: 'toy-tables' }), pre };
  host.append(
    el('p', { className: 'choice-label', id: STRIP_LABEL_ID, textContent: 'Which prediction (select a target chip 2–8)' }), refs.stripHost,
    el('div', { className: 'toy-controls' }, [refs.slider, refs.presets, refs.all]), refs.tables,
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data, ctx),
  );
  return refs;
}

function mountControls(refs, { get, set }) {
  let syncing = false;
  const p = () => get().probs[get().pos - 1];
  // The slider snaps to 0.01; an exact stand-in off that grid (0.3903, 1/16) is shown and kept until the learner drags.
  const slider = mountSlider(refs.slider, {
    id: 'p', label: 'Probability the model gave the selected true token', ...P_SLIDER, value: onGrid(p()),
    format: (v) => formatProb(onGrid(p()) === v ? p() : v),
    onInput: (v) => { if (!syncing) set((s) => setProb(s, v)); },
  });
  const presets = buttonGroup('Set it to', PRESETS.map(({ label, p: value }) => ({ text: label, onClick: () => set((s) => setProb(s, value)) })));
  const all = buttonGroup('Every position', [
    { text: 'uniform guess over 16', onClick: () => set(allUniform) },
    { text: 'Reset', onClick: () => set(resetProbs) },
  ]);
  refs.presets.replaceChildren(presets.node);
  refs.all.replaceChildren(all.node);
  const sync = () => {
    syncing = true;
    slider.set(onGrid(p()));
    syncing = false;
  };
  return { sync, destroy: () => [slider, presets, all].forEach((c) => c.destroy()) };
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data, ctx);
  let toy = null;
  const set = (patch) => toy?.set(patch);
  const get = () => toy?.get() ?? INITIAL_STATE;
  const strip = mountStrip(refs.stripHost, { view: toyView(INITIAL_STATE, data), pos: INITIAL_STATE.pos, labelledBy: STRIP_LABEL_ID, onSelect: (pos) => set((s) => selectPosition(s, pos)) });
  const controls = mountControls(refs, { get, set });
  toy = createToyState(INITIAL_STATE, (s) => {
    const view = toyView(s, data);
    strip.paint(view, s.pos);
    refs.tables.replaceChildren(...readouts(view));
    refs.pre.textContent = view.checkWork;
    if (toy) controls.sync();
  });
  return () => {
    toy.destroy();
    controls.destroy();
    strip.destroy();
    host.replaceChildren();
  };
}
