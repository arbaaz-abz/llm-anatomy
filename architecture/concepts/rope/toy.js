// "Turn the hands" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { TOY } from '@math/attention.js';
import { POSITION_MAX, TOY_BASES, TARGETS, STRETCH_OPTIONS, TRAINED_LENGTH, thousands, fmt3 } from './format.js';
import { INITIAL_STATE, SHIFT, toyView, tryThis } from './toy-view.js';
import { numberRowView, handsView } from './toy-dom.js';
import { SCALE } from './stage.js';

const scroller = () => el('div', { className: 'scroll-x' }); // a wide table scrolls in its own box at 400 px

const cells = (values, texts, scale) => values.map((v, i) => ({ v, text: texts?.[i] ?? fmt3(v), scale }));

function scoreTable(view) {
  return readoutTable({
    head: ['Quantity', 'Value'], name: 'score-table',
    rows: [
      { label: 'Pair 1 dot', cells: [{ value: view.pairDots[0], name: 'pair-dot-1' }] },
      { label: 'Pair 2 dot', cells: [{ value: view.pairDots[1], name: 'pair-dot-2' }] },
      { label: 'Rotated score', sub: 'the sum of the two pair dots', cells: [{ value: view.score, name: 'score' }] },
      { label: 'Unrotated q · k', sub: 'what attention computed before RoPE', cells: [{ value: view.plain, name: 'score-plain' }] },
      { label: `Same offset, ${SHIFT} later`, sub: view.shiftedNote, cells: [{ value: view.shifted, name: 'score-shifted' }] },
    ],
  });
}

function coverageTable(view) {
  const [a, b] = view.columns;
  const row = (label, key, name) => ({ label, cells: [{ value: a[key], name: `${name}-1` }, { value: b[key], name: `${name}-2` }] });
  return readoutTable({
    head: ['', 'Pair 1', 'Pair 2'], name: 'coverage',
    rows: [
      row('Turns once every', 'wavelength', 'wavelength'),
      row('Seen in training, up to', 'seen', 'seen'),
      row(view.reachTitle, 'reached', 'reached'),
      row('Every angle seen?', 'verdict', 'verdict'),
    ],
  });
}

function realTable(view) {
  return readoutTable({
    head: ['Model', 'Base θ', 'Slowest pair turns once every (tokens)'], name: 'real-heads',
    caption: 'A real head: 128 numbers, 64 pairs',
    rows: view.realHeads.map((r) => ({ label: r.name, cells: [{ value: r.baseText, name: `base-${r.id}` }, { value: r.tokensText, name: `real-${r.id}` }] })),
  });
}

function buildDom(host, data) {
  const refs = {
    sliders: [el('div'), el('div'), el('div'), el('div')], choices: [el('div'), el('div')],
    shift: el('button', { type: 'button', id: 'shift10', className: 'choice-option', textContent: 'same offset, +10' }),
    hands: handsView(), q: numberRowView('q-rotated', 'q_sat, turned'), k: numberRowView('k-rotated', 'key, turned'),
    offsets: numberRowView('offset-row', 'score at offset 0 to 7'),
    score: scroller(), coverage: scroller(), real: scroller(),
    check: el('pre', { className: 'check-work', ariaLive: 'polite' }),
  };
  refs.check.dataset.readout = 'check-work';
  const [qPos, kPos, base, target] = refs.sliders;
  const [kToken, stretch] = refs.choices;
  const list = el('ol', { className: 'try-this' }, tryThis(data).map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest])));
  host.append(
    el('div', { className: 'toy-controls scroll-x' }, [qPos, kToken, kPos, refs.shift, base, target, stretch]),
    el('div', { className: 'scroll-x' }, [refs.hands.node]),
    el('div', { className: 'toy-rows' }, [refs.q.node, refs.k.node]),
    refs.score,
    el('div', { className: 'toy-rows' }, [refs.offsets.node]),
    refs.coverage, refs.real,
    el('p', { className: 'toy-note', textContent: 'cos and sin need a calculator; the products use the rounded numbers shown, so ±0.001 is rounding, not a mistake.' }),
    el('h4', { textContent: 'Check my work' }), refs.check,
    el('h4', { textContent: 'Try this' }), list,
  );
  return refs;
}

function paint(refs, view, state) {
  refs.hands.paint(view.hands);
  refs.q.set(cells(view.qRotated));
  refs.k.set(cells(view.kRotated));
  refs.offsets.set(view.offsetValues.map((v, i) => ({ v, text: view.offsetRow[i], scale: SCALE.score })));
  refs.score.replaceChildren(scoreTable(view));
  refs.coverage.replaceChildren(coverageTable(view));
  refs.real.replaceChildren(realTable(view));
  refs.check.textContent = view.checkWork;
  refs.shift.disabled = state.qPos + SHIFT > POSITION_MAX;
}

function mountControls(refs, toy) {
  const [qHost, kHost, baseHost, targetHost] = refs.sliders;
  const [kTokenHost, stretchHost] = refs.choices;
  const set = (patch) => toy.current()?.set(patch);
  let kSlider = null;
  const keepKeyBehind = (q) => { if (toy.current()?.get().kPos > q) kSlider?.set(q); };
  const qSlider = mountSlider(qHost, { id: 'qPos', label: 'Query position ("sat")', min: 1, max: POSITION_MAX, step: 1, value: INITIAL_STATE.qPos, format: (v) => `token ${v}`, onInput: (v) => { set({ qPos: v }); keepKeyBehind(v); } });
  kSlider = mountSlider(kHost, { id: 'kPos', label: 'Key position', min: 1, max: POSITION_MAX, step: 1, value: INITIAL_STATE.kPos, format: (v) => `token ${v}`, onInput: (v) => {
    const q = toy.current()?.get().qPos ?? v;
    if (v > q) kSlider.set(q); else set({ kPos: v });
  } });
  const kToken = mountChoice(kTokenHost, { id: 'kToken', label: 'Key', value: INITIAL_STATE.kToken, options: TOY.tokens.map((t, i) => ({ value: i, label: t })), onChange: (v) => set({ kToken: v }) });
  const base = mountSlider(baseHost, { id: 'base', label: 'Base θ', values: TOY_BASES, value: INITIAL_STATE.base, format: thousands, onInput: (v) => set({ base: v }) });
  const target = mountSlider(targetHost, { id: 'target', label: `Read up to (trained: ${TRAINED_LENGTH} tokens)`, values: TARGETS, value: INITIAL_STATE.target, unit: 'tokens', onInput: (v) => set({ target: v }) });
  const stretch = mountChoice(stretchHost, { id: 'stretch', label: 'Stretch method', value: INITIAL_STATE.stretch, options: STRETCH_OPTIONS, onChange: (v) => set({ stretch: v }) });
  const onShift = () => {
    const s = toy.current().get();
    const d = Math.min(SHIFT, POSITION_MAX - s.qPos);
    if (d > 0) { qSlider.set(s.qPos + d); kSlider.set(s.kPos + d); }
  };
  refs.shift.addEventListener('click', onShift);
  return () => { refs.shift.removeEventListener('click', onShift); [qSlider, kSlider, kToken, base, target, stretch].forEach((c) => c.destroy()); };
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data);
  let state = null;
  const toy = { current: () => state };
  const unmountControls = mountControls(refs, toy);
  state = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s));
  return () => {
    state.destroy();
    unmountControls();
    host.replaceChildren();
  };
}
