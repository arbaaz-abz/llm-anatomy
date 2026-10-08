// "Compute one row yourself" (storyboard §6): one state object, one render(state); destroy tears everything down.
import { TOY, multiHead } from '@math/attention.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { fmt2, fmt3, checkWork, divisorText, runHead } from './format.js';
import { numberRowView, heatmapView } from './toy-view.js';

const START = Object.freeze({ query: 2, divisor: 2, causal: true, head: 'A' }); // where the animation ends
const DIVISORS = [0.5, 1, 2, 4, 8];
const ROOT_D = Math.sqrt(TOY.dHead);
const SUBSCRIPTS = ['₁', '₂', '₃', '₄'];
const tokenName = (i) => `${TOY.tokens[i]}${SUBSCRIPTS[i]}`;
const SCALES = Object.freeze({ score: 4.5, weight: 1, out: 1.5 });
const ROWS = [
  ['scores', 'scores (q·k)'], ['scaled', 'scaled'], ['masked', 'masked'], ['weights', 'weights'], ['output', 'output'],
];

function controls(host, set) {
  const parts = ['query', 'divisor', 'causal', 'head'].map(() => el('div'));
  const reset = el('button', { type: 'button', id: 'divisor-reset', className: 'choice-option', textContent: '√d_head' });
  const divisorBox = el('div', {}, [parts[1], reset,
    el('p', { className: 'toy-intro', textContent: 'This slider is a temperature knob on the attention softmax. Real models fix it at √d_head; the slider is here so you can feel why.' }),
    el('p', { className: 'toy-intro' }, [el('code', { textContent: 'typical score size for random ±1 entries: 4 terms → 2; 128 terms → 11' })]),
  ]);
  host.append(parts[0], divisorBox, parts[2], parts[3]);
  const query = mountChoice(parts[0], { id: 'query', label: 'Query token', value: START.query, onChange: (q) => set({ query: q }),
    options: TOY.tokens.map((_, i) => ({ value: i, label: tokenName(i) })) });
  const divisor = mountSlider(parts[1], { id: 'divisor', label: 'Divide scores by', values: DIVISORS, value: START.divisor, format: divisorText, onInput: (d) => set({ divisor: d }) });
  const onReset = () => divisor.set(ROOT_D);
  reset.addEventListener('click', onReset);
  const causal = mountToggle(parts[2], { id: 'causal', label: 'Causal mask', value: START.causal, onChange: (on) => set({ causal: on }) });
  const head = mountChoice(parts[3], { id: 'head', label: 'Head', value: START.head, onChange: (h) => set({ head: h }),
    options: [{ value: 'A', label: 'A' }, { value: 'B', label: 'B' }, { value: 'both', label: 'both' }] });
  return () => { reset.removeEventListener('click', onReset); [query, divisor, causal, head].forEach((c) => c.destroy()); };
}

function readouts(host) {
  const title = el('p', { className: 'choice-label' });
  const rows = Object.fromEntries(ROWS.map(([name, text]) => [name, numberRowView(name, text)]));
  const weightSum = el('output', { className: 'readout' });
  weightSum.dataset.readout = 'weight-sum';
  const concat = numberRowView('concat', 'joined (A ‖ B)');
  const concatBox = el('div', {}, [concat.node]);
  concatBox.style.cssText = 'overflow-x:auto;max-width:100%';
  const maps = el('div', { style: 'display:flex;flex-wrap:wrap;gap:var(--space-4)' });
  const check = el('pre', { className: 'check-work', ariaLive: 'polite' });
  check.dataset.readout = 'check-work';
  const rowBox = el('div', { style: 'display:grid;gap:3px;overflow-x:auto;max-width:100%' }, ROWS.map(([name]) => rows[name].node));
  host.append(title, rowBox, weightSum, concatBox, maps,
    el('p', { className: 'toy-intro', textContent: 'exp needs a calculator; everything else is arithmetic, and ±0.001 is rounding, not a mistake.' }),
    el('h4', { textContent: 'Check my work' }), check);
  return { title, rows, weightSum, concat, concatBox, maps, check };
}

function render(view, state) {
  const r = runHead(state);
  const q = state.query;
  const hidden = (j) => !r.mask[q][j];
  const cells = (row, fmt, scale, hatch = hidden) => row.map((v, j) => ({ text: fmt(v), v, scale, hatched: hatch(j) }));
  const rowsHead = state.head === 'B' ? 'B' : 'A';
  view.title.textContent = `${state.head === 'both' ? 'Rows show head A' : `Head ${rowsHead}`}, query ${tokenName(q)}`;
  view.rows.scores.set(cells(r.scores[q], fmt2, SCALES.score));
  view.rows.scaled.set(cells(r.scaled[q], fmt3, SCALES.score, () => false));
  view.rows.masked.set(cells(r.masked[q], fmt3, SCALES.score));
  view.rows.weights.set(cells(r.weights[q], fmt3, SCALES.weight));
  view.rows.output.set(cells(r.output[q], fmt3, SCALES.out, () => false));
  view.weightSum.textContent = `Σ = ${r.weights[q].reduce((s, w) => s + w, 0).toFixed(3)}`;
  const both = state.head === 'both';
  view.concatBox.hidden = !both;
  const opts = { causal: state.causal, divisor: state.divisor };
  if (both) view.concat.set(cells(multiHead([TOY.heads.A, TOY.heads.B], opts).concat[q], fmt3, SCALES.out, () => false));
  const shown = both ? ['A', 'B'] : [rowsHead];
  view.maps.replaceChildren(...shown.map((name) => {
    const h = name === rowsHead ? r : runHead({ ...state, head: name });
    return heatmapView({ name, weights: h.weights, mask: h.mask, query: q });
  }));
  view.check.textContent = checkWork(state);
}

export function mount(host) {
  const controlBox = el('div', { style: 'display:grid;gap:var(--space-3)' });
  const readoutBox = el('div', { style: 'display:grid;gap:var(--space-3);min-width:0' });
  host.append(controlBox, readoutBox);
  const view = readouts(readoutBox);
  const toy = createToyState(START, (state) => render(view, state));
  const unmountControls = controls(controlBox, (patch) => toy.set(patch));
  return () => {
    toy.destroy();
    unmountControls();
    host.replaceChildren();
  };
}
