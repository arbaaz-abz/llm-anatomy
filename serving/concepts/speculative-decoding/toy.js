// speculative-decoding toy, "Guess and check": mount(el, ctx) → destroy. One state object, one render; every string it
// prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { mountPresetButtons } from '@shared/ui/preset-buttons.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { formatInt } from '@math/core.js';
import { ZOOM_WORDS } from './numbers.js';
import { probText } from './format.js';
import { ALPHA_RANGE, C_OPTIONS, INITIAL_STATE, K_LIMITS, batchStops, mtpPresets, toyView } from './toy-view.js';
import { tryThis } from './try-this.js';
import { batchTable, output, paintBatchPlot, paintKPlot, paintPosition, positionTable, roundTable } from './toy-dom.js';

const STAND_IN = 'Acceptance rates, probabilities and drafter cost are stand-ins; real acceptance falls with depth and varies by task (code high, creative text low). The drafter cost is a fraction of one target step.';
const rich = (text, ctx, props = {}) => appendRich(el('p', { className: 'toy-note', ...props }), text, ctx);

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    alpha: el('div'), mtp: el('div'), k: el('div'), c: el('div'), batch: el('div'), guess: el('div'),
    round: el('div'), batchTable: el('div'), position: el('div'), usersNote: output('users-note'), check: pre,
    kPlot: G.svgEl('svg', { role: 'img' }), batchPlot: G.svgEl('svg', { role: 'img' }), vectors: G.svgEl('svg', { role: 'img', 'aria-label': 'p, q, leftover and result' }),
  };
  host.append(
    rich(STAND_IN, ctx),
    el('div', { className: 'toy-controls' }, [refs.alpha, refs.mtp, refs.k, refs.c, refs.batch]),
    appendRich(el('p', { className: 'toy-note' }, [refs.usersNote, ' ']), 'How the cache sizes are worked out: [[prefill-decode]].', ctx),
    el('div', { className: 'toy-tables' }, [refs.round, refs.batchTable]),
    el('div', { className: 'scroll-x' }, [refs.kPlot]), el('div', { className: 'scroll-x' }, [refs.batchPlot]),
    el('h4', { textContent: 'One position' }), refs.guess, refs.position, el('div', { className: 'scroll-x' }, [refs.vectors]),
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(ctx?.data, ctx),
  );
  return refs;
}

function paint(refs, state, data, max) {
  const view = toyView(state, data);
  refs.round.replaceChildren(roundTable(view));
  refs.batchTable.replaceChildren(batchTable(view));
  refs.position.replaceChildren(positionTable(view, state.guess));
  refs.usersNote.textContent = view.usersNote;
  refs.check.textContent = view.checkWork;
  paintKPlot(refs.kPlot, view, state);
  paintBatchPlot(refs.batchPlot, view, state, max);
  paintPosition(refs.vectors, view, state.guess);
}

function mountControls(refs, data, set) {
  const stops = batchStops(data);
  const alpha = mountSlider(refs.alpha, { id: 'alpha', label: 'Acceptance rate α', ...ALPHA_RANGE, value: INITIAL_STATE.alpha, format: probText, onInput: (v) => set({ alpha: v }) });
  const k = mountSlider(refs.k, { id: 'k', label: 'Guesses per round', ...K_LIMITS, step: 1, value: INITIAL_STATE.k, unit: 'guesses', onInput: (v) => set({ k: v }) });
  const mtp = mountPresetButtons(refs.mtp, {
    id: 'mtp-preset', label: 'Multi-token prediction', options: mtpPresets(data),
    onPick: (a) => { alpha.set(a); k.set(1); },
  });
  const c = mountChoice(refs.c, { id: 'c', label: 'Drafter cost per guess', variant: 'chips', value: INITIAL_STATE.c, options: C_OPTIONS.map((v) => ({ value: v, label: probText(v) })), onChange: (v) => set({ c: v }) });
  const batch = mountSlider(refs.batch, { id: 'batch', label: 'Users in the batch', values: stops, value: INITIAL_STATE.batch, format: (v) => `${formatInt(v)} ${v === 1 ? 'user' : 'users'}`, onInput: (v) => set({ batch: v }) });
  const guess = mountChoice(refs.guess, { id: 'guess', label: "Drafter's guess", variant: 'chips', value: INITIAL_STATE.guess, options: ZOOM_WORDS.map((w) => ({ value: w, label: w })), onChange: (v) => set({ guess: v }) });
  return [alpha, k, mtp, c, batch, guess];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, ctx);
  const max = batchStops(data).at(-1);
  let toy = null;
  const controls = mountControls(refs, data, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, s, data, max));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
