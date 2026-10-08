// rlhf-dpo toy, "Tune a DPO pair": mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { INITIAL_STATE, TOY_LIMITS, fmtChange, trim1 } from './format.js';
import { toyView } from './toy-view.js';
import { tryThis } from './try-this.js';
import { mountRewardCells, tryThisList } from './toy-dom.js';

const REWARDS_LABEL_ID = 'rlhf-dpo-rewards-label';

function buildDom(host, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    sliderA: el('div'), sliderB: el('div'), beta: el('div'), reset: el('button', { type: 'button', className: 'choice-option', id: 'toy-reset', textContent: 'Reset' }),
    cells: el('div'), table: el('div', { className: 'toy-tables' }), pre,
  };
  host.append(
    el('div', { className: 'toy-controls' }, [refs.sliderA, refs.sliderB, refs.beta, refs.reset]),
    el('p', { className: 'choice-label', id: REWARDS_LABEL_ID, textContent: 'Implicit reward of A and of B (β × the change)' }), refs.cells,
    refs.table, el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(tryThis(), ctx),
  );
  return refs;
}

const outputsTable = (view) => readoutTable({
  head: ['DPO on this pair', 'Value'],
  rows: [
    { label: 'Gap (margin)', sub: 'implicit reward of A minus B', cells: [{ value: view.margin, name: 'margin' }] },
    { label: 'P(A preferred)', sub: 'as the loss sees it, σ(gap)', cells: [{ value: view.pChosen, name: 'p-chosen' }] },
    { label: 'Loss', sub: view.lossRef, cells: [{ value: view.loss, name: 'loss' }] },
    { label: 'Update weight', sub: 'how hard DPO still pushes this pair', cells: [{ value: view.weight, name: 'weight' }] },
  ],
});

function mountControls(refs, { get, set }) {
  const { change, betas } = TOY_LIMITS;
  const slider = (host, id, label, key) => mountSlider(host, { id, label, ...change, value: INITIAL_STATE[key], format: fmtChange, onInput: (v) => set({ [key]: v }) });
  const a = slider(refs.sliderA, 'dChosen', 'Change in log-probability of A (chosen) vs the reference', 'dChosen');
  const b = slider(refs.sliderB, 'dRejected', 'Change in log-probability of B (rejected) vs the reference', 'dRejected');
  const beta = mountChoice(refs.beta, { id: 'beta', label: 'β (how tightly DPO is tied to the reference)', variant: 'chips', value: INITIAL_STATE.beta, options: betas.map((v) => ({ value: v, label: trim1(v) })), onChange: (v) => set({ beta: v }) });
  const reset = () => { a.set(INITIAL_STATE.dChosen); b.set(INITIAL_STATE.dRejected); if (get().beta !== INITIAL_STATE.beta) beta.set(INITIAL_STATE.beta); };
  refs.reset.addEventListener('click', reset);
  return { destroy: () => { [a, b, beta].forEach((c) => c.destroy()); refs.reset.removeEventListener('click', reset); } };
}

export function mount(host, ctx) {
  const refs = buildDom(host, ctx);
  const cells = mountRewardCells(refs.cells, { labelledBy: REWARDS_LABEL_ID });
  let toy = null;
  const controls = mountControls(refs, { get: () => toy?.get() ?? INITIAL_STATE, set: (patch) => toy?.set(patch) });
  toy = createToyState(INITIAL_STATE, (state) => {
    const view = toyView(state);
    cells.paint(view.rewardValues);
    refs.table.replaceChildren(outputsTable(view));
    refs.pre.textContent = view.checkWork;
  });
  return () => {
    toy.destroy();
    controls.destroy();
    cells.destroy();
    host.replaceChildren();
  };
}
