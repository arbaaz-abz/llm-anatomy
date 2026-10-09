// batching toy, "Seat timeline": mount(el, ctx) → destroy. One state object, one render; every string it prints comes from
// toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { INITIAL_STATE, REQUESTS_TITLE, TOY_LIMITS, PROMPT_OPTIONS, BUDGET_OPTIONS, toyView, tryThis } from './toy-view.js';
import { drawToy, TOY_STAGE } from './toy-draw.js';

const STEP_HEADS = ['admitted', 'done', 'waited'];

function summaryTable(view) {
  return readoutTable({ head: ['', 'Static', 'Continuous'], name: 'summary', rows: view.summaryRows });
}

function requestsTable(view) {
  const head = ['Request', ...['Static', 'Continuous'].flatMap((lane) => STEP_HEADS.map((h) => `${lane} ${h}`))];
  return readoutTable({ head, name: 'requests', rows: view.requestRows.map((r) => ({ label: r.id, cells: r.cells })) });
}

function timingTable(view) {
  return readoutTable({
    head: ['Time on the running example', 'Continuous lane'], name: 'timing',
    rows: view.timingRows.map((r) => ({ label: r.label, cells: [{ value: r.value, name: r.name }] })),
  });
}

function tryThisList() {
  const items = tryThis().map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host) {
  const refs = {
    sliders: [el('div'), el('div')], prompt: el('div'), budget: el('div'), budgetNote: el('p', { className: 'toy-note', textContent: 'The token budget applies to the continuous lane only.' }),
    svg: G.svgEl('svg', { width: TOY_STAGE.w, height: TOY_STAGE.h, viewBox: `0 0 ${TOY_STAGE.w} ${TOY_STAGE.h}`, role: 'img' }),
    summary: el('div'), summaryNote: el('p', { className: 'toy-note' }), requestsTitle: el('h4', { textContent: REQUESTS_TITLE }), requests: el('div'), timing: el('div'),
  };
  refs.budgetBox = el('div', {}, [refs.budget, refs.budgetNote]);
  refs.controls = el('div', { className: 'toy-controls' }, [...refs.sliders, refs.prompt, refs.budgetBox]);
  host.append(
    refs.controls, el('div', { className: 'scroll-x' }, [refs.svg]),
    el('h4', { textContent: 'Both lanes' }), refs.summary, refs.summaryNote,
    refs.requestsTitle, el('div', { className: 'scroll-x' }, [refs.requests]),
    el('h4', { textContent: 'In milliseconds' }), refs.timing,
    ...tryThisList(),
  );
  return refs;
}

function paint(refs, view) {
  drawToy(refs.svg, view);
  refs.summary.replaceChildren(summaryTable(view));
  refs.summaryNote.textContent = view.summaryNote;
  refs.summaryNote.hidden = !view.summaryNote;
  refs.requestsTitle.textContent = view.requestsTitle;
  refs.requests.replaceChildren(requestsTable(view));
  refs.timing.replaceChildren(timingTable(view));
  refs.budgetBox.hidden = !view.showBudget;
}

function mountControls(refs, set) {
  const seats = mountSlider(refs.sliders[0], { id: 'seats', label: 'Seats (max requests running at once)', values: TOY_LIMITS.seats, value: INITIAL_STATE.seats, unit: 'seats', onInput: (v) => set({ seats: v }) });
  const cOutput = mountSlider(refs.sliders[1], { id: 'cOutput', label: 'C\'s answer length', min: TOY_LIMITS.cOutput[0], max: TOY_LIMITS.cOutput[1], step: 1, value: INITIAL_STATE.cOutput, unit: 'tokens', onInput: (v) => set({ cOutput: v }) });
  const prompt = mountChoice(refs.prompt, { id: 'dPrompt', label: 'D\'s prompt', variant: 'chips', value: INITIAL_STATE.dPrompt, options: PROMPT_OPTIONS, onChange: (v) => set({ dPrompt: v }) });
  const budget = mountChoice(refs.budget, { id: 'budget', label: 'Token budget per step (chunked prefill)', variant: 'chips', value: INITIAL_STATE.budget, options: BUDGET_OPTIONS, onChange: (v) => set({ budget: v }) });
  return [seats, cOutput, prompt, budget];
}

export function mount(host) {
  const refs = buildDom(host);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s)));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
