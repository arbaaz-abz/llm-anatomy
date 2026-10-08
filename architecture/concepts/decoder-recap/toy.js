// decoder-recap toy, "Modernize GPT-3, one switch at a time": mount(el, ctx) → destroy. One state object, one render;
// every string it prints comes from toyView (format.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { appendRich } from '@shared/lesson-page.js';
import { OPTIONS, INITIAL_STATE, MODERN_STATE, ACTIVE_DEFINITION, applySwitch, toyView, tryThis } from './format.js';

const SWITCHES = Object.freeze([
  ['norm', 'Norm'], ['position', 'Position'], ['mlp', 'MLP'], ['biases', 'Biases'], ['kvHeads', 'KV heads'], ['experts', 'Experts (illustrative: not a real model)'],
]);
const PRESET_BUTTONS = Object.freeze([{ value: 'gpt3', label: 'GPT-3 (2020)', state: INITIAL_STATE }, { value: '2026', label: '2026-style', state: MODERN_STATE }]);

function totalsTable(view) {
  return readoutTable({
    head: ['Readout', 'Value'],
    rows: [
      { label: 'Total parameters', cells: [{ value: view.total, sub: view.totalSub, name: 'total' }] },
      { label: 'Active parameters', cells: [{ value: view.active, sub: view.activeSub, name: 'active' }] },
      { label: 'Change vs GPT-3', cells: [{ value: view.change, sub: view.changeSub, name: 'change' }] },
      { label: 'Cache per token', cells: [{ value: view.cache, sub: view.cacheSub, name: 'cache' }] },
      { label: 'Longest input', cells: [{ value: view.longest, name: 'longest' }] },
    ],
  });
}

function partsTable(view) {
  return readoutTable({
    head: ['Part', 'Parameters', 'vs GPT-3'], caption: 'Which switch changed which number',
    rows: view.parts.map(({ part, label, count, delta }) => ({ label, cells: [{ value: count, name: `part-${part}` }, { value: delta, name: `part-${part}-delta` }] })),
  });
}

function tryThisList(ctx) {
  const items = tryThis().map(({ prompt, insight, rest }) => {
    const li = el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]);
    return appendRich(li, rest, ctx);
  });
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, ctx) {
  const refs = {
    presets: el('div', { id: 'presets', className: 'choice choice--chips', role: 'group', ariaLabel: 'Presets' }),
    controls: Object.fromEntries(SWITCHES.map(([key]) => [key, el('div')])),
    totals: el('div'), parts: el('div'),
  };
  PRESET_BUTTONS.forEach(({ value, label }) => {
    const button = el('button', { type: 'button', className: 'choice-option', textContent: label });
    button.dataset.value = value;
    refs.presets.append(button);
  });
  host.append(
    refs.presets, el('div', { className: 'toy-controls' }, Object.values(refs.controls)), refs.totals,
    el('p', { className: 'toy-note', textContent: ACTIVE_DEFINITION }), el('div', { className: 'scroll-x' }, [refs.parts]), ...tryThisList(ctx),
  );
  return refs;
}

function paint(refs, view) {
  refs.totals.replaceChildren(totalsTable(view));
  refs.parts.replaceChildren(partsTable(view));
}

export function mount(host, ctx) {
  const refs = buildDom(host, ctx);
  let toy = null;
  let syncing = false;
  const choices = SWITCHES.map(([key, label]) => mountChoice(refs.controls[key], {
    id: key, label, variant: 'segmented', value: INITIAL_STATE[key], options: OPTIONS[key],
    onChange: (value) => { if (!syncing) toy.set((s) => applySwitch(s, key, value)); },
  }));
  // A preset or a linked switch (experts need SwiGLU) moves other controls: show the state, without echoing it back.
  const sync = (state) => {
    syncing = true;
    SWITCHES.forEach(([key], i) => choices[i].set(state[key]));
    syncing = false;
  };
  const onPreset = (event) => {
    const chosen = PRESET_BUTTONS.find((b) => b.value === event.target.closest('button')?.dataset.value);
    if (chosen) toy.set(chosen.state);
  };
  refs.presets.addEventListener('click', onPreset);
  toy = createToyState(INITIAL_STATE, (state) => { paint(refs, toyView(state)); sync(state); });
  return () => {
    refs.presets.removeEventListener('click', onPreset);
    toy.destroy();
    choices.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
