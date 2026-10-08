// "Which tokens teach?" (storyboard §6): one state object, one paint(state); destroy tears everything down.
import { el } from '@shared/ui/dom.js';
import { appendRich } from '@shared/lesson-page.js';
import { mountToggle, mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { INITIAL_STATE } from './format.js';
import { TEMPLATE_OPTIONS } from './numbers.js';
import { view, tryThis } from './toy-view.js';
import { drawTranscript, countsTable, kindsTable } from './toy-dom.js';

const TOGGLES = Object.freeze([
  { id: 'maskPrompt', label: 'Mask the prompt and template tags' },
  { id: 'maskObservation', label: 'Mask the tool\'s reply' },
  { id: 'maskError', label: 'Mask the mistake in the trace (GLM-5)' },
]);

function boxes(host) {
  const box = () => el('div', {});
  const refs = { toggles: TOGGLES.map(box), template: box() };
  host.append(...refs.toggles, refs.template);
  return refs;
}

function mountControls(refs, set) {
  const toggles = TOGGLES.map(({ id, label }, i) => mountToggle(refs.toggles[i], { id, label, value: INITIAL_STATE[id], onChange: (on) => set({ [id]: on }) }));
  const template = mountChoice(refs.template, { id: 'template', label: 'Show the template as (counts unchanged)', options: TEMPLATE_OPTIONS, value: INITIAL_STATE.template, variant: 'chips', onChange: (value) => set({ template: value }) });
  return () => { toggles.forEach((t) => t.destroy()); template.destroy(); };
}

function readouts(host, ctx) {
  const named = (tag, props, name) => { const node = el(tag, props); node.dataset.readout = name; return node; };
  const refs = {
    transcript: el('div', { className: 'toy-transcript stepper-stage' }),
    selection: named('output', { className: 'readout', ariaLive: 'polite' }, 'selection'),
    note: named('p', { className: 'toy-note' }, 'template-note'),
    counts: el('div', {}),
    kinds: el('div', {}),
    check: named('pre', { className: 'check-work', ariaLive: 'polite' }, 'check-work'),
  };
  const tries = tryThis().map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  host.append(refs.transcript, refs.selection, refs.note, refs.counts, refs.kinds, el('h4', { textContent: 'Check my work' }), refs.check,
    el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, tries));
  return refs;
}

export function mount(host, ctx) {
  const controlHost = el('div', { className: 'toy-controls' });
  host.append(controlHost);
  // Transcript and readouts are direct grid items of the toy (no wrapper), so the 580 px scroller and the check box shrink on a phone.
  const refs = readouts(host, ctx);
  const controlBoxes = boxes(controlHost);
  let toy = null;
  let refocus = false;
  const select = (flat, keyboard) => { refocus = keyboard; toy.set({ selected: flat }); };
  const paint = (state) => {
    const v = view(state, ctx?.data);
    const nodes = drawTranscript(refs.transcript, v.chips, state.selected, select);
    if (refocus) nodes[state.selected].focus();
    refocus = false;
    refs.selection.textContent = v.selection;
    refs.note.textContent = v.templateNote;
    refs.counts.replaceChildren(countsTable(v));
    refs.kinds.replaceChildren(kindsTable(v));
    refs.check.textContent = v.checkWork;
  };
  toy = createToyState(INITIAL_STATE, paint);
  const unmountControls = mountControls(controlBoxes, (patch) => toy.set(patch));
  return () => {
    toy.destroy();
    unmountControls();
    host.replaceChildren();
  };
}
