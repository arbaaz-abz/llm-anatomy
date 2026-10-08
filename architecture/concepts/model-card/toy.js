// model-card toy, "Decode a card": mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { INITIAL_STATE, CARDS, NONE, CONTEXT_OPTIONS, toyView, tryThis } from './toy-view.js';
import { fieldsTable, costsTable, barsBlock, fitBars, notesBlock } from './toy-dom.js';

const CARD_OPTIONS = Object.freeze(CARDS.map(({ id, name }) => ({ value: id, label: name })));
const RIGHT_OPTIONS = Object.freeze([...CARD_OPTIONS, { value: NONE, label: 'none' }]);

function buildDom(host, data) {
  const refs = {
    left: el('div'), right: el('div'), context: el('div'),
    fields: el('div', { className: 'scroll-x' }), costs: el('div', { className: 'scroll-x' }),
    bars: el('div', { className: 'scroll-x toy-bars' }), notes: el('div'),
  };
  const items = tryThis(data).map(({ prompt, insight }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]));
  host.append(
    el('div', { className: 'toy-controls' }, [refs.left, refs.right, refs.context]),
    refs.fields, refs.costs, el('h4', { textContent: 'Active share, drawn' }), refs.bars, refs.notes,
    el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items),
  );
  return refs;
}

function paint(refs, view, ctx) {
  refs.fields.replaceChildren(fieldsTable(view, ctx));
  refs.costs.replaceChildren(costsTable(view));
  refs.bars.replaceChildren(...barsBlock(view));
  fitBars(refs.bars);
  refs.notes.replaceChildren(...notesBlock(view));
}

function mountControls(refs, set) {
  return [
    mountChoice(refs.left, { id: 'left', label: 'Card', variant: 'chips', value: INITIAL_STATE.left, options: CARD_OPTIONS, onChange: (v) => set({ left: v }) }),
    mountChoice(refs.right, { id: 'right', label: 'Compare with', variant: 'chips', value: INITIAL_STATE.right, options: RIGHT_OPTIONS, onChange: (v) => set({ right: v }) }),
    mountChoice(refs.context, { id: 'context', label: 'Conversation length for the cache lines', variant: 'segmented', value: INITIAL_STATE.context, options: CONTEXT_OPTIONS, onChange: (v) => set({ context: v }) }),
  ];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  toy = createToyState(INITIAL_STATE, (state) => paint(refs, toyView(state, data), ctx));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
