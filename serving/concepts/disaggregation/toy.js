// disaggregation toy, "Ship the KV, feed the experts": mount(host, ctx) → destroy. One state object, one render;
// every string it prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { formatInt } from '@math/core.js';
import { INITIAL_STATE, PROMPT_STOPS, EP_STOPS, USER_PRESETS, KV_FORMATS, setupFrom } from './format.js';
import { toyView, TOY_NOTE } from './toy-view.js';
import { tryThis } from './try-this.js';
import { output, shipTables, expertTables, paintBar } from './toy-dom.js';

const rich = (text, ctx, props = {}) => appendRich(el('p', { className: 'toy-note', ...props }), text, ctx);

function tryThisList(setup, ctx) {
  const items = tryThis(setup).map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, setup, ctx) {
  const refs = {
    prompt: el('div'), link: el('div'), kv: el('div'), ep: el('div'), users: el('div'),
    shortNote: output('short-note', { className: 'toy-note' }), shipTables: el('div', { className: 'toy-readouts' }),
    expertTables: el('div', { className: 'toy-readouts' }), hbmNote: output('hbm-note'), bar: G.svgEl('svg', { role: 'img' }), check: el('pre', { className: 'check-work', ariaLive: 'polite' }),
  };
  refs.check.dataset.readout = 'check-work';
  host.append(
    rich(TOY_NOTE, ctx),
    el('h4', { textContent: 'Ship the KV' }),
    el('div', { className: 'toy-controls' }, [refs.prompt, refs.link, refs.kv]),
    el('p', { className: 'toy-note' }, [refs.shortNote]),
    refs.shipTables,
    el('h4', { textContent: 'Feed the experts' }),
    el('div', { className: 'toy-controls' }, [refs.ep, refs.users]),
    refs.expertTables,
    el('p', { className: 'toy-note' }, [refs.hbmNote]),
    el('div', { className: 'scroll-x' }, [refs.bar]),
    el('h4', { textContent: 'Check my work' }), refs.check, ...tryThisList(setup, ctx),
  );
  return refs;
}

function paint(refs, view) {
  refs.shipTables.replaceChildren(...shipTables(view.ship));
  refs.shortNote.textContent = view.ship.shortNote;
  refs.shortNote.parentElement.hidden = view.ship.shortNote === '';
  refs.expertTables.replaceChildren(expertTables(view.experts));
  refs.hbmNote.textContent = view.experts.hbmNote;
  paintBar(refs.bar, view.experts.bar);
  refs.check.textContent = view.checkWork;
}

function mountControls(refs, setup, toy) {
  const set = (patch) => toy().set(patch);
  return [
    mountSlider(refs.prompt, { id: 'prompt', label: 'Prompt length', values: PROMPT_STOPS, value: INITIAL_STATE.prompt, unit: 'tokens', format: formatInt, onInput: (prompt) => set({ prompt }) }),
    mountChoice(refs.link, { id: 'link', label: 'Link between pools (per GPU, each way)', variant: 'chips', value: INITIAL_STATE.link, options: setup.links.map((l) => ({ value: l.id, label: l.label })), onChange: (link) => set({ link }) }),
    mountChoice(refs.kv, { id: 'kv', label: 'KV cache', value: INITIAL_STATE.kv, options: Object.entries(KV_FORMATS).map(([value, f]) => ({ value, label: f.label })), onChange: (kv) => set({ kv }) }),
    mountSlider(refs.ep, { id: 'ep', label: 'GPUs sharing the experts (EP size)', values: EP_STOPS, value: INITIAL_STATE.ep, unit: 'GPUs', format: formatInt, onInput: (ep) => set({ ep }) }),
    mountChoice(refs.users, { id: 'users', label: 'Users per GPU', variant: 'chips', value: INITIAL_STATE.users, options: USER_PRESETS.map((u) => ({ value: u, label: formatInt(u) })), onChange: (users) => set({ users }) }),
  ];
}

export function mount(host, ctx) {
  const setup = setupFrom(ctx?.data);
  const refs = buildDom(host, setup, ctx);
  let toy = null;
  const controls = mountControls(refs, setup, () => toy);
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, ctx.data)));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy?.());
    host.replaceChildren();
  };
}
