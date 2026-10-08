// training-pipeline toy, "Read a recipe": mount(el, ctx) → destroy. One state object, one render; every string it
// prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { MODEL_CHIPS, INITIAL_STATE, toyView, tryThis } from './toy-view.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { mountStrip, STRIP } from './select.js';
import { stageStrip } from './toy-view.js';

const BAR_W = 400;

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

// One labelled number as a readout table (template layout: readoutTable for any labelled numbers).
const totalTable = () => readoutTable({ rows: [{ label: 'Known tokens total', cells: [{ value: '', name: 'known-total' }] }] });

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    chips: el('div'),
    strip: G.svgEl('svg', { role: 'group', 'aria-label': 'the six stages', width: STRIP.w, height: STRIP.h, viewBox: `0 0 ${STRIP.w} ${STRIP.h}` }),
    inspector: output('inspector'),
    stops: el('p', { className: 'toy-note' }),
    bar: G.svgEl('svg', { role: 'group', 'aria-label': 'token budget by stage' }),
    totalTable: totalTable(),
    pre,
  };
  host.append(
    refs.chips,
    el('div', { className: 'scroll-x' }, [refs.strip]),
    el('p', {}, [refs.inspector]), refs.stops,
    el('div', { className: 'scroll-x' }, [refs.bar]),
    refs.totalTable,
    el('p', { className: 'toy-note' }, ['The bar uses published figures only. The toy never estimates an unpublished token count, so it cannot make post-training look smaller or larger than it is.']),
    el('h4', { textContent: 'Check my work' }), pre,
    ...tryThisList(data, ctx),
  );
  return refs;
}

function paintBar(svg, parts) {
  svg.replaceChildren();
  G.shareBar(svg, { x: 4, y: 6, w: BAR_W, parts, label: 'token budget', minSegment: 12 });
  G.fitViewBox(svg, 8);
}

function paint(refs, view, state, set, ctx) {
  refs.stripView.paint(view.strip, state.stage);
  refs.inspector.textContent = view.inspector;
  refs.stops.replaceChildren();
  appendRich(refs.stops, view.stops, ctx);
  paintBar(refs.bar, view.parts);
  refs.totalTable.querySelector('[data-readout="known-total"]').textContent = view.knownTotal;
  refs.pre.textContent = view.checkWork;
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data, ctx);
  let toy = null;
  const set = (patch) => toy?.set(patch);
  refs.stripView = mountStrip(refs.strip, stageStrip(INITIAL_STATE.model), INITIAL_STATE.stage, (stage) => set({ stage }));
  const chips = mountChoice(refs.chips, {
    id: 'model', label: 'Whose recipe', variant: 'chips', value: INITIAL_STATE.model, options: MODEL_CHIPS.map(({ value, label }) => ({ value, label })), onChange: (model) => set({ model }),
  });
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data), s, set, ctx));
  return () => {
    toy.destroy();
    refs.stripView.destroy();
    chips.destroy();
    host.replaceChildren();
  };
}
