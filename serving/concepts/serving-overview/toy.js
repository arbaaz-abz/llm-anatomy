// serving-overview toy, "Where does the time go?" (storyboard §6): mount(host, ctx) → destroy. One state object, one paint;
// every string it prints comes from toyView (toy-view.js) and tryThis (try-this.js), so the page shows what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { formatDuration, formatInt } from '@math/core.js';
import { INITIAL_STATE, PROMPT_STOPS, OUTPUT_STOPS, QUEUE } from './format.js';
import { toyView, speeds } from './toy-view.js';
import { tryThis } from './try-this.js';
import { output, timelineTable, paintFigure } from './toy-dom.js';

const LONG_PROMPT_NOTE = 'The floor counts the weights\' math only and leaves out attention\'s own math, which is small at chat sizes and grows with the square of the prompt, so the longest prompts here would take noticeably longer in practice.';
const UNDER_TOY = 'TTFT and TPOT are the two numbers every later page optimizes.';

const rich = (text, ctx) => appendRich(el('p', { className: 'toy-note' }), text, ctx);

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx, chips) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    prompt: el('div'), output: el('div'), queue: el('div'), rate: el('div'),
    table: el('div', { className: 'scroll-x' }),
    widthNote: output('timeline-width'), tickNote: output('tick-note'),
    figure: G.svgEl('svg', { role: 'img', 'aria-label': 'your request on one time axis, drawn to scale' }),
    check: pre,
  };
  host.append(
    el('div', { className: 'toy-controls' }, [refs.prompt, refs.output, refs.queue, refs.rate]),
    rich(chips.note, ctx), rich(LONG_PROMPT_NOTE, ctx),
    refs.table,
    el('p', { className: 'toy-note' }, [refs.widthNote, '; ', refs.tickNote]), el('div', { className: 'scroll-x' }, [refs.figure]),
    rich(UNDER_TOY, ctx),
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data, ctx),
  );
  return refs;
}

function paint(refs, view) {
  refs.table.replaceChildren(timelineTable(view.readouts));
  refs.widthNote.textContent = view.widthNote;
  refs.tickNote.textContent = view.tickNote;
  paintFigure(refs.figure, view.figure);
  refs.check.textContent = view.checkWork;
}

function mountControls(refs, chips, toy) {
  const set = (key) => (v) => toy().set({ [key]: v });
  return [
    mountSlider(refs.prompt, { id: 'prompt', label: 'Prompt length', values: PROMPT_STOPS, value: INITIAL_STATE.prompt, unit: 'tokens', format: formatInt, onInput: set('prompt') }),
    mountSlider(refs.output, { id: 'output', label: 'Answer length', values: OUTPUT_STOPS, value: INITIAL_STATE.output, unit: 'tokens', format: formatInt, onInput: set('output') }),
    mountSlider(refs.queue, { id: 'queue', label: 'Time in queue', ...QUEUE, value: INITIAL_STATE.queue, format: formatDuration, onInput: set('queue') }),
    mountChoice(refs.rate, { id: 'decodeRate', label: 'Decode speed per user', variant: 'chips', value: INITIAL_STATE.decodeRate, options: chips.options, onChange: set('decodeRate') }),
  ];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const chips = speeds(data);
  const refs = buildDom(host, data, ctx, chips);
  let toy = null;
  const controls = mountControls(refs, chips, () => toy);
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data)));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
