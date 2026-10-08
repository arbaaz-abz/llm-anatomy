// gpu-primer toy, "Where does this multiply sit on the roof?": mount(host, ctx) → destroy. One state object, one render;
// every string it prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { CHIPS, chipPreset, formatOptions, usableFormat } from './hardware.js';
import { INITIAL_STATE, TOKEN_STOPS, TOKEN_PRESETS, int } from './format.js';
import { toyView, FP8_NOTE, SAME_FORMAT_NOTE, RESIDUAL_NOTE } from './toy-view.js';
import { tryThis } from './try-this.js';
import { output, readoutTables, paintPlot, paintLanes } from './toy-dom.js';
import { mountPresetButtons } from '@shared/ui/preset-buttons.js';

const rich = (text, ctx, props = {}) => appendRich(el('p', { className: 'toy-note', ...props }), text, ctx);

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => {
    const li = el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]);
    return appendRich(li, rest, ctx);
  });
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    chip: el('div'), fmt: el('div'), tokens: el('div'), presets: el('div'),
    tables: el('div', { className: 'toy-readouts' }), // stacked: Rubin's two-ended values need the full width
    plotTitle: output('plot-title', { className: 'choice-label' }),
    plot: G.svgEl('svg', { role: 'img' }), lanes: G.svgEl('svg', { role: 'img', 'aria-label': 'memory and compute time on one axis' }),
    conflict: output('conflict'), lanesWidth: output('lanes-width'), check: pre,
  };
  host.append(
    el('div', { className: 'toy-controls' }, [refs.chip, refs.fmt, rich(SAME_FORMAT_NOTE, ctx), rich(FP8_NOTE, ctx), refs.tokens, refs.presets]),
    refs.tables,
    el('p', {}, [refs.plotTitle]), el('div', { className: 'scroll-x' }, [refs.plot]),
    el('p', { className: 'toy-note' }, [refs.conflict]), rich(`The plain dot is a ${RESIDUAL_NOTE}.`, ctx),
    el('p', { className: 'toy-note' }, [refs.lanesWidth]), el('div', { className: 'scroll-x' }, [refs.lanes]),
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data, ctx),
  );
  return refs;
}

function paint(refs, view) {
  refs.tables.replaceChildren(...readoutTables(view.readouts));
  refs.plotTitle.textContent = view.plot.title;
  paintPlot(refs.plot, view.plot);
  refs.conflict.textContent = view.conflict;
  refs.conflict.parentElement.hidden = view.conflict === '';
  refs.lanesWidth.textContent = view.lanesWidth;
  paintLanes(refs.lanes, view.lanes);
  refs.check.textContent = view.checkWork;
}

function mountControls(refs, data, toy) {
  const fmt = mountChoice(refs.fmt, {
    id: 'fmt', label: 'Number format', value: INITIAL_STATE.fmt,
    options: formatOptions(chipPreset(data, INITIAL_STATE.chip)), onChange: (v) => toy().set({ fmt: v }),
  });
  // A chip change moves the format first when the new chip cannot run it (P3-R12), so no state is ever unrunnable.
  const chip = mountChoice(refs.chip, {
    id: 'chip', label: 'Chip', variant: 'chips', value: INITIAL_STATE.chip,
    options: CHIPS.map((c) => ({ value: c.id, label: c.label })),
    onChange: (id) => {
      const preset = chipPreset(data, id);
      toy().set({ chip: id, fmt: usableFormat(preset, toy().get().fmt) });
      fmt.update(formatOptions(preset));
    },
  });
  const tokens = mountSlider(refs.tokens, {
    id: 'tokens', label: 'Tokens sharing each weight read', values: TOKEN_STOPS, value: INITIAL_STATE.tokens, unit: 'tokens', format: int,
    onInput: (v) => toy().set({ tokens: v }),
  });
  const presets = mountPresetButtons(refs.presets, {
    id: 'tokens-preset', label: 'Token presets', options: TOKEN_PRESETS.map((v) => ({ value: v, label: int(v) })), onPick: (v) => tokens.set(v),
  });
  return [chip, fmt, tokens, presets];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data, ctx);
  let toy = null;
  const controls = mountControls(refs, data, () => toy);
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data)));
  return () => {
    toy.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
