// quantization toy, "Round a block, then shrink a model": mount(host, ctx) → destroy. One state object, one render; every string
// it prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { CHIPS, chipPreset, chipLabel, modelFormatOptions, usableModelFormat } from './hardware.js';
import { INITIAL_STATE, FORMAT_LABEL } from './format.js';
import { BLOCK_SIZES } from './numbers.js';
import { toyView, NOTE_LINE } from './toy-view.js';
import { tryThis } from './try-this.js';
import { output, paintFigure, roundTable, modelTable } from './toy-dom.js';

const rich = (text, ctx) => appendRich(el('p', { className: 'toy-note' }), text, ctx);

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => appendRich(el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` })]), rest, ctx));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const refs = {
    format: el('div'), blockSize: el('div'), outlier: el('div'), modelFormat: el('div'), hw: el('div'), kv: el('div'),
    figure: G.svgEl('svg', { role: 'img' }), roundTable: el('div'), modelTable: el('div'), check: pre,
  };
  host.append(
    el('h4', { textContent: 'Round a block' }),
    rich(NOTE_LINE, ctx),
    el('div', { className: 'toy-controls' }, [refs.format, refs.blockSize, refs.outlier]),
    el('div', { className: 'scroll-x' }, [refs.figure]),
    refs.roundTable,
    el('h4', { textContent: 'Check my work' }), pre,
    el('h4', { textContent: 'Shrink a model' }),
    el('div', { className: 'toy-controls' }, [refs.modelFormat, refs.hw, refs.kv]),
    refs.modelTable,
    ...tryThisList(data, ctx),
  );
  return refs;
}

function paint(refs, view) {
  paintFigure(refs.figure, view.figure);
  refs.roundTable.replaceChildren(roundTable(view.figure.readouts));
  refs.modelTable.replaceChildren(modelTable(view.model));
  refs.check.textContent = view.checkWork;
}

const choices = (pairs) => pairs.map(([value, label]) => ({ value, label }));

function mountControls(refs, data, toy) {
  const set = (patch) => toy().set(patch);
  const format = mountChoice(refs.format, { id: 'format', label: 'Format', variant: 'chips', value: INITIAL_STATE.format, options: choices(Object.entries(FORMAT_LABEL)), onChange: (v) => set({ format: v }) });
  const blockSize = mountChoice(refs.blockSize, { id: 'blockSize', label: 'Weights per scale', variant: 'chips', value: INITIAL_STATE.blockSize, options: choices(BLOCK_SIZES.map((n) => [n, String(n)])), onChange: (v) => set({ blockSize: v }) });
  const outlier = mountChoice(refs.outlier, { id: 'outlier', label: 'Last weight', value: 'outlier', options: choices([['outlier', '2.10 (outlier)'], ['calm', '0.30']]), onChange: (v) => set({ outlier: v === 'outlier' }) });
  const modelFormat = mountChoice(refs.modelFormat, {
    id: 'modelFormat', label: 'Model weights', variant: 'chips', value: INITIAL_STATE.modelFormat, options: modelFormatOptions(chipPreset(data, INITIAL_STATE.hw)), onChange: (v) => set({ modelFormat: v }),
  });
  // A chip change moves the format first when the new chip cannot run it (P3-R12), so no state is ever unrunnable: the old chip
  // still runs the format the new one falls back to, then the chip changes, then the options are re-disabled.
  const hw = mountChoice(refs.hw, {
    id: 'hw', label: 'GPU', variant: 'chips', value: INITIAL_STATE.hw, options: CHIPS.map((c) => ({ value: c.id, label: chipLabel(chipPreset(data, c.id)) })),
    onChange: (id) => {
      const preset = chipPreset(data, id);
      const keep = usableModelFormat(preset, toy().get().modelFormat);
      if (keep !== toy().get().modelFormat) modelFormat.set(keep);
      set({ hw: id });
      modelFormat.update(modelFormatOptions(preset));
    },
  });
  const kv = mountChoice(refs.kv, { id: 'kv', label: 'KV cache', value: INITIAL_STATE.kv, options: choices([['bf16', 'BF16'], ['fp8', 'FP8']]), onChange: (v) => set({ kv: v }) });
  return [format, blockSize, outlier, modelFormat, hw, kv];
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
