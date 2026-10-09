// prefill-decode toy, "Step-time calculator": mount(host, ctx) → destroy. One state object, one render; every string it
// prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import { formatInt } from '@math/core.js';
import { GPUS, WEIGHT_FORMATS } from './hardware.js';
import { INITIAL_STATE, PROMPT_STOPS, CONTEXT_STOPS, usersLabel } from './format.js';
import { toyView, gpuNote } from './toy-view.js';
import { tryThis } from './try-this.js';
import { output, readoutTables, paintStepBar, paintMemoryBar, paintCurve, figureBlocks } from './toy-dom.js';

const PHASES = Object.freeze([{ value: 'prefill', label: 'prefill' }, { value: 'decode', label: 'decode' }]);
const tokensLabel = (v) => `${formatInt(v)} token${v === 1 ? '' : 's'}`;

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
  const usersNote = output('users-note');
  const sliders = { promptTokens: el('div'), users: el('div'), context: el('div') };
  const refs = {
    phase: el('div'), hw: el('div'), weights: el('div'), ...sliders,
    // A slider that does not apply to the phase is hidden through a plain wrapper (.slider sets its own display).
    boxes: Object.fromEntries(Object.entries(sliders).map(([k, node]) => [k, el('div', {}, [node])])),
    usersNote, usersNoteP: el('p', { className: 'toy-note' }, [usersNote]),
    fit: el('p'), tables: el('div', { className: 'toy-readouts' }), figures: figureBlocks(), check: pre,
  };
  host.append(
    el('div', { className: 'toy-controls' }, [refs.phase, refs.hw, el('p', { className: 'toy-note' }, [output('gpu-memory', { textContent: gpuNote(data) })]), refs.weights, refs.boxes.promptTokens, refs.boxes.users, refs.boxes.context, refs.usersNoteP]),
    refs.fit, refs.tables, ...refs.figures.step.nodes, ...refs.figures.memory.nodes, ...refs.figures.curve.nodes,
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data, ctx),
  );
  return refs;
}

// "does not fit on one GPU: see [[serving-calculator]]" as a readout with the lesson link; absent while the weights fit.
function paintFit(p, fit, ctx) {
  if (!fit) { p.replaceChildren(); p.hidden = true; return; }
  p.replaceChildren(appendRich(output('fit'), fit, ctx));
  p.hidden = false;
}

function paintFigure(block, spec, paint) {
  block.nodes.forEach((n) => { n.hidden = !spec; });
  block.caption.textContent = spec?.caption ?? '';
  paint(block.svg, spec);
}

// The users slider's stops end at the max that fits, which moves with GPU, format and context: remount it when they change.
function syncUsers(refs, controls, users, set) {
  const key = users.stops.join(',');
  if (controls.users && controls.usersKey === key) return;
  controls.users?.destroy();
  const max = users.stops.at(-1);
  controls.users = mountSlider(refs.users, {
    id: 'users', label: 'Users in the batch (decode)', values: users.stops, value: users.value,
    format: (v) => usersLabel(v, max), onInput: (v) => set({ users: v }),
  });
  controls.usersKey = key;
}

function paint(refs, controls, state, view, ctx, set) {
  const decode = state.phase === 'decode';
  refs.boxes.promptTokens.hidden = decode;
  refs.boxes.users.hidden = !decode;
  refs.boxes.context.hidden = !decode;
  syncUsers(refs, controls, view.users, set);
  refs.usersNote.textContent = view.usersNote;
  refs.usersNoteP.hidden = view.usersNote === '';
  paintFit(refs.fit, view.fit, ctx);
  refs.tables.replaceChildren(...readoutTables(view.tables));
  paintFigure(refs.figures.step, view.stepBar, paintStepBar);
  paintFigure(refs.figures.memory, view.memoryBar, paintMemoryBar);
  paintFigure(refs.figures.curve, view.curve, paintCurve);
  refs.check.textContent = view.checkWork;
}

function mountControls(refs, data, set) {
  return [
    mountChoice(refs.phase, { id: 'phase', label: 'Phase', options: PHASES, value: INITIAL_STATE.phase, onChange: (v) => set({ phase: v }) }),
    mountChoice(refs.hw, {
      id: 'hw', label: 'GPU', variant: 'chips', value: INITIAL_STATE.hw,
      options: GPUS.map((g) => ({ value: g.id, label: g.label })), onChange: (v) => set({ hw: v }),
    }),
    mountChoice(refs.weights, {
      id: 'weights', label: 'Weight format', variant: 'chips', value: INITIAL_STATE.weights,
      options: Object.entries(WEIGHT_FORMATS).map(([value, f]) => ({ value, label: f.label })), onChange: (v) => set({ weights: v }),
    }),
    mountSlider(refs.promptTokens, {
      id: 'promptTokens', label: 'Prompt tokens (prefill)', values: PROMPT_STOPS, value: INITIAL_STATE.promptTokens, format: tokensLabel,
      onInput: (v) => set({ promptTokens: v }),
    }),
    mountSlider(refs.context, {
      id: 'context', label: 'Context per user (decode)', values: CONTEXT_STOPS, value: INITIAL_STATE.context, format: tokensLabel,
      onInput: (v) => set({ context: v }),
    }),
  ];
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data, ctx);
  const users = { users: null, usersKey: null };
  let toy = null;
  const set = (patch) => toy?.set(patch);
  const controls = mountControls(refs, data, set);
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, users, s, toyView(s, data), ctx, set));
  return () => {
    toy.destroy();
    users.users?.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
