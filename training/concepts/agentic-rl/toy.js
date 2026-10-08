// agentic-rl toy, "Fix the mismatch." (storyboard §6): mount(el, ctx) → destroy. One state object, one render; every string it
// prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { CORRECTIONS, INITIAL_STATE, LAMBDAS, PRECISIONS, toyView, tryThis } from './toy-view.js';
import { mountTable, mountTimeline } from './toy-dom.js';

function tryThisList() {
  const items = tryThis().map(({ prompt, insight, rest }) => el('li', {}, [prompt, ' → ', el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

const options = (list) => list.map(({ value, label }) => ({ value, label }));

function mountControls(host, set) {
  const parts = ['correction', 'precision', 'lambda'].map(() => el('div'));
  host.append(...parts);
  const controls = [
    mountChoice(parts[0], { id: 'correction', label: 'How the trainer treats the mismatch', options: options(CORRECTIONS), value: INITIAL_STATE.correction, variant: 'chips', onChange: (correction) => set({ correction }) }),
    mountChoice(parts[1], { id: 'precision', label: 'Rollout and trainer number format', options: options(PRECISIONS), value: INITIAL_STATE.precision, variant: 'chips', onChange: (precision) => set({ precision }) }),
    mountChoice(parts[2], { id: 'lambda', label: 'Update when this share of episodes is done', options: options(LAMBDAS), value: INITIAL_STATE.lambda, variant: 'chips', onChange: (lambda) => set({ lambda }) }),
  ];
  return () => controls.forEach((c) => c.destroy());
}

const tableSpec = (head, name, rows) => readoutTable({ head, name, rows: rows.map(([label, sub, value, cellName]) => ({ label, sub, cells: [{ value, name: cellName }] })) });

function paintReadouts(host, view) {
  const inspector = readoutTable({
    head: ['Selected token', 'Value'], name: 'inspector',
    rows: view.inspector.map(([name, label, value]) => ({ label, cells: [{ value, name }] })),
  });
  const timeline = tableSpec(['Rollout timeline', 'Value'], 'timeline', [
    ['Iteration length', 'when the trainer updates', view.timeline.iteration, 'iteration'],
    ['Utilization', 'busy ÷ (8 × iteration)', view.timeline.utilization, 'utilization'],
    ['Finishing under newer weights', 'off-policy rows', view.timeline.carried, 'carried'],
  ]);
  const masked = tableSpec(['Across the group', 'Value'], 'group', [['Masked tokens', 'out of all policy tokens', view.masked, 'masked']]);
  host.replaceChildren(inspector, masked, timeline);
}

export function mount(host) {
  const readouts = el('div', { className: 'toy-tables' });
  const note = el('p', { className: 'toy-intro', textContent: toyView(INITIAL_STATE).note });
  let toy = null;
  const set = (patch) => toy?.set(patch);
  const unmountControls = mountControls(host, set);
  const table = mountTable((row, pos) => set({ row, pos }), INITIAL_STATE);
  const timeline = mountTimeline();
  host.append(el('div', { className: 'scroll-x' }, [table.node]), el('div', { className: 'scroll-x' }, [timeline.node]), note, readouts, ...tryThisList());
  toy = createToyState(INITIAL_STATE, (s) => {
    table.paint(s);
    timeline.paint(s);
    paintReadouts(readouts, toyView(s));
  });
  return () => {
    toy.destroy();
    unmountControls();
    table.destroy();
    host.replaceChildren();
  };
}
