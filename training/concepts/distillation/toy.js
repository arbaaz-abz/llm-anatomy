// distillation toy, "Grade the student": mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { METHOD_CHIPS, STUDENT_CHIPS, TEACHER_CHIPS, INITIAL_STATE, toyView, tryThis } from './toy-view.js';
import { TOY_STAGE, paintStage, mountPicks } from './toy-dom.js';

function tryThisList() {
  const items = tryThis().map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host) {
  const refs = {};
  refs.method = el('div');
  refs.student = el('div');
  refs.teacher = el('div');
  refs.svg = G.svgEl('svg', { class: 'toy-stage', role: 'group', 'aria-label': 'teacher and student probabilities for the four candidate tokens', width: TOY_STAGE.w, height: TOY_STAGE.h, viewBox: `0 0 ${TOY_STAGE.w} ${TOY_STAGE.h}` });
  refs.hint = el('p', { className: 'toy-note', textContent: 'On-policy: click, tap or use the arrow keys on a student cell to choose the token the student sampled.' });
  refs.rewards = el('div');
  refs.note = el('p', { className: 'toy-note' });
  refs.readouts = el('div');
  refs.pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  refs.pre.dataset.readout = 'check-work';
  host.append(
    refs.method, refs.student, refs.teacher, el('div', { className: 'scroll-x' }, [refs.svg]), refs.hint,
    refs.readouts, refs.rewards, refs.note,
    el('h4', { textContent: 'Check my work' }), refs.pre, ...tryThisList(),
  );
  return refs;
}

const readoutRow = (label, name, value, sub) => ({ label, sub, cells: [{ value, name }] });

// The labelled numbers for the chosen method, in one table (XT-4); the data-readout names are the ones the specs read.
function readoutRows(view, method) {
  const byMethod = {
    traces: [readoutRow('Loss on the teacher\'s token 56', 'trace-loss', view.traceLoss, '−ln of the student\'s probability')],
    logits: [readoutRow('Forward KL(teacher ‖ student)', 'forward-kl', view.forwardKl)],
    onPolicy: [
      readoutRow(`Reward for the sampled token (${view.sampledToken})`, 'sampled-reward', view.sampledReward),
      readoutRow('Student\'s expected reward', 'expected-reward', view.expectedReward, 'minus the reverse KL'),
    ],
  };
  return [...byMethod[method], readoutRow('Multi-teacher loss', 'multi-teacher-loss', view.multiLoss, 'Σ weight × KL(student ‖ teacher)')];
}

function paint(refs, view, state) {
  const onPolicy = state.method === 'onPolicy';
  refs.hint.hidden = !onPolicy;
  refs.rewards.hidden = !onPolicy;
  refs.readouts.replaceChildren(readoutTable({ head: ['Output', 'Value'], name: 'outputs', rows: readoutRows(view, state.method) }));
  refs.pre.textContent = view.checkWork;
  refs.note.textContent = view.teacherNote;
  refs.note.hidden = view.teacherNote === '';
  refs.rewards.replaceChildren(readoutTable({
    head: ['Sampled token', 'Reward'], name: 'rewards',
    rows: view.rewardRows.map((r) => ({ label: r.token, cells: [{ value: r.value }] })),
  }));
  paintStage(refs.svg, view.model, { onPolicy, picks: refs.picks });
}

function mountControls(refs, set) {
  const choose = (host, id, label, options, key) => mountChoice(host, { id, label, variant: 'chips', value: INITIAL_STATE[key], options, onChange: (v) => set({ [key]: v }) });
  return [
    choose(refs.method, 'method', 'How the student learns', METHOD_CHIPS, 'method'),
    choose(refs.student, 'student', 'Student\'s probabilities', STUDENT_CHIPS, 'student'),
    choose(refs.teacher, 'teacher', 'Who grades', TEACHER_CHIPS, 'teacher'),
  ];
}

export function mount(host) {
  const refs = buildDom(host);
  let toy = null;
  const controls = mountControls(refs, (patch) => toy?.set(patch));
  refs.picks = mountPicks(refs.svg, (sampled) => toy?.set({ sampled }));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s), s));
  return () => {
    toy.destroy();
    refs.picks.destroy();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
