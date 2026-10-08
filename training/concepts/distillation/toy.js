// distillation toy, "Grade the student": mount(el, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { METHOD_CHIPS, STUDENT_CHIPS, TEACHER_CHIPS, INITIAL_STATE, toyView, tryThis } from './toy-view.js';
import { TOY_STAGE, paintStage, bindPicks } from './toy-dom.js';

const output = (name) => {
  const o = el('output');
  o.dataset.readout = name;
  return o;
};
const line = (label, name, refs, extra = []) => {
  const o = output(name);
  refs.outputs[name] = o;
  return el('p', {}, [`${label}: `, o, ...extra]);
};

function tryThisList() {
  const items = tryThis().map(({ prompt, insight, rest }) => el('li', {}, [`${prompt} → `, el('strong', { textContent: `Insight: ${insight}` }), rest]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host) {
  const refs = { outputs: {}, lines: {} };
  refs.method = el('div');
  refs.student = el('div');
  refs.teacher = el('div');
  refs.svg = G.svgEl('svg', { class: 'toy-stage', role: 'group', 'aria-label': 'teacher and student probabilities for the four candidate tokens', width: TOY_STAGE.w, height: TOY_STAGE.h, viewBox: `0 0 ${TOY_STAGE.w} ${TOY_STAGE.h}` });
  refs.hint = el('p', { className: 'toy-note', textContent: 'On-policy: click, tap or use the arrow keys on a student cell, then Enter or Space, to choose the token the student sampled.' });
  refs.rewards = el('div');
  refs.note = el('p', { className: 'toy-note' });
  refs.lines.trace = line('Loss on the teacher\'s token 56, −ln of the student\'s probability', 'trace-loss', refs);
  refs.lines.logits = line('Forward KL(teacher ‖ student)', 'forward-kl', refs);
  refs.lines.sampled = line('Reward for the sampled token', 'sampled-reward', refs, [' (token ', (refs.outputs['sampled-token'] = output('sampled-token')), ')']);
  refs.lines.expected = line('Student\'s expected reward, minus the reverse KL', 'expected-reward', refs);
  const multi = line('Multi-teacher loss, Σ weight × KL(student ‖ teacher)', 'multi-teacher-loss', refs);
  refs.pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  refs.pre.dataset.readout = 'check-work';
  host.append(
    refs.method, refs.student, refs.teacher, el('div', { className: 'scroll-x' }, [refs.svg]), refs.hint,
    refs.lines.trace, refs.lines.logits, refs.lines.sampled, refs.rewards, refs.lines.expected, multi, refs.note,
    el('h4', { textContent: 'Check my work' }), refs.pre, ...tryThisList(),
  );
  return refs;
}

function paint(refs, view, state) {
  const onPolicy = state.method === 'onPolicy';
  refs.lines.trace.hidden = state.method !== 'traces';
  refs.lines.logits.hidden = state.method !== 'logits';
  refs.lines.sampled.hidden = !onPolicy;
  refs.lines.expected.hidden = !onPolicy;
  refs.hint.hidden = !onPolicy;
  refs.rewards.hidden = !onPolicy;
  const text = {
    'trace-loss': view.traceLoss, 'forward-kl': view.forwardKl, 'sampled-reward': view.sampledReward, 'sampled-token': view.sampledToken,
    'expected-reward': view.expectedReward, 'multi-teacher-loss': view.multiLoss,
  };
  Object.entries(text).forEach(([name, value]) => { refs.outputs[name].textContent = value; });
  refs.pre.textContent = view.checkWork;
  refs.note.textContent = view.teacherNote;
  refs.note.hidden = view.teacherNote === '';
  refs.rewards.replaceChildren(readoutTable({
    head: ['Sampled token', 'Reward'], name: 'rewards',
    rows: view.rewardRows.map((r) => ({ label: r.token, sub: r.sampled ? 'sampled' : '', cells: [{ value: r.value }] })),
  }));
  paintStage(refs.svg, view.model, { onPolicy });
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
  const unbind = bindPicks(refs.svg, (sampled) => toy?.set({ sampled }));
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s), s));
  return () => {
    toy.destroy();
    unbind();
    controls.forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
