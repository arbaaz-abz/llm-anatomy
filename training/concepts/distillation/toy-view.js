// distillation toy view model (pure, no DOM): state → the numbers and every string the toy prints.
// Losses and rewards come from math/lm.js and math/distill.js; nothing is typed.
import { tokenLoss, klDivergence } from '@math/lm.js';
import { opdTokenReward, multiTeacherLoss } from '@math/distill.js';
import { CANDIDATES, TEACHERS, STUDENTS, WRONG_TOKEN } from './numbers.js';
import { checkWork, fixed, signed, prob } from './format.js';

export const METHOD_CHIPS = Object.freeze([
  { value: 'traces', label: 'teacher\'s text' },
  { value: 'logits', label: 'teacher\'s probabilities' },
  { value: 'onPolicy', label: 'on-policy' },
]);
export const STUDENT_CHIPS = Object.freeze([
  { value: 'unsure', label: `unsure [${STUDENTS.unsure.map(prob).join(', ')}]` },
  { value: 'wrong', label: `confident and wrong [${STUDENTS.wrong.map(prob).join(', ')}]` },
  { value: 'near', label: `close to the math teacher [${STUDENTS.near.map(prob).join(', ')}]` },
]);
export const TEACHER_CHIPS = Object.freeze([
  { value: 'math', label: `math teacher [${TEACHERS.math.map(prob).join(', ')}]` },
  { value: 'chat', label: `chat teacher [${TEACHERS.chat.map(prob).join(', ')}]` },
  { value: 'mix', label: '50/50 mix' },
]);

export const INITIAL_STATE = Object.freeze({ method: 'onPolicy', student: 'unsure', sampled: WRONG_TOKEN, teacher: 'math' });

const MATH = Object.freeze({ name: 'math teacher', short: 'math', probs: TEACHERS.math });
const CHAT = Object.freeze({ name: 'chat teacher', short: 'chat', probs: TEACHERS.chat });
const TEACHER_SETS = Object.freeze({ math: [[MATH], [1]], chat: [[CHAT], [1]], mix: [[MATH, CHAT], [0.5, 0.5]] });
const sum = (values) => values.reduce((total, v) => total + v, 0);

function validate(state) {
  if (!METHOD_CHIPS.some((c) => c.value === state.method)) throw new RangeError(`toyView: unknown method "${state.method}"`);
  if (!(state.student in STUDENTS)) throw new RangeError(`toyView: unknown student "${state.student}"`);
  if (!(state.teacher in TEACHER_SETS)) throw new RangeError(`toyView: unknown teacher "${state.teacher}"`);
  if (!Number.isInteger(state.sampled) || state.sampled < 0 || state.sampled >= CANDIDATES.length) throw new RangeError(`toyView: sampled must be 0–3, got ${state.sampled}`);
}

// Every number the toy prints, for one state. The weighted forms reduce to the plain ones for a single teacher at weight 1.
export function modelFor(state) {
  validate(state);
  const student = STUDENTS[state.student];
  const [teachers, weights] = TEACHER_SETS[state.teacher];
  const rewards = student.map((p, j) => sum(teachers.map((t, i) => weights[i] * opdTokenReward(t.probs[j], p))));
  const multiLoss = multiTeacherLoss(student, teachers.map((t) => t.probs), weights);
  return {
    method: state.method, student, teachers, weights, sampled: state.sampled,
    traceLoss: tokenLoss(student[0]),
    forwardKl: sum(teachers.map((t, i) => weights[i] * klDivergence(t.probs, student))),
    rewards,
    expectedReward: -multiLoss, // Σ p_S · reward = −Σ w_i KL(student ‖ teacher_i)
    multiLoss,
  };
}

const rewardRows = (model) => CANDIDATES.map((token, i) => ({ token, value: signed(model.rewards[i], 3), sampled: i === model.sampled }));

// Everything the toy prints as text, by readout name.
export function toyView(state) {
  const model = modelFor(state);
  const mix = model.teachers.length > 1;
  return {
    model,
    traceLoss: fixed(model.traceLoss, 3),
    forwardKl: fixed(model.forwardKl, 3),
    sampledReward: signed(model.rewards[model.sampled], 3),
    sampledToken: CANDIDATES[model.sampled],
    rewardRows: rewardRows(model),
    expectedReward: signed(model.expectedReward, 3),
    multiLoss: fixed(model.multiLoss, 3),
    teacherNote: mix ? 'Two teachers at weight ½ each: rewards and the forward KL are the weighted averages of each teacher\'s.' : '',
    checkWork: checkWork(model),
  };
}

// The three try-this items (storyboard §6): numbers computed from the same model, text pinned in the page test.
export function tryThis() {
  const at = (student, teacher, method = 'onPolicy') => modelFor({ ...INITIAL_STATE, method, student, teacher });
  const unsure = at('unsure', 'math');
  const near = at('near', 'math');
  const wrong = at('wrong', 'math');
  const rewards = CANDIDATES.map((token, i) => `${token} ${signed(unsure.rewards[i], 3)}`).join(', ');
  const nearLoss = ['math', 'chat', 'mix'].map((teacher) => fixed(at('near', teacher).multiLoss, 3));
  return [
    {
      prompt: `With on-policy, unsure and the math teacher, select each sampled token: ${rewards}; the expected reward is ${signed(unsure.expectedReward, 3)}.`,
      insight: 'the teacher grades the student\'s own choices, wrong ones included.',
      rest: ' Every token the student might write gets its own signed grade, which is what fixes the blind spot of frame 4.',
    },
    {
      prompt: `Switch student to close to the math teacher: every reward shrinks toward 0 (${CANDIDATES[0]} ${signed(near.rewards[0], 3)}, ${CANDIDATES[1]} ${signed(near.rewards[1], 3)}) and the expected reward is ${signed(near.expectedReward, 3)}. `
        + `Then switch to confident and wrong: ${CANDIDATES[1]} falls to ${signed(wrong.rewards[1], 3)} and the expected reward to ${signed(wrong.expectedReward, 3)}.`,
      insight: 'the signal is largest where the student is most wrong and vanishes as it matches the teacher.',
      rest: ' The distillation reward stops pushing once the student matches its teacher; going past it needs another signal, such as MiMo-V2-Flash\'s outcome advantage (see the math panel).',
    },
    {
      prompt: `Keep close to the math teacher and switch teacher: math ${nearLoss[0]}, chat ${nearLoss[1]}, 50/50 mix ${nearLoss[2]}.`,
      insight: 'averaging teachers pulls the student toward a blend.',
      rest: ' A student that already matches the math specialist is still pushed away from it by the mix, which is why Kimi K3 picks one teacher per prompt by domain and effort level; DeepSeek-V4 sums its teachers\' KLs with weights.',
    },
  ];
}
