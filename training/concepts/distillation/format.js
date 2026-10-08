// distillation formatters and "Check my work" (pure, no DOM; in the coverage gate). Real minus everywhere.
import { CANDIDATES } from './numbers.js';

export const MINUS = '−';

// fixed(-1.7918, 3) → '−1.792'; a value that rounds to zero prints '0.000', never '−0.000'.
export function fixed(value, digits) {
  const text = Math.abs(value).toFixed(digits);
  return (value < 0 && Number(text) !== 0 ? MINUS : '') + text;
}

// signed(0.8109, 3) → '+0.811' · signed(-1.7918, 3) → '−1.792' · signed(0, 3) → '0.000'
export function signed(value, digits) {
  const text = fixed(value, digits);
  return value > 0 && Number(text) !== 0 ? `+${text}` : text;
}

export const prob = (p) => p.toFixed(2);
const wrapNeg = (value) => (value < 0 ? `(${fixed(value, 3)})` : fixed(value, 3));
const ln = (p) => Math.log(p);
const HALF = '½';

function klTerms(p, q) {
  return p.map((pi, i) => pi * ln(pi / q[i]));
}
const sum = (values) => values.reduce((total, v) => total + v, 0);
const ROUNDING_TOLERANCE = 5e-4;
// Terms print at 3 decimals; when their rounded sum is a thousandth off the exact value, the line says so.
const roundingNote = (terms, total) => (Math.abs(sum(terms.map((v) => Number(fixed(v, 3)))) - total) > ROUNDING_TOLERANCE ? ' (terms rounded)' : '');

function singleOnPolicy({ student, teachers, sampled, rewards, expectedReward }) {
  const [teacher] = teachers;
  const t = teacher.probs[sampled];
  const s = student[sampled];
  const terms = klTerms(student, teacher.probs);
  const termText = student.map((p, i) => `${prob(p)} ln(${prob(p)}/${prob(teacher.probs[i])})`).join(' + ');
  return [
    `reward(${CANDIDATES[sampled]}) = ln ${prob(t)} − ln ${prob(s)} = ${fixed(ln(t), 3)} − ${wrapNeg(ln(s))} = ${fixed(rewards[sampled], 3)}`,
    'expected reward = −KL(student ‖ teacher)',
    `  = −(${termText})`,
    `  = −(${terms.map((v) => fixed(v, 3)).join(' + ')}) = ${fixed(expectedReward, 3)}${roundingNote(terms, -expectedReward)}`,
  ];
}

function mixOnPolicy({ student, teachers, sampled, rewards, expectedReward }) {
  const parts = teachers.map((teacher) => ln(teacher.probs[sampled]) - ln(student[sampled]));
  const kls = teachers.map((teacher) => sum(klTerms(student, teacher.probs)));
  const names = teachers.map((teacher) => teacher.short);
  return [
    `reward(${CANDIDATES[sampled]}) = ${parts.map((v) => `${HALF} × (${fixed(v, 3)})`).join(' + ')} = ${fixed(rewards[sampled], 3)}`,
    `expected reward = −(${names.map((n) => `${HALF} KL(student ‖ ${n})`).join(' + ')})`,
    `  = −(${kls.map((v) => `${HALF} × ${fixed(v, 3)}`).join(' + ')}) = ${fixed(expectedReward, 3)}`,
  ];
}

function logitLines({ student, teachers, forwardKl }) {
  if (teachers.length > 1) {
    const kls = teachers.map((teacher) => sum(klTerms(teacher.probs, student)));
    return [
      `KL(teacher ‖ student) = ${teachers.map((t) => `${HALF} KL(${t.short} ‖ student)`).join(' + ')}`,
      `  = ${kls.map((v) => `${HALF} × ${fixed(v, 3)}`).join(' + ')} = ${fixed(forwardKl, 3)}`,
    ];
  }
  const [teacher] = teachers;
  const terms = klTerms(teacher.probs, student);
  const termText = teacher.probs.map((p, i) => `${prob(p)} ln(${prob(p)}/${prob(student[i])})`).join(' + ');
  const signedTerms = terms.map((v, i) => (i === 0 ? fixed(v, 3) : `${v < 0 ? MINUS : '+'} ${Math.abs(v).toFixed(3)}`)).join(' ');
  return ['KL(teacher ‖ student)', `  = ${termText}`, `  = ${signedTerms} = ${fixed(forwardKl, 3)}${roundingNote(terms, forwardKl)}`];
}

// The "Check my work" text for one toy state's model (toy-view.js modelFor): templated from the same numbers the toy prints.
export function checkWork(model) {
  if (model.method === 'traces') return `loss = −ln p_student(${CANDIDATES[0]}) = −ln ${prob(model.student[0])} = ${fixed(model.traceLoss, 3)}`;
  if (model.method === 'logits') return logitLines(model).join('\n');
  return (model.teachers.length > 1 ? mixOnPolicy(model) : singleOnPolicy(model)).join('\n');
}
