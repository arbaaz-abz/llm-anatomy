// Distillation (distillation §11): the on-policy reward for one sampled token and the multi-teacher loss.
// Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('<fn>: <arg> must be …').
import { klDivergence } from './lm.js';

const WEIGHT_SUM_TOLERANCE = 1e-9;

function requirePositiveProbability(fn, name, value) {
  if (typeof value !== 'number' || !(value > 0 && value <= 1)) throw new RangeError(`${fn}: ${name} must be a probability with 0 < p ≤ 1, got ${value}`);
}

// On-policy distillation reward for one token the student sampled: ln pTeacher − ln pStudent, clipped to ±clip
// (Kimi K3 clips; the default is no clip).
//   (0.90, 0.40) → 0.8109 · (0.05, 0.30) → −1.7918 · (0.97, 0.97) → 0 · (0.05, 0.30, { clip: 1 }) → −1
export function opdTokenReward(pTeacher, pStudent, { clip = Infinity } = {}) {
  requirePositiveProbability('opdTokenReward', 'pTeacher', pTeacher);
  requirePositiveProbability('opdTokenReward', 'pStudent', pStudent);
  if (typeof clip !== 'number' || !(clip > 0)) throw new RangeError(`opdTokenReward: clip must be a number > 0, got ${clip}`);
  const reward = Math.log(pTeacher) - Math.log(pStudent);
  return Math.min(Math.max(reward, -clip), clip) || 0; // `|| 0` turns −0 into 0
}

function checkWeights(teachers, weights) {
  if (!Array.isArray(teachers) || teachers.length === 0) throw new RangeError('multiTeacherLoss: teachers must be a non-empty array');
  if (!Array.isArray(weights) || weights.length !== teachers.length) {
    throw new RangeError(`multiTeacherLoss: weights must have one entry per teacher (${teachers.length}), got ${weights?.length}`);
  }
  weights.forEach((w, i) => {
    if (typeof w !== 'number' || !(w >= 0)) throw new RangeError(`multiTeacherLoss: weights[${i}] must be a number ≥ 0, got ${w}`);
  });
  const total = weights.reduce((sum, w) => sum + w, 0);
  if (Math.abs(total - 1) > WEIGHT_SUM_TOLERANCE) throw new RangeError(`multiTeacherLoss: weights must sum to 1, got ${total}`);
}

// Σ w_i · KL(student ‖ teacher_i), the DeepSeek-V4 form.
//   ([0.85,0.07,0.05,0.03], [[0.90,0.05,0.03,0.02]], [1]) → 0.0127
//   ([0.85,0.07,0.05,0.03], [[0.90,0.05,0.03,0.02],[0.60,0.20,0.10,0.10]], [0.5,0.5]) → 0.0822
export function multiTeacherLoss(student, teachers, weights) {
  checkWeights(teachers, weights);
  return teachers.reduce((sum, teacher, i) => sum + (weights[i] === 0 ? 0 : weights[i] * klDivergence(student, teacher)), 0);
}
