// Supervised fine-tuning's loss mask (sft §11). Pure: no DOM, inputs never mutated.
// Every bad argument throws RangeError('<fn>: <arg> must be …'). The loss itself is meanLoss(probs, lossMask(...)) in lm.js.
import { deepFreeze } from './core.js';

const KINDS = Object.freeze(['template', 'user', 'assistant', 'error', 'observation']);
export const MASK_DEFAULTS = deepFreeze({ maskPrompt: true, maskObservation: true, maskError: false });

function checkSegments(fn, segments) {
  if (!Array.isArray(segments)) throw new RangeError(`${fn}: segments must be an array of { kind, tokens }`);
  segments.forEach((s, i) => {
    if (!KINDS.includes(s?.kind)) throw new RangeError(`${fn}: segments[${i}].kind must be one of ${KINDS.join(', ')}, got ${s?.kind}`);
    if (!Array.isArray(s.tokens) || s.tokens.some((t) => typeof t !== 'string')) throw new RangeError(`${fn}: segments[${i}].tokens must be an array of strings`);
  });
}

function checkOptions(fn, options) {
  if (options === null || typeof options !== 'object') throw new RangeError(`${fn}: options must be an object`);
  const merged = { ...MASK_DEFAULTS, ...options };
  Object.keys(MASK_DEFAULTS).forEach((key) => {
    if (typeof merged[key] !== 'boolean') throw new RangeError(`${fn}: ${key} must be true or false, got ${merged[key]}`);
  });
  return merged;
}

// Is a token of this kind a loss target? Assistant tokens always are.
const isTrained = (kind, { maskPrompt, maskObservation, maskError }) => {
  if (kind === 'template' || kind === 'user') return !maskPrompt;
  if (kind === 'observation') return !maskObservation;
  if (kind === 'error') return !maskError;
  return true;
};

// One boolean per token (true = trained, false = masked).
//   the §5 transcript → 26 entries, 15 true: 00000000111111111111100011
//   [{ kind: 'user', tokens: ['hi'] }, { kind: 'assistant', tokens: ['hello'] }] → [false, true]
export function lossMask(segments, options = {}) {
  checkSegments('lossMask', segments);
  const o = checkOptions('lossMask', options);
  return segments.flatMap((s) => s.tokens.map(() => isTrained(s.kind, o)));
}

// Counts for the toy and the check box: { total, trained, masked, byKind: { [kind]: { trained, masked } } }.
//   §5 transcript, defaults → 26, 15, 11 · maskError: true → 26, 10, 16
export function maskSummary(segments, options = {}) {
  checkSegments('maskSummary', segments);
  const o = checkOptions('maskSummary', options);
  const byKind = Object.fromEntries(KINDS.map((k) => [k, { trained: 0, masked: 0 }]));
  segments.forEach((s) => {
    const key = isTrained(s.kind, o) ? 'trained' : 'masked';
    byKind[s.kind][key] += s.tokens.length;
  });
  const trained = KINDS.reduce((n, k) => n + byKind[k].trained, 0);
  const masked = KINDS.reduce((n, k) => n + byKind[k].masked, 0);
  return { total: trained + masked, trained, masked, byKind };
}
