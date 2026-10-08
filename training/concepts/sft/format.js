// Pure formatters for the sft page: the toy's start state, the "Check my work" text (storyboard §6) and the strings the
// stage and the toy print. No DOM; every count comes from math/sft.js and every share from formatShare.
import { maskSummary } from '@math/sft.js';
import { formatShare } from '@shared/glyphs.js';
import { SEGMENTS, TOTAL_TOKENS, FOLLOWED } from './numbers.js';

// The toy's start: prompt and template tags masked, the tool's reply masked, the mistake kept (frame 6's state).
export const INITIAL_STATE = Object.freeze({ maskPrompt: true, maskObservation: true, maskError: false, template: 'generic', selected: FOLLOWED });

export const KIND_ORDER = Object.freeze(['template', 'user', 'assistant', 'error', 'observation']);
const KIND_PHRASE = Object.freeze({
  template: 'a template tag',
  user: 'a user token',
  assistant: 'an assistant token',
  error: 'a token of the kept mistake',
  observation: 'a tool-reply token',
});

export const optionsOf = (state) => ({ maskPrompt: state.maskPrompt, maskObservation: state.maskObservation, maskError: state.maskError });
export const summaryOf = (state) => maskSummary(SEGMENTS, optionsOf(state));
export const shareText = (trained, total = TOTAL_TOKENS) => formatShare(trained / total);
export const trainedText = (summary) => `${summary.trained} of ${summary.total}`;

// "assistant 10 + error 5": the kinds that have tokens on that side, in the storyboard's kind order.
function termsOf(summary, side) {
  const terms = KIND_ORDER.filter((kind) => summary.byKind[kind][side] > 0).map((kind) => `${kind} ${summary.byKind[kind][side]}`);
  return terms.length === 0 ? 'none' : terms.join(' + ');
}

// The "Check my work" box, for any toggle state.
export function checkWork(state) {
  const s = summaryOf(state);
  return [
    `trained = ${termsOf(s, 'trained')} = ${s.trained}`,
    `masked  = ${termsOf(s, 'masked')} = ${s.masked}`,
    `share   = ${s.trained} ÷ ${s.total} = ${shareText(s.trained, s.total)}`,
  ].join('\n');
}

// Per-kind rows for the breakdown table: "trained/total".
export function kindRows(state) {
  const { byKind } = summaryOf(state);
  return KIND_ORDER.map((kind) => {
    const { trained, masked } = byKind[kind];
    return { kind, text: `${trained}/${trained + masked}` };
  });
}

// What the selected token is, and whether it is trained (visible without hover).
export function selectionText(chip, trained) {
  return `Selected "${chip.raw}", ${KIND_PHRASE[chip.kind]}: ${trained ? 'trained' : 'masked (context only)'}.`;
}
