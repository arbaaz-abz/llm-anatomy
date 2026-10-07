// The lesson spec contract and its checks (pure). mountLesson refuses a spec that fails them.
import { SLUG_PATTERN } from './concepts.js';

export const CAPTION_MAX_WORDS = 30;
export const CAPTION_MAX_SENTENCES = 2;
const OPERATORS = /[=+×÷√^*<>≤≥]/; // README lesson 2: formulas live in "Numbers shown" and the math panel

const isText = (x) => typeof x === 'string' && x.trim() !== '';
const isFn = (x) => typeof x === 'function';
export const countWords = (text) => text.trim().split(/\s+/).filter(Boolean).length;
export const countSentences = (text) => text.trim().split(/(?<=[.!?]["”’)]*)\s+/).filter(Boolean).length;

export function captionProblems(caption, step) {
  if (!isText(caption)) return [`step ${step}: caption must be a non-empty string`];
  const problems = [];
  if (countWords(caption) > CAPTION_MAX_WORDS) problems.push(`step ${step}: caption has ${countWords(caption)} words (max ${CAPTION_MAX_WORDS})`);
  if (countSentences(caption) > CAPTION_MAX_SENTENCES) problems.push(`step ${step}: caption has ${countSentences(caption)} sentences (max ${CAPTION_MAX_SENTENCES})`);
  if (OPERATORS.test(caption)) problems.push(`step ${step}: caption contains a formula operator`);
  return problems;
}

function animationProblems(animation) {
  const steps = animation?.steps;
  if (!Array.isArray(steps) || steps.length === 0) return ['animation.steps must be a non-empty array'];
  return [
    ...steps.flatMap((s, i) => captionProblems(s?.caption, i + 1)),
    ...(isFn(animation.render) ? [] : ['animation.render must be a function (index, progress, stage)']),
    ...(isText(animation.label) ? [] : ['animation.label must name the animation for screen readers']),
    ...(animation.standIn === undefined || isText(animation.standIn) ? [] : ['animation.standIn must be a string when given']),
    ...(animation.belowFor === undefined || isFn(animation.belowFor) ? [] : ['animation.belowFor must be a function (index) → paragraphs']),
  ];
}

const furtherOk = (f) => isText(f?.title) && /^https:\/\/\S+$/.test(f?.href ?? '');

export function validateLessonSpec(spec) {
  const checks = [
    [SLUG_PATTERN.test(spec?.slug ?? ''), 'slug must be a kebab-case slug'],
    [isText(spec?.hook), 'hook must be a non-empty string'],
    [Array.isArray(spec?.intuition) && spec.intuition.length >= 2 && spec.intuition.length <= 4 && spec.intuition.every(isText), 'intuition must be 2–4 non-empty paragraphs (spec §4)'],
    [isText(spec?.toy?.title) && isFn(spec?.toy?.mount), 'toy needs a title and mount(el, ctx) → destroy'],
    [Array.isArray(spec?.math?.blocks) && spec.math.blocks.length > 0 && spec.math.blocks.every((b) => isText(b?.tex)), 'math.blocks must be a non-empty list of { tex, note? }'],
    [isText(spec?.facts?.framing), 'facts.framing must be the framing paragraph'],
    [Array.isArray(spec?.facts?.rows) && spec.facts.rows.length > 0 && spec.facts.rows.every((r) => isText(r?.claim)), 'facts.rows must be a non-empty list of { claim, derived? }'],
    [Array.isArray(spec?.takeaways) && spec.takeaways.length === 3 && spec.takeaways.every(isText), 'takeaways must be exactly 3 strings (spec §4)'],
    [Array.isArray(spec?.links?.next) && spec.links.next.every((s) => SLUG_PATTERN.test(s)), 'links.next must be a list of slugs'],
    [Array.isArray(spec?.links?.further) && spec.links.further.length >= 1 && spec.links.further.length <= 3 && spec.links.further.every(furtherOk), 'links.further must hold 1–3 { title, href: https://…, note? }'],
  ];
  return [...checks.filter(([ok]) => !ok).map(([, message]) => message), ...animationProblems(spec?.animation)];
}
