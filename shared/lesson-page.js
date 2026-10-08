// The lesson page scaffold (spec §4): every concept page is these eight sections, in this order.
// A concept module supplies a LESSON spec (shared/lesson-spec.js) and calls mountLesson from mount(el, ctx).
import { mountStepper } from './ui/stepper.js';
import { mountMathPanel, linkMathToStage } from './ui/math-panel.js';
import { el } from './ui/dom.js';
import { validateLessonSpec } from './lesson-spec.js';
import { parseRichText } from './rich-text.js';
import { fillClaim } from './claims.js';

// Appends rich text ([[slug]] → lesson link, `x` → code) to `node` and returns it. Never parses HTML.
export function appendRich(node, text, ctx) {
  for (const seg of parseRichText(text)) {
    if (seg.type === 'text') node.append(seg.text);
    else if (seg.type === 'code') node.append(el('code', { textContent: seg.text }));
    else {
      const url = ctx?.href?.(seg.slug);
      const code = el('code', { textContent: seg.slug });
      node.append(url ? el('a', { href: url }, [code]) : code);
    }
  }
  return node;
}

const para = (text, ctx, className = '') => appendRich(el('p', { className }), text, ctx);

function section(name, heading) {
  const node = el('section', { className: `lesson-section lesson-${name}` });
  node.dataset.section = name;
  if (heading) node.append(el('h3', { textContent: heading }));
  return node;
}

function runCleanup(fn) {
  try {
    fn();
  } catch (error) {
    console.error('Lesson cleanup failed', error);
  }
}

function factRow(row, ctx) {
  const filled = fillClaim(row.claim, ctx.data);
  const li = el('li', { className: 'fact-row' });
  if (row.derived) li.dataset.derived = '';
  filled.segments.forEach((seg) => (seg.type === 'text'
    ? appendRich(li, seg.text, ctx)
    : li.append(el('span', { className: 'fact', textContent: seg.text, ...(seg.note ? { title: seg.note } : {}) }))));
  filled.sources.forEach((href, i) => li.append(' ', el('a', {
    className: 'fact-source', href, target: '_blank', rel: 'noopener', textContent: filled.sources.length > 1 ? `source ${i + 1}` : 'source',
  })));
  if (filled.reported) li.append(' ', el('span', { className: 'fact-reported', textContent: 'reported' }));
  if (filled.missing.length) {
    console.error(`Lesson fact missing from data/*.json: ${filled.missing.join(', ')}`);
    li.append(' ', el('span', { className: 'fact-missing', textContent: `missing fact: ${filled.missing.join(', ')}` }));
  }
  return li;
}

function buildShells(spec, ctx) {
  const hook = el('header', { className: 'lesson-head' }, [el('h2', { textContent: ctx.concept?.title ?? spec.slug }), para(spec.hook, ctx, 'lesson-hook')]);
  hook.dataset.section = 'hook';
  const intuition = section('intuition', 'The idea');
  intuition.append(...spec.intuition.map((p) => para(p, ctx)), ...(spec.intuitionNote ? [para(spec.intuitionNote, ctx, 'lesson-note')] : []));
  const animation = section('animation', 'Step through it');
  const stepperHost = el('div', { className: 'lesson-stepper' });
  const below = el('div', { className: 'below-stage', ariaLive: 'polite' });
  animation.append(stepperHost, ...(spec.animation.standIn ? [para(spec.animation.standIn, ctx, 'stand-in')] : []), below);
  const toy = section('toy', spec.toy.title);
  const toyHost = el('div', { className: 'toy' });
  toy.append(...(spec.toy.intro ? [para(spec.toy.intro, ctx, 'toy-intro')] : []), toyHost);
  const math = section('math');
  const mathHost = el('div');
  math.append(mathHost, ...(spec.math.notes ?? []).map((n) => para(n, ctx, 'math-note')));
  return { nodes: [hook, intuition, animation, toy, math, factsSection(spec, ctx), takeawaysSection(spec, ctx), linksSection(spec, ctx)], stepperHost, below, toyHost, mathHost };
}

function factsSection(spec, ctx) {
  const node = section('facts', 'In today\'s models (Oct 2026)');
  node.append(para(spec.facts.framing, ctx), el('ul', { className: 'fact-rows' }, spec.facts.rows.map((r) => factRow(r, ctx))), ...(spec.facts.prose ?? []).map((p) => para(p, ctx)));
  return node;
}

function takeawaysSection(spec, ctx) {
  const node = section('takeaways', 'Takeaways');
  node.append(el('ol', {}, spec.takeaways.map((t) => appendRich(el('li'), t, ctx))));
  return node;
}

function linksSection(spec, ctx) {
  const node = section('links', 'Where next');
  const lessonList = (label, slugs) => (slugs.length ? [el('h4', { textContent: label }), appendRich(el('p'), slugs.map((s) => `[[${s}]]`).join(' · '), ctx)] : []);
  const further = spec.links.further.map((f) => el('li', {}, [el('a', { href: f.href, target: '_blank', rel: 'noopener', textContent: f.title }), ...(f.note ? [`: ${f.note}`] : [])]));
  node.append(...lessonList('Before this', ctx.concept?.prereqs ?? []), ...lessonList('Next', spec.links.next), el('h4', { textContent: 'Go deeper' }), el('ul', {}, further));
  return node;
}

function mountAnimation(spec, ctx, { stepperHost, below }) {
  let shownBelow = -1;
  const render = (index, progress, stage) => {
    spec.animation.render(index, progress, stage);
    if (!spec.animation.belowFor || index === shownBelow) return;
    shownBelow = index;
    below.replaceChildren(...spec.animation.belowFor(index).map((p) => para(p, ctx)));
  };
  const stepper = mountStepper(stepperHost, { steps: spec.animation.steps, render, label: spec.animation.label });
  return stepper;
}

function mountToy(spec, ctx, host) {
  try {
    const destroy = spec.toy.mount(host, ctx);
    return typeof destroy === 'function' ? destroy : null;
  } catch (error) {
    console.error(`Lesson "${spec.slug}": the toy failed to start`, error);
    host.replaceChildren(el('div', { className: 'load-error', role: 'alert' }, [
      el('p', { textContent: 'This toy did not load. The rest of the lesson still works; please report it on GitHub with your browser name.' }),
    ]));
    return null;
  }
}

export function mountLesson(root, ctx, spec) {
  const problems = validateLessonSpec(spec);
  if (problems.length) throw new Error(`Lesson "${spec?.slug}" spec is invalid: ${problems.join('; ')}`);
  const shells = buildShells(spec, ctx);
  root.append(...shells.nodes); // in the document before anything measures text
  const stepper = mountAnimation(spec, ctx, shells);
  const math = mountMathPanel(shells.mathHost, { summary: 'Show me the math', blocks: spec.math.blocks });
  const cleanups = [() => stepper.destroy(), () => math.destroy(), linkMathToStage(shells.mathHost, stepper.stage), mountToy(spec, ctx, shells.toyHost)].filter(Boolean);
  return () => [...cleanups].reverse().forEach(runCleanup);
}
