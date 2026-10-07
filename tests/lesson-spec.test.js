import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateLessonSpec, captionProblems, countWords, countSentences } from '../shared/lesson-spec.js';

const noop = () => {};
const VALID = Object.freeze({
  slug: 'fixture',
  hook: 'Why does this page exist?',
  intuition: ['One.', 'Two.'],
  animation: { label: 'Fixture', steps: [{ caption: 'A short caption.' }], render: noop },
  toy: { title: 'Try it', mount: noop },
  math: { blocks: [{ tex: 'x = 1' }] },
  facts: { framing: 'Framing.', rows: [{ claim: 'GPT-3 had {gpt-3.layers} layers.' }] },
  takeaways: ['a', 'b', 'c'],
  links: { next: ['attention'], further: [{ title: 'Elsewhere', href: 'https://example.org/x' }] },
});

test('a complete spec has no problems', () => {
  assert.deepEqual(validateLessonSpec(VALID), []);
});

test('the approved storyboard captions pass the caption rules (README lesson 2)', () => {
  const approved = [
    'A model is this block repeated N times, each with its own weights: talk, think, talk, think. "Layers: 61" on a model card counts these blocks.',
    'A score is the query dotted with a key. Against the four keys, "sat" scores −1.0, 3.0, 0.5 and −0.75; the biggest is "cat".',
    'After the last block, the last row is normalized and multiplied by the unembedding matrix: one score per vocabulary word. Softmax turns the 16 scores into probabilities: "on" 39%.',
  ];
  approved.forEach((c, i) => assert.deepEqual(captionProblems(c, i + 1), []));
});

test('caption rules: words, sentences, operators', () => {
  assert.equal(countWords('  one two   three '), 3);
  assert.equal(countSentences('One. "Two." Three!'), 3);
  assert.equal(countSentences('Scores −1.0, 3.0 and 0.5 sum.'), 1);
  assert.match(captionProblems(Array(31).fill('w').join(' '), 2)[0], /step 2: caption has 31 words/);
  assert.match(captionProblems('One. Two. Three.', 3)[0], /3 sentences/);
  assert.match(captionProblems('Then x = 2.', 4)[0], /formula operator/);
  assert.match(captionProblems('', 5)[0], /non-empty/);
});

test('each missing or malformed part is named', () => {
  const problems = validateLessonSpec({ ...VALID, slug: 'Bad Slug', takeaways: ['a'], intuition: ['only one'], links: { next: [], further: [] } });
  assert.deepEqual(problems, [
    'slug must be a kebab-case slug',
    'intuition must be 2–4 non-empty paragraphs (spec §4)',
    'takeaways must be exactly 3 strings (spec §4)',
    'links.further must hold 1–3 { title, href: https://…, note? }',
  ]);
  assert.match(validateLessonSpec({ ...VALID, animation: { ...VALID.animation, render: null } }).join(), /animation.render must be a function/);
  assert.match(validateLessonSpec({ ...VALID, animation: { ...VALID.animation, standIn: 3 } }).join(), /standIn must be a string/);
  assert.match(validateLessonSpec(undefined).join(), /slug must be/);
});
