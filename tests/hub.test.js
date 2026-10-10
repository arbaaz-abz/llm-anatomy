import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { hubModel, countLabel, TRACK_BLURBS } from '../hub/hub.js';

const graph = JSON.parse(await readFile(new URL('../shared/concepts.json', import.meta.url), 'utf8'));

test('hubModel: the three tracks in course order with 11, 14 and 9 lessons (34 in all)', () => {
  const tracks = hubModel(graph);
  assert.deepEqual(tracks.map((t) => [t.id, t.count]), [['architecture', 11], ['training', 14], ['serving', 9]]);
  assert.equal(tracks.reduce((sum, t) => sum + t.count, 0), graph.concepts.length);
});

test('hubModel: every track has a blurb and links to its own page', () => {
  for (const track of hubModel(graph)) {
    assert.ok(TRACK_BLURBS[track.id].length > 0, track.id);
    assert.equal(track.blurb, TRACK_BLURBS[track.id]);
    assert.equal(track.href, `./${track.id}/`);
  }
});

test('hubModel: lessons link to their anchor in their track, in nav order', () => {
  const [architecture] = hubModel(graph);
  const first = architecture.sections[0].lessons[0];
  assert.deepEqual(first, { slug: 'decoder-anatomy', title: 'The whole model, end to end', href: './architecture/#decoder-anatomy' });
});

test('hubModel: lesson numbers run on across sections (Training: Recipe 1–9, GPUs & scale from 10)', () => {
  const training = hubModel(graph)[1];
  assert.deepEqual(training.sections.map((s) => [s.title, s.start, s.lessons.length]), [['Recipe', 1, 9], ['GPUs & scale', 10, 5]]);
});

test('hubModel: learned counts only that track\'s lessons and ignores unknown or malformed input', () => {
  const tracks = hubModel(graph, ['attention', 'rope', 'batching', 'no-such-lesson']);
  assert.deepEqual(tracks.map((t) => t.learned), [2, 0, 1]);
  assert.deepEqual(hubModel(graph, 'not a list').map((t) => t.learned), [0, 0, 0]);
});

test('hubModel: does not mutate the graph', () => {
  const before = JSON.stringify(graph);
  hubModel(graph, ['attention']);
  assert.equal(JSON.stringify(graph), before);
});

test('countLabel: plural, singular, and learned only when nonzero', () => {
  assert.equal(countLabel({ count: 11, learned: 0 }), '11 lessons');
  assert.equal(countLabel({ count: 1, learned: 0 }), '1 lesson');
  assert.equal(countLabel({ count: 9, learned: 3 }), '9 lessons · 3 learned');
});
