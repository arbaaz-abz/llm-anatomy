import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { indexConcepts, topoOrder, learningPath, bySection } from '../shared/concepts.js';

const data = JSON.parse(await readFile(new URL('../shared/concepts.json', import.meta.url), 'utf8'));

test('concept file matches the spec counts (11 / 14 / 9)', () => {
  const count = (t) => data.concepts.filter((c) => c.track === t).length;
  assert.deepEqual([count('architecture'), count('training'), count('serving')], [11, 14, 9]);
});

test('every concept references a real track and section', () => {
  const sections = new Map(data.tracks.map((t) => [t.id, new Set(t.sections.map((s) => s.id))]));
  data.concepts.forEach((c) => assert.ok(sections.get(c.track)?.has(c.section), `${c.slug}: ${c.track}/${c.section}`));
});

test('indexConcepts validates slugs, duplicates and prereqs', () => {
  const index = indexConcepts(data.concepts);
  assert.equal(index.size, 34);
  assert.throws(() => indexConcepts([{ slug: 'Bad Slug', prereqs: [] }]), /slug/);
  assert.throws(() => indexConcepts([{ prereqs: [] }]), /slug/);
  assert.throws(() => indexConcepts([{ slug: 42, prereqs: [] }]), /slug/);
  assert.throws(() => indexConcepts([{ slug: 'a', prereqs: [] }, { slug: 'a', prereqs: [] }]), /duplicate/);
  assert.throws(() => indexConcepts([{ slug: 'a', prereqs: ['ghost'] }]), /ghost/);
});

test('graph is acyclic and topoOrder puts prereqs first', () => {
  const order = topoOrder(data.concepts);
  const pos = new Map(order.map((s, i) => [s, i]));
  data.concepts.forEach((c) => c.prereqs.forEach((p) => assert.ok(pos.get(p) < pos.get(c.slug), `${p} before ${c.slug}`)));
  assert.throws(() => topoOrder([{ slug: 'a', prereqs: ['b'] }, { slug: 'b', prereqs: ['a'] }]), /cycle/);
});

test('learningPath returns the transitive prereqs in order, ending with the target', () => {
  const index = indexConcepts(data.concepts);
  const path = learningPath(index, 'paged-attention');
  assert.equal(path.at(-1), 'paged-attention');
  ['attention', 'kv-cache', 'kv-compression', 'batching', 'gpu-primer'].forEach((s) => assert.ok(path.includes(s), s));
  assert.ok(path.indexOf('kv-cache') < path.indexOf('kv-compression'));
  assert.deepEqual(learningPath(index, 'attention'), ['decoder-anatomy', 'attention']);
  assert.throws(() => learningPath(index, 'nope'), /unknown/);
});

test('bySection groups a track in file order', () => {
  const groups = bySection(data.concepts, 'training');
  assert.deepEqual(groups.map((g) => g.section), ['recipe', 'gpus']);
  assert.equal(groups[0].concepts[0].slug, 'training-pipeline');
  assert.equal(groups[1].concepts.length, 5);
});

test('prereq rulings: pretraining, model-card and distillation keep only direct edges', () => {
  const index = indexConcepts(data.concepts);
  assert.deepEqual(index.get('pretraining').prereqs, ['training-pipeline']);
  assert.deepEqual(index.get('model-card').prereqs, ['long-context-attention', 'moe', 'multimodal']);
  assert.deepEqual(index.get('distillation').prereqs, ['agentic-rl']);
  assert.ok(index.get('paged-attention').prereqs.includes('kv-compression'));
});

test('decoder-anatomy is the first architecture concept and has no prereqs', () => {
  const index = indexConcepts(data.concepts);
  assert.deepEqual(index.get('decoder-anatomy').prereqs, []);
  assert.equal(data.concepts.find((c) => c.track === 'architecture').slug, 'decoder-anatomy');
  assert.deepEqual(index.get('attention').prereqs, ['decoder-anatomy']);
  assert.deepEqual(index.get('decoder-recap').prereqs, ['decoder-anatomy']);
});
