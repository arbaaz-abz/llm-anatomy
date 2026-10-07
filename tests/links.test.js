import { test } from 'node:test';
import assert from 'node:assert/strict';
import { conceptHref, hubHref, currentTarget } from '../shared/links.js';

const artifactUrls = {
  hub: 'https://claude.ai/artifact/HUB', architecture: 'https://claude.ai/artifact/ARCH',
  training: 'https://claude.ai/artifact/TRAIN', serving: '',
};

test('same-track links are plain anchors in both targets', () => {
  assert.equal(conceptHref({ from: 'architecture', track: 'architecture', slug: 'rope', target: 'pages', artifactUrls }), '#rope');
  assert.equal(conceptHref({ from: 'architecture', track: 'architecture', slug: 'rope', target: 'artifact', artifactUrls }), '#rope');
});

test('cross-track links on GitHub Pages are relative', () => {
  assert.equal(conceptHref({ from: 'hub', track: 'serving', slug: 'batching', target: 'pages', artifactUrls }), './serving/#batching');
  assert.equal(conceptHref({ from: 'training', track: 'architecture', slug: 'moe', target: 'pages', artifactUrls }), '../architecture/#moe');
  assert.equal(conceptHref({ from: 'hub', track: 'serving', slug: null, target: 'pages', artifactUrls }), './serving/');
  assert.equal(hubHref({ from: 'serving', target: 'pages', artifactUrls }), '../');
});

test('cross-track links in artifacts use the published URL', () => {
  assert.equal(conceptHref({ from: 'hub', track: 'training', slug: 'sft', target: 'artifact', artifactUrls }), 'https://claude.ai/artifact/TRAIN#sft');
  assert.equal(hubHref({ from: 'training', target: 'artifact', artifactUrls }), 'https://claude.ai/artifact/HUB');
});

test('an unpublished artifact link falls back to null instead of a broken URL', () => {
  assert.equal(conceptHref({ from: 'hub', track: 'serving', slug: 'batching', target: 'artifact', artifactUrls }), null);
});

test('currentTarget reads the build meta tag', () => {
  const doc = (content) => ({ querySelector: () => (content ? { content } : null) });
  assert.equal(currentTarget(doc('artifact')), 'artifact');
  assert.equal(currentTarget(doc(null)), 'pages');
});
