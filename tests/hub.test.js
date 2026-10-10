import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { hubModel, countLabel, TRACK_BLURBS } from '../hub/hub.js';
import { storyFrame, FINAL_FRAME, STORY_MS, LEAD_MS, ACT_MS, modelScene, trainScene, serveScene, MODEL, W_START, W_END, LOSS_CURVE, SERVE } from '../hub/story.js';
import { mapLayout, depthsOf, chainOf, captionFor, MAP } from '../hub/map-layout.js';
import { nextLesson, learnedCount, courseOrder } from '../hub/progress.js';

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
  assert.deepEqual(first, { slug: 'decoder-anatomy', title: 'The whole model, end to end', href: './architecture/#decoder-anatomy', learned: false });
  assert.equal(hubModel(graph, ['decoder-anatomy'])[0].sections[0].lessons[0].learned, true);
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

// ---- the story (hub/story.js) ----

test('storyFrame: before the lead nothing has moved; acts play one after the other; the end rests on the final frame', () => {
  assert.deepEqual(storyFrame(0), { progress: [0, 0, 0], act: 0, done: false });
  assert.deepEqual(storyFrame(LEAD_MS + ACT_MS / 2), { progress: [0.5, 0, 0], act: 0, done: false });
  assert.deepEqual(storyFrame(LEAD_MS + ACT_MS * 1.25), { progress: [1, 0.25, 0], act: 1, done: false });
  assert.deepEqual(storyFrame(STORY_MS), { progress: [1, 1, 1], act: -1, done: true });
  assert.deepEqual(storyFrame(Infinity), FINAL_FRAME);
  assert.deepEqual(storyFrame(NaN), storyFrame(0));
  assert.deepEqual(storyFrame(-50), storyFrame(0));
});

test('modelScene: input tokens appear in turn, the dot climbs the spine through the blocks, then the next token is out', () => {
  assert.deepEqual(modelScene(0).tokens.map((t) => t.state), ['dim', 'dim', 'dim']);
  assert.deepEqual(modelScene(0.2).tokens.map((t) => t.state), ['idle', 'idle', 'idle']);
  assert.equal(modelScene(0.2).dotY, null);
  assert.equal(modelScene(0.2).output, 'draft');
  const mid = modelScene(0.55);
  assert.ok(mid.dotY < MODEL.spine.from && mid.dotY > MODEL.spine.to, 'the dot is between the input row and the output token');
  const ys = [0.3, 0.45, 0.6, 0.75].map((p) => modelScene(p).dotY);
  assert.deepEqual([...ys].sort((a, b) => b - a), ys, 'the dot only ever moves up');
  const seen = new Set([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8].map((p) => modelScene(p).active).filter(Boolean));
  assert.deepEqual([...seen], ['attention', 'mlp'], 'the dot lights attention before the MLP');
  assert.deepEqual(modelScene(1), { tokens: MODEL.input.map((text) => ({ text, state: 'idle' })), dotY: null, active: null, output: 'active' });
});

test('trainScene: the weights move from the noisy start to the settled end while the loss curve draws itself', () => {
  assert.deepEqual(trainScene(0).weights, W_START);
  assert.deepEqual(trainScene(1).weights, W_END);
  assert.equal(trainScene(0).curve.length, 1);
  assert.equal(trainScene(1).curve.length, LOSS_CURVE.length);
  assert.equal(trainScene(0).gradient, null);
  assert.equal(trainScene(1).gradient, null);
  const g = trainScene(0.5).gradient;
  assert.ok(g >= 0 && g < 1, 'mid-act the gradient dot is on its way');
  const loss = LOSS_CURVE.map(([, y]) => y);
  assert.deepEqual([...loss].sort((a, b) => b - a), loss, 'the loss only falls');
  assert.ok(W_START.every((row) => row.every((v) => Math.abs(v) <= 1)) && W_END.every((row) => row.every((v) => Math.abs(v) <= 1)));
});

test('serveScene: requests arrive in turn and dock at the GPU, then tokens stream out per request', () => {
  const start = serveScene(0);
  assert.deepEqual(start.requests.map((r) => r.owner), SERVE.owners);
  assert.ok(start.requests.every((r) => r.x === SERVE.enter));
  assert.deepEqual(start.streams, [0, 0, 0]);
  assert.equal(start.memFill, 0);
  const quarter = serveScene(0.2);
  assert.ok(quarter.requests[0].x > quarter.requests[1].x && quarter.requests[1].x > quarter.requests[2].x, 'A leads B leads C');
  const end = serveScene(1);
  assert.ok(end.requests.every((r) => r.x === SERVE.dock));
  assert.deepEqual(end.streams, [SERVE.streamMax, SERVE.streamMax, SERVE.streamMax]);
  assert.equal(end.memFill, 1);
  assert.ok(SERVE.streamX + (SERVE.streamMax - 1) * SERVE.streamStep + 28 <= 320, 'the longest stream stays inside the cover');
});

// ---- the course map (hub/map-layout.js) ----

test('depthsOf: a starting point is 0, everything else one more than its deepest prerequisite', () => {
  const depths = depthsOf(graph.concepts);
  assert.equal(depths.get('decoder-anatomy'), 0);
  assert.equal(depths.get('gpu-primer'), 0);
  assert.equal(depths.get('attention'), 1);
  assert.equal(depths.get('serving-overview'), 3); // kv-cache ← attention ← decoder-anatomy
  for (const c of graph.concepts) {
    const expected = c.prereqs.length ? 1 + Math.max(...c.prereqs.map((p) => depths.get(p))) : 0;
    assert.equal(depths.get(c.slug), expected, c.slug);
  }
});

test('mapLayout: a station per lesson inside its lane, an edge per prerequisite, cross-track edges flagged', () => {
  const layout = mapLayout(graph, ['attention']);
  assert.equal(layout.stations.length, graph.concepts.length);
  assert.equal(layout.lanes.map((l) => l.id).join(','), 'architecture,training,serving');
  assert.equal(layout.edges.length, graph.concepts.reduce((n, c) => n + c.prereqs.length, 0));
  const cross = layout.edges.filter((e) => e.cross);
  assert.equal(cross.length, graph.concepts.reduce((n, c) => n + c.prereqs.filter((p) => graph.concepts.find((o) => o.slug === p).track !== c.track).length, 0));
  assert.ok(cross.some((e) => e.from === 'gpu-primer' && e.to === 'prefill-decode'));
  for (const s of layout.stations) {
    const lane = layout.lanes.find((l) => l.id === s.track);
    assert.ok(s.y > lane.y && s.y < lane.y + lane.height, `${s.slug} sits in its lane`);
    assert.ok(s.x - MAP.r > 0 && s.x + MAP.r < layout.width, `${s.slug} sits inside the map`);
    assert.equal(s.href, `./${s.track}/#${s.slug}`);
  }
  assert.equal(layout.stations.find((s) => s.slug === 'attention').learned, true);
  assert.equal(layout.stations.find((s) => s.slug === 'rope').learned, false);
  const training = layout.stations.filter((s) => s.track === 'training');
  assert.deepEqual(training.slice(0, 2).map((s) => s.number), [1, 2], 'numbers follow the lists on the cards');
  assert.ok(layout.edges.every((e) => /^M[\d. ]+ (C|Q)[\d. -]+$/.test(e.d)), 'every edge is one path');
});

test('mapLayout: no two stations share a spot, edges run from an earlier column to a later one', () => {
  const layout = mapLayout(graph);
  const spots = new Set(layout.stations.map((s) => `${s.x},${s.y}`));
  assert.equal(spots.size, layout.stations.length);
  const at = new Map(layout.stations.map((s) => [s.slug, s]));
  for (const e of layout.edges) assert.ok(at.get(e.from).depth < at.get(e.to).depth, `${e.from} → ${e.to}`);
});

test('mapLayout: does not mutate the graph and accepts malformed learned input', () => {
  const before = JSON.stringify(graph);
  mapLayout(graph, 'nope');
  assert.equal(JSON.stringify(graph), before);
});

test('chainOf and captionFor: a starting point says so; a late lesson lists what it builds on, nearest last', () => {
  const layout = mapLayout(graph);
  assert.deepEqual(chainOf(layout, 'decoder-anatomy'), []);
  assert.equal(captionFor(layout, 'decoder-anatomy'), 'The whole model, end to end (Architecture 1) is a starting point: no prerequisites.');
  const chain = chainOf(layout, 'serving-overview').map((s) => s.slug);
  assert.deepEqual(chain, ['decoder-anatomy', 'attention', 'kv-cache']);
  assert.equal(captionFor(layout, 'serving-overview'), "One request's journey (Serving 1) builds on 3 lessons: The whole model, end to end, Attention, step by step, The KV cache.");
  assert.equal(captionFor(layout, 'no-such-lesson'), '');
});

// ---- progress (hub/progress.js) ----

test('courseOrder: all 34 lessons, track by track, as the cards list them', () => {
  const order = courseOrder(graph);
  assert.equal(order.length, 34);
  assert.equal(order[0].slug, 'decoder-anatomy');
  assert.equal(order[11].slug, 'training-pipeline');
  assert.equal(order[33].slug, 'serving-calculator');
});

test('nextLesson: nothing before anything is learned; then the first unlearned lesson whose prerequisites are done', () => {
  assert.equal(nextLesson(graph, []), null);
  assert.equal(nextLesson(graph, 'nope'), null);
  assert.deepEqual(nextLesson(graph, ['decoder-anatomy']), { slug: 'decoder-recap', title: 'From GPT-3 to 2026', track: 'architecture' });
  // rope is learned but its prerequisite is not: the first unlearned lesson is still the pick
  assert.equal(nextLesson(graph, ['rope']).slug, 'decoder-anatomy');
  assert.equal(nextLesson(graph, graph.concepts.map((c) => c.slug)), null);
});

test('learnedCount: counts only real lessons', () => {
  assert.equal(learnedCount(graph, ['attention', 'rope', 'ghost']), 2);
  assert.equal(learnedCount(graph, null), 0);
});
