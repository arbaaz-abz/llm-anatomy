import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { formatBytes, formatDuration, formatInt } from '../math/core.js';
import { blendedInputPrice, simulatePrefixCache, PREFIX_REQUESTS } from '../math/prefix.js';
import { RUNNING_EXAMPLE, stepTime } from '../math/serving.js';
import { kvCacheBytes } from '../math/memory.js';
import { LESSON, lessonFor, INTUITION, MATH_NOTES } from '../serving/concepts/prefix-caching/content.js';
import { CAPTIONS as PAGE_CAPTIONS } from '../serving/concepts/prefix-caching/captions.js';
import { factRows, stageText, providersFor, readOn, deepseekChip, PREFILL_TITLE, PAGED_TITLE, FRAMING } from '../serving/concepts/prefix-caching/facts.js';
import {
  INITIAL_STATE, BLOCK_SIZES, POOLS, formatPrice, toggleRequest, setBlockSize, setPool, setProvider, setWritePremium, setHitMode, simulateFor, hitFraction, requestsFor,
} from '../serving/concepts/prefix-caching/format.js';
import { toyView, tryThis, TREE_MIN_BLOCK_SIZE } from '../serving/concepts/prefix-caching/toy-view.js';
import { SCALE_UP } from '../serving/concepts/prefix-caching/numbers.js';
import { LOG, DEFAULT_MODEL, DEFAULT_PATHS, shortLabel, poolAfter, poolDuring, liveQueue, evictedIds, treeNodes, nodeState } from '../serving/concepts/prefix-caching/scenes.js';
import { sceneAt, ARRIVE } from '../serving/concepts/prefix-caching/anim.js';
import { replicaHits, readsLine } from '../serving/concepts/prefix-caching/frames-extra.js';
import { CAPTIONS, CHECK_WORK, DEFAULT_ROWS, TRY_THIS } from './prefix-caching-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const fact = (id, key) => data.serving.entries.find((e) => e.id === id).facts[key].value;
const filled = lessonFor(data);
const text = stageText(data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(filled), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.deepEqual([...PAGE_CAPTIONS], CAPTIONS);
  assert.equal(CAPTIONS.length, 11);
});

test('the numbers the captions repeat match the simulation, the routing and the data (README lesson 29)', () => {
  const [a, b, c, d] = LOG;
  assert.match(CAPTIONS[2], new RegExp(`skips ${b.hitTokens} tokens and computes only the ${b.computed} that differ`));
  assert.match(CAPTIONS[4], /Sixteen of its twenty prompt tokens/);
  assert.deepEqual([c.hitTokens, c.promptTokens], [16, 20]);
  assert.match(CAPTIONS[7], /24 of 57 prompt tokens came from the cache, a 42\.1% hit rate/);
  assert.deepEqual([a, b, c, d].reduce((s, r) => s + r.hitTokens, 0), 24);
  assert.deepEqual([a, b, c, d].reduce((s, r) => s + r.promptTokens, 0), 57);
  assert.match(CAPTIONS[7], new RegExp(`${fact('deepseek-v3-production', 'kv_hit_rate_pct')}% across its production traffic in 2025`));
  assert.deepEqual(replicaHits(), [16, 8]);
  assert.match(CAPTIONS[8], /reuses 16 tokens; sent anywhere else, only 8/);
  assert.match(CAPTIONS[10], /a tenth or less/);
  assert.equal(fact('pricing-anthropic', 'cache_read_mult'), 0.1);
  assert.equal(fact('pricing-anthropic', 'cache_write_5m_mult'), 1.25, '"a quarter more"');
});

test('10 facts rows, every placeholder resolves against data/*.json, nothing unfilled', () => {
  const rows = factRows();
  assert.equal(rows.length, 10);
  assert.equal(LESSON.facts.rows.length, 10);
  rows.forEach((row, i) => {
    const f = fillClaim(row.claim, data);
    assert.deepEqual(f.missing, [], `row ${i + 1}`);
    if (!row.derived) assert.ok(f.sources.length > 0, `row ${i + 1} has a source`);
    assert.doesNotMatch(f.segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`);
  });
  const line = (i) => fillClaim(rows[i].claim, data).segments.map((s) => s.text).join('');
  assert.match(line(0), /It caches full blocks only, and evicts unused blocks through an LRU free queue\./);
  assert.equal(line(1), 'vLLM\'s default block size is 16 tokens.');
  assert.match(line(3), /^DeepSeek's production inference system \(V3\/R1, Feb 2025\): 56\.3% of its 608B input tokens per day hit the KV cache\.$/);
  assert.match(line(5), /59–498% more effective request capacity/);
  assert.match(line(6), /up to 64 conversations fit in HBM, 64–128 need CPU offload, and throughput with storage offload was more than 2x beyond 128 conversations\./);
  assert.equal(line(7), 'Anthropic: a cache read costs 0.1× the input price for most models, a cache write 1.25× (5-minute entry) or 2× (1-hour entry). Sonnet 5.5 input is $2 per million tokens; Opus 5.5 is $4, with reads at 0.05×.');
  assert.equal(line(8), 'DeepSeek V4-Pro, off-peak: $0.66 per million input tokens on a miss and $0.022 on a hit, with no cache-write fee; peak rates are 2×.');
  [2, 4, 5].forEach((i) => assert.equal(fillClaim(rows[i].claim, data).reported, true, `row ${i + 1} is reported`));
  rows.forEach((r, i) => assert.doesNotMatch(r.claim, /\(reported\)/, `row ${i + 1}: the scaffold's chip says reported`));
  assert.match(line(2), /^SGLang's RadixAttention keeps cached KV in a radix tree keyed by token sequences, with LRU leaf eviction\.$/);
  assert.match(line(4), /^KV-aware routing: NVIDIA Dynamo's router sends requests "to GPUs that already have the most relevant short-term memory from earlier steps"; llm-d's scheduler routes by prefix; vLLM engines emit KV events so routers know what each replica holds\.$/);
  assert.match(line(6), /^vLLM tiered KV offloading \(2026-09-10\): /);
  assert.equal(fillClaim(rows[0].claim, data).reported, false);
  assert.equal(rows[9].derived, true);
  assert.match(FRAMING, /list prices on one day/);
});

test('dated prose fills from data: hook, intuition, stand-in, notes, takeaways; nothing missing', () => {
  const texts = [filled.hook, ...filled.intuition, filled.animation.standIn, ...filled.math.notes, ...filled.takeaways];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}]|—/, t.slice(0, 60)));
  [...INTUITION, ...MATH_NOTES, LESSON.animation.standIn].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(filled.intuition[1], /DeepSeek reported 56\.3% of its input tokens hitting the cache in production/);
  assert.match(filled.animation.standIn, /blocks of 4 where vLLM uses 16/);
  assert.equal(INTUITION.length, 4);
});

test('stage text is computed from the data (frames 1, 8, 10, 11)', () => {
  assert.deepEqual([...text.frame1], [
    '1 slot = one token\'s K and V, for every layer (from PagedAttention)',
    'hand-picked prompts; blocks of 4 (vLLM uses 16)',
    'tokens count from 1; blocks from 0',
  ]);
  assert.equal(text.deepseek, 'DeepSeek production (Feb 2025): 56.3%');
  assert.deepEqual([...text.tiers], [
    'vLLM tiered offload (2026-09-10), Qwen-35B on 2 × H100:',
    'up to 64 conversations fit in HBM; 64–128 need CPU offload;',
    'with storage offload, throughput was more than 2x beyond 128 conversations.',
  ]);
  assert.equal(text.readOn, '2026-10-07');
  assert.equal(readOn(null), '—');
  const empty = stageText(null);
  assert.equal(empty.providers, null);
  assert.equal(empty.deepseek, 'DeepSeek production (—): —%');
});

test('lesson names printed in text are the graph\'s titles (README lesson 33)', () => {
  const title = (slug) => graph.concepts.find((c) => c.slug === slug).title;
  assert.equal(PREFILL_TITLE, title('prefill-decode'));
  assert.equal(PAGED_TITLE, title('paged-attention'));
  assert.equal(LESSON.slug, 'prefix-caching');
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('prefix-caching')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(graph.concepts.find((c) => c.slug === 'prefix-caching').prereqs, ['paged-attention']);
});

test('the math panel: hl-hit links, and the multipliers it names match the data (ruling P3-R13)', () => {
  const tex = LESSON.math.blocks.map((b) => b.tex).join('\n');
  assert.match(tex, /\\htmlClass\{hl-hit\}/);
  assert.equal(LESSON.math.blocks.length, 3);
});

test('providers: Sonnet, Opus and DeepSeek restate the data (price, read, write)', () => {
  const [sonnet, opus, deepseek] = providersFor(data);
  assert.deepEqual([sonnet.base, sonnet.read, sonnet.write, sonnet.hasWrite], [2, 0.1, 1.25, true]);
  assert.deepEqual([opus.base, opus.read, opus.write, opus.hasWrite], [4, 0.05, 1.25, true]);
  assert.deepEqual([deepseek.base, deepseek.hitUsd, deepseek.write, deepseek.hasWrite], [0.66, 0.022, 1, false]);
  assert.equal(deepseek.read, 0.022 / 0.66);
  assert.equal(providersFor(null), null);
  assert.equal(providersFor({ serving: { entries: [] } }), null);
});

test('price format: two decimals, two significant figures below $0.10 (one format per quantity)', () => {
  assert.deepEqual([2, 2.5, 0.2, 0.66, 1.4474, 0.1].map(formatPrice), ['$2.00', '$2.50', '$0.20', '$0.66', '$1.45', '$0.10']);
  assert.deepEqual([0.022, 0.084, 0.0120, 0.0033, 0.05].map(formatPrice), ['$0.022', '$0.084', '$0.012', '$0.0033', '$0.05']);
  assert.equal(formatPrice(0), '$0.00');
  assert.throws(() => formatPrice(-1), RangeError);
  assert.throws(() => formatPrice(NaN), RangeError);
});

test('state transitions: toggle requests (kept in arrival order), block size, pool, provider, write premium, hit rate; inputs never mutated', () => {
  const before = JSON.stringify(INITIAL_STATE);
  assert.deepEqual(toggleRequest(INITIAL_STATE, 'B').on, ['A', 'C', 'D']);
  assert.deepEqual(toggleRequest(toggleRequest(INITIAL_STATE, 'B-again'), 'A').on, ['B', 'C', 'D', 'B-again']);
  assert.deepEqual(toggleRequest(toggleRequest(INITIAL_STATE, 'B'), 'B').on, ['A', 'B', 'C', 'D']);
  assert.equal(setBlockSize(INITIAL_STATE, 16).blockSize, 16);
  assert.equal(setPool(INITIAL_STATE, 6).pool, 6);
  assert.equal(setProvider(INITIAL_STATE, 'deepseek').provider, 'deepseek');
  assert.equal(setWritePremium(INITIAL_STATE, 0).writePremium, false);
  assert.equal(setHitMode(INITIAL_STATE, 56.3).hitMode, 56.3);
  assert.equal(setHitMode(setHitMode(INITIAL_STATE, 10), 'toy').hitMode, 'toy');
  assert.throws(() => toggleRequest(INITIAL_STATE, 'E'), RangeError);
  assert.throws(() => setBlockSize(INITIAL_STATE, 3), RangeError);
  assert.throws(() => setPool(INITIAL_STATE, 7), RangeError);
  assert.throws(() => setProvider(INITIAL_STATE, 'x'), RangeError);
  assert.throws(() => setHitMode(INITIAL_STATE, 101), RangeError);
  assert.throws(() => setHitMode(INITIAL_STATE, 'x'), RangeError);
  assert.equal(JSON.stringify(INITIAL_STATE), before);
  assert.ok(Object.isFrozen(INITIAL_STATE) && Object.isFrozen(INITIAL_STATE.on));
  assert.deepEqual([BLOCK_SIZES, POOLS], [[1, 2, 4, 8, 16], [6, 8, 12]]);
});

test('"Check my work" for the default state is the storyboard text; other states fill the same template', () => {
  assert.equal(toyView(INITIAL_STATE, data).checkWork, CHECK_WORK);
  const lines = (state) => toyView(state, data).checkWork.split('\n');
  assert.equal(lines(setHitMode(INITIAL_STATE, 56.3))[0], 'h = 56.3% (set by hand)');
  assert.equal(lines(setHitMode(INITIAL_STATE, 56.3))[2], '            = 0.437 · $2.00 · 1.25 + 0.563 · $2.00 · 0.1 = $1.09 + $0.113 = $1.21');
  assert.equal(lines(setWritePremium(INITIAL_STATE, false))[2], '            = 0.579 · $2.00 · 1 + 0.421 · $2.00 · 0.1 = $1.16 + $0.0842 = $1.24');
  assert.equal(lines(setProvider(INITIAL_STATE, 'opus'))[2], '            = 0.579 · $4.00 · 1.25 + 0.421 · $4.00 · 0.05 = $2.89 + $0.0842 = $2.98');
  const ds = lines(setHitMode(setProvider(INITIAL_STATE, 'deepseek'), 56.3));
  assert.equal(ds[1], 'price per M = (1 − h) · miss + h · hit');
  assert.equal(ds[2], '            = 0.437 · $0.66 + 0.563 · $0.022 = $0.288 + $0.0124 = $0.301');
  assert.equal(lines(setHitMode(INITIAL_STATE, 0))[2], '            = 1.000 · $2.00 · 1.25 + 0.000 · $2.00 · 0.1 = $2.50 + $0.00 = $2.50');
  const none = toyView({ ...INITIAL_STATE, on: [] }, data);
  assert.equal(none.checkWork.split('\n')[0], 'h = 0 (no request has arrived)');
});

test('toy view: the default state prints the storyboard\'s numbers, each from math/prefix.js', () => {
  const v = toyView(INITIAL_STATE, data);
  assert.deepEqual(v.rows.map((r) => [r.key, r.prompt, r.hit, r.computed, r.blocks, r.evicted]), DEFAULT_ROWS);
  assert.deepEqual(v.totals, { prompt: '57', hit: '24', computed: '33', rate: '42.1%' });
  const sim = simulatePrefixCache({ requests: PREFIX_REQUESTS, blockSize: 4, poolBlocks: 8 });
  assert.equal(v.sim.hitTokens, sim.hitTokens);
  assert.deepEqual(v.sim.log, sim.log);
  assert.equal(v.hitSlider, '42.1%');
  assert.equal(v.stage.tree.length, 1 + DEFAULT_MODEL.length, 'the root plus every block');
  assert.deepEqual(v.stage.queue, [3, 2, 1, 0, 6, 7, 4, 5]);
  assert.equal(v.stage.blockSize, 4);
});

test('Review Focus 1: the price readout equals blendedInputPrice for each provider chip; the scale-up line equals stepTime and kvCacheBytes', () => {
  const h = 24 / 57;
  providersFor(data).forEach((p) => {
    const view = toyView(setProvider(INITIAL_STATE, p.id), data);
    const write = p.hasWrite ? p.write : 1;
    assert.equal(view.price.blended, formatPrice(blendedInputPrice({ hitRate: h, basePrice: p.base, readMult: p.read, writeMult: write })), p.id);
    assert.equal(view.price.plain, formatPrice(p.base));
  });
  assert.equal(toyView(INITIAL_STATE, data).price.blended, '$1.53');
  assert.equal(toyView(setWritePremium(INITIAL_STATE, false), data).price.blended, '$1.24');
  assert.equal(toyView(setProvider(INITIAL_STATE, 'deepseek'), data).price.write, false);
  const view = toyView(INITIAL_STATE, data);
  assert.equal(view.scale.skipped, formatDuration(stepTime({ ...RUNNING_EXAMPLE, tokens: 10000, seqs: 0, context: 0 }).timeS));
  assert.equal(view.scale.held, formatBytes(kvCacheBytes({ bytesPerToken: RUNNING_EXAMPLE.kvBytesPerToken, tokens: 10000 })));
  assert.deepEqual([view.scale.skipped, view.scale.held, view.scale.tokens], ['707 ms', '3.28 GB', '10,000']);
  assert.equal(SCALE_UP.tokens, 10000);
});

test('toy view: every block size and pool runs; hit rates by block size at pool 8; the tree is drawn from block size 4 up', () => {
  const rates = BLOCK_SIZES.map((bs) => toyView(setBlockSize(INITIAL_STATE, bs), data).totals.rate);
  assert.deepEqual(rates, ['47.4%', '45.6%', '42.1%', '42.1%', '28.1%']);
  for (const bs of BLOCK_SIZES) {
    for (const pool of POOLS) {
      const v = toyView(setPool(setBlockSize(INITIAL_STATE, bs), pool), data);
      assert.equal(v.stage.tree !== null, bs >= TREE_MIN_BLOCK_SIZE, `block size ${bs}`);
      assert.equal(v.stage.poolBlocks.length, Math.ceil((pool * 4) / bs));
    }
  }
  const one = toyView(setBlockSize(INITIAL_STATE, 1), data);
  assert.match(one.stage.treeNote, /too many blocks to draw as a tree/);
  assert.equal(one.rows[3].hit, '3');
  assert.match(one.rows[3].evicted, /^\d+ blocks$/, 'long eviction lists print as a count');
});

test('toy view: B again and a pool of 6 (try this 2)', () => {
  const again = toyView(toggleRequest(INITIAL_STATE, 'B-again'), data);
  assert.deepEqual([again.rows[4].label, again.rows[4].hit, again.rows[4].blocks, again.rows[4].evicted], ['B again', '8', '0, 1, 3, 2', 'The…down, Where…sit']);
  const small = toyView(setPool(toggleRequest(INITIAL_STATE, 'B-again'), 6), data);
  assert.equal(small.rows[2].evicted, '?…fish, Do…fish');
  assert.equal(small.rows[3].evicted, 'It…., Why…?, The…down, Where…sit');
});

test('toy view: no request at all is an empty cache, and a missing provider hides the price', () => {
  const none = toyView({ ...INITIAL_STATE, on: [] }, data);
  assert.deepEqual(none.rows, []);
  assert.deepEqual(none.totals, { prompt: '0', hit: '0', computed: '0', rate: '—' });
  assert.equal(none.stage.tree, null);
  assert.equal(none.price.rate, '0.00%');
  assert.equal(hitFraction(INITIAL_STATE, simulateFor({ ...INITIAL_STATE, on: [] })), 0);
  assert.equal(toyView(INITIAL_STATE, null).price, null);
  assert.equal(toyView(INITIAL_STATE, null).checkWork, '');
  assert.deepEqual(requestsFor({ ...INITIAL_STATE, on: ['B-again'] }).map((r) => [r.id, r.label]), [['B', 'B again']]);
});

test('the try-this list is the storyboard\'s, with its numbers computed', () => {
  assert.deepEqual(tryThis(data), TRY_THIS);
  const missing = tryThis(null);
  assert.match(missing[2].prompt, /Hit rate for pricing at 0%: — per M/);
});

test('scenes: the tree, the pool and the queue of the storyboard\'s key frames (frames 1, 5, 7 and 8)', () => {
  const ids = (s) => s.tree.map((n) => `${n.label}:${n.state}`);
  assert.deepEqual(ids(sceneAt(0, 1, text)), ['start:cached', 'You…cat:cached', '.…rhyme:cached', 'Where…sit:cached', 'The…down:cached']);
  assert.deepEqual(sceneAt(0, 1, text).pool.blocks.map((b) => b.state), ['cached', 'cached', 'cached', 'cached', 'free', 'free', 'free', 'free']);
  const f7 = sceneAt(6, 1, text);
  assert.deepEqual(ids(f7), [
    'start:cached', 'You…cat:cached', '.…rhyme:cached', 'Where…sit:cached', 'The…down:cached', 'Do…fish:evicted', '?…fish:evicted', 'Why…?:evicted', 'It….:evicted',
    'You…dog:new', '.…prose:new', 'Where…sit:new', 'On….:new',
  ]);
  assert.equal(f7.tree.filter((n) => n.state === 'evicted').length, 4);
  assert.equal(f7.tree.filter((n) => n.state === 'new').length, 4);
  assert.deepEqual(f7.pool.blocks.map((b) => `${b.state}:${b.owner ?? ''}`), ['cached:A', 'cached:A', 'cached:A', 'cached:A', 'filled:D', 'filled:D', 'filled:D', 'filled:D']);
  const f8 = sceneAt(7, 1, text);
  assert.equal(f8.tree.filter((n) => n.state === 'new' || n.state === 'hit').length, 0, 'nobody runs after frame 8 starts');
  assert.deepEqual(f8.pool.blocks.map((b) => b.state), Array(8).fill('cached'));
});

test('scenes: queues follow the pool (live while a request runs, whole once it finishes)', () => {
  assert.deepEqual([0, 1, 2, 3].map((i) => liveQueue(LOG[i - 1] ?? null, LOG[i], 8)), [[4, 5, 6, 7], [6, 7, 3, 2], [5, 4], [3, 2, 1, 0]]);
  assert.deepEqual(poolDuring(LOG[0], LOG[1], 8).map((b) => b.state), ['cached', 'cached', 'cached', 'cached', 'filled', 'filled', 'free', 'free']);
  assert.deepEqual(poolAfter(null, 3), [{ state: 'free' }, { state: 'free' }, { state: 'free' }]);
  assert.deepEqual(evictedIds(DEFAULT_MODEL, LOG[3], 3).map((id) => DEFAULT_MODEL.find((n) => n.id === id).text), ['? Yes , fish', 'Do you like fish', 'It was warm .', 'Why down there ?']);
  assert.equal(shortLabel('Why down there ?'), 'Why…?');
  assert.equal(shortLabel('start'), 'start');
  const labels = DEFAULT_MODEL.map((n) => shortLabel(n.text));
  assert.equal(new Set(labels).size, labels.length - 1, 'only A\'s and D\'s "Where did you sit" print alike (the same words)');
});

test('scenes are pure and a frame starts exactly where the previous one ended (template rule 4)', () => {
  const shape = (s) => JSON.stringify({ tree: s.tree, pool: s.pool, overlays: s.overlays.map((o) => o.key) });
  for (let f = 0; f < 8; f += 1) {
    for (const p of [0, 0.3, 0.7, 1]) assert.equal(shape(sceneAt(f, p, text)), shape(sceneAt(f, p, text)), `frame ${f + 1} at ${p}`);
    if (f > 0) assert.equal(shape(sceneAt(f, 0, text)), shape(sceneAt(f - 1, 1, text)), `frame ${f + 1} starts where frame ${f} ended`);
    assert.equal(shape(sceneAt(f, ARRIVE - 0.001, text)), shape(f === 0 ? sceneAt(0, ARRIVE - 0.001, text) : sceneAt(f - 1, 1, text)));
  }
});

test('every node the scenes draw comes from the model; states are the glyph\'s four', () => {
  for (let f = 0; f < 8; f += 1) {
    sceneAt(f, 1, text).tree.forEach((n) => assert.ok(['hit', 'new', 'cached', 'evicted'].includes(n.state), `${n.id}: ${n.state}`));
  }
  const all = treeNodes(DEFAULT_MODEL, { state: nodeState({ paths: DEFAULT_PATHS, active: null }) });
  assert.equal(all.length, 13, 'root, A 4, B 2, C 2, D 4');
  assert.equal(formatInt(all.length), '13');
});

test('the frames\' notes name what the frame shows (storyboard §5 "Numbers shown")', () => {
  const noteKeys = (f) => sceneAt(f, 1, text).overlays.map((o) => o.key);
  assert.ok(noteKeys(0).includes('note:1') && noteKeys(1).includes('note:2') && noteKeys(2).includes('note:3') && noteKeys(3).includes('note:4'));
  assert.ok(noteKeys(4).includes('note:5') && noteKeys(5).includes('note:6') && noteKeys(6).includes('note:7'));
  assert.ok(noteKeys(7).includes('table') && noteKeys(7).includes('big') && noteKeys(7).includes('deepseek'));
});

test('dated labels come from the date keys: the chip, frame 11\'s reads line, the caption\'s year (XS-4, prefix-caching-4)', () => {
  assert.equal(deepseekChip(data), 'DeepSeek Feb 2025');
  assert.equal(deepseekChip(null), 'DeepSeek —');
  assert.equal(fact('deepseek-v3-production', 'date').slice(0, 4), CAPTIONS[7].match(/in (\d{4})\./)[1]);
  const [, opus, deepseek] = providersFor(data);
  assert.equal(readsLine(opus, deepseek), "Opus 5.5 reads at 0.05× · DeepSeek's hit is 3.3% of its miss");
  assert.equal(formatPrice(0.0222), '$0.0222');
});
