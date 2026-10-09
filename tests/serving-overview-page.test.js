import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { formatShare } from '../shared/glyphs.js';
import { formatBytes, formatCount, formatDuration } from '../math/core.js';
import { kvCacheBytes } from '../math/memory.js';
import { stepTime, requestTimeline, freeHbmPerGpu, maxUsersPerGpu, hbmFor, RUNNING_EXAMPLE, TOY_REQUESTS } from '../math/serving.js';
import { simulateContinuous } from '../math/batching.js';
import { LESSON, lessonFor } from '../serving/concepts/serving-overview/content.js';
import { BELOW, FACT_ROWS } from '../serving/concepts/serving-overview/facts.js';
import { INITIAL_STATE, PROMPT_STOPS, OUTPUT_STOPS, arrowText, checkWork, prefillRule, timelineGeometry } from '../serving/concepts/serving-overview/format.js';
import { toyView, speeds, sharedUsers, timelineFor, hardwareEntry, FIGURE_WIDTH } from '../serving/concepts/serving-overview/toy-view.js';
import { tryThis } from '../serving/concepts/serving-overview/try-this.js';
import * as N from '../serving/concepts/serving-overview/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './serving-overview-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const title = (slug) => graph.concepts.find((c) => c.slug === slug).title;
const titled = (text) => text.replace(/\[\[([a-z0-9-]+)\]\]/g, (_, slug) => title(slug));
const view = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);
const h200 = data.hardware.entries.find((e) => e.id === 'h200');
// The rates behind the chips, straight from the shared step time (Review Focus 1).
const rateAt = (users) => 1 / stepTime({ ...RUNNING_EXAMPLE, tokens: users, seqs: users, context: 2048 }).timeS;
const USERS = maxUsersPerGpu(freeHbmPerGpu({ hbmBytes: hbmFor(h200).bytes, weightBytes: RUNNING_EXAMPLE.weightBytesPerGpu, gpus: 1 }), kvCacheBytes({ bytesPerToken: RUNNING_EXAMPLE.kvBytesPerToken, tokens: 2048 }));
const timeline = (patch) => {
  const s = { ...INITIAL_STATE, ...patch };
  return requestTimeline({ queueS: s.queue, promptTokens: s.prompt, model: RUNNING_EXAMPLE, outputTokens: s.output, decodeTokPerS: s.decodeRate === 'alone' ? rateAt(1) : rateAt(USERS) });
};

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('5 facts rows; every placeholder resolves; the filled rows read as the storyboard\'s §8', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 5);
  lesson.facts.rows.forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.doesNotMatch(fillText(row.claim, data), /—/, `row ${i + 1}`);
  });
  const rows = lesson.facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /DeepSeek-V4-Pro on GB300 NVL72 at 27 output tokens\/s per user \(ISL 8K \/ OSL 1K, FP4, disaggregated Dynamo \+ vLLM; InferenceX, measured 2026-05-22\)\.$/);
  assert.match(rows[1], /^A 128K-token prompt took 8\.6 s to its first token: DeepSeek-R1 on GB300 NVL72/);
  assert.match(rows[2], /: 56\.3% of DeepSeek's input tokens hit its KV cache \(V3\/R1 production, Feb 2025\)\.$/);
  assert.equal(rows[3], 'The engines inside a replica: vLLM, SGLang, TensorRT-LLM; orchestration by NVIDIA Dynamo (1.0, GA 2026-03-16) or llm-d (CNCF Sandbox 2026-03-12).');
  assert.equal(fillClaim(lesson.facts.rows[3].claim, data).reported, true, 'engines and llm-d are reported');
  assert.equal(FACT_ROWS[4].derived, true);
  assert.equal(fillClaim(FACT_ROWS[4].claim, data).sources.length, 0);
});

test('hook, intuition, notes under the stage and takeaways fill with nothing missing, and no "—"', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, lesson.intuitionNote, lesson.facts.framing, ...lesson.takeaways, lesson.toy.intro, lesson.animation.standIn,
    ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i)), ...lesson.math.notes];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.equal(lesson.animation.belowFor(42).length, 0);
  assert.equal(BELOW.length, CAPTIONS.length);
  assert.match(lesson.toy.intro, /^Every time here is a floor, not a measurement: one model on one GPU \(Llama-3\.1-70B in FP8 on an H200\)/);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('serving-overview')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('lessons are named by title in text: [[slug]] only after "see", "in" or as a named subject; never a bare slug on stage', () => {
  const lesson = lessonFor(data);
  const prose = [lesson.intuitionNote, lesson.toy.intro, lesson.animation.standIn, ...lesson.math.notes, ...BELOW.flat(), ...tryThis(data).map((t) => t.rest), speeds(data).note];
  prose.forEach((t) => assert.doesNotMatch(t, /\]\]'s|\(\[\[[a-z-]+\]\]\)/, t.slice(0, 60)));
  N.LESSON_MAP.forEach((m) => assert.deepEqual(m.titles, m.slugs.map(title), m.stop));
  assert.equal(N.TITLES.prefixCaching, title('prefix-caching'));
  assert.equal(N.TITLES.batching, title('batching'));
  assert.equal(N.TITLES.sampling, title('sampling'));
});

// ---- "Check my work" and the toy readouts, each equal to the shared function's output (Review Focus 1) ----

test('"Check my work" for the default state is the storyboard text, from requestTimeline\'s unrounded values', () => {
  assert.equal(view({}).checkWork, CHECK_WORK);
  assert.equal(checkWork(timeline({}), 500), CHECK_WORK);
  assert.equal(view({ decodeRate: 'shared', queue: 2 }).checkWork.split('\n')[0], 'TTFT = queue + prefill = 2 s + 141 ms = 2.14 s');
  assert.match(view({ decodeRate: 'shared', queue: 2 }).checkWork.split('\n')[1], /^total = TTFT \+ \(n − 1\) · TPOT = 2\.14 s \+ 499 · 29\.6 ms \(14\.8 s\) = 16\.9 s {3}\(/);
  assert.match(view({ output: 2, prompt: 3 }).checkWork, /= 14\.6 ms \+ 1 · 14\.7 ms \(14\.7 ms\) = 29\.3 ms/);
});

test('the chips\' rates are 1 ÷ stepTime at 2,048 tokens of context; sharing is the batch that fills an H200 (105 users)', () => {
  const s = speeds(data);
  assert.equal(USERS, 105);
  assert.equal(s.users, USERS);
  assert.equal(sharedUsers(hbmFor(h200).bytes), USERS);
  assert.equal(s.rates.alone, rateAt(1));
  assert.equal(s.rates.shared, rateAt(USERS));
  assert.equal(N.decodeRate(1), rateAt(1));
  assert.deepEqual(s.options, [
    { value: 'alone', label: '67.9 tok/s, alone on the GPU' },
    { value: 'shared', label: '33.7 tok/s, sharing with 104 others' },
  ]);
  assert.deepEqual([formatCount(rateAt(1)), formatCount(rateAt(USERS))], ['67.9', '33.7']);
});

test('the chip note prints the H200\'s memory with its basis word, read through hbmFor (Review Focus 3)', () => {
  const { note } = speeds(data);
  assert.equal(hbmFor(h200).basis, 'nominal');
  assert.match(note, new RegExp(`in an H200's ${formatBytes(hbmFor(h200).bytes)} \\(nominal\\)\\.$`));
  assert.match(note, /Sharing means 105 users, as many 2,048-token caches as fit beside the 70 GB of weights/);
  assert.match(note, /^Output tokens per second for one user/);
  assert.throws(() => hardwareEntry({ hardware: { entries: [] } }), RangeError);
});

test('default readouts equal requestTimeline and stepTime for the same inputs', () => {
  const v = view({});
  const tl = timeline({});
  assert.deepEqual(v.timeline, tl);
  assert.equal(tl.prefillS, stepTime({ ...RUNNING_EXAMPLE, tokens: 2000, seqs: 0, context: 0 }).timeS);
  assert.deepEqual(
    [v.readouts.prefill.value, v.readouts.ttft.value, v.readouts.tpot.value, v.readouts.total.value, v.readouts.decodeShare.value],
    ['141 ms', '141 ms', '14.7 ms', '7.49 s', '98.1%'],
  );
  assert.deepEqual(
    [v.readouts.prefill.value, v.readouts.ttft.value, v.readouts.tpot.value, v.readouts.total.value, v.readouts.decodeShare.value],
    [formatDuration(tl.prefillS), formatDuration(tl.ttftS), formatDuration(tl.tpotS), formatDuration(tl.e2eS), formatShare(tl.decodeShare)],
  );
  assert.equal(v.readouts.prefill.sub, 'one read of the weights or the math, whichever is longer (here: the math)');
  assert.equal(view({ prompt: 3 }).readouts.prefill.sub, 'one read of the weights or the math, whichever is longer (here: the weight read)');
  assert.equal(prefillRule('memory'), prefillRule('anything but compute'));
  assert.equal(v.widthNote, 'full width = 7.49 s (the whole request)');
});

test('every prompt stop prints the storyboard\'s prefill time (§6 worked examples)', () => {
  const printed = PROMPT_STOPS.map((prompt) => view({ prompt }).readouts.prefill.value);
  assert.deepEqual(printed, ['14.6 ms', '14.9 ms', '35.4 ms', '141 ms', '566 ms', '1.41 s', '9.06 s']);
  assert.deepEqual(OUTPUT_STOPS, [2, 50, 200, 500, 1000, 4000]);
});

test('the storyboard\'s timeline rows (§6 worked examples)', () => {
  const row = (patch) => {
    const r = view(patch).readouts;
    return [r.ttft.value, r.tpot.value, r.total.value, r.decodeShare.value];
  };
  assert.deepEqual(row({ prompt: 20_000 }), ['1.41 s', '14.7 ms', '8.76 s', '83.9%']);
  assert.deepEqual(row({ output: 50 }), ['141 ms', '14.7 ms', '863 ms', '83.6%']);
  assert.deepEqual(row({ decodeRate: 'shared' }), ['141 ms', '29.6 ms', '14.9 s', '99.1%']);
  assert.deepEqual(row({ decodeRate: 'shared', queue: 2 }), ['2.14 s', '29.6 ms', '16.9 s', '87.4%']);
  assert.deepEqual(row({ prompt: 3, output: 2 }), ['14.6 ms', '14.7 ms', '29.3 ms', '50.2%']);
});

test('an answer of 1 token makes the total equal TTFT (the toy\'s floor at its edge)', () => {
  const one = view({ output: 1 });
  assert.equal(one.timeline.e2eS, one.timeline.ttftS);
  assert.equal(one.readouts.total.value, one.readouts.ttft.value);
  assert.equal(one.figure.ticks, 0);
  assert.equal(one.figure.decodeLabel, '');
  assert.match(one.checkWork, /\+ 0 · 14\.7 ms \(0 s\) = 141 ms/);
});

test('the queue moves TTFT by exactly the queue seconds and never moves TPOT or prefill', () => {
  const base = view({});
  for (const queue of [0.5, 2, 5]) {
    const q = view({ queue });
    assert.equal(q.timeline.ttftS - base.timeline.ttftS, queue);
    assert.equal(q.timeline.tpotS, base.timeline.tpotS);
    assert.equal(q.timeline.prefillS, base.timeline.prefillS);
    assert.equal(q.readouts.ttft.value, formatDuration(base.timeline.prefillS + queue));
  }
});

// ---- the timeline bar: drawn to scale ----

test('the bar is drawn to scale: full width is the whole request, prefill ends at TTFT, one tick per token up to 200', () => {
  const v = view({});
  const { steps, ttftX, ticks, thinned } = v.figure;
  assert.equal(steps[0].kind, 'prefill');
  assert.equal(steps.at(-1).to, FIGURE_WIDTH);
  assert.ok(Math.abs(ttftX - FIGURE_WIDTH * (v.timeline.ttftS / v.timeline.e2eS)) < 1e-9);
  assert.deepEqual([ticks, thinned], [50, true]); // 499 decode steps: one tick per 10 tokens
  steps.slice(1).forEach((s, i) => assert.ok(s.from === steps[i].to && s.to > s.from, `step ${i + 1} follows the last`));
  assert.equal(v.tickNote, 'one tick per 10 answer tokens');
  const short = view({ output: 200 });
  assert.deepEqual([short.figure.ticks, short.figure.thinned, short.tickNote], [199, false, 'one tick per answer token']);
  assert.equal(view({ output: 4000 }).figure.ticks, 400);
  const queued = view({ queue: 2 }).figure;
  assert.deepEqual(queued.steps.slice(0, 2).map((s) => s.kind), ['queue', 'prefill']);
  assert.equal(queued.steps[0].to, queued.queueX);
  assert.throws(() => timelineGeometry(timeline({}), 500, 0), RangeError);
  assert.throws(() => timelineGeometry(timeline({}), 0, 100), RangeError);
});

// ---- try this ----

test('the try-this list prints the storyboard\'s numbers, computed from the same functions', () => {
  const printed = tryThis(data).map(({ prompt, insight, rest }) => titled(`${prompt} → Insight: ${insight}${rest}`));
  assert.deepEqual(printed, TRY_THIS);
  assert.deepEqual([arrowText(0.1415, 1.4149), arrowText(7.49, 8.76), arrowText(7.49, 0.863)], ['141 ms → 1.41 s', '7.49 → 8.76 s', '7.49 s → 863 ms']);
});

// ---- the stage's numbers ----

test('the stage prints the storyboard\'s frame numbers, from stepTime and requestTimeline', () => {
  assert.equal(formatDuration(N.PREFILL_S), '14.6 ms');
  assert.equal(N.PREFILL_S, stepTime({ ...RUNNING_EXAMPLE, tokens: 3, seqs: 0, context: 0 }).timeS);
  assert.equal(formatDuration(N.TPOT_S), '14.7 ms');
  assert.equal(formatCount(N.RATE_ALONE), '67.9');
  assert.equal(formatDuration(N.CHAT_TIMELINE.prefillS), '141 ms');
  assert.equal(formatDuration(N.CHAT_TIMELINE.e2eS), '7.49 s');
  assert.equal(N.PROMPT.length, 3);
  assert.equal(N.DECODE_CONTEXT, 2048);
  assert.equal(N.H200_NOMINAL_BYTES, h200.facts.hbm_gb.value * 1e9);
  assert.equal(N.WEIGHTS_FILL, RUNNING_EXAMPLE.weightBytesPerGpu / hbmFor(h200).bytes);
});

test('frame 9 replays simulateContinuous on TOY_REQUESTS with 3 seats: at step 4, A, C and D advance; B is done', () => {
  assert.deepEqual(N.BATCH, simulateContinuous({ requests: TOY_REQUESTS, seats: 3 }));
  assert.deepEqual(N.batchAt(N.BATCH_STEP), { advancing: ['A', 'C', 'D'], done: ['B'] });
  assert.deepEqual(N.batchAt(3), { advancing: ['A', 'C'], done: ['B'] });
  assert.equal(N.TOY_REQUESTS, TOY_REQUESTS);
});

test('inputs are not mutated', () => {
  const state = Object.freeze({ ...INITIAL_STATE, queue: 2, decodeRate: 'shared' });
  const frozen = JSON.stringify(data);
  toyView(state, data);
  tryThis(data);
  lessonFor(data);
  timelineFor(state, speeds(data).rates);
  assert.equal(JSON.stringify(data), frozen);
  assert.deepEqual(TOY_REQUESTS.map((r) => r.id), ['A', 'B', 'C', 'D']);
});
