import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../architecture/concepts/multimodal/content.js';
import {
  INITIAL_STATE, modelFacts, initialState, applyPreset, applySetting, sideValues, snapSide, counts, checkWork, shareText,
  imagesThatFit, framesOf, int, PHONE, WIDE, VIDEO_FRAME, CONTEXT_1M, CONTEXT_256K, TOY_IMAGE,
} from '../architecture/concepts/multimodal/format.js';
import { toyView, specLine, tryThis } from '../architecture/concepts/multimodal/toy-view.js';
import { MERGE_GROUPS, MERGED, IMAGE_TOKENS, Z1, groupOf, slotOf, FOLLOWED, TOY_COUNT, PATCHES, EAR } from '../architecture/concepts/multimodal/numbers.js';
import { visionTokens } from '../math/vision.js';
import { sharePct } from '../math/memory.js';
import { CAPTIONS, CHECK_WORK } from './multimodal-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const facts = modelFacts(data);
const lesson = lessonFor(data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lesson), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(lesson.animation.steps.map((s) => s.caption), CAPTIONS);
});

test('7 facts rows; every placeholder in rows and prose resolves, and nothing prints "—"', () => {
  assert.equal(lesson.facts.rows.length, 7);
  lesson.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  const below = [0, 2, 3, 4, 6, 9].flatMap((i) => lesson.animation.belowFor(i));
  const prose = [lesson.hook, ...lesson.intuition, lesson.animation.standIn, lesson.facts.framing, ...below, ...lesson.takeaways, ...lesson.math.notes];
  [...prose, ...lesson.facts.rows.map((r) => r.claim)].forEach((text) => {
    assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40));
    assert.ok(!fillText(text, data).includes('—'), text.slice(0, 40));
  });
});

test('the rows print the storyboard\'s numbers from the data; the omitted DeepSeek width and layer count of the projector are not typed', () => {
  const rows = lesson.facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /^Kimi K3: native multimodal.*401M parameters, 27 layers, patch 14.*merges 2 × 2; images up to 3,584 × 3,584 pixels/);
  assert.match(rows[1], /^MiniMax-M3: native multimodal.*ViT of 32 layers, width 1,280, patch 14; images up to 2,016 × 2,016 pixels/);
  assert.match(rows[2], /a 1\.6B-parameter vision encoder/);
  assert.match(rows[3], /32 layers, merges 3 × 3, a small MLP projector/);
  assert.ok(!/1,?024|2-layer/.test(rows[3]));
  assert.match(rows[4], /DeepSeek-V4-Pro \(text;.*GLM-5\.3 \(text\), gpt-oss-120b \(text\) and Qwen3\.8 \(text;/);
  assert.match(rows[6], /factorized spatial-temporal attention \+ temporal pooling/);
  assert.match(rows[5], /^The adapter recipe, LLaVA \(2023\): /);
  assert.equal(lesson.facts.rows[5].derived, undefined, 'the LLaVA row cites its paper entry, so it carries a source link');
  assert.match(rows[3], /^DeepSeek-V4\.1-Flash: the first/);
  assert.match(rows[6], /^Video in Kimi K3: /);
});

test('reported chips: only Mistral\'s encoder and DeepSeek-V4.1-Flash; no row prints a reported release year (X-1)', () => {
  const reported = lesson.facts.rows.map((r) => fillClaim(r.claim, data).reported);
  assert.deepEqual(reported, [false, false, true, true, false, false, false]);
  lesson.facts.rows.forEach((r) => assert.ok(!/(kimi-k3|minimax-m3|deepseek-v4\.1-flash)\.release_date/.test(r.claim), r.claim.slice(0, 40)));
  assert.ok(fillClaim(lesson.facts.rows[5].claim, data).sources.includes('https://arxiv.org/abs/2304.08485'));
  assert.match(fillText(lesson.animation.belowFor(9)[0], data), /LLaVA-style \(2023\)/);
  assert.equal(fillText(lesson.animation.belowFor(4)[0], data), 'The projector is a small MLP in Kimi K3 and, reportedly, in DeepSeek-ViT.');
  assert.equal(fillClaim('{mistral-large-4.vision_encoder_params|count}', data).reported, true);
  assert.equal(fillClaim('{deepseek-v4.1-flash.vision_merge}', data).reported, true);
  assert.equal(fillClaim('{kimi-k3.vision_encoder_params|count} {kimi-k3.max_image_side}', data).reported, false);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('multimodal')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the captions\' numbers equal the data and math/vision.js (README lessons 16 and 29)', () => {
  assert.equal(facts.patch, 14);
  assert.equal(facts.merge, 2);
  assert.equal(facts.maxSide, 3584);
  assert.equal(facts.deepseekMerge, 3);
  assert.equal(visionTokens({ ...PHONE, patch: facts.patch, merge: facts.merge }).tokensPerFrame, 1296);
  assert.equal(visionTokens({ width: facts.maxSide, height: facts.maxSide, patch: facts.patch, merge: facts.merge }).tokensPerFrame, 16384);
  assert.match(CAPTIONS[6], /1,008-pixel square photo costs 1,296 tokens.*16,384/);
  assert.match(CAPTIONS[0], /real models use 14 by 14/);
  assert.match(CAPTIONS[3], /Kimi K3 merges 2 by 2; DeepSeek's newest encoder merges 3 by 3/);
});

test('the fallback facts equal the data, so the pure functions and the page agree', () => {
  assert.deepEqual({ ...modelFacts(null) }, { ...facts });
  assert.deepEqual({ ...INITIAL_STATE }, { ...initialState(facts) });
});

test('"Check my work" for the default state is the storyboard line', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.equal(toyView(INITIAL_STATE, facts).checkWork, CHECK_WORK);
});

test('"Check my work" generalizes: wide photo, no merge, video, toy image', () => {
  assert.equal(checkWork({ ...INITIAL_STATE, height: 504 }), '1008 ÷ 14 = 72 · 504 ÷ 14 = 36 · 72 × 36 = 2,592 · ÷ (2 × 2) = 648');
  assert.equal(checkWork({ ...INITIAL_STATE, merge: 1 }), '1008 ÷ 14 = 72 · 72 × 72 = 5,184 · no merge = 5,184');
  assert.equal(checkWork({ ...INITIAL_STATE, width: 448, height: 448, media: 'video' }), '448 ÷ 14 = 32 · 32 × 32 = 1,024 · ÷ (2 × 2) = 256 · × 120 frames = 30,720');
  assert.equal(checkWork({ ...INITIAL_STATE, ...TOY_IMAGE, preset: 'toy' }), '16 ÷ 4 = 4 · 4 × 4 = 16 · ÷ (2 × 2) = 4');
});

test('toy outputs equal math/vision.js (the six readouts at the default state)', () => {
  const v = toyView(INITIAL_STATE, facts);
  assert.deepEqual([v.grid, v.perFrame, v.tokens, v.share, v.fit], ['72 × 72 = 5,184', '1,296', '1,296', '0.12%', '809']);
  assert.equal(v.shareNote, 'of 1,048,576 tokens');
  assert.equal(v.fitLabel, 'Images of this size that fit');
  assert.equal(v.showVideo, false);
});

test('try this 1: merge none 5,184; the largest input 16,384, 1.6% of 1M, 64 fit', () => {
  assert.equal(toyView({ ...INITIAL_STATE, merge: 1 }, facts).tokens, '5,184');
  const largest = applySetting({ ...INITIAL_STATE }, { width: facts.maxSide, height: facts.maxSide }, facts);
  const v = toyView(largest, facts);
  assert.deepEqual([v.tokens, v.share, v.fit], ['16,384', '1.6%', '64']);
});

test('try this 2: 1,008 × 504 costs 648 tokens, a squashed square 1,296', () => {
  assert.equal(toyView({ ...INITIAL_STATE, ...WIDE }, facts).tokens, '648');
  assert.equal(toyView({ ...INITIAL_STATE, ...PHONE }, facts).tokens, '1,296');
});

test('try this 3: video at 448 × 448: a minute, ten minutes at 1 fps, an hour (does not fit)', () => {
  const base = { ...INITIAL_STATE, ...VIDEO_FRAME, media: 'video' };
  const minute = toyView(base, facts);
  assert.deepEqual([minute.perFrame, minute.tokens, minute.share, minute.frames], ['256', '30,720', '2.9%', '120 (60 s at 2 per second)']);
  assert.equal(minute.fitLabel, 'Frames that fit');
  const ten = toyView({ ...base, seconds: 600, fps: 1 }, facts);
  assert.deepEqual([ten.tokens, ten.share], ['153,600', '14.6%']);
  assert.equal(toyView({ ...base, seconds: 600, fps: 1, context: CONTEXT_256K }, facts).share, '58.6%');
  const hour = toyView({ ...base, seconds: 3600 }, facts);
  assert.deepEqual([hour.tokens, hour.share, hour.shareNote, hour.tokensApprox], ['1,843,200', '175.8%', 'does not fit', '≈ 1.84M']);
});

test('shares use sharePct: one decimal from 1%, two below, "<0.01%" for dust', () => {
  assert.deepEqual([1296, 16384, 30720, 153600, 1843200].map((t) => shareText(t, CONTEXT_1M)), ['0.12%', '1.6%', '2.9%', '14.6%', '175.8%']);
  assert.equal(shareText(1, CONTEXT_1M), '<0.01%');
  assert.equal(shareText(0, CONTEXT_1M), '0.00%');
  assert.equal(sharePct(1296, CONTEXT_1M, { decimals: 2 }), 0.12);
});

test('imagesThatFit and framesOf', () => {
  assert.equal(imagesThatFit(CONTEXT_1M, 16384), 64);
  assert.equal(imagesThatFit(CONTEXT_1M, 1296), 809);
  assert.equal(framesOf({ media: 'image', seconds: 60, fps: 2 }), 1);
  assert.equal(framesOf({ media: 'video', seconds: 3600, fps: 2 }), 7200);
});

test('slider stops are multiples of patch × merge inside [112, the largest side], so every count is whole', () => {
  [[14, 2], [14, 1], [14, 3], [16, 2], [16, 3], [16, 1]].forEach(([patch, merge]) => {
    const values = sideValues({ patch, merge, maxSide: facts.maxSide });
    assert.ok(values[0] >= 112 && values[0] < 112 + patch * merge, `${patch}/${merge} first stop`);
    assert.ok(values.at(-1) <= facts.maxSide && values.at(-1) > facts.maxSide - patch * merge, `${patch}/${merge} last stop`);
    values.forEach((v) => assert.doesNotThrow(() => visionTokens({ width: v, height: v, patch, merge })));
  });
  assert.equal(sideValues({ patch: 14, merge: 2, maxSide: 3584 }).length, 125);
  assert.equal(snapSide(1000, { patch: 14, merge: 2, maxSide: 3584 }), 1008);
});

test('presets: Kimi K3, the 3 × 3 chip (patch kept), the toy image and back; editing makes the preset custom', () => {
  const deepseek = applyPreset(INITIAL_STATE, 'deepseek', facts);
  assert.deepEqual([deepseek.preset, deepseek.patch, deepseek.merge, deepseek.width], ['deepseek', 14, 3, 1008]);
  assert.equal(toyView(deepseek, facts).tokens, '576');
  const toy = applyPreset(INITIAL_STATE, 'toy', facts);
  assert.deepEqual([toy.width, toy.height, toy.patch, toy.merge], [16, 16, 4, 2]);
  assert.equal(toyView(toy, facts).tokens, String(TOY_COUNT.tokens));
  assert.equal(toyView(toy, facts).showSize, false);
  const back = applyPreset(toy, 'kimi', facts);
  assert.deepEqual([back.width, back.patch, back.merge], [1008, 14, 2]);
  const custom = applySetting(INITIAL_STATE, { patch: 16 }, facts);
  assert.equal(custom.preset, 'custom');
  assert.equal(custom.width % 32, 0);
  assert.equal(counts(custom).patches % 4, 0);
});

test('the spec line names the preset; DeepSeek\'s says the patch size is not in the data', () => {
  assert.match(specLine(INITIAL_STATE, facts), /^Kimi K3 settings: patch 14, 2 × 2 merge, images up to 3,584 pixels a side\.$/);
  assert.match(specLine(applyPreset(INITIAL_STATE, 'deepseek', facts), facts), /\(reported\).*keeps patch 14/);
});

test('every "try this" number is the toy\'s own', () => {
  const [one, two, three] = tryThis(facts);
  assert.match(one.prompt, /1,296 tokens.*5,184.*3,584 × 3,584: 16,384 tokens, 1\.6% of 1M; 64/);
  assert.match(two.prompt, /648 tokens.*1,296/);
  assert.match(three.prompt, /30,720.*153,600 \(14\.6% of 1M; 58\.6% of 262,144\).*1,843,200/);
});

test('the stand-in image: patch 6 is decoder-anatomy\'s ear, patch 6 lands in group 1 slot 4, merged vectors have 32 numbers', () => {
  assert.deepEqual([...PATCHES[FOLLOWED]], [...EAR]);
  assert.deepEqual([groupOf(FOLLOWED), slotOf(FOLLOWED)], [0, 3]);
  assert.deepEqual(MERGE_GROUPS[0], [0, 1, 4, 5]);
  assert.deepEqual(MERGED.map((r) => r.length), [32, 32, 32, 32]);
  assert.deepEqual(IMAGE_TOKENS.map((r) => r.length), [8, 8, 8, 8]);
  assert.deepEqual(MERGED[0].slice(24), Z1[5], 'the last slot of img₁ is patch 6\'s encoded vector');
});

test('int prints thousands separators and the toy image has 4 tokens', () => {
  assert.equal(int(1843200), '1,843,200');
  assert.equal(TOY_COUNT.tokens, 4);
});

test('format.js never mutates its inputs', () => {
  const before = JSON.stringify(INITIAL_STATE);
  applyPreset(INITIAL_STATE, 'toy', facts);
  applySetting(INITIAL_STATE, { merge: 3 }, facts);
  toyView(INITIAL_STATE, facts);
  assert.equal(JSON.stringify(INITIAL_STATE), before);
  const dataBefore = JSON.stringify(data);
  lessonFor(data);
  assert.equal(JSON.stringify(data), dataBefore);
});
