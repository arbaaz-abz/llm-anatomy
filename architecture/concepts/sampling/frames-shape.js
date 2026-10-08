// Frames 4–7: temperature, top-k, top-p and 20 seeded draws (storyboard §5). Each frame is a pure function of its
// progress p (0 → 1); the end of frame n is the start of frame n + 1 (what leaves fades out over [0, HANDOFF]).
import * as G from '@shared/glyphs.js';
import { VOCAB, topP, drawSamples } from '@math/sampling.js';
import { fmt3, trimNumber } from './format.js';
import { barBody } from './frames-pick.js';
import {
  CELL, ROW_X, ROW_W, RIGHT_X, SHAPE_Y, NOTE_PAD, SCORE_CELLS, HANDOFF, RESULT, seg, lerp, ease, arriving, leaving, layer, note,
  chips, wordLabels, fiveRow, cellsAt, fiveCells, probsAt, othersNote, temperatureText,
} from './stage.js';

const BEFORE = cellsAt(1);
const ROW_MID = CELL / 2;
const FILTER = Object.freeze({ k3: RESULT.k3, p7: RESULT.p7 }); // at T = 1 both keep on, "." and and
const RESCALED = Object.freeze(fiveCells(FILTER.k3.probs)); // 0.547, 0.331, 0.122, 0, 0
const KEPT = 3; // survivors of top-k 3 and of top-p 0.7 at T = 1
const SLOT = [{ slot: 'five', text: '?' }];
const flags = (theCut, othersCut) => [false, false, false, theCut, othersCut];
const cumulative = (cells) => cells.reduce((acc, v) => [...acc, (acc.at(-1) ?? 0) + v], []);
const SUMS = Object.freeze(cumulative(BEFORE.slice(0, KEPT)));

// ---- the scene frames 4–6 share: scores ÷ T, the T = 1 row ("before") and the row after the change ("after") ----
function zRow(svg, temperature, opacity) {
  if (opacity <= 0) return;
  const inner = layer(svg, opacity);
  wordLabels(inner, SHAPE_Y.scores - 8);
  fiveRow(inner, { y: SHAPE_Y.scores, values: SCORE_CELLS.map((s) => s / temperature), kind: 'score', name: 'scores ÷ T', link: 'z' });
  note(inner, RIGHT_X, SHAPE_Y.scores + ROW_MID, temperatureText(temperature), { link: 't' });
}

function beforeRow(svg, labelOpacity) {
  if (labelOpacity > 0) wordLabels(svg, SHAPE_Y.before - 8, labelOpacity);
  fiveRow(svg, { y: SHAPE_Y.before, values: BEFORE, kind: 'prob', name: 'before', link: 'p' });
  note(svg, RIGHT_X, SHAPE_Y.before + ROW_MID, 'T = 1');
}

function afterRow(svg, { cells, hatch = null, probs }) {
  fiveRow(svg, { y: SHAPE_Y.after, values: cells, kind: 'prob', name: 'after', link: 'p', hatch });
  note(svg, ROW_X + ROW_W, SHAPE_Y.after + CELL + NOTE_PAD, othersNote(probs), { anchor: 'end' });
}

// The two lines to the right of the after row: which filter, and what the survivors are rescaled by.
function filterNotes(svg, { title, titleOpacity = 1, keptOpacity }) {
  note(svg, RIGHT_X, SHAPE_Y.after + ROW_MID - 10, title, { opacity: titleOpacity });
  note(svg, RIGHT_X, SHAPE_Y.after + ROW_MID + 10, `kept ${fmt3(FILTER.k3.mass)} → rescaled to 1`, { opacity: keptOpacity });
}

// ---- frame 4: temperature ----
// T goes 1 → 0.5, holds, then 0.5 → 2; every intermediate row is the real softmax at that temperature.
const temperatureAt = (p) => (p < 0.5 ? lerp(1, 0.5, ease(seg(p, 0.15, 0.4))) : lerp(0.5, 2, ease(seg(p, 0.55, 0.85))));

export function temperatureBody(svg, p, opacity = 1) {
  const temperature = temperatureAt(p);
  const inner = layer(svg, opacity);
  zRow(inner, temperature, 1);
  beforeRow(inner, 0);
  afterRow(inner, { cells: cellsAt(temperature), probs: probsAt(temperature) });
}

export function drawFrame4(svg, p) {
  const swap = seg(p, 0, HANDOFF);
  chips(svg, [{ slot: 'five', text: '.', opacity: 1 - swap, mark: 0 }, { slot: 'five', text: '?', opacity: swap }]);
  if (p < HANDOFF) barBody(svg, 1, leaving(p));
  temperatureBody(svg, p, arriving(p));
}

// ---- frame 5: top-k keeps the three most likely tokens and rescales them ----
// The after row follows T from 2 back to 1, the cut cells hatch from the right, then the survivors grow by 1 / 0.714.
export function topKBody(svg, p, opacity = 1) {
  const inner = layer(svg, opacity);
  const temperature = lerp(2, 1, ease(seg(p, 0, 0.2)));
  const zOpacity = 1 - seg(p, 0.2, 0.3);
  zRow(inner, temperature, zOpacity);
  beforeRow(inner, 1 - zOpacity);
  const cut = { others: p >= 0.35, the: p >= 0.45 };
  const grow = ease(seg(p, 0.55, 0.85));
  const cells = cellsAt(temperature).map((v, i) => {
    if ((i === 4 && cut.others) || (i === 3 && cut.the)) return 0;
    return i < KEPT ? lerp(v, RESCALED[i], grow) : v;
  });
  afterRow(inner, { cells, hatch: flags(cut.the, cut.others), probs: cut.others ? FILTER.k3.probs : probsAt(temperature) });
  filterNotes(inner, { title: 'top-k: k = 3', titleOpacity: seg(p, 0.3, 0.4), keptOpacity: seg(p, 0.55, 0.65) });
}

export function drawFrame5(svg, p) {
  chips(svg, SLOT);
  topKBody(svg, p);
}

// ---- frame 6: top-p adds up the sorted probabilities until they reach p ----
const P = 0.7;
const SUM_AT = Object.freeze([0.2, 0.32, 0.44]); // when each running total appears
const BOTTOM_Y = 330;
const keptAt = (temperature) => topP(probsAt(temperature), P).kept;

// The after row starts as frame 5 ended (top-3 hatched), clears, then top-p cuts and rescales the same three tokens.
function topPCells(p) {
  const reset = ease(seg(p, 0, 0.15));
  const grow = ease(seg(p, 0.8, 1));
  const cut = { others: p < 0.05 || p >= 0.62, the: p < 0.05 || p >= 0.7 };
  const cells = BEFORE.map((v, i) => {
    if ((i === 4 && cut.others) || (i === 3 && cut.the)) return 0;
    if (i >= KEPT) return lerp(0, v, reset);
    return p < 0.8 ? lerp(RESCALED[i], v, reset) : lerp(v, RESCALED[i], grow);
  });
  return { cells, hatch: flags(cut.the, cut.others) };
}

function runningSums(svg, p) {
  const y = SHAPE_Y.before + CELL + 12;
  note(svg, ROW_X - 10, y, 'sum', { anchor: 'end', opacity: seg(p, 0.18, 0.23) });
  SUMS.forEach((s, i) => note(svg, ROW_X + i * CELL + CELL / 2, y, fmt3(s), { anchor: 'middle', opacity: seg(p, SUM_AT[i], SUM_AT[i] + 0.05) }));
  note(svg, RIGHT_X, y, `${fmt3(SUMS[KEPT - 1])} ≥ ${P}: stop`, { opacity: seg(p, 0.5, 0.58) });
}

export function topPBody(svg, p, opacity = 1) {
  const inner = layer(svg, opacity);
  beforeRow(inner, 1);
  const { cells, hatch } = topPCells(p);
  afterRow(inner, { cells, hatch, probs: hatch[4] ? FILTER.p7.probs : probsAt(1) });
  runningSums(inner, p);
  filterNotes(inner, { title: 'top-k: k = 3', titleOpacity: 1 - seg(p, 0, 0.1), keptOpacity: p < 0.8 ? 1 - seg(p, 0, 0.1) : seg(p, 0.8, 0.9) });
  note(inner, RIGHT_X, SHAPE_Y.after + ROW_MID - 10, `top-p: p = ${P}`, { opacity: seg(p, 0.1, 0.2) });
  note(inner, ROW_X, BOTTOM_Y, `at T = 0.5 it keeps ${keptAt(0.5)}; at T = 2 it keeps ${keptAt(2)}`, { opacity: seg(p, 0.15, 0.3) });
}

export function drawFrame6(svg, p) {
  chips(svg, SLOT);
  topPBody(svg, p);
}

// ---- frame 7: 20 seeded draws at T = 1, then at T = 0.5 ----
const DRAWS = 20;
const SEED = 1;
const GROUPS = Object.freeze([
  { x: 30, temperature: 1, from: 0.12, to: 0.45 },
  { x: 300, temperature: 0.5, from: 0.55, to: 0.9 },
].map((g) => ({ ...g, draws: drawSamples(probsAt(g.temperature), DRAWS, SEED) })));
const BARS = Object.freeze({ y: 92, w: 260, h: 120, labelsY: 226, firstY: 268 }); // 52 px a bar: "the" and "12 others" stay apart
const CATEGORY_LABELS = Object.freeze(['on', '.', 'and', 'the', '12 others']);
const NAMED_COUNT = 4;

// Draws per category: the four named words, then "everything else" (the 12 others).
const categoryCounts = (draws) => {
  const named = Array.from({ length: NAMED_COUNT }, (_, c) => draws.filter((d) => d === VOCAB.indexOf(['on', '.', 'and', 'the'][c])).length);
  return [...named, draws.length - named.reduce((a, b) => a + b, 0)];
};

function drawGroup(svg, group, p) {
  const shown = Math.round(DRAWS * seg(p, group.from, group.to));
  const first = group.draws.slice(0, Math.min(shown, 8));
  const opacity = seg(p, group.from - 0.05, group.from);
  note(svg, group.x, BARS.y - 18, `T = ${trimNumber(group.temperature, 2)}, seed ${SEED}`, { opacity });
  G.bars(layer(svg, opacity), { x: group.x, y: BARS.y, w: BARS.w, h: BARS.h, values: categoryCounts(group.draws.slice(0, shown)), labels: CATEGORY_LABELS, max: DRAWS, label: `draws at T = ${group.temperature}` });
  note(svg, group.x, BARS.labelsY + 22, 'first eight draws', { opacity });
  note(svg, group.x, BARS.firstY, first.map((i) => VOCAB[i]).join(' ') || ' ', { opacity });
}

export function drawsBody(svg, p, opacity = 1) {
  const inner = layer(svg, opacity);
  GROUPS.forEach((group) => drawGroup(inner, group, p));
}

export function drawFrame7(svg, p) {
  chips(svg, SLOT);
  if (p < HANDOFF) topPBody(svg, 1, leaving(p));
  drawsBody(svg, p);
}
