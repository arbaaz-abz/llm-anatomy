// Frames 1–6: one group of eight answers, from sampling to advantages to the no-spread branch (storyboard §5).
// Each frame is a pure function of its progress p (0 → 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { fixed, signed2 } from './format.js';
import { GROUP_SIZE } from './numbers.js';
import {
  GROUP, ALL_RIGHT, ROW_H, rowY, GUTTER_X, VERDICT_X, R_COL, A_COL, MARGIN_X, HEADER_BLOCK, CHIP_X0,
  seg, lerp, ease, fade, label, marginLines, advantageFill, chipRow, chipX, chipY, rowIndexes, promptChips, column, rewardFills,
  meanRule,
} from './stage.js';

const ROWS = Array.from({ length: GROUP_SIZE }, (_, i) => i);
// The verifier footnote under the checker block, wrapped to the right margin's width.
const FOOTNOTE = Object.freeze(['Real checkers parse a', 'boxed answer and compare', 'it symbolically; this', 'one reads the last token.']);
const ADVANTAGE_NOTE_Y = 36;
const ZERO_FILL = advantageFill(0);
const MEAN = GROUP.stats.mean; // 0.25
const rewards = GROUP.rewards;
const advantages = GROUP.advantages;
const centered = rewards.map((r) => r - MEAN); // R − mean: +0.75 on rows 1 and 5, −0.25 on the others
const countOf = (p) => Math.floor(p * GROUP_SIZE + 1e-9);

function policy(svg, state = 'active', opacity = 1) {
  fade(G.block(svg, { ...HEADER_BLOCK, label: 'policy: the model being trained', state }), opacity);
}

function checker(svg, state = 'active', opacity = 1) {
  fade(G.block(svg, { x: MARGIN_X - 4, y: 3, w: 164, h: 24, label: 'checker: last token = 56?', state }), opacity);
}

const verdicts = (svg, count, group = GROUP) => ROWS.slice(0, count).forEach((i) => G.verdict(svg, { x: VERDICT_X, y: rowY(i) + ROW_H / 2, ok: group.rewards[i] === 1 }));

// ---- frame 1: eight samples ----
export function drawFrame1(svg, p) {
  promptChips(svg);
  policy(svg);
  rowIndexes(svg);
  const filling = Math.min(countOf(p), GROUP_SIZE - 1);
  ROWS.forEach((i) => {
    const tokens = GROUP.rows[i].tokens;
    if (i < filling || p >= 1) chipRow(svg, tokens, i);
    else if (i === filling) chipRow(svg, tokens, i, { shown: Math.floor(seg(p * GROUP_SIZE - i, 0, 1) * (tokens.length + 1) - 1e-9) });
  });
  if (p >= 1) return;
  const y = rowY(filling) + ROW_H / 2;
  G.flow(svg, { from: [GUTTER_X, HEADER_BLOCK.y + HEADER_BLOCK.h], to: [GUTTER_X, y - 2], carry: 'token', progress: p * GROUP_SIZE - filling });
}

// ---- frame 2: the checker scores each answer ----
export function drawFrame2(svg, p) {
  promptChips(svg);
  policy(svg, 'idle');
  const scored = Math.min(Math.ceil(p * GROUP_SIZE - 1e-9), GROUP_SIZE);
  checker(svg, 'active', ease(seg(p, 0, 0.12)));
  FOOTNOTE.forEach((line, i) => label(svg, MARGIN_X - 2, 44 + i * 16, line, { opacity: ease(seg(p, 0, 0.12)) }));
  rowIndexes(svg);
  ROWS.forEach((i) => chipRow(svg, GROUP.rows[i].tokens, i));
  verdicts(svg, scored);
  column(svg, { origin: R_COL, values: rewards.map((r, i) => (i < scored ? r : null)), header: 'R', fills: rewardFills(rewards, scored), link: 'r' });
}

// ---- frame 3: the baseline ----
const BASELINE_AT = 0.2; // the checker and policy have faded; the mean rule slides up, the second column counts R → R − mean
export function drawFrame3(svg, p) {
  const gone = 1 - ease(seg(p, 0, BASELINE_AT));
  promptChips(svg);
  if (gone > 0) { policy(svg, 'idle', gone); checker(svg, 'active', gone); }
  rowIndexes(svg);
  ROWS.forEach((i) => chipRow(svg, GROUP.rows[i].tokens, i));
  verdicts(svg, GROUP_SIZE);
  column(svg, { origin: R_COL, values: rewards, header: 'R', fills: rewardFills(rewards), link: 'r' });
  const slide = ease(seg(p, BASELINE_AT, 1));
  column(svg, { origin: A_COL, values: rewards.map((r, i) => lerp(r, centered[i], slide)), header: 'R − mean', opacity: ease(seg(p, 0, BASELINE_AT)), link: 'a' });
  meanRule(svg, MEAN * slide, `mean ${fixed(MEAN * slide, 2)}`);
}

// ---- frame 4: divide by the spread ----
export function drawFrame4(svg, p) {
  const t = ease(seg(p, 0.1, 1));
  promptChips(svg);
  rowIndexes(svg);
  ROWS.forEach((i) => chipRow(svg, GROUP.rows[i].tokens, i));
  verdicts(svg, GROUP_SIZE);
  column(svg, { origin: R_COL, values: rewards, header: 'R', fills: rewardFills(rewards), link: 'r' });
  column(svg, { origin: A_COL, values: centered.map((c, i) => lerp(c, advantages[i], t)), header: 'A', link: 'a' });
  meanRule(svg, MEAN, `mean ${fixed(MEAN, 2)}`, 1 - ease(seg(p, 0, 0.1)));
  label(svg, MARGIN_X, ADVANTAGE_NOTE_Y, 'A = advantage', { opacity: ease(seg(p, 0, 0.1)) });
  marginLines(svg, [`mean ${fixed(MEAN, 2)}`, `std ${fixed(GROUP.stats.std, 2)}`, `Σ A = ${fixed(advantages.reduce((s, a) => s + a, 0) * t, 2)}`], { opacity: ease(seg(p, 0, 0.1)) });
}

// ---- frame 5 (key frame): the advantage flows back onto every token ----
const PUSH = Object.freeze({ stagger: 0.04, span: 0.6 });
function chipPassed(p, row, tokens, j) {
  const t = seg(p, row * PUSH.stagger, row * PUSH.stagger + PUSH.span);
  const dotX = lerp(A_COL.x, CHIP_X0, t);
  return t > 0 && dotX <= chipX(tokens, j) + G.tokenWidth(tokens[j]) / 2;
}

export function drawFrame5(svg, p) {
  promptChips(svg);
  rowIndexes(svg);
  verdicts(svg, GROUP_SIZE);
  column(svg, { origin: R_COL, values: rewards, header: 'R', fills: rewardFills(rewards), link: 'r' });
  column(svg, { origin: A_COL, values: advantages, header: 'A', link: 'a' });
  label(svg, MARGIN_X, ADVANTAGE_NOTE_Y, 'A = advantage');
  ROWS.forEach((i) => {
    const { tokens } = GROUP.rows[i];
    chipRow(svg, tokens, i, { fill: (j) => (chipPassed(p, i, tokens, j) ? advantageFill(advantages[i]) : null) });
  });
  ROWS.forEach((i) => {
    const t = seg(p, i * PUSH.stagger, i * PUSH.stagger + PUSH.span);
    if (t > 0 && t < 1) G.flow(svg, { from: [A_COL.x, rowY(i) + ROW_H / 2], to: [CHIP_X0 - 4, rowY(i) + ROW_H / 2], carry: 'gradient', progress: t });
  });
  const counts = pushCounts();
  marginLines(svg, [`mean ${fixed(MEAN, 2)}`, `std ${fixed(GROUP.stats.std, 2)}`, `Σ A = ${fixed(advantages.reduce((s, a) => s + a, 0), 2)}`, `Σ|A| = ${fixed(GROUP.totalPush, 2)}`]);
  marginLines(svg, [`${signed2(counts.up.advantage)} on ${counts.up.chips} chips`, `${signed2(counts.down.advantage)} on ${counts.down.chips} chips`], { y: 176, opacity: ease(seg(p, 0.5, 0.9)) });
}

// Chips carrying the right answers' advantage and the wrong answers' (frame 5's "per-chip A").
function pushCounts() {
  const chips = (sign) => GROUP.rows.filter((r) => Math.sign(r.advantage) === sign).reduce((n, r) => n + r.tokens.length, 0);
  return { up: { advantage: Math.max(...advantages), chips: chips(1) }, down: { advantage: Math.min(...advantages), chips: chips(-1) } };
}

// ---- frame 6: the what-if branch, every answer right ----
const SWAP = { from: 0, span: 0.25 }; // rows are replaced one by one by all-right answers
export function drawFrame6(svg, p) {
  const swapped = (i) => p >= SWAP.from + (i / GROUP_SIZE) * SWAP.span + 0.01;
  const stamp = seg(p, 0.3, 0.5);
  const fillIn = seg(p, 0.5, 0.75);
  const hatch = ease(seg(p, 0.75, 0.95));
  promptChips(svg);
  label(svg, MARGIN_X, ADVANTAGE_NOTE_Y, 'A = advantage');
  label(svg, HEADER_BLOCK.x, 15, 'what if all 8 were right?', { opacity: ease(seg(p, 0, SWAP.span)) });
  rowIndexes(svg);
  ROWS.forEach((i) => {
    if (!swapped(i)) {
      chipRow(svg, GROUP.rows[i].tokens, i, { fill: advantageFill(advantages[i]) });
      return;
    }
    chipRow(svg, ALL_RIGHT.rows[i].tokens, i, { fill: hatch > 0 ? ZERO_FILL : null, hatched: hatch > 0.5 });
  });
  const old = ROWS.filter((i) => !swapped(i));
  const stamped = ROWS.filter((i) => swapped(i) && stamp > 0 && i < Math.ceil(stamp * GROUP_SIZE - 1e-9));
  old.forEach((i) => G.verdict(svg, { x: VERDICT_X, y: rowY(i) + ROW_H / 2, ok: rewards[i] === 1 }));
  stamped.forEach((i) => G.verdict(svg, { x: VERDICT_X, y: rowY(i) + ROW_H / 2, ok: true }));
  const rValues = ROWS.map((i) => (!swapped(i) ? rewards[i] : i < Math.ceil(fillIn * GROUP_SIZE - 1e-9) ? 1 : null));
  column(svg, { origin: R_COL, values: rValues, header: 'R', fills: ROWS.map((i) => (rValues[i] == null ? null : rValues[i] === 1 ? 'ok' : 'bad')), link: 'r' });
  const aValues = ROWS.map((i) => (!swapped(i) ? advantages[i] : p >= 0.55 && i < Math.ceil(seg(p, 0.55, 0.8) * GROUP_SIZE - 1e-9) ? 0 : null));
  column(svg, { origin: A_COL, values: aValues, header: 'A', link: 'a' });
  const rise = ease(seg(p, 0.5, 0.8));
  meanRule(svg, lerp(MEAN, 1, rise), `mean ${fixed(lerp(MEAN, 1, rise), 2)}`, seg(p, 0.3, 0.5));
  const counts = pushCounts();
  const leaving = 1 - seg(p, 0, 0.1);
  if (leaving > 0) {
    marginLines(svg, [`mean ${fixed(MEAN, 2)}`, `std ${fixed(GROUP.stats.std, 2)}`, `Σ A = ${fixed(0, 2)}`, `Σ|A| = ${fixed(GROUP.totalPush, 2)}`], { opacity: leaving });
    marginLines(svg, [`${signed2(counts.up.advantage)} on ${counts.up.chips} chips`, `${signed2(counts.down.advantage)} on ${counts.down.chips} chips`], { y: 176, opacity: leaving });
  }
  const done = seg(p, 0.8, 1);
  marginLines(svg, ['std 0', 'Σ A = 0.00', 'Σ|A| = 0'], { y: 96, opacity: done });
  ['no signal: filtered', '(dynamic sampling)'].forEach((line, k) => label(svg, MARGIN_X, 322 + k * 18, line, { opacity: hatch }));
}
