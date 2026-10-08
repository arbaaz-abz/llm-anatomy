// agentic-rl frames 1–4: an episode in a sandbox, what is trained, the group table, and the judge branch.
import * as G from '@shared/glyphs.js';
import { ADVANTAGES, EPISODE, OBSERVATION_TOKENS, POLICY_TOKENS } from './numbers.js';
import { ofTotal } from './format.js';
import {
  CELL, CHIP_H, TABLE, arriving, chipFill, chipXs, groupTable, layer, leaving, litBlock, note, readout, seg, selectChip,
} from './stage.js';

const LABEL_X = 8;
const EPI = Object.freeze({ x: 64, prompt: 14, turn1: 72, obs: 130, turn2: 188 });
const PROMPT = Object.freeze(['7', '×', '8', '=', '?']);
const TURN1 = EPISODE.slice(0, 7).map((t) => t.text);
const OBS = EPISODE.slice(7, 10).map((t) => t.text);
const TURN2 = EPISODE.slice(10).map((t) => t.text);
const POLICY = Object.freeze({ x: 64, y: 262, w: 90, h: 40 });
const SANDBOX = Object.freeze({ x: 430, y: 262, w: 140, h: 40 });
const CALL_FLOW = Object.freeze({ from: [158, 272], to: [426, 272] });
const BACK_FLOW = Object.freeze({ from: [426, 292], to: [158, 292] });
const labelAt = (svg, y, str) => note(svg, LABEL_X, y + CHIP_H / 2 + 4, str);

// Chips of one episode row; `opacityOf(i)` fades a chip in, `hatchedOf(i)` hatches it.
function chipRow(svg, words, y, { opacityOf = () => 1, hatchedOf = () => false } = {}) {
  const xs = chipXs(words, EPI.x);
  words.forEach((text, i) => {
    const o = opacityOf(i);
    if (o > 0) G.token(layer(svg, o), { x: xs[i], y, text, hatched: hatchedOf(i) });
  });
}

// Chips typing in left to right across [a, b] of the frame.
const typing = (p, a, b, count) => (i) => seg(p, a + ((b - a) * i) / count, a + ((b - a) * (i + 1)) / count);

function episodeLabels(svg, opacity = 1) {
  const g = layer(svg, opacity);
  labelAt(g, EPI.prompt, 'prompt');
  labelAt(g, EPI.turn1, 'turn 1');
  labelAt(g, EPI.obs, 'sandbox');
  labelAt(g, EPI.turn2, 'turn 2');
}

export function drawFrame1(svg, p) {
  episodeLabels(svg);
  chipRow(svg, PROMPT, EPI.prompt);
  chipRow(svg, TURN1, EPI.turn1, { opacityOf: typing(p, 0.05, 0.4, TURN1.length) });
  const toSandbox = seg(p, 0.4, 0.55);
  const back = seg(p, 0.6, 0.75);
  litBlock(svg, POLICY, 'policy', Math.max(seg(p, 0.02, 0.1) - seg(p, 0.4, 0.45), seg(p, 0.85, 0.9)), 'idle');
  litBlock(svg, SANDBOX, 'sandbox (runs tools)', seg(p, 0.55, 0.6) - seg(p, 0.75, 0.8), 'idle');
  if (toSandbox > 0) G.flow(svg, { ...CALL_FLOW, carry: 'token', progress: toSandbox });
  note(svg, 292, 266, 'tool call', { anchor: 'middle' });
  if (back > 0) G.flow(svg, { ...BACK_FLOW, carry: 'token', progress: back });
  note(svg, 292, 310, 'observation', { anchor: 'middle' });
  chipRow(svg, OBS, EPI.obs, { opacityOf: typing(p, 0.7, 0.85, OBS.length) });
  const turn2 = seg(p, 0.85, 1);
  chipRow(svg, TURN2, EPI.turn2, { opacityOf: () => turn2 });
  if (turn2 > 0) selectChip(layer(svg, turn2), EPI.x, EPI.turn2, TURN2[0]);
  readout(svg, ['2 policy turns · 1 tool call'], { y: 346, opacity: seg(p, 0.9, 1) });
}

// Frame 2: the same chips; the three observation chips take the hatch one by one and the readout counts down.
export function drawFrame2(svg, p) {
  const gone = leaving(p); // frame 1's blocks and flows fade out as the hatch starts
  if (gone > 0) {
    const g = layer(svg, gone);
    litBlock(g, POLICY, 'policy', 0, 'idle');
    litBlock(g, SANDBOX, 'sandbox (runs tools)', 0, 'idle');
    G.flow(g, { ...CALL_FLOW, carry: 'token', progress: 1 });
    G.flow(g, { ...BACK_FLOW, carry: 'token', progress: 1 });
  }
  episodeLabels(svg);
  chipRow(svg, PROMPT, EPI.prompt);
  chipRow(svg, TURN1, EPI.turn1);
  const hatchedAt = [0.2, 0.35, 0.5];
  chipRow(svg, OBS, EPI.obs, { hatchedOf: (i) => p >= hatchedAt[i] });
  chipRow(svg, TURN2, EPI.turn2);
  selectChip(svg, EPI.x, EPI.turn2, TURN2[0]);
  const hatched = hatchedAt.filter((t) => p >= t).length;
  const obsEnd = chipXs(OBS, EPI.x).at(-1) + G.tokenWidth(OBS.at(-1));
  note(layer(svg, seg(p, 0.5, 0.65)), obsEnd + 14, EPI.obs + CHIP_H / 2 + 4, 'observation: masked');
  readout(svg, [`trained ${ofTotal(POLICY_TOKENS + hatchedAt.length - hatched, EPISODE.length)}`], { y: 270 });
  readout(svg, [`${POLICY_TOKENS} policy tokens · ${OBSERVATION_TOKENS} observation tokens`], { y: 288, opacity: seg(p, 0.65, 0.85) });
}

const COLUMN = Object.freeze({ x: 432, y: 44, stride: 26 });
const GRADIENT_FLOW = Object.freeze({ from: [TABLE.aX + CELL + 4, TABLE.y + CELL / 2], to: [COLUMN.x - 6, COLUMN.y + CELL / 2] });

// The episode as a column beside the table: observation chips hatched, the policy chips take row 1's advantage.
function episodeColumn(svg, p) {
  const flow = seg(p, 0.72, 0.88);
  const filled = p >= 0.9;
  EPISODE.forEach((t, i) => {
    const fill = filled && !t.observation ? chipFill(ADVANTAGES[0]) : undefined;
    G.token(svg, { x: COLUMN.x, y: COLUMN.y + i * COLUMN.stride, text: t.text, fill, hatched: t.observation });
  });
  note(svg, COLUMN.x, 14, 'an agent version of');
  note(svg, COLUMN.x, 28, 'row 1: same final 56');
  if (flow > 0) G.flow(svg, { ...GRADIENT_FLOW, carry: 'gradient', progress: flow });
}

export function drawFrame3(svg, p) {
  const g = layer(svg, arriving(p));
  const row = (i) => seg(p, 0.05 + 0.04 * i, 0.12 + 0.04 * i);
  const advantage = (i) => seg(p, 0.45 + 0.03 * i, 0.53 + 0.03 * i);
  groupTable(g, {
    chip: (r) => (advantage(r) >= 0.5 ? { fill: chipFill(ADVANTAGES[r]) } : {}),
    shown: (r) => ({ verdict: row(r), r: row(r), a: advantage(r) }),
  });
  episodeColumn(g, p);
  const sum = ADVANTAGES.reduce((s, a) => s + a, 0);
  readout(g, [`Σ A = ${Math.abs(sum) < 1e-9 ? 0 : sum.toFixed(2)}`], { x: COLUMN.x, y: 352, opacity: advantage(7) });
}

// ---- frame 4: the judge branch ----
const JUDGE = Object.freeze({ x: 20, y: 44, w: 120, h: 40 });
const RUBRIC = Object.freeze(['cites a source', 'answers the question asked', 'under the length budget']);
const ATTEMPTS = Object.freeze([{ name: 'attempt A', met: [true, true, true], x: 310 }, { name: 'attempt B', met: [false, true, false], x: 400 }]);
const RUBRIC_Y = 132;
const RUBRIC_STRIDE = 32;

export function drawFrame4(svg, p) {
  const g = layer(svg, arriving(p));
  note(g, 20, 22, 'branch: when nothing can be tested');
  litBlock(g, JUDGE, 'judge model', 1, 'idle');
  G.flow(g, { from: [80, 88], to: [80, 118], carry: 'token', progress: seg(p, 0.03, 0.3) });
  RUBRIC.forEach((line, k) => {
    const o = seg(p, 0.05 + 0.15 * k, 0.2 + 0.15 * k);
    note(layer(g, o), 20, RUBRIC_Y + k * RUBRIC_STRIDE + 4, line, { cls: 'g-readout' });
    ATTEMPTS.forEach((a) => {
      const t = seg(p, 0.5 + 0.1 * k, 0.6 + 0.1 * k);
      if (t > 0) G.verdict(layer(g, t), { x: a.x, y: RUBRIC_Y + k * RUBRIC_STRIDE, ok: a.met[k] });
    });
  });
  ATTEMPTS.forEach((a) => {
    note(layer(g, seg(p, 0.45, 0.55)), a.x, RUBRIC_Y - 22, a.name, { anchor: 'middle' });
    const score = seg(p, 0.85, 1);
    if (score > 0) G.block(layer(g, score), { x: a.x - 30, y: RUBRIC_Y + 3 * RUBRIC_STRIDE - 8, w: 60, h: 36, label: `${a.met.filter(Boolean).length} / ${RUBRIC.length}`, state: 'idle' });
  });
  note(layer(g, seg(p, 0.9, 1)), 14, 300, 'Kimi K3\'s judge makes an answer over its length budget lose automatically');
}
