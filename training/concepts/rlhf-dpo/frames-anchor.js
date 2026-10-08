// rlhf-dpo frames 5–8: the clip range switches an update off; the reference model; reward hacking; the KL term.
import * as G from '@shared/glyphs.js';
import { ANSWER_FRESH, CANDIDATES, ROWS, CLIP_LINE, EPSILON, ADVANTAGE, RATIO_START, RATIO_MOVED, RATIO_SAFE, CLIP, KL, TOTALS, REWARD_HONEST, REWARD_HACKED, BETA_RLHF, HACKED_ANSWER } from './numbers.js';
import { fmt1, fmt2, fmt3 } from './format.js';
import { klPenalizedReward } from '@math/preference.js';
import { lerp, seg, typed, layer, note, select, linked, signed } from './stage.js';

const CELL43 = 48; // a probability cell; 4 × 48 = 192 px per row, wide enough for the header words ('Rayleigh' is 53 px)
const ROWS_X = 112;
const prob = (v) => fmt2(v);

// ---- frame 5: the clip range ----
const CLIP_AT = Object.freeze({ x: 200, y: 84, w: 350 });
const TOKEN_AT = Object.freeze({ x: 34, y: 66 });
const objectiveText = (r, out) => `min(${fmt2(r)} · ${fmt1(ADVANTAGE)}, ${fmt1(1 + EPSILON)} · ${fmt1(ADVANTAGE)}) = ${fmt2(out.objective)}, ${out.clipped ? 'clipped' : 'not clipped'}`;

export function drawFrame5(svg, p) {
  const slide = seg(p, 0.15, 0.7);
  const ratio = lerp(RATIO_START, RATIO_MOVED, slide);
  const off = ratio > CLIP_LINE.band[1];
  note(svg, TOKEN_AT.x, TOKEN_AT.y - 14, "the answer's last token");
  G.token(svg, { ...TOKEN_AT, text: ANSWER_FRESH[ANSWER_FRESH.length - 1], hatched: off });
  select(svg, { x: TOKEN_AT.x, y: TOKEN_AT.y, w: G.tokenWidth(ANSWER_FRESH[ANSWER_FRESH.length - 1]), h: 24 });
  G.clipLine(svg, { ...CLIP_AT, lo: CLIP_LINE.lo, hi: CLIP_LINE.hi, band: CLIP_LINE.band, marker: ratio, label: CLIP_LINE.label });
  const lines = [
    [`ε = ${fmt1(EPSILON)} · band [${fmt1(CLIP_LINE.band[0])}, ${fmt1(CLIP_LINE.band[1])}]`, 'g-label'],
    [`r = ${fmt2(RATIO_MOVED)}, beat it by ${signed(ADVANTAGE)}`, ''],
    [objectiveText(RATIO_MOVED, CLIP.moved), ''],
    ['past the band the gradient switches off (the token is hatched)', 'g-label'],
    [`at r = ${fmt2(RATIO_SAFE)}: ${objectiveText(RATIO_SAFE, CLIP.safe)}`, ''],
  ];
  lines.forEach(([text, cls], i) => note(svg, 34, 170 + i * 22, typed(text, seg(p, 0.7 + i * 0.05, 0.8 + i * 0.05)), { cls }));
}

// ---- frames 6–8: the three probability rows ----
const HEADER_Y = 42;
const ROW_Y = Object.freeze({ reference: 52, policy: 104, hacked: 156 });
const ROW_LABEL = Object.freeze({ reference: 'reference', policy: 'policy', hacked: 'hacked policy' });
const VALUE_MAX_ABS = 1;

function headers(parent) {
  CANDIDATES.forEach((word, i) => note(parent, ROWS_X + i * CELL43 + CELL43 / 2 + 0, HEADER_Y, word, { anchor: 'middle' }));
}

function probRow(parent, key, values) {
  const link = key === 'reference' ? 'ref' : key === 'policy' ? 'pol' : null;
  const host = link ? linked(parent, link, { x: ROWS_X, y: ROW_Y[key], w: 4 * CELL43, h: CELL43 }) : parent;
  G.vector(host, { x: ROWS_X, y: ROW_Y[key], values, cell: CELL43, orient: 'row', maxAbs: VALUE_MAX_ABS, label: ROW_LABEL[key], format: prob });
}
const policyMark = (parent, opacity = 1) => select(parent, { x: ROWS_X - 2, y: ROW_Y.policy - 2, w: 4 * CELL43 + 4, h: CELL43 + 4 }, opacity);
const mixRow = (from, to, t) => from.map((v, i) => lerp(v, to[i], t));

export function drawFrame6(svg, p) {
  headers(svg);
  probRow(layer(svg, seg(p, 0.05, 0.4)), 'reference', ROWS.reference);
  G.block(layer(svg, seg(p, 0.05, 0.4)), { x: 330, y: ROW_Y.reference + 7, w: 230, h: 34, label: 'reference (frozen SFT copy)', state: 'dim' });
  G.block(svg, { x: 330, y: ROW_Y.policy + 7, w: 230, h: 34, label: 'policy (being trained)', state: 'active' });
  probRow(svg, 'policy', mixRow(ROWS.reference, ROWS.policy, seg(p, 0.45, 0.9)));
  policyMark(svg);
  note(svg, ROWS_X, 180, 'probability of each first word, at one position', { cls: 'g-label' });
}

export function drawFrame7(svg, p) {
  note(svg, 16, 22, 'what if nothing anchors the policy?', { cls: '' });
  headers(svg);
  probRow(svg, 'reference', ROWS.reference);
  probRow(svg, 'policy', ROWS.policy);
  policyMark(svg);
  const grow = seg(p, 0.05, 0.6);
  probRow(layer(svg, seg(p, 0.05, 0.2)), 'hacked', mixRow(ROWS.policy, ROWS.hacked, grow));
  const x = 330;
  note(svg, x + CELL43 / 2, HEADER_Y, 'reward', { anchor: 'middle' });
  G.vector(svg, { x, y: ROW_Y.policy, values: [REWARD_HONEST], cell: CELL43, orient: 'row', maxAbs: 1.5, format: fmt1 });
  const climb = seg(p, 0.45, 0.8);
  G.vector(layer(svg, seg(p, 0.05, 0.2)), { x, y: ROW_Y.hacked, values: [lerp(REWARD_HONEST, REWARD_HACKED, climb)], cell: CELL43, orient: 'row', maxAbs: 1.5, format: fmt1 });
  note(svg, ROWS_X, 232, `hacked answer: ${HACKED_ANSWER}`, { cls: '' });
  note(layer(svg, seg(p, 0.8, 0.95)), ROWS_X, 256, 'reward model overrates flattery', { cls: '' });
}

export function drawFrame8(svg, p) {
  headers(svg);
  probRow(svg, 'reference', ROWS.reference);
  probRow(svg, 'policy', ROWS.policy);
  policyMark(svg);
  probRow(layer(svg, 0.4), 'hacked', ROWS.hacked);
  const cols = Object.freeze({ reward: 200, kl: 330, total: 480 });
  const head = 238;
  const rowsOut = [
    { name: 'honest', y: 258, reward: REWARD_HONEST, kl: KL.honest, total: TOTALS.honest },
    { name: 'hacked', y: 282, reward: REWARD_HACKED, kl: KL.hacked, total: TOTALS.hacked },
  ];
  const beta = linked(svg, 'beta', { x: 16, y: head - 14, w: 60, h: 18 });
  note(beta, 16, head, `β = ${fmt1(BETA_RLHF)}`, { cls: '' });
  note(svg, cols.reward, head, 'reward', { anchor: 'end' });
  note(svg, cols.kl, head, 'KL from reference', { anchor: 'end' });
  note(svg, cols.total, head, `total (β = ${fmt1(BETA_RLHF)})`, { anchor: 'end' });
  const klT = seg(p, 0.1, 0.4);
  const totalT = seg(p, 0.4, 0.9);
  rowsOut.forEach((r) => {
    note(svg, 16, r.y + 14, r.name, { cls: '' });
    note(svg, cols.reward, r.y + 14, fmt3(r.reward), { cls: '', anchor: 'end' });
    if (klT > 0) note(layer(svg, klT), cols.kl, r.y + 14, fmt3(r.kl), { cls: '', anchor: 'end' });
    note(svg, cols.total, r.y + 14, fmt3(lerp(r.reward, r.total, totalT)), { cls: '', anchor: 'end' });
  });
  const free = (r) => fmt3(klPenalizedReward(r.reward, r.kl, 0));
  note(layer(svg, seg(p, 0.85, 1)), 16, 332, `β = 0: ${free(rowsOut[0])} vs ${free(rowsOut[1])}, flattery wins · β = ${fmt1(BETA_RLHF)}: ${fmt3(TOTALS.honest)} vs ${fmt3(TOTALS.hacked)}, honest wins`);
}
