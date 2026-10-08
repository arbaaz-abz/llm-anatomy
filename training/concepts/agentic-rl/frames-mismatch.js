// agentic-rl frames 7–10: the engine/trainer mismatch, the importance-sampling correction (key frame 8), the KL term, the scale.
import * as G from '@shared/glyphs.js';
import { ADVANTAGES, AGREE_COUNT, ROWS, TOKEN_COUNT, FOLLOWED } from './numbers.js';
import { fixed2, fixed3, ofTotal, push2, wrapText } from './format.js';
import { INITIAL_STATE, tokenAt } from './toy-view.js';
import {
  CELL, CHIP_H, TABLE, arriving, chipXs, layer, lerp, linked, litBlock, note, readout, seg, selectChip,
} from './stage.js';

const row4 = tokenAt(INITIAL_STATE, FOLLOWED.row, FOLLOWED.pos);
const withCorrection = (correction) => tokenAt({ ...INITIAL_STATE, correction }, FOLLOWED.row, FOLLOWED.pos);
const ICEPOP = { ...INITIAL_STATE, correction: 'icepop' };

// ---- frame 7: the same token, two probabilities ----
const CHIP = Object.freeze({ y: 40 });
const ENGINE = Object.freeze({ x: 70, y: 130, w: 160, h: 40 });
const TRAINER = Object.freeze({ x: 350, y: 130, w: 160, h: 40 });
const CENTER = 290;

function probCell(parent, geom, p, opacity) {
  const g = layer(parent, opacity);
  G.cell(g, { x: geom.x + geom.w / 2 - CELL / 2, y: geom.y + geom.h + 14, size: CELL, v: p, maxAbs: 1, format: fixed3 });
}

export function drawFrame7(svg, p) {
  const text = row4.text;
  const chipX = CENTER - G.tokenWidth(text) / 2;
  note(svg, CENTER, 22, 'row 4, token 8: the second 48', { anchor: 'middle' });
  G.token(svg, { x: chipX, y: CHIP.y, text });
  selectChip(svg, chipX, CHIP.y, text);
  const toEngine = seg(p, 0.05, 0.25);
  const toTrainer = seg(p, 0.4, 0.6);
  if (toEngine > 0) G.flow(svg, { from: [CENTER - 20, CHIP.y + CHIP_H + 4], to: [ENGINE.x + ENGINE.w / 2, ENGINE.y - 4], carry: 'token', progress: toEngine });
  if (toTrainer > 0) G.flow(svg, { from: [CENTER + 20, CHIP.y + CHIP_H + 4], to: [TRAINER.x + TRAINER.w / 2, TRAINER.y - 4], carry: 'token', progress: toTrainer });
  litBlock(svg, ENGINE, 'engine (samples)', seg(p, 0.2, 0.3), 'idle');
  litBlock(svg, TRAINER, 'trainer (updates)', seg(p, 0.55, 0.65), 'idle');
  probCell(svg, ENGINE, row4.engine, seg(p, 0.3, 0.45));
  probCell(svg, TRAINER, row4.trainer, seg(p, 0.65, 0.8));
  const rho = linked(layer(svg, seg(p, 0.8, 1)), 'rho', { x: 150, y: 244, w: 280, h: 24 });
  note(rho, CENTER, 261, `ρ = trainer / engine = ${fixed2(row4.rho)}`, { cls: 'g-readout', anchor: 'middle' });
  const fade = seg(p, 0.85, 1);
  note(layer(svg, fade), 14, 298, 'ρ is not the r of the last page: same weights, different programs.');
  note(layer(svg, fade), 14, 322, `most tokens agree within a few percent (${AGREE_COUNT} of ${TOKEN_COUNT} here);`);
  note(layer(svg, fade), 14, 338, 'a few outliers like this one are what the corrections are for');
}

// ---- frame 8 (key frame): the clip line, three masked tokens ----
const CLIP = Object.freeze({ x: 80, y: 56, w: 420, lo: 0, hi: 3.5, band: [0.5, 2] });
const SHOWN_ROWS = Object.freeze([3, 5, 6]); // 0-based; rows 4, 6 and 7 on screen
const ROW_Y = Object.freeze({ first: 112, stride: 44 });
const MASKED_AT = Object.freeze({ '3:7': 0.4, '5:0': 0.58, '6:4': 0.74 });
const RHO_X = 412;
const maskedFor = (row, pos) => tokenAt(ICEPOP, row, pos).masked;

function frame8Rows(svg, p) {
  const g = layer(svg, arriving(p));
  note(g, TABLE.rX + CELL / 2, ROW_Y.first - 8, 'R', { anchor: 'middle' });
  note(g, TABLE.aX + CELL / 2, ROW_Y.first - 8, 'A', { anchor: 'middle' });
  SHOWN_ROWS.forEach((row, k) => {
    const y = ROW_Y.first + k * ROW_Y.stride;
    const tokens = ROWS[row];
    const xs = chipXs(tokens, TABLE.chipsX);
    note(g, TABLE.x, y + CELL / 2 + 4, String(row + 1));
    tokens.forEach((text, pos) => {
      const masked = maskedFor(row, pos) && p >= MASKED_AT[`${row}:${pos}`];
      G.token(g, { x: xs[pos], y: y + (CELL - CHIP_H) / 2, text, hatched: masked });
      if (row === FOLLOWED.row && pos === FOLLOWED.pos) selectChip(g, xs[pos], y + (CELL - CHIP_H) / 2, text);
      if (maskedFor(row, pos) && p >= MASKED_AT[`${row}:${pos}`]) {
        const t = tokenAt(ICEPOP, row, pos);
        const tag = linked(g, 'rho', { x: RHO_X, y: y + 8, w: 150, h: 20 });
        note(tag, RHO_X, y + CELL / 2 + 4, `ρ = ${fixed2(t.rho)}: ${text}`, { cls: 'g-readout' });
      }
    });
    G.cell(g, { x: TABLE.rX, y, size: CELL, v: 0, maxAbs: 1, fill: 'bad' });
    G.cell(linked(g, 'a', { x: TABLE.aX, y, w: CELL, h: CELL }), { x: TABLE.aX, y, size: CELL, v: ADVANTAGES[row], maxAbs: 2.65 });
  });
  const lastY = ROW_Y.first + SHOWN_ROWS.length * ROW_Y.stride;
  note(g, TABLE.x, lastY + 6, '5 other rows: ρ within [½, 2], unmasked', { cls: 'g-readout' });
  return lastY;
}

export function drawFrame8(svg, p) {
  const land = seg(p, 0.05, 0.4);
  const marker = lerp(1, row4.rho, land);
  const clip = linked(layer(svg, arriving(p)), 'rho', { x: CLIP.x - 4, y: CLIP.y - 26, w: CLIP.w + 8, h: 60 });
  G.clipLine(clip, { ...CLIP, marker, label: 'ρ = trainer / engine' });
  note(layer(svg, seg(p, 0.1, 0.3)), CLIP.x + (CLIP.w * (CLIP.band[0] + CLIP.band[1])) / 2 / CLIP.hi, CLIP.y + 36, 'same factor either way: ½ and 2', { anchor: 'middle' });
  const lastY = frame8Rows(svg, p);
  const counted = Object.values(MASKED_AT).filter((at) => p >= at).length;
  const [full, tis, ice] = ['full', 'tis', 'icepop'].map(withCorrection);
  const none = withCorrection('none');
  readout(svg, [
    `masked ${ofTotal(counted, TOKEN_COUNT)} · band [1/2, 2] (β = 2)`,
    `row 4's 48: full IS ${fixed2(full.weight)} · truncated at 2 ${fixed2(tis.weight)} · IcePop ${ice.masked ? 'masked' : fixed2(ice.weight)}`,
    `push ${push2(none.push)} → ${push2(full.push)} / ${push2(tis.push)} / ${push2(ice.push)}`,
  ], { y: lastY + 30, opacity: seg(p, 0.4, 0.6) });
}

// ---- frame 9: the reference model leaves memory ----
const SHELF = Object.freeze({ y: 110, w: 150, h: 44 });
const SHELF_X = Object.freeze({ policy: 24, checker: 214, reference: 404 });

export function drawFrame9(svg, p, text) {
  const fade = seg(p, 0.25, 0.6);
  note(svg, 24, 90, 'what stays loaded for each update', { cls: 'g-label' });
  G.block(svg, { x: SHELF_X.policy, y: SHELF.y, w: SHELF.w, h: SHELF.h, label: 'policy', state: 'active' });
  G.block(svg, { x: SHELF_X.checker, y: SHELF.y, w: SHELF.w, h: SHELF.h, label: 'checker (a program)', state: 'active' });
  litBlock(svg, { x: SHELF_X.reference, y: SHELF.y, w: SHELF.w, h: SHELF.h }, 'reference', 1 - fade, 'dim');
  note(layer(svg, seg(p, 0.45, 0.7)), SHELF_X.reference + SHELF.w / 2, SHELF.y + SHELF.h + 18, 'KL term β = 0: not loaded', { anchor: 'middle' });
  const models = p >= 0.55 ? '2 → 1' : '2';
  readout(svg, [`models in memory ${models}`], { y: 214 });
  readout(svg, [text.beta], { y: 250, opacity: seg(p, 0.7, 0.9) });
}

// ---- frame 10: the scale ----
const TABLE_TEXT = Object.freeze({ x: 24, y: 48, gap: 17, block: 18, wrap: 84 });

export function drawFrame10(svg, p, text) {
  let y = TABLE_TEXT.y;
  text.table.forEach((entry, i) => {
    const lines = wrapText(entry, TABLE_TEXT.wrap);
    const g = layer(svg, seg(p, 0.05 + 0.17 * i, 0.2 + 0.17 * i));
    lines.forEach((line, k) => note(g, TABLE_TEXT.x + (k === 0 ? 0 : 14), y + k * TABLE_TEXT.gap, line, { cls: 'g-readout' }));
    y += lines.length * TABLE_TEXT.gap + TABLE_TEXT.block;
  });
}
