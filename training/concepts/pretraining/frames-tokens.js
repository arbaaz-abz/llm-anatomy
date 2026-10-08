// pretraining frames 1–3: text cut into tokens, one prediction's probabilities, and its cross-entropy loss.
import * as G from '@shared/glyphs.js';
import { tokenLoss } from '@math/lm.js';
import { TOKENS, SHOWN_WORDS, OTHERS, PROB_CELLS, PROB_TOTAL, PROB_MAX_ABS, P_ON, REFERENCE_PS, FOLLOWED } from './numbers.js';
import { formatProb, formatLoss } from './format.js';
import { CELL, CHIPS, CHIP_H, DIM, CHAR_W, SLOT, seg, lerp, arriving, leaving, typed, slotX, slotCenter, layer, note, chips, linked, select } from './stage.js';

const TARGET = FOLLOWED; // chip index of "on" (0-based 4): the token position 4 predicts
const SENTENCE = 'The cat sat down on the mat.';
const LINE = Object.freeze({ y: 32, center: slotCenter(0) + 3.5 * SLOT });
const NOTES_X = LINE.center;
const MODEL = Object.freeze({ x: slotX(0) + 6, y: 112, w: 4 * SLOT - 12, h: 30 });
const PROBS = Object.freeze({ x: slotX(0) + 6, y: 210 });
const BIG = Object.freeze({ x: PROBS.x, y: 150, size: 56 }); // frame 3: the "on" cell, enlarged
const P3 = (v) => v.toFixed(3); // probabilities at the storyboard's 3 d.p.
const WORD_LABELS = Object.freeze(SHOWN_WORDS.map((w) => (w === '.' ? '"."' : w)));

// Where each word sits in the plain text line (mono, so a word's center follows from its character offset).
const offsets = TOKENS.map((w, i) => SENTENCE.indexOf(w, i === 0 ? 0 : SENTENCE.indexOf(TOKENS[i - 1]) + TOKENS[i - 1].length));
const lineLeft = LINE.center - (SENTENCE.length * CHAR_W) / 2;
const wordCenter = (i) => lineLeft + (offsets[i] + TOKENS[i].length / 2) * CHAR_W;

function sentence(parent) {
  note(parent, LINE.center, LINE.y, SENTENCE, { anchor: 'middle' });
}

// A piece of text on its way from the line to its chip (ink, centered on its own position).
function piece(parent, i, x, y) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, g);
  t.textContent = TOKENS[i];
}

function frame1Notes(parent, text) {
  note(parent, NOTES_X, CHIPS.y + CHIP_H + 22, `${TOKENS.length} tokens`, { cls: '', anchor: 'middle' });
  note(parent, NOTES_X, 150, text.vocab, { cls: '', anchor: 'middle' });
  note(parent, NOTES_X, 170, text.vocabRange, { anchor: 'middle' });
  note(parent, NOTES_X, 190, text.embedding, { anchor: 'middle' });
  note(parent, NOTES_X, 226, 'Bigger vocabularies mean fewer tokens per document but a bigger table.', { anchor: 'middle' });
}

// Frame 1: the line splits at the piece boundaries and each piece drops into its chip; the period is its own chip.
export function drawFrame1(svg, p, text) {
  sentence(svg);
  const split = seg(p, 0.1, 0.4);
  const drop = seg(p, 0.4, 0.7);
  const fadePiece = 1 - seg(p, 0.6, 0.7);
  if (split > 0 && fadePiece > 0) {
    const g = layer(svg, fadePiece);
    TOKENS.forEach((_, i) => piece(g, i, lerp(wordCenter(i), slotCenter(i), split), lerp(LINE.y + 4, CHIPS.y + CHIP_H / 2, drop)));
  }
  const chipIn = seg(p, 0.55, 0.75);
  if (chipIn > 0) chips(svg, { opacity: () => chipIn });
  const notesIn = seg(p, 0.75, 1);
  if (notesIn > 0) frame1Notes(layer(svg, notesIn), text);
}

// Chips 1–4 are what the model reads; 5–8 wait (dim). `dimT` fades 5–8 down from lit.
const contextChips = (svg, dimT, { follow = [], selOpacity = 1 } = {}) => chips(svg, { opacity: (i) => (i < 4 ? 1 : lerp(1, DIM, dimT)), follow, selOpacity });
const trueNext = (parent) => note(parent, slotCenter(TARGET), CHIPS.y - 8, 'true next token', { anchor: 'middle' });

function probRow(parent, shown) {
  const g = linked(parent, 'p', { x: PROBS.x, y: PROBS.y, w: 5 * CELL, h: CELL });
  G.vector(g, { x: PROBS.x, y: PROBS.y, values: PROB_CELLS.slice(0, shown), cell: CELL, orient: 'row', maxAbs: PROB_MAX_ABS, format: P3 });
  WORD_LABELS.slice(0, shown).forEach((w, i) => note(parent, PROBS.x + i * CELL + CELL / 2, PROBS.y - 8, w, { anchor: 'middle' }));
  if (shown === 5) note(parent, PROBS.x + 4 * CELL + 2, PROBS.y - 8, `${OTHERS} others`);
}

function probNotes(parent) {
  note(parent, PROBS.x + 4.5 * CELL, PROBS.y + CELL + 16, `${P3(PROB_CELLS[4])} each`, { anchor: 'middle' });
  note(parent, PROBS.x + 5 * CELL + 14, PROBS.y + CELL / 2 + 4, `Σ = ${PROB_TOTAL.toFixed(3)}`);
}

const MODEL_CX = MODEL.x + MODEL.w / 2;
function model(parent, { into = 1, out = 1 } = {}) {
  G.block(parent, { x: MODEL.x, y: MODEL.y, w: MODEL.w, h: MODEL.h, label: 'model', state: 'idle' });
  if (into > 0) G.flow(parent, { from: [slotCenter(3), CHIPS.y + CHIP_H + 2], to: [slotCenter(3), MODEL.y - 2], carry: 'activation', progress: into });
  if (out > 0) G.flow(parent, { from: [MODEL_CX, MODEL.y + MODEL.h + 2], to: [MODEL_CX, PROBS.y - 22], carry: 'activation', progress: out });
}

// Frame 2: the model reads chips 1–4 and puts out a probability for every next token; "on" got 0.390.
export function drawFrame2(svg, p, text) {
  const out = leaving(p);
  if (out > 0) {
    const old = layer(svg, out);
    sentence(old);
    frame1Notes(old, text);
  }
  const markIn = seg(p, 0.85, 1);
  contextChips(svg, arriving(p), { follow: [TARGET], selOpacity: markIn });
  trueNext(layer(svg, arriving(p)));
  const modelIn = seg(p, 0.1, 0.25);
  if (modelIn > 0) model(layer(svg, modelIn), { into: seg(p, 0.25, 0.45), out: seg(p, 0.45, 0.65) });
  const shown = Math.ceil(5 * seg(p, 0.65, 0.85));
  if (shown > 0) probRow(svg, shown);
  if (markIn > 0) probNotes(layer(svg, markIn));
  select(svg, { x: PROBS.x, y: PROBS.y, w: CELL, h: CELL }, markIn);
}

// Frame 3: the "on" cell alone, enlarged; its loss types in, then the two reference marks.
export function drawFrame3(svg, p) {
  const out = leaving(p);
  contextChips(svg, 1, { follow: [TARGET] });
  trueNext(svg);
  if (out > 0) {
    const old = layer(svg, out);
    model(old);
    WORD_LABELS.slice(1).forEach((w, i) => note(old, PROBS.x + (i + 1) * CELL + CELL / 2, PROBS.y - 8, w, { anchor: 'middle' }));
    note(old, PROBS.x + 4 * CELL + 2, PROBS.y - 8, `${OTHERS} others`);
    G.vector(old, { x: PROBS.x + CELL, y: PROBS.y, values: PROB_CELLS.slice(1), cell: CELL, orient: 'row', maxAbs: PROB_MAX_ABS, format: P3 });
    probNotes(old);
  }
  const grow = seg(p, 0.15, 0.4);
  lossReadout(svg, { grow, typeT: seg(p, 0.45, 0.75), refs: seg(p, 0.75, 0.95) });
}

// The "on" cell growing from the probability row to BIG, its loss typed to typeT, the reference marks at `refs`.
export function lossReadout(parent, { grow = 1, typeT = 1, refs = 1 } = {}) {
  const size = lerp(CELL, BIG.size, grow);
  const box = { x: BIG.x, y: lerp(PROBS.y, BIG.y, grow), w: size, h: size };
  const g = linked(parent, 'p', box);
  G.vector(g, { x: box.x, y: box.y, values: [P_ON], cell: size, orient: 'row', maxAbs: PROB_MAX_ABS, format: P3 });
  note(parent, box.x + size / 2, box.y - 8, 'on', { anchor: 'middle' });
  select(parent, box);
  const readout = `loss = −ln ${formatProb(P_ON)} = ${formatLoss(tokenLoss(P_ON))}`;
  if (typeT > 0) note(parent, BIG.x + BIG.size + 16, BIG.y + BIG.size / 2 + 4, typed(readout, typeT), { cls: '' });
  if (refs > 0) {
    const r = layer(parent, refs);
    REFERENCE_PS.forEach((q, i) => note(r, BIG.x + BIG.size + 16, BIG.y + BIG.size + 24 + i * 20, `if ${q}: ${formatLoss(tokenLoss(q))}`));
  }
}

export { trueNext, TARGET };
