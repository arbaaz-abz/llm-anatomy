// speculative-decoding frames 1–5: one plain step, the drafter's guesses, the verify pass, the verdicts, the corrected token.
// Each draw is a pure function of progress p and starts from the previous frame's end state.
import * as G from '@shared/glyphs.js';
import { formatDuration } from '@math/core.js';
import { C, K, PICK, VERDICTS } from './numbers.js';
import { stepParts } from './sweeps.js';
import { probText } from './format.js';
import {
  BAR, BLOCK, CHIPS, DRAFTER_CX, NOTE_Y, PICK_Y, ROW_XS, TARGET_CX, VERDICT_Y,
  arriving, blocks, chip, chipCenter, followChip, layer, leaving, note, seg, standIn, stepBarAt, flowTo,
} from './stage.js';

const PLAIN = stepParts({ tokens: 1, seqs: 1 });
const VERIFY = stepParts({ tokens: K + 1, seqs: 1 });
const FOOT = CHIPS.y + CHIPS.h + 2; // below a chip
const PLAIN_TITLE = 'plain step: 1 token';
const VERIFY_TITLE = `verify pass: ${K + 1} tokens`;
const PER_GUESS_S = C * PLAIN.timeS;

// The numbers of the "Numbers shown" column, one line per frame, each from the functions above.
const LINES = Object.freeze({
  1: `batch 1, context 1,024: ${formatDuration(PLAIN.timeS)} per token`,
  2: `k = ${K} guesses · drafter cost c = ${probText(C)} of a step each`,
  3: `verify pass: ${K + 1} tokens · ${formatDuration(VERIFY.timeS)} vs ${formatDuration(PLAIN.timeS)} for 1 token`,
  4: `accepted ${VERDICTS.filter(Boolean).length} of ${K}`,
  5: `${VERDICTS.filter(Boolean).length} accepted + 1 corrected = ${VERDICTS.filter(Boolean).length + 1} tokens in ${formatDuration(VERIFY.timeS)} + ${K} × ${formatDuration(PER_GUESS_S)} of drafting`,
});

const line = (parent, n, opacity = 1) => note(layer(parent, opacity), CHIPS.x, NOTE_Y, LINES[n]);
const prompt = (parent) => [0, 1, 2].forEach((i) => chip(parent, i));
const drafts = (parent, opacity = () => 1, state = 'draft') => [3, 4, 5].forEach((i) => chip(parent, i, { opacity: opacity(i), state }));

// Frame 1: one normal decode step. The prompt goes into the target; one token comes out.
export function drawFrame1(svg, p) {
  prompt(svg);
  chip(svg, 3, { opacity: seg(p, 0.55, 0.85) });
  blocks(svg, { drafter: 0 });
  flowTo(svg, [chipCenter(2), FOOT], [TARGET_CX, BLOCK.y], seg(p, 0.05, 0.45));
  flowTo(svg, [TARGET_CX, BLOCK.y], [chipCenter(3), FOOT], seg(p, 0.45, 0.8));
  stepBarAt(svg, { x: BAR.left, parts: PLAIN, t: seg(p, 0.3, 0.95), title: PLAIN_TITLE });
  line(svg, 1);
}

// Frame 2: the drafter guesses down, on, a; each guess appears as a dashed chip.
export function drawFrame2(svg, p) {
  prompt(svg);
  chip(svg, 3, { opacity: leaving(p) });
  blocks(svg, { drafter: arriving(p) });
  const at = (i) => seg(p, 0.15 + 0.22 * (i - 3), 0.35 + 0.22 * (i - 3));
  drafts(svg, at);
  [3, 4, 5].forEach((i) => flowTo(svg, [DRAFTER_CX, BLOCK.y], [chipCenter(i), FOOT], at(i)));
  stepBarAt(svg, { x: BAR.left, parts: PLAIN, title: PLAIN_TITLE });
  line(svg, 1, leaving(p));
  line(svg, 2, arriving(p));
  standIn(svg, arriving(p));
}

function resting(svg) {
  prompt(svg);
  blocks(svg);
  stepBarAt(svg, { x: BAR.left, parts: PLAIN, title: PLAIN_TITLE });
  standIn(svg);
}

// Frame 3: the target takes `sat` and the three guesses in one pass; the verify bar draws to the plain bar's length.
export function drawFrame3(svg, p) {
  resting(svg);
  drafts(svg);
  [2, 3, 4, 5].forEach((i) => flowTo(svg, [chipCenter(i), FOOT], [TARGET_CX, BLOCK.y], seg(p, 0.05, 0.5)));
  stepBarAt(layer(svg, seg(p, 0.35, 0.5)), { x: BAR.right, parts: VERIFY, t: seg(p, 0.4, 0.95), title: VERIFY_TITLE });
  line(svg, 2, leaving(p));
  line(svg, 3, arriving(p));
}

function verdicts(svg, count, stamp) {
  VERDICTS.slice(0, count).forEach((ok, i) => G.verdict(layer(svg, stamp(i)), { x: chipCenter(3 + i), y: VERDICT_Y, ok }));
}

// Frame 4: verdicts stamp left to right; the followed frame moves along the guesses.
export function drawFrame4(svg, p) {
  resting(svg);
  drafts(svg);
  stepBarAt(svg, { x: BAR.right, parts: VERIFY, title: VERIFY_TITLE });
  verdicts(svg, K, (i) => seg(p, 0.05 + 0.25 * i, 0.25 + 0.25 * i));
  followChip(svg, 3 + Math.min(K - 1, Math.floor(p * K)), 1);
  line(svg, 3, leaving(p));
  line(svg, 4, seg(p, 0.7, 0.9));
}

// Frame 5: the failed guess is replaced by the target's own token; the round made three tokens.
export function drawFrame5(svg, p) {
  resting(svg);
  [3, 4].forEach((i) => {
    chip(svg, i, { state: 'draft', opacity: 1 - seg(p, 0, 0.3) });
    chip(svg, i, { opacity: seg(p, 0, 0.3) });
  });
  chip(svg, 5, { state: 'draft', opacity: 1 - seg(p, 0.1, 0.4) });
  chip(svg, 5, { text: PICK, opacity: seg(p, 0.3, 0.6), dy: -8 * (1 - seg(p, 0.3, 0.6)) });
  stepBarAt(svg, { x: BAR.right, parts: VERIFY, title: VERIFY_TITLE });
  verdicts(svg, K - 1, () => 1);
  G.verdict(layer(svg, 1 - seg(p, 0.1, 0.4)), { x: chipCenter(5), y: VERDICT_Y, ok: VERDICTS[2] });
  note(layer(svg, seg(p, 0.5, 0.8)), ROW_XS[5], PICK_Y + 8, "target's pick");
  followChip(svg, 5, 1, PICK);
  line(svg, 4, leaving(p));
  line(svg, 5, seg(p, 0.7, 0.9));
}
