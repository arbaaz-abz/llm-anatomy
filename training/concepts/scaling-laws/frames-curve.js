// Frames 3, 4, 6 and 7: the valley along a fixed budget, its bottom, what serving adds, and the over-trained model.
import { computeOptimal, inferenceAwareOptimum, lifetimeFlops, tokensForLoss } from '@math/scaling.js';
import { trainingFlops } from '@math/scale.js';
import { FRAME_2_SIZES, CURVE_TABLE_SIZES, STAGE_BUDGET, SERVE_WHAT_IF } from './numbers.js';
import { sci, perParam, sizeText, tokensText, lossText, savingText } from './format.js';
import { COLUMN, UNDER, seg, ease, logLerp, label, lines, wrap, drawCurve, marker, lossAt, countedSci } from './stage.js';

const OPT = computeOptimal(STAGE_BUDGET);
const SERVED = inferenceAwareOptimum({ targetLoss: OPT.loss, inferenceTokens: SERVE_WHAT_IF });
const OPT_LIFE = lifetimeFlops(OPT.N, OPT.D, SERVE_WHAT_IF);
const ONE_B = FRAME_2_SIZES[0];
const NEAR_OPT = FRAME_2_SIZES[1]; // frame 2's middle block: the optimum itself
const ONE_T = FRAME_2_SIZES[2];
const POINTS = Object.freeze([ONE_B, NEAR_OPT, ONE_T]);
const SERVE_BASIS = '2 FLOPs per active parameter per token';
const NOTES_2020_2022 = Object.freeze(['2020 (Kaplan): mostly bigger', '2022 (Chinchilla): about 20 tokens per parameter']);
// The end points are named in the column and on the axis ticks; only the followed point is labeled on the curve.
const pointMarker = (n, followed = false) => marker(n, followed ? sizeText(n) : '', { followed });

// ---- frame 3: three points drop onto the valley, then the fitted curve draws through them ----
export function drawFrame3(svg, p) {
  const shown = POINTS.filter((_, i) => seg(p, i * 0.12, i * 0.12 + 0.12) >= 1);
  drawCurve(svg, { drawn: ease(seg(p, 0.4, 1)), markers: shown.map((n) => pointMarker(n, n === NEAR_OPT)) });
  const table = CURVE_TABLE_SIZES.map((n) => `${sizeText(n).padEnd(5)} ${lossText(lossAt(n))}`);
  label(svg, COLUMN.x, COLUMN.y, 'fitted loss', { opacity: seg(p, 0.6, 0.8) });
  lines(svg, COLUMN.x, COLUMN.y + COLUMN.lineH, table, { opacity: seg(p, 0.7, 1) });
}

// ---- frame 4: the marker settles at the bottom of the valley ----
export function drawFrame4(svg, p) {
  const at = logLerp(NEAR_OPT, OPT.N, ease(seg(p, 0, 0.6)));
  const text = `${sizeText(OPT.N)} · ${perParam(OPT.tokensPerParam)} per param`;
  drawCurve(svg, {
    markers: [pointMarker(ONE_B), marker(at, seg(p, 0.5, 0.6) >= 1 ? text : sizeText(at), { followed: true }), pointMarker(ONE_T)],
  });
  const rows = [`N* = ${sci(OPT.N)}`, `D* = ${sci(OPT.D)}`, `${perParam(OPT.tokensPerParam)} tokens/param`, `loss ${lossText(OPT.loss)}`];
  label(svg, COLUMN.x, COLUMN.y, 'compute-optimal', { opacity: seg(p, 0.5, 0.7) });
  lines(svg, COLUMN.x, COLUMN.y + COLUMN.lineH, rows, { opacity: seg(p, 0.6, 0.9) });
  lines(svg, UNDER.x, UNDER.y, NOTES_2020_2022, { lineH: UNDER.lineH, opacity: seg(p, 0.8, 1) });
}

// ---- frame 6: serving adds 2 FLOPs per parameter per token ----
const SERVE_FINAL = lifetimeFlops(OPT.N, OPT.D, SERVE_WHAT_IF) - trainingFlops({ params: OPT.N, tokens: OPT.D });
const TRAIN_OPT = trainingFlops({ params: OPT.N, tokens: OPT.D });

export function drawFrame6(svg, p) {
  const count = ease(seg(p, 0.2, 0.8));
  const showServe = seg(p, 0.2, 0.3);
  drawCurve(svg, { markers: [marker(OPT.N, sizeText(OPT.N), { followed: true })] });
  const rows = [
    { text: `served: ${tokensText(SERVE_WHAT_IF)} tokens`, bold: true },
    '(what-if)',
    ...wrap(`serving: ${SERVE_BASIS}`, COLUMN.chars),
    `train    ${sci(TRAIN_OPT)}`,
    { text: `serve    ${count > 0 ? countedSci(SERVE_FINAL, count, sci(SERVE_FINAL)) : '0'}`, opacity: showServe },
    { text: `lifetime ${sci(TRAIN_OPT + SERVE_FINAL * count)}`, bold: true, opacity: showServe },
  ];
  lines(svg, COLUMN.x, COLUMN.y, rows, { opacity: seg(p, 0, 0.2) });
}

// ---- frame 7: the same loss from a smaller model fed more tokens ----
export function drawFrame7(svg, p) {
  const n = logLerp(OPT.N, SERVED.N, ease(seg(p, 0, 0.7)));
  const tokens = tokensForLoss(n, OPT.loss);
  const train = trainingFlops({ params: n, tokens });
  const serve = lifetimeFlops(n, tokens, SERVE_WHAT_IF) - train;
  drawCurve(svg, {
    markers: [marker(OPT.N, sizeText(OPT.N)), marker(n, sizeText(n), { followed: true, y: OPT.loss })],
    refY: { value: OPT.loss, label: `same loss ${lossText(OPT.loss)}` },
  });
  lines(svg, COLUMN.x, COLUMN.y, [
    { text: `compute-optimal ${sizeText(OPT.N)}`, bold: true },
    `lifetime ${sci(OPT_LIFE)}`,
    { text: `over-trained ${sizeText(n)}`, bold: true },
    `${tokensText(tokens)} tokens`,
    `${perParam(tokens / n, 0)} per param`,
    `train    ${sci(train)}`,
    `serve    ${sci(serve)}`,
    `lifetime ${sci(train + serve)}`,
    { text: `${savingText(Math.max(0, 1 - (train + serve) / OPT_LIFE))} less`, bold: true },
  ]);
}
