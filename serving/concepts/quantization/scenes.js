// quantization frames 1–6 as scenes: the end state of each frame as plain data (rows of cell texts, scale chips, the number line,
// notes), computed from math/quant.js. frames-block.js draws a scene and cross-fades the parts that differ from the previous scene,
// so a frame is a pure function of (index, progress). No DOM.
import { weightBytes } from '@math/serving.js';
import { WEIGHTS, OUTLIER, FOLLOWED } from './numbers.js';
import { blockView, lineSpec, cellText, codeText, scaleText, errorText, restoredText, gridFor, LINE_DOMAIN, INT4_GRID } from './format.js';

const STAND_IN = 'hand-picked weights; real blocks hold 16 or 32';
const NVFP4_RULE = ['this toy\'s NVFP4 scale: largest value over six, rounded to FP8', '(a stand-in; the exact rule is not in our sources)'];
const OTHER_SEVEN = WEIGHTS.slice(0, -1).map(Math.abs);

const texts = (cells, f = cellText) => cells.map((v) => ({ value: v, text: f(v) }));

// The rows of one quantized state: codes, restored, error, "became 0" tags, scale chips.
function rowsOf(state) {
  const { result } = blockView(state);
  return {
    chips: result.blocks.map((b, i) => ({ from: i * state.blockSize, to: Math.min((i + 1) * state.blockSize, WEIGHTS.length) - 1, text: `scale ${scaleText(b.scale)}` })),
    codes: texts(result.blocks.flatMap((b) => b.codes), codeText),
    restored: texts(result.restored),
    errors: texts(result.err),
    tags: result.restored.map((r, i) => (r === 0 && WEIGHTS[i] !== 0 ? i : -1)).filter((i) => i >= 0),
    result,
  };
}

// A number line in the code domain: each dot goes from `value ÷ scale` (a) to its grid value (b).
function codeLine(state) {
  const l = lineSpec(state);
  return { key: `code-${state.format}`, lo: l.lo, hi: l.hi, grid: [...l.grid], label: 'weight ÷ scale', points: l.exact.map((a, i) => ({ a, b: l.snapped[i], followed: i === FOLLOWED })) };
}

// Frame 4's line: the weights themselves against a grid whose ticks are one scale apart.
function weightLine(scale) {
  const grid = INT4_GRID.map((k) => Number((k * scale).toFixed(4)));
  const edge = Math.ceil((OUTLIER + scale) * 10) / 10;
  return { key: 'weights', lo: -edge, hi: edge, grid, label: 'weight', points: WEIGHTS.map((v, i) => ({ a: v, b: v, followed: i === FOLLOWED })) };
}

const INT8 = { format: 'int4', blockSize: 8, outlier: true };
const INT4B = { format: 'int4', blockSize: 4, outlier: true };
const NV4 = { format: 'nvfp4', blockSize: 4, outlier: true };

// The scene of frame `index` (0–5). Fields left out are not on the stage in that frame.
export function sceneFor(index) {
  const int8 = rowsOf(INT8);
  const int4 = rowsOf(INT4B);
  const nv4 = rowsOf(NV4);
  const scale8 = scaleText(int8.result.blocks[0].scale);
  const fullRow = (rows) => ({ chips: rows.chips, codes: rows.codes, restored: rows.restored, errors: rows.errors, tags: rows.tags });
  const scenes = [
    { counter: true, notes: [STAND_IN, `8 × 16 bits = ${weightBytes({ params: WEIGHTS.length, bitsPerParam: 16 })} bytes`] },
    { chips: int8.chips, codes: int8.codes, line: codeLine(INT8), notes: [`scale = ${cellText(OUTLIER)} ÷ 7 = ${scale8}`] },
    { ...fullRow(int8), line: { ...codeLine(INT8), points: codeLine(INT8).points.map((p) => ({ ...p, a: p.b })) }, notes: [`mean error ${errorText(int8.result.meanAbsErr)} · ${int8.result.zeroed} zeroed`] },
    { ...fullRow(int8), outlierTag: true, line: weightLine(int8.result.blocks[0].scale), notes: [`tick spacing ${scale8} · the other seven weights: ${cellText(Math.min(...OTHER_SEVEN))} to ${cellText(Math.max(...OTHER_SEVEN))} in size`] },
    { ...fullRow(int4), rule: true, line: codeLine(INT4B), notes: [`mean error ${errorText(int8.result.meanAbsErr)} → ${errorText(int4.result.meanAbsErr)} · zeroed ${int8.result.zeroed} → ${int4.result.zeroed}`] },
    { ...fullRow(nv4), rule: true, line: codeLine(NV4), notes: [...NVFP4_RULE, `mean error ${errorText(nv4.result.meanAbsErr)}`] },
  ];
  const scene = scenes[index];
  if (!scene) throw new RangeError(`quantization: no block scene ${index + 1}`);
  return scene;
}

// Frame 7: MXFP4 and NVFP4 side by side, both blocks of four.
export function compareScene() {
  const mx = rowsOf({ format: 'mxfp4', blockSize: 4, outlier: true });
  const nv = rowsOf(NV4);
  const ratio = (rows) => WEIGHTS[FOLLOWED] / rows.result.blocks[0].scale; // the followed weight is in block 0 at blocks of four
  const mxRatio = ratio(mx);
  const nvRatio = ratio(nv);
  const exponent = Math.log2(mx.result.blocks[0].scale);
  return {
    mx, nv, mxRatio, nvRatio, grid: gridFor('mxfp4'), lo: LINE_DOMAIN.lo, hi: LINE_DOMAIN.hi,
    notes: [
      `MX scale 2${exponent < 0 ? '⁻' : ''}${String(Math.abs(exponent)).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])} = ${scaleText(mx.result.blocks[0].scale)}: ${cellText(WEIGHTS[FOLLOWED])} → ${mxRatio.toFixed(2)} → clamped to 6 → ${restoredText(mx.result.restored[FOLLOWED])}`,
      `NV scale ${scaleText(nv.result.blocks[0].scale)}: ${cellText(WEIGHTS[FOLLOWED])} → ${nvRatio.toFixed(2)} → ${codeText(nv.result.blocks[0].codes[FOLLOWED])} → ${restoredText(nv.result.restored[FOLLOWED])}`,
    ],
    errors: { mx: errorText(mx.result.meanAbsErr), nv: errorText(nv.result.meanAbsErr) },
  };
}
