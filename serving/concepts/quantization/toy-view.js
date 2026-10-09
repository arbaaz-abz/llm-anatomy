// quantization toy view model (pure, no DOM): state + data → every string the toy prints and the specs of its figure.
// Numbers come from format.js (math/quant.js, math/serving.js); durations through formatDuration, bytes through formatBytes,
// counts through formatInt (README lesson 35).
import { CHIPS, chipPreset, chipLabel, modelFormatOptions } from './hardware.js';
import { blockView, lineSpec, checkWork, shrink, cellText, codeText, scaleText, errorText, bitsText, freeText, usersText, timeText, memoryText } from './format.js';
import { formatBytes, formatInt } from '@math/core.js';
import { FOLLOWED } from './numbers.js';

export const NOTE_LINE = 'On a hand-sized block INT4 can match FP4\'s error; FP4\'s advantage is that Blackwell tensor cores run it natively (frame 9).';

// The "Round a block" figure and readouts. One entry per weight; a block's scale chip spans its weights.
export function blockFigure(state) {
  const { weights, result } = blockView(state);
  const line = lineSpec(state);
  const blocks = result.blocks.map((b, i) => ({ from: i * state.blockSize, to: Math.min((i + 1) * state.blockSize, weights.length) - 1, scale: scaleText(b.scale) }));
  return {
    weights: weights.map((v) => ({ value: v, text: cellText(v) })),
    codes: result.blocks.flatMap((b) => b.codes).map((c) => ({ value: c, text: codeText(c) })),
    restored: result.restored.map((v) => ({ value: v, text: cellText(v) })),
    errors: result.err.map((v) => ({ value: v, text: cellText(v) })),
    blocks,
    line: { lo: line.lo, hi: line.hi, grid: [...line.grid], points: line.exact.map((value, i) => ({ value, snapped: true, followed: i === FOLLOWED })) },
    readouts: { meanError: errorText(result.meanAbsErr), zeroed: formatInt(result.zeroed), clipped: formatInt(result.clipped) },
  };
}

// The "Shrink a model" readouts for one chip; `sub` qualifiers carry the basis word and the bound.
export function modelReadouts(state, preset) {
  const m = shrink(state, preset);
  return {
    bits: bitsText(m.bits),
    weights: formatBytes(m.weights),
    memory: { value: memoryText(preset), sub: preset.basis === 'usable' ? `of ${preset.nominalGb} GB nominal` : '' },
    free: freeText(m),
    kvPerUser: formatBytes(m.kvPerUser),
    users: usersText(m),
    decode: { value: timeText(m.decode), sub: `${m.decode.bound}-bound` },
    prefill: { value: timeText(m.prefill), sub: `${m.prefill.bound}-bound` },
  };
}

export function toyView(state, data) {
  const preset = chipPreset(data, state.hw);
  return {
    figure: blockFigure(state),
    model: modelReadouts(state, preset),
    chips: CHIPS.map((c) => ({ value: c.id, label: chipLabel(chipPreset(data, c.id)) })),
    modelFormatOptions: modelFormatOptions(preset),
    checkWork: checkWork(state),
  };
}
