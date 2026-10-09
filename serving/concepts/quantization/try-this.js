// Storyboard §6 "Try this": four prompts, each leading to a named insight. Every number is computed from math/quant.js and
// the shrink analysis (format.js) on the data/hardware.json presets (hardware.js). Controls are named by their visible labels
// (XT-2). Pure, no DOM.
import { bitsPerElement } from '@math/roofline.js';
import { CHIPS, chipPreset } from './hardware.js';
import { INITIAL_STATE, blockView, errorText, cellText, shrink, timeText, usersText } from './format.js';

const view = (patch) => blockView({ ...INITIAL_STATE, ...patch });
const err = (patch) => errorText(view(patch).result.meanAbsErr);
const withFormat = (format, blockSize) => `${err({ format, blockSize })}`;

function first() {
  const base = view({});
  const calm = view({ outlier: false });
  const small = view({ blockSize: 4 });
  return {
    prompt: `With Format INT4 and Weights per scale 8: mean error ${errorText(base.result.meanAbsErr)}, ${base.result.zeroed} weights zeroed. Set Last weight to ${cellText(calm.weights[7])}: error ${errorText(calm.result.meanAbsErr)}. Set it back to ${cellText(base.weights[7])} (outlier) and Weights per scale to 4: ${errorText(small.result.meanAbsErr)}, ${small.result.zeroed} zeroed.`,
    insight: 'one outlier ruins a shared scale, and smaller blocks contain the damage.',
    rest: ' That is why 2026 formats scale every 16 or 32 weights (DeepSeek\'s FP8 weights use 128 × 128 tiles).',
  };
}

function second() {
  const mx4 = view({ format: 'mxfp4', blockSize: 4 });
  const mx2 = view({ format: 'mxfp4', blockSize: 2 });
  const row = (blockSize) => `INT4 ${withFormat('int4', blockSize)} · MXFP4 ${withFormat('mxfp4', blockSize)}${blockSize === 4 ? ` (0.47 clipped to ${mx4.result.restored[3]})` : blockSize === 2 ? ` (${mx2.result.clipped === 2 ? 'two' : mx2.result.clipped} clipped)` : ''} · NVFP4 ${withFormat('nvfp4', blockSize)}`;
  const extra = bitsPerElement('nvfp4') - bitsPerElement('mxfp4');
  return {
    prompt: `Weights per scale 4: ${row(4)}. Weights per scale 8: ${row(8)}. Weights per scale 2: ${row(2)}.`,
    insight: 'the scale\'s precision matters as much as the grid.',
    rest: ` A power-of-two scale is cheap to store but can waste range or clip; an 8-bit float scale fits each block more closely, for ${extra} more bits per weight.`,
  };
}

function third(data) {
  const [h200, b200] = CHIPS.map((c) => chipPreset(data, c.id));
  const on = (modelFormat, preset) => shrink({ modelFormat, kv: 'bf16' }, preset);
  const [bf, f8, w4] = ['bf16', 'fp8', 'w4a16'].map((m) => on(m, h200));
  const nv = on('nvfp4', b200);
  const chain = (pick) => [bf, f8, w4].map(pick).join(' → ');
  return {
    prompt: `With Model weights BF16 → FP8 → 4-bit weights, 16-bit math on the H200: decode ${[bf, f8, w4].map((m) => timeText(m.decode)).join(' → ')}; prefill ${chain((m) => timeText(m.prefill))}; users ${chain(usersText)}. Switch GPU to B200 (${b200.hbmGb} GB ${b200.basis}) and Model weights to NVFP4: prefill ${timeText(nv.prefill)}, decode ${timeText(nv.decode)}, ${usersText(nv)} users.`,
    insight: 'bytes speed up decode; only low-precision math speeds up prefill.',
    rest: ' Weight-only 4-bit is a decode trick; native FP8 or FP4 helps both.',
  };
}

function fourth(data) {
  const h200 = chipPreset(data, 'h200');
  const [plain, fp8Kv] = ['bf16', 'fp8'].map((kv) => shrink({ modelFormat: 'fp8', kv }, h200));
  return {
    prompt: `On the H200 with Model weights FP8, set KV cache to FP8: users ${usersText(plain)} → ${usersText(fp8Kv)}.`,
    insight: 'at long context the KV cache is the bigger target,',
    rest: ' and its quantization needs its own care (step 10).',
  };
}

// [{ prompt, insight, rest }]; the page prints "prompt → Insight: insight rest", with [[slug]] as the lesson's title.
export const tryThis = (data) => [first(), second(), third(data), fourth(data)];
