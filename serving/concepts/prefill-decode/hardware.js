// prefill-decode's GPU presets (pure, no DOM): data/hardware.json → { dense peaks, bandwidth, HBM with its basis word }.
// The one place the toy reads chip numbers (builder notes §2): dense peaks only; HBM bandwidth `hbm_tbps`; HBM capacity
// through hbmFor (usable when the data has it, else nominal), always printed with its basis word.
import { lookupFact } from '@shared/claims.js';
import { hbmFor } from '@math/serving.js';
import { formatBytes } from '@math/core.js';

export const GPUS = Object.freeze([
  Object.freeze({ id: 'h100', label: 'H100' }),
  Object.freeze({ id: 'h200', label: 'H200' }),
  Object.freeze({ id: 'b200', label: 'B200' }),
]);

// The weight formats the toy offers: the exact FORMATS key (math/roofline.js) and the dense-peak key per format.
export const WEIGHT_FORMATS = Object.freeze({
  bf16: Object.freeze({ label: 'BF16', key: 'bf16', peakKey: 'bf16_dense_tflops' }),
  fp8: Object.freeze({ label: 'FP8', key: 'fp8_e4m3', peakKey: 'fp8_e4m3_dense_tflops' }),
});

const value = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;
const positive = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;

// "141 GB nominal" · "180 GB usable (192 nominal)": the memory a GPU's chip and readouts print, basis word included.
export function hbmText(preset) {
  if (preset.hbm.basis === 'usable') return `${formatBytes(preset.hbm.bytes)} usable (${preset.nominalGb} nominal)`;
  return `${formatBytes(preset.hbm.bytes)} nominal`;
}

// One GPU's preset; a RangeError when the data lacks the GPU or a figure the toy needs.
export function gpuPreset(data, id) {
  const gpu = GPUS.find((g) => g.id === id);
  if (!gpu) throw new RangeError(`gpuPreset: gpu must be one of ${GPUS.map((g) => g.id).join(', ')}, got ${id}`);
  const entry = data?.hardware?.entries?.find((e) => e.id === id);
  if (!entry) throw new RangeError(`gpuPreset: data/hardware.json has no entry ${id}`);
  const bandwidthTBps = value(data, id, 'hbm_tbps');
  const peak = Object.fromEntries(Object.entries(WEIGHT_FORMATS).map(([f, w]) => [f, value(data, id, w.peakKey)]));
  if (!positive(bandwidthTBps) || !Object.values(peak).every(positive)) throw new RangeError(`gpuPreset: data/hardware.json lacks hbm_tbps or a dense peak for ${id}`);
  return Object.freeze({
    id, label: gpu.label, bandwidthTBps,
    peak: Object.freeze(peak),
    hbm: Object.freeze(hbmFor(entry)),
    nominalGb: value(data, id, 'hbm_gb'),
  });
}

// The GPU chip's visible label: "H200 · 141 GB nominal" (the basis word rides on the chip, Review Focus 3).
export const gpuLabel = (preset) => `${preset.label} · ${hbmText(preset)}`;
