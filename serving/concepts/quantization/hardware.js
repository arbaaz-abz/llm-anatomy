// quantization's hardware presets (pure, no DOM): data/hardware.json → one chip's memory (with its basis word), bandwidth and
// dense peaks, through hbmFor. The one place the page reads chip numbers; the toy, the try-this list and the stage-constant
// test all go through it. Conventions (builder notes §2): dense peaks only; HBM capacity always through hbmFor.
import { lookupFact } from '@shared/claims.js';
import { hbmFor } from '@math/serving.js';
import { MODEL_FORMATS } from './format.js';
import { STAGE_CHIPS } from './numbers.js';

export const CHIPS = Object.freeze([{ id: 'h200', label: 'H200' }, { id: 'b200', label: 'B200' }].map(Object.freeze));

export const NO_FP4 = 'no FP4 tensor cores; use 4-bit weights, 16-bit math';
const FP4_FALLBACK = 'w4a16';

const value = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;
const positive = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;

// A chip's preset. peak.fp4 is null on a chip without FP4 tensor cores (the H200). Blackwell runs MXFP4 and NVFP4 at the
// same dense rate, and data/hardware.json carries it as nvfp4_dense_tflops, so both 4-bit math formats read that key.
export function chipPreset(data, id) {
  if (!CHIPS.some((c) => c.id === id)) throw new RangeError(`chipPreset: chip must be one of ${CHIPS.map((c) => c.id).join(', ')}, got ${id}`);
  const entry = data?.hardware?.entries?.find((e) => e.id === id);
  if (!entry) throw new RangeError(`chipPreset: data/hardware.json has no entry ${id}`);
  const { bytes, basis } = hbmFor(entry);
  const bandwidthTBps = value(data, id, 'hbm_tbps');
  if (!positive(bandwidthTBps)) throw new RangeError(`chipPreset: data/hardware.json has no hbm_tbps for ${id}`);
  const peak = Object.fromEntries([['bf16', 'bf16_dense_tflops'], ['fp8', 'fp8_e4m3_dense_tflops'], ['fp4', 'nvfp4_dense_tflops']].map(([f, key]) => {
    const v = value(data, id, key);
    return [f, positive(v) ? v : null];
  }));
  return Object.freeze({
    id,
    label: CHIPS.find((c) => c.id === id).label,
    hbmGb: bytes / 1e9,
    hbmBytes: bytes,
    basis,
    nominalGb: value(data, id, 'hbm_gb'),
    bandwidthTBps,
    peak: Object.freeze(peak),
  });
}

// The model-format options for a chip: a format whose math needs FP4 tensor cores is disabled, with its visible reason (P3-R12).
export function modelFormatOptions(preset) {
  return Object.entries(MODEL_FORMATS).map(([key, f]) => (f.math === 'fp4' && preset.peak.fp4 == null
    ? { value: key, label: f.label, disabled: true, note: NO_FP4 }
    : { value: key, label: f.label }));
}

// The format shown after a chip change: the wanted one if the chip can run it, else 4-bit weights with 16-bit math.
export function usableModelFormat(preset, wanted) {
  return MODEL_FORMATS[wanted].math === 'fp4' && preset.peak.fp4 == null ? FP4_FALLBACK : wanted;
}

// The chip chip's label: "H200 (141 GB nominal)", "B200 (180 GB usable of 192 nominal)".
export function chipLabel(preset) {
  const gb = `${preset.hbmGb} GB ${preset.basis}`;
  return preset.basis === 'usable' ? `${preset.label} (${gb} of ${preset.nominalGb} nominal)` : `${preset.label} (${gb})`;
}

// A chip preset for the stage in chipPreset's shape, built from the stage constants (numbers.js).
export function stagePreset(id) {
  const c = STAGE_CHIPS[id];
  return Object.freeze({
    id, label: c.label, hbmGb: c.hbmGb, hbmBytes: c.hbmGb * 1e9, basis: c.basis, nominalGb: c.nominalGb, bandwidthTBps: c.bandwidthTBps, peak: c.peak,
  });
}
