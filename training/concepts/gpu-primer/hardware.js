// gpu-primer's hardware presets (pure, no DOM): data/hardware.json → { peak per format, bandwidth ends, HBM with its basis
// word }. The one place the page reads chip numbers; the toy, the try-this list and the stage-constant test all go through it.
// Conventions (builder notes §2): dense peaks only; HBM bandwidth `hbm_tbps` (Rubin's is a range: both ends);
// HBM capacity `hbm_gb` is nominal.
import { lookupFact } from '@shared/claims.js';

// Toy chip order (storyboard §6 controls); `fp4` names the chip's FP4 flavor (NVFP4 on NVIDIA, MXFP4 on AMD).
export const CHIPS = Object.freeze([
  { id: 'h100', label: 'H100', fp4: 'nvfp4' },
  { id: 'h200', label: 'H200', fp4: 'nvfp4' },
  { id: 'b200', label: 'B200', fp4: 'nvfp4' },
  { id: 'b300', label: 'B300', fp4: 'nvfp4' },
  { id: 'mi355x', label: 'MI355X', fp4: 'mxfp4' },
  { id: 'tpu-v7', label: 'TPU v7', fp4: 'nvfp4' },
  { id: 'rubin', label: 'Rubin', fp4: 'nvfp4' },
].map(Object.freeze));

// The format control's three options; each maps to an exact FORMATS key per chip (never a bare 'fp8').
export const FORMAT_CHOICES = Object.freeze([
  Object.freeze({ value: 'bf16', label: 'BF16' }),
  Object.freeze({ value: 'fp8', label: 'FP8' }),
  Object.freeze({ value: 'fp4', label: 'FP4' }),
]);

// Why an option is disabled (ruling P3-R12): a chip with no FP4 figure, or Rubin with no settled BF16/FP8 figure.
export const NO_FP4 = 'no FP4 figure in data';
export const NO_BF16_FP8 = 'no settled BF16/FP8 figure';

const value = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;
const positive = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;

// One chip's preset. bandwidths: one entry, or two for a published range, fastest first (Rubin [22, 19.2]), so the
// ridge and the crossing read low → high. Throws a RangeError when the data lacks the chip or its bandwidth.
export function chipPreset(data, id) {
  const chip = CHIPS.find((c) => c.id === id);
  if (!chip) throw new RangeError(`chipPreset: chip must be one of ${CHIPS.map((c) => c.id).join(', ')}, got ${id}`);
  const bw = value(data, id, 'hbm_tbps');
  const bandwidths = (Array.isArray(bw) ? [...bw] : [bw]).sort((a, b) => b - a);
  if (!bandwidths.every(positive)) throw new RangeError(`chipPreset: data/hardware.json has no hbm_tbps for ${id}`);
  const formats = { bf16: 'bf16', fp8: 'fp8_e4m3', fp4: chip.fp4 };
  const peak = Object.fromEntries(Object.entries(formats).map(([f, key]) => {
    const v = value(data, id, `${key}_dense_tflops`);
    return [f, positive(v) ? v : null];
  }));
  if (!Object.values(peak).some(positive)) throw new RangeError(`chipPreset: data/hardware.json has no dense peak for ${id}`);
  return Object.freeze({
    id, label: chip.label,
    formats: Object.freeze(formats),
    peak: Object.freeze(peak),
    bandwidths: Object.freeze(bandwidths),
    hbm: Object.freeze({ gb: value(data, id, 'hbm_gb'), basis: 'nominal' }),
  });
}

const reason = (f) => (f === 'fp4' ? NO_FP4 : NO_BF16_FP8);

// The format options for a chip: an option the chip has no peak for is disabled with its visible reason.
export function formatOptions(preset) {
  return FORMAT_CHOICES.map((o) => (preset.peak[o.value] == null ? { ...o, disabled: true, note: reason(o.value) } : { ...o }));
}

// The format the toy shows after a chip change: the wanted one if the chip has it, else the first it has (what mountChoice's
// update() reselects too).
export function usableFormat(preset, wanted) {
  return preset.peak[wanted] != null ? wanted : FORMAT_CHOICES.find((o) => preset.peak[o.value] != null).value;
}
