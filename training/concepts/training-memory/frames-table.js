// training-memory frame 11: a trillion parameters at 16 B need 16 TB of state, and a mixture of experts pays for every
// expert, active or not (DeepSeek-V4-Pro's size). The counts are gpusToHoldStates with each chip's HBM, labeled.
import { formatBytes, formatCount } from '@math/core.js';
import { gpusToHoldStates, TRAINING_RECIPES, trainingBytesPerParam } from '@math/training-memory.js';
import { ONE_T, V4_PRO, FRAME11_CHIPS } from './numbers.js';
import { seg, lerp, note } from './stage.js';

const ADAM_BYTES = trainingBytesPerParam(TRAINING_RECIPES.adam).total;
const MUON_BYTES = trainingBytesPerParam(TRAINING_RECIPES.muon).total;
const COLUMNS = Object.freeze([
  { x: 300, head: ['1T model', '16 B'], params: ONE_T, bytes: ADAM_BYTES },
  { x: 410, head: ['V4-Pro size', '16 B (Adam)'], params: V4_PRO.total, bytes: ADAM_BYTES },
  { x: 520, head: ['V4-Pro size', '12 B (Muon)'], params: V4_PRO.total, bytes: MUON_BYTES },
]);
const ROW_Y = (i) => 100 + i * 24;
const holders = (column, chip) => gpusToHoldStates({ params: column.params, bytesPerParam: column.bytes, hbmBytes: chip.hbm });

export function drawFrame11(svg, p) {
  note(svg, 20, 22, `${formatCount(ONE_T)} parameters × ${ADAM_BYTES} B = ${formatBytes(ONE_T * ADAM_BYTES)} of training state`, { cls: 'g-text' });
  note(svg, 20, 42, 'GPUs just to hold it, before any activations');
  COLUMNS.forEach((col) => col.head.forEach((line, k) => note(svg, col.x, 64 + k * 14, line, { anchor: 'middle' })));
  FRAME11_CHIPS.forEach((chip, i) => {
    const typed = seg(p, 0.05 + i * 0.12, 0.2 + i * 0.12);
    note(svg, 20, ROW_Y(i), `${chip.name} · ${chip.capacity}`, { opacity: typed });
    COLUMNS.forEach((col) => note(svg, col.x, ROW_Y(i), String(holders(col, chip)), { cls: 'g-text', anchor: 'middle', opacity: typed }));
  });
  const dim = lerp(1, 0.4, seg(p, 0.6, 0.9));
  note(svg, 20, 208, `DeepSeek-V4-Pro's size at the ${ADAM_BYTES}-byte recipe:`);
  note(svg, 20, 226, `${formatCount(V4_PRO.total)} total`, { cls: 'g-text' });
  note(svg, 110, 226, `${formatCount(V4_PRO.active)} active`, { cls: 'g-text', opacity: dim });
  note(svg, 20, 244, `${formatBytes(V4_PRO.total * ADAM_BYTES)} of state → ${holders(COLUMNS[1], FRAME11_CHIPS[0])} H100s; its own recipe, Muon (${MUON_BYTES} B): ${holders(COLUMNS[2], FRAME11_CHIPS[0])}`);
  note(svg, 20, 284, 'mixture of experts: each token uses a few of many expert blocks');
  note(svg, 20, 302, 'See "Mixture of Experts" for how a token is routed.', { opacity: seg(p, 0.5, 0.8) });
}
