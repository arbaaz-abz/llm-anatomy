// training-memory frames 1–4: what one parameter costs, byte by byte (weight, gradient, Adam moments, master copy),
// beside GPT-3's running total and how many H100s it fills. Each frame starts where the previous one ended.
import { sharePct } from '@math/memory.js';
import { gpuCountText } from './format.js';
import { GPT3, H100_HBM } from './numbers.js';
import { PER_PARAM, BAR, COUNTER, seg, note, byteBar, gpuRow, countGb } from './stage.js';

const GROWTH = [0.1, 0.8]; // the growing segment's sub-phase

function chrome(svg, bytesPerParam, { totalLabel = 0 } = {}) {
  note(svg, 20, 20, "GPT-3's shape, today's recipe");
  note(svg, 20, 38, 'bar: what one parameter costs · GPUs: how full 80 GB gets');
  note(svg, BAR.x, BAR.y - 8, 'one parameter');
  const bytes = GPT3.params * bytesPerParam;
  const need = bytes / H100_HBM;
  const h100s = gpuCountText(need);
  note(svg, COUNTER.x, COUNTER.y, `GPT-3 needs: ${countGb(bytes)} = ${h100s} H100s`, { cls: 'g-text' });
  gpuRow(svg, bytes);
  if (totalLabel > 0) note(svg, BAR.x + PER_PARAM.total * BAR.perByte + 6, BAR.y + BAR.h / 2 + 4, `${PER_PARAM.total} B`, { cls: 'g-text', opacity: totalLabel });
}

function frame(svg, p, grown, key) {
  const shown = { ...grown, [key]: PER_PARAM[key] * seg(p, GROWTH[0], GROWTH[1]) };
  const total = byteBar(svg, shown);
  return total;
}

export function drawFrame1(svg, p) {
  const shown = { weight: PER_PARAM.weight * seg(p, ...GROWTH) };
  const total = byteBar(svg, shown);
  chrome(svg, total);
}

export function drawFrame2(svg, p) {
  const total = frame(svg, p, { weight: PER_PARAM.weight }, 'grad');
  chrome(svg, total);
}

export function drawFrame3(svg, p) {
  const base = { weight: PER_PARAM.weight, grad: PER_PARAM.grad };
  const total = frame(svg, p, base, 'optimizer');
  chrome(svg, total);
}

export function drawFrame4(svg, p) {
  const base = { weight: PER_PARAM.weight, grad: PER_PARAM.grad, optimizer: PER_PARAM.optimizer };
  const total = frame(svg, p, base, 'master');
  chrome(svg, total, { totalLabel: seg(p, 0.8, 1) });
  const share = sharePct(PER_PARAM.weight, PER_PARAM.total, { decimals: 1 });
  note(svg, BAR.x, 186, `weights alone are ${PER_PARAM.weight} of ${PER_PARAM.total} B (${share}%)`, { opacity: seg(p, 0.8, 1) });
}
