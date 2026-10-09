// serving-calculator frames 9–10: what a million tokens costs, and why output is priced above input.
import * as G from '@shared/glyphs.js';
import { formatCount, formatInt, formatRatio } from '@math/core.js';
import { costPerMillion } from '@math/serving.js';
import { MEASURED, PRODUCTION, LIST_PRICES, LIST_PRICES_DATE, MEASURED_DATE, PRODUCTION_DATE, CONTEXT } from './numbers.js';
import { PREFILL_CEILING, atTarget, tokSGpu, costOfMeasured } from './figures.js';
import { LEFT, seg, leaving, layer, note, lines, footer } from './stage.js';
import { frame8Content } from './frames-b.js';
import { usd } from './format.js';

const BAR_AT = Object.freeze({ x: 8, y: 52, h: 150 });
const COL = 330;

// Frame 9's chart: $ per million tokens for two measured GPU-hours and DeepSeek's list prices.
function costBars(g, o) {
  if (o <= 0) return;
  const layerG = layer(g, o);
  const gb300 = costOfMeasured(MEASURED.gb300);
  const gb200 = costOfMeasured(MEASURED.gb200);
  note(layerG, LEFT, 32, '$ per million tokens');
  G.bars(layerG, {
    ...BAR_AT, w: 300, values: [gb300, gb200, LIST_PRICES.inputUsd, LIST_PRICES.outputUsd], labels: ['GB300', 'GB200', 'list in', 'list out'],
    max: 2, format: usd, label: 'dollars per million tokens',
  });
  lines(layerG, LEFT, 240, [
    'GB300 · GB200: measured, input and output tokens together',
    `list: DeepSeek V4-Pro off-peak, input (cache miss) · output`,
    `list prices read ${LIST_PRICES_DATE}; measured ${MEASURED_DATE}`,
  ]);
}

function costReadout(g, p) {
  const m = MEASURED.gb300;
  const lineG = (a, b) => layer(g, seg(p, a, b));
  note(lineG(0.15, 0.3), COL, 60, `${usd(m.usdPerGpuHour)} per GPU-hour`, { cls: '' });
  note(lineG(0.3, 0.45), COL, 76, `÷ ${formatInt(m.tokSGpu)} tok/s per GPU`, { cls: '' });
  note(lineG(0.45, 0.6), COL, 92, `= ${usd(costPerMillion(m.usdPerGpuHour, m.tokSGpu))} per M tokens`, { cls: '' });
  lines(lineG(0.6, 0.75), COL, 108, ['(input and output together,', 'InferenceX\'s convention)']);
  const gb200 = MEASURED.gb200;
  note(lineG(0.75, 0.9), COL, 140, `GB200: ${usd(gb200.usdPerGpuHour)} ÷ ${formatInt(gb200.tokSGpu)} = ${usd(costOfMeasured(gb200))}`, { cls: '' });
}

export function drawFrame9(svg, p) {
  const out = leaving(p);
  if (out > 0) frame8Content(layer(svg, out), 1);
  costBars(svg, seg(p, 0.1, 0.5));
  costReadout(svg, p);
  footer(svg);
}

const NODE_BARS = Object.freeze({ x: 8, y: 52, w: 150, h: 130 });

function nodeBars(g, p) {
  const o = seg(p, 0.1, 0.4);
  if (o <= 0) return;
  const layerG = layer(g, o);
  note(layerG, LEFT, 24, `tokens per second per node, DeepSeek production, ${PRODUCTION_DATE}`);
  G.bars(layerG, {
    ...NODE_BARS, values: [PRODUCTION.inputNodeTokS, PRODUCTION.outputNodeTokS], labels: ['input', 'output'], max: 80000,
    format: (n) => formatCount(n), label: 'input and output tokens per second per node',
  });
  note(layerG, 180, 80, `${formatCount(PRODUCTION.inputNodeTokS)} ÷ ${formatCount(PRODUCTION.outputNodeTokS)}`, { cls: '' });
  note(layerG, 180, 96, `= ${formatRatio(PRODUCTION.inputNodeTokS / PRODUCTION.outputNodeTokS)}`, { cls: '' });
}

function priceRatios(g, o) {
  if (o <= 0) return;
  const layerG = layer(g, o);
  lines(layerG, 300, 80, ['price ratios, output ÷ input:', `Anthropic ${formatRatio(LIST_PRICES.anthropicRatio)}`, `DeepSeek V4-Pro ${formatRatio(LIST_PRICES.outputUsd / LIST_PRICES.inputUsd)}`], { cls: '' });
}

function floorReadout(g, o) {
  if (o <= 0) return;
  const layerG = layer(g, o);
  const long = atTarget(CONTEXT.mid);
  const decode = tokSGpu(long.users, CONTEXT.mid);
  lines(layerG, LEFT, 222, [
    `floor, per GPU: input ceiling ${formatInt(PREFILL_CEILING)} tok/s`,
    `vs decode at 128K + 8K ${formatInt(decode)} tok/s: ${formatInt(PREFILL_CEILING)} ÷ ${formatInt(decode)} = ${formatRatio(PREFILL_CEILING / decode)}`,
  ], { cls: '' });
  lines(layerG, LEFT, 262, [
    'different hardware and units (per node vs per GPU): only the ratios are compared,',
    'each as tokens per second on one basis, input tokens vs output tokens',
  ]);
}

export function drawFrame10(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    costBars(g, 1);
    costReadout(g, 1);
  }
  nodeBars(svg, p);
  priceRatios(svg, seg(p, 0.4, 0.65));
  floorReadout(svg, seg(p, 0.6, 0.9));
  footer(svg);
}
