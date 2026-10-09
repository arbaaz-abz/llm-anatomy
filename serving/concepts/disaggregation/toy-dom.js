// disaggregation toy DOM helpers: the readout tables and the per-GPU memory bar, repainted from toyView.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

const BOUND_CLASS = Object.freeze({ 'memory-bound': 'sem-text--memory', 'compute-bound': 'sem-text--compute' });

// The verdict word keeps its text (it carries the meaning); a class only colours it, memory teal and compute amber.
function colourVerdict(table) {
  const out = table.querySelector('[data-readout="verdict"]');
  out.replaceChildren(el('span', { className: BOUND_CLASS[out.textContent], textContent: out.textContent }));
}

// Panel 1: what moves between the pools, how long it takes, and the MLA-sized comparison (a different model: no ratio).
export function shipTables(s) {
  const ship = readoutTable({
    head: ['Ship the KV', ''], name: 'ship',
    rows: [
      { label: 'KV to ship', sub: `${s.bytesPerToken} B per token × prompt`, cells: [{ value: s.kvBytes, name: 'kv-bytes' }] },
      { label: 'Transfer time', sub: `over ${s.linkLabel}`, cells: [{ value: s.transfer, name: 'transfer' }] },
      { label: 'Prefill time', sub: 'one forward pass, H200, FP8', cells: [{ value: s.prefill, name: 'prefill' }] },
      { label: 'Transfer as a share of prefill', sub: 'basis: one request\'s prompt', cells: [{ value: s.ratio, name: 'ratio' }] },
    ],
  });
  const mla = readoutTable({
    head: [`Same prompt, MLA-sized KV (DeepSeek-V3, ${s.mlaBytesPerToken} B per token)`, ''], name: 'mla',
    rows: [
      { label: 'KV to ship', cells: [{ value: s.mlaKv, name: 'mla-kv' }] },
      { label: 'Transfer time', sub: 'no share: a different model', cells: [{ value: s.mlaTransfer, name: 'mla-transfer' }] },
    ],
  });
  return [ship, mla];
}

// Panel 2: V4-Pro's experts over the chosen GPUs.
export function expertTables(e) {
  const table = readoutTable({
    head: ['Feed the experts', ''], name: 'experts',
    rows: [
      { label: 'Experts per GPU', cells: [{ value: e.expertsPerGpu, name: 'experts-per-gpu' }] },
      { label: 'Weights per GPU', cells: [{ value: e.weights, name: 'weights' }] },
      { label: 'Free memory per GPU', cells: [{ value: e.free, name: 'free' }] },
      { label: 'Tokens per expert per step', cells: [{ value: e.tokens, name: 'tokens-per-expert' }] },
      { label: 'Expert multiply intensity', sub: 'FLOPs per byte, FP4 weights', cells: [{ value: e.intensity, name: 'intensity' }] },
      { label: 'Ridge point', sub: 'FP4 on GB300', cells: [{ value: e.ridge, name: 'ridge' }] },
      { label: 'Tokens per expert to be compute-bound', cells: [{ value: e.tokensNeeded, name: 'tokens-needed' }] },
      { label: 'Verdict', cells: [{ value: e.verdict, name: 'verdict' }] },
    ],
  });
  colourVerdict(table);
  return table;
}

const BAR_W = 240;
// The memory bar: weights and free HBM; a weights segment too narrow to print is named in the legend (README lesson 19).
export function paintBar(svg, bar) {
  svg.replaceChildren();
  if (!bar) {
    svg.setAttribute('aria-label', 'weights do not fit in one GPU');
    return;
  }
  G.shareBar(svg, { x: 0, y: 4, w: BAR_W, parts: bar.parts, tail: 'none', label: 'memory per GPU' });
  G.fitViewBox(svg, 6);
  svg.setAttribute('aria-label', 'memory per GPU');
}
