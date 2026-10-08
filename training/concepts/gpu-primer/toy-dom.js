// gpu-primer toy DOM helpers: the readout tables and the two figures (roofline, time lanes), repainted from toyView.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';

export const TOY_PLOT = Object.freeze({ w: 520, h: 280 });
const LANES_W = 520;

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

// Token presets as action buttons: they move the slider and keep no "selected" state, because the slider can sit between
// presets (64, 512) and a chip left pressed there would claim a value the toy is not showing. Same markup as mountChoice's
// chips, minus aria-pressed. → { destroy }
export function mountPresetButtons(root, { id, label, values, format, onPick }) {
  const caption = el('span', { id: `${id}-label`, className: 'choice-label', textContent: label });
  const buttons = values.map((v) => {
    const b = el('button', { type: 'button', className: 'choice-option', textContent: format(v) });
    b.dataset.value = String(v);
    return b;
  });
  const onClick = (event) => {
    const b = event.target.closest('.choice-option');
    if (b && root.contains(b)) onPick(values[buttons.indexOf(b)]);
  };
  root.id = id;
  root.className = 'choice choice--chips';
  root.setAttribute('role', 'group');
  root.setAttribute('aria-labelledby', caption.id);
  root.replaceChildren(caption, ...buttons);
  root.addEventListener('click', onClick);
  return { destroy() { root.removeEventListener('click', onClick); root.replaceChildren(); } };
}

// Two ruled tables: what the multiply costs, and what the chosen chip makes of it.
export function readoutTables(r) {
  const multiply = readoutTable({
    head: ['The multiply', ''], name: 'multiply',
    rows: [
      { label: 'FLOPs', sub: '2 × tokens × 8,192 × 8,192', cells: [{ value: r.flops, name: 'flops' }] },
      { label: 'Bytes moved', sub: 'read X and W once, write Y once', cells: [{ value: r.bytes, name: 'bytes' }] },
      { label: 'Arithmetic intensity', cells: [{ value: r.intensity, sub: 'FLOPs per byte', name: 'intensity' }] },
    ],
  });
  const chip = readoutTable({
    head: ['On this chip', ''], name: 'chip-readouts',
    rows: [
      { label: 'Ridge point', sub: 'peak ÷ bandwidth', cells: [{ value: r.ridge, sub: 'FLOPs per byte', name: 'ridge' }] },
      { label: 'Tokens needed to be compute-bound', cells: [{ value: r.tokensNeeded, sub: 'tokens', name: 'tokens-needed' }] },
      { label: 'Verdict', cells: [{ value: r.verdict, name: 'verdict' }] },
      { label: 'Attainable speed', cells: [{ value: r.attainable, name: 'attainable' }] },
      { label: 'Share of peak', cells: [{ value: r.peakShare, name: 'peak-share' }] },
      { label: 'Memory time', sub: 'bytes ÷ bandwidth', cells: [{ value: r.memoryTime, name: 'memory-time' }] },
      { label: 'Compute time', sub: 'FLOPs ÷ peak', cells: [{ value: r.computeTime, name: 'compute-time' }] },
      { label: 'Longer by', cells: [{ value: r.timeRatio.value, sub: r.timeRatio.sub, name: 'time-ratio' }] },
    ],
  });
  return [multiply, chip];
}

// The chip's roofline with this multiply (followed) and a residual add; sized to its drawing, scrolls in its box at 400 px.
export function paintPlot(svg, plot) {
  svg.replaceChildren();
  G.roofline(svg, { x: 0, y: 0, w: TOY_PLOT.w, h: TOY_PLOT.h, peakTflops: plot.peakTflops, bandwidthTBps: plot.bandwidthTBps, xDomain: plot.xDomain, yDomain: plot.yDomain, ridgeRange: plot.ridgeRange, points: plot.points, label: plot.title });
  G.fitViewBox(svg, 6);
  svg.setAttribute('aria-label', `${plot.title} roofline`);
}

export function paintLanes(svg, lanes) {
  svg.replaceChildren();
  G.laneTimeline(svg, { x: 0, y: 0, w: LANES_W, lanes: lanes.lanes, label: 'memory and compute time' });
  G.fitViewBox(svg, 4);
}
