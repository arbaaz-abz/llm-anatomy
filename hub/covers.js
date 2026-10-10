// The three card covers: each draws one act of the story (hub/story.js) with the course's own glyphs, and
// mountStory runs them on one clock. Under prefers-reduced-motion every cover shows its final frame at once.
import { svgEl, token, tokenWidth, block, flow, matrix, gpu, curvePlot } from '../shared/glyphs.js';
import { group, text } from '../shared/glyphs/core.js';
import { el } from '../shared/ui/dom.js';
import { storyFrame, FINAL_FRAME, modelScene, trainScene, serveScene, MODEL, SERVE } from './story.js';

export const COVER = Object.freeze({ w: 320, h: 150 });

export const COVER_CAPTIONS = Object.freeze({
  architecture: 'A token goes in, the next one comes out.',
  training: 'The loss falls as the weights move.',
  serving: 'Three requests, one GPU, tokens out.',
});

const labels = (svg) => group(svg, 'hub-cover-labels');

// ---- act 1: a token through the model ----
function drawModel(svg, p) {
  const S = modelScene(p);
  svg.replaceChildren();
  const { spine } = MODEL;
  const rail = group(svg, 'g-flow g-flow--token');
  svgEl('line', { class: 'g-path', x1: spine.x, y1: spine.from, x2: spine.x, y2: spine.to + 6 }, rail);
  svgEl('polygon', { class: 'g-head', points: `${spine.x},${spine.to} ${spine.x - 4},${spine.to + 7} ${spine.x + 4},${spine.to + 7}` }, rail);
  MODEL.blocks.forEach((b) => block(svg, { x: 110, y: b.y, w: 100, h: MODEL.blockH, label: b.label, state: S.active === b.id ? 'active' : 'idle' }));
  const notes = labels(svg);
  svgEl('path', { class: 'hub-bracket', d: `M216 ${MODEL.blocks[1].y}h5v${MODEL.blocks[0].y + MODEL.blockH - MODEL.blocks[1].y}h-5` }, notes);
  text(notes, 227, (MODEL.blocks[1].y + MODEL.blocks[0].y + MODEL.blockH) / 2, '× N layers', 'g-label', { 'dominant-baseline': 'central' });
  text(notes, 130, 20, 'next token', 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  const widths = S.tokens.map((t) => tokenWidth(t.text));
  let x = spine.x - (widths.reduce((a, b) => a + b, 0) + 6 * (widths.length - 1)) / 2;
  S.tokens.forEach((t, i) => {
    token(svg, { x, y: spine.from, text: t.text, state: t.state });
    x += widths[i] + 6;
  });
  if (S.dotY != null) svgEl('circle', { class: 'g-dot', cx: spine.x, cy: S.dotY, r: 5 }, rail);
  const out = S.output === 'active' ? MODEL.output : '?';
  token(svg, { x: spine.x - tokenWidth(out) / 2, y: 8, text: out, state: S.output });
}

// ---- act 2: the loss falls as the weights move ----
const PLOT = Object.freeze({ x: 150, y: 24, w: 150, h: 110 });
function drawTrain(svg, p) {
  const S = trainScene(p);
  svg.replaceChildren();
  matrix(svg, { x: 24, y: 40, values: S.weights, cell: 20, maxAbs: 1, label: 'weights', format: (v) => v.toFixed(2) });
  curvePlot(svg, {
    ...PLOT, label: 'training loss',
    xAxis: { domain: [0, 1], label: 'steps' }, yAxis: { domain: [0, 1], label: 'loss' },
    series: [{ points: S.curve, tone: 'ink' }],
  });
  flow(svg, { from: [156, 100], to: [112, 100], carry: 'gradient', progress: S.gradient ?? (p > 0.5 ? 1 : 0) });
  text(labels(svg), 134, 118, 'gradient', 'g-label', { 'text-anchor': 'middle' });
}

// ---- act 3: three requests share one GPU ----
function drawServe(svg, p) {
  const S = serveScene(p);
  svg.replaceChildren();
  gpu(svg, { x: 112, y: 22, w: 96, h: 72, memFill: S.memFill, label: 'one GPU' });
  const notes = labels(svg);
  text(notes, 12, 18, 'requests in', 'g-label');
  text(notes, SERVE.streamX, 18, 'tokens out', 'g-label');
  S.requests.forEach((r, i) => {
    const y = SERVE.laneY + i * SERVE.laneStep;
    token(svg, { x: r.x, y, text: r.owner, owner: r.owner });
    for (let k = 0; k < S.streams[i]; k += 1) token(svg, { x: SERVE.streamX + k * SERVE.streamStep, y, text: r.owner, owner: r.owner });
  });
}

export const DRAWERS = Object.freeze({ architecture: drawModel, training: drawTrain, serving: drawServe });

// <figure> with the cover's <svg> and its caption; the svg is named by the caption.
export function coverFigure(trackId) {
  const captionId = `hub-cover-${trackId}-caption`;
  const svg = svgEl('svg', { viewBox: `0 0 ${COVER.w} ${COVER.h}`, class: 'hub-cover-svg', role: 'img', 'aria-labelledby': captionId });
  const figure = el('figure', { className: 'hub-cover' }, [svg, el('figcaption', { id: captionId, textContent: COVER_CAPTIONS[trackId] ?? '' })]);
  return { figure, svg, draw: (p) => DRAWERS[trackId]?.(svg, p) };
}

export const prefersReducedMotion = () => globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

// Plays the acts in order on one requestAnimationFrame clock. `scenes[i]` = { draw(p), onProgress?(p) };
// `onState` hears 'playing', 'done' or 'still' (reduced motion: the final frames, nothing moves).
// A cover is only redrawn when its act's progress changed, so a resting act costs nothing per frame.
export function mountStory(scenes, { reduced = prefersReducedMotion(), onState = () => {}, raf = (fn) => requestAnimationFrame(fn), cancel = (id) => cancelAnimationFrame(id) } = {}) {
  const shown = scenes.map(() => -1);
  const paint = (frame) => frame.progress.forEach((p, i) => {
    if (p === shown[i]) return;
    shown[i] = p;
    scenes[i].draw(p);
    scenes[i].onProgress?.(p);
  });
  if (reduced) {
    paint(FINAL_FRAME);
    onState('still');
    return { replay: () => paint(FINAL_FRAME) };
  }
  let start = null;
  let handle = 0;
  const tick = (now) => {
    const frame = storyFrame(now - start);
    paint(frame);
    if (frame.done) onState('done');
    else handle = raf(tick);
  };
  const play = () => {
    cancel(handle);
    shown.fill(-1);
    onState('playing');
    handle = raf((now) => { start = now; tick(now); });
  };
  play();
  return { replay: play };
}
