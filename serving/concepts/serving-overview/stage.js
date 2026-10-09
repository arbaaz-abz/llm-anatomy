// serving-overview stage kit: the fixed layout (storyboard §4: pipeline on top, your request's time axis below), the timing
// helpers seg / lerp / ease, and the drawing helpers every frame shares. Pure: nothing touches the DOM at import time.
// Your request (frames 1–8) wears G.selectionMark everywhere it appears and has no letter and no --req hue (P4-R7).
import * as G from '@shared/glyphs.js';
import { PROMPT, ANSWER, PREFILL_S, TPOT_S, WEIGHTS_FILL } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
const CHAR_W = 6.6; // one mono character at the 11 px label size
const CHIP_GAP = 4;

// ---- the pipeline (top half) ----
export const YOU = Object.freeze({ x: 8, y: 40, w: 52, h: 32 });
export const ROUTER = Object.freeze({ x: 80, y: 40, w: 64, h: 32 });
export const REPLICAS = Object.freeze([4, 41, 78].map((y) => Object.freeze({ x: 164, y, w: 76, h: 30 })));
export const REGION = Object.freeze({ x: 252, y: 2, w: 326, h: 110 }); // "inside replica 2": a faint fill, never an outline
export const SCHEDULER = Object.freeze({ x: 264, y: 40, w: 80, h: 32 });
export const GPU = Object.freeze({ x: 366, y: 26, w: 80, h: 60 });
export const KV = Object.freeze({ x: 476, y: 42, tile: 14, pitch: 16 });
export const PIPE_CY = 56;
export const center = (box) => box.x + box.w / 2;

// ---- the chip row: the request's tokens travel under the pipeline ----
export const CHIP_Y = 124;
export const CHIP_H = 24;
export const LABEL_Y = 166; // a chip group's label
export const RETURN_Y = 156; // the flow that carries a picked token back to you
export const QUEUE_LANE = Object.freeze({ x: 238, y: 118, w: 132, h: 36 });
export const chipWidths = (texts) => texts.map((t) => G.tokenWidth(t));
export const groupWidth = (texts) => chipWidths(texts).reduce((s, w) => s + w, 0) + CHIP_GAP * Math.max(texts.length - 1, 0);
const PROMPT_W = groupWidth(PROMPT);
export const PROMPT_AT = Object.freeze({
  you: YOU.x, router: center(ROUTER) - PROMPT_W / 2, replica: center(REPLICAS[1]) - PROMPT_W / 2,
  queue: QUEUE_LANE.x + (QUEUE_LANE.w - PROMPT_W) / 2, gpu: center(GPU) - PROMPT_W / 2,
});
export const outX = (text) => center(GPU) - G.tokenWidth(text) / 2; // the out slot: a picked token under the GPU
export const ANSWER_X = Object.freeze(ANSWER.map((_, i) => YOU.x + groupWidth(ANSWER.slice(0, i)) + (i > 0 ? CHIP_GAP : 0)));

// ---- your request's time axis (bottom half): drawn to scale ----
export const BAR = Object.freeze({ x: 100, y: 206, h: 10 });
export const PX_PER_S = 8000; // 14.6 ms ≈ 117 px
export const PREFILL_W = PREFILL_S * PX_PER_S;
export const TPOT_W = TPOT_S * PX_PER_S;
export const tickX = (k) => PREFILL_W + k * TPOT_W; // the end of decode step k (k = 0: the end of prefill)
export const BRACKET_Y = 228;
export const AXIS_Y = 258;
export const NOTE_Y = Object.freeze([284, 304, 324, 344]);

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF]
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

// ---- drawing helpers ----
export function fade(node, opacity) {
  if (node && opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? Math.max(opacity, 0).toFixed(3) : null }, parent);

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(parent, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

// A note line in the bottom band: ink text, 0-based row of NOTE_Y.
export const note = (parent, row, str, opacity = 1) => label(parent, 12, NOTE_Y[row], str, { cls: '', opacity });

export const select = (parent, x, y, w, h, opacity = 1) => (opacity > 0 && w > 0 ? fade(G.selectionMark(parent, { x, y, w, h }), opacity) : null);

// A solid 1 px muted line (a rule, an axis, a bracket): never dashed (P4-R15); currentColor at the muted strength, no inline style.
export function rule(parent, d) {
  return G.svgEl('path', { d, fill: 'none', stroke: 'currentColor', 'stroke-opacity': 0.55, 'stroke-width': 1 }, parent);
}

// A faint region with no outline (outlines mean selection): "inside replica 2", the queue lane.
export function region(parent, { x, y, w, h }, opacity = 1) {
  if (opacity <= 0) return null;
  return fade(G.svgEl('rect', { x, y, width: w, height: h, rx: 6, fill: 'currentColor', 'fill-opacity': 0.06 }, parent), opacity);
}

// A labelled bracket under a span of the time axis, ends turned up; `reveal` draws it left to right. `link` names the
// math term it answers to (hl-ttft / hl-tpot): an invisible g-frame the math panel outlines (template rule 8).
export function bracket(parent, { x0, x1, y = BRACKET_Y, text, link, reveal = 1, anchor = 'middle', opacity = 1 }) {
  if (reveal <= 0 || opacity <= 0 || x1 <= x0) return null;
  const g = fade(G.svgEl('g', link ? { 'data-link': link } : {}, parent), opacity);
  const right = lerp(x0, x1, reveal);
  if (link) G.svgEl('rect', { class: 'g-frame', x: x0 - 2, y: y - 7, width: x1 - x0 + 4, height: 28, rx: 3, fill: 'none', stroke: 'none' }, g);
  rule(g, `M${x0} ${y - 4}V${y}H${right}${reveal >= 1 ? `V${y - 4}` : ''}`);
  if (reveal >= 1) {
    const tx = anchor === 'middle' ? (x0 + x1) / 2 : anchor === 'end' ? x1 : x0;
    label(g, tx, y + 12, text, { anchor, cls: '' });
  }
  return g;
}

// ---- the pipeline ----
const BLOCK_ORDER = ['you', 'router', 'r1', 'r2', 'r3', 'scheduler'];
export const IDLE = Object.freeze(Object.fromEntries(BLOCK_ORDER.map((k) => [k, 'idle'])));
export const CHOSEN = Object.freeze({ ...IDLE, r1: 'dim', r2: 'active', r3: 'dim' }); // replica 2 picked (frames 2–8)

export function pipeline(svg, states = IDLE) {
  region(svg, REGION);
  label(svg, REGION.x + 8, 14, 'inside replica 2');
  G.block(svg, { ...YOU, label: 'you', state: states.you });
  G.block(svg, { ...ROUTER, label: 'router', state: states.router });
  REPLICAS.forEach((box, i) => G.block(svg, { ...box, label: `replica ${i + 1}`, state: states[`r${i + 1}`] }));
  G.block(svg, { ...SCHEDULER, label: 'scheduler', state: states.scheduler });
  G.gpu(svg, { ...GPU, memFill: WEIGHTS_FILL });
  label(svg, center(GPU), 16, 'GPU (H200)', { anchor: 'middle' });
}

export const HOPS = Object.freeze({
  toRouter: [[YOU.x + YOU.w, PIPE_CY], [ROUTER.x, PIPE_CY]],
  toReplica: [[ROUTER.x + ROUTER.w, PIPE_CY], [REPLICAS[1].x, PIPE_CY]],
  toScheduler: [[REPLICAS[1].x + REPLICAS[1].w, PIPE_CY], [SCHEDULER.x, PIPE_CY]],
  toGpu: [[SCHEDULER.x + SCHEDULER.w, PIPE_CY], [GPU.x, PIPE_CY]],
});
export function hop(svg, name, progress) {
  if (progress <= 0 || progress >= 1) return;
  const [from, to] = HOPS[name];
  G.flow(svg, { from, to, carry: 'token', progress });
}

// ---- your request's pieces ----
// The prompt chips The cat sat at x (typed: how many have appeared), framed as "your request".
export function promptChips(svg, { x, y = CHIP_Y, typed = PROMPT.length, opacity = 1, caption = true }) {
  if (opacity <= 0 || typed <= 0) return;
  const g = layer(svg, opacity);
  const widths = chipWidths(PROMPT);
  let cx = x;
  PROMPT.slice(0, typed).forEach((t, i) => {
    G.token(g, { x: cx, y, text: t, index: i + 1 });
    cx += widths[i] + CHIP_GAP;
  });
  select(g, x, y, cx - CHIP_GAP - x, CHIP_H);
  if (caption) label(g, x, y + 42, 'your request');
}

// One picked token (answer position i) at x, y, framed as part of your request.
export function answerChip(svg, i, { x, y = CHIP_Y, opacity = 1 }) {
  if (opacity <= 0) return;
  const g = layer(svg, opacity);
  G.token(g, { x, y, text: ANSWER[i], index: 4 + i });
  select(g, x, y, G.tokenWidth(ANSWER[i]), CHIP_H);
}

// The tokens already streamed back to you, as one framed group under "you".
export function answerRow(svg, count, opacity = 1) {
  if (count <= 0 || opacity <= 0) return;
  const g = layer(svg, opacity);
  ANSWER.slice(0, count).forEach((t, i) => G.token(g, { x: ANSWER_X[i], y: CHIP_Y, text: t, index: 4 + i }));
  select(g, YOU.x, CHIP_Y, groupWidth(ANSWER.slice(0, count)), CHIP_H);
  label(g, YOU.x, LABEL_Y, 'streamed to you');
}

// Your request's K/V tiles (one pair per stored token), framed; `faint` (0 → 1) steps them back once the cache is freed and
// `tiles` fades the tiles alone (the "KV cache" label stays).
export function cacheTiles(svg, count, { faint = 0, tiles = 1 } = {}) {
  label(svg, KV.x, KV.y - 8, 'KV cache');
  if (count <= 0 || tiles <= 0) return;
  const g = layer(svg, tiles);
  const strength = 1 - 0.65 * faint;
  G.kvStack(layer(g, strength), { x: KV.x, y: KV.y, count, tile: KV.tile });
  select(g, KV.x, KV.y, count * KV.pitch - 2, 2 * KV.tile + 2, strength);
}

// Your request's bar on the time axis: `steps` in px (G.request steps); none yet → only its label.
export function yourBar(svg, steps, opacity = 1) {
  if (opacity <= 0) return;
  const g = layer(svg, opacity);
  rule(g, `M${BAR.x} ${AXIS_Y}H${STAGE.w - 40}`);
  label(g, STAGE.w - 34, AXIS_Y, 'time', { cls: 'g-label' });
  const end = steps.length ? steps.at(-1).to : 0;
  if (steps.length) G.request(g, { x: BAR.x, y: BAR.y, steps, label: 'your request' });
  else label(g, BAR.x - 8, BAR.y + BAR.h / 2, 'your request', { anchor: 'end' });
  const left = BAR.x - 8 - 'your request'.length * CHAR_W;
  select(g, left - 2, BAR.y - 2, BAR.x + end - left + 4, BAR.h + 4);
}

// The bar's steps after prefill grew to `prefill` (0–1) and `ticks` decode steps landed (frames 4–8).
export function barSteps(prefill, ticks) {
  if (prefill <= 0) return [];
  const steps = [{ from: 0, to: PREFILL_W * prefill, kind: 'prefill' }];
  for (let k = 1; k <= ticks; k += 1) steps.push({ from: tickX(k - 1), to: tickX(k), kind: 'decode' });
  return steps;
}

export const ttftBracket = (svg, reveal, opacity = 1) => bracket(svg, { x0: BAR.x, x1: BAR.x + PREFILL_W, text: 'TTFT', link: 'ttft', reveal, opacity });
export const tpotBracket = (svg, reveal, opacity = 1) => bracket(svg, { x0: BAR.x + tickX(1), x1: BAR.x + tickX(2), text: 'TPOT', link: 'tpot', reveal, opacity });

// The return trip of a picked token: the chip slides from the out slot to its place under "you" along a token flow.
export function streamBack(svg, i, t) {
  if (t <= 0) return;
  const from = outX(ANSWER[i]);
  const x = lerp(from, ANSWER_X[i], ease(t));
  if (t < 1) G.flow(svg, { from: [from + G.tokenWidth(ANSWER[i]) / 2, RETURN_Y], to: [ANSWER_X[i] + G.tokenWidth(ANSWER[i]) / 2, RETURN_Y], carry: 'token', progress: ease(t) });
  answerChip(svg, i, { x });
}

// A token rising from the chip row into the GPU (t 0 → 1), fading as it enters.
export function intoGpu(svg, x, t, draw) {
  if (t >= 1) return;
  const y = lerp(CHIP_Y, GPU.y + 18, ease(t));
  draw({ x, y, opacity: 1 - seg(t, 0.55, 1) });
}

// A picked token leaving the GPU (t 0 → 1) down to the out slot.
export function outOfGpu(svg, i, t) {
  if (t <= 0) return;
  const y = lerp(GPU.y + 18, CHIP_Y, ease(t));
  answerChip(svg, i, { x: outX(ANSWER[i]), y, opacity: seg(t, 0, 0.4) });
}
