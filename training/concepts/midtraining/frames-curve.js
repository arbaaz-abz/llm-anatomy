// Frames 1-5: the learning-rate schedule, then why its end matters and the named mid-training stage (storyboard §5).
// Each frame is a pure function of its progress p (0 -> 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { lrAt } from '@math/schedule.js';
import { WARMUP_DRAW, BRANCH, DECAY_START, PLAIN_COSINE_AT } from './numbers.js';
import { countText } from './facts.js';
import { ofPeak, sci, trimNumber, percentText } from './format.js';
import {
  PLOT, TEXT_Y, TOP_Y, seg, lerp, ease, label, note, select, linked, rule, outgoing, drawPlot, bandBox, partialCurve,
} from './stage.js';

const WARMUP = Object.freeze({ kind: 'wsd', total: 1, warmup: WARMUP_DRAW, decayStart: DECAY_START });
const COSINE = Object.freeze({ kind: 'cosine', total: 1 });
const BRANCH_OPTIONS = Object.freeze({ kind: 'wsd', total: BRANCH.to, decayStart: BRANCH.from });
const BANDS = Object.freeze([
  { from: 0, to: WARMUP_DRAW, label: 'warmup' },
  { from: WARMUP_DRAW, to: DECAY_START, label: 'stable' },
  { from: DECAY_START, to: 1, label: 'decay' },
]);
const STOP = 0.6; // the stop point of frame 2 and of the toy's default

const decayBox = () => bandBox(BANDS, 2);

// Frame 1: the warmup ramp from 0 to the peak.
export function drawFrame1(svg, p, { run }) {
  const upto = WARMUP_DRAW * ease(seg(p, 0.05, 0.7));
  const { nemotron } = run;
  drawPlot(svg, { series: upto > 0 ? [{ points: partialCurve(WARMUP, upto), label: '' }] : [], bands: [BANDS[0]] });
  label(svg, 4, TOP_Y, `Nemotron 3 Super: warmup over the first ${countText(nemotron.warmupTokens)} of ${countText(nemotron.total)} tokens, to ${sci(nemotron.peak)}`, { opacity: seg(p, 0.5, 0.9) });
  note(svg, 4, TEXT_Y[0], `The warmup is drawn ${percentText(WARMUP_DRAW * 100)} of the run wide so it shows; Nemotron's real one is ${trimNumber(nemotron.warmupPercent)}% of its run.`, { opacity: seg(p, 0.6, 0.95) });
}

// Frame 2: the cosine curve, and the 60% mark dropping onto it.
export function drawFrame2(svg, p, ctx) {
  outgoing(svg, (g) => drawFrame1(g, 1, ctx), 1 - seg(p, 0, 0.25));
  const upto = ease(seg(p, 0.1, 0.75));
  const stop = lrAt(STOP, COSINE);
  const markers = upto >= STOP ? [{ x: STOP, y: lerp(1, stop, ease(seg(p, 0.75, 1))), label: `60%: ${ofPeak(stop)} of peak`, followed: true }] : [];
  drawPlot(svg, { series: upto > 0 ? [{ points: partialCurve(COSINE, upto), label: 'cosine' }] : [], markers });
  const line = PLAIN_COSINE_AT.map((t) => `${percentText(t * 100)} ${ofPeak(lrAt(t, COSINE))}`).join(' · ');
  label(svg, 4, TEXT_Y[0], `cosine at ${line}`, { opacity: seg(p, 0.7, 0.95) });
}

// The WSD curve with its three bands, finished (frames 3-5). `branch` adds the muted branch series.
function wsdPlot(svg, { upto = 1, branchUpto = null, bands = BANDS } = {}) {
  const series = [];
  if (upto > 0) series.push({ points: partialCurve(WARMUP, upto), label: 'WSD' });
  if (branchUpto !== null) series.push({ points: partialCurve(BRANCH_OPTIONS, branchUpto, { from: BRANCH.from }), label: 'branch', style: 'muted', labelAt: 'mid' });
  drawPlot(svg, { series, bands: bands.filter((b) => upto > b.from) });
}

// The decay window, outlined (frames 3-5), with its math link.
function decayWindow(svg, opacity = 1) {
  const box = decayBox();
  linked(svg, 'd', box);
  select(svg, box, opacity);
}

// Frame 3: warmup-stable-decay, and a branch peeling off the plateau at 60%.
export function drawFrame3(svg, p, ctx) {
  outgoing(svg, (g) => drawFrame2(g, 1, ctx), 1 - seg(p, 0, 0.2));
  const upto = ease(seg(p, 0.1, 0.5));
  const branchUpto = p < 0.6 ? null : lerp(BRANCH.from, BRANCH.to, ease(seg(p, 0.6, 0.95)));
  wsdPlot(svg, { upto, branchUpto });
  decayWindow(svg, seg(p, 0.5, 0.7));
  linked(svg, 'peak', { x: PLOT.x + 40, y: PLOT.y + 20, w: 200, h: 14 });
  label(svg, 4, TOP_Y, 'DeepSeek-V4: constant, then a cosine tail to 10% (start not published)', { opacity: seg(p, 0.5, 0.9) });
  const v = (t) => trimNumber(lrAt(t, WARMUP));
  label(svg, 4, TEXT_Y[0], `WSD: ${v(0.5)} until ${percentText(DECAY_START * 100)}, linear decay ${v(0.9)} at 90%, ${v(1)} at 100% · branch from ${percentText(BRANCH.from * 100)}`, { opacity: seg(p, 0.7, 0.95) });
}

const MIX = Object.freeze([
  { label: 'web', w: 64, lit: false },
  { label: 'high-quality', w: 100, lit: true },
  { label: 'reasoning', w: 88, lit: true },
  { label: 'code', w: 60, lit: true },
  { label: 'agent traces', w: 96, lit: true },
]);
const MIX_GAP = 6;
const MIX_Y = 6;
const MIX_H = 28;

// Frame 4: the data mixture changes as the curve enters the decay.
export function drawFrame4(svg, p, ctx) {
  outgoing(svg, (g) => drawFrame3(g, 1, ctx), 1 - seg(p, 0, 0.25));
  wsdPlot(svg);
  decayWindow(svg);
  const switched = seg(p, 0.3, 0.6) >= 0.5;
  const total = MIX.reduce((sum, m) => sum + m.w, 0) + MIX_GAP * (MIX.length - 1);
  const box = decayBox();
  let x = PLOT.x + PLOT.w - total;
  MIX.forEach((m) => {
    G.block(svg, { x, y: MIX_Y, w: m.w, h: MIX_H, label: m.label, state: !switched ? 'idle' : m.lit ? 'active' : 'dim' });
    x += m.w + MIX_GAP;
  });
  const cx = box.x + box.w / 2;
  G.flow(svg, { from: [cx, MIX_Y + MIX_H + 2], to: [cx, PLOT.y + 22], carry: 'token', progress: seg(p, 0.3, 0.9) });
  label(svg, 4, TEXT_Y[0], 'dims: web · lights: high-quality · reasoning · code · agent traces', { opacity: seg(p, 0.6, 0.85) });
  label(svg, 4, TEXT_Y[1], ctx.text.nemotronDecay, { opacity: seg(p, 0.7, 0.9) });
  label(svg, 4, TEXT_Y[2], ctx.text.minimaxDecay, { opacity: seg(p, 0.8, 1) });
}

// Frame 5: the decay band is the named stage, mid-training.
export function drawFrame5(svg, p, ctx) {
  outgoing(svg, (g) => drawFrame4(g, 1, ctx), 1 - seg(p, 0, 0.25));
  wsdPlot(svg);
  decayWindow(svg);
  const box = decayBox();
  const reach = ease(seg(p, 0.1, 0.5));
  rule(svg, box.x + box.w * (1 - reach), 38, box.x + box.w, 38);
  label(svg, box.x + box.w / 2, 26, 'mid-training', { anchor: 'middle', cls: '', opacity: seg(p, 0.1, 0.4) });
  const { text } = ctx;
  [text.glmMid, text.olmo, text.deepseekMid].forEach((line, i) => label(svg, 4, TEXT_Y[i], line, { opacity: seg(p, 0.45 + i * 0.12, 0.6 + i * 0.12) }));
  label(svg, 4, TEXT_Y[3], text.glmShare, { opacity: seg(p, 0.8, 0.95) });
  label(svg, 4, TEXT_Y[4], text.glmSum, { opacity: seg(p, 0.8, 0.95) });
}
