// model-card frames, the diagram's right panel: the expert row (frames 1 and 4), the KV cache (1 and 5), the active
// share bars (2), the cache readout with its GPU (6), the context labels (7) and the modality note (8).
import * as G from '@shared/glyphs.js';
import { PANEL, layer, note, mark, rule, lerp, seg } from './stage.js';
import { sharePercent } from './format.js';

const P = PANEL.x;
const ROW = Object.freeze({ shared: { w: 40 }, boxH: 22, routed: { x: P + 46, w: 130 }, small: { w: 14, step: 16 }, rest: { w: 36 } });

// ---- experts ----

function expertRow(svg, model, scene) {
  const { experts } = model;
  const lit = scene.split > 0.5;
  const y = 44;
  note(svg, P, 36, 'experts, one MoE layer');
  G.block(svg, { x: P, y, w: ROW.shared.w, h: ROW.boxH, label: 'shared', state: lit ? 'active' : 'idle' });
  if (scene.split < 1) G.block(layer(svg, 1 - scene.split), { x: ROW.routed.x, y, w: ROW.routed.w, h: ROW.boxH, label: `${experts.total} routed`, state: 'idle' });
  if (scene.split > 0) {
    const g = layer(svg, scene.split);
    for (let i = 0; i < experts.used; i += 1) G.block(g, { x: ROW.routed.x + i * ROW.small.step, y, w: ROW.small.w, h: ROW.boxH, label: '', state: 'active' });
    G.block(g, { x: ROW.routed.x + experts.used * ROW.small.step + 4, y, w: ROW.rest.w, h: ROW.boxH, label: String(experts.others), state: 'idle' });
    note(g, P, 84, `${experts.used} ÷ ${experts.total} = ${experts.routedText} routed per token`);
    note(g, P, 98, `${experts.total} experts, ${experts.hidden} wide each`);
  }
}

// ---- KV cache ----

const KV = Object.freeze({ title: 122, y: 130, tile: 14, step: 16, groups: 2, x: 20, boxH: 142 }); // two merge groups of tiles stand for the stored tokens

function kvPanel(svg, model, scene) {
  const { kv } = model;
  const count = KV.groups * kv.merge;
  const x = P + KV.x; // the K and V letters sit 8 px left of the tiles, inside the selection box
  note(svg, P, KV.title, `KV cache: ${kv.heads} head × ${kv.headDim}`);
  G.kvStack(svg, { x, y: KV.y, count, tile: KV.tile });
  const fuse = scene.fuse;
  if (fuse <= 0) return;
  const bracketY = KV.y + 2 * KV.tile + 8;
  const merge = seg(fuse, 0, 0.35);
  for (let g = 0; g < KV.groups; g += 1) rule(layer(svg, merge), x + g * kv.merge * KV.step, bracketY, x + (g + 1) * kv.merge * KV.step - 2, bracketY);
  const merged = seg(fuse, 0.35, 0.7);
  if (merged > 0) {
    const g = layer(svg, merged);
    G.kvStack(g, { x, y: bracketY + 8, count: KV.groups, tile: KV.tile });
    note(g, x + KV.groups * KV.step + 8, bracketY + 8 + KV.tile + 4, `merge ${kv.merge}: ${count} → ${KV.groups} entries`);
  }
  const window = seg(fuse, 0.7, 1);
  if (window > 0) {
    const g = layer(svg, window);
    const hcaY = bracketY + 8 + 2 * KV.tile + 14;
    G.kvStack(g, { x, y: hcaY, count: 1, tile: KV.tile });
    note(g, x + KV.step + 8, hcaY + KV.tile + 4, `merge ${kv.hcaMerge}: ${kv.hcaMerge} → 1 entry`);
    rule(g, x + (count - 2) * KV.step, bracketY, x + count * KV.step - 2, bracketY);
    note(g, x + count * KV.step + 8, KV.y + KV.tile + 4, `window ${kv.window}`);
  }
}

// ---- active share bar ----

// One bar over the first 10% of all parameters, drawn x10: the active slice is wide enough to read, and the other 90%
// is said in words (README lesson 19). The bar's own shares are of that 10%, so its labels are scaled back to the total.
function shareBars(svg, model) {
  const { active } = model;
  const w = PANEL.w;
  const zoomed = active.zoom * active.total;
  note(svg, P, 26, active.formula);
  note(svg, P, 42, active.line);
  note(svg, P, 62, `The bar is the first ${active.zoomText} of all`);
  note(svg, P, 76, 'parameters, drawn ×10. The other');
  note(svg, P, 90, `${100 - Number.parseFloat(active.zoomText)}% is not used either.`);
  const link = G.svgEl('g', { 'data-link': 'act' }, svg);
  G.svgEl('rect', { class: 'g-frame', x: P - 2, y: 102, width: w + 4, height: 18, rx: 3, fill: 'none', stroke: 'none' }, link);
  const parts = [{ name: 'active', value: active.active, hue: 1 }, { name: `not used (in this ${active.zoomText})`, value: zoomed - active.active, hue: 2 }];
  G.shareBar(link, { x: P, y: 104, w, parts, label: `active share, the first ${active.zoomText} drawn ×10`, tail: 'none', format: (share) => sharePercent(share * active.zoom) });
  note(svg, P, 196, 'work per token follows active');
}

// ---- cache readout, GPU ----

function cacheReadout(svg, model, scene) {
  const { cache } = model;
  const g = G.svgEl('g', { 'data-link': 'kv' }, svg);
  G.svgEl('rect', { class: 'g-frame', x: P - 4, y: 22, width: PANEL.w + 4, height: 82, rx: 3, fill: 'none', stroke: 'none' }, g);
  note(g, P, 34, 'cache per token (an estimate)');
  note(g, P, 54, `about ${cache.perTokenText}`, { cls: 'g-text' });
  G.token(g, { x: P + 130, y: 40, text: 'reported' });
  note(g, P, 76, `× ${cache.tokensText} tokens =`);
  note(g, P, 94, `about ${cache.conversationText}`, { cls: 'g-text' });
  note(svg, P, 124, 'for scale, Llama-3.1-70B GQA,');
  note(svg, P, 138, `same context: ${cache.scaleText}`);
  const fill = lerp(0, cache.gpuFill[0], scene.gpu);
  G.gpu(svg, { x: P, y: 154, w: 96, h: 64, memFill: fill, label: cache.gpuLabel });
  note(svg, P + 104, 170, `low end: ${cache.lowShare}%`);
  note(svg, P + 104, 186, `high end: ${cache.highShare}%`);
  note(svg, P + 104, 202, 'of GPU memory');
  note(svg, P, 250, `paper: about ${cache.paperRatioText} of V3.2's KV`);
}

// ---- context, modality ----

function contextLabels(svg, model) {
  const { context } = model;
  note(svg, P, 36, `RoPE base ${context.ropeBase}`);
  note(svg, P, 52, `(${context.ropeBaseCompressed} for compressed streams)`);
  note(svg, P, 72, `YaRN factor ${context.yarn}`);
  note(svg, P, 98, 'trained in stages');
  note(svg, P, 114, context.stages);
}

function modalityNote(svg, model) {
  note(svg, P, 36, `${model.modality.text} (${model.modality.confidence})`, { cls: 'g-text' });
  note(svg, P, 60, 'some secondary sites call V4');
  note(svg, P, 74, 'multimodal; its paper lists');
  note(svg, P, 88, 'multimodality as future work');
}

const PANELS = Object.freeze([['ex', expertRow], ['kv', kvPanel], ['bars', shareBars], ['cache', cacheReadout], ['ctx', contextLabels], ['modal', modalityNote]]);

export function drawPanel(svg, model, scene) {
  PANELS.forEach(([key, draw]) => {
    if (scene[key] > 0) draw(layer(svg, scene[key]), model, scene);
  });
  mark(svg, scene.selKv, { x: P - 4, y: KV.title - 12, w: PANEL.w + 4, h: KV.boxH });
  mark(svg, scene.selExperts, { x: P - 4, y: 28, w: PANEL.w + 4, h: 78 });
}
