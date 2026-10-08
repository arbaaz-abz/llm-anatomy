// Frames 8–9: the comparison. Frame 8 sets the toy layer beside DeepSeek-V3's shape; frame 9 is the four real models.
// The real shapes come from data/models.json (`shapes`), never from this file.
import * as G from '@shared/glyphs.js';
import { formatBytes } from '@math/core.js';
import { kvCacheBytes } from '@math/memory.js';
import { LINES, STACK, seg, lerp, ease, key, label, ghost } from './stage.js';
import { drawFrame7 } from './frames-mla.js';
import { TOY_SHAPE, DEFAULT_CONTEXT } from './numbers.js';
import { ladder, mlaRatio, realHeadBaseline, compare } from './ladder.js';
import { int, timesText } from './format.js';

const ROW = Object.freeze({ y: 82, pitch: 34, name: 29, toy: 300, v3: 520 });
const LADDER_ORDER = Object.freeze(['mha', 'gqa8', 'gqa2', 'mqa', 'mla']); // MLA lands last, into the slot its toy number gives it
const SLOT = Object.freeze({ mha: 0, gqa8: 1, gqa2: 2, mla: 3, mqa: 4 });
const COLUMN = Object.freeze({ centers: [80, 220, 360, 500], name: 56, bytes: 142, size: 218 });

const cell = (value) => (value === null ? '—' : int(value));

export function drawFrame8(svg, p, { shapes }) {
  const v3 = shapes.v3;
  const toy = ladder(TOY_SHAPE);
  const real = ladder(v3);
  ghost(svg, drawFrame7, 1 - seg(p, 0, 0.25));
  const head = seg(p, 0.25, 0.4);
  key(svg, ROW.name, 44, 'per token per layer (numbers)', { opacity: head });
  key(svg, ROW.toy, 62, `toy: ${TOY_SHAPE.queryHeads} heads × ${TOY_SHAPE.headDim}`, { anchor: 'end', opacity: head });
  key(svg, ROW.v3, 62, `${v3.name}: ${v3.queryHeads} heads × ${v3.headDim}`, { anchor: 'end', opacity: head });
  LADDER_ORDER.forEach((name, order) => {
    const t = seg(p, 0.3 + 0.09 * order, 0.4 + 0.09 * order);
    const i = toy.findIndex((r) => r.key === name);
    const y = ROW.y + SLOT[name] * ROW.pitch + (name === 'mla' ? -12 * (1 - ease(t)) : 0);
    label(svg, ROW.name, y, toy[i].label, { cls: 'g-key', opacity: t });
    label(svg, ROW.toy, y, cell(toy[i].value), { cls: 'g-key', anchor: 'end', opacity: t });
    label(svg, ROW.v3, y, cell(real[i].value), { cls: 'g-key', anchor: 'end', opacity: t });
  });
  const notes = seg(p, 0.8, 1);
  label(svg, LINES.x, LINES.y[0] - 22, '— GQA-8 needs more than 8 query heads', { opacity: notes });
  key(svg, LINES.x, LINES.y[0], `MHA ÷ MLA at ${v3.name}'s shape = ${timesText(mlaRatio(real))}`, { opacity: notes });
  label(svg, LINES.x, LINES.y[1], `at ${v3.queryHeads} heads of ${v3.headDim} numbers; the real query/key head is ${v3.qkWidth} wide,`, { opacity: notes });
  label(svg, LINES.x, LINES.y[2], `which would make the baseline ${Math.round(realHeadBaseline(v3))}×`, { opacity: notes });
}

// One model's column: name, scheme, its four-token stack, bytes per token (all layers), and the cache at the default context.
function column(svg, x, c, { appear, size }) {
  const { shape } = c;
  label(svg, x, COLUMN.name, shape.name, { cls: 'g-key', anchor: 'middle', opacity: appear });
  label(svg, x, COLUMN.name + 18, c.scheme, { anchor: 'middle', opacity: appear });
  const box = G.svgEl('g', { 'data-link': 'kvh' }, svg);
  G.svgEl('rect', { class: 'g-frame', x: x - STACK.w / 2 - 16, y: 92, width: STACK.w + 16, height: STACK.h + 4, rx: 3, fill: 'none', stroke: 'none' }, box);
  G.kvStack(box, { x: x - STACK.w / 2, y: 94, count: STACK.tiles, tile: STACK.tile });
  box.style.opacity = String(appear);
  label(svg, x, COLUMN.bytes, formatBytes(c.perToken), { cls: 'g-key', anchor: 'middle', opacity: appear });
  label(svg, x, COLUMN.bytes + 16, `${int(c.perToken)} B`, { anchor: 'middle', opacity: appear });
  label(svg, x, COLUMN.size - 18, `at ${int(DEFAULT_CONTEXT)} tokens`, { anchor: 'middle', opacity: size });
  label(svg, x, COLUMN.size, formatBytes(c.atContext), { cls: 'g-key', anchor: 'middle', opacity: size });
  if (c.isWhatIf) label(svg, x, COLUMN.size + 18, 'a what-if', { anchor: 'middle', opacity: size });
}

export function drawFrame9(svg, p, { shapes }) {
  const models = [shapes.gpt3, shapes.llama, shapes.minimax, shapes.v3].map(compare);
  ghost(svg, (g, q) => drawFrame8(g, q, { shapes }), 1 - seg(p, 0, 0.25));
  models.forEach((c, i) => column(svg, COLUMN.centers[i], c, { appear: seg(p, 0.2 + 0.1 * i, 0.4 + 0.1 * i), size: seg(p, 0.7, 0.95) }));
  const [first, last] = [models[0], models[3]];
  const ratio = first.perToken / last.perToken;
  key(svg, LINES.x, LINES.y[1] + 8, `${first.shape.name} ÷ ${last.shape.name} = ${timesText(ratio)} per token`, { opacity: seg(p, 0.85, 1) });
  label(svg, LINES.x, LINES.y[2] + 8, 'all layers, 2 bytes a number', { opacity: seg(p, 0.85, 1) });
}
