// DOM helpers for the kv-compression toy: the wiring figure (frames 1–3 at rest) and the frame-4 pattern maps. Layout comes from theme.css.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { TOY } from '@math/attention.js';
import { Q, QUERY_HEADS, STACK, STAGE, qx, qcx, kvHead, queryRow, wire, select, CELL, weightGrid, key } from './stage.js';

const FIG_H = 190;

// Query heads Q1…Q8 above the stored heads each one reads (`groups[h]` = its KV head). MLA draws one latent block instead.
export function wiringFigure({ groups, mla, dLatent, dRope }) {
  const svg = G.svgEl('svg', { width: STAGE.w, height: FIG_H, viewBox: `0 0 ${STAGE.w} ${FIG_H}`, role: 'img', 'aria-label': 'which query head reads which stored head' });
  queryRow(svg);
  if (mla) {
    const block = { x: qx(0), y: STACK.y, w: qx(QUERY_HEADS - 1) + Q.w - qx(0), h: 28 };
    for (let h = 0; h < QUERY_HEADS; h += 1) wire(svg, { fromX: qcx(h), head: h });
    G.block(svg, { ...block, label: `latent ${dLatent} + position key ${dRope}` });
    return svg;
  }
  const kv = Math.max(...groups) + 1;
  const centers = Array.from({ length: kv }, (_, k) => {
    const heads = groups.flatMap((g, h) => (g === k ? [qcx(h)] : []));
    return (heads[0] + heads[heads.length - 1]) / 2;
  });
  groups.forEach((g, h) => wire(svg, { fromX: centers[g], head: h }));
  centers.forEach((c) => kvHead(svg, c));
  return svg;
}

const MAP = Object.freeze({ x: 44, y: 44, w: 300, h: 44 + 4 * CELL + 8 });
const TOKENS = TOY.tokens;
const QUERY = 2;

// One 4 × 4 weight heatmap (numbers printed at NUMBER_CELL), the followed row "sat" outlined; data-readout makes it testable.
export function patternMap({ name, title, weights, mask }) {
  const svg = G.svgEl('svg', { width: MAP.w, height: MAP.h, viewBox: `0 0 ${MAP.w} ${MAP.h}`, role: 'img', 'aria-label': `${title}: attention weights; rows are queries, columns are keys` });
  key(svg, MAP.x, 14, title);
  weightGrid(svg, { x: MAP.x, y: MAP.y, weights, mask, rowLabels: TOKENS, colLabels: TOKENS });
  select(svg, MAP.x, MAP.y + QUERY * CELL, 4 * CELL, CELL);
  const box = el('div', {}, [svg]);
  box.dataset.readout = name;
  return box;
}
