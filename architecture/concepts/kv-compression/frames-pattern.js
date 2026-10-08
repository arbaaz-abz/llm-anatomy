// Frame 4: the branch to attention's two heads. Same keys and values, different queries, so different patterns.
import * as G from '@shared/glyphs.js';
import { TOY } from '@math/attention.js';
import { CELL, HEADER, LINES, seg, ghost, fade, key, select, textWidth, weightGrid } from './stage.js';
import { drawFrame3 } from './frames-share.js';
import { patternFor } from './pattern.js';

const QUERY = 2; // "sat", the row we follow
const TOKENS = TOY.tokens;
const CHIP_X = Object.freeze([50, 98, 146, 194]);
const CHIP_Y = 36;
const MAP_Y = 124;
const MAP_X = Object.freeze({ a: 62, b: 342 });
const TITLE_Y = 88;

const { a: HEAD_A, shared: HEAD_B } = patternFor();
export const rowFilled = (p) => Math.floor(seg(p, 0.5, 1) * 4 + 1e-9); // head B's heatmap fills row by row

function head(svg, name, x, { weights, mask }, { rowsShown, opacity, followed }) {
  const title = name === 'a' ? 'head A' : 'head B, reading head A\'s keys';
  const g = G.svgEl('g', {}, svg);
  key(g, x, TITLE_Y, title);
  if (followed) select(g, x - 4, TITLE_Y - 8, textWidth(title) + 8, 16);
  weightGrid(g, { x, y: MAP_Y, weights, mask, rowsShown, rowLabels: TOKENS, colLabels: TOKENS });
  select(g, x, MAP_Y + QUERY * CELL, 4 * CELL, CELL);
  return fade(g, opacity);
}

export function drawFrame4(svg, p) {
  ghost(svg, drawFrame3, 1 - seg(p, 0, 0.3));
  const enter = seg(p, 0.3, 0.5);
  if (enter <= 0) return;
  key(svg, HEADER.x, HEADER.y, 'branch: attention\'s two heads, same keys, different questions', { opacity: seg(p, 0.2, 0.4) });
  TOKENS.forEach((t, i) => fade(G.token(svg, { x: CHIP_X[i], y: CHIP_Y, text: t, index: i + 1 }), enter));
  select(svg, CHIP_X[QUERY], CHIP_Y, G.tokenWidth(TOKENS[QUERY]), 24, enter);
  head(svg, 'a', MAP_X.a, HEAD_A, { rowsShown: 4, opacity: enter, followed: true });
  head(svg, 'b', MAP_X.b, HEAD_B, { rowsShown: rowFilled(p), opacity: enter, followed: false });
  key(svg, LINES.x, LINES.y[1], 'both heads read head A\'s keys and values; each keeps its own queries', { opacity: seg(p, 0.5, 0.8) });
}
