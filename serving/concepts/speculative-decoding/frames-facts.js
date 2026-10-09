// speculative-decoding frame 10: three measured systems, each row with its own setup. `text` is facts.js stageText(data),
// so every number printed here comes from data/serving.json (a "—" where the data is missing).
import * as G from '@shared/glyphs.js';
import { layer, leaving, note, seg } from './stage.js';
import { drawFrame9 } from './frames-curves.js';

const ROWS_Y = Object.freeze([62, 140, 218]);
const BLOCK = Object.freeze({ x: 24, w: 120, h: 44 });
const DETAIL_X = 164;

export function drawFrame10(svg, p, text) {
  if (p < 0.15) drawFrame9(layer(svg, leaving(p)), 1);
  note(layer(svg, seg(p, 0, 0.15)), 24, 28, "measured by the papers' authors, not this toy");
  text.rows.forEach((row, i) => {
    const g = layer(svg, seg(p, 0.1 + 0.25 * i, 0.35 + 0.25 * i));
    G.block(g, { x: BLOCK.x, y: ROWS_Y[i], w: BLOCK.w, h: BLOCK.h, label: row.name });
    note(g, DETAIL_X, ROWS_Y[i] + 18, row.setup);
    note(g, DETAIL_X, ROWS_Y[i] + 36, row.result);
  });
}
