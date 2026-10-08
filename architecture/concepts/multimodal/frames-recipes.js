// multimodal frame 10 (storyboard §5): the adapter recipe and the native recipe, side by side. A comparison, not a timeline step.
import * as G from '@shared/glyphs.js';
import * as S from './stage.js';
import { drawFrame9 } from './frames-counts.js';

const { seg, layer, note, noteLines, guide, leaving, NOTE_Y } = S;
const BLOCK = Object.freeze({ w: 76, h: 40, gap: 18 });
const COLUMN_X = Object.freeze({ adapter: 16, native: 304 });
const BLOCKS_Y = 112;
const NAMES = Object.freeze(['encoder', 'projector', 'backbone']);
const blockX = (side, i) => COLUMN_X[side] + i * (BLOCK.w + BLOCK.gap);

// One pipeline: encoder → projector → backbone, with a short flow between neighbors.
function pipeline(parent, side, opacity) {
  const g = layer(parent, opacity);
  NAMES.forEach((label, i) => G.block(g, { x: blockX(side, i), y: BLOCKS_Y, w: BLOCK.w, h: BLOCK.h, label, state: 'idle' }));
  [0, 1].forEach((i) => {
    const y = BLOCKS_Y + BLOCK.h / 2;
    G.flow(g, { from: [blockX(side, i) + BLOCK.w + 1, y], to: [blockX(side, i + 1) - 2, y], carry: 'activation', progress: 0 });
  });
}

// A bracket under blocks [from, to] with a centered label.
function bracket(parent, side, from, to, text, opacity) {
  const [x1, x2] = [blockX(side, from), blockX(side, to) + BLOCK.w];
  const y = BLOCKS_Y + BLOCK.h + 12;
  const g = layer(parent, opacity);
  guide(g, x1, y, x2, y, 0.6);
  guide(g, x1, y - 4, x1, y, 0.6);
  guide(g, x2, y - 4, x2, y, 0.6);
  note(g, (x1 + x2) / 2, y + 16, text, { anchor: 'middle' });
}

export function drawFrame10(svg, p, ctx) {
  const gone = leaving(p);
  if (gone > 0) drawFrame9(layer(svg, gone), 1, ctx);
  const enter = seg(p, 0.1, 0.3);
  note(svg, 16, 24, 'two recipes side by side (a comparison, not a step in time)', { opacity: enter });
  note(svg, COLUMN_X.adapter, 80, 'adapter', { opacity: enter });
  note(svg, COLUMN_X.native, 80, 'native', { opacity: enter });
  pipeline(svg, 'adapter', enter);
  pipeline(svg, 'native', enter);
  bracket(svg, 'adapter', 0, 1, 'pretrained, attached later', seg(p, 0.3, 0.5));
  note(svg, blockX('adapter', 2) + BLOCK.w / 2, BLOCKS_Y + BLOCK.h + 28, 'already trained on text', { anchor: 'middle', opacity: seg(p, 0.3, 0.5) });
  bracket(svg, 'native', 0, 2, 'trained together from step 0', seg(p, 0.55, 0.85));
  noteLines(svg, 16, NOTE_Y[0], ['both keep an encoder: native means trained together,', 'not "no encoder"'], seg(p, 0.85, 1));
}
