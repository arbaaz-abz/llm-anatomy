// decoder-recap panels 8–10: the sink joins the softmax, what each kind of layer keeps, and the side-by-side summary.
import * as G from '@shared/glyphs.js';
import { PANEL, CELL, seg, lerp, layer, note, lines } from './stage.js';
import { GPT3_FACTS, MODERN_FACTS, KEY_NAMES, QK } from './numbers.js';
import { int, fixed3 } from './format.js';

const NOTE_X = PANEL.x + 6;

// Panel 8: the three scores of the QK-norm row, a fourth "sink" cell joins them, and softmax runs over all four.
const SINK = Object.freeze({ x: 322, y: 52, restX: 450, fromX: 520 });
export function panel8(g, p) {
  note(g, NOTE_X, 22, 'softmax over the keys, plus a sink');
  note(g, SINK.x - 8, SINK.y + 25, 'scores', { anchor: 'end' });
  KEY_NAMES.forEach((name, i) => note(g, SINK.x + i * CELL + CELL / 2, SINK.y - 8, name, { anchor: 'middle' }));
  G.vector(g, { x: SINK.x, y: SINK.y, values: QK.normed(1), cell: CELL, orient: 'row', maxAbs: 2, format: fixed3 });
  const arrive = seg(p, 0.2, 0.5);
  const sink = layer(g, arrive);
  G.block(sink, { x: lerp(SINK.fromX, SINK.restX, arrive), y: SINK.y, w: CELL, h: CELL - 3, label: 'sink', state: 'active' });
  lines(layer(g, seg(p, 0.45, 0.65)), SINK.x, SINK.y + CELL + 22, ['sink: a learned "nothing here"', 'score per head']);
  const flow = layer(g, seg(p, 0.6, 0.7));
  G.flow(flow, { from: [SINK.x + 2 * CELL, SINK.y + CELL + 54], to: [SINK.x + 2 * CELL, SINK.y + CELL + 88], carry: 'activation', progress: seg(p, 0.65, 0.85) });
  G.block(flow, { x: SINK.x, y: SINK.y + CELL + 92, w: 4 * CELL + 8, h: 26, label: 'softmax' });
  lines(layer(g, seg(p, 0.85, 1)), SINK.x, SINK.y + CELL + 142, ['the weights add to 1 over the keys', 'and the sink; the sink takes what', 'no key deserves']);
  note(layer(g, seg(p, 0.85, 1)), NOTE_X, 300, 'details: long-context-attention');
}

// Panel 9: what each kind of attention layer keeps in the cache.
const KINDS = Object.freeze([
  { count: 6, text: 'full: a K, V pair per token', y: 78 },
  { count: 2, text: 'window: just the latest few', y: 138 },
  { count: 1, text: 'linear: one fixed-size state', y: 198 },
]);
export function panel9(g, p) {
  note(g, NOTE_X, 22, 'what each kind of layer keeps');
  KINDS.forEach((k, i) => {
    const o = layer(g, seg(p, 0.35 + i * 0.12, 0.5 + i * 0.12));
    G.kvStack(o, { x: NOTE_X + 16, y: k.y, count: k.count, tile: 8 });
    note(o, NOTE_X + 86, k.y + 14, k.text);
  });
  note(layer(g, seg(p, 0.75, 0.95)), NOTE_X, 262, 'only full layers grow a cache');
}

// Panel 10: GPT-3 against the same shape with every swap (counts from math/params.js and math/memory.js).
const SUM = Object.freeze({ label: NOTE_X, left: 394, right: 464, head: 46 });
export function panel10(g, p) {
  const o = layer(g, seg(p, 0.1, 0.3));
  note(o, SUM.left, SUM.head, 'GPT-3', { cls: '' });
  note(o, SUM.right, SUM.head, 'every swap', { cls: '' });
  note(o, SUM.label, 74, 'parameters');
  note(o, SUM.left, 74, GPT3_FACTS.totalShort);
  lines(o, SUM.right, 74, [`${MODERN_FACTS.stored} stored`, `${MODERN_FACTS.active} active`]);
  note(o, SUM.label, 116, 'cache per token');
  note(o, SUM.left, 116, GPT3_FACTS.cacheText);
  note(o, SUM.right, 116, MODERN_FACTS.cacheText);
  note(o, SUM.label, 144, 'position limit');
  note(o, SUM.left, 144, int(GPT3_FACTS.positions));
  note(o, SUM.right, 144, 'none');
  note(o, SUM.label, 180, 'some blocks use window or linear attention');
}
