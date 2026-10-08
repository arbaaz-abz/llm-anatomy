// The layer-stack scenes: frame 3's six toy layers (5 window : 1 full), frame 9's four-layer groups (3 linear : 1 full) and
// frame 10's two routes. Layers are bottom to top; the residual lane runs up the left. Every function takes an opacity.
import * as G from '@shared/glyphs.js';
import { fade, scene, label, seg, ease } from './stage.js';

const LANE_X = 46;
const LAYER_X = 70;
const LAYER_W = 176;
const NOTE_X = 262;

// Draws one block per entry of `kinds`, bottom first, and returns y(i): the top edge of layer i (counted from the bottom).
function layerBlocks(holder, { kinds, top, h, gap, names, activeFrom = Infinity }) {
  const y = (i) => top + (kinds.length - 1 - i) * (h + gap);
  kinds.forEach((kind, i) => {
    const state = kind === 'full' && i >= activeFrom ? 'active' : 'idle';
    G.block(holder, { x: LAYER_X, y: y(i), w: LAYER_W, h, label: names[kind], state });
  });
  return y;
}

// ---- frame 3: 5 window layers and 1 full layer; a dot from token 1 rides the lane up to the full layer ----
const SIX = Object.freeze(['window', 'window', 'window', 'window', 'window', 'full']);
const SIX_NAMES = Object.freeze({ window: 'window attention', full: 'full attention' });

export function sixLayers(parent, p = 1, opacity = 1) {
  const holder = scene(parent, opacity);
  const top = 36;
  const h = 30;
  const gap = 10;
  const y = layerBlocks(holder, { kinds: SIX, top, h, gap, names: SIX_NAMES, activeFrom: p >= 0.8 ? 5 : Infinity });
  const bottom = y(0) + h;
  G.token(holder, { x: 10, y: bottom + 22, text: 'token 1' });
  G.flow(holder, { from: [LANE_X, bottom + 18], to: [LANE_X, top - 12], carry: 'activation', progress: ease(seg(p, 0.1, 1)) });
  label(holder, 10, 14, 'a toy stack: 5 window layers : 1 full layer');
  label(holder, NOTE_X, y(2) + h / 2, 'window layers see only their last tokens', { opacity: 1 });
  label(holder, NOTE_X, y(5) + h / 2 + 24, 'the full layer reads token 1 itself');
  G.kvStack(holder, { x: NOTE_X + 8, y: y(5) - 6, count: 10, tile: 13, label: undefined });
  label(holder, NOTE_X, bottom + 34, '1 of 6 layers grows a cache');
  return holder;
}

// ---- frame 9: groups of 3 linear layers and 1 full layer ----
const EIGHT = Object.freeze(['linear', 'linear', 'linear', 'full', 'linear', 'linear', 'linear', 'full']);
const EIGHT_NAMES = Object.freeze({ linear: 'linear: fixed state', full: 'full attention' });

export function eightLayers(parent, p = 1, opacity = 1) {
  const holder = scene(parent, opacity);
  const top = 36;
  const h = 26;
  const gap = 8;
  const built = Math.ceil(p * EIGHT.length - 1e-9); // layers appear in groups of four, bottom up
  const shown = p >= 1 ? EIGHT.length : Math.min(built, EIGHT.length);
  const y = (i) => top + (EIGHT.length - 1 - i) * (h + gap);
  EIGHT.slice(0, shown).forEach((kind, i) => {
    G.block(holder, { x: LAYER_X, y: y(i), w: LAYER_W, h, label: EIGHT_NAMES[kind], state: 'idle' });
    if (kind === 'full') G.kvStack(holder, { x: NOTE_X + 8, y: y(i) + 4, count: 10, tile: 10, label: undefined });
  });
  label(holder, 10, 14, 'a toy stack: 3 linear layers : 1 full layer, repeated');
  label(holder, NOTE_X, y(0) + h / 2, 'no cache: a fixed state');
  label(holder, NOTE_X, y(7) + h + 24, 'only full layers grow a cache', { opacity: shown === EIGHT.length ? 1 : 0 });
  return holder;
}

// ---- frame 10: two routes to one label ----
const ROUTES = Object.freeze([
  { title: 'softmax, made sparse or compressed', models: ['DeepSeek-V4-Pro', 'GLM-5.3', 'MiniMax-M3'], y: 34 },
  { title: 'linear hybrids', models: ['Qwen3.8', 'Kimi K3'], y: 200 },
]);
const MODEL_H = 26;
const TARGET = Object.freeze({ x: 430, y: 150, w: 130, h: 44 });

export function twoRoutes(parent, p = 1, opacity = 1) {
  const holder = scene(parent, opacity);
  const typed = seg(p, 0.1, 0.6) * (ROUTES[0].models.length + ROUTES[1].models.length);
  let index = 0;
  ROUTES.forEach((route) => {
    G.block(holder, { x: 20, y: route.y, w: 270, h: 28, label: route.title, state: 'idle' });
    route.models.forEach((name, k) => {
      const appear = Math.min(Math.max(typed - index, 0), 1);
      index += 1;
      if (appear > 0) fade(G.token(holder, { x: 40, y: route.y + 40 + k * (MODEL_H + 8), text: name }), appear);
    });
  });
  const arrows = ease(seg(p, 0.65, 1));
  G.block(holder, { ...TARGET, label: '1M tokens', state: arrows >= 1 ? 'active' : 'idle' });
  const targets = [[TARGET.x - 4, TARGET.y + 14], [TARGET.x - 4, TARGET.y + TARGET.h - 14]];
  [[296, 48], [296, 214]].forEach((from, i) => G.flow(holder, { from, to: targets[i], carry: 'activation', progress: arrows }));
  return holder;
}
