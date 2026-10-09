// prefix-caching stage: renderFor(text) → render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg>.
// `text` is the dated stage text (facts.js stageText, filled from the data by lessonFor). Every frame is a pure function of
// (index, progress, text): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { STAGE, POOL, TREE, layer, leaving, note, seg } from './stage.js';
import { CAPTIONS } from './captions.js';
import { sceneAt, ARRIVE } from './anim.js';
import { drawTree, drawPool } from './parts.js';
import { drawFrame9, drawFrame10, drawFrame11 } from './frames-extra.js';

const SCENE_FRAMES = 8;
const FADE = 0.12; // an overlay that arrives mid-frame fades in over this much progress

// Draws one scene: the tree and the pool as they stand, then its overlays. `prev` (the previous frame's end scene) lends the
// overlays the new frame drops, which fade out during [ARRIVE, ARRIVE + HANDOFF] while the new ones fade in.
function drawScene(parent, scene, { prev = null, p = 1 } = {}) {
  note(parent, TREE.x, TREE.labelY, 'prefix tree: each node is one full block of 4 tokens (first…last word)');
  drawTree(parent, scene.tree);
  note(parent, POOL.x, POOL.labelY, 'KV pool: 8 blocks of 4 slots');
  drawPool(parent, scene.pool.blocks, { overrides: scene.pool.overrides });
  const known = new Set(scene.overlays.map((o) => o.key));
  const before = new Set(prev?.overlays.map((o) => o.key) ?? []);
  const out = 1 - seg(p, ARRIVE, ARRIVE + 0.15);
  prev?.overlays.filter((o) => !known.has(o.key)).forEach((o) => { if (out > 0) o.draw(layer(parent, out)); });
  scene.overlays.forEach((o) => {
    const opacity = before.has(o.key) ? 1 : seg(p, o.at, o.at + FADE);
    if (opacity > 0) o.draw(layer(parent, opacity));
  });
}

const SPECIAL = Object.freeze([drawFrame9, drawFrame10, drawFrame11]);

// What a special frame fades out: frame 9 fades the whole of frame 8's end scene; 10 and 11 the previous special frame's end.
function previousFor(index, text) {
  if (index === SCENE_FRAMES) return (g) => drawScene(g, sceneAt(SCENE_FRAMES - 1, 1, text));
  return (g) => SPECIAL[index - SCENE_FRAMES - 1](g, 1, text, () => {});
}

export function renderFor(text) {
  return function render(index, progress, stage) {
    if (!(index >= 0 && index < CAPTIONS.length)) throw new RangeError(`prefix-caching: no frame ${index + 1} (there are ${CAPTIONS.length})`);
    const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
    svg.replaceChildren();
    G.hatchFill(svg); // claim the hatch pattern first, so every frame's DOM is identical however it was reached
    svg.setAttribute('aria-label', CAPTIONS[index]);
    if (index < SCENE_FRAMES) {
      const prev = index === 0 ? null : sceneAt(index - 1, 1, text);
      drawScene(svg, sceneAt(index, progress, text), { prev, p: progress });
    } else {
      SPECIAL[index - SCENE_FRAMES](svg, progress, text, previousFor(index, text));
    }
  };
}
