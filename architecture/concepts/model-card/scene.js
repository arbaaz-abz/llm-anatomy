// model-card scenes (pure, no DOM): every frame is the mix of the previous frame's end scene and its own, so a frame
// starts exactly where the last one ended and scrubbing back and forth is exact (spec §5.3).
// A scene is a flat table of numbers 0–1: group opacities, selection-mark opacities and animation progress.
import { clamp01, seg, lerp } from './stage.js';

const CHIPS = 10;
const chipKeys = (prefix) => Array.from({ length: CHIPS }, (_, i) => `${prefix}${i}`);

export const SCENE_KEYS = Object.freeze([
  'col', 'chips', ...chipKeys('s'), 'sa', 'sb', 'x4', 'x7', 'x9', // card column: chips typed in, selection marks, extra chips
  'diag', 'typed', 'litAttn', 'litMoe', 'selStack', 'selExperts', 'selKv', 'selLane', 'selPatch', // the diagram
  'ex', 'split', 'kv', 'fuse', 'bars', 'cache', 'gpu', 'ctx', 'stretch', 'modal', // the right panel and the input lane
  'exits', 'conf', 'cs0', 'cs1', 'cs2', 'cs3', 'walk', // frames 9 and 10
]);

const blank = Object.fromEntries(SCENE_KEYS.map((k) => [k, 0]));
const next = (prev, patch) => Object.freeze({ ...prev, ...patch });

const START = next(blank, { col: 1, diag: 1, ex: 1, kv: 1 });
const F1 = next(START, { chips: 1 });
const F2 = next(F1, { s0: 1, s1: 1, litMoe: 1, bars: 1, ex: 0, kv: 0 });
const F3 = next(F2, { s0: 0, s1: 0, s2: 1, litMoe: 0, bars: 0, ex: 1, kv: 1, typed: 1, selStack: 1 });
const F4 = next(F3, { s2: 0, s3: 1, selStack: 0, split: 1, x4: 1, selExperts: 1 });
const F5 = next(F4, { s3: 0, s4: 1, selExperts: 0, split: 0, x4: 0, litAttn: 1, fuse: 1, selKv: 1 });
const F6 = next(F5, { litAttn: 0, selKv: 0, fuse: 0, ex: 0, kv: 0, cache: 1, gpu: 1 });
const F7 = next(F6, { s4: 0, s5: 1, cache: 0, gpu: 0, ctx: 1, stretch: 1, x7: 1, selLane: 1 });
const F8 = next(F7, { s5: 0, s6: 1, ctx: 0, stretch: 0, x7: 0, selLane: 0, modal: 1, selPatch: 1 });
const F9 = next(F8, { s6: 0, s7: 1, s8: 1, sa: 1, sb: 1, modal: 0, selPatch: 0, x9: 1, exits: 1, diag: 0.25 });
const F10 = next(F9, { s7: 0, s8: 0, sa: 0, sb: 0, col: 0, x9: 0, exits: 0, diag: 0, conf: 1, cs0: 1, cs1: 1, cs2: 1, cs3: 1, walk: 1 });

export const ENDS = Object.freeze([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]);
export const FRAME_COUNT = ENDS.length;

// What leaves fades first, what arrives fades in after it, so two groups never overlap at half strength.
const LEAVE = Object.freeze([0, 0.4]);
const ARRIVE = Object.freeze([0.3, 1]);
// Frame 10's four conflicts split one after another.
const WINDOWS = Object.freeze({ cs0: [0.3, 0.55], cs1: [0.45, 0.7], cs2: [0.6, 0.85], cs3: [0.75, 1] });

const windowFor = (key, from, to) => WINDOWS[key] ?? (to < from ? LEAVE : ARRIVE);

export function sceneAt(index, progress) {
  if (!Number.isInteger(index) || index < 0 || index >= ENDS.length) throw new RangeError(`model-card: no frame ${index + 1} (there are ${ENDS.length})`);
  const [from, to] = [index === 0 ? START : ENDS[index - 1], ENDS[index]];
  const p = clamp01(progress);
  return Object.fromEntries(SCENE_KEYS.map((key) => {
    const [a, b] = windowFor(key, from[key], to[key]);
    return [key, lerp(from[key], to[key], seg(p, a, b))];
  }));
}
