// prefix-caching stage layout: fixed positions shared by every frame, the timing helpers and small drawing helpers built on
// the glyph library. The tree, pool and counters keep one anchor in frames 1-8; frames 9-11 draw their own scenes.
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const CHAR_W = 6.6; // JetBrains Mono at the 11 px label size
export const TREE = Object.freeze({ x: 8, y: 22, labelY: 12 });
export const WALK = Object.freeze({ chipY: 131, x: 44, y: Object.freeze([147, 161]), max: 74 }); // the followed request's blocks in full, two lines at most
export const NOTE_Y = 177; // the numbers a frame names
export const POOL = Object.freeze({ x: 8, y: 198, labelY: 192, perRow: 4, blockSize: 4, blocks: 8 });
export const STRIP = Object.freeze({ x: 8, y: 300, gap: 8 }); // the active request's block table
export const QUEUE_Y = 350;
export const FRAME1_NOTES_Y = Object.freeze([308, 322, 336]);
export const COUNTERS = Object.freeze({ labelX: 474, cellX: 480, y: 200, pitch: 41, headY: 190 });
export const TABLE = Object.freeze({ x: 372, step: 48, y: 200, pitch: 41, headY: 190, letterX: 336 });
export const BIG_Y = 330;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
// Each frame starts from the previous frame's end state (template rule 4): what leaves fades out and what arrives fades in
// during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);
export const at = (p, t) => p >= t; // a discrete change that happens at progress t

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15), styled like glyph labels; cls '' prints in ink instead of muted.
// `scale` enlarges the text (the theme fixes the diagram text size, so a big number is the same text drawn scaled).
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start', scale = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note', transform: scale === 1 ? null : `translate(${x} ${y}) scale(${scale})` }, parent);
  const t = G.svgEl('text', { x: scale === 1 ? x : 0, y: scale === 1 ? y : 0, class: cls || null, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// Words wrapped to lines of at most `max` characters (a word is never split).
export function wrapLines(parts, max, joiner = ' · ') {
  const lines = [];
  let line = '';
  parts.forEach((part) => {
    const next = line === '' ? part : `${line}${joiner}${part}`;
    if (next.length > max && line !== '') { lines.push(line); line = part; } else line = next;
  });
  return line === '' ? lines : [...lines, line];
}

export function select(parent, box, opacity = 1) {
  if (opacity > 0) G.selectionMark(layer(parent, opacity), box);
}
