// The sft stage's shared kit: size, timing helpers, plain labeled notes and the transcript layout every frame reuses.
// Nothing touches the DOM at import time.
import * as G from '@shared/glyphs.js';
import { SEGMENTS, LINES, FOLLOWED } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF]
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

// ---- layout: the transcript, one line per segment group (storyboard §4) ----
export const TX0 = 76; // chips start here; role labels sit in the gutter to the left
export const LINE_Y0 = 8;
export const LINE_PITCH = 40;
export const CHIP_H = 24;
export const CHIP_GAP = 4;
export const lineY = (line) => LINE_Y0 + line * LINE_PITCH;
export const NOTE_Y = Object.freeze([248, 268, 288, 308]); // plain text lines below the transcript

// The 26 chips with their positions. `rename` maps a raw token to the text a template shows (label-only: widths change, counts never).
export function layoutTranscript(rename = {}) {
  const chips = [];
  LINES.forEach((line, lineIndex) => {
    let x = TX0;
    line.segments.forEach((segIndex) => {
      const { kind, tokens } = SEGMENTS[segIndex];
      tokens.forEach((raw) => {
        const text = rename[raw] ?? raw;
        const w = G.tokenWidth(text);
        chips.push({ flat: chips.length, segment: segIndex, kind, raw, text, x, y: lineY(lineIndex), w, line: lineIndex });
        x += w + CHIP_GAP;
      });
    });
  });
  return chips;
}
export const CHIPS = Object.freeze(layoutTranscript().map((c) => Object.freeze(c)));

// ---- drawing helpers ----
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { anchor = 'start', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  const t = G.svgEl('text', { x, y, class: 'g-label', 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return g;
}

// Role labels (user / assistant / tool) in the gutter at the start of their lines.
export function roleLabels(parent, opacityOf = () => 1) {
  LINES.forEach((line, i) => {
    const opacity = opacityOf(i);
    if (line.role && opacity > 0) note(parent, 4, lineY(i) + CHIP_H / 2, line.role, { opacity });
  });
}

// Draw chips. Options (all pure functions of the chip): opacityOf, hatchedOf, stateOf, dxOf, dyOf. The followed token carries the one selection mark.
export function drawChips(parent, chips, { opacityOf = () => 1, hatchedOf = () => false, stateOf = defaultState, dxOf = () => 0, dyOf = () => 0, followed = true } = {}) {
  chips.forEach((chip) => {
    const opacity = opacityOf(chip);
    if (opacity <= 0) return;
    const at = { x: chip.x + dxOf(chip), y: chip.y + dyOf(chip) };
    const box = layer(parent, opacity);
    G.token(box, { ...at, text: chip.text, state: stateOf(chip), hatched: hatchedOf(chip) });
    if (followed && chip.flat === FOLLOWED) G.selectionMark(box, { ...at, w: chip.w, h: CHIP_H });
  });
}

// Template tags are drawn dim so they read as markup; everything else is a plain chip.
export const defaultState = (chip) => (chip.kind === 'template' ? 'dim' : 'idle');

// Wrap a drawing in the math panel's `m` link: an invisible frame around the transcript so hovering the term outlines it.
export function linkedTranscript(parent, name = 'm') {
  const g = G.svgEl('g', { 'data-link': name }, parent);
  const right = Math.max(...CHIPS.map((c) => c.x + c.w));
  G.svgEl('rect', { class: 'g-frame', x: TX0 - 6, y: LINE_Y0 - 5, width: right - TX0 + 12, height: lineY(LINES.length - 1) + CHIP_H + 5 - LINE_Y0 + 5, rx: 4, fill: 'none', stroke: 'none' }, g);
  return g;
}
