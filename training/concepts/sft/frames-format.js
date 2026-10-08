// Frames 1–3: a base model only continues text, the chat template wraps the turns, the think tags open the answer
// (storyboard §5). Each frame is a pure function of its progress p (0 → 1); a frame starts where the previous one ends.
import * as G from '@shared/glyphs.js';
import { CHIPS, CHIP_H, TX0, NOTE_Y, lineY, seg, ease, arriving, leaving, layer, note, roleLabels, drawChips, defaultState, HANDOFF } from './stage.js';
import { CONTINUATION, THINK_TAGS, CALL_TOKENS, OBS_TOKENS, FIRST_ASSISTANT_TOKEN } from './numbers.js';

// ---- frame 1: the base model continues the prompt ----
const MODEL = Object.freeze({ x: 90, y: 72, w: 140, h: 40 });
const MODEL_MID = MODEL.x + MODEL.w / 2;
const CONT_Y = 154;

const promptChips = (parent) => drawChips(parent, CHIPS.filter((c) => c.flat >= 1 && c.flat <= 6), { followed: false });

// The base model, the two arrows and the continuation row (this body fades out when frame 2 begins).
export function baseBody(svg, p, opacity = 1) {
  const g = layer(svg, opacity);
  G.block(g, { ...MODEL, label: 'base model', state: 'idle' });
  const into = seg(p, 0, 0.25);
  const out = seg(p, 0.25, 0.4);
  if (into > 0) G.flow(g, { from: [MODEL_MID, lineY(0) + CHIP_H + 4], to: [MODEL_MID, MODEL.y - 2], carry: 'token', progress: into });
  if (out > 0) G.flow(g, { from: [MODEL_MID, MODEL.y + MODEL.h + 2], to: [MODEL_MID, CONT_Y - 6], carry: 'token', progress: out });
  note(g, 4, CONT_Y + CHIP_H / 2, 'continues');
  let x = TX0;
  CONTINUATION.forEach((word, i) => {
    const w = G.tokenWidth(word);
    const shown = seg(p, 0.4 + i * 0.04, 0.46 + i * 0.04);
    if (shown > 0) G.token(layer(g, shown), { x, y: CONT_Y, text: word, state: 'idle' });
    x += w + 4;
  });
  note(g, TX0, CONT_Y + CHIP_H + 16, 'illustrative', { opacity: seg(p, 0.9, 1) });
}

export function drawFrame1(svg, p) {
  note(svg, 4, lineY(0) + CHIP_H / 2, 'prompt');
  promptChips(svg);
  baseBody(svg, p, 1);
}

// ---- frame 2: the template tags slide in ----
const SLIDE = 24;
export function drawFrame2(svg, p) {
  if (p < HANDOFF) baseBody(svg, 1, leaving(p));
  const user = ease(seg(p, 0.2, 0.6));
  const assistant = ease(seg(p, 0.4, 0.8));
  const tags = [{ chip: CHIPS[0], t: user, dx: -SLIDE * (1 - user) }, { chip: CHIPS[7], t: assistant, dx: SLIDE * (1 - assistant) }];
  promptChips(svg);
  tags.forEach(({ chip, t, dx }) => drawChips(svg, [chip], { opacityOf: () => t, dxOf: () => dx, followed: false }));
  roleLabels(svg, (line) => (line <= 1 ? seg(p, 0.6, 0.9) : 0));
  note(svg, TX0, NOTE_Y[0], '2 template tokens', { opacity: seg(p, 0.85, 1) });
}

// ---- frame 3: the assistant line fills, with its think tags ----
// Chips type in from <think> to <end>; the tool chips are dim for now; the think tags lift once the line is complete.
const FILL = Object.freeze({ start: 0.04, step: 0.045, fade: 0.08 });
const LIFT = 6;
const firstFlatOfLine = (line) => CHIPS.find((c) => c.line === line).flat;
export const typedAt = (p, flat) => {
  const start = FILL.start + (flat - FIRST_ASSISTANT_TOKEN) * FILL.step;
  return flat < FIRST_ASSISTANT_TOKEN ? 1 : seg(p, start, start + FILL.fade);
};
const isToolChip = (chip) => CALL_TOKENS.includes(chip.flat) || OBS_TOKENS.includes(chip.flat);

export function drawFrame3(svg, p) {
  const lift = Math.sin(Math.PI * seg(p, 0.85, 1)) * LIFT;
  roleLabels(svg, (line) => (line <= 1 ? 1 : typedAt(p, firstFlatOfLine(line))));
  drawChips(svg, CHIPS, {
    opacityOf: (chip) => typedAt(p, chip.flat),
    stateOf: (chip) => (isToolChip(chip) ? 'dim' : defaultState(chip)),
    dyOf: (chip) => (THINK_TAGS.includes(chip.flat) ? -lift : 0),
  });
}
