// Frames 4–7: the loss mask (storyboard §5). The followed token is the final "56". Hatch = masked (README lesson 24).
// Each frame starts where the previous one ends: frame 4's hatch sweeps the prompt, frame 5 adds the tool's reply,
// frame 6 counts what is left, frame 7 masks the kept mistake.
import * as G from '@shared/glyphs.js';
import { maskSummary } from '@math/sft.js';
import { CHIPS, CHIP_H, NOTE_Y, lineY, seg, layer, note, roleLabels, drawChips, defaultState, linkedTranscript } from './stage.js';
import { SEGMENTS, CALL_TOKENS, OBS_TOKENS, ERROR_TOKENS } from './numbers.js';
import { shareText } from './format.js';

const PROMPT_TOKENS = Object.freeze(CHIPS.filter((c) => c.kind === 'template' || c.kind === 'user').map((c) => c.flat)); // 0–7
const rightEnd = (line) => Math.max(...CHIPS.filter((c) => c.line === line).map((c) => c.x + c.w));

// The transcript with a per-chip "is it hatched yet" rule, inside the math panel's `m` link.
function transcript(svg, hatchedOf, extra = {}) {
  const group = linkedTranscript(svg);
  roleLabels(group);
  drawChips(group, CHIPS, { hatchedOf, ...extra });
}

// ---- frame 4: the prompt and the template tags are context only ----
const SWEEP4 = Object.freeze({ start: 0.1, step: 0.07 });
export function drawFrame4(svg, p) {
  const hatchedOf = (chip) => PROMPT_TOKENS.includes(chip.flat) && p >= SWEEP4.start + chip.flat * SWEEP4.step;
  const toolDim = (chip) => (p < 0.1 && (CALL_TOKENS.includes(chip.flat) || OBS_TOKENS.includes(chip.flat)) ? 'dim' : defaultState(chip));
  transcript(svg, hatchedOf, { stateOf: toolDim });
  const label = seg(p, 0.6, 0.75);
  note(svg, rightEnd(0) + 10, lineY(0) + CHIP_H / 2, 'context only', { opacity: label });
  note(svg, rightEnd(1) + 10, lineY(1) + CHIP_H / 2, 'context only', { opacity: label });
  note(svg, 76, NOTE_Y[0], 'opening role tags are masked; the assistant\'s own <end> is trained', { opacity: seg(p, 0.75, 0.9) });
  const s = maskSummary(SEGMENTS, { maskObservation: false });
  note(svg, 76, NOTE_Y[1], `masked ${s.byKind.template.masked + s.byKind.user.masked} (template ${s.byKind.template.masked}, user ${s.byKind.user.masked})`, { opacity: seg(p, 0.85, 1) });
}

// ---- frame 5: the tool call is trained, the tool's reply is masked ----
const SWEEP5 = Object.freeze({ start: 0.6, step: 0.08 });
const TOOL = Object.freeze({ x: 400, y: lineY(4) - 4, w: 90, h: CHIP_H + 8 });
export function drawFrame5(svg, p) {
  const hatchedOf = (chip) => PROMPT_TOKENS.includes(chip.flat) || (OBS_TOKENS.includes(chip.flat) && p >= SWEEP5.start + (chip.flat - OBS_TOKENS[0]) * SWEEP5.step);
  transcript(svg, hatchedOf);
  const g = layer(svg);
  G.block(g, { ...TOOL, label: 'tool', state: 'idle' });
  const callEnd = CHIPS[CALL_TOKENS[2]];
  const out = seg(p, 0.05, 0.3);
  const back = seg(p, 0.35, 0.6);
  const fromX = callEnd.x + callEnd.w / 2;
  if (out > 0) G.flow(g, { from: [fromX, callEnd.y + CHIP_H + 2], to: [fromX, TOOL.y - 2], carry: 'token', progress: out });
  if (back > 0) G.flow(g, { from: [TOOL.x - 2, lineY(4) + CHIP_H / 2], to: [rightEnd(4) + 4, lineY(4) + CHIP_H / 2], carry: 'token', progress: back });
  const s = maskSummary(SEGMENTS);
  note(svg, 76, NOTE_Y[0], 'the tool\'s reply (an observation: see Agentic RL): masked', { opacity: seg(p, 0.7, 0.85) });
  note(svg, 76, NOTE_Y[1], `call ${CALL_TOKENS.length} tokens trained · reply ${OBS_TOKENS.length} masked · masked so far ${PROMPT_TOKENS.length} + ${OBS_TOKENS.length} = ${s.masked}`, { opacity: seg(p, 0.8, 0.95) });
}

// ---- frame 6: the loss counts the trained tokens only ----
const DEFAULT_MASKED = Object.freeze([...PROMPT_TOKENS, ...OBS_TOKENS]);
const isMaskedDefault = (chip) => DEFAULT_MASKED.includes(chip.flat);

export function drawFrame6(svg, p) {
  transcript(svg, isMaskedDefault);
  const s = maskSummary(SEGMENTS);
  const count = Math.floor(s.trained * seg(p, 0.1, 0.8) + 1e-9);
  note(svg, 76, NOTE_Y[0], `trained ${count} of ${s.total}`);
  note(svg, 76, NOTE_Y[1], 'same loss as pretraining, on these tokens only', { opacity: seg(p, 0.8, 0.95) });
  note(svg, 76, NOTE_Y[2], `trained ${s.trained} · masked ${s.masked} · ${s.trained} ÷ ${s.total} = ${shareText(s.trained, s.total)}`, { opacity: seg(p, 0.85, 1) });
}

// ---- frame 7: a kept mistake is masked ----
const SWEEP7 = Object.freeze({ start: 0.15, step: 0.1 });
export function drawFrame7(svg, p) {
  const errorHatched = (chip) => ERROR_TOKENS.includes(chip.flat) && p >= SWEEP7.start + (chip.flat - ERROR_TOKENS[0]) * SWEEP7.step;
  transcript(svg, (chip) => isMaskedDefault(chip) || errorHatched(chip));
  note(svg, rightEnd(2) + 10, lineY(2) + CHIP_H / 2, 'mistake kept in the text', { opacity: seg(p, 0, 0.15) });
  const before = maskSummary(SEGMENTS);
  const after = maskSummary(SEGMENTS, { maskError: true });
  const hatchedNow = ERROR_TOKENS.filter((flat) => p >= SWEEP7.start + (flat - ERROR_TOKENS[0]) * SWEEP7.step).length;
  note(svg, 76, NOTE_Y[0], `trained ${before.trained - hatchedNow} of ${before.total}`);
  note(svg, 76, NOTE_Y[1], `trained ${before.trained} → ${after.trained} · ${shareText(after.trained, after.total)} of the tokens · masked ${after.masked}`, { opacity: seg(p, 0.7, 0.85) });
}
