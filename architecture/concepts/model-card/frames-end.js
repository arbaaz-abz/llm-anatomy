// model-card frames 9 and 10: the exits (fields that lead out of the block, to the training and serving lessons) and
// the key frame, where sources disagree and each conflict chip splits into its two numbers.
import * as G from '@shared/glyphs.js';
import { COL, PANEL, STAGE, TOKEN_H, layer, note, mark, rule, chipY, lerp, seg } from './stage.js';

// ---- frame 9: four flows from lit chips to labeled exits at the stage edge ----

function exitY(exit, model) {
  const labels = exit.chip === undefined ? model.extras.frame9[exit.extra] : model.chips[exit.chip].label;
  return { y: (exit.chip === undefined ? COL.extraY[exit.extra] : chipY(exit.chip)) + TOKEN_H / 2, x: COL.x + G.tokenWidth(labels) + 6 };
}

export function drawExits(svg, model, scene) {
  if (scene.exits <= 0) return;
  const g = layer(svg, scene.exits);
  const edge = STAGE.w - 10;
  model.exits.forEach((exit) => {
    const { x, y } = exitY(exit, model);
    G.flow(g, { from: [x, y], to: [edge, y], carry: 'token', progress: scene.exits });
    note(g, edge, y - 6, `→ ${exit.title}`, { anchor: 'end' });
  });
}

// ---- frame 10: conflicts ----

const ROW = Object.freeze({ x: 12, y: 44, stride: 78, lowX: 12, highX: 332, merged: 172 });

function token(parent, x, y, text, opacity) {
  if (opacity > 0) G.token(layer(parent, opacity), { x, y, text });
}

function conflictRow(svg, c, index, scene) {
  const y = ROW.y + index * ROW.stride;
  const split = scene[`cs${index}`];
  note(svg, ROW.x, y, c.title, { cls: 'g-text' });
  const row = y + 10;
  const highW = G.tokenWidth(c.high);
  const highX = lerp(ROW.merged, ROW.highX - highW, split);
  token(svg, ROW.lowX, row, c.low, 1);
  if (split > 0) rule(layer(svg, split), ROW.lowX + G.tokenWidth(c.low) + 6, row + TOKEN_H / 2, highX - 6, row + TOKEN_H / 2);
  token(svg, highX, row, c.high, 1);
  if (c.reported) G.token(layer(svg, split), { x: ROW.highX + 12, y: row, text: 'reported' });
  const sources = layer(svg, split);
  note(sources, ROW.lowX, row + TOKEN_H + 14, c.lowSource);
  note(sources, ROW.highX, row + TOKEN_H + 14, c.highSource, { anchor: 'end' });
  if (c.shares) note(sources, ROW.x, row + TOKEN_H + 28, c.shares);
}

export function drawConflicts(svg, model, scene) {
  if (scene.conf <= 0) return;
  const g = layer(svg, scene.conf);
  note(g, ROW.x, 20, 'Sources disagree', { cls: 'g-text' });
  model.conflicts.forEach((c, i) => conflictRow(g, c, i, scene));
  const row = Math.min(model.conflicts.length - 1, Math.floor(scene.walk * model.conflicts.length));
  mark(svg, scene.conf, { x: ROW.x - 4, y: ROW.y + row * ROW.stride - 14, w: ROW.highX + 90 - ROW.x, h: 82 });
}
