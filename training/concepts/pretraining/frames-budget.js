// pretraining frames 9–10: rephrased data in, templated text out; then the 2026 pretraining token budgets (key frame).
import { DIM, seg, lerp, arriving, leaving, layer, note } from './stage.js';
import { STAGES, pipeline, stream, doc, factLines, rowY, docX, laneX, laneY, rowEnd } from './pipeline.js';
import { STATES, mixScene } from './frames-data.js';

const DS_LINE = 'DeepSeek-V4 filters out batched auto-generated and templated text';
const REPHRASE_LINE = 'Kimi K3: knowledge and math rephrased by another model, with fidelity checks';
const NOTES = Object.freeze({ rephrase: 'rewordings of C, checked, added back', templated: 'T: templated, dropped' });
const REPHRASED = Object.freeze(['C′', 'C″']);

const rephraseNote = (parent) => note(parent, rowEnd(2), rowY(STAGES.rephrase) + 16, NOTES.rephrase);
const templatedNote = (parent) => note(parent, rowEnd(1), rowY(STAGES.classifier) + 16, NOTES.templated);

// Frame 9's moving parts at their sub-phases (all 1 = its end state).
function rephrasing(parent, { move = 1, split = 1, noteT = 1, tIn = 1, tDim = 1 } = {}) {
  const y = lerp(laneY(3), rowY(STAGES.rephrase), move);
  doc(parent, { x: lerp(laneX(0), docX(0), move), y, label: 'C', opacity: 1 - split });
  doc(parent, { x: docX(0), y, label: REPHRASED[0], opacity: split });
  doc(parent, { x: lerp(docX(0), docX(1), split), y, label: REPHRASED[1], opacity: split });
  if (noteT > 0) rephraseNote(layer(parent, noteT));
  doc(parent, { x: docX(0), y: rowY(STAGES.classifier), label: 'T', dim: tDim, opacity: tIn });
  if (tDim > 0) templatedNote(layer(parent, tDim));
  stream(parent, lerp(rowY(STAGES.mixture), rowY(STAGES.rephrase), move));
}

// Frame 9: a knowledge document is reworded twice and added back; a templated one is filtered out at the classifier.
export function drawFrame9(svg, p, text) {
  const out = leaving(p);
  const inn = arriving(p);
  pipeline(svg, STATES[9], { from: STATES[8], t: inn });
  if (out > 0) mixScene(layer(svg, out), text);
  factLines(layer(svg, inn), [DS_LINE, REPHRASE_LINE]);
  rephrasing(svg, { move: seg(p, 0.15, 0.35), split: seg(p, 0.4, 0.65), noteT: seg(p, 0.6, 0.75), tIn: seg(p, 0.6, 0.7), tDim: seg(p, 0.75, 0.85) });
}

const TABLE = Object.freeze({ name: 150, value: 430, title: 44, y0: 76, pitch: 26 });
const rowT = (r) => 0.15 + r * 0.09;

// Frame 10 (key frame): 2026 open frontier models' pretraining tokens, typed in top to bottom; Llama 3.1 405B dim below.
export function drawFrame10(svg, p, text) {
  const out = leaving(p);
  if (out > 0) {
    const old = layer(svg, out);
    pipeline(old, STATES[9]);
    factLines(old, [DS_LINE, REPHRASE_LINE]);
    rephrasing(old);
  }
  note(layer(svg, arriving(p)), TABLE.name, TABLE.title, 'pretraining tokens', { cls: '' });
  text.budget.forEach((row, r) => {
    const t = seg(p, rowT(r), rowT(r) + 0.07);
    if (t <= 0) return;
    const g = layer(svg, t * (row.dim ? DIM : 1));
    const y = TABLE.y0 + r * TABLE.pitch;
    note(g, TABLE.name, y, row.name, { cls: '' });
    note(g, TABLE.value, y, row.tokens, { cls: '', anchor: 'end' });
  });
  const rangeT = seg(p, 0.9, 1);
  if (rangeT > 0) note(layer(svg, rangeT), TABLE.name, TABLE.y0 + text.budget.length * TABLE.pitch + 12, text.range);
}
