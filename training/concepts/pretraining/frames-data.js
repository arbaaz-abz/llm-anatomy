// pretraining frames 6–10: the data pipeline (filters, dedup, mixture, rephrasing) and the 2026 token budgets.
// Documents are lettered chips; C appears twice (an exact copy). No keep ratios are published, so none are shown.
import { seg, lerp, arriving, leaving, layer, note } from './stage.js';
import { lossScene } from './frames-loss.js';
import { STAGES, pipeline, stream, doc, factLines, laneLabels, rowY, docX, laneX, laneY, rowEnd } from './pipeline.js';

const DOCS = Object.freeze(['A', 'B', 'C', 'D', 'C', 'F']);
const FAILS_RULES = Object.freeze([1, 5]);
const FAILS_CLASSIFIER = Object.freeze([3]);
const SURVIVORS = Object.freeze([0, 2, 4]); // A, C and its copy
const COPY = 4;
const STATES = Object.freeze({
  6: ['active', 'active', 'active', 'dim', 'dim', 'dim', 'dim'],
  7: ['idle', 'idle', 'idle', 'active', 'dim', 'dim', 'dim'],
  8: ['idle', 'idle', 'idle', 'idle', 'active', 'dim', 'dim'],
  9: ['idle', 'idle', 'idle', 'idle', 'idle', 'active', 'dim'],
});
const KIMI_LINE = 'Kimi K3: rule-based heuristics, classifier quality scoring, dedup, per domain';
const MIX_LINE = 'MiniMax-M2: code, mathematics and STEM significantly upsampled';
const NOTES = Object.freeze({ crawl: 'six crawled documents; C appears twice', rules: 'B, F: dropped by the rules', classifier: 'D: scored low, dropped', dedup: 'the copy of C is removed', lanes: 'chip counts are illustrative, not published rates' });

const at = (k) => rowY(k);
const dropped = (parent, { rules = 1, classifier = 1 } = {}) => {
  FAILS_RULES.forEach((i) => doc(parent, { x: docX(i), y: at(STAGES.rules), label: DOCS[i], dim: rules }));
  FAILS_CLASSIFIER.forEach((i) => doc(parent, { x: docX(i), y: at(STAGES.classifier), label: DOCS[i], dim: classifier }));
};
const droppedNotes = (parent, { rules = 1, classifier = 1, crawl = 1 } = {}) => {
  if (crawl > 0) note(layer(parent, crawl), docX(0), at(STAGES.crawl) - 7, NOTES.crawl);
  if (rules > 0) note(layer(parent, rules), rowEnd(DOCS.length), at(STAGES.rules) + 16, NOTES.rules);
  if (classifier > 0) note(layer(parent, classifier), rowEnd(DOCS.length), at(STAGES.classifier) + 16, NOTES.classifier);
};

// Frame 6's documents: they appear at crawl, move to rules (B and F drop), then on to the classifier (D drops).
function frame6Docs(svg, p) {
  const appear = seg(p, 0.15, 0.3);
  const move1 = seg(p, 0.3, 0.5);
  const move2 = seg(p, 0.6, 0.8);
  const [dim1, dim2] = [seg(p, 0.5, 0.6), seg(p, 0.8, 0.9)];
  const y1 = lerp(at(STAGES.crawl), at(STAGES.rules), move1);
  DOCS.forEach((label, i) => {
    const failsRules = FAILS_RULES.includes(i);
    const y = failsRules ? y1 : lerp(y1, at(STAGES.classifier), move2);
    const dim = failsRules ? dim1 : (FAILS_CLASSIFIER.includes(i) ? dim2 : 0);
    doc(svg, { x: docX(i), y, label, dim, opacity: appear });
  });
  droppedNotes(svg, { rules: dim1, classifier: dim2, crawl: appear });
  if (appear > 0) stream(layer(svg, appear), lerp(y1, at(STAGES.classifier), move2));
}

// Frame 6: rules drop the obvious junk; the quality classifier scores the rest.
export function drawFrame6(svg, p) {
  const out = leaving(p);
  if (out > 0) lossScene(layer(svg, out));
  const inn = arriving(p);
  pipeline(layer(svg, inn), STATES[6]);
  factLines(layer(svg, inn), [KIMI_LINE]);
  frame6Docs(svg, p);
}

// Frame 7: the survivors reach dedup; C and its exact copy meet there, and the copy fades out.
export function drawFrame7(svg, p) {
  const out = leaving(p);
  pipeline(svg, STATES[7], { from: STATES[6], t: arriving(p) });
  factLines(svg, [KIMI_LINE]);
  if (out > 0) {
    const old = layer(svg, out);
    dropped(old);
    droppedNotes(old);
  }
  const move = seg(p, 0.15, 0.45);
  const y = lerp(at(STAGES.classifier), at(STAGES.dedup), move);
  const meet = seg(p, 0.5, 0.7);
  const gone = seg(p, 0.7, 0.9);
  SURVIVORS.forEach((i) => {
    const copy = i === COPY;
    doc(svg, { x: copy ? lerp(docX(i), docX(i - 1), meet) : docX(i), y, label: DOCS[i], opacity: copy ? 1 - gone : 1 });
  });
  if (gone > 0) note(layer(svg, gone), rowEnd(DOCS.length), at(STAGES.dedup) + 16, NOTES.dedup);
  stream(svg, y);
}

// The mixture's lanes (a new chip slides in along its lane): A starts web, C starts knowledge; code and math take twice as many chips.
const LANE_DOCS = Object.freeze([['A', 'G', 'H'], ['I', 'J', 'K', 'L', 'M', 'N'], ['O', 'P', 'Q', 'R', 'S', 'T'], ['C', 'U', 'V']]);
const FAST = Object.freeze([false, true, true, false]);
const laneChipT = (lane, slot) => (FAST[lane] ? 0.35 + slot * 0.1 : 0.35 + slot * 0.2);
const SLIDE_IN = 24; // a lane's new chip slides in along its lane (never across the lane labels)

function lanes(parent, p) {
  LANE_DOCS.forEach((labels, lane) => labels.forEach((label, slot) => {
    const first = slot === 0 && (lane === 0 || lane === 3);
    if (first) return; // A and C travel from dedup (drawn by the frame)
    const t = seg(p, laneChipT(lane, slot), laneChipT(lane, slot) + 0.08);
    if (t > 0) doc(parent, { x: lerp(laneX(slot) - SLIDE_IN, laneX(slot), t), y: laneY(lane), label, opacity: t });
  }));
}

function travellers(parent, move) {
  doc(parent, { x: lerp(docX(0), laneX(0), move), y: lerp(at(STAGES.dedup), laneY(0), move), label: 'A' });
  doc(parent, { x: lerp(docX(2), laneX(0), move), y: lerp(at(STAGES.dedup), laneY(3), move), label: 'C' });
}

// Frame 8: the survivors are sorted into domain lanes; the code and math lanes fill faster.
export function drawFrame8(svg, p, text) {
  const out = leaving(p);
  const inn = arriving(p);
  pipeline(svg, STATES[8], { from: STATES[7], t: inn });
  if (out > 0) {
    const old = layer(svg, out);
    factLines(old, [KIMI_LINE]);
    note(old, rowEnd(DOCS.length), at(STAGES.dedup) + 16, NOTES.dedup);
  }
  factLines(layer(svg, inn), [MIX_LINE, text.glm]);
  const move = seg(p, 0.15, 0.35);
  travellers(svg, move);
  const labelsIn = seg(p, 0.15, 0.3);
  if (labelsIn > 0) laneLabels(layer(svg, labelsIn));
  lanes(svg, p);
  const fineprint = seg(p, 0.85, 1);
  if (fineprint > 0) note(layer(svg, fineprint), laneX(0), laneY(3) + 40, NOTES.lanes);
  stream(svg, lerp(at(STAGES.dedup), at(STAGES.mixture), move));
}

// Frame 8's end state, as frame 9 starts from it (everything but the knowledge lane's C, which frame 9 moves on).
export function mixScene(parent, text) {
  factLines(parent, [MIX_LINE, text.glm]);
  laneLabels(parent);
  lanes(parent, 1);
  doc(parent, { x: laneX(0), y: laneY(0), label: 'A' });
  note(parent, laneX(0), laneY(3) + 40, NOTES.lanes);
}

export { STATES, NOTES, at };
