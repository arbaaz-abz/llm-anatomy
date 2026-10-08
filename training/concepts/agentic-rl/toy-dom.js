// agentic-rl toy DOM: the group table with selectable tokens, and the rollout timeline. Both paint from the toy state;
// the item groups of the table persist across repaints (so focus survives) and are redrawn each time.
import * as G from '@shared/glyphs.js';
import { ADVANTAGES, DURATIONS, GROUP_SIZE, REWARDS, ROWS } from './numbers.js';
import { minutes } from './format.js';
import { CELL, STAGE, TABLE, TIMELINE, chipFill, linked, note, selectChip, timelineBars, timelineX, timelineY, tokenBox, A_MAX_ABS_STAGE } from './stage.js';
import { schedule, tokenAt } from './toy-view.js';
import { mountStageSelect } from './stage-select.js';

const MIN_IDLE_ROOM = 36; // px an "idle" label needs between a bar's end and the cut
const TABLE_H = STAGE.h;
const TIMELINE_H = 304;
const itemValue = (row, pos) => `${row}:${pos}`;
const parseValue = (value) => value.split(':').map(Number);

const svgOf = (w, h, label) => G.svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': label });

// The group table: 39 selectable chips, the verdict, R and A columns. Returns { node, paint(state), destroy }.
export function mountTable(onSelect, state) {
  const svg = svgOf(STAGE.w, TABLE_H, 'The group of eight answers: click a token to inspect it');
  G.hatchFill(svg);
  const layer = G.svgEl('g', {}, svg);
  const items = ROWS.flatMap((tokens, row) => tokens.map((text, pos) => ({ value: itemValue(row, pos), label: `row ${row + 1}, token ${pos + 1}: ${text}` })));
  const select = mountStageSelect(svg, { items, value: itemValue(state.row, state.pos), onSelect: (value) => onSelect(...parseValue(value)) });
  const paintRows = () => {
    layer.replaceChildren();
    note(layer, TABLE.rX + CELL / 2, TABLE.y - 10, 'R', { anchor: 'middle' });
    note(layer, TABLE.aX + CELL / 2, TABLE.y - 10, 'A', { anchor: 'middle' });
    ROWS.forEach((_, row) => {
      const y = TABLE.y + row * TABLE.stride;
      note(layer, TABLE.x, y + CELL / 2 + 4, String(row + 1));
      G.verdict(layer, { x: TABLE.verdictX, y: y + CELL / 2, ok: REWARDS[row] === 1 });
      G.cell(layer, { x: TABLE.rX, y, size: CELL, v: REWARDS[row], maxAbs: 1, fill: REWARDS[row] === 1 ? 'ok' : 'bad' });
      G.cell(linked(layer, 'a', { x: TABLE.aX, y, w: CELL, h: CELL }), { x: TABLE.aX, y, size: CELL, v: ADVANTAGES[row], maxAbs: A_MAX_ABS_STAGE });
    });
  };
  const paint = (s) => {
    paintRows();
    ROWS.forEach((tokens, row) => tokens.forEach((text, pos) => {
      const node = select.node(itemValue(row, pos));
      node.replaceChildren();
      const t = tokenAt(s, row, pos);
      const box = tokenBox(row, pos);
      G.token(node, { x: box.x, y: box.y, text, fill: chipFill(t.advantage), hatched: t.masked });
      if (row === s.row && pos === s.pos) selectChip(node, box.x, box.y, text);
    }));
    select.sync(itemValue(s.row, s.pos));
  };
  return { node: svg, paint, destroy: () => select.destroy() };
}

// The rollout timeline: one request bar per episode, the cut for a partial iteration, idle and carried labels.
export function mountTimeline() {
  const svg = svgOf(STAGE.w, TIMELINE_H, 'The eight episodes on a time axis in minutes');
  G.hatchFill(svg);
  const layer = G.svgEl('g', {}, svg);
  const paint = (s) => {
    layer.replaceChildren();
    const sched = schedule(s);
    timelineBars(layer);
    const cutX = timelineX(sched.iterationTime);
    const axisY = timelineY(GROUP_SIZE) - 6;
    DURATIONS.forEach((d, row) => {
      if (sched.carried[row]) {
        note(layer, cutX + 4, timelineY(row) + 22, 'finishes under newer weights');
      } else if ((sched.iterationTime - d) * TIMELINE.unit >= MIN_IDLE_ROOM) {
        note(layer, timelineX(d) + 6, timelineY(row) + 8, 'idle');
      }
    });
    if (sched.iterationTime < TIMELINE.minutes) {
      G.svgEl('line', { x1: cutX, y1: timelineY(0) - 8, x2: cutX, y2: axisY, stroke: 'var(--ink-muted)', 'stroke-width': 1 }, layer);
      note(layer, cutX, 14, `cut here: ${GROUP_SIZE - sched.carried.filter(Boolean).length} of ${GROUP_SIZE} done`, { anchor: 'middle' });
    }
    G.selectionMark(layer, { x: TIMELINE.x - 26, y: timelineY(s.row) - 4, w: TIMELINE.minutes * TIMELINE.unit + 34, h: 18 });
    note(layer, TIMELINE.x, axisY + 34, `iteration ${minutes(sched.iterationTime)}`);
  };
  return { node: svg, paint };
}
