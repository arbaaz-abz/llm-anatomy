// The course map, geometry only: one lane per track, lessons placed by how many prerequisites stand before
// them (longest chain, left to right), every "read this first" link drawn as a curve. Pure; map.js renders it.
import { indexConcepts, learningPath } from '../shared/concepts.js';

export const MAP = Object.freeze({ col: 104, row: 26, lanePad: 14, laneGap: 8, labelW: 100, padRight: 20, r: 10 });

const px = (v) => Number(v.toFixed(1));

// slug → length of the longest prerequisite chain before it (0 for a starting point).
export function depthsOf(concepts) {
  const prereqs = new Map(concepts.map((c) => [c.slug, c.prereqs]));
  const depths = new Map();
  const depthOf = (slug) => {
    if (depths.has(slug)) return depths.get(slug);
    const before = prereqs.get(slug) ?? [];
    const d = before.length ? 1 + Math.max(...before.map(depthOf)) : 0;
    depths.set(slug, d);
    return d;
  };
  concepts.forEach((c) => depthOf(c.slug));
  return depths;
}

function groupByDepth(concepts, depths) {
  const groups = new Map();
  concepts.forEach((c) => {
    const d = depths.get(c.slug);
    groups.set(d, [...(groups.get(d) ?? []), c]);
  });
  return groups;
}

function laneStations({ track, concepts, depths, top, href, done }) {
  const groups = groupByDepth(concepts, depths);
  const rows = Math.max(...[...groups.values()].map((g) => g.length));
  const height = rows * MAP.row + 2 * MAP.lanePad;
  const stations = concepts.map((c, i) => {
    const depth = depths.get(c.slug);
    const group = groups.get(depth);
    const groupTop = top + MAP.lanePad + ((rows - group.length) * MAP.row) / 2;
    return {
      slug: c.slug, title: c.title, track: track.id, trackTitle: track.title, number: i + 1, depth,
      x: MAP.labelW + depth * MAP.col + MAP.col / 2, y: px(groupTop + (group.indexOf(c) + 0.5) * MAP.row),
      href: href(track.id, c.slug), learned: done.has(c.slug),
    };
  });
  return { lane: { id: track.id, title: track.title, href: href(track.id, ''), y: top, height }, stations };
}

// A curve from a prerequisite to the lesson that needs it. Same row and more than a column apart: an arc over the
// stations between; otherwise an S-curve with horizontal tangents (straight when the two sit side by side).
function edge(a, b) {
  const x1 = a.x + MAP.r;
  const x2 = b.x - MAP.r;
  const half = (x2 - x1) / 2;
  const d = a.y === b.y && b.depth - a.depth > 1
    ? `M${x1} ${a.y} Q${px(x1 + half)} ${px(a.y - MAP.row * 0.9)} ${x2} ${b.y}`
    : `M${x1} ${a.y} C${px(x1 + half)} ${a.y} ${px(x2 - half)} ${b.y} ${x2} ${b.y}`;
  return { from: a.slug, to: b.slug, track: a.track, cross: a.track !== b.track, d };
}

export function mapLayout(graph, learned = [], href = (track, slug) => `./${track}/${slug ? `#${slug}` : ''}`) {
  const index = indexConcepts(graph.concepts);
  const depths = depthsOf(graph.concepts);
  const done = new Set(Array.isArray(learned) ? learned : []);
  const lanes = [];
  const stations = [];
  let top = 0;
  for (const track of graph.tracks) {
    const concepts = graph.concepts.filter((c) => c.track === track.id);
    if (!concepts.length) continue;
    const placed = laneStations({ track, concepts, depths, top, href, done });
    lanes.push(placed.lane);
    stations.push(...placed.stations);
    top += placed.lane.height + MAP.laneGap;
  }
  const at = new Map(stations.map((s) => [s.slug, s]));
  const edges = graph.concepts.flatMap((c) => c.prereqs.map((p) => edge(at.get(p), at.get(c.slug))));
  const maxDepth = Math.max(0, ...depths.values());
  return { width: MAP.labelW + (maxDepth + 1) * MAP.col + MAP.padRight, height: top - MAP.laneGap, lanes, stations, edges, index };
}

// Everything a lesson builds on, nearest last, as stations (the lesson itself excluded).
export function chainOf(layout, slug) {
  const at = new Map(layout.stations.map((s) => [s.slug, s]));
  return learningPath(layout.index, slug).filter((s) => s !== slug).map((s) => at.get(s));
}

export function captionFor(layout, slug) {
  const station = layout.stations.find((s) => s.slug === slug);
  if (!station) return '';
  const chain = chainOf(layout, slug);
  const who = `${station.title} (${station.trackTitle} ${station.number})`;
  if (!chain.length) return `${who} is a starting point: no prerequisites.`;
  return `${who} builds on ${chain.length} lesson${chain.length === 1 ? '' : 's'}: ${chain.map((s) => s.title).join(', ')}.`;
}
