// Where the reader is: lessons marked learned (shared/storage.js) and the one to open next. Pure.
import { bySection } from '../shared/concepts.js';

// Lessons in the order the course lists them: track by track, section by section.
export const courseOrder = (graph) => graph.tracks.flatMap((t) => bySection(graph.concepts, t.id).flatMap((g) => g.concepts));

// The first unlearned lesson whose prerequisites are all learned, else the first unlearned lesson at all;
// null before anything is learned (the page says "Start with Architecture" then) or once everything is.
export function nextLesson(graph, learned) {
  const done = new Set(Array.isArray(learned) ? learned : []);
  if (done.size === 0) return null;
  const order = courseOrder(graph);
  const unlearned = order.filter((c) => !done.has(c.slug));
  const ready = unlearned.find((c) => c.prereqs.every((p) => done.has(p)));
  const pick = ready ?? unlearned[0] ?? null;
  return pick ? { slug: pick.slug, title: pick.title, track: pick.track } : null;
}

export const learnedCount = (graph, learned) => {
  const known = new Set(graph.concepts.map((c) => c.slug));
  return (Array.isArray(learned) ? learned : []).filter((slug) => known.has(slug)).length;
};
