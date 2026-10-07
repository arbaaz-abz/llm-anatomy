// Pure helpers over the concept list in concepts.json. Used by the track apps and the hub map.

export const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export function indexConcepts(concepts) {
  const index = new Map();
  for (const concept of concepts) {
    if (typeof concept.slug !== 'string' || !SLUG_PATTERN.test(concept.slug)) throw new Error(`invalid slug "${concept.slug}"`);
    if (index.has(concept.slug)) throw new Error(`duplicate slug "${concept.slug}"`);
    index.set(concept.slug, concept);
  }
  for (const concept of concepts) {
    const missing = concept.prereqs.filter((p) => !index.has(p));
    if (missing.length) throw new Error(`${concept.slug}: unknown prereq ${missing.join(', ')}`);
  }
  return index;
}

export function topoOrder(concepts) {
  const prereqsOf = new Map(concepts.map((c) => [c.slug, c.prereqs]));
  const state = new Map(); // slug -> 'visiting' | 'done'
  const order = [];
  const visit = (slug) => {
    if (state.get(slug) === 'done') return;
    if (state.get(slug) === 'visiting') throw new Error(`concept graph has a cycle at "${slug}"`);
    state.set(slug, 'visiting');
    (prereqsOf.get(slug) ?? []).forEach(visit);
    state.set(slug, 'done');
    order.push(slug);
  };
  concepts.forEach((c) => visit(c.slug));
  return order;
}

export function learningPath(index, slug) {
  if (!index.has(slug)) throw new Error(`learningPath: unknown concept "${slug}"`);
  const needed = new Set();
  const collect = (s) => {
    if (needed.has(s)) return;
    needed.add(s);
    index.get(s).prereqs.forEach(collect);
  };
  collect(slug);
  return topoOrder([...index.values()]).filter((s) => needed.has(s));
}

export function bySection(concepts, trackId) {
  const groups = [];
  for (const concept of concepts.filter((c) => c.track === trackId)) {
    const group = groups.find((g) => g.section === concept.section);
    if (group) group.concepts.push(concept);
    else groups.push({ section: concept.section, concepts: [concept] });
  }
  return groups;
}
