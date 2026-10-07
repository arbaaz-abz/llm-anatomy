export const TRACK_IDS = ['architecture', 'training', 'serving'];

export function conceptHref({ from, track, slug, target, artifactUrls }) {
  const anchor = slug ? `#${slug}` : '';
  if (from === track) return anchor || '#';
  if (target === 'artifact') {
    const base = artifactUrls[track];
    return base ? `${base}${anchor}` : null;
  }
  const prefix = from === 'hub' ? './' : '../';
  return `${prefix}${track}/${anchor}`;
}

export function hubHref({ from, target, artifactUrls }) {
  if (target === 'artifact') return artifactUrls.hub || null;
  return from === 'hub' ? './' : '../';
}

export function currentTarget(doc = globalThis.document) {
  return doc?.querySelector('meta[name="llm-anatomy-target"]')?.content === 'artifact' ? 'artifact' : 'pages';
}
