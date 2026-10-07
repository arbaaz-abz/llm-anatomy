import { SLUG_PATTERN } from './concepts.js';

export function parseRoute(hash, knownSlugs) {
  const raw = String(hash ?? '').replace(/^#/, '');
  if (!raw) return { slug: null };
  let decoded;
  try {
    decoded = decodeURIComponent(raw);
  } catch {
    return { slug: null, invalid: raw };
  }
  return SLUG_PATTERN.test(decoded) && knownSlugs.has(decoded)
    ? { slug: decoded }
    : { slug: null, invalid: decoded };
}

export function startRouter({ knownSlugs, onRoute, win = globalThis }) {
  const handle = () => onRoute(parseRoute(win.location.hash, knownSlugs));
  win.addEventListener('hashchange', handle);
  handle();
  return () => win.removeEventListener('hashchange', handle);
}
