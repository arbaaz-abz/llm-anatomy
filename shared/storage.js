// localStorage is a per-viewer convenience here. Failures (private windows, blocked
// site data, quota) are expected, so they degrade to "nothing remembered" by design.

export const LEARNED_KEY = 'llm-anatomy:learned';

export function safeStorage(getStorage = () => globalThis.localStorage) {
  let storage = null;
  try {
    storage = getStorage() ?? null;
  } catch {
    storage = null;
  }
  return {
    get(key, fallback) {
      try {
        const raw = storage?.getItem(key);
        return raw == null ? fallback : JSON.parse(raw);
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        if (!storage) return false;
        storage.setItem(key, JSON.stringify(value));
        return true;
      } catch {
        return false;
      }
    },
  };
}

export function toggleLearned(store, slug) {
  const stored = store.get(LEARNED_KEY, []);
  const current = Array.isArray(stored) ? stored : [];
  const next = current.includes(slug)
    ? current.filter((s) => s !== slug)
    : [...current, slug].sort();
  // Report the truth: if nothing could be saved, nothing was learned.
  return store.set(LEARNED_KEY, next) ? next : current;
}
