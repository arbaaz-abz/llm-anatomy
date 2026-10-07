// Loads JSON relative to this file, so it works from any page depth and in both build targets.
export async function loadJSON(relativeToShared) {
  const url = new URL(relativeToShared, import.meta.url);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${relativeToShared}: HTTP ${response.status}`);
  return response.json();
}
