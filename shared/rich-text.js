// Inline markup for lesson prose (pure): [[slug]] links a lesson, `x` is code. Everything else is plain text;
// nothing is ever parsed as HTML.
const TOKEN = /\[\[([a-z0-9]+(?:-[a-z0-9]+)*)\]\]|`([^`\n]+)`/g;

export function parseRichText(text) {
  if (typeof text !== 'string') throw new TypeError(`parseRichText: text must be a string, got ${typeof text}`);
  const segments = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    if (m.index > last) segments.push({ type: 'text', text: text.slice(last, m.index) });
    segments.push(m[1] ? { type: 'lesson', slug: m[1] } : { type: 'code', text: m[2] });
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ type: 'text', text: text.slice(last) });
  return segments;
}
