// Ruled readout tables: a header row, a bold label with a grey sub-label per row, and value cells
// with an optional grey qualifier. spec: { head?, caption?, name?, rows: [{ label, sub?, cells: [{ value, sub?, name? }] }] }.
// readoutTableHtml is pure (and tested); readoutTable returns the DOM element.

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ESCAPES[c]);
const sub = (text) => (text ? `<span class="ro-sub">${esc(text)}</span>` : '');
const named = (name) => (name ? ` data-readout="${esc(name)}"` : '');

const cellHtml = ({ value, sub: qualifier, name }) => `<td><output class="ro-value"${named(name)}>${esc(value)}</output>${sub(qualifier)}</td>`;
const rowHtml = ({ label, sub: note, cells }) => `<tr><th scope="row"><span class="ro-label">${esc(label)}</span>${sub(note)}</th>${cells.map(cellHtml).join('')}</tr>`;

export function readoutTableHtml({ head, caption, name, rows }) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError('readoutTable: rows must be a non-empty array');
  const header = head ? `<thead><tr>${head.map((t) => `<th scope="col">${esc(t)}</th>`).join('')}</tr></thead>` : '';
  const cap = caption ? `<caption>${esc(caption)}</caption>` : '';
  return `<table class="readout-table"${named(name)}>${cap}${header}<tbody>${rows.map(rowHtml).join('')}</tbody></table>`;
}

export function readoutTable(spec) {
  const template = document.createElement('template');
  template.innerHTML = readoutTableHtml(spec);
  return template.content.firstElementChild;
}
