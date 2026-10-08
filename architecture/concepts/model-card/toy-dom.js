// model-card toy DOM helpers: the field table, the cost table and the active-share bars. Layout comes from theme.css
// (.readout-table, .fact-*, .toy-note); nothing here sets an inline style.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { appendRich } from '@shared/lesson-page.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { sharePercent, ZOOM_FRACTION } from './format.js';
import { note } from './stage.js';

const BAR_W = 300; // wide enough that the active slice (30% of the bar) prints its share
const BAR_Y = 6;

const LONG_VALUE = 28; // a longer value wraps inside its cell instead of widening the table
const output = (name, text) => {
  const o = el('output', { className: text.length > LONG_VALUE ? 'fact-wrap' : 'ro-value', textContent: text });
  o.dataset.readout = name;
  return o;
};

function fieldCell(column, row) {
  const parts = [output(`${column.side}-${row.key}`, row.value)];
  if (row.tag) parts.push(' ', el('span', { className: 'fact-reported', textContent: row.tag }));
  if (row.sourceUrl) parts.push(' ', el('a', { className: 'fact-source', href: row.sourceUrl, target: '_blank', rel: 'noopener', textContent: 'source' }));
  if (row.detail) parts.push(el('span', { className: 'ro-sub', textContent: row.detail }));
  const td = el('td', {}, parts);
  td.dataset.side = column.side;
  return td;
}

function fieldRowHeader(row, ctx) {
  const lessons = row.lessons.length ? appendRich(el('span', { className: 'ro-sub fact-lesson' }, ['→ ']), row.lessons.map((s) => `[[${s}]]`).join(', '), ctx) : null;
  return el('th', { scope: 'row' }, [el('span', { className: 'ro-label', textContent: row.label }), el('span', { className: 'ro-sub', textContent: row.gloss }), ...(lessons ? [lessons] : [])]);
}

// The card, field by field: one row per data key, one column per card; a row's gloss and lesson stay visible.
export function fieldsTable(view, ctx) {
  const head = el('thead', {}, [el('tr', {}, [el('th', { scope: 'col', textContent: 'Field' }), ...view.columns.map((c) => el('th', { scope: 'col', textContent: c.name }))])]);
  const body = el('tbody', {}, view.columns[0].rows.map((row, i) => {
    const tr = el('tr', {}, [fieldRowHeader(row, ctx), ...view.columns.map((c) => fieldCell(c, c.rows[i]))]);
    tr.dataset.field = row.key;
    return tr;
  }));
  const table = el('table', { className: 'readout-table' }, [el('caption', { textContent: 'The card, field by field' }), head, body]);
  table.dataset.readout = 'fields';
  return table;
}

// What the fields cost: active share, routed share, cache per token and per conversation, and the GPU share.
export function costsTable(view) {
  const row = (label, sub, name, pick) => ({ label, sub, cells: view.columns.map((c) => ({ ...pick(c), name: `${c.side}-${name}` })) });
  const rows = [
    row('Active share', 'published active ÷ published total', 'active-share', (c) => ({ value: c.costs.activeShare })),
    ...(view.showEmbedding ? [row('Active share, embedding counted', 'when a lab also quotes that figure', 'active-share-embedding', (c) => ({ value: c.costs.activeEmbedding ?? '—' }))] : []),
    row('Routed experts used per token', 'experts per token ÷ routed experts', 'routed-share', (c) => ({ value: c.costs.routedShare })),
    row('Cache per token', 'keys and values kept for each token', 'cache-token', (c) => ({ value: c.costs.cacheToken, sub: c.costs.cacheTokenSub })),
    row('Cache for one conversation', view.conversationLabel, 'cache-conversation', (c) => ({ value: c.costs.cacheConversation, sub: c.tokensText })),
    row(`Share of one ${view.hbmGb} GB GPU`, 'cache for one conversation ÷ GPU memory', 'gpu-share', (c) => ({ value: c.costs.gpuShare, sub: c.costs.gpuSub })),
  ];
  return readoutTable({ head: ['Cost', ...view.columns.map((c) => c.name)], caption: 'What the fields cost', name: 'costs', rows });
}

// One card's active share: one bar over the first 10% of the total, drawn x10, with the headline in words (README lesson 19).
function barSvg(column) {
  const { total, active, formula } = column.bars;
  const svg = G.svgEl('svg', { role: 'group', 'aria-label': `${column.name}: ${formula}` });
  note(svg, 4, 0, `${column.name}: ${column.bars.line}`, { cls: 'g-text' });
  const parts = [{ name: 'active', value: active, hue: 1 }, { name: 'not used (in this 10%)', value: ZOOM_FRACTION * total - active, hue: 2 }];
  G.shareBar(svg, { x: 4, y: BAR_Y + 12, w: BAR_W, parts, label: `${column.name}: active share, the first 10% drawn ×10`, tail: 'none', format: (share) => sharePercent(share * ZOOM_FRACTION) });
  return svg;
}

export function barsBlock(view) {
  const drawn = view.columns.filter((c) => c.bars);
  return drawn.map((c) => barSvg(c));
}

export function fitBars(host) {
  host.querySelectorAll('svg').forEach((svg) => G.fitViewBox(svg, 8));
}

export function notesBlock(view) {
  const notes = view.columns.flatMap((c) => c.notes);
  if (notes.length === 0) return [];
  return [el('h4', { textContent: 'Labs differ' }), el('ul', { className: 'labs-differ' }, notes.map((n) => el('li', {}, [el('strong', { textContent: `${n.label}: ` }), n.text])))];
}
