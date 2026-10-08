import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readoutTableHtml } from '../shared/ui/readout-table.js';

const rows = [
  { label: 'Total', sub: 'every parameter', cells: [{ value: '1,576', name: 'total' }] },
  { label: 'Active', cells: [{ value: '1,448', sub: '91.9% of total', name: 'active' }] },
];

test('a header row, then one row per entry with a row-header label and value cells', () => {
  const html = readoutTableHtml({ head: ['Count', 'Parameters'], rows });
  assert.match(html, /^<table class="readout-table">/);
  assert.equal((html.match(/<th scope="col"/g) ?? []).length, 2);
  assert.equal((html.match(/<th scope="row"/g) ?? []).length, 2);
  assert.equal((html.match(/<tbody><tr/g) ?? []).length, 1);
});

test('labels carry a grey sub-label, values carry a qualifier, both only when given', () => {
  const html = readoutTableHtml({ rows });
  assert.match(html, /<th scope="row"><span class="ro-label">Total<\/span><span class="ro-sub">every parameter<\/span><\/th>/);
  assert.match(html, /<th scope="row"><span class="ro-label">Active<\/span><\/th>/);
  assert.match(html, /<output class="ro-value" data-readout="active">1,448<\/output><span class="ro-sub">91.9% of total<\/span>/);
  assert.match(html, /<output class="ro-value" data-readout="total">1,576<\/output><\/td>/);
});

test('a caption and a data-readout name on the table are optional', () => {
  const html = readoutTableHtml({ caption: 'Per block', name: 'per-block', rows });
  assert.match(html, /<table class="readout-table" data-readout="per-block"><caption>Per block<\/caption>/);
});

test('text is escaped, so a value can never inject markup', () => {
  const html = readoutTableHtml({ rows: [{ label: '<b>x</b>', cells: [{ value: 'a & b', sub: '"q"' }] }] });
  assert.ok(!html.includes('<b>'));
  assert.match(html, /&lt;b&gt;x&lt;\/b&gt;/);
  assert.match(html, /a &amp; b/);
  assert.match(html, /&quot;q&quot;/);
});

test('an empty table is an error, not silent markup', () => {
  assert.throws(() => readoutTableHtml({ rows: [] }), RangeError);
});
