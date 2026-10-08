// Course conventions in the 14 Training storyboards (Plan 3 Review Focus 5; README lessons 23, 28; X-1, X-3).
// Each failure reads "<file>:<line>: <rule>".
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { extractRefs, DATA_FILES } from '../scripts/check-data-refs.js';

const ROOT = new URL('../', import.meta.url);
const readJson = async (rel) => JSON.parse(await readFile(new URL(rel, ROOT), 'utf8'));

const graph = await readJson('shared/concepts.json');
const SLUGS = graph.concepts.filter((c) => c.track === 'training').map((c) => c.slug);
const datasets = Object.fromEntries(await Promise.all(DATA_FILES.map(async (f) => [f, await readJson(`data/${f}.json`)])));
const idsByFile = Object.fromEntries(DATA_FILES.map((f) => [f, new Set(datasets[f].entries.map((e) => e.id))]));
const STORYBOARDS = await Promise.all(SLUGS.map(async (slug) => {
  const file = `${slug}.md`;
  const text = await readFile(new URL(`docs/storyboards/${file}`, ROOT), 'utf8');
  return { file, lines: text.split('\n'), refs: extractRefs(text, idsByFile) };
}));

/** Every cited data key, with the text of the line that cites it. */
const citations = () => STORYBOARDS.flatMap(({ file, lines, refs }) => refs
  .filter((r) => r.key)
  .map((r) => ({ where: `${file}:${r.line}`, id: r.id, key: r.key, line: lines[r.line - 1] })));

/** Every line of every Training storyboard. */
const allLines = () => STORYBOARDS.flatMap(({ file, lines }) => lines.map((text, i) => ({ where: `${file}:${i + 1}`, text })));

test('the 14 Training storyboards are read', () => {
  assert.equal(SLUGS.length, 14);
});

// (a) README lesson 23: peaks are dense; a sparse peak is never cited.
const sparsePeakProblems = (cites) => cites
  .filter((c) => /_sparse_tflops$/.test(c.key))
  .map((c) => `${c.where}: sparse peak \`${c.id}.${c.key}\` cited (cite the dense key)`);

// (b) README lessons 23, 28: a link speed names its direction convention.
const LINK_DIRECTION_WORDS = /both directions|each way|direction not given/i;
const isLinkKey = (c) => /^nvlink/.test(c.key) || /^network-/.test(c.id) || /_gb_s/.test(c.key);
const linkDirectionProblems = (cites) => cites
  .filter((c) => isLinkKey(c) && !/(_each_way|_tbps)$/.test(c.key) && !LINK_DIRECTION_WORDS.test(c.line))
  .map((c) => `${c.where}: link key \`${c.id}.${c.key}\` cited without "both directions" / "each way" / "direction not given"`);

// (c) README lesson 23: usable vs nominal HBM carries its label word.
const hbmLabelProblems = (cites) => cites.flatMap((c) => {
  if (c.key === 'hbm_usable_gb' && !/usable/i.test(c.line)) return [`${c.where}: \`${c.id}.hbm_usable_gb\` cited on a line without "usable"`];
  if (c.key === 'hbm_gb' && /usable/i.test(c.line) && !/nominal/i.test(c.line)) return [`${c.where}: nominal \`${c.id}.hbm_gb\` on a "usable" line without "nominal"`];
  return [];
});

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
/** Printed forms of an entry name: as stored, without a trailing "(…)", without a leading vendor word. */
const nameForms = (name) => {
  const bare = name.replace(/\s*\([^)]*\)\s*$/, '').trim();
  const noVendor = bare.replace(/^(NVIDIA|AMD|Google|Meta's)\s+/, '');
  return [...new Set([name, bare, noVendor])].filter((n) => n.length >= 4);
};

// (d) X-1: "<name> (YYYY)" only when the entry's release_date is confirmed and in that year.
const yearProblems = (lines, entries) => lines.flatMap(({ where, text }) => entries.flatMap((entry) => {
  const date = entry.facts?.release_date;
  const confirmedYear = date?.confidence === 'confirmed' ? String(date.value).slice(0, 4) : null;
  return nameForms(entry.name).flatMap((form) => [...text.matchAll(new RegExp(`${escapeRe(form)}\\s*\\((?:\\w+ )?(20\\d\\d)\\b`, 'g'))]
    .filter((m) => m[1] !== confirmedYear)
    .map((m) => `${where}: "${m[0]}" prints a year without a confirmed ${entry.id}.release_date in ${m[1]} (X-1)`));
}));

// (e) X-3: an "N×" ratio prints at most 3 significant figures, no trailing zero after the point.
// A ratio is a number written straight against "×" and not followed by an operand ("8 × 16" is a product).
const RATIO = /(\d[\d,]*(?:\.\d+)?)×(?!\s*[\d(])/g;
const significantFigures = (text) => {
  const digits = text.replace(/,/g, '');
  if (digits.includes('.')) return digits.replace('.', '').replace(/^0+/, '').length;
  return digits.replace(/^0+/, '').replace(/0+$/, '').length || 1;
};
const ratioProblems = (lines) => lines.flatMap(({ where, text }) => [...text.matchAll(RATIO)]
  .filter((m) => significantFigures(m[1]) > 3 || /\.\d*0$/.test(m[1]))
  .map((m) => `${where}: ratio "${m[0]}" is not 3 s.f. with trailing zeros dropped (X-3)`));

test('(a) no sparse peak is cited', () => assert.deepEqual(sparsePeakProblems(citations()), []));
test('(b) every link key names its direction convention', () => assert.deepEqual(linkDirectionProblems(citations()), []));
test('(c) usable and nominal HBM keys carry their label word', () => assert.deepEqual(hbmLabelProblems(citations()), []));
test('(d) X-1: a printed year needs a confirmed release_date in that year', () => {
  const entries = [...datasets.models.entries, ...datasets.hardware.entries];
  assert.deepEqual(yearProblems(allLines(), entries), []);
});
test('(e) X-3: ratios print 3 significant figures, trailing zeros dropped', () => assert.deepEqual(ratioProblems(allLines()), []));

// The rules themselves, on planted lines (so a scan that finds nothing is not a broken scan).
test('the rules catch planted violations', () => {
  const cite = (id, key, line) => ({ where: 'x.md:1', id, key, line });
  assert.equal(sparsePeakProblems([cite('h100', 'bf16_sparse_tflops', '')]).length, 1);
  assert.equal(linkDirectionProblems([cite('h100', 'nvlink_gb_s', '900 GB/s')]).length, 1);
  assert.equal(linkDirectionProblems([cite('h100', 'nvlink_gb_s', '900 GB/s both directions')]).length, 0);
  assert.equal(linkDirectionProblems([cite('h100', 'nvlink_gb_s_each_way', '450')]).length, 0);
  assert.equal(linkDirectionProblems([cite('network-400g', 'gbps', '400 Gb/s')]).length, 1);
  assert.equal(hbmLabelProblems([cite('b200', 'hbm_usable_gb', '180 GB')]).length, 1);
  assert.equal(hbmLabelProblems([cite('b200', 'hbm_gb', '180 GB usable')]).length, 1);
  assert.equal(hbmLabelProblems([cite('b200', 'hbm_gb', '180 GB usable (192 nominal)')]).length, 0);
  const kimi = { id: 'kimi-k3', name: 'Kimi K3', facts: { release_date: { value: '2026-07-27', confidence: 'reported' } } };
  const glm = { id: 'glm-5', name: 'GLM-5', facts: { release_date: { value: '2026-02', confidence: 'confirmed' } } };
  const h100 = { id: 'h100', name: 'NVIDIA H100 SXM', facts: {} };
  const line = (text) => [{ where: 'x.md:1', text }];
  assert.equal(yearProblems(line('Kimi K3 (2026) trains'), [kimi]).length, 1);
  assert.equal(yearProblems(line('GLM-5 (2026) trains'), [glm]).length, 0);
  assert.equal(yearProblems(line('GLM-5 (2025) trains'), [glm]).length, 1);
  assert.equal(yearProblems(line('H100 SXM (2022): 80 GB'), [h100]).length, 1);
  assert.equal(yearProblems(line('Llama (2024 paper)'), [{ id: 'l', name: 'Llama', facts: {} }]).length, 1);
  assert.deepEqual(['12×', '1,180×', '73.9×', '2×', '0.27×'].flatMap((t) => ratioProblems(line(t))), []);
  assert.equal(ratioProblems(line('(1.069×)')).length, 1);
  assert.equal(ratioProblems(line('(2.00×)')).length, 1);
  assert.equal(ratioProblems(line('5.0× faster')).length, 1);
  assert.equal(ratioProblems(line('16,384 × 8 GPUs; 4096 × 4096')).length, 0);
});
