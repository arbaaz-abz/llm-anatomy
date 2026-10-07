import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractRefs, findMissing, formatReport, splitIdKey } from '../scripts/check-data-refs.js';

const ids = {
  models: new Set(['llama-3.1-70b', 'kimi-k2.5', 'glm-5']),
  hardware: new Set(['h100', 'b200', 'b300']),
  serving: new Set(['vllm']),
  papers: new Set(['chinchilla-refit-2024']),
};
const pairs = (refs) => refs.map((r) => `${r.file}:${r.id}.${r.key}`);

test('splitIdKey prefers known dotted ids, then the first valid split', () => {
  assert.deepEqual(splitIdKey('llama-3.1-70b.layers', ids.models), { id: 'llama-3.1-70b', key: 'layers' });
  assert.deepEqual(splitIdKey('kimi-k2.5.weight_format'), { id: 'kimi-k2.5', key: 'weight_format' });
  assert.deepEqual(splitIdKey('qwen3.8.*'), { id: 'qwen3.8', key: '*' });
  assert.equal(splitIdKey('scale.js'), null);
  assert.equal(splitIdKey('no-dot-here'), null);
});

test('reads file.json/id.key with .key continuations and values', () => {
  const md = '| a | `hardware.json/h100.hbm_gb`, `.hbm_tbps` = 3.35; `b200.hbm_usable_gb = 180` | 03 |';
  assert.deepEqual(pairs(extractRefs(md, ids)), ['hardware:h100.hbm_gb', 'hardware:h100.hbm_tbps', 'hardware:b200.hbm_usable_gb']);
});

test('reads file.id.key, ellipsis keys and wildcards', () => {
  const md = '| a | `models.glm-5.pretrain_tokens` · `papers.chinchilla-refit-2024` · `hardware.json/h100.nvlink_gb_s_each_way` = 450, `b300.…` = 900 · `serving.json/vllm.*` |';
  assert.deepEqual(pairs(extractRefs(md, ids)), [
    'models:glm-5.pretrain_tokens', 'papers:chinchilla-refit-2024.',
    'hardware:h100.nvlink_gb_s_each_way', 'hardware:b300.nvlink_gb_s_each_way', 'serving:vllm.*',
  ]);
});

test('bare id.key uses the file in context or the file that knows the id', () => {
  const md = '| a | `models.json` `glm-5.rl_algorithm`, `new-model.rl_group_size` = 32 |\n| b | `b200.hbm_gb` |';
  assert.deepEqual(pairs(extractRefs(md, ids)), ['models:glm-5.rl_algorithm', 'models:new-model.rl_group_size', 'hardware:b200.hbm_gb']);
});

test('bare key = value spans attach to the current id; context resets per row', () => {
  const md = '- `hardware.json/meta-llama3-cluster` (new entry): `gpus_per_rack 16`,\n  `racks_per_pod = 192`\n- `max_num_seqs = 4`';
  assert.deepEqual(pairs(extractRefs(md, ids)), [
    'hardware:meta-llama3-cluster.', 'hardware:meta-llama3-cluster.gpus_per_rack', 'hardware:meta-llama3-cluster.racks_per_pod',
  ]);
});

test('ignores code spans that are not data refs, fenced blocks and id wildcards', () => {
  const md = 'Uses `math/scale.js`, `core.softmax`, `config.json`, `concepts.json`.\n```\n`models.glm-5.x`\n```\n`hardware.json/*_dense_tflops`';
  assert.deepEqual(extractRefs(md, ids), []);
});

test('expands brace lists and handles spans that wrap lines', () => {
  const md = 'See `hardware.json/{h100,b200}` and `models.glm-5.context\n_stages`';
  const refs = extractRefs(md, ids);
  assert.deepEqual(pairs(refs).slice(0, 2), ['hardware:h100.', 'hardware:b200.']);
  assert.equal(refs[0].line, 1);
});

test('findMissing reports absent entries and keys; formatReport groups by file', () => {
  const datasets = { models: { entries: [{ id: 'glm-5', facts: { layers: {} } }] } };
  const refs = [
    { file: 'models', id: 'glm-5', key: 'layers', line: 1 },
    { file: 'models', id: 'glm-5', key: '*', line: 1 },
    { file: 'models', id: 'glm-5', key: 'vocab_size', line: 2 },
    { file: 'models', id: 'kimi-k9', key: '', line: 3 },
    { file: null, id: 'x-1', key: 'y', line: 4 },
  ];
  const missing = findMissing(refs, datasets);
  assert.deepEqual(missing.map((m) => m.reason), ['no key', 'no entry', 'no data file in context and id unknown']);
  const report = formatReport(missing.map((m) => ({ ...m, storyboard: 's.md' })));
  assert.match(report, /## models\.json: 2 missing/);
  assert.match(report, /- glm-5\.vocab_size \(no key\): s\.md:2/);
  assert.match(report, /\(unknown file\)/);
});

test('a nested ref stored as a flattened key counts as present', () => {
  const datasets = { hardware: { entries: [{ id: 'formats', facts: { mxfp4_block_size: {}, fp8_e4m3_layout: {} } }] } };
  const refs = [
    { file: 'serving', id: 'mxfp4', key: 'block_size', line: 1 },
    { file: 'hardware', id: 'fp8_e4m3', key: 'layout', line: 2 },
    { file: 'hardware', id: 'mxfp4', key: 'scale_bits', line: 3 },
  ];
  assert.deepEqual(findMissing(refs, datasets).map((m) => m.key), ['scale_bits']);
});
