// Reports storyboard references to data/*.json facts that have no matching entry and key.
// A report, not a gate: storyboard reference formats vary, so it always exits 0 on a clean scan.
// Usage: node scripts/check-data-refs.js [file ...]   (e.g. `node scripts/check-data-refs.js models`)
import { readFile, readdir } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

export const DATA_FILES = ['models', 'hardware', 'serving', 'papers'];

const FILE_ALT = DATA_FILES.join('|');
const FILE_SLASH = new RegExp(`^(?:data/)?(${FILE_ALT})\\.json(?:/(.*))?$`); // models.json, data/models.json/id.key
const FILE_DOT = new RegExp(`^(${FILE_ALT})\\.(?!json\\b)(.+)$`); // models.id.key, papers.id
const ID_SHAPE = /^[a-z0-9][a-z0-9.-]*$/;
const KEY_SHAPE = /^[a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)*$/;
const BARE_KEY_WITH_VALUE = /^([a-z][a-z0-9]*_[a-z0-9_]*)(?:\s*=\s*|\s+)\S/; // `gpus_per_rack 16`
const NOT_A_KEY = new Set(['js', 'json', 'md', 'mjs', 'py', 'html', 'css', 'txt', 'svg', 'png']);
const ELLIPSIS = /^(?:…|\.\.\.)$/;
const BLOCK_START = /^\s*(?:$|#|\||[-*] |\d+\. )/; // blank line, heading, table row or list item
const STRONG_ID = /[-\d]/; // a bare id.key with no context counts only if its id has a hyphen or digit

/** First token of a span, with any `= value` or trailing value dropped. */
function refText(span) {
  return span.replace(/\s+/g, ' ').trim().split(/\s|=/)[0];
}

/** Splits `id.key` using known ids first (ids may contain dots), then the first syntactic split. */
export function splitIdKey(text, knownIds = new Set()) {
  if (text.endsWith('.*')) return { id: text.slice(0, -2), key: '*' };
  const known = [...knownIds].filter((id) => text === id || text.startsWith(`${id}.`))
    .sort((a, b) => b.length - a.length)[0];
  if (known) return { id: known, key: text.slice(known.length + 1) };
  for (let i = text.indexOf('.'); i > 0; i = text.indexOf('.', i + 1)) {
    const id = text.slice(0, i);
    const key = text.slice(i + 1);
    const lastKey = key.split('.').pop();
    if (ID_SHAPE.test(id) && (KEY_SHAPE.test(key) || ELLIPSIS.test(key)) && !NOT_A_KEY.has(lastKey)) return { id, key };
  }
  return null;
}

function expandBraces(text) {
  const m = /^\{([^}]*)\}(.*)$/.exec(text);
  return m ? m[1].split(',').map((id) => `${id.trim()}${m[2]}`) : [text];
}

/** Blanks fenced code blocks (keeping line breaks) so offsets still map to line numbers. */
function stripFences(markdown) {
  let inFence = false;
  return markdown.split('\n').map((line) => {
    if (/^\s*```/.test(line)) { inFence = !inFence; return ''; }
    return inFence ? '' : line;
  }).join('\n');
}

const lineAt = (text, offset) => text.slice(0, offset).split('\n').length;

/**
 * Extracts data references from one storyboard.
 * @param {string} markdown
 * @param {Record<string, Set<string>>} idsByFile known ids per data file (used to split dotted ids)
 * @returns {{file: string|null, id: string, key: string, line: number}[]} key '' = entry-only, '*' = any key
 */
export function extractRefs(markdown, idsByFile = {}) {
  const text = stripFences(markdown);
  const lines = text.split('\n');
  const refs = [];
  const EMPTY = { file: null, id: null, key: null };
  let ctx = EMPTY;
  let lastLine = 0;
  // Context (current file, id, key) carries across wrapped lines but not into a new row, item or paragraph.
  const advanceTo = (line) => {
    for (let n = lastLine + 1; n <= line; n += 1) if (BLOCK_START.test(lines[n - 1])) ctx = EMPTY;
    lastLine = Math.max(lastLine, line);
  };
  const fileOfId = (id) => DATA_FILES.find((f) => idsByFile[f]?.has(id)) ?? null;
  const push = (file, id, key, line) => {
    if (!id || id.includes('*')) return;
    const resolved = file && idsByFile[file]?.has(id) ? file : (fileOfId(id) ?? file);
    const finalKey = ELLIPSIS.test(key) ? (ctx.key ?? '') : key;
    refs.push({ file: resolved, id, key: finalKey, line });
    ctx = { file: resolved, id, key: finalKey || ctx.key };
  };
  const pushPath = (file, rest, line, isExplicit) => {
    for (const item of expandBraces(rest)) {
      const split = splitIdKey(item, idsByFile[file] ?? new Set());
      if (split) push(file, split.id, split.key, line);
      else if (isExplicit && ID_SHAPE.test(item)) push(file, item, '', line);
    }
  };

  for (const match of text.matchAll(/`([^`]+)`/g)) {
    const span = match[1];
    const line = lineAt(text, match.index);
    advanceTo(line);
    const ref = refText(span);
    const slash = FILE_SLASH.exec(ref);
    const dot = slash ? null : FILE_DOT.exec(ref);
    if (slash) {
      ctx = { file: slash[1], id: null, key: null };
      if (slash[2]) pushPath(slash[1], slash[2], line, true);
    } else if (dot) {
      pushPath(dot[1], dot[2], line, true);
    } else if (ref.startsWith('.') && ctx.id) {
      const key = ref.slice(1);
      if (KEY_SHAPE.test(key) || key === '*') push(ctx.file, ctx.id, key, line);
    } else if (BARE_KEY_WITH_VALUE.test(span.trim()) && ctx.id && !ref.includes('.')) {
      push(ctx.file, ctx.id, BARE_KEY_WITH_VALUE.exec(span.trim())[1], line);
    } else if (ref.includes('.') || ref.startsWith('{')) {
      const before = refs.length;
      for (const item of expandBraces(ref)) {
        const split = splitIdKey(item, new Set(Object.values(idsByFile).flatMap((s) => [...s])));
        // A bare id.key counts when its id is known, a data file is in context, or it looks like a model id.
        if (split && (fileOfId(split.id) || ctx.file || STRONG_ID.test(split.id))) push(ctx.file, split.id, split.key, line);
      }
      if (refs.length === before && ref.startsWith('{')) continue;
    }
  }
  return refs;
}

/** Returns refs whose entry or key is absent, each tagged with a reason. */
export function findMissing(refs, datasets) {
  const allEntries = Object.values(datasets).flatMap((d) => d?.entries ?? []);
  // A nested ref like `mxfp4.block_size` (inside a `formats` entry) may be stored flattened as `mxfp4_block_size`.
  const isFlattened = (ref) => {
    const flat = `${ref.id}.${ref.key}`.replace(/[.-]/g, '_');
    return ref.key && allEntries.some((e) => Object.hasOwn(e.facts ?? {}, flat));
  };
  return refs.flatMap((ref) => {
    if (isFlattened(ref)) return [];
    const entries = datasets[ref.file]?.entries ?? [];
    const entry = entries.find((e) => e.id === ref.id);
    if (!ref.file) return [{ ...ref, reason: 'no data file in context and id unknown' }];
    if (!entry) return [{ ...ref, reason: 'no entry' }];
    if (ref.key === '' || ref.key === '*') return [];
    return Object.hasOwn(entry.facts ?? {}, ref.key) ? [] : [{ ...ref, reason: 'no key' }];
  });
}

/** Groups missing refs by data file, then by id.key, listing where each is cited. */
export function formatReport(missing) {
  const byFile = new Map();
  for (const m of missing) {
    const file = m.file ?? '(unknown file)';
    const label = m.key ? `${m.id}.${m.key}` : m.id;
    const group = byFile.get(file) ?? new Map();
    const item = group.get(label) ?? { reason: m.reason, where: new Set() };
    item.where.add(`${m.storyboard}:${m.line}`);
    byFile.set(file, group.set(label, item));
  }
  const lines = [];
  for (const [file, group] of [...byFile].sort(([a], [b]) => a.localeCompare(b))) {
    lines.push(`\n## ${file}.json: ${group.size} missing`);
    for (const [label, { reason, where }] of [...group].sort(([a], [b]) => a.localeCompare(b))) {
      lines.push(`- ${label} (${reason}): ${[...where].join(', ')}`);
    }
  }
  return lines.join('\n');
}

async function loadDatasets(root) {
  const datasets = {};
  for (const file of DATA_FILES) {
    try {
      datasets[file] = JSON.parse(await readFile(new URL(`data/${file}.json`, root), 'utf8'));
    } catch (err) {
      if (err.code !== 'ENOENT') process.stderr.write(`warning: cannot parse data/${file}.json (${err.message})\n`);
      datasets[file] = { entries: [] };
    }
  }
  return datasets;
}

async function main() {
  const root = new URL('../', import.meta.url);
  const only = process.argv.slice(2).map((f) => f.replace(/\.json$/, ''));
  const datasets = await loadDatasets(root);
  const idsByFile = Object.fromEntries(DATA_FILES.map((f) => [f, new Set(datasets[f].entries.map((e) => e.id))]));
  const dir = new URL('docs/storyboards/', root);
  const names = (await readdir(dir)).filter((n) => n.endsWith('.md') && !n.startsWith('_') && n !== 'README.md');
  let total = 0;
  const missing = [];
  for (const name of names.sort()) {
    const refs = extractRefs(await readFile(new URL(name, dir), 'utf8'), idsByFile);
    total += refs.length;
    missing.push(...findMissing(refs, datasets).map((m) => ({ ...m, storyboard: name })));
  }
  const shown = only.length ? missing.filter((m) => only.includes(m.file ?? '(unknown file)')) : missing;
  const distinct = new Set(shown.map((m) => `${m.file}/${m.id}.${m.key}`)).size;
  process.stdout.write(`${total} references scanned in ${names.length} storyboards; ${shown.length} missing citations (${distinct} distinct facts)`);
  process.stdout.write(`${formatReport(shown)}\n`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) await main();
