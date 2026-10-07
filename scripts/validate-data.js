// Validates data/*.json: every dated fact must carry a source, a confidence label and a date.
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const DATE_SHAPE = /^(\d{4})-(\d{2})-(\d{2})$/;
const CONFIDENCE = new Set(['confirmed', 'reported']);

const isObject = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const isNonEmptyString = (x) => typeof x === 'string' && x.trim() !== '';

function isCalendarDate(text) {
  const match = DATE_SHAPE.exec(typeof text === 'string' ? text : '');
  if (!match) return false;
  const [year, month, day] = match.slice(1).map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day;
}

function valueError(value, note) {
  if (typeof value === 'string') return value.trim() === '' ? 'value must not be an empty string' : null;
  if (typeof value === 'number' || typeof value === 'boolean') return null;
  const isRange = Array.isArray(value) && value.length === 2 && value.every((v) => typeof v === 'number');
  if (!isRange) return 'value must be a number, string, boolean or [lo, hi]';
  if (value[0] > value[1]) return 'range must be [lo, hi] with lo ≤ hi';
  if (!note) return 'a range needs a note naming the conflicting sources';
  return null;
}

function factErrors(where, fact) {
  if (!isObject(fact)) return [`${where}: fact must be an object`];
  const errors = [];
  const vErr = valueError(fact.value, fact.note);
  if (vErr) errors.push(`${where}: ${vErr}`);
  if (!/^https:\/\/\S+$/.test(fact.source_url ?? '')) errors.push(`${where}: source_url must be an https URL`);
  if (!CONFIDENCE.has(fact.confidence)) errors.push(`${where}: confidence must be "confirmed" or "reported"`);
  if (!isCalendarDate(fact.last_verified)) errors.push(`${where}: last_verified must be a real YYYY-MM-DD date`);
  return errors;
}

function entryErrors(label, entry, index, seen) {
  if (!isObject(entry)) return [`${label}: entry #${index} must be an object`];
  const errors = [];
  if (!isNonEmptyString(entry.id)) errors.push(`${label}: entry #${index} needs a non-empty string id`);
  else if (seen.has(entry.id)) errors.push(`${label}: duplicate id "${entry.id}"`);
  else seen.add(entry.id);
  if (!isNonEmptyString(entry.name)) errors.push(`${label}: entry #${index} (${entry.id ?? 'no id'}) needs a non-empty string name`);
  const facts = entry.facts ?? {};
  if (!isObject(facts)) return [...errors, `${label}: entry #${index} facts must be an object`];
  for (const [key, fact] of Object.entries(facts)) {
    errors.push(...factErrors(`${label}/${entry.id}.${key}`, fact));
  }
  return errors;
}

export function validateDataset(json, label) {
  if (!isObject(json)) return [`${label}: file must contain a JSON object`];
  const errors = [];
  if (!isCalendarDate(json.as_of)) errors.push(`${label}: as_of must be a real YYYY-MM-DD date`);
  if (!Array.isArray(json.entries)) return [...errors, `${label}: entries must be an array`];
  const seen = new Set();
  json.entries.forEach((entry, i) => errors.push(...entryErrors(label, entry, i, seen)));
  return errors;
}

export const DATA_FILES = ['models', 'hardware', 'serving', 'papers'];

async function main() {
  const files = DATA_FILES;
  const results = await Promise.all(files.map(async (file) => {
    const url = new URL(`../data/${file}.json`, import.meta.url);
    try {
      return validateDataset(JSON.parse(await readFile(url, 'utf8')), file);
    } catch (err) {
      return [`${file}: cannot read or parse data/${file}.json (${err.message})`];
    }
  }));
  const errors = results.flat();
  if (errors.length) {
    process.stderr.write(`${errors.join('\n')}\n${errors.length} data error(s)\n`);
    process.exit(1);
  }
  process.stdout.write('data files OK\n');
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) await main();
