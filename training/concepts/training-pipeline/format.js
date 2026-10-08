// training-pipeline pure formatters: the "Check my work" text, built from budgetShares. No DOM.
import { formatCount } from '@math/core.js';
import { budgetShares } from '@math/pipeline.js';
import { formatShare } from '@shared/glyphs/bars.js';

export const NOTHING_PUBLISHED = 'nothing published: no shares to compute';
const TOTAL_DIGITS = 4; // "28.55T tokens"
const GAP = 2; // spaces after the widest expression

export const totalText = (knownTotal) => (knownTotal > 0 ? `${formatCount(knownTotal, { digits: TOTAL_DIGITS })} tokens` : 'none published');

// parts: [{ name, value | null }]. One aligned line per known part and one per unpublished part.
export function checkWork(parts) {
  const r = budgetShares(parts);
  if (r.knownTotal === 0) return NOTHING_PUBLISHED;
  const known = r.parts.filter((q) => !q.unknown);
  const total = formatCount(r.knownTotal, { digits: TOTAL_DIGITS });
  const sum = known.map((q) => formatCount(q.value)).join(' + ');
  const rows = known.map((q) => [q.name, `${formatCount(q.value)} ÷ ${total}`, `= ${formatShare(q.share)}`]);
  const head = known.length > 1 ? ['known total', sum, `= ${total}`] : ['known total', total, null];
  const all = [head, ...rows];
  const labelW = Math.max(...all.map((row) => row[0].length));
  const exprW = Math.max(...all.filter((row) => row[2]).map((row) => row[1].length)) + GAP;
  const line = ([label, expr, result]) => (result ? `${label.padEnd(labelW)} = ${expr.padEnd(exprW)}${result}` : `${label.padEnd(labelW)} = ${expr}`);
  const unpublished = r.parts.filter((q) => q.unknown).map((q) => `${q.name}: not published, so not in the total`);
  return [...all.map(line), ...unpublished].join('\n');
}
