// Training budgets (training-pipeline §11; reused by midtraining for its context stages).
// Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('budgetShares: … must be …').
// FROZEN after Plan 3 S3: budgetShares keeps this exact signature.

function checkPart(part, i) {
  if (part === null || typeof part !== 'object') throw new RangeError(`budgetShares: parts[${i}] must be an object { name, value }`);
  if (typeof part.name !== 'string') throw new RangeError(`budgetShares: parts[${i}].name must be a string`);
  const { value } = part;
  if (value === null) return;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`budgetShares: parts[${i}].value must be null or a finite number ≥ 0, got ${value}`);
  }
}

// Shares of a budget among named parts; value null = not published. Shares are of the known total only,
// so unknown parts never change the known shares; every share is null when nothing is published
// (knownTotal 0). The printed percentage of a share goes through sharePct (math/memory.js).
//   [27e12, 1.55e12, null]         → knownTotal 2.855e13, unknownCount 1, shares [0.9457, 0.0543, null]
//   [27e12, 1e12, 0.5e12, 0.05e12] → shares [0.9457, 0.0350, 0.0175, 0.0018]   (midtraining)
//   [null, null, null]             → knownTotal 0, unknownCount 3, shares all null
export function budgetShares(parts) {
  if (!Array.isArray(parts) || parts.length === 0) throw new RangeError('budgetShares: parts must be a non-empty array');
  parts.forEach(checkPart);
  const known = parts.filter((p) => p.value !== null);
  const knownTotal = known.reduce((sum, p) => sum + p.value, 0);
  const unknownCount = parts.length - known.length;
  return {
    knownTotal,
    unknownCount,
    parts: parts.map(({ name, value }) => ({
      name,
      value,
      unknown: value === null,
      share: value === null || knownTotal === 0 ? null : value / knownTotal,
    })),
  };
}
