// Row normalization (decoder-recap §6): RMSNorm, LayerNorm and the two statistics they print. Pure: no DOM, inputs never mutated.
// rmsNorm(row)  = row / sqrt(mean(row²)) · γ                  (no mean, no β)
// layerNorm(row) = (row − mean) / sqrt(variance) · γ + β      (variance over the row, divided by d, not d − 1)

function requireRow(fn, row) {
  if (!Array.isArray(row) || row.length === 0) throw new RangeError(`${fn}: row must be a non-empty array`);
  row.forEach((v, i) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new RangeError(`${fn}: row[${i}] must be a finite number, got ${v}`);
  });
}

// A scale or shift is one number for the whole row or one number per element.
function perElement(fn, name, value, length) {
  if (Array.isArray(value)) {
    if (value.length !== length) throw new RangeError(`${fn}: ${name} has ${value.length} entries, the row has ${length}`);
    value.forEach((v, i) => {
      if (typeof v !== 'number' || !Number.isFinite(v)) throw new RangeError(`${fn}: ${name}[${i}] must be a finite number, got ${v}`);
    });
    return value;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${fn}: ${name} must be a finite number or an array, got ${value}`);
  return Array.from({ length }, () => value);
}

const mean = (row) => row.reduce((s, v) => s + v, 0) / row.length;

// Root mean square: sqrt(mean(row²)).
export function rms(row) {
  requireRow('rms', row);
  return Math.sqrt(mean(row.map((v) => v * v)));
}

// The mean and the standard deviation (population: variance divided by the row length).
export function meanAndSd(row) {
  requireRow('meanAndSd', row);
  const mu = mean(row);
  return { mean: mu, sd: Math.sqrt(mean(row.map((v) => (v - mu) ** 2))) };
}

export function rmsNorm(row, { gamma = 1 } = {}) {
  requireRow('rmsNorm', row);
  const g = perElement('rmsNorm', 'gamma', gamma, row.length);
  const scale = rms(row);
  if (scale === 0) throw new RangeError('rmsNorm: row is all zeros, so there is nothing to scale');
  return row.map((v, i) => (v / scale) * g[i]);
}

export function layerNorm(row, { gamma = 1, beta = 0 } = {}) {
  requireRow('layerNorm', row);
  const g = perElement('layerNorm', 'gamma', gamma, row.length);
  const b = perElement('layerNorm', 'beta', beta, row.length);
  const { mean: mu, sd } = meanAndSd(row);
  if (sd === 0) throw new RangeError('layerNorm: row has no spread (every entry is equal), so its variance is 0');
  return row.map((v, i) => ((v - mu) / sd) * g[i] + b[i]);
}
