// Pure numeric helpers shared by every toy. No DOM access in this file.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomMatrix(rows, cols, seed, scale = 1) {
  const next = mulberry32(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (next() * 2 - 1) * scale));
}

export function dot(a, b) {
  if (a.length !== b.length) throw new RangeError(`dot: length ${a.length} vs ${b.length}`);
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}

export function transpose(m) {
  return m[0].map((_, j) => m.map((row) => row[j]));
}

export function matmul(a, b) {
  if (a[0].length !== b.length) {
    throw new RangeError(`matmul: [${a.length}×${a[0].length}] · [${b.length}×${b[0].length}] shapes don't line up`);
  }
  const columns = transpose(b);
  return a.map((row) => columns.map((col) => dot(row, col)));
}

function assertSoftmaxInput(logits, temperature) {
  if (!(temperature > 0)) throw new RangeError('softmax: temperature must be > 0');
  if (logits.length === 0) throw new RangeError('softmax: logits must not be empty');
  logits.forEach((x, i) => {
    if (Number.isNaN(x)) throw new RangeError(`softmax: logit ${i} is NaN`);
    if (x === Infinity) throw new RangeError(`softmax: logit ${i} is +Infinity (use -Infinity to mask)`);
  });
}

export function softmax(logits, temperature = 1) {
  assertSoftmaxInput(logits, temperature);
  const scaled = logits.map((x) => x / temperature);
  // A loop, not Math.max(...scaled): spreading 100K+ logits overflows the call stack.
  const max = scaled.reduce((m, x) => (x > m ? x : m), -Infinity);
  if (max === -Infinity) throw new RangeError('softmax: every logit is masked');
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((acc, x) => acc + x, 0);
  return exps.map((x) => x / sum);
}

export function causalMask(n) {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => j <= i));
}

const DECIMAL = ['B', 'kB', 'MB', 'GB', 'TB', 'PB'];
const BINARY = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];

const significant = (x, digits) => String(Number(x.toPrecision(digits)));
const threeSig = (x) => significant(x, 3);

export function formatBytes(bytes, { binary = false } = {}) {
  const base = binary ? 1024 : 1000;
  const units = binary ? BINARY : DECIMAL;
  let value = bytes;
  let unit = 0;
  // Round first so 999.6 kB becomes "1 MB" rather than "1000 kB".
  while (Number(value.toPrecision(3)) >= base && unit < units.length - 1) {
    value /= base;
    unit += 1;
  }
  return `${threeSig(value)} ${units[unit]}`;
}

const COUNT_SUFFIXES = ['', 'K', 'M', 'B', 'T'];

// digits = significant digits (3 by default; the params toy prints 5: "116.83B").
export function formatCount(n, { digits = 3 } = {}) {
  if (!Number.isInteger(digits) || digits < 1 || digits > 21) throw new RangeError(`formatCount: digits must be an integer 1–21, got ${digits}`);
  let value = n;
  let unit = 0;
  // Round first so 999.6K becomes "1M" rather than "1000K".
  while (Math.abs(Number(value.toPrecision(digits))) >= 1000 && unit < COUNT_SUFFIXES.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${significant(value, digits)}${COUNT_SUFFIXES[unit]}`;
}

// A frozen deep copy for module-level constants (arrays stay arrays); the input is never touched.
export function deepFreeze(value) {
  if (value === null || typeof value !== 'object') return value;
  const copy = Array.isArray(value)
    ? value.map(deepFreeze)
    : Object.fromEntries(Object.entries(value).map(([k, v]) => [k, deepFreeze(v)]));
  return Object.freeze(copy);
}

// ---- Shared prep S3 (Plan 3, ruling P3-R15): one formatter per quantity (README lesson 35) ----

// 3 significant figures, trailing zeros dropped, thousands grouped ("1,180"). x ≥ 1, so at most 2 decimals survive.
const GROUPED = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2, useGrouping: true });
const groupedThreeSig = (x) => GROUPED.format(Number(x.toPrecision(3)));

// Mantissa of a duration: 3 significant figures at any magnitude ("0.543", "1,180"), unlike GROUPED, which caps decimals at 2.
const THREE_SIG = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 3, useGrouping: true });

const roundedThreeSig = (x) => Number(THREE_SIG.format(x).replaceAll(',', ''));

const isFiniteNumber = (x) => typeof x === 'number' && Number.isFinite(x);

// "N×" for a ratio (X-3): 12×, 2×, 73.9×, 3.56×, 1,180×. A ratio below 1 is never "N×", so it throws.
export function formatRatio(x) {
  if (!isFiniteNumber(x) || x < 1) throw new RangeError(`formatRatio: x must be a finite number ≥ 1, got ${x}`);
  return `${groupedThreeSig(x)}×`;
}

// Unit by magnitude: µs < 1 ms ≤ ms < 1 s ≤ s < 60 s ≤ min < 60 min ≤ h < 48 h ≤ days.
const DURATION_UNITS = [
  { name: 'µs', seconds: 1e-6, below: 1e-3 },
  { name: 'ms', seconds: 1e-3, below: 1 },
  { name: 's', seconds: 1, below: 60 },
  { name: 'min', seconds: 60, below: 3600 },
  { name: 'h', seconds: 3600, below: 48 * 3600 },
  { name: 'days', seconds: 86400, below: Infinity },
];

// 3 significant figures, trailing zeros dropped: 2.5 µs, 14.6 ms, 1.25 min, 54 days. Rounds first, so
// 59.97 s prints "1 min", not "60 s". Zero prints "0 s".
export function formatDuration(seconds) {
  if (!isFiniteNumber(seconds) || seconds < 0) throw new RangeError(`formatDuration: seconds must be a finite number ≥ 0, got ${seconds}`);
  if (seconds === 0) return '0 s';
  const firstUnit = DURATION_UNITS.findIndex((u) => seconds < u.below);
  const unit = DURATION_UNITS.slice(firstUnit).find((u) => roundedThreeSig(seconds / u.seconds) * u.seconds < u.below);
  // A value that rounded up into this unit sits just under 1 of it (59.97 s): print the 1, not 0.999.
  const mantissa = unit === DURATION_UNITS[firstUnit] ? seconds / unit.seconds : Math.max(1, seconds / unit.seconds);
  return `${THREE_SIG.format(mantissa)} ${unit.name}`;
}

// ---- Shared prep S6 (Plan 4, ruling P4-R6) ----

const INTEGER = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0, useGrouping: true });

// An exact count, rounded to a whole number: users per GPU, tokens, tok/s per GPU ("51,020", "−3").
// Uses a real minus sign (U+2212), never "-0".
export function formatInt(n) {
  if (!isFiniteNumber(n)) throw new RangeError(`formatInt: n must be a finite number, got ${n}`);
  const text = INTEGER.format(Math.abs(n));
  return n < 0 && text !== '0' ? `−${text}` : text;
}
