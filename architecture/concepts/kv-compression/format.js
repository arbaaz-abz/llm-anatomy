// Pure formatters and the toy's "Check my work" text (storyboard §6). No DOM.
const MINUS = '−';
const realMinus = (s) => s.replace(/^-/, MINUS);

// 4,718,592 (thousands separators, real minus).
export const int = (n) => realMinus(Math.round(n).toLocaleString('en-US'));

// "Times smaller than MHA": one decimal, and no ".0" on a whole ratio ("12×", "56.9×", "1×").
export const timesText = (ratio) => `${Number(ratio.toFixed(1))}×`;

// A frame-4 weight: exact 0 and 1 print as integers, the rest at 3 decimals; a masked cell's weight is 0; a cell not computed yet is blank.
export function weightText(v) {
  if (v == null || Number.isNaN(v)) return '';
  if (v === -Infinity) return '0';
  return Number.isInteger(v) ? String(v) : realMinus(v.toFixed(3));
}

const plural = (n, one, many) => `${int(n)} ${n === 1 ? one : many}`;

// numbers = { scheme, stored (KV heads kept), headDim, dLatent, dRope, perLayer, layers, bytesPerElem, bytes, mhaPerLayer, ratio }
export function checkWork(n) {
  const perLayer = n.scheme === 'mla'
    ? `${int(n.dLatent)} (latent) + ${int(n.dRope)} (position key) = ${int(n.perLayer)}`
    : `2 (K and V) × ${plural(n.stored, 'KV head', 'KV heads')} × ${int(n.headDim)} numbers = ${int(n.perLayer)}`;
  return [
    `numbers per token per layer: ${perLayer}`,
    `bytes per token: ${int(n.perLayer)} numbers × ${plural(n.layers, 'layer', 'layers')} × ${n.bytesPerElem} bytes = ${int(n.bytes)} B`,
    `compared with MHA at this shape (${int(n.mhaPerLayer)} numbers per layer): ${timesText(n.ratio)} smaller`,
  ].join('\n');
}

// The line under the wiring figure: how many query heads read how many stored heads.
export function groupsLine({ scheme, queryHeads, stored }) {
  if (scheme === 'mla') return `${plural(queryHeads, 'query head', 'query heads')} rebuild their own keys and values from 1 latent per token`;
  const per = queryHeads / stored;
  return `${plural(queryHeads, 'query head', 'query heads')} read ${plural(stored, 'KV head', 'KV heads')}: ${plural(per, 'query head', 'query heads')} per KV head`;
}
