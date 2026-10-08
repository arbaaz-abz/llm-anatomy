// The numbers of frames 8 and 9 (pure): the per-layer ladder for one shape, and the four real models side by side.
import { bytesPerToken, numbersPerLayer } from './scheme.js';
import { kvCacheBytes } from '@math/memory.js';
import { BYTES_PER_NUMBER, DEFAULT_CONTEXT } from './numbers.js';

const GQA_STOP = 8; // the ladder's "common" GQA rung and the fixed 2-head rung

// Rows MHA · GQA-8 · GQA-2 · MLA · MQA (MLA sits where its number does in the toy); GQA-8 is null when the shape has no more than 8 query heads.
export function ladder({ queryHeads, headDim, dLatent, dRope }) {
  const at = (scheme, kvHeads) => numbersPerLayer({ scheme, queryHeads, kvHeads, headDim, dLatent, dRope });
  return [
    { key: 'mha', label: 'MHA', value: at('mha') },
    { key: 'gqa8', label: 'GQA-8', value: queryHeads > GQA_STOP ? at('gqa', GQA_STOP) : null },
    { key: 'gqa2', label: 'GQA-2', value: at('gqa', 2) },
    { key: 'mla', label: 'MLA', value: at('mla') },
    { key: 'mqa', label: 'MQA', value: at('mqa') },
  ];
}

// "MHA ÷ MLA" at one shape, and the same ratio if V3's real (wider) query/key and value heads were the baseline.
export const mlaRatio = (rows) => rows[0].value / rows.find((r) => r.key === 'mla').value;
export const realHeadBaseline = ({ queryHeads, qkWidth, vWidth, dLatent, dRope }) => (queryHeads * (qkWidth + vWidth)) / (dLatent + dRope);

export const schemeLabel = (shape) => {
  if (shape.scheme === 'gqa') return `GQA-${shape.kvHeads}`;
  return shape.scheme.toUpperCase();
};

// Frame 9: each model's bytes per token (all layers, 2 bytes a number) and its cache at the default context.
export function compare(shape) {
  const perToken = bytesPerToken({ ...shape, kvHeads: shape.kvHeads }, { layers: shape.layers, bytesPerElem: BYTES_PER_NUMBER });
  return { shape, scheme: schemeLabel(shape), perToken, atContext: kvCacheBytes({ bytesPerToken: perToken, tokens: DEFAULT_CONTEXT }), isWhatIf: DEFAULT_CONTEXT > shape.contextLength };
}
