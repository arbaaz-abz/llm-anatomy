// The "Show me the math" blocks (storyboard §7). Pure: the DeepSeek-V3 line is templated from the data, the checks from the seeded toy.
import { modelShapes } from './shapes.js';
import { ladder, mlaRatio } from './ladder.js';
import { bytesPerToken } from './scheme.js';
import { BYTES_PER_NUMBER } from './numbers.js';
import { absorptionCheck } from './absorb.js';
import { int } from './format.js';

const tex = String.raw;
const num = (n) => int(n).replace(/,/g, '{,}'); // 70,272 → 70{,}272 (TeX would add a space after a bare comma)

function v3Block(data) {
  const v3 = modelShapes(data).v3;
  const head = tex`\text{MLA bytes per token} = L\,(\htmlClass{hl-lat}{d_c} + \htmlClass{hl-rope}{d_{\text{rope}}})\,b`;
  if (!v3) return { tex: head };
  const bytes = bytesPerToken(v3, { layers: v3.layers, bytesPerElem: BYTES_PER_NUMBER });
  const mha = ladder(v3)[0].value;
  return {
    tex: tex`${head},\qquad \text{${v3.name}: } ${v3.layers} \cdot (${v3.dLatent} + ${v3.dRope}) \cdot ${BYTES_PER_NUMBER} = ${num(bytes)}\ \text{B};\quad \frac{${BYTES_PER_NUMBER} \cdot ${v3.queryHeads} \cdot ${v3.headDim}}{${v3.dLatent + v3.dRope}} = ${Number(mlaRatio(ladder(v3)).toFixed(1))}`,
    note: `Per layer, MHA at ${v3.queryHeads} heads of ${v3.headDim} stores ${num(mha)} numbers against MLA's ${v3.dLatent + v3.dRope}.`,
  };
}

export function mathBlocks(data) {
  const check = absorptionCheck();
  return [
    { tex: tex`\text{MHA: } K_h = X W_K^{h},\; V_h = X W_V^{h}\ \ (h = 1..n_H) \qquad \text{GQA: } K_{g(h)},\; V_{g(h)},\ \ g(h) = \Big\lfloor \tfrac{h-1}{n_H / n_{kv}} \Big\rfloor + 1 \qquad \text{MQA: } n_{kv} = 1` },
    { tex: tex`\text{bytes per token} = 2 \cdot L \cdot \htmlClass{hl-kvh}{n_{kv}} \cdot d_{\text{head}} \cdot b \qquad\text{(kv-cache)}` },
    { tex: tex`\text{MLA: } \htmlClass{hl-lat}{c_t} = x_t W_{DKV} \in \mathbb{R}^{d_c},\quad K_h = \htmlClass{hl-lat}{c}\, W_{UK}^{h},\quad V_h = \htmlClass{hl-lat}{c}\, W_{UV}^{h},\quad \htmlClass{hl-rope}{k^{R}_t} = \operatorname{RoPE}(x_t W_{KR}) \in \mathbb{R}^{d_{\text{rope}}}` },
    { tex: tex`\text{absorbed score: } q_h \cdot \big(c\, W_{UK}^{h}\big) = \big(q_h\, W_{UK}^{h\top}\big) \cdot c \qquad \text{(checked numerically: } ${check.rebuilt.toFixed(6)} = ${check.folded.toFixed(6)})` },
    { tex: tex`\text{by hand: } c = [1, 2],\ W_{UK}^{h} = [3, 4]^{\top},\ q = 5:\quad q\,(c\,W) = 5 \cdot 11 = 55,\qquad (q\,W^{\top}) \cdot c = [15, 20] \cdot [1, 2] = 55` },
    v3Block(data),
  ];
}

export const MATH_NOTES = Object.freeze([
  'Shapes (toy in parentheses): X [n × d_model]; W_K^h, W_V^h [d_model × d_head] (· × 4); per token per layer the cache holds 2 · n_kv · d_head numbers (64 / 16 / 8) or d_c + d_rope (8 + 2 = 10). W_DKV [d_model × d_c], W_UK^h and W_UV^h [d_c × d_head]. Hover a highlighted term to outline its glyph on the stage.',
  'Why the position key sits outside the latent: the position rotation depends on the token\'s position, so it cannot be folded into a fixed matrix the way W_UK is; the position key is kept apart for that reason.',
]);
