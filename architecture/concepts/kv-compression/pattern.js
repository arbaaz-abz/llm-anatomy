// Frame 4 and the toy's pattern toggle: head A and head B from attention's toy, with head B reading head A's keys and values or its own.
import { TOY, attentionHead } from '@math/attention.js';

const run = (Q, K, V) => attentionHead(Q, K, V);

// → { a: { weights, mask }, shared: { weights, mask } (head B on head A's K, V), own: head B's own }.
export function patternFor() {
  const A = TOY.heads.A;
  const B = TOY.heads.B;
  const a = run(A.Q, A.K, A.V);
  const shared = run(B.Q, A.K, A.V);
  const own = run(B.Q, B.K, B.V);
  const pick = ({ weights, mask }) => ({ weights, mask });
  return { a: pick(a), shared: pick(shared), own: pick(own) };
}
