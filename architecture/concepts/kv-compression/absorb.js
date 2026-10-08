// The absorption check of storyboard §7 (pure): scoring against a rebuilt key equals scoring the folded query against the latent.
import { randomMatrix, matmul, transpose } from '@math/core.js';

// q · (c W_UK) and (q W_UKᵀ) · c for the seeded toy c [1 × 8], W_UK [8 × 4], q [1 × 4] (seeds 21, 22, 23).
export function absorptionCheck() {
  const c = randomMatrix(1, 8, 21, 1);
  const W = randomMatrix(8, 4, 22, 0.5);
  const q = randomMatrix(1, 4, 23, 1);
  return {
    rebuilt: matmul(q, transpose(matmul(c, W)))[0][0],
    folded: matmul(matmul(q, transpose(W)), transpose(c))[0][0],
  };
}

// The by-hand line: c = [1, 2], W = [3, 4]ᵀ, q = 5 → both ways 55.
export function byHand() {
  const c = [[1, 2]];
  const W = [[3], [4]];
  const q = [[5]];
  return {
    rebuilt: matmul(q, matmul(c, W))[0][0],
    folded: matmul(matmul(q, transpose(W)), transpose(c))[0][0],
  };
}
