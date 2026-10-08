// multimodal's stand-in numbers (storyboard §4): a 16 × 16 grey image, the vectors it turns into and the toy sequence.
// Everything is computed from seeded matrices with math/core.js, so the stage never types a value. Pure, no DOM.
import { deepFreeze, randomMatrix, matmul, transpose, softmax, causalMask } from '@math/core.js';
import { visionTokens } from '@math/vision.js';
import { maxAbsOf } from '@shared/glyphs.js';

export const SIDE = 16; // image pixels per side
export const PATCH = 4; // toy patch size
export const GRID_N = SIDE / PATCH; // 4 × 4 patches
export const D_VIT = 8; // encoder width = d_model of the toy
export const MERGE = 2;
export const FOLLOWED = 5; // patch 6 (0-based 5), the patch we follow through every frame
export const TOKEN_WORDS = Object.freeze(['The', 'cat', 'sat', 'down']);
export const IMAGE_NAMES = Object.freeze(['img₁', 'img₂', 'img₃', 'img₄']);
export const SEQUENCE = Object.freeze([...IMAGE_NAMES, ...TOKEN_WORDS]); // positions 1–8
export const SAT = 6; // "sat" is position 7 (0-based 6)

// The same cat's-ear crop decoder-anatomy draws (4 × 4 greys); it is patch 6 of this image.
export const EAR = Object.freeze([0.18, 0.22, 0.78, 0.9, 0.2, 0.55, 0.86, 0.95, 0.42, 0.76, 0.9, 0.84, 0.7, 0.86, 0.8, 0.62]);

const smooth = (t) => { const c = Math.min(Math.max(t, 0), 1); return c * c * (3 - 2 * c); };
const grey = (x, y) => {
  const edge = smooth((x + 0.9 * y - 9.1) / 1.4 + 0.5); // dark background top-left, light fur bottom-right
  const grain = ((x * 7 + y * 13) % 5) * 0.012; // fixed texture, no randomness
  return Math.round((0.16 + 0.56 * edge + grain) * 100) / 100;
};

// Row-major 16 × 16 greys with patch 6 replaced by the ear crop.
const pixelAt = (x, y) => {
  const inFollowed = Math.floor(x / PATCH) === FOLLOWED % GRID_N && Math.floor(y / PATCH) === Math.floor(FOLLOWED / GRID_N);
  return inFollowed ? EAR[(y % PATCH) * PATCH + (x % PATCH)] : grey(x, y);
};

// Patch i (row-major, 0-based) as its 16 greys, row by row inside the patch.
export const patchPixels = (i) => {
  const [row, col] = [Math.floor(i / GRID_N), i % GRID_N];
  return Array.from({ length: PATCH * PATCH }, (_, k) => pixelAt(col * PATCH + (k % PATCH), row * PATCH + Math.floor(k / PATCH)));
};
export const PATCHES = deepFreeze(Array.from({ length: GRID_N * GRID_N }, (_, i) => patchPixels(i)));

// Patch embedding: Z0 = flatten(patch) · W_patch, W_patch [16 × 8].
export const W_PATCH = deepFreeze(randomMatrix(PATCH * PATCH, D_VIT, 31, 0.5));
export const Z0 = deepFreeze(matmul(PATCHES, W_PATCH)); // [16 × 8]

// Vision encoder, one stand-in layer with no mask: every patch attends to every patch (a [16 × 16] pattern).
const patchScores = matmul(Z0, transpose(Z0)).map((row) => row.map((v) => v / Math.sqrt(D_VIT)));
export const PATTERN = deepFreeze(patchScores.map((row) => softmax(row)));
const mixed = matmul(PATTERN, Z0);
export const Z1 = deepFreeze(Z0.map((row, i) => row.map((v, j) => v + 0.5 * mixed[i][j]))); // each vector now carries its neighbors

// Merge: a 2 × 2 group of patches becomes one vector of 4 × 8 = 32 numbers.
export const groupOf = (i) => Math.floor(i / GRID_N / MERGE) * (GRID_N / MERGE) + Math.floor((i % GRID_N) / MERGE);
export const slotOf = (i) => ((Math.floor(i / GRID_N)) % MERGE) * MERGE + ((i % GRID_N) % MERGE);
export const MERGE_GROUPS = deepFreeze(Array.from({ length: (GRID_N / MERGE) ** 2 }, (_, g) => (
  Array.from({ length: MERGE * MERGE }, (__, s) => Z1.findIndex((_r, i) => groupOf(i) === g && slotOf(i) === s)))));
export const MERGED = deepFreeze(MERGE_GROUPS.map((members) => members.flatMap((i) => Z1[i]))); // [4 × 32]

// Projector [32 → 8]: the image tokens at the width of the stream.
export const W_PROJ = deepFreeze(randomMatrix(MERGE * MERGE * D_VIT, D_VIT, 32, 0.25));
export const IMAGE_TOKENS = deepFreeze(matmul(MERGED, W_PROJ)); // [4 × 8]

// The toy sequence "img₁ … img₄ The cat sat down": a causal pattern over 8 positions (stand-in scores, seeded).
const SEQ_SCORES = randomMatrix(SEQUENCE.length, SEQUENCE.length, 33, 2);
const SEQ_MASK = causalMask(SEQUENCE.length);
export const SEQ_PATTERN = deepFreeze(SEQ_SCORES.map((row, i) => softmax(row.map((v, j) => (SEQ_MASK[i][j] ? v : -Infinity)))));
export const SEQ_VISIBLE = SEQ_MASK;

// Value scales (each matrix's own largest magnitude, so colors use the whole scale).
export const SCALE = Object.freeze({ z0: maxAbsOf(Z0), z1: maxAbsOf(Z1), merged: maxAbsOf(MERGED), tokens: maxAbsOf(IMAGE_TOKENS), pattern: 0.3, seq: 0.5 });

// The toy image's own count, from the math module (storyboard: 16 patches → 4 tokens).
export const TOY_COUNT = visionTokens({ width: SIDE, height: SIDE, patch: PATCH, merge: MERGE });
