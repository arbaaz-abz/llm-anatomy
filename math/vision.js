// What an image or a video costs a language model (multimodal §6). Pure: no DOM, inputs never mutated.
// An image is cut into patch × patch squares; a merge × merge group of neighboring patches becomes one token.

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer`);
}

// The patch grid of a width × height image. Both sides must be multiples of the patch size.
export function patchGrid({ width, height, patch }) {
  requireCount('patchGrid', 'width', width);
  requireCount('patchGrid', 'height', height);
  requireCount('patchGrid', 'patch', patch);
  if (width % patch !== 0) throw new RangeError('patchGrid: width must be a multiple of patch');
  if (height % patch !== 0) throw new RangeError('patchGrid: height must be a multiple of patch');
  const cols = width / patch;
  const rows = height / patch;
  return { cols, rows, patches: cols * rows };
}

// Tokens after merging merge × merge neighbors, for `frames` frames (no pooling over time).
// merge = 1 means no merge. Both grid sides must be divisible by merge.
export function visionTokens({ width, height, patch, merge = 1, frames = 1 }) {
  const { cols, rows, patches } = patchGrid({ width, height, patch });
  requireCount('visionTokens', 'merge', merge);
  requireCount('visionTokens', 'frames', frames);
  if (cols % merge !== 0) throw new RangeError('visionTokens: cols must be divisible by merge');
  if (rows % merge !== 0) throw new RangeError('visionTokens: rows must be divisible by merge');
  const tokensPerFrame = patches / (merge * merge);
  return { patches, tokensPerFrame, tokens: tokensPerFrame * frames };
}
