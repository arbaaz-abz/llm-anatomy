// Frames 3, 9 and 10 of storyboard §5: the layer-stack scenes (6 toy layers, groups of four, the two routes).
import { seg } from './stage.js';
import { END, gridView } from './scenes-grid.js';
import { sixLayers, eightLayers, twoRoutes } from './scenes-layers.js';

// ---- frame 3: frame 2's window grid cross-fades into the stack of layers ----
export function drawFrame3(svg, p) {
  const come = seg(p, 0, 0.25);
  if (come < 1) gridView(svg, { ...END[2], opacity: 1 - come });
  sixLayers(svg, seg(p, 0.25, 1), come);
}

// ---- frame 9: groups of three linear layers and one full layer (frame 8's state scene fades out first) ----
export function drawFrame9(svg, p, previous) {
  const come = seg(p, 0, 0.2);
  if (come < 1) previous(svg, 1 - come);
  eightLayers(svg, seg(p, 0.2, 1), come);
}

// ---- frame 10: the two routes to a million tokens ----
export function drawFrame10(svg, p) {
  const come = seg(p, 0, 0.1);
  if (come < 1) eightLayers(svg, 1, 1 - come);
  twoRoutes(svg, seg(p, 0.1, 1), come);
}
