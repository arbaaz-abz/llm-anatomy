// Frame 8 of storyboard §5: linear attention's fixed state. Frame 7's compressed grid cross-fades into the state scene.
import { seg } from './stage.js';
import { compressedScene } from './scenes-grid.js';
import { stateScene } from './scenes-state.js';

const FADE = 0.12;

export function drawFrame8(svg, p) {
  const come = seg(p, 0, FADE);
  if (come < 1) compressedScene(svg, 1 - come);
  stateScene(svg, p, come);
}

// The scene frame 9 fades out first: frame 8's end state.
export const stateEnd = (parent, opacity) => stateScene(parent, 1, opacity);
