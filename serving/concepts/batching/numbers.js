// batching stand-ins and constants (hand-authored; frozen). The four requests and the running example live in math/serving.js.
import { deepFreeze } from '@math/core.js';

// Lesson titles printed on the stage (README lesson 33); the page test pins them to shared/concepts.json.
export const TITLES = deepFreeze({ pagedAttention: 'PagedAttention', prefillDecode: 'Prefill vs decode' });
export const FOLLOWED = 'D';
export const MS_AXIS = 400; // milliseconds across the stage in the What-if frames (8-9), held across both
export const BRANCH_BUDGET = 512;
export const BRANCH_PROMPT = 4096;
export const MIXED_STEP = 3; // the step frame 7 zooms into
