// The glyph library (spec §5.1): one drawing per recurring object, used on every page.
// Every builder appends to `parent` (an SVG element) and returns the <g> it created.
// Nothing here touches the DOM at import time or reads global state.
export { NUMBER_CELL, svgEl, valueColor, valueLevel, levelFromFill, formatCell, tokenWidth, maxAbsOf, hatchFill, fitViewBox, cell, token, vector, matrix, heatmap, block, flow } from './glyphs/core.js';
export { kvStack, gpu, rack, request } from './glyphs/systems.js';
export { requestSlot, verdict, clipLine, blockPool, blockTable, memBar } from './glyphs/serving.js';
