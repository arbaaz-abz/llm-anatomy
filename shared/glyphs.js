// The glyph library (spec §5.1): one drawing per recurring object, used on every page.
// Every builder appends to `parent` (an SVG element) and returns the <g> it created.
// Nothing here touches the DOM at import time or reads global state.
export { NUMBER_CELL, svgEl, valueColor, valueLevel, levelFromFill, formatCell, tokenWidth, maxAbsOf, hatchFill, fitViewBox, cell, token, vector, matrix, heatmap, block, flow } from './glyphs/core.js';
export { kvStack, gpu, rack, request, laneTimelineLayout, laneTimeline, bitFields, bitLayoutLayout, bitLayout } from './glyphs/systems.js';
export { requestSlot, verdict, clipLine, blockPool, blockTable } from './glyphs/serving.js';
export { barSegments, formatShare, shareBarLayout, shareBar, memBar, barsLayout, bars } from './glyphs/bars.js';
export { selectionMark, pixelFill, patch, adder, blockStackLayout, blockStack, dialLayout, dial } from './glyphs/architecture.js';
export { formatTick, curvePlotLayout, curvePlot, rooflineLayout, roofline } from './glyphs/plots.js';
