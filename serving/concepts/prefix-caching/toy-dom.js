// prefix-caching toy DOM helpers: the stage (tree, pool, free queue) and the readout tables for one toy view. No state.
import * as G from '@shared/glyphs.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { PREFILL_TITLE } from './facts.js';
import { drawPool, drawQueue, drawTree } from './parts.js';
import { note } from './stage.js';

const STAGE_W = 580;
const PAD = 8;
const TREE_H_FALLBACK = 20;
const POOL_ROW_H = 47; // one row of blocks (blockPool's stepY at cell 14)
const POOL_FRAME_H = 12 + 29; // first row's offset and one block's height
const LEGEND_GAP = 18;

// Draws the tree (or the reason there is none), the pool and the free queue into `svg`, sized to what it draws.
export function paintStage(svg, stage) {
  svg.replaceChildren();
  G.hatchFill(svg);
  let y = PAD;
  if (stage.tree) {
    note(svg, PAD, y + 8, 'prefix tree: each node is one full block (first…last word)');
    y += 16;
    drawTree(svg, stage.tree, { x: PAD, y });
    y += G.prefixTreeLayout({ nodes: stage.tree, wrapAfter: 6 }).height + LEGEND_GAP;
  } else {
    note(svg, PAD, y + 8, stage.treeNote ?? 'no request has arrived');
    y += TREE_H_FALLBACK;
  }
  note(svg, PAD, y, stage.legend);
  y += 8;
  const rows = Math.ceil(stage.poolBlocks.length / stage.perRow);
  drawPool(svg, stage.poolBlocks, { x: PAD, y: y + 6, blockSize: stage.blockSize, perRow: stage.perRow });
  y += 6 + (rows - 1) * POOL_ROW_H + POOL_FRAME_H + 22;
  drawQueue(svg, stage.queue, { x: PAD, y });
  y += 12;
  svg.setAttribute('viewBox', `0 0 ${STAGE_W} ${y}`);
  svg.setAttribute('width', STAGE_W);
  svg.setAttribute('height', y);
  svg.setAttribute('aria-label', `${stage.legend}. Pool: ${stage.poolBlocks.length} blocks of ${stage.blockSize}.`);
}

const EMPTY_ROW = Object.freeze({ label: 'No request', sub: 'turn a request on', cells: [{ value: '—' }] });

// The toy's readout tables: each request, the totals, the scale-up line and the price. `name` is the data-readout the tests read.
export function readoutTables(view) {
  const perRequest = view.rows.length === 0 ? [EMPTY_ROW] : view.rows.map((r) => ({
    label: r.label,
    cells: [
      { value: r.prompt, name: `prompt-${r.key}` }, { value: r.hit, name: `hit-${r.key}` }, { value: r.computed, name: `computed-${r.key}` },
      { value: r.blocks, name: `blocks-${r.key}` }, { value: r.evicted, name: `evicted-${r.key}` },
    ],
  }));
  return [
    readoutTable({ head: ['Request', 'Prompt tokens', 'From cache', 'Computed', 'Blocks used', 'Evicted'], name: 'requests', rows: perRequest }),
    readoutTable({
      head: ['All requests', 'Value'], name: 'totals',
      rows: [
        { label: 'Prompt tokens', cells: [{ value: view.totals.prompt, name: 'total-prompt' }] },
        { label: 'From cache', cells: [{ value: view.totals.hit, name: 'total-hit' }] },
        { label: 'Computed', cells: [{ value: view.totals.computed, name: 'total-computed' }] },
        { label: 'Hit rate', cells: [{ value: view.totals.rate, name: 'hit-rate' }] },
      ],
    }),
    readoutTable({
      head: [`A shared prefix of ${view.scale.tokens} tokens (the H200 example from ${PREFILL_TITLE})`, 'Value'], name: 'scale-up',
      rows: [
        { label: 'Prefill time skipped', cells: [{ value: view.scale.skipped, name: 'skipped' }] },
        { label: 'KV held until reuse', cells: [{ value: view.scale.held, name: 'held' }] },
      ],
    }),
    ...(view.price ? [readoutTable({
      head: ['Price per million input tokens', 'Value'], name: 'price-table',
      rows: [
        { label: 'Hit rate used', cells: [{ value: view.price.rate, name: 'rate-used' }] },
        { label: 'Blended price', sub: view.price.write ? 'cache write charged' : 'no cache-write fee', cells: [{ value: view.price.blended, name: 'price' }] },
        { label: 'Plain input price', sub: 'no caching', cells: [{ value: view.price.plain, name: 'price-plain' }] },
      ],
    })] : []),
  ];
}
