// prefix-caching frames 1-8 as scenes (pure): sceneAt(frame, p, text) → { tree, pool, overlays }. The tree and the pool are always
// drawn; an overlay is a keyed extra (walk lines, counters, table, notes) that fades in when its key is new to the frame and
// out when the next frame drops it, so a frame's start is exactly the previous frame's end (template rule 4).
// Every state comes from the simulation in scenes.js; the times below only order what happens inside a frame.
import * as G from '@shared/glyphs.js';
import { formatInt } from '@math/core.js';
import { PREFIX_REQUESTS, blockTexts } from '@math/prefix.js';
import {
  LOG, DEFAULT_MODEL, DEFAULT_PATHS, POOL_BLOCKS, treeNodes, nodeState, evictedIds, poolAfter, poolDuring, liveQueue, tableChips,
} from './scenes.js';
import { drawWalk, drawQueue, drawStrip, drawCounters, drawTable, drawTwin, pctText } from './parts.js';
import { BIG_Y, FRAME1_NOTES_Y, NOTE_Y, WALK, layer, note, seg } from './stage.js';

const [A, B, C, D] = [0, 1, 2, 3];
const ID = PREFIX_REQUESTS.map((r) => r.id);
const passed = (p, times) => times.filter((t) => p >= t).length;
const textsOf = (i) => blockTexts([...PREFIX_REQUESTS[i].prompt, ...PREFIX_REQUESTS[i].output], 4);
const overlay = (key, draw, at = 0) => ({ key, at, draw });
const pathIds = (i) => DEFAULT_PATHS[i].path;
const blocksFor = (blocks, overrides = []) => ({ blocks, overrides });

function tree({ visibleUpTo, freshCount = Infinity, active = null, hitCount = Infinity, evicted = [], followed = [] }) {
  const visible = (n) => n.creatorIndex < visibleUpTo || (n.creatorIndex === visibleUpTo && DEFAULT_PATHS[visibleUpTo].fresh.indexOf(n.id) < freshCount);
  const state = nodeState({ paths: DEFAULT_PATHS, active, hitCount, evicted: new Set(evicted) });
  return treeNodes(DEFAULT_MODEL, { visible, state, followed: new Set(followed) });
}

// "B: You are a cat (hit) · . Reply in rhyme (hit) · Do you like fish (new)": the followed request's blocks in full.
const walk = (i, words, status, at = 0.1) => overlay(`walk:${ID[i]}`, (g) => drawWalk(g, ID[i], words.map((w, k) => (status[k] ? `${w} (${status[k]})` : w))), at);
const queueOverlay = (queue, { suffix = '', label, at = 0 } = {}) => overlay('queue', (g) => drawQueue(g, queue, { suffix, label }), at);
const noteOverlay = (key, str, at = 0.1) => overlay(key, (g) => note(g, 8, NOTE_Y, str, { cls: '' }), at);
const counters = (title, row, p, times) => overlay(`counters:${title}`, (g) => drawCounters(g, `request ${title}`, [
  row.promptTokens, p >= times[0] ? row.hitTokens : null, p >= times[1] ? row.computed : null, p >= times[2] ? pctText(row.hitTokens, row.promptTokens) : null,
]), 0.1);

// The key of the followed block: its own tokens and its parents', arriving one after another from the left (frame 2).
function drawKey(parent, words, slide) {
  let x = 8;
  words.forEach((w, i) => {
    const g = layer(parent, seg(slide, i * 0.3, i * 0.3 + 0.4));
    if (i > 0) note(g, x - 11, 150, '·', { cls: '' });
    G.token(g, { x, y: 134, text: w });
    x += G.tokenWidth(w) + 16;
  });
}

function frame1(p, text) {
  const done = p >= 0.65;
  const shown = passed(p, [0.1, 0.22, 0.34, 0.46]);
  return {
    tree: tree({ visibleUpTo: A, freshCount: shown, active: done ? null : A, followed: pathIds(A) }),
    pool: blocksFor(done ? poolAfter(LOG[A], POOL_BLOCKS) : poolDuring(null, LOG[A], POOL_BLOCKS)),
    overlays: [
      walk(A, textsOf(A), []),
      noteOverlay('note:1', 'A: 12 prompt + 4 answer = 16 tokens = 4 blocks (0-3)'),
      overlay('f1-notes', (g) => text.frame1.forEach((line, i) => note(g, 8, FRAME1_NOTES_Y[i], line)), 0.1),
      ...(done ? [queueOverlay(LOG[A].freeQueue, { at: 0.65 })] : []),
    ],
  };
}

function frame2(p) {
  return {
    tree: tree({ visibleUpTo: A, followed: [pathIds(A)[2]] }),
    pool: blocksFor(poolAfter(LOG[A], POOL_BLOCKS)),
    overlays: [
      overlay('keychips', (g) => drawKey(g, textsOf(A).slice(0, 3), seg(p, 0.15, 0.6)), 0.1),
      ...(p >= 0.5 ? [noteOverlay('note:2', 'the key of block 2 covers 3 blocks = 12 tokens', 0.5)] : []),
      queueOverlay(LOG[A].freeQueue),
    ],
  };
}

const B_PROMPT = ['You are a cat', '. Reply in rhyme', 'Do you like fish', '?'];

// B's block 5 holds the prompt's last token ('?') and waits for three answer tokens: slot 0 filled, the rest reserved.
function partialBlock(answer) {
  const block = LOG[B].blocks[3];
  return [0, 1, 2, 3].map((slot) => ({ block, slot, state: slot <= answer ? 'filled' : 'reserved', owner: 'B' }));
}

function frame3(p) {
  const row = LOG[B];
  const hits = passed(p, [0.25, 0.35]);
  const fresh = passed(p, [0.5]);
  return {
    tree: tree({ visibleUpTo: B, freshCount: fresh, active: B, hitCount: hits, followed: pathIds(B) }),
    pool: fresh ? blocksFor(poolDuring(LOG[A], row, POOL_BLOCKS), partialBlock(0)) : blocksFor(poolAfter(LOG[A], POOL_BLOCKS)),
    overlays: [
      walk(B, B_PROMPT, [hits >= 1 ? 'hit' : null, hits >= 2 ? 'hit' : null, fresh ? 'new' : null, fresh ? 'partial' : null]),
      counters('B', row, p, [0.55, 0.65, 0.75]),
      noteOverlay('note:3', 'B: 13 prompt tokens · from cache 8 · computed 5', 0.55),
      ...(p >= 0.7 ? [overlay('strip:B', (g) => drawStrip(g, 'B uses blocks', tableChips(row)), 0.7)] : []),
      queueOverlay(fresh ? liveQueue(LOG[A], row, POOL_BLOCKS) : LOG[A].freeQueue),
    ],
  };
}

function frame4(p) {
  const row = LOG[B];
  const answer = passed(p, [0.15, 0.3, 0.45]);
  const full = p >= 0.75;
  const finished = p >= 0.9;
  return {
    tree: tree({ visibleUpTo: B, freshCount: full ? 2 : 1, active: B, hitCount: 2, followed: pathIds(B) }),
    pool: finished ? blocksFor(poolAfter(row, POOL_BLOCKS)) : blocksFor(poolDuring(LOG[A], row, POOL_BLOCKS), partialBlock(answer)),
    overlays: [
      walk(B, full ? [...B_PROMPT.slice(0, 3), '? Yes , fish'] : B_PROMPT, ['hit', 'hit', 'new', full ? 'new' : 'partial']),
      counters('B', row, 1, [0, 0, 0]),
      noteOverlay('note:4', 'partial block: 1 of 4 slots, full after 3 answer tokens · B uses blocks 0, 1, 4, 5'),
      overlay('strip:B', (g) => drawStrip(g, 'B uses blocks', tableChips(row)), 0.7),
      queueOverlay(finished ? row.freeQueue : liveQueue(LOG[A], row, POOL_BLOCKS)),
    ],
  };
}

function frame5(p) {
  const row = LOG[C];
  const hits = passed(p, [0.2, 0.3, 0.4, 0.5]);
  const fresh = passed(p, [0.6, 0.85]);
  const finished = p >= 0.9;
  const words = ['You are a cat', '. Reply in rhyme', 'Where did you sit', 'The cat sat down', 'Why down there ?'];
  return {
    tree: tree({ visibleUpTo: C, freshCount: fresh, active: C, hitCount: hits, followed: pathIds(C) }),
    pool: blocksFor(finished ? poolAfter(row, POOL_BLOCKS) : p >= 0.6 ? poolDuring(LOG[B], row, POOL_BLOCKS) : poolAfter(LOG[B], POOL_BLOCKS)),
    overlays: [
      walk(C, words, [...[1, 2, 3, 4].map((k) => (hits >= k ? 'hit' : null)), fresh ? 'new' : null]),
      overlay('counters:C', (g) => drawCounters(g, 'request C (A\'s second turn)', [row.promptTokens, p >= 0.55 ? row.hitTokens : null, p >= 0.65 ? row.computed : null, p >= 0.75 ? pctText(row.hitTokens, row.promptTokens) : null]), 0.1),
      noteOverlay('note:5', 'C: 20 prompt tokens · from cache 16 · computed 4', 0.55),
      overlay('strip:C', (g) => drawStrip(g, 'C uses blocks', tableChips(row)), 0.7),
      queueOverlay(finished ? row.freeQueue : p >= 0.6 ? liveQueue(LOG[B], row, POOL_BLOCKS) : LOG[B].freeQueue),
    ],
  };
}

function frame6(p) {
  const row = LOG[D];
  const fresh = passed(p, [0.25, 0.35, 0.45]);
  const nodes = tree({ visibleUpTo: D, freshCount: fresh, active: D, hitCount: 0, followed: pathIds(D) });
  const words = ['You are a dog', '. Reply in prose', 'Where did you sit'];
  return {
    tree: nodes,
    pool: blocksFor(poolAfter(LOG[C], POOL_BLOCKS)),
    overlays: [
      walk(D, words, [0, 1, 2].map((k) => (fresh > k ? 'new' : null))),
      overlay('counters:D', (g) => drawCounters(g, 'request D', [row.promptTokens, p >= 0.55 ? 0 : null, p >= 0.65 ? row.computed : null, p >= 0.75 ? pctText(0, row.promptTokens) : null]), 0.1),
      noteOverlay('note:6', 'D: 12 prompt tokens · from cache 0 · computed 12', 0.55),
      overlay('token-level', (g) => note(g, WALK.x, WALK.y[1], '(token-level matching would reuse 3: You are a)'), 0.55),
      ...(fresh >= 3 ? [overlay('twin', (g) => drawTwin(g, nodes, pathIds(A)[2], pathIds(D)[2]), 0.45)] : []),
      queueOverlay(LOG[C].freeQueue),
    ],
  };
}

function frame7(p) {
  const row = LOG[D];
  const pops = passed(p, [0.2, 0.35, 0.5, 0.65]);
  const answer = passed(p, [0.8]);
  const taken = row.blocks.slice(row.hitBlocks).slice(0, pops);
  const full = evictedIds(DEFAULT_MODEL, row, D);
  return {
    tree: tree({ visibleUpTo: D, freshCount: 3 + answer, active: D, hitCount: 0, evicted: full.slice(0, pops), followed: pathIds(D) }),
    pool: blocksFor(poolAfter(LOG[C], POOL_BLOCKS).map((b, i) => (taken.includes(i) ? { state: 'filled', owner: 'D' } : b))),
    overlays: [
      walk(D, ['You are a dog', '. Reply in prose', 'Where did you sit', ...(answer ? ['On the mat .'] : [])], ['new', 'new', 'new', answer ? 'new' : null]),
      overlay('counters:D', (g) => drawCounters(g, 'request D', [row.promptTokens, 0, row.computed, pctText(0, row.promptTokens)])),
      noteOverlay('note:7', 'evicted: blocks 5, 4, 7, 6 · kept: blocks 0-3 (system prompt, A\'s turn)', 0.2),
      queueOverlay(LOG[C].freeQueue.slice(pops), { label: 'free queue:', suffix: `· after D: ${row.freeQueue.join(' ')}` }),
    ],
  };
}

function frame8(p, text) {
  const rows = LOG.map((r, i) => ({ id: ID[i], prompt: r.promptTokens, cache: r.hitTokens, computed: r.computed }));
  const shown = passed(p, [0.2, 0.3, 0.4, 0.5]);
  const totalPrompt = rows.reduce((s, r) => s + r.prompt, 0);
  const totalCache = rows.reduce((s, r) => s + r.cache, 0);
  return {
    tree: tree({ visibleUpTo: D, evicted: evictedIds(DEFAULT_MODEL, LOG[D], D) }),
    pool: blocksFor(poolAfter(LOG[D], POOL_BLOCKS)),
    overlays: [
      overlay('table', (g) => drawTable(g, rows.slice(0, shown)), 0.15),
      overlay('big', (g) => {
        note(g, 8, BIG_Y - 30, `${formatInt(totalCache)} of ${formatInt(totalPrompt)} prompt tokens came from the cache`);
        note(g, 8, BIG_Y, pctText(totalCache, totalPrompt), { cls: '', scale: 2.4 });
      }, 0.65),
      overlay('deepseek', (g) => note(g, 8, BIG_Y + 20, text.deepseek, { cls: '' }), 0.85),
    ],
  };
}

export const SCENES = Object.freeze([frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8]);
export const ARRIVE = 0.1; // a frame keeps the previous frame's end state until here, then changes it

// Frame `index` at progress p. Before ARRIVE the frame still shows the previous frame's end state (template rule 4).
export function sceneAt(index, p, text) {
  if (index > 0 && p < ARRIVE) return SCENES[index - 1](1, text);
  return SCENES[index](p, text);
}
