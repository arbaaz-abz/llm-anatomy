// Gallery page script: palette, glyph figures, and the control demos.
import * as G from '@shared/glyphs.js';
import { mountStepper } from '@shared/ui/stepper.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountMathPanel, linkMathToStage } from '@shared/ui/math-panel.js';
import { renderFact } from '@shared/facts.js';
import { loadJSON } from '@shared/data.js';
import { matmul, transpose, softmax, causalMask, randomMatrix, formatBytes, formatCount } from '@math/core.js';
import { TOY as ATTENTION_TOY } from '@math/attention.js';

const $ = (sel) => document.querySelector(sel);
const svg = (parent, w, h, attrs = {}) => G.svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, ...attrs }, parent);

// ---- theme toggle (per-viewer convenience; storage may throw) ----
const THEMES = ['system', 'light', 'dark'];
const readTheme = () => { try { return localStorage.getItem('theme') ?? 'system'; } catch { return 'system'; } };
function applyTheme(theme) {
  if (theme === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
  $('#theme-toggle').textContent = `Theme: ${theme}`;
  try { localStorage.setItem('theme', theme); } catch { /* private window: toggle still works for this view */ }
}
applyTheme(readTheme());
$('#theme-toggle').addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') ?? 'system';
  applyTheme(THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
});

// ---- palette ----
function swatches(root, list) {
  root.replaceChildren(...list.map(([name, note]) => {
    const el = document.createElement('div');
    el.className = 'swatch';
    el.innerHTML = `<i></i><b></b><small></small>`;
    el.querySelector('i').style.background = `var(${name})`;
    el.querySelector('b').textContent = name;
    el.querySelector('small').textContent = note;
    return el;
  }));
}
swatches($('#sw-base'), [['--bg', 'page'], ['--surface', 'panels'], ['--surface-2', 'inset'], ['--ink', 'text, 16:1 / 15:1'], ['--ink-muted', 'notes, 5.5:1 / 7.3:1'], ['--line', 'hairlines']]);
swatches($('#sw-accent'), [['--accent-arch', 'architecture'], ['--accent-train', 'training'], ['--accent-serve', 'serving']]);
swatches($('#sw-sem'), [['--sem-compute', 'compute-bound'], ['--sem-memory', 'memory-bound, KV'], ['--sem-ok', 'pass, hit, useful'], ['--sem-bad', 'fail, miss, wasted']]);
swatches($('#sw-req'), [['--req-1', 'request A'], ['--req-2', 'request B'], ['--req-3', 'request C'], ['--req-4', 'request D']]);
for (let i = -10; i <= 10; i += 1) {
  const cellEl = document.createElement('i');
  cellEl.style.background = G.valueColor(i, 10);
  cellEl.title = `${i / 10}`;
  $('#value-scale').append(cellEl);
}

// ---- glyph figures ----
const TOKENS = ATTENTION_TOY.tokens;
const TOY = ATTENTION_TOY.heads.A; // the one copy of the hand-picked Q, K, V (attention §4)
const S = matmul(TOY.Q, transpose(TOY.K));
const SCALED = S.map((row) => row.map((v) => v / 2));
const MASK = causalMask(4);
const MASKED = SCALED.map((row, i) => row.map((v, j) => (MASK[i][j] ? v : -Infinity)));
const A = MASKED.map((row) => softmax(row));
const O = matmul(A, TOY.V);
const X = randomMatrix(4, 8, 1);
const VISIBLE_SCALED_SUM = SCALED[2].filter((_, j) => MASK[2][j]).reduce((acc, v) => acc + v, 0);
const poolSlots = [...'AAAAAAAA'.split('').map((o) => ({ owner: o, state: 'filled' })),
  ...'BBBBB'.split('').map((o) => ({ owner: o, state: 'filled' })), { owner: 'B', state: 'reserved' }, { owner: 'B', state: 'reserved' }, { owner: 'B', state: 'reserved' },
  ...'CCCCCCCCCC'.split('').map((o) => ({ owner: o, state: 'filled' })), { owner: 'C', state: 'reserved' }, { owner: 'C', state: 'reserved' },
  ...Array.from({ length: 20 }, () => ({ state: 'free' }))];

const FIGURES = [
  ['token', 'idle · active · dim · value fill · hatched (gradient off)', 300, 44, (s) => {
    G.token(s, { x: 4, y: 8, text: 'The', index: 1 });
    G.token(s, { x: 52, y: 8, text: 'cat', index: 2, state: 'active' });
    G.token(s, { x: 100, y: 8, text: 'sat', index: 3, state: 'dim' });
    G.token(s, { x: 150, y: 8, text: '56', fill: G.valueColor(1.73, 2.65) });
    G.token(s, { x: 192, y: 8, text: '48', fill: G.valueColor(-0.58, 2.65), hatched: true });
    G.token(s, { x: 236, y: 8, text: '?', state: 'idle' });
  }],
  ['vector', 'column and row; numbers print in cells of 36 px or more (40 px here), and every cell has its value as a tooltip', 480, 190, (s) => {
    G.vector(s, { x: 24, y: 24, values: O[2], cell: G.NUMBER_CELL, label: 'o_sat' });
    G.vector(s, { x: 120, y: 24, values: TOY.Q[2], cell: G.NUMBER_CELL, orient: 'row', label: 'q_sat' });
    G.vector(s, { x: 120, y: 90, values: X[2], cell: G.NUMBER_CELL, orient: 'row', label: 'x_sat', maxAbs: 1 });
  }],
  ['matrix', 'shape label [rows × cols]; row labels by token', 300, 130, (s) => {
    G.matrix(s, { x: 44, y: 30, values: TOY.K, cell: 22, label: 'K', rowLabels: TOKENS, maxAbs: 3 });
    G.matrix(s, { x: 180, y: 30, values: TOY.V, cell: 22, label: 'V', maxAbs: 3 });
  }],
  ['heatmap', 'rows = queries, columns = keys; masked cells hatched', 300, 140, (s) => {
    G.heatmap(s, { x: 44, y: 38, values: MASKED, mask: MASK, cell: 22, label: 'S / 2', rowLabels: TOKENS, colLabels: TOKENS.map((t) => t[0]), maxAbs: 1.5 });
    G.heatmap(s, { x: 180, y: 38, values: A, cell: 22, label: 'A', maxAbs: 1 });
  }],
  ['block', 'head / expert / layer: active = filled accent, idle = outline, dim = dashed', 300, 50, (s) => {
    G.block(s, { x: 4, y: 5, w: 88, h: 40, label: 'head A', state: 'active' });
    G.block(s, { x: 104, y: 5, w: 88, h: 40, label: 'head B' });
    G.block(s, { x: 204, y: 5, w: 88, h: 40, label: 'W_O [8×8]', state: 'dim' });
  }],
  ['kvStack', 'one K and one V tile per token, growing right; highlight = this step', 300, 50, (s) => {
    G.kvStack(s, { x: 24, y: 20, count: 12, tile: 14, highlight: [11], label: 'KV cache, 12 tokens' });
  }],
  ['gpu', 'die with SM grid, HBM stacks at the side, memory fill bar', 300, 100, (s) => {
    G.gpu(s, { x: 10, y: 6, memFill: 0.3, label: 'H100 · 30 %' });
    G.gpu(s, { x: 130, y: 6, memFill: 0.85, label: 'GB300 · 85 %' });
  }],
  ['rack', 'GPUs in a frame; link thickness = bandwidth (NVLink vs network)', 300, 100, (s) => {
    G.rack(s, { x: 10, y: 6, gpus: 8, linkWidth: 4, label: 'HGX, 8 NVLink' });
    G.rack(s, { x: 150, y: 6, gpus: 8, linkWidth: 1, label: 'scale-out, 8 Ethernet' });
  }],
  ['request', 'prefill segment (compute) then decode ticks (memory)', 300, 70, (s) => {
    G.request(s, { x: 30, y: 8, prefill: 8, decode: 4, label: 'A' });
    G.request(s, { x: 30, y: 28, prefill: 5, decode: 2, label: 'B' });
    G.request(s, { x: 30, y: 48, prefill: 10, decode: 6, label: 'C' });
  }],
  ['flow', 'arrow with a moving dot; dot color = what it carries', 300, 100, (s) => {
    [['activation', 0.25], ['gradient', 0.5], ['kv', 0.75], ['token', 1]].forEach(([carry, p], i) => {
      const label = G.svgEl('text', { x: 8, y: 16 + i * 22, class: 'g-label' }, G.svgEl('g', { class: 'glyph' }, s));
      label.textContent = carry;
      G.flow(s, { from: [80, 12 + i * 22], to: [290, 12 + i * 22], carry, progress: p });
    });
  }],
  ['verdict', 'checker decision on one answer; aria-label "correct" / "wrong"', 300, 36, (s) => {
    G.token(s, { x: 4, y: 6, text: '7 × 8 = 56' });
    G.verdict(s, { x: 110, y: 18, ok: true });
    G.token(s, { x: 150, y: 6, text: '7 + 8 = 15' });
    G.verdict(s, { x: 256, y: 18, ok: false });
  }],
  ['clipLine', 'number line, shaded safe band, marker at the current ratio', 300, 70, (s) => {
    G.clipLine(s, { x: 30, y: 40, w: 240, lo: 0.6, hi: 1.6, band: [0.8, 1.2], marker: 1.25, label: 'r = π_θ / π_old' });
  }],
  ['blockPool', 'fixed-size KV blocks; filled · reserved (hatched) · free', 310, 150, (s) => {
    G.blockPool(s, { x: 4, y: 4, blocks: 12, blockSize: 4, slots: poolSlots, cell: 14, perRow: 4 });
  }],
  ['blockTable', 'logical → physical lookup, optional ref count', 300, 100, (s) => {
    G.blockTable(s, { x: 10, y: 24, title: 'A', rows: [{ logical: 0, physical: 0 }, { logical: 1, physical: 1 }, { logical: 2, physical: 7 }] });
    G.blockTable(s, { x: 150, y: 24, title: 'D₁ (shared)', rows: [{ logical: 0, physical: 8, ref: 2 }, { logical: 1, physical: 9, ref: 2 }] });
  }],
  ['selectionMark', 'the one selection outline: "the item we follow", in every frame; never an amount', 400, 70, (s) => {
    G.vector(s, { x: 70, y: 14, values: TOY.Q[2], cell: G.NUMBER_CELL, orient: 'row', maxAbs: 3, label: 'q_sat' });
    G.selectionMark(s, { x: 70, y: 14, w: 4 * G.NUMBER_CELL, h: G.NUMBER_CELL });
  }],
  ['patch', 'one image patch as an input piece (4 × 4 crop of a cat\'s ear) · idle · active · dim', 300, 50, (s) => {
    const ear = [0.18, 0.22, 0.78, 0.9, 0.2, 0.55, 0.86, 0.95, 0.42, 0.76, 0.9, 0.84, 0.7, 0.86, 0.8, 0.62];
    G.patch(s, { x: 8, y: 8, pixels: ear, index: 5 });
    G.patch(s, { x: 60, y: 8, pixels: ear, index: 5, state: 'active' });
    G.patch(s, { x: 112, y: 8, pixels: ear, state: 'dim' });
  }],
  ['adder', 'the residual add: a junction on the stream lane, no quantity', 300, 40, (s) => {
    G.adder(s, { x: 20, y: 20 });
  }],
  ['blockStack', '"this block × N" with the residual lane; count is printed, never a height', 300, 200, (s) => {
    G.blockStack(s, { x: 8, y: 8, w: 240, count: 61, active: { block: 1, half: 0 } });
  }],
  ['token draft', 'the draft state: a guess that may still count (dashed, muted), never hatched · idle · active · draft', 300, 50, (s) => {
    G.token(s, { x: 8, y: 10, text: 'on', index: 5 });
    G.token(s, { x: 70, y: 10, text: 'on', index: 5, state: 'active' });
    G.token(s, { x: 132, y: 10, text: 'on', index: 5, state: 'draft' });
  }],
  ['memBar', 'useful · reserved-empty (hatched) · free, with percentages', 300, 60, (s) => {
    G.memBar(s, { x: 4, y: 6, w: 280, useful: 23, reserved: 5, free: 20 });
  }],
  ['shareBar', 'categorical parts: five hues; parts under 18 px fold into "others" and a bracketed zoomed bar; "not published" is neutral, off the scale, never hatched', 360, 210, (s) => {
    G.shareBar(s, { x: 4, y: 6, w: 300, label: 'toy model', parts: [
      { name: 'embedding', value: 128, hue: 1 }, { name: 'attention', value: 512, hue: 2 }, { name: 'MLP', value: 768, hue: 3 },
      { name: 'other (norms)', value: 40, hue: 4 }, { name: 'head', value: 128, hue: 5 }, { name: 'not published', value: 60, unknown: true },
    ] });
  }],
];
$('#figures').replaceChildren(...FIGURES.map(([name, note, w, h, draw]) => {
  const fig = document.createElement('figure');
  if (w > 400) fig.dataset.wide = ''; // an 8-cell row at 40 px needs two columns of the figure grid
  const box = document.createElement('div');
  box.className = 'scroll-x';
  // role="group", not "img": children such as verdict badges carry their own labels.
  const figureSvg = svg(box, w, h, { role: 'group', 'aria-labelledby': `cap-${name}` });
  draw(figureSvg);
  G.fitViewBox(figureSvg, 8);
  document.fonts?.ready.then(() => G.fitViewBox(figureSvg, 8));
  const cap = document.createElement('figcaption');
  cap.id = `cap-${name}`;
  cap.innerHTML = '<b></b><span></span>';
  cap.querySelector('b').textContent = `${name}()`;
  cap.querySelector('span').textContent = note;
  fig.append(box, cap);
  return fig;
}));

// ---- stepper demo: token → embedding → attention row → weighted sum ----
const lerp = (a, b, t) => a + (b - a) * t;
const STEPS = [
  { caption: 'Four tokens. We follow "sat", token 3.' },
  { caption: 'Each token starts as its embedding, a vector of numbers. Here d_model = 8.' },
  { caption: 'A learned matrix turns the embedding into a query. Its dot product with every key gives one score per token: "cat" scores highest.' },
  { caption: 'Divide by √d_head = 2, mask the future to −∞, softmax. The row now adds to 1: 70 % of "sat" goes to "cat".' },
  { caption: 'Output = weights × values: 70 % of cat\'s value, 20 % of sat\'s, 10 % of The\'s.' },
];
const STAGE_W = 580;
const STAGE_H = 366;
const CELL = G.NUMBER_CELL; // 40: numbered rows
const GRID = 20; // un-numbered K and V matrices
const ROW_X = 236;
const ROWS = { x: 60, q: 118, s: 176, a: 244, o: 316 };
const LEFT = { k: { x: 48, y: 118 }, v: { x: 48, y: 246 } };
const TOKEN_X = [8, 60, 112, 164];
const QUERY_ROW = 2; // "sat": the token we follow in every frame
const WEIGHT_MAX = 1.5; // one scale for the weights row and the "× w" chips, so their colors match
// Selection marks (storyboard rule: outlines and accent marks mean "the one we're following", never a quantity).
function queryLabel(s, x, y) {
  const t = G.svgEl('text', { x, y, class: 'g-sub', 'text-anchor': 'middle' }, G.svgEl('g', { class: 'glyph' }, s));
  t.textContent = 'query';
  t.style.fill = 'var(--accent)';
  return t;
}
function rowMarker(s, x, y) {
  const marker = G.svgEl('polygon', { class: 'glyph g-row-marker', points: `${x},${y - 4} ${x + 6},${y} ${x},${y + 4}` }, s);
  marker.style.fill = 'var(--accent)'; // fills via style, per the glyph CSS contract
  return marker;
}
const easeOut = (p) => 1 - (1 - p) ** 2;
function renderDemo(index, progress, stage) {
  // One <svg> per stage, cleared each frame: the DOM of a frame is then a pure function of (index, progress).
  const s = stage.querySelector('svg') ?? svg(stage, STAGE_W, STAGE_H, { role: 'img' });
  G.hatchFill(s); // claims the svg's hatch id up front, so frames with and without hatching share one <svg> element state
  s.replaceChildren();
  s.setAttribute('aria-label', STEPS[index].caption);
  TOKENS.forEach((t, i) => {
    const visible = index > 0 ? 1 : Math.min(Math.max(progress * 4 - i, 0), 1);
    const g = G.token(s, { x: TOKEN_X[i], y: 12, text: t, index: i + 1, state: i === 2 ? 'active' : 'idle' });
    g.style.opacity = String(visible);
  });
  queryLabel(s, TOKEN_X[2] + 18, 50).style.opacity = String(index > 0 ? 1 : Math.min(Math.max(progress * 4 - 2, 0), 1));
  if (index === 0) return;
  // step 2: the embedding row appears as the dot arrives
  const p1 = index === 1 ? progress : 1;
  const xg = G.vector(s, { x: ROW_X, y: ROWS.x, values: X[2], cell: CELL, orient: 'row', maxAbs: 1, label: 'x_sat [1×8]' });
  xg.dataset.link = 'x';
  xg.style.opacity = String(Math.min(p1 * 2, 1));
  if (index === 1) G.flow(s, { from: [TOKEN_X[2] + 40, 24], to: [150, ROWS.x + 14], carry: 'token', progress: p1 });
  if (index === 1) return;
  // step 3: query, keys, one score per key
  const p2 = index === 2 ? progress : 1;
  const qg = G.vector(s, { x: ROW_X, y: ROWS.q, values: TOY.Q[2], cell: CELL, orient: 'row', maxAbs: 3, label: 'q_sat' });
  qg.dataset.link = 'q';
  qg.style.opacity = String(Math.min(p2 * 3, 1));
  const kg = G.matrix(s, { x: LEFT.k.x, y: LEFT.k.y, values: TOY.K, cell: GRID, label: 'K', rowLabels: TOKENS, maxAbs: 3 });
  kg.dataset.link = 'k';
  rowMarker(s, LEFT.k.x - 44, LEFT.k.y + 2 * GRID + GRID / 2);
  const shown = Math.floor(p2 * 4 + 1e-9);
  const sg = G.vector(s, { x: ROW_X, y: ROWS.s, values: S[2].slice(0, shown), cell: CELL, orient: 'row', maxAbs: 4.5, label: 'q·k' });
  sg.dataset.link = 's';
  if (index === 2 && p2 < 1) G.flow(s, { from: [LEFT.k.x + 94, LEFT.k.y + 10], to: [LEFT.k.x + 94, LEFT.k.y + 80], carry: 'activation', progress: p2 });
  if (index === 2) return;
  // step 4: scale, mask and softmax in place; only the numbers and colors change
  const p3 = index === 3 ? progress : 1;
  const rowVals = MASKED[2].map((v, j) => (j === 3 ? -Infinity : lerp(SCALED[2][j], A[2][j], p3)));
  const ag = G.heatmap(s, { x: ROW_X, y: ROWS.a, values: [rowVals], mask: [[true, true, true, false]], cell: CELL, maxAbs: WEIGHT_MAX, label: p3 < 0.5 ? '÷ 2, masked' : 'weights' });
  ag.dataset.link = 'a';
  const sumT = G.svgEl('text', { x: ROW_X + 4 * CELL + 14, y: ROWS.a + CELL / 2, class: 'g-label', 'dominant-baseline': 'central' }, G.svgEl('g', { class: 'glyph' }, s));
  sumT.textContent = `Σ = ${lerp(VISIBLE_SCALED_SUM, 1, p3).toFixed(2)}`; // 1.25 (scaled row) → 1.00 (softmax)
  if (index === 3) return;
  // step 5: V stays plain; each row's weight is a value-colored "× w" chip (same scale as the weights row);
  // the weighted rows slide together into o_sat, whose cells fill.
  const p4 = progress;
  const vg = G.matrix(s, { x: LEFT.v.x, y: LEFT.v.y, values: TOY.V, cell: GRID, label: 'V', rowLabels: TOKENS, maxAbs: 3 });
  vg.dataset.link = 'v';
  rowMarker(s, LEFT.v.x - 44, LEFT.v.y + QUERY_ROW * GRID + GRID / 2);
  const slide = easeOut(Math.min(p4 / 0.6, 1)); // rows travel during the first 60 % of the step…
  const merge = Math.min(Math.max((p4 - 0.6) / 0.3, 0), 1); // …then dissolve into the output as its cells fill
  const target = { x: ROW_X, y: ROWS.o + (CELL - GRID) / 2 };
  TOY.V.forEach((row, i) => {
    const w = A[QUERY_ROW][i];
    const y = LEFT.v.y + i * GRID;
    const tag = G.token(s, { x: LEFT.v.x + 4 * GRID + 10, y: y + 1, text: w === 0 ? '× 0' : `× ${w.toFixed(2)}`, fill: G.valueColor(w, WEIGHT_MAX), hatched: w === 0 });
    tag.classList.add('g-tag');
    // Chips fit the matrix's row pitch: 18px tall on 20px rows, so they stack without touching.
    tag.querySelectorAll('rect').forEach((r) => { r.setAttribute('height', GRID - 2); r.setAttribute('rx', 5); });
    tag.querySelector('.g-text').setAttribute('y', (GRID - 2) / 2);
  });
  const og = G.vector(s, { x: ROW_X, y: ROWS.o, values: O[QUERY_ROW].map((v) => v * Math.min(Math.max((p4 - 0.5) / 0.5, 0), 1)), cell: CELL, orient: 'row', maxAbs: 1.5, label: 'o_sat' });
  og.dataset.link = 'o';
  // The travelling copies are drawn last, on top of o_sat, so the dissolve (0.6–0.9) stays visible while its cells fill.
  TOY.V.forEach((row, i) => {
    const w = A[QUERY_ROW][i];
    if (w === 0 || p4 === 0 || merge === 1) return;
    const ghost = G.vector(s, { x: lerp(LEFT.v.x, target.x, slide), y: lerp(LEFT.v.y + i * GRID, target.y, slide), values: row, cell: GRID, orient: 'row', maxAbs: 3 });
    ghost.classList.add('g-ghost');
    ghost.style.opacity = String(1 - merge);
  });
}
const stepper = mountStepper($('#stepper-root'), { steps: STEPS, render: renderDemo, label: 'One row of attention' });

// ---- slider demo ----
const BYTES_PER_TOKEN = 2 * 36 * 8 * 64 * 2; // 2 (K,V) · layers · kv heads · head dim · bytes (BF16)
const CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const kvOut = $('#kv-readout');
const showKv = (tokens) => { kvOut.value = formatBytes(tokens * BYTES_PER_TOKEN); };
mountSlider($('#slider-root'), { id: 'ctx', label: 'Context length', values: CONTEXTS, value: 131072, unit: 'tokens', format: formatCount, onInput: showKv });
showKv(131072);

// ---- math panel ----
const math = mountMathPanel($('#math-root'), { summary: 'Show me the math', blocks: [
  { tex: '\\htmlClass{hl-q}{Q} = X\\,W_Q,\\qquad \\htmlClass{hl-k}{K} = X\\,W_K,\\qquad \\htmlClass{hl-v}{V} = X\\,W_V', note: 'X [n × d_model] (4 × 8); W [d_model × d_head] (8 × 4); Q, K, V [n × d_head] (4 × 4).' },
  { tex: '\\htmlClass{hl-a}{A} = \\operatorname{softmax}\\!\\left(\\frac{\\htmlClass{hl-q}{Q}\\,\\htmlClass{hl-k}{K}^{\\top}}{\\sqrt{d_{\\text{head}}}} + M\\right),\\qquad \\htmlClass{hl-o}{O} = \\htmlClass{hl-a}{A}\\,\\htmlClass{hl-v}{V}', note: 'M is 0 where j ≤ i and −∞ where j > i. Worked row: softmax([−0.5, 1.5, 0.25, −∞]) = [0.095, 0.703, 0.202, 0].' },
] });
linkMathToStage($('#math-root'), stepper.stage);

// ---- fact ----
const FACT = { entry: 'deepseek-v4-pro', key: 'kv_bytes_per_token' };
loadJSON('../data/models.json').then((models) => {
  const entry = models.entries.find((e) => e.id === FACT.entry);
  const fact = entry?.facts[FACT.key];
  if (!fact) {
    $('#fact-label').textContent = `fact not found: ${FACT.entry}.${FACT.key} is missing from data/models.json`;
    return;
  }
  $('#fact-label').textContent = `${entry.name} · KV cache per token (as of ${models.as_of})`;
  renderFact($('#fact-root'), fact, (v) => formatBytes(v));
}).catch((error) => {
  console.error('Could not load data/models.json', error);
  $('#fact-label').textContent = 'data/models.json did not load';
});
