// prefix-caching toy, "Grow the prefix tree.": mount(host, ctx) → destroy. One state object, one render; every string it prints
// comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice, mountToggle } from '@shared/ui/choice.js';
import { mountPresetButtons } from '@shared/ui/preset-buttons.js';
import { formatShare } from '@shared/glyphs.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { appendRich } from '@shared/lesson-page.js';
import {
  BLOCK_SIZES, POOLS, INITIAL_STATE, REQUEST_KEYS, setBlockSize, setHitMode, setPool, setProvider, setWritePremium, toggleRequest,
} from './format.js';
import { deepseekHitRate, providersFor } from './facts.js';
import { toyView, tryThis } from './toy-view.js';
import { paintStage, readoutTables } from './toy-dom.js';

const WHO_LABEL_ID = 'who-label';
const DEEPSEEK_NOTE = 'DeepSeek charges no cache write';

function tryThisList(data, ctx) {
  const items = tryThis(data).map(({ prompt, insight, rest }) => {
    const li = appendRich(el('li'), prompt, ctx);
    li.append(' → ', el('strong', { textContent: `Insight: ${insight}` }));
    return appendRich(li, rest, ctx);
  });
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data, ctx) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const toggles = REQUEST_KEYS.map(() => el('span', { className: 'toggle-slot' }));
  const refs = {
    who: el('div', { className: 'choice choice--chips' }, [el('span', { id: WHO_LABEL_ID, className: 'choice-label', textContent: 'Who arrives (in order)' }), ...toggles]),
    toggles,
    size: el('div'), pool: el('div'), provider: el('div'), write: el('div'), hit: el('div'), hitChips: el('div'),
    stage: G.svgEl('svg', { role: 'img' }), requests: el('div'), tables: el('div', { className: 'toy-tables' }), pre,
  };
  refs.who.setAttribute('role', 'group');
  refs.who.setAttribute('aria-labelledby', WHO_LABEL_ID);
  host.append(
    el('div', { className: 'toy-controls' }, [refs.who, refs.size, refs.pool, el('p', { className: 'toy-note', textContent: 'Pool sizes count blocks of 4 tokens; at another block size the pool keeps the same token slots.' }), refs.provider, refs.write, refs.hit, refs.hitChips]),
    el('div', { className: 'scroll-x' }, [refs.stage]), el('div', { className: 'scroll-x' }, [refs.requests]), refs.tables,
    el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data, ctx),
  );
  return refs;
}

function mountControls(refs, data, { get, set }) {
  let syncing = false;
  const guard = (fn) => (...args) => { if (!syncing) fn(...args); };
  const providers = providersFor(data) ?? [];
  const toggles = REQUEST_KEYS.map((r, i) => mountToggle(refs.toggles[i], {
    id: `req-${r.key}`, label: r.label, value: INITIAL_STATE.on.includes(r.key), onChange: guard(() => set((s) => toggleRequest(s, r.key))),
  }));
  const size = mountSlider(refs.size, { id: 'blockSize', label: 'Block size', values: BLOCK_SIZES, unit: 'tokens', value: INITIAL_STATE.blockSize, onInput: guard((v) => set((s) => setBlockSize(s, v))) });
  const pool = mountChoice(refs.pool, { id: 'pool', label: 'Pool size', variant: 'chips', value: INITIAL_STATE.pool, options: POOLS.map((n) => ({ value: n, label: `${n} blocks` })), onChange: guard((v) => set((s) => setPool(s, v))) });
  const provider = mountChoice(refs.provider, { id: 'provider', label: 'Price', variant: 'chips', value: INITIAL_STATE.provider, options: providers.map((p) => ({ value: p.id, label: p.label })), onChange: guard((v) => set((s) => setProvider(s, v))) });
  const writeOptions = (deepseek) => [{ value: 'on', label: 'on', ...(deepseek ? { disabled: true, note: DEEPSEEK_NOTE } : {}) }, { value: 'off', label: 'off' }];
  const write = mountChoice(refs.write, { id: 'writePremium', label: 'Charge the cache write', value: 'on', options: writeOptions(false), onChange: guard((v) => set((s) => setWritePremium(s, v === 'on'))) });
  const shown = { grid: null, text: null }; // the slider is on a whole percent; its label may carry the exact live or production rate
  const hit = mountSlider(refs.hit, { id: 'hitRate', label: 'Hit rate for pricing', min: 0, max: 100, step: 1, value: 42, format: (v) => (shown.grid === v ? shown.text : formatShare(v / 100)), onInput: guard((v) => set((s) => setHitMode(s, v))) });
  const chips = mountPresetButtons(refs.hitChips, {
    id: 'hit-presets', label: 'Set it to', options: [{ value: 'toy', label: 'this toy' }, { value: 'deepseek', label: 'DeepSeek 2025' }],
    onPick: (v) => set((s) => setHitMode(s, v === 'toy' ? 'toy' : deepseekHitRate(data))),
  });
  const sync = (state, view) => {
    syncing = true;
    size.set(state.blockSize);
    const deepseek = state.provider === 'deepseek';
    write.update(writeOptions(deepseek));
    if (!deepseek) write.set(state.writePremium ? 'on' : 'off');
    const live = view.sim.promptTokens === 0 ? 0 : (100 * view.sim.hitTokens) / view.sim.promptTokens;
    shown.grid = Math.round(state.hitMode === 'toy' ? live : state.hitMode);
    shown.text = view.hitSlider;
    hit.set(shown.grid);
    syncing = false;
  };
  return { sync, destroy: () => [...toggles, size, pool, provider, write, hit, chips].forEach((c) => c.destroy()) };
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data, ctx);
  let toy = null;
  let latest = null;
  const controls = mountControls(refs, data, { get: () => toy?.get() ?? INITIAL_STATE, set: (patch) => toy?.set(patch) });
  toy = createToyState(INITIAL_STATE, (s) => {
    latest = toyView(s, data);
    paintStage(refs.stage, latest.stage);
    const tables = readoutTables(latest);
    refs.requests.replaceChildren(tables.requests);
    refs.tables.replaceChildren(...tables.others);
    refs.pre.textContent = latest.checkWork;
    if (toy) controls.sync(s, latest);
  });
  controls.sync(toy.get(), latest); // the first render ran before the controls could be synced
  return () => {
    toy.destroy();
    controls.destroy();
    host.replaceChildren();
  };
}
