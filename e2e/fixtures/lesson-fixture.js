// Test-only fixture lesson (never published): proves the scaffold and runs the lesson contract
// (e2e/lesson-contract.spec.js). #away unmounts it (the contract's "leave" route); #broken-toy mounts it
// with a toy that throws. `render` is exported so the contract can compare a clock sample with it.
import * as G from '@shared/glyphs.js';
import { mountLesson } from '@shared/lesson-page.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { loadJSON } from '@shared/data.js';

export const CAPTIONS = Object.freeze([
  'Four tokens arrive. We follow "sat", token 3.',
  'The query row of "sat" lifts out so its numbers can be read.',
  'The row slides back into place, still marked as the one we follow.',
]);
const STAGE = { w: 580, h: 366 };
const ROW = [0, 2, 0.5, 0];

export function render(index, progress, stage) {
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  svg.setAttribute('aria-label', CAPTIONS[index]);
  ['The', 'cat', 'sat', 'down'].forEach((t, i) => G.token(svg, { x: 20 + i * 60, y: 20, text: t, index: i + 1, state: i === 2 ? 'active' : 'idle' }));
  const lift = [0, progress, 1 - progress][index];
  const y = 200 - 80 * lift;
  G.vector(svg, { x: 120, y, values: ROW, cell: G.NUMBER_CELL, orient: 'row', maxAbs: 3, label: 'q_sat' });
  G.selectionMark(svg, { x: 120, y, w: 4 * G.NUMBER_CELL, h: G.NUMBER_CELL });
}

function mountToy(host) {
  const [sliderHost, choiceHost] = [document.createElement('div'), document.createElement('div')];
  const out = document.createElement('output');
  out.dataset.readout = 'result';
  host.append(sliderHost, choiceHost, out);
  const toy = createToyState({ n: 4, mode: 'double' }, (s) => { out.textContent = String(s.mode === 'double' ? 2 * s.n : s.n * s.n); });
  const slider = mountSlider(sliderHost, { id: 'fixture-n', label: 'n', values: [1, 2, 4, 8], value: 4, unit: 'items', onInput: (n) => toy.set({ n }) });
  const choice = mountChoice(choiceHost, { id: 'fixture-mode', label: 'Mode', value: 'double', onChange: (mode) => toy.set({ mode }),
    options: [{ value: 'double', label: 'double' }, { value: 'square', label: 'square' }] });
  return () => { toy.destroy(); slider.destroy(); choice.destroy(); };
}

const lesson = (toyMount) => ({
  slug: 'fixture',
  hook: 'Does the lesson scaffold hold every section a page needs?',
  intuition: ['A fixture page uses the real scaffold with toy content.', 'It links to [[attention]] and prints `W_O` as code.'],
  animation: { label: 'Fixture animation', steps: CAPTIONS.map((caption) => ({ caption })), render, standIn: 'These numbers are hand-picked stand-ins.', belowFor: (i) => (i === 2 ? ['Page text under the stage on step 3.'] : []) },
  toy: { title: 'Try the fixture', intro: 'Doubles or squares n.', mount: toyMount },
  math: { blocks: [{ tex: '\\htmlClass{hl-q}{q} = x\\,W_Q', note: 'x [1 × 8], W_Q [8 × 4].' }] },
  facts: { framing: 'Three facts read live from [[decoder-anatomy]]\'s data.', rows: [
    { claim: 'DeepSeek-V4-Pro has {deepseek-v4-pro.total_params|count} parameters.' },
    { claim: 'GPT-3 ({gpt-3.release_date|year}) stacked {gpt-3.layers} blocks.' },
    { claim: 'DeepSeek-V3 accepts {sv:deepseek-v3-mtp.acceptance_pct}% of its multi-token-prediction drafts.' },
  ] },
  takeaways: ['One.', 'Two.', 'Three.'],
  links: { next: ['attention'], further: [{ title: 'Example', href: 'https://example.org/', note: 'a stand-in link' }] },
});

const root = document.getElementById('lesson');
let unmount = null;
try {
  // loadJSON resolves relative to shared/data.js, so these paths are the same from any page.
  const [models, hardware, serving, papers] = await Promise.all(['models', 'hardware', 'serving', 'papers'].map((f) => loadJSON(`../data/${f}.json`)));
  const ctx = { concept: { slug: 'fixture', title: 'Lesson fixture', prereqs: ['decoder-anatomy'] }, data: { models, hardware, serving, papers }, href: (slug) => `../../architecture/#${slug}` };
  const sync = () => {
    unmount?.();
    unmount = null;
    root.replaceChildren();
    if (location.hash === '#away') return;
    const broken = () => { throw new Error('fixture: this toy fails on purpose'); };
    unmount = mountLesson(root, ctx, lesson(location.hash === '#broken-toy' ? broken : mountToy));
  };
  addEventListener('hashchange', sync);
  sync();
} catch (error) {
  console.error('Lesson fixture failed to start', error);
  root.innerHTML = '<div class="load-error" role="alert"><p>The lesson fixture did not load.</p></div>';
}
