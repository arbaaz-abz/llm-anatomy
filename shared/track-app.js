// Track shell: nav, index view, and lazy loading of concepts/<slug>.js.
import { indexConcepts, bySection } from './concepts.js';
import { startRouter } from './router.js';
import { safeStorage, toggleLearned, LEARNED_KEY } from './storage.js';
import { conceptHref, hubHref, currentTarget } from './links.js';
import { loadJSON } from './data.js';
import { el } from './ui/dom.js';

// Chrome, Safari and Firefox word a failed dynamic import differently.
const isMissingModule = (error) =>
  error?.code === 'ERR_MODULE_NOT_FOUND'
  || /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module/i.test(error?.message ?? '');

export async function loadConcept(slug, importer) {
  try {
    const module = (await importer(slug)).default;
    if (typeof module?.mount !== 'function') throw new Error(`concepts/${slug}.js must default-export { mount(el, ctx) }`);
    return { status: 'ok', module };
  } catch (error) {
    if (!isMissingModule(error)) return { status: 'error', error };
    // A concept whose own imports fail looks identical to a missing file in Chrome,
    // so keep a trace while pages are being built.
    console.warn(`concepts/${slug}.js did not load; showing "coming soon"`, error);
    return { status: 'missing' };
  }
}

// A lesson's mount() or unmount() must never take the shell down with it.
export function mountSafely(module, container, ctx) {
  try {
    const unmount = module.mount(container, ctx);
    return { status: 'ok', unmount: typeof unmount === 'function' ? unmount : null };
  } catch (error) {
    console.error(`Lesson "${ctx.concept?.slug}" threw while mounting`, error);
    return { status: 'error', unmount: null, error };
  }
}

export function safeUnmount(unmount) {
  try {
    unmount?.();
  } catch (error) {
    console.error('Lesson threw while unmounting', error);
  }
}

export async function mountTrack({ root, track }) {
  const [graph, links, models, hardware, serving, papers] = await Promise.all([
    loadJSON('./concepts.json'), loadJSON('./links.json'), loadJSON('../data/models.json'), loadJSON('../data/hardware.json'),
    loadJSON('../data/serving.json'), loadJSON('../data/papers.json'),
  ]);
  const data = Object.freeze({ models, hardware, serving, papers }); // every lesson's facts (fillClaim prefixes: none, hw:, sv:, paper:)
  const index = indexConcepts(graph.concepts);
  const trackInfo = graph.tracks.find((t) => t.id === track);
  if (!trackInfo) throw new Error(`mountTrack: unknown track "${track}"`);
  const target = currentTarget();
  const store = safeStorage();
  const href = (slug) => {
    const concept = index.get(slug);
    return concept ? conceptHref({ from: track, track: concept.track, slug, target, artifactUrls: links.artifactUrls }) : null;
  };
  const title = (slug) => index.get(slug)?.title ?? null;
  const importer = (slug) => import(new URL(`concepts/${slug}.js`, document.baseURI).href);

  const nav = el('nav', { className: 'app-nav', ariaLabel: `${trackInfo.title} lessons` });
  const main = el('main', { className: 'app-main', id: 'main', tabIndex: -1 });
  const home = hubHref({ from: track, target, artifactUrls: links.artifactUrls });
  const header = el('header', { className: 'app-header' }, [
    home ? el('a', { href: home, textContent: 'LLM Anatomy' }) : el('span', { textContent: 'LLM Anatomy' }),
    el('h1', { textContent: trackInfo.title }),
  ]);
  root.replaceChildren(el('div', { className: 'app' }, [header, nav, main]));

  const sectionTitle = (id) => trackInfo.sections.find((s) => s.id === id)?.title ?? id;
  const groups = bySection(graph.concepts, track);

  function renderNav(current) {
    nav.replaceChildren(...groups.map((g) => el('div', { className: 'nav-section' }, [
      el('h2', { textContent: sectionTitle(g.section) }),
      ...g.concepts.map((c) => {
        const link = el('a', { className: 'nav-link', href: `#${c.slug}`, textContent: c.title });
        if (c.slug === current) link.setAttribute('aria-current', 'page');
        return link;
      }),
    ])));
  }

  function learnedButton(slug) {
    const button = el('button', { type: 'button', className: 'learned-toggle' });
    const paint = (list) => {
      const on = list.includes(slug);
      button.setAttribute('aria-pressed', String(on));
      button.textContent = on ? 'Learned ✓' : 'Mark as learned';
    };
    paint(store.get(LEARNED_KEY, []));
    button.addEventListener('click', () => paint(toggleLearned(store, slug)));
    return button;
  }

  function prereqLinks(concept) {
    return concept.prereqs.map((p) => {
      const url = href(p);
      return url ? el('a', { href: url, textContent: index.get(p).title }) : el('span', { textContent: index.get(p).title });
    });
  }

  function renderIndex() {
    main.replaceChildren(el('section', { className: 'track-index' }, groups.flatMap((g) => [
      el('h2', { textContent: sectionTitle(g.section) }),
      ...g.concepts.map((c) => el('a', { className: 'concept-card', href: `#${c.slug}` }, [
        el('h3', { textContent: c.title }), el('p', { textContent: c.summary }),
      ])),
    ])));
  }

  let unmount = null;
  let generation = 0;

  const settle = () => {
    main.focus?.({ preventScroll: true });
    globalThis.scrollTo?.(0, 0);
  };

  const loadErrorCard = (concept) => el('div', { className: 'load-error', role: 'alert' }, [
    el('h2', { textContent: `“${concept.title}” didn't load` }),
    el('p', { textContent: 'Reload the page. If it keeps happening, please open an issue on GitHub with your browser name.' }),
  ]);

  async function show(route) {
    const mine = ++generation;
    safeUnmount(unmount);
    unmount = null;
    renderNav(route.slug);
    if (!route.slug) { renderIndex(); settle(); return; }
    const concept = index.get(route.slug);
    // Drop the previous lesson's DOM right away; the import may take a moment.
    main.replaceChildren(el('p', { className: 'boot', textContent: 'Loading…' }));
    const result = await loadConcept(route.slug, importer);
    if (mine !== generation) return; // a newer navigation won the race
    const container = el('article', { className: 'concept' });
    main.replaceChildren(container);
    if (result.status === 'ok') {
      const mounted = mountSafely(result.module, container, { concept, data, href, title });
      if (mounted.status === 'ok') {
        unmount = mounted.unmount;
        container.append(learnedButton(concept.slug));
      } else {
        container.replaceChildren(loadErrorCard(concept));
      }
    } else if (result.status === 'missing') {
      container.append(el('div', { className: 'coming-soon' }, [
        el('h2', { textContent: concept.title }), el('p', { textContent: concept.summary }),
        el('p', { textContent: 'This lesson is being built. Start with:' }), ...prereqLinks(concept),
      ]));
    } else {
      console.error(`Lesson "${route.slug}" failed to load`, result.error);
      container.append(loadErrorCard(concept));
    }
    settle();
  }

  startRouter({ knownSlugs: new Set(graph.concepts.filter((c) => c.track === track).map((c) => c.slug)), onRoute: show });
}
