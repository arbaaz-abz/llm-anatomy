// The home page: the course in one screen, three tracks with their lessons, built from shared/concepts.json.
import { bySection } from '../shared/concepts.js';
import { conceptHref } from '../shared/links.js';
import { loadJSON } from '../shared/data.js';
import { safeStorage, LEARNED_KEY } from '../shared/storage.js';
import { el } from '../shared/ui/dom.js';
import { mountThemeToggle } from '../shared/ui/theme-toggle.js';

export const TRACK_BLURBS = Object.freeze({
  architecture: 'What is inside the model: attention, positions, the KV cache, experts, and how a token gets picked.',
  training: 'How the weights get there: pretraining through RL and distillation, then the GPUs and clusters that do it.',
  serving: 'How it answers millions of people: batching, caching, guessing ahead, fewer bits, and what a token costs.',
});

const href = (track, slug) => conceptHref({ from: 'hub', track, slug, target: 'pages', artifactUrls: {} });

// Pure: the tracks in course order, each with its sections, lessons and how many the reader marked learned.
export function hubModel(graph, learned = []) {
  const done = new Set(Array.isArray(learned) ? learned : []);
  return graph.tracks.map((track) => {
    let numbered = 0; // lesson numbers run on across a track's sections, as one list
    const sections = bySection(graph.concepts, track.id).map((group) => {
      const start = numbered + 1;
      numbered += group.concepts.length;
      return {
        title: track.sections.find((s) => s.id === group.section)?.title ?? group.section,
        start,
        lessons: group.concepts.map((c) => ({ slug: c.slug, title: c.title, href: href(track.id, c.slug) })),
      };
    });
    const slugs = sections.flatMap((s) => s.lessons.map((l) => l.slug));
    return {
      id: track.id,
      title: track.title,
      blurb: TRACK_BLURBS[track.id] ?? '',
      href: href(track.id, ''),
      count: slugs.length,
      learned: slugs.filter((slug) => done.has(slug)).length,
      sections,
    };
  });
}

export const countLabel = ({ count, learned }) =>
  `${count} lesson${count === 1 ? '' : 's'}${learned > 0 ? ` · ${learned} learned` : ''}`;

function trackCard(track, position) {
  const showSectionTitles = track.sections.length > 1;
  return el('section', { className: 'hub-track', ariaLabel: track.title }, [
    el('p', { className: 'eyebrow', textContent: `Part ${position} · ${countLabel(track)}` }),
    el('h2', {}, [el('a', { href: track.href, textContent: track.title })]),
    el('p', { className: 'hub-blurb', textContent: track.blurb }),
    ...track.sections.map((section) => el('div', { className: 'hub-section' }, [
      ...(showSectionTitles ? [el('h3', { textContent: section.title })] : []),
      el('ol', { start: section.start }, section.lessons.map((lesson) => el('li', {}, [el('a', { href: lesson.href, textContent: lesson.title })]))),
    ])),
  ]);
}

export async function mountHub({ root }) {
  const graph = await loadJSON('./concepts.json');
  const tracks = hubModel(graph, safeStorage().get(LEARNED_KEY, []));
  const total = tracks.reduce((sum, t) => sum + t.count, 0);
  const cards = tracks.map((track, i) => {
    const card = trackCard(track, i + 1);
    card.dataset.track = track.id;
    return card;
  });
  root.replaceChildren(
    el('header', { className: 'app-header' }, [
      el('span', { textContent: 'LLM Anatomy' }),
      el('div', { className: 'app-header-actions' }, [mountThemeToggle()]),
    ]),
    el('main', { className: 'hub', id: 'main' }, [
      el('div', { className: 'hub-hero' }, [
        el('h1', { textContent: 'How an LLM is built, trained and served' }),
        el('p', { textContent: `${total} short lessons on the models of 2026. Each one is an animation you step through and a toy whose numbers you can check by hand.` }),
        el('a', { className: 'hub-start', href: tracks[0].href, textContent: `Start with ${tracks[0].title}` }),
      ]),
      el('div', { className: 'hub-tracks' }, cards),
      el('footer', { className: 'hub-foot' }, [
        el('p', {}, [
          'Every real-world number is sourced and dated. Code MIT, lessons CC BY 4.0. ',
          el('a', { href: 'https://github.com/arbaaz-abz/llm-anatomy', textContent: 'Source on GitHub' }),
        ]),
      ]),
    ]),
  );
}
