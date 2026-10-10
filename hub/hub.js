// The home page: the course in one screen, built from shared/concepts.json. A hero, three track cards whose
// covers play the course's story (built, trained, served) in turn, and a map of how the lessons build on each other.
import { bySection } from '../shared/concepts.js';
import { conceptHref } from '../shared/links.js';
import { loadJSON } from '../shared/data.js';
import { safeStorage, LEARNED_KEY } from '../shared/storage.js';
import { el } from '../shared/ui/dom.js';
import { mountThemeToggle } from '../shared/ui/theme-toggle.js';
import { coverFigure, mountStory, prefersReducedMotion } from './covers.js';
import { mapLayout } from './map-layout.js';
import { mountMap } from './map.js';
import { nextLesson, learnedCount } from './progress.js';

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
        lessons: group.concepts.map((c) => ({ slug: c.slug, title: c.title, href: href(track.id, c.slug), learned: done.has(c.slug) })),
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

const lessonItem = (lesson) => {
  const li = el('li', { className: lesson.learned ? 'is-learned' : '' }, [el('a', { href: lesson.href, textContent: lesson.title })]);
  li.dataset.slug = lesson.slug;
  return li;
};

function trackCard(track, position) {
  const showSectionTitles = track.sections.length > 1;
  const cover = coverFigure(track.id);
  const card = el('section', { className: 'hub-track', ariaLabel: track.title }, [
    cover.figure,
    el('p', { className: 'eyebrow', textContent: `Part ${position} · ${countLabel(track)}` }),
    el('h2', {}, [el('a', { href: track.href, textContent: track.title })]),
    el('p', { className: 'hub-blurb', textContent: track.blurb }),
    ...track.sections.map((section) => el('div', { className: 'hub-section' }, [
      ...(showSectionTitles ? [el('h3', { textContent: section.title })] : []),
      el('ol', { start: section.start }, section.lessons.map(lessonItem)),
    ])),
  ]);
  card.dataset.track = track.id;
  return { card, scene: { draw: cover.draw, onProgress: (p) => card.style.setProperty('--act-p', String(p)) } };
}

// "How an LLM is built, trained and served", each verb in the color of the track that covers it.
const VERBS = ['built', 'trained', 'served'];
const word = (text, track) => {
  const span = el('span', { className: 'hub-word', textContent: text });
  if (track) span.dataset.track = track;
  return span;
};
const heading = (tracks) => el('h1', {}, [
  'How an LLM is ', word(VERBS[0], tracks[0]?.id), ', ', word(VERBS[1], tracks[1]?.id), ' and ', word(VERBS[2], tracks[2]?.id),
]);

function progressLine(graph, learned, total) {
  const done = learnedCount(graph, learned);
  if (done === 0) return null;
  const next = nextLesson(graph, learned);
  return el('p', { className: 'hub-progress' }, [
    `${done} of ${total} learned. `,
    ...(next ? ['Next for you: ', el('a', { href: href(next.track, next.slug), textContent: next.title })] : ['That is the whole course.']),
  ]);
}

function hero({ graph, tracks, learned, total }) {
  const start = el('a', { className: 'hub-start', href: tracks[0].href, textContent: `Start with ${tracks[0].title}` });
  return el('div', { className: 'hub-hero' }, [
    el('p', { className: 'eyebrow', textContent: `${total} lessons · ${tracks.length} tracks · open source` }),
    heading(tracks),
    el('p', { className: 'hub-intro', textContent: 'The models of 2026, from one token\'s path through the network to a cluster of GPUs answering millions of people. Each lesson is an animation you step through and a toy whose numbers you can check by hand.' }),
    el('div', { className: 'hub-actions' }, [start, progressLine(graph, learned, total)].filter(Boolean)),
  ]);
}

export async function mountHub({ root }) {
  const graph = await loadJSON('./concepts.json');
  const learned = safeStorage().get(LEARNED_KEY, []);
  const tracks = hubModel(graph, learned);
  const total = tracks.reduce((sum, t) => sum + t.count, 0);
  const cards = tracks.map((track, i) => trackCard(track, i + 1));
  const grid = el('div', { className: 'hub-tracks' }, cards.map((c) => c.card));
  const reduced = prefersReducedMotion();
  const replay = el('button', { type: 'button', className: 'hub-replay', textContent: 'Replay the story', hidden: reduced });
  const map = mountMap({
    layout: mapLayout(graph, learned, href),
    onFocus: (slug, chain) => {
      const on = new Set(chain);
      root.querySelectorAll('.hub-section li[data-slug]').forEach((li) => li.classList.toggle('is-path', on.has(li.dataset.slug)));
      grid.classList.toggle('has-focus', Boolean(slug));
    },
  });
  root.replaceChildren(
    el('header', { className: 'app-header' }, [
      el('span', { textContent: 'LLM Anatomy' }),
      el('div', { className: 'app-header-actions' }, [mountThemeToggle()]),
    ]),
    el('main', { className: 'hub', id: 'main' }, [
      hero({ graph, tracks, learned, total }),
      grid,
      el('div', { className: 'hub-replay-row' }, [replay]),
      map.section,
      el('footer', { className: 'hub-foot' }, [
        el('p', {}, [
          'Every real-world number is sourced and dated. Code MIT, lessons CC BY 4.0. ',
          el('a', { href: 'https://github.com/arbaaz-abz/llm-anatomy', textContent: 'Source on GitHub' }),
        ]),
      ]),
    ]),
  );
  // The grid carries the story's state (playing, done, still), so tests and tools read it through the DOM.
  const story = mountStory(cards.map((c) => c.scene), { reduced, onState: (state) => { grid.dataset.story = state; } });
  replay.addEventListener('click', () => story.replay());
}
