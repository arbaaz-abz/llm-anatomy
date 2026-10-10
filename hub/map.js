// The course map on the page: lanes, stations (each a link to its lesson) and "read this first" curves from
// hub/map-layout.js. Hovering or focusing a station lights its whole chain and says it in words.
import { svgEl } from '../shared/glyphs.js';
import { el } from '../shared/ui/dom.js';
import { MAP, chainOf, captionFor } from './map-layout.js';

const SVG_TEXT_ID = 'hub-map-title';
export const MAP_HINT = 'Hover or tab to a lesson to see what it builds on.';

function drawLane(svg, lane) {
  const g = svgEl('g', { class: 'map-lane', 'data-track': lane.id }, svg);
  svgEl('rect', { class: 'map-lane-bg', x: 0, y: lane.y, width: '100%', height: lane.height, rx: 6 }, g);
  const link = svgEl('a', { class: 'map-lane-title', href: lane.href, 'aria-label': `${lane.title} track` }, g);
  svgEl('rect', { class: 'map-lane-swatch', x: 14, y: lane.y + lane.height / 2 - 5, width: 10, height: 10, rx: 2 }, link);
  const label = svgEl('text', { x: 30, y: lane.y + lane.height / 2, 'dominant-baseline': 'central' }, link);
  label.textContent = lane.title.toUpperCase(); // in JS: text-transform is unreliable on SVG text
}

function drawEdge(svg, e) {
  return svgEl('path', { class: `map-edge${e.cross ? ' map-edge--cross' : ''}`, d: e.d, 'data-from': e.from, 'data-to': e.to, 'data-track': e.track }, svg);
}

function drawStation(svg, s) {
  const a = svgEl('a', { class: `map-station${s.learned ? ' is-learned' : ''}`, href: s.href, 'data-slug': s.slug, 'data-track': s.track, 'aria-label': `${s.title} (${s.trackTitle} ${s.number})` }, svg);
  svgEl('title', {}, a).textContent = s.title;
  svgEl('circle', { class: 'map-ring', cx: s.x, cy: s.y, r: MAP.r + 4 }, a);
  svgEl('circle', { class: 'map-dot', cx: s.x, cy: s.y, r: MAP.r }, a);
  const num = svgEl('text', { class: 'map-num', x: s.x, y: s.y, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, a);
  num.textContent = String(s.number);
  return a;
}

// Returns the section and setFocus(slug | null); `onFocus(slug, chainSlugs)` lets the cards follow along.
export function mountMap({ layout, onFocus = () => {} }) {
  const svg = svgEl('svg', { class: 'hub-map-svg', viewBox: `0 0 ${layout.width} ${layout.height}`, role: 'group', 'aria-labelledby': SVG_TEXT_ID });
  svg.style.minWidth = `${Math.round(layout.width * 0.85)}px`;
  layout.lanes.forEach((lane) => drawLane(svg, lane));
  const edges = layout.edges.map((e) => drawEdge(svg, e));
  const stations = layout.stations.map((s) => drawStation(svg, s));
  const caption = el('p', { className: 'hub-map-caption', textContent: MAP_HINT });
  caption.setAttribute('aria-live', 'polite');

  const setFocus = (slug) => {
    const chain = slug ? new Set([...chainOf(layout, slug).map((s) => s.slug), slug]) : null;
    svg.classList.toggle('has-focus', Boolean(slug));
    stations.forEach((node) => node.classList.toggle('is-path', Boolean(chain?.has(node.dataset.slug))));
    edges.forEach((node) => node.classList.toggle('is-path', Boolean(chain?.has(node.dataset.from) && chain?.has(node.dataset.to))));
    caption.textContent = slug ? captionFor(layout, slug) : MAP_HINT;
    onFocus(slug, chain ? [...chain] : []);
  };
  stations.forEach((node) => {
    const slug = node.dataset.slug;
    node.addEventListener('pointerenter', () => setFocus(slug));
    node.addEventListener('focus', () => setFocus(slug));
    node.addEventListener('pointerleave', () => setFocus(null));
    node.addEventListener('blur', () => setFocus(null));
  });

  const cross = layout.edges.filter((e) => e.cross).length;
  const section = el('section', { className: 'hub-map' }, [
    el('h2', { id: SVG_TEXT_ID, textContent: 'How the lessons build on each other' }),
    el('p', { className: 'hub-map-intro', textContent: `Each dot is a lesson, numbered as in the lists above, placed by how many lessons come before it. A line means "read this first"; ${cross} of them cross tracks.` }),
    el('div', { className: 'scroll-x hub-map-scroll' }, [svg]),
    caption,
  ]);
  return { section, setFocus };
}
