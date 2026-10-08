// cluster-topology toy, "Put a cut on a link": mount(el, ctx) → destroy. One state object, one render; every string it
// prints comes from toyView (toy-view.js), so the page shows exactly what the tests check.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountChoice } from '@shared/ui/choice.js';
import { createToyState } from '@shared/ui/toy-state.js';
import { INITIAL_STATE, SYSTEM_CHIPS, CUT_OPTIONS, TOKENS_NOTE, toyView, tryThis, stopsFor, snapDegree, whereOptions, systemFromData } from './toy-view.js';
import { TOKEN_STOPS, int } from './format.js';
import { LANES } from './numbers.js';

const LANE_BOX = Object.freeze({ x: 0, y: 4, w: 52 + LANES.trackPx });
const DEGREE_UNIT = Object.freeze({ tensor: 'GPUs', pipeline: 'stages', data: 'replicas' });

const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};
const line = (label, name) => { const out = output(name); return { out, row: el('p', { className: 'toy-note' }, [`${label}: `, out]) }; };

function tryThisList(data) {
  const items = tryThis(data).map(({ text, insight }) => el('li', {}, [`${text} → `, el('strong', { textContent: `Insight: ${insight}` })]));
  return [el('h4', { textContent: 'Try this' }), el('ol', { className: 'try-this' }, items)];
}

function buildDom(host, data) {
  const pre = el('pre', { className: 'check-work', ariaLive: 'polite' });
  pre.dataset.readout = 'check-work';
  const rows = {
    ratio: line('Comm as % of compute', 'ratio'), hide: line('Can it hide?', 'hide'), need: line('Link needed for expert traffic to hide', 'link-needed'),
    bytes: line('Bytes per GPU behind the ratio, per layer per step', 'bytes'),
  };
  const refs = {
    chips: el('div'), cut: el('div'), degree: el('div'), degreeWrap: el('div'), where: el('div'), tokens: el('div'), tokensSlider: el('div'), tokensNote: el('p', { className: 'toy-note', textContent: TOKENS_NOTE }),
    system: output('system-line'), cutLine: output('cut-line'), laneNote: output('lane-note'), rows, pre,
    lanes: G.svgEl('svg', { role: 'group', 'aria-label': 'communication against compute' }),
  };
  refs.degreeWrap.append(refs.degree);
  refs.tokens.append(refs.tokensSlider, refs.tokensNote);
  host.append(
    refs.chips, el('p', { className: 'toy-note' }, [refs.system]), el('div', { className: 'toy-controls' }, [refs.cut, refs.degreeWrap, refs.where, refs.tokens]),
    el('p', { className: 'toy-note' }, [refs.cutLine]), rows.ratio.row, el('div', { className: 'scroll-x' }, [refs.lanes]), el('p', { className: 'toy-note' }, [refs.laneNote]),
    rows.hide.row, rows.need.row, rows.bytes.row, el('h4', { textContent: 'Check my work' }), pre, ...tryThisList(data),
  );
  return refs;
}

function paintLanes(svg, { lanes, state }) {
  svg.replaceChildren();
  const compute = [{ from: 0, to: lanes.compute, kind: 'compute' }];
  const comm = [{ from: 0, to: lanes.comm, kind: 'comm', label: null }];
  G.laneTimeline(svg, { ...LANE_BOX, scale: LANES.pxPerUnit, cap: lanes.cap, label: `${state.cut} communication`, lanes: [{ label: 'compute', segments: compute }, { label: 'comm', segments: comm }] });
  G.fitViewBox(svg, 4);
}

function paint(refs, view) {
  const { rows } = refs;
  rows.ratio.out.textContent = view.ratioText;
  refs.laneNote.textContent = view.lanes.note;
  rows.hide.out.textContent = view.hide;
  rows.need.out.textContent = view.linkNeeded ?? '';
  rows.need.row.hidden = view.linkNeeded === null;
  rows.bytes.out.textContent = view.bytes ?? '';
  rows.bytes.row.hidden = view.bytes === null;
  refs.system.textContent = view.systemLine;
  refs.cutLine.textContent = view.cutLine;
  refs.pre.textContent = view.check;
  refs.degreeWrap.hidden = view.state.cut === 'expert';
  refs.tokens.hidden = view.state.cut !== 'data';
  paintLanes(refs.lanes, view);
}

export function mount(host, ctx) {
  const data = ctx?.data;
  const refs = buildDom(host, data);
  let toy = null;
  let degree = null;
  let where = null;
  const sysOf = (id) => systemFromData(id, data);
  const mountDegree = (cut, value) => {
    degree?.destroy();
    degree = mountSlider(refs.degree, {
      id: 'degree', label: 'Degree', values: stopsFor(cut).length ? stopsFor(cut) : [INITIAL_STATE.degree], value, unit: DEGREE_UNIT[cut] ?? '', format: int,
      onInput: (v) => change({ degree: v }),
    });
  };
  // Every control goes through here: re-disable "inside" for the next state first (it may move the choice to "network").
  function change(patch) {
    const next = { ...toy.get(), ...patch };
    where.update(whereOptions(next, sysOf(next.system)));
    toy.set(patch);
  }
  const first = toyView(INITIAL_STATE, data);
  const controls = [
    mountChoice(refs.chips, { id: 'system', label: 'System', variant: 'chips', value: INITIAL_STATE.system, options: SYSTEM_CHIPS, onChange: (v) => change({ system: v }) }),
    mountChoice(refs.cut, {
      id: 'cut', label: 'Parallelism', value: INITIAL_STATE.cut, options: CUT_OPTIONS,
      onChange: (v) => { const d = snapDegree(v, toy.get().degree); mountDegree(v, d); change({ cut: v, degree: d }); },
    }),
    mountSlider(refs.tokensSlider, {
      id: 'tokens', label: 'Tokens per replica per step', values: TOKEN_STOPS, value: INITIAL_STATE.tokens, format: int, onInput: (v) => change({ tokens: v }),
    }),
  ];
  where = mountChoice(refs.where, { id: 'where', label: 'Runs on', options: first.where, value: INITIAL_STATE.where, onChange: (v) => toy.set({ where: v }) });
  mountDegree(INITIAL_STATE.cut, INITIAL_STATE.degree);
  toy = createToyState(INITIAL_STATE, (s) => paint(refs, toyView(s, data)));
  return () => {
    toy.destroy();
    [...controls, where, degree].forEach((c) => c.destroy());
    host.replaceChildren();
  };
}
