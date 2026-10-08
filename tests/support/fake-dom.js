// A tiny stand-in DOM for unit tests of the shared UI helpers (the repo has no jsdom): elements with attributes,
// children, class lookup, `closest`, `contains`, listeners, `focus` and `dispatch`. Enough for click / keydown helpers.
export class FakeEl {
  constructor(tag) {
    this.tag = tag; this.attrs = {}; this.dataset = {}; this.children = []; this.parent = null;
    this.listeners = {}; this.className = ''; this.textContent = ''; this.id = '';
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  append(...kids) { kids.forEach((k) => { k.parent = this; this.children.push(k); }); }
  replaceChildren(...kids) { this.children.forEach((k) => { k.parent = null; }); this.children = []; this.append(...kids); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; }
  hasClass(name) { return `${this.className} ${this.attrs.class ?? ''}`.split(/\s+/).includes(name); }
  closest(selector) {
    const name = selector.replace(/^\./, '');
    for (let n = this; n; n = n.parent) if (n.hasClass(name)) return n;
    return null;
  }
  contains(other) { for (let n = other; n; n = n.parent) if (n === this) return true; return false; }
  addEventListener(type, fn) { (this.listeners[type] ??= new Set()).add(fn); }
  removeEventListener(type, fn) { this.listeners[type]?.delete(fn); }
  focus() { globalThis.document.activeElement = this; }
  // Bubbles from this element up through its ancestors, like a DOM event.
  dispatch(type, props = {}) {
    const event = { type, target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...props };
    for (let n = this; n; n = n.parent) [...(n.listeners[type] ?? [])].forEach((fn) => fn(event));
    return event;
  }
}

// Installs a fake `document` for the run; returns what `run` returns.
export function withFakeDom(run) {
  const saved = globalThis.document;
  globalThis.document = { activeElement: null, createElement: (tag) => new FakeEl(tag), createElementNS: (_ns, tag) => new FakeEl(tag) };
  try { return run(); } finally { globalThis.document = saved; }
}
