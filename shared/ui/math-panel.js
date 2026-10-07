// Equations are author-written; the only trusted command is \htmlClass, used to color terms to match diagrams.
export const allowHtmlClassOnly = (context) => context.command === '\\htmlClass';

export function renderTex(tex, target, { display = true } = {}) {
  const katex = globalThis.katex;
  if (katex?.render) {
    try {
      katex.render(tex, target, { displayMode: display, throwOnError: false, trust: allowHtmlClassOnly, strict: false });
      return 'katex';
    } catch (error) {
      console.error('KaTeX could not render an equation; showing raw TeX instead.', error);
    }
  }
  target.textContent = tex;
  target.classList.add('tex-fallback');
  return 'fallback';
}

export function mountMathPanel(root, { summary = 'Show me the math', blocks }) {
  root.innerHTML = '<details class="math-panel"><summary></summary><div class="math-body"></div></details>';
  root.querySelector('summary').textContent = summary;
  const body = root.querySelector('.math-body');
  for (const { tex, note } of blocks) {
    const eq = document.createElement('div');
    renderTex(tex, eq);
    body.append(eq);
    if (note) {
      const p = document.createElement('p');
      p.textContent = note;
      body.append(p);
    }
  }
  return { destroy() { root.innerHTML = ''; } };
}
