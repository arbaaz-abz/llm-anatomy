// The one DOM-builder helper: el('p', { className: 'x' }, [child, 'text']).
export const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
