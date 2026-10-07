export function formatFact(fact, format = String) {
  const unit = fact.unit ? ` ${fact.unit}` : '';
  const text = Array.isArray(fact.value)
    ? `${format(fact.value[0])}–${format(fact.value[1])}${unit}`
    : `${format(fact.value)}${unit}`;
  return { text, href: fact.source_url, reported: fact.confidence === 'reported', note: fact.note ?? '' };
}

export function renderFact(el, fact, format = String) {
  const { text, href, reported, note } = formatFact(fact, format);
  el.classList.add('fact');
  el.textContent = text;
  if (note) el.title = note;
  const link = document.createElement('a');
  link.className = 'fact-source';
  link.href = href;
  link.target = '_blank';
  link.rel = 'noopener';
  link.textContent = 'source';
  el.append(' ', link);
  if (reported) {
    const chip = document.createElement('span');
    chip.className = 'fact-reported';
    chip.textContent = 'reported';
    el.append(' ', chip);
  }
}
