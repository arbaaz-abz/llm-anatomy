import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mountPresetButtons } from '../shared/ui/preset-buttons.js';
import { FakeEl, withFakeDom } from './support/fake-dom.js';

const OPTIONS = [{ value: 64, label: '64' }, { value: 512, label: '512' }, { value: 4096, label: '4,096' }];
const mount = (onPick = () => {}) => {
  const root = new FakeEl('div');
  const handle = mountPresetButtons(root, { id: 'presets', label: 'Tokens', options: OPTIONS, onPick });
  return { root, handle, buttons: root.children.slice(1) };
};

test('mountPresetButtons draws a labelled group of choice-option buttons, with no pressed state', () => {
  withFakeDom(() => {
    const { root, buttons } = mount();
    assert.equal(root.id, 'presets');
    assert.equal(root.className, 'choice choice--chips');
    assert.equal(root.getAttribute('role'), 'group');
    assert.equal(root.getAttribute('aria-labelledby'), 'presets-label');
    assert.equal(root.children[0].id, 'presets-label');
    assert.equal(root.children[0].textContent, 'Tokens');
    assert.deepEqual(buttons.map((b) => b.textContent), ['64', '512', '4,096']);
    assert.deepEqual(buttons.map((b) => b.dataset.value), ['64', '512', '4096']);
    for (const b of buttons) {
      assert.equal(b.className, 'choice-option');
      assert.equal(b.type, 'button');
      assert.equal(b.getAttribute('aria-pressed'), null);
    }
  });
});

test('a click calls onPick with the option value and leaves every button unpressed', () => {
  withFakeDom(() => {
    const picked = [];
    const { buttons } = mount((v) => picked.push(v));
    buttons[1].dispatch('click');
    buttons[2].dispatch('click');
    assert.deepEqual(picked, [512, 4096]);
    for (const b of buttons) assert.equal(b.getAttribute('aria-pressed'), null);
  });
});

test('a click outside the buttons (the caption) picks nothing', () => {
  withFakeDom(() => {
    const picked = [];
    const { root } = mount((v) => picked.push(v));
    root.children[0].dispatch('click');
    assert.deepEqual(picked, []);
  });
});

test('destroy removes the click listener and the buttons', () => {
  withFakeDom(() => {
    const picked = [];
    const { root, handle, buttons } = mount((v) => picked.push(v));
    handle.destroy();
    assert.equal(root.children.length, 0);
    assert.equal(root.listeners.click.size, 0);
    buttons[0].dispatch('click');
    assert.deepEqual(picked, []);
  });
});

test('empty options throw', () => {
  withFakeDom(() => {
    assert.throws(() => mountPresetButtons(new FakeEl('div'), { id: 'p', label: 'x', options: [], onPick() {} }), /non-empty/);
  });
});
