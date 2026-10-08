import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mountStageSelect, keyStep } from '../shared/ui/stage-select.js';
import { FakeEl, withFakeDom } from './support/fake-dom.js';

const ITEMS = [{ value: 'a', label: 'Alpha' }, { value: 'b', label: 'Beta' }, { value: 'c', label: 'Gamma' }];
const setup = (value = 'a') => {
  const parent = new FakeEl('svg');
  const picked = [];
  let handle;
  handle = mountStageSelect(parent, { items: ITEMS, value, onSelect: (v) => { picked.push(v); handle.sync(v); } });
  return { parent, picked, handle, nodes: parent.children };
};

test('keyStep: arrows move by one and stop at the ends; Home / End jump; other keys are null', () => {
  assert.equal(keyStep('ArrowRight', 2, 7), 3);
  assert.equal(keyStep('ArrowDown', 2, 7), 3);
  assert.equal(keyStep('ArrowLeft', 2, 7), 1);
  assert.equal(keyStep('ArrowUp', 0, 7), 0);
  assert.equal(keyStep('ArrowRight', 6, 7), 6);
  assert.equal(keyStep('Home', 4, 7), 0);
  assert.equal(keyStep('End', 1, 7), 6);
  assert.equal(keyStep('a', 3, 7), null);
});

test('one button-role group per item, with a roving tab stop on the selected item', () => {
  withFakeDom(() => {
    const { nodes } = setup('b');
    assert.deepEqual(nodes.map((n) => n.getAttribute('role')), ['button', 'button', 'button']);
    assert.deepEqual(nodes.map((n) => n.getAttribute('aria-label')), ['Alpha', 'Beta', 'Gamma']);
    assert.deepEqual(nodes.map((n) => n.dataset.value), ['a', 'b', 'c']);
    assert.deepEqual(nodes.map((n) => n.getAttribute('aria-pressed')), ['false', 'true', 'false']);
    assert.deepEqual(nodes.map((n) => n.getAttribute('tabindex')), ['-1', '0', '-1']);
    assert.ok(nodes.every((n) => n.hasClass('stage-item')));
  });
});

test('a click selects the item', () => {
  withFakeDom(() => {
    const { nodes, picked } = setup();
    nodes[2].dispatch('click');
    assert.deepEqual(picked, ['c']);
    assert.deepEqual(nodes.map((n) => n.getAttribute('tabindex')), ['-1', '-1', '0']);
  });
});

test('Enter and Space select the focused item without moving', () => {
  withFakeDom(() => {
    const { nodes, picked } = setup();
    const enter = nodes[1].dispatch('keydown', { key: 'Enter' });
    const space = nodes[2].dispatch('keydown', { key: ' ' });
    assert.deepEqual(picked, ['b', 'c']);
    assert.ok(enter.defaultPrevented && space.defaultPrevented);
  });
});

test('arrow keys move the selection and the focus together; the ends hold; Home / End jump', () => {
  withFakeDom(() => {
    const { nodes, picked } = setup('a');
    nodes[0].dispatch('keydown', { key: 'ArrowRight' });
    assert.equal(document.activeElement, nodes[1]);
    nodes[1].dispatch('keydown', { key: 'ArrowDown' });
    assert.equal(document.activeElement, nodes[2]);
    nodes[2].dispatch('keydown', { key: 'ArrowRight' });
    assert.equal(document.activeElement, nodes[2]);
    nodes[2].dispatch('keydown', { key: 'Home' });
    assert.equal(document.activeElement, nodes[0]);
    nodes[0].dispatch('keydown', { key: 'End' });
    assert.equal(document.activeElement, nodes[2]);
    assert.deepEqual(picked, ['b', 'c', 'c', 'a', 'c']);
  });
});

test('an unrelated key does nothing and is not prevented', () => {
  withFakeDom(() => {
    const { nodes, picked } = setup();
    const event = nodes[0].dispatch('keydown', { key: 'x' });
    assert.deepEqual(picked, []);
    assert.equal(event.defaultPrevented, false);
  });
});

test('the item groups persist across repaints: sync changes the state, not the nodes, so focus survives', () => {
  withFakeDom(() => {
    const { nodes, handle } = setup('a');
    nodes[0].focus();
    handle.sync('b');
    assert.equal(document.activeElement, nodes[0]);
    assert.equal(handle.node('b'), nodes[1]);
    assert.equal(handle.node('c'), nodes[2]);
  });
});

test('destroy removes the listeners and the groups', () => {
  withFakeDom(() => {
    const { parent, nodes, picked, handle } = setup();
    const first = nodes[0];
    handle.destroy();
    assert.equal(parent.children.length, 0);
    assert.equal(parent.listeners.click.size, 0);
    assert.equal(parent.listeners.keydown.size, 0);
    first.dispatch('click');
    assert.deepEqual(picked, []);
  });
});

test('empty items throw', () => {
  withFakeDom(() => {
    assert.throws(() => mountStageSelect(new FakeEl('svg'), { items: [], value: 'a', onSelect() {} }), /non-empty/);
  });
});
