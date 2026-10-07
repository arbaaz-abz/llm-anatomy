import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initialStepperState, stepperReducer, easeInOut } from '../shared/ui/stepper.js';

const s0 = initialStepperState(4);
const run = (state, ...actions) => actions.reduce(stepperReducer, state);

test('next/prev move within bounds', () => {
  assert.equal(run(s0, { type: 'next' }).index, 1);
  assert.equal(run(s0, { type: 'prev' }).index, 0);
  assert.equal(run(s0, { type: 'next' }, { type: 'next' }, { type: 'next' }, { type: 'next' }).index, 3);
});

test('seek clamps out-of-range and non-numeric input', () => {
  assert.equal(run(s0, { type: 'seek', index: 2 }).index, 2);
  assert.equal(run(s0, { type: 'seek', index: -5 }).index, 0);
  assert.equal(run(s0, { type: 'seek', index: 99 }).index, 3);
  assert.equal(run(s0, { type: 'seek', index: Number.NaN }).index, 0);
  assert.equal(run(s0, { type: 'seek', index: 1.7 }).index, 1);
});

test('play from the last step restarts; stepDone advances then stops at the end', () => {
  const atEnd = run(s0, { type: 'seek', index: 3 });
  assert.deepEqual(run(atEnd, { type: 'play' }), { ...s0, playing: true, index: 0 });
  const playing = run(s0, { type: 'play' });
  const end = run(playing, { type: 'stepDone' }, { type: 'stepDone' }, { type: 'stepDone' }, { type: 'stepDone' });
  assert.equal(end.index, 3);
  assert.equal(end.playing, false);
});

test('stepDone is ignored while paused; manual moves pause playback', () => {
  assert.equal(run(s0, { type: 'stepDone' }).index, 0);
  assert.equal(run(s0, { type: 'play' }, { type: 'prev' }).playing, false);
  assert.equal(run(s0, { type: 'play' }, { type: 'seek', index: 2 }).playing, false);
  assert.equal(run(s0, { type: 'toggle' }).playing, true);
  assert.equal(run(s0, { type: 'toggle' }, { type: 'toggle' }).playing, false);
});

test('speed is clamped to the offered values; unknown actions throw', () => {
  assert.equal(run(s0, { type: 'speed', speed: 2 }).speed, 2);
  assert.equal(run(s0, { type: 'speed', speed: 1.25 }).speed, 1.25);
  assert.equal(run(s0, { type: 'speed', speed: 0.75 }).speed, 0.75);
  assert.equal(run(s0, { type: 'speed', speed: 7 }).speed, 1);
  assert.throws(() => stepperReducer(s0, { type: 'explode' }), /unknown/);
});

test('reducer never mutates its input', () => {
  const frozen = Object.freeze({ ...s0 });
  assert.doesNotThrow(() => run(frozen, { type: 'next' }, { type: 'play' }, { type: 'speed', speed: 0.5 }));
});

test('easeInOut is monotonic from 0 to 1', () => {
  assert.equal(easeInOut(0), 0);
  assert.equal(easeInOut(1), 1);
  assert.equal(easeInOut(0.5), 0.5);
  assert.ok(easeInOut(0.25) < easeInOut(0.75));
});

