import test from "node:test";
import assert from "node:assert/strict";
import { stepMotionLabBlink } from "../src/motionLab/blinkState.ts";

const durations = {
  centerMs: 140,
  closeMs: 90,
  openMs: 130,
  settleMs: 100,
  intervalMin: 2,
  intervalMax: 10,
};

function createState(wait = 0) {
  return { blinkWait: wait, blinkPhase: "idle", blinkT: 0 };
}

test("coordinated blink returns to center before RIFE owns the eye", () => {
  const state = createState();
  let frame = stepMotionLabBlink(state, 0, true, 1, durations, () => 0);
  assert.equal(frame.phase, "centering");
  assert.equal(frame.rifeOwnsEye, false);
  assert.equal(frame.dynamicEyeAlpha, 1);

  frame = stepMotionLabBlink(state, 0.077, true, 1, durations, () => 0);
  assert.equal(frame.phase, "centering");
  assert.ok(frame.dynamicEyeAlpha > 0 && frame.dynamicEyeAlpha < 1);

  frame = stepMotionLabBlink(state, 0.0315, true, 1, durations, () => 0);
  assert.equal(frame.phase, "centering");
  assert.ok(frame.dynamicEyeAlpha > 0 && frame.dynamicEyeAlpha < 1);

  frame = stepMotionLabBlink(state, 0.0315, true, 1, durations, () => 0);
  assert.equal(frame.phase, "closing");
  assert.equal(frame.rifeProgress, 0);
  assert.equal(frame.rifeOwnsEye, true);
  assert.equal(frame.dynamicEyeAlpha, 0);

  frame = stepMotionLabBlink(state, 0.045, true, 1, durations, () => 0);
  assert.equal(frame.phase, "closing");
  assert.equal(frame.rifeProgress, 0.5);

  frame = stepMotionLabBlink(state, 0.045, true, 1, durations, () => 0);
  assert.equal(frame.phase, "opening");
  assert.equal(frame.rifeProgress, 1);

  frame = stepMotionLabBlink(state, 0.13, true, 1, durations, () => 0);
  assert.equal(frame.phase, "settling");
  assert.equal(frame.rifeProgress, 0);
  assert.equal(frame.rifeOwnsEye, false);
  assert.equal(frame.dynamicEyeAlpha, 0);

  frame = stepMotionLabBlink(state, 0.05, true, 1, durations, () => 0);
  assert.equal(frame.phase, "settling");
  assert.equal(frame.dynamicEyeAlpha, 0.5);

  frame = stepMotionLabBlink(state, 0.05, true, 1, durations, () => 0);
  assert.equal(frame.phase, "idle");
  assert.equal(frame.dynamicEyeAlpha, 1);
  assert.equal(frame.sequenceActive, false);
  assert.equal(state.blinkWait, 2);
});

test("a delayed frame can cross the whole blink without leaving a stale phase", () => {
  const state = createState();
  const sequenceMs = durations.centerMs + durations.closeMs + durations.openMs + durations.settleMs;
  const frame = stepMotionLabBlink(
    state,
    (sequenceMs + 500) / 1000,
    true,
    2,
    durations,
    () => 0.25,
  );
  assert.equal(frame.phase, "idle");
  assert.equal(frame.rifeOwnsEye, false);
  assert.equal(state.blinkWait, 1.5);
});

test("disabling blink immediately restores procedural eye ownership", () => {
  const state = { blinkWait: 0, blinkPhase: "closing", blinkT: 30 };
  const frame = stepMotionLabBlink(state, 0.016, false, 1, durations, () => 0);
  assert.equal(frame.phase, "idle");
  assert.equal(frame.rifeOwnsEye, false);
  assert.equal(state.blinkT, 0);
  assert.equal(state.blinkWait, 2);
});

test("an old hot-reloaded runtime is normalized without starting a stray blink", () => {
  const state = { blinkWait: 1, blinkT: -1 };
  const frame = stepMotionLabBlink(state, 0, true, 1, durations, () => 0);
  assert.equal(frame.phase, "idle");
  assert.equal(state.blinkT, 0);
  assert.equal(state.blinkWait, 1);
});
