import test from "node:test";
import assert from "node:assert/strict";
import {
  motionLabEarTwitchImpulse,
  motionLabInitialEarTwitchWait,
  motionLabNextEarTwitchWait,
} from "../src/motionLab/earTwitch.ts";

test("ear twitch modes separate bounce, tilt, and double motion", () => {
  assert.deepEqual(motionLabEarTwitchImpulse("bounce", 1, 1), {
    bounceVelocity: -110,
    rotationVelocity: 0,
  });
  assert.deepEqual(motionLabEarTwitchImpulse("tilt", 1, -1), {
    bounceVelocity: 0,
    rotationVelocity: -1.2,
  });
  assert.deepEqual(motionLabEarTwitchImpulse("double", 1, 1), {
    bounceVelocity: -110,
    rotationVelocity: 1.2,
  });
  assert.deepEqual(motionLabEarTwitchImpulse("double", 1, 1, true), {
    bounceVelocity: -79.2,
    rotationVelocity: 0.864,
  });
});

test("double twitch queues one short follow-up before returning to the normal interval", () => {
  assert.equal(motionLabNextEarTwitchWait("double", true, 0), 0.12);
  assert.equal(motionLabNextEarTwitchWait("double", true, 1), 0.24);
  assert.equal(motionLabNextEarTwitchWait("double", false, 0), 3);
  assert.equal(motionLabNextEarTwitchWait("tilt", true, 1), 9);
});

test("the first twitch is scheduled quickly enough to compare modes", () => {
  assert.equal(motionLabInitialEarTwitchWait(0), 0.55);
  assert.ok(Math.abs(motionLabInitialEarTwitchWait(1) - 1.2) < 1e-9);
});
