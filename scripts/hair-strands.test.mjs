import test from "node:test";
import assert from "node:assert/strict";
import {
  createHairStrandSpring,
  detectHairStrandsFromAlpha,
  stepHairStrandSpring,
} from "../src/motionLabPhysics.ts";

function makeHairAlpha(width, height, peaks) {
  const alpha = new Uint8Array(width * height);
  for (let x = 20; x < width - 20; x += 1) {
    let bottom = 100;
    for (const peak of peaks) {
      const distance = Math.abs(x - peak.x);
      bottom = Math.max(bottom, 100 + peak.depth * Math.max(0, 1 - distance / peak.radius));
    }
    const top = 20 + (Math.floor(x / 80) % 3);
    for (let y = top; y <= Math.round(bottom); y += 1) alpha[y * width + x] = 255;
  }
  return alpha;
}

test("strand detection keeps the contour peak coordinates and vertical span", () => {
  const width = 300;
  const alpha = makeHairAlpha(width, 180, [
    { x: 60, depth: 55, radius: 28 },
    { x: 150, depth: 55, radius: 28 },
    { x: 240, depth: 55, radius: 28 },
  ]);

  const strands = detectHairStrandsFromAlpha(alpha, width, 180, 3);

  assert.deepEqual(strands.map(strand => strand.x), [60, 150, 240]);
  assert.deepEqual(strands.map(strand => strand.tipY), [155, 155, 155]);
  assert.deepEqual(strands.map(strand => strand.rootY), [20, 21, 20]);
});

test("nearby contour peaks are suppressed by prominence and minimum distance", () => {
  const width = 320;
  const alpha = makeHairAlpha(width, 190, [
    { x: 75, depth: 62, radius: 24 },
    { x: 125, depth: 32, radius: 20 },
    { x: 250, depth: 58, radius: 26 },
  ]);

  const strands = detectHairStrandsFromAlpha(alpha, width, 190, 2);

  assert.equal(strands.length, 2);
  assert.ok(Math.abs(strands[0].x - 75) <= 1);
  assert.ok(Math.abs(strands[1].x - 250) <= 1);
});

test("each strand uses two scalar springs with a slower soft response", () => {
  const spring = createHairStrandSpring(1.25);
  for (let frame = 0; frame < 12; frame += 1) {
    stepHairStrandSpring(spring, 10, 1 / 60, 70, 9);
  }

  assert.equal(spring.phase, 1.25);
  assert.ok(spring.stiff.x > spring.soft.x);
  assert.equal(typeof spring.stiff.x, "number");
  assert.equal(typeof spring.soft.x, "number");
});

test("strand spring output respects the visible displacement safety limit", () => {
  const spring = createHairStrandSpring();
  for (let frame = 0; frame < 240; frame += 1) {
    const output = stepHairStrandSpring(spring, frame % 40 < 20 ? 40 : -40, 1 / 60, 55, 5, 6);
    assert.ok(Math.abs(output.stiffDx) <= 6);
    assert.ok(Math.abs(output.softDx) <= 6);
  }
});
