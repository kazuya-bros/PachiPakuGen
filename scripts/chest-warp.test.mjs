import test from "node:test";
import assert from "node:assert/strict";

import {
  motionLabChestWarpBounds,
  motionLabChestWarpSourceY,
  resolveMotionLabChestWarpRegion,
} from "../src/motionLab/chestWarp.ts";

test("chest cutout guide is used as-is for the warp center", () => {
  const region = resolveMotionLabChestWarpRegion(
    1000,
    800,
    { x: 200, y: 80, w: 600, h: 700 },
    { x: 390, y: 350, w: 220, h: 120 },
  );
  assert.equal(region.centerX, 500);
  assert.equal(region.centerY, 410);
  assert.ok(region.radiusX >= 220 * 0.42);
  assert.ok(region.radiusY >= 120 * 0.42);
});

test("low-painted cutouts are NOT relocated to an estimated bust position", () => {
  const body = { x: 390, y: 214, w: 548, h: 1066 };
  const lowGuide = { x: 403, y: 785, w: 420, h: 327 };
  const region = resolveMotionLabChestWarpRegion(1280, 1280, body, lowGuide);
  assert.equal(region.centerX, 403 + 420 * 0.5);
  assert.equal(region.centerY, 785 + 327 * 0.5);
});

test("body bounds provide a fallback only when chest.png is absent", () => {
  const region = resolveMotionLabChestWarpRegion(1000, 800, { x: 200, y: 80, w: 600, h: 700 });
  assert.equal(region.centerX, 500);
  assert.equal(region.centerY, 80 + 700 * 0.42);
  const bounds = motionLabChestWarpBounds(1000, 800, region);
  assert.ok(bounds.x > 0 && bounds.y > 0);
  assert.ok(bounds.x + bounds.w < 1000 && bounds.y + bounds.h < 800);
});

test("Gaussian warp moves the center and fixes the outer boundary", () => {
  const region = { centerX: 100, centerY: 100, radiusX: 20, radiusY: 20 };
  assert.equal(motionLabChestWarpSourceY(100, 100, region, 4), 96);
  const outer = motionLabChestWarpSourceY(160, 100, region, 4);
  assert.ok(Math.abs(outer - 100) < 0.05);
});
