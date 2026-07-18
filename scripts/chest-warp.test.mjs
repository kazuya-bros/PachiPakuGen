import test from "node:test";
import assert from "node:assert/strict";

import {
  motionLabChestWarpBounds,
  motionLabChestWarpSourceY,
  resolveMotionLabChestWarpRegion,
} from "../src/motionLab/chestWarp.ts";

test("chest guide selects the local warp center without becoming a draw layer", () => {
  const region = resolveMotionLabChestWarpRegion(
    1000,
    800,
    { x: 200, y: 80, w: 600, h: 700 },
    { x: 390, y: 350, w: 220, h: 120 },
  );
  assert.equal(region.centerX, 500);
  assert.equal(region.centerY, 410);
});

test("body bounds provide a restrained fallback when chest.png is absent", () => {
  const region = resolveMotionLabChestWarpRegion(1000, 800, { x: 200, y: 80, w: 600, h: 700 });
  assert.equal(region.centerX, 500);
  assert.equal(region.centerY, 430);
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
