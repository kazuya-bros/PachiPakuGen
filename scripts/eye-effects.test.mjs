import test from "node:test";
import assert from "node:assert/strict";
import {
  detectMotionLabEyeRegions,
  MOTION_LAB_BROW_MAX_TILT_DEG,
  motionLabBrowMotion,
  motionLabBrowRotationDeg,
  motionLabGazeRangePx,
  motionLabHorizontalGazeAt,
  motionLabIrisBreathScale,
  motionLabWetnessOpacity,
  motionLabWetnessGeometry,
} from "../src/motionLab/eyeEffects.ts";

function alphaPixels(width, height, points) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (const [x, y] of points) pixels[(y * width + x) * 4 + 3] = 255;
  return pixels;
}

test("detects the two largest separated iris regions from alpha", () => {
  const points = [];
  for (let y = 2; y <= 4; y += 1) {
    for (let x = 1; x <= 3; x += 1) points.push([x, y]);
    for (let x = 8; x <= 10; x += 1) points.push([x, y]);
  }
  points.push([6, 0]); // 小さなノイズは候補から除外

  const regions = detectMotionLabEyeRegions(alphaPixels(12, 7, points), 12, 7);

  assert.deepEqual(regions.map(({ x, y, w, h }) => ({ x, y, w, h })), [
    { x: 1, y: 2, w: 3, h: 3 },
    { x: 8, y: 2, w: 3, h: 3 },
  ]);
});

test("keeps detached highlights with each iris and does not split a single visible eye", () => {
  const paired = [];
  for (let y = 4; y <= 6; y += 1) {
    for (let x = 2; x <= 4; x += 1) paired.push([x, y]);
    for (let x = 11; x <= 13; x += 1) paired.push([x, y]);
  }
  paired.push([3, 1], [12, 1]);
  const pairedRegions = detectMotionLabEyeRegions(alphaPixels(16, 9, paired), 16, 9);
  assert.deepEqual(pairedRegions.map(({ x, y, w, h }) => ({ x, y, w, h })), [
    { x: 2, y: 1, w: 3, h: 6 },
    { x: 11, y: 1, w: 3, h: 6 },
  ]);

  const single = [];
  for (let y = 2; y <= 6; y += 1) {
    for (let x = 4; x <= 9; x += 1) single.push([x, y]);
  }
  const singleRegions = detectMotionLabEyeRegions(alphaPixels(14, 9, single), 14, 9);
  assert.deepEqual(singleRegions.map(({ x, y, w, h }) => ({ x, y, w, h })), [
    { x: 4, y: 2, w: 6, h: 5 },
  ]);
});

test("gaze range is based on iris width, fully stops at zero, and stays capped", () => {
  const regions = [
    { x: 0, y: 0, w: 48, h: 44, pixels: 1000 },
    { x: 80, y: 0, w: 52, h: 45, pixels: 1100 },
  ];
  assert.equal(motionLabGazeRangePx(regions, 0), 0);
  assert.equal(motionLabGazeRangePx(regions, 1), 1.5);
  assert.equal(motionLabGazeRangePx(regions, 2), 3);
  assert.equal(motionLabGazeRangePx(regions, 10), 4.8);
  assert.deepEqual(motionLabHorizontalGazeAt(Math.PI / 2, regions, 0), { x: 0, y: 0 });
  assert.deepEqual(motionLabHorizontalGazeAt(Math.PI / 2, regions, 1), { x: 1.5, y: 0 });
  assert.deepEqual(motionLabHorizontalGazeAt(Math.PI * 1.5, regions, 2), { x: -3, y: 0 });
});

test("iris breathing stays subtle while wetness remains visible at maximum", () => {
  for (let timeMs = 0; timeMs <= 10_000; timeMs += 137) {
    assert.ok(motionLabIrisBreathScale(timeMs, 1) >= 0.96);
    assert.ok(motionLabIrisBreathScale(timeMs, 1) <= 1.04);
    assert.ok(motionLabWetnessOpacity(timeMs, 1) >= 0.34);
    assert.ok(motionLabWetnessOpacity(timeMs, 1) <= 0.52);
    assert.ok(motionLabIrisBreathScale(timeMs, 0.5) >= 0.99);
    assert.ok(motionLabIrisBreathScale(timeMs, 0.5) <= 1.01);
    assert.ok(motionLabWetnessOpacity(timeMs, 0.5) <= 0.12);
  }
  const irisPeakMs = ((Math.PI / 2 - 0.65) / (Math.PI * 2)) * 5200;
  const irisTroughMs = ((Math.PI * 1.5 - 0.65) / (Math.PI * 2)) * 5200;
  assert.ok(Math.abs(motionLabIrisBreathScale(irisPeakMs, 1) - 1.04) < 1e-9);
  assert.ok(Math.abs(motionLabIrisBreathScale(irisTroughMs, 1) - 0.96) < 1e-9);

  const wetnessPeakMs = ((Math.PI / 2 - 1.1) / (Math.PI * 2)) * 4600;
  const wetnessTroughMs = ((Math.PI * 1.5 - 1.1) / (Math.PI * 2)) * 4600;
  assert.ok(Math.abs(motionLabWetnessOpacity(wetnessPeakMs, 1) - 0.52) < 1e-9);
  assert.ok(Math.abs(motionLabWetnessOpacity(wetnessTroughMs, 1) - 0.34) < 1e-9);
  assert.equal(motionLabIrisBreathScale(1000, 0), 1);
  assert.equal(motionLabWetnessOpacity(1000, 0), 0);
});

test("brow motion is subtle at idle and reacts more clearly while speaking", () => {
  assert.equal(MOTION_LAB_BROW_MAX_TILT_DEG, 1.5);
  assert.deepEqual(motionLabBrowMotion(1000, 1, 0), {
    liftPx: 0,
    tiltDeg: 0,
    asymmetryDeg: 0,
  });
  let idleMax = 0;
  let speakingMax = 0;
  for (let timeMs = 0; timeMs <= 7200; timeMs += 41) {
    const idle = motionLabBrowMotion(timeMs, 0, 1);
    const speaking = motionLabBrowMotion(timeMs, 1, 1);
    const leftRotation = motionLabBrowRotationDeg("left", speaking);
    const rightRotation = motionLabBrowRotationDeg("right", speaking);
    idleMax = Math.max(idleMax, idle.liftPx);
    speakingMax = Math.max(speakingMax, speaking.liftPx);
    assert.ok(idle.liftPx >= 0 && idle.liftPx <= 0.55);
    assert.ok(idle.tiltDeg >= 0 && idle.tiltDeg <= MOTION_LAB_BROW_MAX_TILT_DEG);
    assert.equal(Math.abs(idle.asymmetryDeg), 0);
    assert.ok(speaking.liftPx >= 0 && speaking.liftPx <= 4.2);
    assert.ok(speaking.tiltDeg >= 0 && speaking.tiltDeg <= MOTION_LAB_BROW_MAX_TILT_DEG);
    assert.ok(Math.abs(speaking.asymmetryDeg) <= MOTION_LAB_BROW_MAX_TILT_DEG * 0.18);
    assert.ok(Math.abs(leftRotation) <= MOTION_LAB_BROW_MAX_TILT_DEG);
    assert.ok(Math.abs(rightRotation) <= MOTION_LAB_BROW_MAX_TILT_DEG);
  }
  assert.ok(speakingMax > idleMax * 4);
});

test("brow rotations mirror the base tilt and share a small asymmetry", () => {
  const motion = { tiltDeg: 0.8, asymmetryDeg: 0.15 };
  const left = motionLabBrowRotationDeg("left", motion);
  const right = motionLabBrowRotationDeg("right", motion);

  assert.ok(Math.abs(left - (-0.65)) < 1e-12);
  assert.ok(Math.abs(right - 0.95) < 1e-12);
  assert.ok(left < 0, "screen-left brow must rotate counter-clockwise");
  assert.ok(right > 0, "screen-right brow must rotate clockwise");
  assert.ok(Math.abs((left + right) / 2 - motion.asymmetryDeg) < 1e-12);
  assert.ok(Math.abs((right - left) / 2 - motion.tiltDeg) < 1e-12);

  let sawPositiveAsymmetry = false;
  let sawNegativeAsymmetry = false;
  for (let timeMs = 0; timeMs <= 3500; timeMs += 25) {
    const speaking = motionLabBrowMotion(timeMs, 1, 1);
    sawPositiveAsymmetry ||= speaking.asymmetryDeg > 0.1;
    sawNegativeAsymmetry ||= speaking.asymmetryDeg < -0.1;
  }
  assert.ok(sawPositiveAsymmetry && sawNegativeAsymmetry);
});

test("brow lift and tilt use independent speaking periods", () => {
  const responseDelta = (timeMs) => {
    const idle = motionLabBrowMotion(timeMs, 0, 1);
    const speaking = motionLabBrowMotion(timeMs, 1, 1);
    return {
      liftPx: speaking.liftPx - idle.liftPx,
      tiltDeg: speaking.tiltDeg - idle.tiltDeg,
    };
  };

  const start = responseDelta(0);
  const afterLiftPeriod = responseDelta(820);
  assert.ok(Math.abs(start.liftPx - afterLiftPeriod.liftPx) < 1e-12);
  assert.ok(Math.abs(start.tiltDeg - afterLiftPeriod.tiltDeg) > 0.1);

  const afterTiltPeriod = responseDelta(1240);
  assert.ok(Math.abs(start.tiltDeg - afterTiltPeriod.tiltDeg) < 1e-12);
  assert.ok(Math.abs(start.liftPx - afterTiltPeriod.liftPx) > 0.1);
});

test("wetness geometry survives a normal scaled iris", () => {
  const geometry = motionLabWetnessGeometry({ x: 10, y: 20, w: 53, h: 52 });
  assert.ok(geometry.surfaceRadiusY * 2 >= 10);
  assert.ok(geometry.crescentLineWidth >= 2.6);
  assert.ok(geometry.crescentRadiusX * 2 >= 34);
  assert.ok(geometry.crescentCenterY > 20 + 52 * 0.6);
});
