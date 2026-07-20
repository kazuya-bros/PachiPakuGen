import test, { after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";

const vite = await createServer({
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom",
});
after(() => vite.close());

const {
  noise1d,
  noise1dLoop,
  snapAngularRateToPeriod,
  snapCyclesToPeriod,
  createArmSway,
  updateArmSway,
} = await vite.ssrLoadModule("/src/motionLabPhysics.ts");
const {
  motionLabVowelLoopTimeline,
  MOTION_LAB_VOWEL_KEYS,
} = await vite.ssrLoadModule("/src/motionLab/constants.ts");
const { motionLabTimelineAt } = await vite.ssrLoadModule("/src/motionLab/render.ts");

test("single-vowel loop timeline's 1200ms period divides every loop-export length evenly and holds each state 600ms", () => {
  for (const vowel of MOTION_LAB_VOWEL_KEYS) {
    const { timeline, durationMs } = motionLabVowelLoopTimeline(vowel);
    assert.equal(durationMs, 1200);
    for (const periodMs of [7200, 14400, 21600]) {
      assert.equal(periodMs % durationMs, 0, `${periodMs} must divide evenly by ${durationMs}`);
    }
    // 開いて閉じる、閉じたまま次周期へ（境界の口形が周期の先頭と一致＝シームレス）
    const atStart = motionLabTimelineAt(0, timeline, durationMs);
    // 開状態が600ms保持される（開始直後〜599msまでvowelのまま = 二重の平滑化が
    // 収束しきるだけの時間がある）
    const atOpen = motionLabTimelineAt(600, timeline, durationMs);
    const atOpenHeld = motionLabTimelineAt(1150, timeline, durationMs);
    const atSeam = motionLabTimelineAt(durationMs, timeline, durationMs);
    assert.equal(atStart.mouth, "closed");
    assert.equal(atOpen.mouth, vowel);
    assert.equal(atOpenHeld.mouth, vowel);
    assert.equal(atSeam.mouth, atStart.mouth);
    assert.equal(atSeam.energy, atStart.energy);
  }
});

test("noise1dLoop repeats exactly at the loop period and stays continuous at the seam", () => {
  const L = 4.32;
  for (const x of [0, 0.3, 1.7, 2.9, 4.1]) {
    assert.ok(Math.abs(noise1dLoop(x, L) - noise1dLoop(x + L, L)) < 1e-9);
    assert.ok(Math.abs(noise1dLoop(x, L) - noise1dLoop(x + 3 * L, L)) < 1e-9);
  }
  // シーム両側の値差がノイズの通常変動量を超えない（連続性）
  const eps = 1e-4;
  const before = noise1dLoop(L - eps, L);
  const after_ = noise1dLoop(L + eps, L);
  assert.ok(Math.abs(before - after_) < 0.01, `seam jump ${Math.abs(before - after_)}`);
  // period<=0はフォールバックで通常noise1dと一致
  assert.equal(noise1dLoop(1.234, 0), noise1d(1.234));
});

test("snapped angular rates complete an integer number of cycles per loop", () => {
  const TAU = Math.PI * 2;
  for (const T of [7.2, 14.4, 21.6]) {
    for (const rate of [TAU / 3.6, 1.35, 1.7, 0.9, TAU / 11.5]) {
      const snapped = snapAngularRateToPeriod(rate, T);
      const cycles = (snapped * T) / TAU;
      assert.ok(Math.abs(cycles - Math.round(cycles)) < 1e-9, `rate=${rate} T=${T}`);
      assert.ok(Math.round(cycles) >= 1);
      // 元のレートから極端に離れない（Tが対象周期の3倍以上なら±20%以内）
      if (T * rate / TAU >= 3) {
        assert.ok(Math.abs(snapped / rate - 1) < 0.2, `rate=${rate} T=${T} -> ${snapped}`);
      }
    }
  }
});

test("snapped cycle frequencies complete an integer number of cycles per loop", () => {
  const bps = 160 / 60;
  for (const T of [7.2, 14.4, 21.6]) {
    for (const r of [0.42, 0.72, 0.5, 0.36]) {
      const snapped = snapCyclesToPeriod(bps * r, T);
      const cycles = snapped * T;
      assert.ok(Math.abs(cycles - Math.round(cycles)) < 1e-9);
    }
  }
});

test("arm sway with loopPeriodSeconds produces a periodic trajectory after warm-up", () => {
  const T = 14.4;
  const dt = 1 / 30;
  const stepsPerLoop = Math.round(T / dt);
  const params = {
    k: 90, c: 10, coupling: 0.02, noise: 0.008, maxAngle: 0.12,
    idleSwing: 0.05, liftEnabled: true, liftCoupling: 0.08, liftBounce: 26, liftMax: 6,
    loopPeriodSeconds: T,
  };
  const state = createArmSway(3, false);
  // T周期のroot入力（呼吸・体揺れ相当）でウォームアップ2周
  const rootAt = (t) => ({
    x: Math.sin((2 * Math.PI / T) * 4 * t) * 1.2,
    y: Math.sin((2 * Math.PI / T) * 4 * t + 1) * 3.2,
  });
  let t = 0;
  const run = (loops, record) => {
    const samples = [];
    for (let i = 0; i < stepsPerLoop * loops; i += 1) {
      t += dt;
      const root = rootAt(t);
      const out = updateArmSway(state, params, dt, root.x, root.y, false);
      if (record) samples.push(out.left.rigid);
    }
    return samples;
  };
  run(2, false); // ウォームアップ（過渡応答の減衰）
  const loopA = run(1, true);
  const loopB = run(1, true);
  // 収束後の連続する2周期がほぼ一致（バネの残留過渡のみの誤差）
  let maxDiff = 0;
  for (let i = 0; i < stepsPerLoop; i += 1) {
    maxDiff = Math.max(maxDiff, Math.abs(loopA[i] - loopB[i]));
  }
  assert.ok(maxDiff < 1e-4, `max periodic mismatch ${maxDiff}`);
});
