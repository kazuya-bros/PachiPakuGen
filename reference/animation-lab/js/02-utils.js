/**
 * 数学・アニメーションユーティリティ
 * SmoothDamp / イージング / 1D Perlinノイズ / FPSカウンタ
 */
(function (NS) {
  'use strict';

  const U = {};

  U.clamp = (v, min, max) => (v < min ? min : v > max ? max : v);
  U.lerp = (a, b, t) => a + (b - a) * t;

  /**
   * Unity互換 SmoothDamp（臨界減衰バネによる滑らかな追従）
   * @param {number} current 現在値
   * @param {number} target  目標値
   * @param {{v:number}} velRef 速度保持オブジェクト（呼び出し側で保持）
   * @param {number} smoothTime 到達時定数（秒）
   * @param {number} dt デルタタイム（秒）
   */
  U.smoothDamp = function (current, target, velRef, smoothTime, dt) {
    smoothTime = Math.max(0.0001, smoothTime);
    const omega = 2 / smoothTime;
    const x = omega * dt;
    const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = current - target;
    const temp = (velRef.v + omega * change) * dt;
    velRef.v = (velRef.v - omega * temp) * exp;
    let output = target + (change + temp) * exp;
    // オーバーシュート防止
    if ((target - current > 0) === (output > target)) {
      output = target;
      velRef.v = (output - target) / dt;
    }
    return output;
  };

  // イージング
  U.easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  U.easeInOutSine = (t) => -(Math.cos(Math.PI * t) - 1) / 2;
  U.easeLinear = (t) => t;
  U.EASINGS = { linear: U.easeLinear, outCubic: U.easeOutCubic, inOutSine: U.easeInOutSine };

  /**
   * 1D Perlin風グラデーションノイズ（-1..1）
   * 顔向きドリフト等の「ゆっくりした不規則な動き」用
   */
  const PERM = new Uint8Array(512);
  (function seedPerm() {
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    let s = 1234567;
    for (let i = 255; i > 0; i--) {
      s = (s * 16807) % 2147483647;
      const j = s % (i + 1);
      const t = p[i]; p[i] = p[j]; p[j] = t;
    }
    for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
  })();

  U.noise1d = function (x) {
    const i0 = Math.floor(x) & 255;
    const i1 = (i0 + 1) & 255;
    const f = x - Math.floor(x);
    const u = f * f * f * (f * (f * 6 - 15) + 10); // smootherstep
    const g0 = (PERM[i0] / 127.5 - 1);
    const g1 = (PERM[i1] / 127.5 - 1);
    const n0 = g0 * f;
    const n1 = g1 * (f - 1);
    return U.lerp(n0, n1, u) * 2;
  };

  /** FPSカウンタ（1秒ごとに集計） */
  U.FpsCounter = class {
    constructor() { this.frames = 0; this.last = performance.now(); this.fps = 0; }
    tick(now) {
      this.frames++;
      if (now - this.last >= 1000) {
        this.fps = Math.round((this.frames * 1000) / (now - this.last));
        this.frames = 0;
        this.last = now;
      }
      return this.fps;
    }
  };

  /**
   * 不透明ピクセルのバウンディングボックス（肩ピボット自動推定用）
   * 結果はビットマップ単位でキャッシュ
   */
  const bboxCache = new WeakMap();
  U.alphaBBox = function (bitmap) {
    let b = bboxCache.get(bitmap);
    if (b) return b;
    const c = document.createElement('canvas');
    c.width = bitmap.width; c.height = bitmap.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let minX = c.width, minY = c.height, maxX = -1, maxY = -1;
    for (let y = 0; y < c.height; y++) {
      for (let x = 0; x < c.width; x++) {
        if (d[(y * c.width + x) * 4 + 3] > 8) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    b = maxX < 0
      ? { x: 0, y: 0, w: bitmap.width, h: bitmap.height }
      : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
    bboxCache.set(bitmap, b);
    return b;
  };

  /** バイト数の見やすい表記 */
  U.formatBytes = function (bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  };

  NS.U = U;
})(window.AnimLab);
