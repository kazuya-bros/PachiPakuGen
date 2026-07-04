/**
 * 軸スクラブ基盤 — 「RIFE連番＝焼き込みパラメータ軸」のコアエンジン
 *
 * 連番フレームを 0..1 の連続パラメータ値として扱い、
 * 駆動源（手動 / タイマー / エンベロープ / バネ / ノイズ）で値を動かす。
 * frameIndex(n) で現在値に対応するフレーム番号を得る。
 */
(function (NS) {
  'use strict';

  const U = NS.U;

  class Scrub {
    constructor(init = 0) {
      this.value = init;
      this._vel = { v: 0 };   // smoothDamp用
      this._w = 0;            // バネ角速度
      this._t = Math.random() * 97; // ノイズ位相
      this._dir = 1;          // ピンポン方向
    }

    set(x) { this.value = U.clamp(x, 0, 1); }

    /** SmoothDampで目標へ滑らかに追従 */
    smooth(target, smoothTime, dt) {
      this.value = U.clamp(U.smoothDamp(this.value, target, this._vel, smoothTime, dt), 0, 1);
      return this.value;
    }

    /**
     * バネ-ダンパーで目標へ（半陰的オイラー、dtが大きい時はサブステップ）
     * k: 剛性(1/s^2), c: 減衰(1/s)
     */
    spring(target, k, c, dt) {
      const steps = dt > 1 / 60 ? Math.ceil(dt / (1 / 120)) : 1;
      const h = dt / steps;
      for (let i = 0; i < steps; i++) {
        const a = -k * (this.value - target) - c * this._w;
        this._w += a * h;
        this.value += this._w * h;
      }
      this.value = U.clamp(this.value, 0, 1);
      return this.value;
    }

    /** バネに撃力を加える（頷き等のイベント駆動） */
    impulse(v) { this._w += v; }

    /** ノイズドリフト: center±range をゆっくり不規則に漂う */
    noiseDrift(speed, range, center, dt) {
      this._t += dt * speed;
      this.value = U.clamp(center + U.noise1d(this._t) * range, 0, 1);
      return this.value;
    }

    /** 0→1→0 の往復再生 */
    pingpong(speed, dt) {
      this.value += this._dir * speed * dt;
      if (this.value >= 1) { this.value = 1; this._dir = -1; }
      if (this.value <= 0) { this.value = 0; this._dir = 1; }
      return this.value;
    }

    /** 現在値に対応するフレーム番号 */
    frameIndex(frameCount) {
      return Math.round(this.value * (frameCount - 1));
    }
  }

  /**
   * 汎用軸ビューア用の駆動源定義（axis_* ラック）
   * update(scrub, dt, ctx, params) — ctx: {mora, speechStarted}
   */
  NS.AXIS_DRIVERS = {
    manual: {
      label: '手動スライダー',
      update(scrub, dt, ctx, params) { scrub.smooth(params.manualValue, 0.08, dt); },
    },
    timer: {
      label: '等速往復（従来相当）',
      update(scrub, dt, ctx, params) { scrub.pingpong(params.speed || 0.8, dt); },
    },
    envelope: {
      label: '開度エンベロープ連動',
      update(scrub, dt, ctx) { scrub.smooth(ctx.mora.openness, 0.05, dt); },
    },
    noise: {
      label: 'ノイズドリフト',
      update(scrub, dt, ctx, params) {
        scrub.noiseDrift(params.speed || NS.P.face.driftSpeed, params.range || NS.P.face.driftRange, 0.5, dt);
      },
    },
    springNod: {
      label: 'バネ＋発話で頷き',
      update(scrub, dt, ctx, params) {
        if (ctx.mora.speechStarted) scrub.impulse(NS.P.face.nodAmp * 6);
        scrub.spring(0.5, NS.P.face.nodK, NS.P.face.nodC, dt);
      },
    },
  };

  NS.Scrub = Scrub;
})(window.AnimLab);
