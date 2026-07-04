/**
 * 軸デモ
 * - BlinkController: 既存eye連番の「軸スクラブ」拡張デモ
 *     auto=自動瞬き（従来） / half=半目保持（ジト目） / sleepy=眠気 / off
 *   → 中間フレームを「保持」できることが等速再生との本質的な違い
 * - AxisRack: axis_* フォルダ（顔向き等）の汎用ビューア。駆動源を差し替えて実験
 */
(function (NS) {
  'use strict';

  const U = NS.U;

  // ============================================================
  // まばたき（eye軸）
  // eye連番: frame0=全開 → 最終=閉じ（PachiPakuGen出力順）
  // ============================================================
  class BlinkController {
    constructor() {
      this.mode = 'auto';
      this.scrub = new NS.Scrub(0);
      this._nextBlinkAt = this._scheduleNext(performance.now());
      this._blinkPhase = null; // {t0} 瞬き実行中
    }

    _scheduleNext(now) {
      const P = NS.P.blink;
      // SpriTalk仕様: ランダム間隔 2〜10秒
      return now + (P.intervalMin + Math.random() * (P.intervalMax - P.intervalMin)) * 1000;
    }

    setMode(mode) {
      this.mode = mode;
      this._blinkPhase = null;
    }

    /** @returns {number} eye軸の値 0(開)..1(閉) */
    update(dt, now) {
      const P = NS.P.blink;
      switch (this.mode) {
        case 'off':
          this.scrub.smooth(0, 0.1, dt);
          break;
        case 'half':
          // 半目で保持（＝軸スクラブなら中間フレームを維持できる）
          this.scrub.smooth(P.halfLevel, 0.25, dt);
          break;
        case 'sleepy':
          // ゆっくり閉眼→わずかに開くを繰り返す（眠気）
          {
            const wave = 0.75 + 0.25 * Math.sin(now / 1400);
            this.scrub.smooth(wave, 0.9, dt);
          }
          break;
        case 'auto':
        default:
          if (this._blinkPhase) {
            const t = now - this._blinkPhase.t0;
            if (t < P.closeMs) {
              this.scrub.set(U.easeOutCubic(t / P.closeMs));
            } else if (t < P.closeMs + 40) {
              this.scrub.set(1);
            } else if (t < P.closeMs + 40 + P.openMs) {
              this.scrub.set(1 - U.easeOutCubic((t - P.closeMs - 40) / P.openMs));
            } else {
              this.scrub.set(0);
              this._blinkPhase = null;
              this._nextBlinkAt = this._scheduleNext(now);
            }
          } else if (now >= this._nextBlinkAt) {
            this._blinkPhase = { t0: now };
          } else {
            this.scrub.set(0);
          }
          break;
      }
      return this.scrub.value;
    }
  }

  // ============================================================
  // 汎用軸ビューア（axis_* ラック）
  // ============================================================
  class AxisViewer {
    /**
     * @param {HTMLElement} rackEl
     * @param {string} name  'axis_face_lr' 等
     * @param {LayerAsset} axis
     */
    constructor(rackEl, name, axis) {
      this.name = name;
      this.axis = axis;
      this.scrub = new NS.Scrub(0.5);
      this.params = { manualValue: 0.5, speed: 0.8, range: NS.P.face.driftRange };
      // デフォルト駆動源: 名前から推定
      this.driverId = name.includes('_ud') ? 'springNod' : name.includes('face') ? 'noise' : 'manual';

      const el = document.createElement('div');
      el.className = 'axis-viewer';
      el.innerHTML = `
        <div class="axis-title">
          <span>${name}（${axis.frames.length}f）</span>
          <select class="axis-driver"></select>
        </div>
        <div class="axis-stage"></div>
        <div class="axis-meter"><div class="axis-meter-fill"></div></div>
        <input type="range" class="axis-manual" min="0" max="1" step="0.005" value="0.5">
      `;
      rackEl.appendChild(el);
      this.el = el;

      const sel = el.querySelector('.axis-driver');
      for (const [id, d] of Object.entries(NS.AXIS_DRIVERS)) {
        const o = document.createElement('option');
        o.value = id; o.textContent = d.label;
        if (id === this.driverId) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => { this.driverId = sel.value; });

      const manual = el.querySelector('.axis-manual');
      manual.addEventListener('input', () => {
        this.params.manualValue = parseFloat(manual.value);
        this.driverId = 'manual';
        sel.value = 'manual';
      });

      // フレームimgを全てプリマウント
      const stage = el.querySelector('.axis-stage');
      this.imgs = axis.frames.map((f) => {
        const img = document.createElement('img');
        img.src = f.url;
        stage.appendChild(img);
        return img;
      });
      this._shown = -1;
      this.meterFill = el.querySelector('.axis-meter-fill');
    }

    update(dt, ctx) {
      const driver = NS.AXIS_DRIVERS[this.driverId];
      if (driver) driver.update(this.scrub, dt, ctx, this.params);
      const idx = this.scrub.frameIndex(this.axis.frames.length);
      if (idx !== this._shown) {
        if (this._shown >= 0) this.imgs[this._shown].style.visibility = 'hidden';
        this.imgs[idx].style.visibility = 'visible';
        this._shown = idx;
      }
      this.meterFill.style.width = (this.scrub.value * 100).toFixed(1) + '%';
    }

    destroy() { this.el.remove(); }
  }

  /** axis_* をラックに並べる */
  function buildAxisRack(rackEl, assets) {
    rackEl.innerHTML = '';
    const viewers = [];
    for (const [name, axis] of assets.axes) {
      viewers.push(new AxisViewer(rackEl, name, axis));
    }
    return viewers;
  }

  NS.BlinkController = BlinkController;
  NS.buildAxisRack = buildAxisRack;
})(window.AnimLab);
