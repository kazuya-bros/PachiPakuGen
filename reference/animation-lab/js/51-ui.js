/**
 * UI: パラメータスライダー / 比較セル管理 / トランスポート
 */
(function (NS) {
  'use strict';

  // ===== パラメータ定義（NS.P のパスにバインド） =====
  const PARAM_DEFS = [
    {
      sec: 'A4 開度エンベロープ', open: true,
      items: [
        { path: 'envelope.mode', label: 'モード', type: 'select', options: [['envelope', 'エンベロープ'], ['rect', '矩形（現行相当）']] },
        { path: 'envelope.attackMs', label: 'attack (ms)', min: 5, max: 250, step: 5 },
        { path: 'envelope.releaseMs', label: 'release (ms)', min: 10, max: 400, step: 5 },
      ],
    },
    {
      sec: 'A1 調音結合', items: [
        { path: 'a1.smoothTime', label: 'smoothTime (s)', min: 0.01, max: 0.3, step: 0.01 },
      ],
    },
    {
      sec: 'A2 クロスフェード', items: [
        { path: 'a2.blendTimeMs', label: 'blend (ms)', min: 20, max: 400, step: 10 },
        { path: 'a2.smoothTime', label: 'smoothTime (s)', min: 0.01, max: 0.3, step: 0.01 },
      ],
    },
    {
      sec: 'A3 メッシュモーフ', items: [
        { path: 'a3.smoothTime', label: 'smoothTime (s)', min: 0.02, max: 0.3, step: 0.01 },
        { path: 'a3.innerStart', label: '口内出現 開始', min: 0, max: 0.6, step: 0.05 },
        { path: 'a3.innerFull', label: '口内出現 完了', min: 0.2, max: 1, step: 0.05 },
        { path: 'a3.jawFactor', label: '顎の下がり量', min: 0.05, max: 0.7, step: 0.05 },
      ],
    },
    {
      sec: 'B0 待機（現行）', items: [
        { path: 'b0.breathAmp', label: '呼吸 振幅(px)', min: 0, max: 15, step: 0.5 },
        { path: 'b0.breathSpeed', label: '呼吸 速度', min: 0.2, max: 3, step: 0.1 },
        { path: 'b0.swayAmpX', label: '揺れ X(px)', min: 0, max: 12, step: 0.5 },
        { path: 'b0.swayAmpY', label: '揺れ Y(px)', min: 0, max: 12, step: 0.5 },
        { path: 'b0.swaySpeed', label: '揺れ 速度', min: 0.2, max: 3, step: 0.1 },
        { path: 'b0.reduceOnSpeech', label: '発話中20%減衰', type: 'check' },
        { path: 'b0.hairAmp', label: '髪sin 振幅(px)', min: 0, max: 15, step: 0.5 },
        { path: 'b0.hairRot', label: '髪sin 回転(rad)', min: 0, max: 0.08, step: 0.005 },
      ],
    },
    {
      sec: 'B1 スプリング髪', items: [
        { path: 'b1.k', label: '剛性 k', min: 10, max: 300, step: 5 },
        { path: 'b1.c', label: '減衰 c', min: 1, max: 40, step: 0.5, extra: 'critical' },
        { path: 'b1.coupling', label: '速度カップリング', min: 0, max: 0.1, step: 0.005 },
        { path: 'b1.maxAngle', label: '最大角(rad)', min: 0.05, max: 0.5, step: 0.01 },
      ],
    },
    {
      sec: 'B2 発話反応', items: [
        { path: 'b2.bounceAmp', label: 'バウンス強度', min: 0, max: 0.15, step: 0.005 },
        { path: 'b2.bounceLambda', label: '減衰 λ', min: 1, max: 12, step: 0.5 },
        { path: 'b2.bounceFreq', label: '周波数 ω', min: 4, max: 40, step: 1 },
      ],
    },
    {
      sec: 'B3 メッシュ髪', items: [
        { path: 'b3.k', label: '根元剛性 k', min: 10, max: 200, step: 5 },
        { path: 'b3.c', label: '減衰 c', min: 1, max: 30, step: 0.5 },
        { path: 'b3.wind', label: '風の強さ', min: 0, max: 0.06, step: 0.002 },
        { path: 'b3.drive', label: '頭追従', min: 0, max: 0.2, step: 0.01 },
      ],
    },
    {
      sec: '腕揺れ（待機拡張）', items: [
        { path: 'arm.enabled', label: '有効', type: 'check' },
        { path: 'arm.k', label: '剛性 k', min: 20, max: 250, step: 5 },
        { path: 'arm.c', label: '減衰 c', min: 2, max: 30, step: 0.5 },
        { path: 'arm.coupling', label: '頭追従', min: 0, max: 0.08, step: 0.005 },
        { path: 'arm.noise', label: '常時微揺れ', min: 0, max: 0.03, step: 0.001 },
        { path: 'arm.maxAngle', label: '最大角(rad)', min: 0.02, max: 0.3, step: 0.01 },
        { path: 'arm.liftEnabled', label: '肩の弾み', type: 'check' },
        { path: 'arm.liftCoupling', label: '肩弾み 追従量', min: 0, max: 0.3, step: 0.01 },
        { path: 'arm.liftBounce', label: '発話バウンス', min: 0, max: 80, step: 2 },
        { path: 'arm.liftMax', label: '肩弾み 最大(px)', min: 1, max: 15, step: 0.5 },
      ],
    },
    {
      sec: 'まばたき軸', items: [
        { path: 'blink.closeMs', label: '閉じ (ms)', min: 40, max: 300, step: 10 },
        { path: 'blink.openMs', label: '開き (ms)', min: 40, max: 400, step: 10 },
        { path: 'blink.intervalMin', label: '間隔 最小(s)', min: 0.5, max: 8, step: 0.5 },
        { path: 'blink.intervalMax', label: '間隔 最大(s)', min: 2, max: 15, step: 0.5 },
        { path: 'blink.halfLevel', label: '半目レベル', min: 0.2, max: 0.8, step: 0.05 },
      ],
    },
    {
      sec: '顔向き軸', items: [
        { path: 'face.driftRange', label: 'ドリフト幅', min: 0, max: 0.5, step: 0.02 },
        { path: 'face.driftSpeed', label: 'ドリフト速度', min: 0.02, max: 0.6, step: 0.02 },
        { path: 'face.nodAmp', label: '頷き強度', min: 0, max: 1.5, step: 0.05 },
        { path: 'face.nodK', label: '頷きバネ k', min: 30, max: 400, step: 10 },
        { path: 'face.nodC', label: '頷き減衰 c', min: 2, max: 40, step: 1 },
      ],
    },
  ];

  function getPath(path) {
    const [a, b] = path.split('.');
    return NS.P[a][b];
  }
  function setPath(path, v) {
    const [a, b] = path.split('.');
    NS.P[a][b] = v;
  }

  function buildParamPanel() {
    const host = document.getElementById('param-sections');
    host.innerHTML = '';
    for (const def of PARAM_DEFS) {
      const details = document.createElement('details');
      details.className = 'param-section';
      if (def.open) details.open = true;
      const summary = document.createElement('summary');
      summary.textContent = def.sec;
      details.appendChild(summary);
      const body = document.createElement('div');
      body.className = 'param-body';
      details.appendChild(body);

      for (const item of def.items) {
        const row = document.createElement('div');
        row.className = 'param-row';
        const label = document.createElement('label');
        label.textContent = item.label;
        row.appendChild(label);

        if (item.type === 'select') {
          const sel = document.createElement('select');
          for (const [v, l] of item.options) {
            const o = document.createElement('option');
            o.value = v; o.textContent = l;
            if (getPath(item.path) === v) o.selected = true;
            sel.appendChild(o);
          }
          sel.addEventListener('change', () => setPath(item.path, sel.value));
          row.appendChild(sel);
        } else if (item.type === 'check') {
          const chk = document.createElement('input');
          chk.type = 'checkbox';
          chk.checked = !!getPath(item.path);
          chk.addEventListener('change', () => setPath(item.path, chk.checked ? 1 : 0));
          row.appendChild(chk);
        } else {
          const range = document.createElement('input');
          range.type = 'range';
          range.min = item.min; range.max = item.max; range.step = item.step;
          range.value = getPath(item.path);
          const val = document.createElement('span');
          val.className = 'pval';
          val.textContent = String(getPath(item.path));
          range.addEventListener('input', () => {
            const v = parseFloat(range.value);
            setPath(item.path, v);
            val.textContent = String(v);
          });
          row.appendChild(range);
          row.appendChild(val);
          if (item.extra === 'critical') {
            const btn = document.createElement('button');
            btn.className = 'btn small';
            btn.textContent = '臨界';
            btn.title = '臨界減衰 c = 2√k をセット';
            btn.addEventListener('click', () => {
              const c = Math.round(2 * Math.sqrt(NS.P.b1.k) * 10) / 10;
              setPath('b1.c', c);
              range.value = c;
              val.textContent = String(c);
            });
            row.appendChild(btn);
          }
        }
        body.appendChild(row);
      }
      host.appendChild(details);
    }
  }

  // ===== 比較セル =====
  const CELL_DEFAULTS = [
    { mouth: 'A0', body: 'B0' },
    { mouth: 'A1', body: 'B1' },
    { mouth: 'A2', body: 'B2' },
    { mouth: 'A3', body: 'B3' },
  ];

  class Cell {
    constructor(gridEl, index) {
      this.index = index;
      this.mouthId = CELL_DEFAULTS[index % 4].mouth;
      this.bodyId = CELL_DEFAULTS[index % 4].body;
      this.compositor = null;
      this.mouthInstance = null;
      this.bodyInstance = null;
      this._rebuilding = false;

      this.el = document.createElement('div');
      this.el.className = 'cell';
      this.el.innerHTML = `
        <div class="cell-header">
          <select class="mouth-sel"></select>
          <select class="body-sel"></select>
          <span class="comp-badge">--</span>
        </div>
        <div class="cell-stage"></div>
      `;
      gridEl.appendChild(this.el);
      this.stageEl = this.el.querySelector('.cell-stage');
      this.badge = this.el.querySelector('.comp-badge');

      const mouthSel = this.el.querySelector('.mouth-sel');
      for (const m of NS.MouthMethods) {
        const o = document.createElement('option');
        o.value = m.id; o.textContent = m.label;
        if (m.id === this.mouthId) o.selected = true;
        mouthSel.appendChild(o);
      }
      mouthSel.addEventListener('change', () => { this.mouthId = mouthSel.value; this.rebuild(); });

      const bodySel = this.el.querySelector('.body-sel');
      for (const b of NS.BodyMethods) {
        const o = document.createElement('option');
        o.value = b.id; o.textContent = b.label;
        if (b.id === this.bodyId) o.selected = true;
        bodySel.appendChild(o);
      }
      bodySel.addEventListener('change', () => { this.bodyId = bodySel.value; this.rebuild(); });
    }

    mouthDef() { return NS.MouthMethods.find((m) => m.id === this.mouthId); }
    bodyDef() { return NS.BodyMethods.find((b) => b.id === this.bodyId); }

    async rebuild() {
      // 連続変更（口と体を続けて切替等）を取りこぼさないよう直列化する
      this._pending = true;
      if (this._rebuilding) return;
      this._rebuilding = true;
      try {
        while (this._pending) {
          this._pending = false;
          await this._doRebuild();
        }
      } finally {
        this._rebuilding = false;
      }
    }

    async _doRebuild() {
      const assets = NS.state.assets;
      if (!assets) return;

      if (this.compositor) { this.compositor.destroy(); this.compositor = null; }
      this.mouthInstance = null;
      this.bodyInstance = null;

      const mouthDef = this.mouthDef();
      const bodyDef = this.bodyDef();
      const needPixi = mouthDef.requiresPixi || bodyDef.requiresPixi;

      if (needPixi) {
        this.badge.textContent = 'pixi (WebGL)';
        this.badge.classList.add('pixi');
        const comp = new NS.PixiCompositor(this.stageEl, assets, {
          mouthMesh: mouthDef.requiresPixi,
          hairMesh: bodyDef.requiresPixi,
          hairSegments: bodyDef.SEGMENTS || 6,
        });
        try {
          await comp.init();
          this.compositor = comp;
        } catch (e) {
          console.error('[Cell] pixi初期化失敗:', e);
          this.badge.textContent = 'pixi失敗→DOM';
          this.compositor = new NS.DomCompositor(this.stageEl, assets);
        }
      } else {
        this.badge.textContent = 'DOM (本体同等)';
        this.badge.classList.remove('pixi');
        this.compositor = new NS.DomCompositor(this.stageEl, assets);
      }

      this.mouthInstance = mouthDef.create(assets);
      this.bodyInstance = bodyDef.create(assets);
      // 腕揺れ: 体方式から独立したコントローラ（arm_l/arm_r 読込時のみ）
      this.armSway = NS.ARM_LAYER_NAMES.some((n) => assets.byName.has(n)) ? new NS.ArmSway() : null;
    }

    resetMethods() {
      if (this.mouthInstance && this.mouthInstance.reset) this.mouthInstance.reset();
    }

    update(dtMs, now, dt, mora, eyeValue) {
      if (!this.compositor || !this.mouthInstance) return;
      const cmd = this.mouthInstance.update(dtMs, now, mora);
      const ctx = {
        speaking: mora.speaking,
        openness: mora.openness,
        speechStarted: mora.speechStarted,
        time: now / 1000,
      };
      const pose = this.bodyInstance.update(dt, ctx);
      pose.arms = this.armSway
        ? this.armSway.update(dt, pose.root.x, pose.root.y, mora.speechStarted)
        : null;
      this.compositor.applyMouth(cmd);
      this.compositor.applyPose(pose);
      // まばたき軸（全セル共通駆動 = 公平比較）
      const eye = NS.state.assets.byName.get('eye');
      if (eye) {
        const idx = Math.round(eyeValue * (eye.frames.length - 1));
        this.compositor.showFrame('eye', idx);
      }
    }

    destroy() {
      if (this.compositor) this.compositor.destroy();
      this.el.remove();
    }
  }

  NS.Cell = Cell;
  NS.buildParamPanel = buildParamPanel;
})(window.AnimLab);
