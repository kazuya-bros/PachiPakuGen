/**
 * SpriTalk移植UI案（A1調音結合＋B3メッシュ髪揺れ＋腕揺れ採用前提）
 *
 * 「キャラクター設定 > アニメーション」に追加する想定の設定UIを
 * 実際に動くオーバーレイとして実装。値はライブで比較セルに反映される。
 * 保存データ構造案（AnimationSettingsV2）も現在値から生成して表示する。
 * 仕様: docs/animation-lab-tech.md §5
 */
(function (NS) {
  'use strict';

  let overlay = null;

  // 髪揺れモード → 比較セルの体方式マッピング
  const HAIR_MODE_TO_BODY = { sine: 'B0', spring: 'B1', mesh: 'B3' };
  const MOUTH_MODE_TO_METHOD = { step: 'A0', coarticulation: 'A1' };

  function applyToAllCells(kind, methodId) {
    for (const cell of NS.state.cells) {
      const sel = cell.el.querySelector(kind === 'mouth' ? '.mouth-sel' : '.body-sel');
      if (kind === 'mouth') cell.mouthId = methodId;
      else cell.bodyId = methodId;
      if (sel) sel.value = methodId;
      cell.rebuild();
    }
  }

  /** 現在値から保存データ構造案（AnimationSettingsV2）を生成 */
  function currentSettings(state) {
    const P = NS.P;
    return {
      lipSync: {
        mode: state.mouthMode,
        attackMs: P.envelope.attackMs,
        releaseMs: P.envelope.releaseMs,
      },
      hairPhysics: {
        mode: state.hairMode,
        segments: 6,
        stiffness: state.hairMode === 'spring' ? P.b1.k : P.b3.k,
        damping: state.hairMode === 'spring' ? P.b1.c : P.b3.c,
        wind: P.b3.wind,
        drive: P.b3.drive,
      },
      armSway: {
        enabled: !!P.arm.enabled,
        stiffness: P.arm.k,
        damping: P.arm.c,
        maxAngle: P.arm.maxAngle,
      },
    };
  }

  function open() {
    if (overlay) return;
    const state = { mouthMode: 'coarticulation', hairMode: 'mesh', tab: 'lip' };

    overlay = document.createElement('div');
    overlay.id = 'port-ui-overlay';
    overlay.innerHTML = `
      <div class="pu-panel">
        <div class="pu-header">
          <span class="pu-breadcrumb">キャラクター設定 › アニメーション <span class="pu-tag">移植UI案</span></span>
          <button class="btn small pu-close">✕</button>
        </div>
        <div class="pu-tabs">
          <button class="pu-tab active" data-tab="lip">口パク</button>
          <button class="pu-tab" data-tab="hair">髪揺れ</button>
          <button class="pu-tab" data-tab="arm">腕揺れ</button>
          <button class="pu-tab" data-tab="data">保存データ案</button>
        </div>
        <div class="pu-body"></div>
        <div class="pu-footer">値は左の比較セルに即時反映（ライブプレビュー相当）</div>
      </div>
    `;
    document.body.appendChild(overlay);
    const body = overlay.querySelector('.pu-body');

    overlay.querySelector('.pu-close').addEventListener('click', close);
    overlay.addEventListener('click', (ev) => { if (ev.target === overlay) close(); });
    for (const tabBtn of overlay.querySelectorAll('.pu-tab')) {
      tabBtn.addEventListener('click', () => {
        state.tab = tabBtn.dataset.tab;
        for (const b of overlay.querySelectorAll('.pu-tab')) b.classList.toggle('active', b === tabBtn);
        render();
      });
    }

    // ---- 部品ビルダー ----
    function radioRow(label, name, options, current, onChange, note) {
      const row = document.createElement('div');
      row.className = 'pu-row';
      row.innerHTML = `<label>${label}</label>`;
      const group = document.createElement('div');
      group.className = 'pu-radio-group';
      for (const [value, text, recommended] of options) {
        const l = document.createElement('label');
        l.className = 'pu-radio';
        const r = document.createElement('input');
        r.type = 'radio'; r.name = name; r.value = value;
        r.checked = value === current;
        r.addEventListener('change', () => { onChange(value); render(); });
        l.appendChild(r);
        l.appendChild(document.createTextNode(text + (recommended ? '（推奨）' : '')));
        group.appendChild(l);
      }
      row.appendChild(group);
      if (note) {
        const n = document.createElement('div');
        n.className = 'pu-note';
        n.textContent = note;
        row.appendChild(n);
      }
      return row;
    }

    function sliderRow(label, get, set, min, max, step, suffix) {
      const row = document.createElement('div');
      row.className = 'pu-row';
      row.innerHTML = `<label>${label}</label>`;
      const range = document.createElement('input');
      range.type = 'range'; range.min = min; range.max = max; range.step = step;
      range.value = get();
      const val = document.createElement('span');
      val.className = 'pu-val';
      val.textContent = get() + (suffix || '');
      range.addEventListener('input', () => {
        set(parseFloat(range.value));
        val.textContent = range.value + (suffix || '');
      });
      row.appendChild(range);
      row.appendChild(val);
      return row;
    }

    function toggleRow(label, get, set, note) {
      const row = document.createElement('div');
      row.className = 'pu-row';
      row.innerHTML = `<label>${label}</label>`;
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = !!get();
      chk.addEventListener('change', () => set(chk.checked ? 1 : 0));
      row.appendChild(chk);
      if (note) {
        const n = document.createElement('div');
        n.className = 'pu-note';
        n.textContent = note;
        row.appendChild(n);
      }
      return row;
    }

    // ---- タブ描画 ----
    function render() {
      body.innerHTML = '';
      const P = NS.P;

      if (state.tab === 'lip') {
        body.appendChild(radioRow('方式', 'pu-lip-mode', [
          ['coarticulation', '調音結合', true],
          ['step', '従来（ステップ送り）', false],
        ], state.mouthMode, (v) => {
          state.mouthMode = v;
          applyToAllCells('mouth', MOUTH_MODE_TO_METHOD[v]);
        }, '調音結合: 母音間で口を閉じずに遷移。素材（RIFE連番）は従来のまま使用'));
        body.appendChild(sliderRow('開く速さ attack', () => P.envelope.attackMs, (v) => (P.envelope.attackMs = v), 5, 250, 5, ' ms'));
        body.appendChild(sliderRow('閉じる速さ release', () => P.envelope.releaseMs, (v) => (P.envelope.releaseMs = v), 10, 400, 5, ' ms'));
      }

      if (state.tab === 'hair') {
        body.appendChild(radioRow('方式', 'pu-hair-mode', [
          ['mesh', 'メッシュ物理', true],
          ['spring', '遅延バネ（軽量）', false],
          ['sine', 'sin波（従来）', false],
        ], state.hairMode, (v) => {
          state.hairMode = v;
          applyToAllCells('body', HAIR_MODE_TO_BODY[v]);
        }, 'メッシュ物理はWebGL描画。遅延バネはレイヤー全体を追従させる軽量フォールバック'));
        if (state.hairMode === 'mesh') {
          body.appendChild(sliderRow('柔らかさ（剛性k・低いほど揺れる）', () => P.b3.k, (v) => (P.b3.k = v), 10, 200, 5));
          body.appendChild(sliderRow('収まり（減衰c）', () => P.b3.c, (v) => (P.b3.c = v), 1, 30, 0.5));
          body.appendChild(sliderRow('風の強さ', () => P.b3.wind, (v) => (P.b3.wind = v), 0, 0.06, 0.002));
          body.appendChild(sliderRow('体への追従', () => P.b3.drive, (v) => (P.b3.drive = v), 0, 0.2, 0.01));
        } else if (state.hairMode === 'spring') {
          body.appendChild(sliderRow('柔らかさ（剛性k）', () => P.b1.k, (v) => (P.b1.k = v), 10, 300, 5));
          body.appendChild(sliderRow('収まり（減衰c）', () => P.b1.c, (v) => (P.b1.c = v), 1, 40, 0.5));
        } else {
          body.appendChild(sliderRow('揺れ振幅', () => P.b0.hairAmp, (v) => (P.b0.hairAmp = v), 0, 15, 0.5, ' px'));
        }
      }

      if (state.tab === 'arm') {
        const hasArms = NS.state.assets && NS.ARM_LAYER_NAMES.some((n) => NS.state.assets.byName.has(n));
        if (!hasArms) {
          const n = document.createElement('div');
          n.className = 'pu-note';
          n.textContent = 'arm_l.png / arm_r.png が読み込まれていません（本体UIでは腕レイヤーがある場合のみこのセクションを表示する想定）';
          body.appendChild(n);
        }
        body.appendChild(toggleRow('腕揺れを有効にする', () => P.arm.enabled, (v) => (P.arm.enabled = v),
          '肩ピボットは腕画像の不透明領域から自動推定'));
        body.appendChild(sliderRow('揺れ幅（最大角）', () => P.arm.maxAngle, (v) => (P.arm.maxAngle = v), 0.02, 0.3, 0.01, ' rad'));
        body.appendChild(sliderRow('柔らかさ（剛性k）', () => P.arm.k, (v) => (P.arm.k = v), 20, 250, 5));
        body.appendChild(sliderRow('収まり（減衰c）', () => P.arm.c, (v) => (P.arm.c = v), 2, 30, 0.5));
        body.appendChild(toggleRow('肩の弾み', () => P.arm.liftEnabled, (v) => (P.arm.liftEnabled = v),
          '体の上下動（呼吸・バウンス）に肩が遅れて追従＋発話開始でぽよんと弾む'));
        body.appendChild(sliderRow('肩弾みの量', () => P.arm.liftCoupling, (v) => (P.arm.liftCoupling = v), 0, 0.3, 0.01));
        body.appendChild(sliderRow('発話バウンス強度', () => P.arm.liftBounce, (v) => (P.arm.liftBounce = v), 0, 80, 2));
      }

      if (state.tab === 'data') {
        const note = document.createElement('div');
        note.className = 'pu-note';
        note.textContent = 'この設定はキャラクターごとに保存される想定（AnimationSettingsV2、docs/animation-lab-tech.md §5.2）。hairPhysics.mode="sine" で従来動作＝完全後方互換。';
        body.appendChild(note);
        const pre = document.createElement('pre');
        pre.className = 'pu-json';
        pre.textContent = JSON.stringify(currentSettings(state), null, 2);
        body.appendChild(pre);
        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn small';
        copyBtn.textContent = '📋 コピー';
        copyBtn.addEventListener('click', async () => {
          try { await navigator.clipboard.writeText(pre.textContent); copyBtn.textContent = '✓ コピー済'; } catch (e) { /* 無視 */ }
        });
        body.appendChild(copyBtn);
      }
    }

    // 初期状態を比較セルへ適用（A1＋B3の採用構成）
    applyToAllCells('mouth', 'A1');
    applyToAllCells('body', 'B3');
    render();
  }

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  NS.PortUI = { open, close };
})(window.AnimLab);
