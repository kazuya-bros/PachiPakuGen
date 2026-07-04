/**
 * ブートストラップ＋メインループ
 * 全セル・全軸は単一のrAFループから同一dt・同一MoraStateで駆動する（比較の公平性）。
 */
(function (NS) {
  'use strict';

  const state = NS.state;
  const player = new NS.MoraPlayer();
  const blink = new NS.BlinkController();
  const fpsCounter = new NS.U.FpsCounter();
  let lastTs = 0;
  let seeking = false;

  const $ = (id) => document.getElementById(id);

  // ===== 素材読込 =====

  async function onAssetsLoaded(assets) {
    // 既存のセル・軸を破棄してから旧素材を解放
    for (const c of state.cells) c.destroy();
    state.cells = [];
    for (const v of state.axes) v.destroy();
    state.axes = [];
    if (state.assets) NS.Loader.disposeAssets(state.assets);

    state.assets = assets;
    state.morphData = null; // 新素材でMorphDataを作り直す
    NS.A3.ensureMorphData(assets);

    // UI表示切替
    $('drop-zone').classList.add('hidden');
    $('grid-controls').classList.remove('hidden');
    $('grid').classList.remove('hidden');
    $('transport').classList.remove('hidden');
    $('mesh-edit-btn').disabled = false;
    $('port-ui-btn').disabled = false;

    const mouthCount = NS.MOUTH_LAYER_NAMES.filter((n) => assets.byName.has(n)).length;
    $('asset-info').textContent =
      `${assets.width}x${assets.height} / mouth×${mouthCount} / axis×${assets.axes.size}` +
      (assets.isMock ? '（モック）' : '');
    $('asset-info').classList.add('ok');
    $('mem-display').textContent = 'mem: ' + NS.U.formatBytes(assets.totalBytes);

    await buildCells();

    // 軸ラック
    if (assets.axes.size > 0) {
      $('axis-rack-wrap').classList.remove('hidden');
      state.axes = NS.buildAxisRack($('axis-rack'), assets);
    } else {
      $('axis-rack-wrap').classList.add('hidden');
    }
  }

  async function buildCells() {
    const gridEl = $('grid');
    for (const c of state.cells) c.destroy();
    state.cells = [];
    const count = parseInt($('grid-mode').value, 10);
    gridEl.className = count === 1 ? 'g1' : count === 2 ? 'g2' : 'g4';
    for (let i = 0; i < count; i++) {
      state.cells.push(new NS.Cell(gridEl, i));
    }
    for (const c of state.cells) await c.rebuild();
  }

  // ===== トランスポート =====

  function setupTransport() {
    const patternSel = $('pattern-select');
    for (const p of NS.MORA_PATTERNS) {
      const o = document.createElement('option');
      o.value = p.id; o.textContent = p.label;
      patternSel.appendChild(o);
    }
    patternSel.addEventListener('change', () => {
      const p = NS.MORA_PATTERNS.find((x) => x.id === patternSel.value);
      if (p) {
        player.setTimings(p.timings, null);
        resetAllMethods();
        updatePlayBtn();
      }
    });
    player.setTimings(NS.MORA_PATTERNS[0].timings, null);
    player._onLoop = resetAllMethods;

    $('play-btn').addEventListener('click', () => {
      if (player.playing) player.pause();
      else player.play();
      updatePlayBtn();
    });
    $('stop-btn').addEventListener('click', () => {
      player.stop();
      resetAllMethods();
      updatePlayBtn();
    });
    $('loop-chk').addEventListener('change', () => { player.loop = $('loop-chk').checked; });
    player.loop = true;

    const seek = $('seek-bar');
    seek.addEventListener('pointerdown', () => { seeking = true; });
    seek.addEventListener('pointerup', () => { seeking = false; });
    seek.addEventListener('input', () => {
      player.seek((parseInt(seek.value, 10) / 1000) * player.durationMs);
    });
  }

  function updatePlayBtn() {
    $('play-btn').textContent = player.playing ? '⏸ 一時停止' : '▶ 再生';
  }

  function resetAllMethods() {
    for (const c of state.cells) c.resetMethods();
  }

  // ===== VOICEVOX =====

  async function setupVoicevox() {
    const status = $('vv-status');
    const btn = $('vv-btn');
    const version = await NS.Voicevox.probe();
    if (version) {
      status.textContent = 'VOICEVOX: 接続OK';
      status.classList.add('ok');
      btn.disabled = false;
      btn.addEventListener('click', async () => {
        const text = $('vv-text').value.trim();
        if (!text) return;
        btn.disabled = true;
        try {
          const { timings, audio } = await NS.Voicevox.synthesize(text);
          player.setTimings(timings, audio);
          resetAllMethods();
          player.play();
          updatePlayBtn();
        } catch (e) {
          alert('VOICEVOX合成失敗: ' + e.message);
        } finally {
          btn.disabled = false;
        }
      });
    } else {
      status.textContent = 'VOICEVOX: 未接続（埋込データで動作）';
    }
  }

  // ===== 素材読込UI =====

  function setupLoaders() {
    $('folder-input').addEventListener('change', async (ev) => {
      try {
        await onAssetsLoaded(await NS.Loader.loadFromFileList(ev.target.files));
      } catch (e) {
        alert('読込失敗: ' + e.message);
      }
    });

    $('mock-btn').addEventListener('click', async () => {
      $('mock-btn').disabled = true;
      try {
        await onAssetsLoaded(await NS.Mock.generateAssets());
      } finally {
        $('mock-btn').disabled = false;
      }
    });

    // DnD（ステージ全域）
    const area = $('stage-area');
    area.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      $('drop-zone').classList.add('dragover');
    });
    area.addEventListener('dragleave', () => $('drop-zone').classList.remove('dragover'));
    area.addEventListener('drop', async (ev) => {
      ev.preventDefault();
      $('drop-zone').classList.remove('dragover');
      try {
        await onAssetsLoaded(await NS.Loader.loadFromDataTransfer(ev.dataTransfer.items));
      } catch (e) {
        alert('読込失敗: ' + e.message);
      }
    });
  }

  // ===== メインループ =====

  function loop(ts) {
    requestAnimationFrame(loop);
    const now = performance.now();
    let dt = lastTs ? (now - lastTs) / 1000 : 1 / 60;
    lastTs = now;
    dt = Math.min(dt, 0.1); // タブ非アクティブ時のジャンプ防止（本体仕様）

    $('fps-display').textContent = 'FPS: ' + fpsCounter.tick(now);
    $('pixi-count').textContent = 'pixi: ' + NS.PixiCompositor.activeCount;

    if (!state.assets) return;

    const mora = player.update(now, dt);
    const eyeValue = blink.update(dt, now);

    for (const cell of state.cells) {
      cell.update(dt * 1000, now, dt, mora, eyeValue);
    }

    const axisCtx = { mora };
    for (const viewer of state.axes) viewer.update(dt, axisCtx);

    // シークバー・時刻表示
    if (!seeking && player.durationMs > 0) {
      $('seek-bar').value = String(Math.round((mora.elapsed / player.durationMs) * 1000));
    }
    $('time-display').textContent =
      (mora.elapsed / 1000).toFixed(2) + ' / ' + (player.durationMs / 1000).toFixed(2) + 's';
    if (!player.playing) updatePlayBtn();
  }

  // ===== 起動 =====

  function init() {
    NS.buildParamPanel();
    setupTransport();
    setupLoaders();
    setupVoicevox();

    $('grid-mode').addEventListener('change', buildCells);
    $('blink-mode').addEventListener('change', () => blink.setMode($('blink-mode').value));
    $('mesh-edit-btn').addEventListener('click', () => {
      if (state.assets) NS.MeshEditor.open(state.assets);
    });
    $('port-ui-btn').addEventListener('click', () => {
      if (state.assets) NS.PortUI.open();
    });

    if (!window.PIXI) {
      console.warn('[AnimLab] pixi.js が読み込まれていません。A3/B3はDOMフォールバックになります。');
    }
    requestAnimationFrame(loop);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window.AnimLab);
