/**
 * A3 対応点編集UI
 * - 母音キーフレームを半透明トレース表示し、その上でメッシュ頂点をドラッグ編集
 * - ROI（口領域矩形）のドラッグ指定
 * - MorphData の JSONエクスポート/インポート（将来のPachiPakuGen出力仕様の叩き台）
 */
(function (NS) {
  'use strict';

  const U = NS.U;
  let overlay = null;

  function open(assets) {
    if (overlay) return;
    const md = NS.A3.ensureMorphData(assets);

    overlay = document.createElement('div');
    overlay.id = 'mesh-editor-overlay';

    // ---- ツールバー ----
    const bar = document.createElement('div');
    bar.className = 'me-toolbar';
    overlay.appendChild(bar);

    const state = {
      vowel: 'a',
      mode: 'vertex',       // 'vertex' | 'roi'
      traceAlpha: 0.5,
      previewOpen: 1.0,
      dragging: null,       // {type:'vertex', index} | {type:'roi', startX, startY}
    };

    function btn(label, onClick, cls = 'btn small') {
      const b = document.createElement('button');
      b.className = cls;
      b.textContent = label;
      b.addEventListener('click', onClick);
      bar.appendChild(b);
      return b;
    }

    // 母音選択
    const vowelBtns = {};
    for (const v of NS.VOWELS) {
      vowelBtns[v] = btn(v.toUpperCase(), () => { state.vowel = v; syncButtons(); draw(); });
    }
    // モード切替
    const modeVertexBtn = btn('頂点編集', () => { state.mode = 'vertex'; syncButtons(); });
    const modeRoiBtn = btn('ROI設定', () => { state.mode = 'roi'; syncButtons(); });

    btn('デフォルト生成', () => {
      md.targets = NS.A3.defaultTargets(md.grid, md.roi);
      bumpRev(); draw();
    });

    // グリッド分割数
    const gridLabel = document.createElement('label');
    gridLabel.style.cssText = 'color:#9aa0ae;font-size:11px;';
    gridLabel.innerHTML = '分割 ';
    const gridSel = document.createElement('select');
    for (const g of ['4x3', '6x5', '8x6', '10x8']) {
      const o = document.createElement('option');
      o.value = g; o.textContent = g;
      if (g === md.grid.cols + 'x' + md.grid.rows) o.selected = true;
      gridSel.appendChild(o);
    }
    gridSel.addEventListener('change', () => {
      const [c, r] = gridSel.value.split('x').map(Number);
      md.grid = { cols: c, rows: r };
      md.targets = NS.A3.defaultTargets(md.grid, md.roi);
      bumpRev(); draw();
    });
    gridLabel.appendChild(gridSel);
    bar.appendChild(gridLabel);

    // トレース透明度 / プレビュー開度
    function slider(label, get, set, min, max, step) {
      const l = document.createElement('label');
      l.style.cssText = 'color:#9aa0ae;font-size:11px;display:flex;align-items:center;gap:4px;';
      l.textContent = label;
      const s = document.createElement('input');
      s.type = 'range'; s.min = min; s.max = max; s.step = step; s.value = get();
      s.style.width = '80px';
      s.addEventListener('input', () => { set(parseFloat(s.value)); draw(); });
      l.appendChild(s);
      bar.appendChild(l);
    }
    slider('トレース', () => state.traceAlpha, (v) => (state.traceAlpha = v), 0, 1, 0.05);
    slider('開度プレビュー', () => state.previewOpen, (v) => (state.previewOpen = v), 0, 1, 0.05);

    btn('📋 JSONエクスポート', async () => {
      const json = NS.A3.morphDataToJson(md);
      // ダウンロード（file://でも download 属性は動作する）
      const blob = new Blob([json], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'mouth-morph.json';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      try { await navigator.clipboard.writeText(json); } catch (e) { /* クリップボード不可でも続行 */ }
    });

    const importInput = document.createElement('input');
    importInput.type = 'file';
    importInput.accept = '.json';
    importInput.hidden = true;
    importInput.addEventListener('change', async () => {
      const file = importInput.files[0];
      if (!file) return;
      try {
        const next = NS.A3.morphDataFromJson(await file.text());
        NS.state.morphData = next;
        bumpRev();
        close();
        open(assets); // 再構築
      } catch (e) {
        alert('インポート失敗: ' + e.message);
      }
    });
    overlay.appendChild(importInput);
    btn('📂 インポート', () => importInput.click());
    btn('✕ 閉じる', () => close(), 'btn small');

    // ---- キャンバス ----
    const wrap = document.createElement('div');
    wrap.className = 'me-canvas-wrap';
    const canvas = document.createElement('canvas');
    const maxH = Math.min(window.innerHeight * 0.72, assets.height);
    const scale = maxH / assets.height;
    canvas.width = Math.round(assets.width * scale);
    canvas.height = Math.round(assets.height * scale);
    wrap.appendChild(canvas);
    overlay.appendChild(wrap);

    const help = document.createElement('div');
    help.className = 'me-help';
    help.textContent = '頂点編集: ○をドラッグして母音の口形状に合わせる（開度プレビュー=1で全開時の位置） / ROI設定: 口領域を矩形ドラッグ / 変更は再生中のA3セルに即反映';
    overlay.appendChild(help);

    document.body.appendChild(overlay);
    const ctx = canvas.getContext('2d');

    function bumpRev() { NS.state.morphRev = (NS.state.morphRev || 0) + 1; }

    function syncButtons() {
      for (const v of NS.VOWELS) vowelBtns[v].classList.toggle('active', state.vowel === v);
      modeVertexBtn.classList.toggle('active', state.mode === 'vertex');
      modeRoiBtn.classList.toggle('active', state.mode === 'roi');
    }

    /** 頂点のキャンバス座標（オフセット×プレビュー開度込み） */
    function vertexPos(index) {
      const { cols, rows } = md.grid;
      const r = Math.floor(index / (cols + 1));
      const c = index % (cols + 1);
      const off = md.targets[state.vowel][index];
      return {
        x: (md.roi.x + (c * md.roi.w) / cols + off[0] * state.previewOpen) * scale,
        y: (md.roi.y + (r * md.roi.h) / rows + off[1] * state.previewOpen) * scale,
      };
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // ベース合成（hair_back → body → eye0 → hair）
      for (const name of ['hair_back', 'body', 'eye', 'hair']) {
        const l = assets.byName.get(name);
        if (l) ctx.drawImage(l.frames[0].bitmap, 0, 0, canvas.width, canvas.height);
      }
      // トレース: 選択母音の全開キーフレーム
      const ml = assets.byName.get('mouth_' + state.vowel);
      if (ml && state.traceAlpha > 0) {
        ctx.globalAlpha = state.traceAlpha;
        ctx.drawImage(ml.frames[ml.frames.length - 1].bitmap, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1;
      }
      // ROI矩形
      ctx.strokeStyle = '#f0a35e';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(md.roi.x * scale, md.roi.y * scale, md.roi.w * scale, md.roi.h * scale);
      ctx.setLineDash([]);

      // ワイヤーフレーム＋頂点
      const { cols, rows } = md.grid;
      ctx.strokeStyle = 'rgba(110,168,254,0.7)';
      ctx.lineWidth = 1;
      for (let r = 0; r <= rows; r++) {
        ctx.beginPath();
        for (let c = 0; c <= cols; c++) {
          const p = vertexPos(r * (cols + 1) + c);
          c === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }
      for (let c = 0; c <= cols; c++) {
        ctx.beginPath();
        for (let r = 0; r <= rows; r++) {
          const p = vertexPos(r * (cols + 1) + c);
          r === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }
      ctx.fillStyle = '#6ea8fe';
      const n = (cols + 1) * (rows + 1);
      for (let i = 0; i < n; i++) {
        const p = vertexPos(i);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function canvasPos(ev) {
      const rect = canvas.getBoundingClientRect();
      return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
    }

    canvas.addEventListener('mousedown', (ev) => {
      const p = canvasPos(ev);
      if (state.mode === 'roi') {
        state.dragging = { type: 'roi', startX: p.x, startY: p.y };
        return;
      }
      // 最寄り頂点をピック
      const n = (md.grid.cols + 1) * (md.grid.rows + 1);
      let best = -1, bestD = 12;
      for (let i = 0; i < n; i++) {
        const v = vertexPos(i);
        const d = Math.hypot(v.x - p.x, v.y - p.y);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0) state.dragging = { type: 'vertex', index: best };
    });

    canvas.addEventListener('mousemove', (ev) => {
      if (!state.dragging) return;
      const p = canvasPos(ev);
      if (state.dragging.type === 'vertex') {
        const i = state.dragging.index;
        const { cols, rows } = md.grid;
        const r = Math.floor(i / (cols + 1));
        const c = i % (cols + 1);
        const baseX = md.roi.x + (c * md.roi.w) / cols;
        const baseY = md.roi.y + (r * md.roi.h) / rows;
        const po = Math.max(0.05, state.previewOpen); // 開度0では編集不能なので下限
        md.targets[state.vowel][i] = [
          (p.x / scale - baseX) / po,
          (p.y / scale - baseY) / po,
        ];
        bumpRev();
        draw();
      } else {
        const x0 = Math.min(state.dragging.startX, p.x) / scale;
        const y0 = Math.min(state.dragging.startY, p.y) / scale;
        const x1 = Math.max(state.dragging.startX, p.x) / scale;
        const y1 = Math.max(state.dragging.startY, p.y) / scale;
        md.roi = {
          x: Math.round(x0), y: Math.round(y0),
          w: Math.max(8, Math.round(x1 - x0)), h: Math.max(8, Math.round(y1 - y0)),
        };
        draw();
      }
    });

    window.addEventListener('mouseup', () => {
      if (state.dragging && state.dragging.type === 'roi') {
        // ROI変更後はデフォルトターゲットを再生成
        md.targets = NS.A3.defaultTargets(md.grid, md.roi);
        bumpRev();
        draw();
      }
      state.dragging = null;
    });

    syncButtons();
    draw();
  }

  function close() {
    if (overlay) {
      overlay.remove();
      overlay = null;
    }
  }

  NS.MeshEditor = { open, close };
})(window.AnimLab);
