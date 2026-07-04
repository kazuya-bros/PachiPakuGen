/**
 * A3: メッシュモーフ（Live2D方式）＋ハイブリッド合成
 *
 * - 口ROIをグリッドメッシュ化し、閉じ⇔各母音の頂点オフセットを開度で補間
 * - 口内ピクセル問題は「同一頂点バッファを共有する2枚のメッシュ」で解決:
 *     下: 目標母音の全開キーフレーム（alpha=1、口内ピクセル担当）
 *     上: 閉じ口キーフレーム（alpha = 1-innerAlpha、閉じ→開きのクロスフェード）
 *   → 形状はメッシュモーフ・ピクセルはクロスフェードのハイブリッド
 *
 * MorphData（JSONエクスポート形式 = 将来のPachiPakuGen出力仕様の叩き台）:
 *   { version:1, roi:{x,y,w,h}, grid:{cols,rows},
 *     targets:{ a:[[dx,dy]...], i:..., u:..., e:..., o:... } }  // 頂点は行優先
 */
(function (NS) {
  'use strict';

  const U = NS.U;

  // ===== MorphData ユーティリティ =====

  function vertexCount(grid) { return (grid.cols + 1) * (grid.rows + 1); }

  /** 手動編集前のデフォルト: 顎が下がる程度の手続き的オフセット */
  function defaultTargets(grid, roi) {
    const targets = {};
    for (const v of NS.VOWELS) {
      const scale = NS.VOWEL_OPENNESS[v];
      const offs = [];
      for (let r = 0; r <= grid.rows; r++) {
        const rowFrac = r / grid.rows;
        // 下の行ほど大きく下がる（顎）。境界上端は固定
        const dy = NS.P.a3.jawFactor * roi.h * Math.pow(rowFrac, 1.6) * scale;
        for (let c = 0; c <= grid.cols; c++) {
          const colFrac = c / grid.cols;
          // u/o は横をすぼめる
          let dx = 0;
          if (v === 'u' || v === 'o') {
            dx = (0.5 - colFrac) * roi.w * 0.12 * scale * rowFrac;
          } else if (v === 'i' || v === 'e') {
            dx = (colFrac - 0.5) * roi.w * 0.08 * scale * rowFrac;
          }
          offs.push([dx, dy]);
        }
      }
      targets[v] = offs;
    }
    return targets;
  }

  function createMorphData(assets, cols, rows) {
    const roi = {
      x: Math.round(assets.width * 0.36),
      y: Math.round(assets.height * 0.42),
      w: Math.round(assets.width * 0.28),
      h: Math.round(assets.height * 0.22),
    };
    const grid = { cols, rows };
    return { version: 1, roi, grid, targets: defaultTargets(grid, roi) };
  }

  function morphDataToJson(md) {
    return JSON.stringify({
      version: md.version,
      roi: md.roi,
      grid: md.grid,
      targets: Object.fromEntries(
        Object.entries(md.targets).map(([v, offs]) => [v, offs.map((o) => [Math.round(o[0] * 10) / 10, Math.round(o[1] * 10) / 10])])
      ),
    }, null, 1);
  }

  function morphDataFromJson(text) {
    const md = JSON.parse(text);
    if (!md.roi || !md.grid || !md.targets) throw new Error('MorphData形式ではありません');
    const n = vertexCount(md.grid);
    for (const v of NS.VOWELS) {
      if (!Array.isArray(md.targets[v]) || md.targets[v].length !== n) {
        throw new Error(`targets.${v} の頂点数が grid と一致しません`);
      }
    }
    return md;
  }

  /** グローバルの MorphData を取得（なければデフォルト生成） */
  function ensureMorphData(assets) {
    if (!NS.state.morphData) {
      NS.state.morphData = createMorphData(assets, 6, 5);
      NS.state.morphRev = 1; // 変更検知用リビジョン
    }
    return NS.state.morphData;
  }

  // ===== A3 方式ロジック（コンポジター非依存） =====

  const A3 = {
    id: 'A3', label: 'A3: メッシュモーフ（pixi）', requiresPixi: true,
    create(assets) {
      ensureMorphData(assets);
      let front = null;   // 現在の母音（'a'等）
      let back = null;
      let blendT = 1;
      let openness = 0;
      const vel = { v: 0 };

      return {
        reset() { front = null; back = null; blendT = 1; openness = 0; vel.v = 0; },
        update(dtMs, now, mora) {
          const dt = dtMs / 1000;
          const v = mora.playing ? mora.vowel : undefined;
          if (v && NS.VOWELS.includes(v) && v !== front) {
            back = front;
            front = v;
            blendT = 0;
          }
          blendT = U.clamp(blendT + dt / Math.max(0.001, NS.P.a3.smoothTime), 0, 1);
          openness = U.clamp(U.smoothDamp(openness, mora.openness, vel, NS.P.a3.smoothTime, dt), 0, 1);
          return { kind: 'mesh', mesh: { vowel: front, prevVowel: back, blendT, openness } };
        },
      };
    },
  };
  NS.MouthMethods.push(A3);

  // ===== pixi ランタイム（41-compositor-pixi.js から利用） =====

  class MouthMeshRuntime {
    /**
     * @param {Object} pixiCtx { container, textureFor(bitmap) }
     * @param {CharacterAssets} assets
     */
    constructor(pixiCtx, assets) {
      this.ctx = pixiCtx;
      this.assets = assets;
      this.md = ensureMorphData(assets);
      this.rev = -1;
      this.meshOpen = null;   // 下: 母音全開キーフレーム
      this.meshClosed = null; // 上: 閉じ口キーフレーム
      this.currentVowelTex = null;
      this._build();
    }

    _cropTexture(frameImage, roi) {
      const base = this.ctx.textureFor(frameImage.bitmap);
      return new PIXI.Texture({
        source: base.source,
        frame: new PIXI.Rectangle(roi.x, roi.y, roi.w, roi.h),
      });
    }

    _closedFrame() {
      // 閉じ口キーフレーム: mouth_a等の frame0（なければbody）
      const m = this.assets.byName.get('mouth_a') || this.assets.byName.get('mouth');
      return m ? m.frames[0] : this.assets.byName.get('body').frames[0];
    }

    _openFrame(vowel) {
      const l = this.assets.byName.get('mouth_' + vowel);
      return l ? l.frames[l.frames.length - 1] : this._closedFrame();
    }

    _build() {
      this._destroyMeshes();
      const md = this.md = NS.state.morphData;
      this.rev = NS.state.morphRev;
      const { cols, rows } = md.grid;

      const mk = (tex) => {
        const mesh = new PIXI.MeshPlane({ texture: tex, verticesX: cols + 1, verticesY: rows + 1 });
        mesh.position.set(md.roi.x, md.roi.y);
        mesh.zIndex = NS.LAYER_Z_INDEX.mouth_a * 10; // mouthスロットと同格
        this.ctx.container.addChild(mesh);
        return mesh;
      };
      this.meshOpen = mk(this._cropTexture(this._openFrame('a'), md.roi));
      this.meshClosed = mk(this._cropTexture(this._closedFrame(), md.roi));
      this.meshClosed.zIndex = NS.LAYER_Z_INDEX.mouth_a * 10 + 1; // 閉じ口を上に（フェードで口内が現れる）
      this.currentVowelTex = 'a';
      this._writeVertices('a', null, 1, 0);
    }

    _destroyMeshes() {
      for (const m of [this.meshOpen, this.meshClosed]) {
        if (m) {
          m.removeFromParent();
          m.destroy({ children: true });
        }
      }
      this.meshOpen = this.meshClosed = null;
    }

    _writeVertices(vowel, prevVowel, blendT, openness) {
      const md = this.md;
      const { cols, rows } = md.grid;
      const tgt = md.targets[vowel] || md.targets.a;
      const prev = prevVowel ? md.targets[prevVowel] : null;
      const stepX = md.roi.w / cols;
      const stepY = md.roi.h / rows;

      for (const mesh of [this.meshOpen, this.meshClosed]) {
        const buf = mesh.geometry.getBuffer('aPosition');
        const data = buf.data;
        let i = 0, vi = 0;
        for (let r = 0; r <= rows; r++) {
          for (let c = 0; c <= cols; c++) {
            let dx = tgt[vi][0], dy = tgt[vi][1];
            if (prev && blendT < 1) {
              dx = U.lerp(prev[vi][0], dx, blendT);
              dy = U.lerp(prev[vi][1], dy, blendT);
            }
            data[i++] = c * stepX + dx * openness;
            data[i++] = r * stepY + dy * openness;
            vi++;
          }
        }
        buf.update();
      }
    }

    /** @param {{vowel:string|null, prevVowel:string|null, blendT:number, openness:number}} m */
    update(m) {
      // 編集UIによる変更を反映
      if (NS.state.morphRev !== this.rev) this._build();

      const vowel = m.vowel || 'a';
      if (vowel !== this.currentVowelTex) {
        this.meshOpen.texture = this._cropTexture(this._openFrame(vowel), this.md.roi);
        this.currentVowelTex = vowel;
      }
      this._writeVertices(vowel, m.prevVowel, m.blendT, m.openness);

      // ハイブリッド合成: 開度に応じて閉じ口をフェードアウトし口内を見せる
      const P = NS.P.a3;
      const t = U.clamp((m.openness - P.innerStart) / Math.max(0.01, P.innerFull - P.innerStart), 0, 1);
      const innerAlpha = t * t * (3 - 2 * t); // smoothstep
      this.meshOpen.alpha = m.openness > 0.01 ? 1 : 0;
      this.meshClosed.alpha = 1 - innerAlpha;
    }

    destroy() { this._destroyMeshes(); }
  }

  NS.A3 = { ensureMorphData, createMorphData, defaultTargets, morphDataToJson, morphDataFromJson, MouthMeshRuntime };
})(window.AnimLab);
