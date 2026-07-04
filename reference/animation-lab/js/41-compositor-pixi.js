/**
 * pixiコンポジター（pixi.js v8）
 * - A0〜A2/B0〜B2はDOMコンポジターと同一のロジック出力を適用（パリティ検証用）
 * - A3（口メッシュモーフ）/ B3（髪メッシュ揺れ）はpixi専用ランタイムに委譲
 * - v8注意: await app.init() 必須 / MeshPlane / geometry.getBuffer('aPosition')
 */
(function (NS) {
  'use strict';

  let ACTIVE_COUNT = 0;

  class PixiCompositor {
    /**
     * @param {HTMLElement} stageEl
     * @param {CharacterAssets} assets
     * @param {{mouthMesh:boolean, hairMesh:boolean, hairSegments:number}} opts
     */
    constructor(stageEl, assets, opts) {
      this.stageEl = stageEl;
      this.assets = assets;
      this.opts = opts || {};
      this.kind = 'pixi';
      this.ready = false;
      this._texCache = new Map(); // ImageBitmap -> PIXI.Texture
      this._destroyed = false;
    }

    async init() {
      if (!window.PIXI) throw new Error('pixi.js が読み込まれていません（CDN接続または vendor/pixi.min.js を確認）');
      const w = this.stageEl.clientWidth || 400;
      const h = this.stageEl.clientHeight || 400;

      this.app = new PIXI.Application();
      await this.app.init({ width: w, height: h, backgroundAlpha: 0, antialias: true });
      if (this._destroyed) { this.app.destroy(true, { children: true }); return; }
      ACTIVE_COUNT++;

      this.app.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;';
      this.stageEl.appendChild(this.app.canvas);
      this.viewW = w; this.viewH = h;

      const assets = this.assets;
      this.fit = Math.min(w / assets.width, h / assets.height);

      // poseC: 画面座標系（原点=キャラ下端中央）。BodyPose.root はここに適用（DOMとパリティ）
      this.poseC = new PIXI.Container();
      this.poseC.position.set(w / 2, h);
      this.app.stage.addChild(this.poseC);

      // charC: 素材の自然座標系
      this.charC = new PIXI.Container();
      this.charC.sortableChildren = true;
      this.charC.scale.set(this.fit);
      this.charC.position.set((-assets.width * this.fit) / 2, -assets.height * this.fit);
      this.poseC.addChild(this.charC);

      const pixiCtx = { container: this.charC, textureFor: (bmp) => this.textureFor(bmp) };

      // レイヤーごとに1スプライト（フレームはテクスチャ差替え）
      this.sprites = new Map();
      for (const layer of assets.layers) {
        if (NS.MOUTH_LAYER_NAMES.includes(layer.name)) continue; // 口はスロット方式
        if (layer.name === 'hair' && this.opts.hairMesh) continue; // B3はメッシュに置換
        if (NS.ARM_LAYER_NAMES.includes(layer.name)) continue;     // 腕はメッシュチェーン
        const sp = new PIXI.Sprite(this.textureFor(layer.frames[0].bitmap));
        sp.zIndex = layer.zIndex * 10;
        if (layer.name === 'hair' || layer.name === 'hair_back') {
          sp.pivot.set(assets.width / 2, 0);
          sp.position.set(assets.width / 2, 0);
        }
        this.charC.addChild(sp);
        this.sprites.set(layer.name, { sp, layer, shownIdx: 0 });
      }

      // 口スロット（front/back の2枚 = entries方式のパリティ）
      this.mouthSlots = [0, 1].map(() => {
        const sp = new PIXI.Sprite();
        sp.zIndex = NS.LAYER_Z_INDEX.mouth_a * 10;
        sp.visible = false;
        this.charC.addChild(sp);
        return sp;
      });

      // メッシュランタイム
      this.mouthMesh = this.opts.mouthMesh ? new NS.A3.MouthMeshRuntime(pixiCtx, assets) : null;
      this.hairMesh = this.opts.hairMesh ? new NS.HairMeshRuntime(pixiCtx, assets, this.opts.hairSegments || 6) : null;
      this.armMeshes = [];
      for (const name of NS.ARM_LAYER_NAMES) {
        if (assets.byName.has(name)) {
          this.armMeshes.push({ side: name === 'arm_l' ? 'left' : 'right', rt: new NS.ArmMeshRuntime(pixiCtx, assets, name) });
        }
      }

      this.ready = true;
    }

    textureFor(bitmap) {
      let tex = this._texCache.get(bitmap);
      if (!tex) {
        tex = PIXI.Texture.from(bitmap);
        this._texCache.set(bitmap, tex);
      }
      return tex;
    }

    showFrame(layerName, idx) {
      if (!this.ready) return;
      const rec = this.sprites.get(layerName);
      if (!rec) return;
      idx = Math.max(0, Math.min(idx, rec.layer.frames.length - 1));
      if (rec.shownIdx === idx) return;
      rec.sp.texture = this.textureFor(rec.layer.frames[idx].bitmap);
      rec.shownIdx = idx;
    }

    applyMouth(cmd) {
      if (!this.ready || !cmd) return;
      if (cmd.kind === 'mesh') {
        for (const sp of this.mouthSlots) sp.visible = false;
        if (this.mouthMesh) this.mouthMesh.update(cmd.mesh);
        return;
      }
      const entries = cmd.entries || [];
      for (let i = 0; i < this.mouthSlots.length; i++) {
        const sp = this.mouthSlots[i];
        const e = entries[i];
        if (!e) { sp.visible = false; continue; }
        const layer = this.assets.byName.get(e.layer);
        if (!layer) { sp.visible = false; continue; }
        const idx = Math.max(0, Math.min(e.frame, layer.frames.length - 1));
        sp.texture = this.textureFor(layer.frames[idx].bitmap);
        sp.alpha = e.opacity != null ? e.opacity : 1;
        sp.visible = true;
      }
    }

    applyPose(pose) {
      if (!this.ready) return;
      const r = pose.root;
      this.poseC.position.set(this.viewW / 2 + r.x, this.viewH + r.y);
      this.poseC.rotation = r.rot;
      this.poseC.scale.set(r.scaleX, r.scaleY);

      // hair/hair_back: DOMは表示px、charC内は自然座標なので fit で割る
      const hairRec = this.sprites.get('hair');
      if (hairRec) {
        hairRec.sp.position.set(this.assets.width / 2 + pose.hair.x / this.fit, 0);
        hairRec.sp.rotation = pose.hair.rot;
      }
      const hbRec = this.sprites.get('hair_back');
      if (hbRec) {
        hbRec.sp.position.set(this.assets.width / 2 + pose.hairBack.x / this.fit, 0);
        hbRec.sp.rotation = pose.hairBack.rot;
      }
      if (this.hairMesh && pose.hairMeshAngles) {
        this.hairMesh.update(pose.hairMeshAngles);
      }
      for (const am of this.armMeshes) {
        // lift は表示px → charC内は自然座標なので fit で割る
        am.rt.update(
          pose.arms ? pose.arms[am.side].angles : null,
          pose.arms ? pose.arms[am.side].lift / this.fit : 0
        );
      }
    }

    destroy() {
      this._destroyed = true;
      if (this.app) {
        if (this.mouthMesh) this.mouthMesh.destroy();
        if (this.hairMesh) this.hairMesh.destroy();
        for (const am of this.armMeshes || []) am.rt.destroy();
        this.app.destroy(true, { children: true });
        this.app = null;
        ACTIVE_COUNT--;
      }
      this._texCache.clear();
      this.ready = false;
    }

    static get activeCount() { return ACTIVE_COUNT; }
  }

  NS.PixiCompositor = PixiCompositor;
})(window.AnimLab);
