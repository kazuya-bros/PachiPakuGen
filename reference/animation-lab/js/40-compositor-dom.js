/**
 * DOMコンポジター
 * SpriTalk本体（LayeredEmotion）と同一条件のレンダリングパス:
 *   position:absolute の <img> を z-index 順に重ね、visibility/opacity を切替。
 *   全フレームをプリマウントしてデコードジャンクを回避（本体と同じ戦略）。
 *   BodyPose は wrapper（transform-origin: bottom center）と
 *   hair/hair_back レイヤー（top center）への CSS transform。
 */
(function (NS) {
  'use strict';

  class DomCompositor {
    /**
     * @param {HTMLElement} stageEl .cell-stage
     * @param {CharacterAssets} assets
     */
    constructor(stageEl, assets) {
      this.assets = assets;
      this.kind = 'dom';

      this.rootEl = document.createElement('div');
      this.rootEl.className = 'char-root';
      this.wrapper = document.createElement('div');
      this.wrapper.className = 'char-wrapper';
      this.wrapper.style.aspectRatio = `${assets.width} / ${assets.height}`;
      this.wrapper.style.height = '100%';
      this.wrapper.style.maxWidth = '100%';
      this.rootEl.appendChild(this.wrapper);
      stageEl.appendChild(this.rootEl);

      /** layerName -> { el, imgs[], shownIdx } */
      this.layerMap = new Map();
      for (const layer of assets.layers) {
        const el = document.createElement('div');
        el.className = 'char-layer';
        el.style.zIndex = String(layer.zIndex + 10);
        if (layer.name === 'hair' || layer.name === 'hair_back') {
          el.style.transformOrigin = 'top center';
        }
        const imgs = layer.frames.map((f) => {
          const img = document.createElement('img');
          img.src = f.url;
          el.appendChild(img);
          return img;
        });
        this.wrapper.appendChild(el);
        this.layerMap.set(layer.name, { el, imgs, shownIdx: -1, shownOpacity: 1 });
      }

      // 初期表示: 静的レイヤーはframe0
      for (const name of ['hair_back', 'body', 'arm_l', 'arm_r', 'hair', 'eye']) {
        this.showFrame(name, 0);
      }

      // 腕レイヤー: 肩ピボット（不透明bboxの上端中央）を transform-origin に設定
      for (const name of NS.ARM_LAYER_NAMES) {
        const rec = this.layerMap.get(name);
        const layer = assets.byName.get(name);
        if (rec && layer) {
          const bb = NS.U.alphaBBox(layer.frames[0].bitmap);
          const ox = (((bb.x + bb.w / 2) / assets.width) * 100).toFixed(2);
          const oy = ((bb.y / assets.height) * 100).toFixed(2);
          rec.el.style.transformOrigin = `${ox}% ${oy}%`;
        }
      }

      /** 表示中のmouthエントリ: layerName -> frame */
      this._mouthShown = new Map();
    }

    /** 単一フレームレイヤー表示（eye等） */
    showFrame(layerName, idx) {
      const rec = this.layerMap.get(layerName);
      if (!rec) return;
      idx = Math.max(0, Math.min(idx, rec.imgs.length - 1));
      if (rec.shownIdx === idx) return;
      if (rec.shownIdx >= 0) rec.imgs[rec.shownIdx].style.visibility = 'hidden';
      rec.imgs[idx].style.visibility = 'visible';
      rec.imgs[idx].style.opacity = '1';
      rec.shownIdx = idx;
    }

    /** @param {MouthCommand} cmd */
    applyMouth(cmd) {
      if (!cmd) return;
      if (cmd.kind === 'mesh') {
        // DOMではメッシュ不可（セル側でpixi強制されるため通常来ない）
        this._hideAllMouth();
        return;
      }
      const entries = cmd.entries || [];
      const nextKeys = new Set();

      for (const e of entries) {
        const rec = this.layerMap.get(e.layer);
        if (!rec) continue;
        nextKeys.add(e.layer);
        const idx = Math.max(0, Math.min(e.frame, rec.imgs.length - 1));
        if (rec.shownIdx !== idx) {
          if (rec.shownIdx >= 0) rec.imgs[rec.shownIdx].style.visibility = 'hidden';
          rec.imgs[idx].style.visibility = 'visible';
          rec.shownIdx = idx;
        }
        const op = e.opacity != null ? e.opacity : 1;
        if (rec.shownOpacity !== op) {
          rec.imgs[idx].style.opacity = String(op);
          rec.shownOpacity = op;
        }
        this._mouthShown.set(e.layer, idx);
      }

      // 前回表示していて今回いないmouthレイヤーを隠す
      for (const [layer] of this._mouthShown) {
        if (!nextKeys.has(layer)) {
          const rec = this.layerMap.get(layer);
          if (rec && rec.shownIdx >= 0) {
            rec.imgs[rec.shownIdx].style.visibility = 'hidden';
            rec.shownIdx = -1;
          }
          this._mouthShown.delete(layer);
        }
      }
    }

    _hideAllMouth() {
      for (const [layer] of this._mouthShown) {
        const rec = this.layerMap.get(layer);
        if (rec && rec.shownIdx >= 0) {
          rec.imgs[rec.shownIdx].style.visibility = 'hidden';
          rec.shownIdx = -1;
        }
      }
      this._mouthShown.clear();
    }

    /** @param {BodyPose} pose */
    applyPose(pose) {
      const r = pose.root;
      this.wrapper.style.transform =
        `translate(${r.x.toFixed(2)}px, ${r.y.toFixed(2)}px) rotate(${r.rot.toFixed(4)}rad) scale(${r.scaleX.toFixed(4)}, ${r.scaleY.toFixed(4)})`;

      const hair = this.layerMap.get('hair');
      if (hair) {
        hair.el.style.transform =
          `translateX(${pose.hair.x.toFixed(2)}px) rotate(${pose.hair.rot.toFixed(4)}rad)`;
      }
      const hb = this.layerMap.get('hair_back');
      if (hb) {
        hb.el.style.transform =
          `translateX(${pose.hairBack.x.toFixed(2)}px) rotate(${pose.hairBack.rot.toFixed(4)}rad)`;
      }

      // 腕: 肩の弾み（translateY）＋肩ピボットの剛体回転（DOMは段階1フォールバック）
      const armL = this.layerMap.get('arm_l');
      const armR = this.layerMap.get('arm_r');
      if (armL) {
        armL.el.style.transform = pose.arms
          ? `translateY(${pose.arms.left.lift.toFixed(2)}px) rotate(${pose.arms.left.rigid.toFixed(4)}rad)`
          : '';
      }
      if (armR) {
        armR.el.style.transform = pose.arms
          ? `translateY(${pose.arms.right.lift.toFixed(2)}px) rotate(${pose.arms.right.rigid.toFixed(4)}rad)`
          : '';
      }
    }

    destroy() {
      this.rootEl.remove();
      this.layerMap.clear();
      this._mouthShown.clear();
    }
  }

  NS.DomCompositor = DomCompositor;
})(window.AnimLab);
