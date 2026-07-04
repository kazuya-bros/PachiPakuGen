/**
 * 腕揺れ（待機モーション拡張・段階2）
 * - ArmSway: B3と同じ角度チェーン物理を左右の腕に適用する純ロジック
 *   （体方式B0〜B3から独立。どの組合せでも併用できる）
 * - ArmMeshRuntime: pixi用。腕画像の不透明bboxを切り出し、
 *   肩（bbox上端中央）固定のメッシュチェーンとして描画
 * - DOMコンポジターでは剛体回転（チェーン角の平均）にフォールバック（段階1相当）
 */
(function (NS) {
  'use strict';

  const U = NS.U;

  function mkChain(n) {
    return { angles: new Float32Array(n), omegas: new Float32Array(n), t: Math.random() * 100 };
  }

  class ArmSway {
    constructor() {
      const n = NS.P.arm.segments;
      this.left = mkChain(n);
      this.right = mkChain(n);
      this.prevRootX = null;
      this.prevRootY = null;
      this.lift = { x: 0, v: 0 }; // 肩の上下（両肩共通、px）
    }

    /**
     * @param {number} dt 秒
     * @param {number} rootX 体（root）の現在X（px）
     * @param {number} rootY 体（root）の現在Y（px）
     * @param {boolean} speechStarted このフレームで発話開始（肩バウンスの撃力）
     * @returns {{left:{angles:number[],rigid:number,lift:number}, right:{...}}|null}
     */
    update(dt, rootX, rootY, speechStarted) {
      const P = NS.P.arm;
      if (!P.enabled) { this.prevRootX = rootX; this.prevRootY = rootY; return null; }

      const vx = this.prevRootX === null || dt <= 0 ? 0 : (rootX - this.prevRootX) / dt;
      const vy = this.prevRootY === null || dt <= 0 ? 0 : (rootY - this.prevRootY) / dt;
      this.prevRootX = rootX;
      this.prevRootY = rootY;

      // ===== 肩の弾み（縦方向の遅延追従＋発話バウンス） =====
      if (P.liftEnabled) {
        if (speechStarted) this.lift.v += P.liftBounce; // 撃力（px/s）
        const driveY = U.clamp(-P.liftCoupling * vy, -P.liftMax, P.liftMax);
        const kL = P.k * 0.8, cL = P.c * 0.9;
        const hL = Math.min(dt, 0.1);
        const a = -kL * (this.lift.x - driveY) - cL * this.lift.v;
        this.lift.v += a * hL;
        this.lift.x = U.clamp(this.lift.x + this.lift.v * hL, -P.liftMax, P.liftMax);
      } else {
        this.lift.x *= Math.max(0, 1 - dt * 8); // OFF時は滑らかにゼロへ
        this.lift.v = 0;
      }

      const dtc = Math.min(dt, 0.1);
      const steps = dtc > 1 / 60 ? 2 : 1;
      const h = dtc / steps;
      const drive = U.clamp(-P.coupling * vx * 0.05, -P.maxAngle, P.maxAngle);

      const out = {};
      for (const side of ['left', 'right']) {
        const ch = this[side];
        ch.t += dtc;
        // 左右で位相をずらした常時微揺れ（完全同期だと機械的に見える）
        const noise = U.noise1d(ch.t * 0.5 + (side === 'left' ? 0 : 37)) * P.noise;
        const n = ch.angles.length;
        for (let s = 0; s < steps; s++) {
          for (let i = 0; i < n; i++) {
            const target = i === 0 ? drive + noise : ch.angles[i - 1];
            const k = P.k * (1 - 0.6 * (i / n)); // 手先ほど柔らかい
            const a = -k * (ch.angles[i] - target) - P.c * ch.omegas[i];
            ch.omegas[i] += a * h;
            ch.angles[i] += ch.omegas[i] * h;
            ch.angles[i] = U.clamp(ch.angles[i], -P.maxAngle, P.maxAngle);
          }
        }
        let sum = 0;
        for (let i = 0; i < n; i++) sum += ch.angles[i];
        out[side] = { angles: Array.from(ch.angles), rigid: sum / n, lift: this.lift.x };
      }
      return out;
    }
  }

  /**
   * pixi用: 腕のbbox切り出しメッシュチェーン
   */
  class ArmMeshRuntime {
    /**
     * @param {Object} pixiCtx { container, textureFor(bitmap) }
     * @param {CharacterAssets} assets
     * @param {string} layerName 'arm_l' | 'arm_r'
     */
    constructor(pixiCtx, assets, layerName) {
      const layer = assets.byName.get(layerName);
      this.available = !!layer;
      if (!this.available) return;

      const frame = layer.frames[0];
      this.bbox = U.alphaBBox(frame.bitmap);
      this.segments = NS.P.arm.segments;

      const base = pixiCtx.textureFor(frame.bitmap);
      const tex = new PIXI.Texture({
        source: base.source,
        frame: new PIXI.Rectangle(this.bbox.x, this.bbox.y, this.bbox.w, this.bbox.h),
      });
      this.mesh = new PIXI.MeshPlane({
        texture: tex,
        verticesX: 3,
        verticesY: this.segments + 1,
      });
      this.mesh.position.set(this.bbox.x, this.bbox.y);
      this.mesh.zIndex = NS.LAYER_Z_INDEX[layerName] * 10;
      pixiCtx.container.addChild(this.mesh);
      this.cols = 2; // verticesX - 1
      this.update(null, 0);
    }

    /**
     * @param {number[]|null} angles 肩→手先のチェーン角。nullで直立
     * @param {number} liftY 肩の上下オフセット（自然座標px）
     */
    update(angles, liftY) {
      if (!this.available) return;
      this.mesh.position.y = this.bbox.y + (liftY || 0);
      const buf = this.mesh.geometry.getBuffer('aPosition');
      const data = buf.data;
      const rows = this.segments;
      const stepY = this.bbox.h / rows;
      const cx = this.bbox.w / 2;

      // 肩（row0 = bbox上端中央）固定。各行は親行から累積回転で下る
      let originX = cx, originY = 0, cum = 0;
      let i = 0;
      for (let r = 0; r <= rows; r++) {
        if (r > 0) {
          cum += angles ? angles[Math.min(r - 1, angles.length - 1)] : 0;
          originX += Math.sin(cum) * stepY;
          originY += Math.cos(cum) * stepY;
        }
        const cos = Math.cos(cum);
        for (let c = 0; c <= this.cols; c++) {
          const lx = (c * this.bbox.w) / this.cols - cx;
          data[i++] = originX + lx * cos;
          data[i++] = originY;
        }
      }
      buf.update();
    }

    destroy() {
      if (this.mesh) {
        this.mesh.removeFromParent();
        this.mesh.destroy({ children: true });
        this.mesh = null;
      }
    }
  }

  NS.ArmSway = ArmSway;
  NS.ArmMeshRuntime = ArmMeshRuntime;
})(window.AnimLab);
