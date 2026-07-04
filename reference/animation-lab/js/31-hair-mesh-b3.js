/**
 * B3 髪メッシュランタイム（pixi専用）
 * hair.png を縦方向に分割した MeshPlane に、B3方式が計算した
 * 角度チェーン（根元固定・毛先ほど大きく曲がる）を適用する。
 */
(function (NS) {
  'use strict';

  class HairMeshRuntime {
    /**
     * @param {Object} pixiCtx { container, textureFor(bitmap) }
     * @param {CharacterAssets} assets
     * @param {number} segments 縦分割数
     */
    constructor(pixiCtx, assets, segments) {
      this.segments = segments;
      const hair = assets.byName.get('hair');
      this.available = !!hair;
      if (!this.available) return;

      const tex = pixiCtx.textureFor(hair.frames[0].bitmap);
      this.w = assets.width;
      this.h = assets.height;
      this.mesh = new PIXI.MeshPlane({
        texture: tex,
        verticesX: 4,                 // 横方向も少し分割して曲げを滑らかに
        verticesY: segments + 1,
      });
      this.mesh.zIndex = NS.LAYER_Z_INDEX.hair * 10;
      pixiCtx.container.addChild(this.mesh);
      this.cols = 3; // verticesX - 1
    }

    /**
     * @param {number[]} angles 各行のバネ角（rad, 根元→毛先）
     */
    update(angles) {
      if (!this.available || !angles) return;
      const buf = this.mesh.geometry.getBuffer('aPosition');
      const data = buf.data;
      const rows = this.segments;
      const stepY = this.h / rows;
      const cx = this.w / 2;

      // 根元(row0)は固定。各行は親行の位置から累積回転した段ベクトルで下る
      let originX = cx, originY = 0, cum = 0;
      let i = 0;
      for (let r = 0; r <= rows; r++) {
        if (r > 0) {
          cum += angles[r - 1];
          originX += Math.sin(cum) * stepY;
          originY += Math.cos(cum) * stepY;
        }
        const cos = Math.cos(cum), sin = Math.sin(cum);
        for (let c = 0; c <= this.cols; c++) {
          const lx = (c * this.w) / this.cols - cx; // 中心からの横オフセット
          data[i++] = originX + lx * cos;
          data[i++] = originY - lx * sin * 0.15; // 横方向の傾きは控えめに
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

  NS.HairMeshRuntime = HairMeshRuntime;
})(window.AnimLab);
