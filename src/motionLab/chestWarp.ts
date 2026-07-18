export interface MotionLabRasterBounds {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface MotionLabChestWarpRegion {
  centerX: number;
  centerY: number;
  radiusX: number;
  radiusY: number;
}

/**
 * 胸部の局所ワープ範囲を決める。
 * chest.png がある場合は位置決め用ガイドとしてだけ使い、無い場合は素体の
 * 上半身から控えめな既定範囲を推定する。
 */
export function resolveMotionLabChestWarpRegion(
  width: number,
  height: number,
  bodyBounds: MotionLabRasterBounds,
  guideBounds?: MotionLabRasterBounds | null,
): MotionLabChestWarpRegion {
  const guideUsable = !!guideBounds && guideBounds.w > 2 && guideBounds.h > 2;
  if (guideUsable) {
    const guide = guideBounds;
    return {
      centerX: guide.x + guide.w * 0.5,
      centerY: guide.y + guide.h * 0.5,
      radiusX: Math.max(width * 0.035, guide.w * 0.34),
      radiusY: Math.max(height * 0.025, guide.h * 0.42),
    };
  }

  return {
    centerX: bodyBounds.x + bodyBounds.w * 0.5,
    centerY: bodyBounds.y + bodyBounds.h * 0.5,
    radiusX: Math.max(width * 0.05, bodyBounds.w * 0.16),
    radiusY: Math.max(height * 0.035, bodyBounds.h * 0.1),
  };
}

/** ガウス重みで境界を固定した、出力画素から入力画素へのY座標変換。 */
export function motionLabChestWarpSourceY(
  x: number,
  y: number,
  region: MotionLabChestWarpRegion,
  offsetY: number,
): number {
  const nx = (x - region.centerX) / Math.max(1, region.radiusX);
  const ny = (y - region.centerY) / Math.max(1, region.radiusY);
  const weight = Math.exp(-0.5 * (nx * nx + ny * ny));
  return y - offsetY * weight;
}

/** 3σ外は変位がほぼゼロなので、処理対象を上半身の小さな矩形に限定する。 */
export function motionLabChestWarpBounds(
  width: number,
  height: number,
  region: MotionLabChestWarpRegion,
): MotionLabRasterBounds {
  const x0 = Math.max(0, Math.floor(region.centerX - region.radiusX * 3));
  const y0 = Math.max(0, Math.floor(region.centerY - region.radiusY * 3));
  const x1 = Math.min(width, Math.ceil(region.centerX + region.radiusX * 3));
  const y1 = Math.min(height, Math.ceil(region.centerY + region.radiusY * 3));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}
