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
  /** 胸より下方向の減衰半径 */
  radiusY: number;
  /**
   * 胸より上方向の減衰半径。
   * 切り出し範囲内でも顔側へ広がりすぎないよう、未指定なら radiusY×0.7。
   */
  radiusYUp?: number;
}

function chestRegion(
  centerX: number,
  centerY: number,
  radiusX: number,
  radiusY: number,
  radiusYUp: number,
): MotionLabChestWarpRegion {
  return { centerX, centerY, radiusX, radiusY, radiusYUp };
}

/** 画素Yに応じた縦方向のガウス半径（上方向をやや短くする）。 */
export function motionLabChestWarpVerticalRadius(
  region: MotionLabChestWarpRegion,
  y: number,
): number {
  const up = region.radiusYUp ?? region.radiusY * 0.7;
  return Math.max(1, y < region.centerY ? up : region.radiusY);
}

/**
 * 胸部の局所ワープ範囲を決める。
 *
 * chest.png（STEP4の胸部切り出し）がある場合は、その不透明領域を唯一の根拠にする。
 * 位置の「推定し直し」はしない。無い場合だけ素体bboxから控えめな既定範囲を使う。
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
    // 切り出しマスクのbboxそのものへフィットさせる（中心＝切り出し中心）
    const centerX = guide.x + guide.w * 0.5;
    const centerY = guide.y + guide.h * 0.5;
    // bboxの半分より少し広めに取り、縁で急に切れないようにする
    const radiusX = Math.max(width * 0.02, guide.w * 0.42);
    const radiusY = Math.max(height * 0.015, guide.h * 0.42);
    const radiusYUp = Math.max(height * 0.012, guide.h * 0.34);
    return chestRegion(centerX, centerY, radiusX, radiusY, radiusYUp);
  }

  // ガイド無し: 胴体中央やや上を既定の胸位置とする
  return chestRegion(
    bodyBounds.x + bodyBounds.w * 0.5,
    bodyBounds.y + bodyBounds.h * 0.42,
    Math.max(width * 0.045, bodyBounds.w * 0.16),
    Math.max(height * 0.028, bodyBounds.h * 0.055),
    Math.max(height * 0.02, bodyBounds.h * 0.038),
  );
}

/** ガウス重みで境界を固定した、出力画素から入力画素へのY座標変換。 */
export function motionLabChestWarpSourceY(
  x: number,
  y: number,
  region: MotionLabChestWarpRegion,
  offsetY: number,
): number {
  const nx = (x - region.centerX) / Math.max(1, region.radiusX);
  const ny = (y - region.centerY) / motionLabChestWarpVerticalRadius(region, y);
  const weight = Math.exp(-0.5 * (nx * nx + ny * ny));
  return y - offsetY * weight;
}

/** 3σ外は変位がほぼゼロなので、処理対象を胸周辺の小さな矩形に限定する。 */
export function motionLabChestWarpBounds(
  width: number,
  height: number,
  region: MotionLabChestWarpRegion,
): MotionLabRasterBounds {
  const up = region.radiusYUp ?? region.radiusY * 0.7;
  const x0 = Math.max(0, Math.floor(region.centerX - region.radiusX * 3));
  const y0 = Math.max(0, Math.floor(region.centerY - up * 3));
  const x1 = Math.min(width, Math.ceil(region.centerX + region.radiusX * 3));
  const y1 = Math.min(height, Math.ceil(region.centerY + region.radiusY * 3));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}
