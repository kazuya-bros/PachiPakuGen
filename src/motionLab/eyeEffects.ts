export interface MotionLabEyeRegion {
  x: number;
  y: number;
  w: number;
  h: number;
  pixels: number;
}

export interface MotionLabWetnessGeometry {
  centerX: number;
  surfaceCenterY: number;
  surfaceRadiusX: number;
  surfaceRadiusY: number;
  crescentCenterY: number;
  crescentRadiusX: number;
  crescentRadiusY: number;
  crescentLineWidth: number;
}

export interface MotionLabBrowMotionState {
  liftPx: number;
  /** 両眉の内側を持ち上げる基本角度。左右で符号を反転して使う。 */
  tiltDeg: number;
  /** 発話中だけ左右差を作る、ごく小さい共通回転成分。 */
  asymmetryDeg: number;
}

const TAU = Math.PI * 2;
export const MOTION_LAB_GAZE_MAX_RANGE_PX = 4.8;
export const MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA = 0.04;
export const MOTION_LAB_WETNESS_MIN_ALPHA = 0.34;
export const MOTION_LAB_WETNESS_MAX_ALPHA = 0.52;
export const MOTION_LAB_BROW_MAX_LIFT_PX = 4.2;
export const MOTION_LAB_BROW_MAX_TILT_DEG = 1.5;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * 虹彩画像のアルファから左右の目を検出する。
 * 全キャンバス幅ではなく実際の虹彩幅を、視線と収縮の安全な可動域に使う。
 */
export function detectMotionLabEyeRegions(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  alphaThreshold = 8,
): MotionLabEyeRegion[] {
  if (width <= 0 || height <= 0 || pixels.length < width * height * 4) return [];
  // irides.png can contain detached highlights. A horizontal projection keeps
  // those fragments with the correct eye and avoids allocating a full-frame
  // flood-fill queue for large source images.
  const columns = new Uint32Array(width);
  let maxColumnPixels = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (pixels[(y * width + x) * 4 + 3] <= alphaThreshold) continue;
      const count = ++columns[x];
      maxColumnPixels = Math.max(maxColumnPixels, count);
    }
  }
  if (maxColumnPixels === 0) return [];

  // A single isolated alpha pixel is a common antialiasing/export artefact,
  // never a usable iris column.
  const significantColumnPixels = Math.max(2, Math.ceil(maxColumnPixels * 0.08));
  let minX = 0;
  while (minX < width && columns[minX] < significantColumnPixels) minX += 1;
  let maxX = width - 1;
  while (maxX >= minX && columns[maxX] < significantColumnPixels) maxX -= 1;
  if (maxX - minX < 3) return [];

  const span = maxX - minX;
  const searchStart = minX + Math.floor(span * 0.3);
  const searchEnd = minX + Math.ceil(span * 0.7);
  const center = (minX + maxX) * 0.5;
  let splitX = Math.round(center);
  for (let x = searchStart; x <= searchEnd; x += 1) {
    if (
      columns[x] < columns[splitX]
      || (columns[x] === columns[splitX] && Math.abs(x - center) < Math.abs(splitX - center))
    ) {
      splitX = x;
    }
  }

  const regionForRange = (startX: number, endX: number): MotionLabEyeRegion | null => {
    let regionMinX = width;
    let regionMinY = height;
    let regionMaxX = -1;
    let regionMaxY = -1;
    let count = 0;
    for (let x = startX; x <= endX; x += 1) {
      if (columns[x] < significantColumnPixels) continue;
      for (let y = 0; y < height; y += 1) {
        if (pixels[(y * width + x) * 4 + 3] <= alphaThreshold) continue;
        regionMinX = Math.min(regionMinX, x);
        regionMinY = Math.min(regionMinY, y);
        regionMaxX = Math.max(regionMaxX, x);
        regionMaxY = Math.max(regionMaxY, y);
        count += 1;
      }
    }
    if (count < 4 || regionMaxX < regionMinX || regionMaxY < regionMinY) return null;
    return {
      x: regionMinX,
      y: regionMinY,
      w: regionMaxX - regionMinX + 1,
      h: regionMaxY - regionMinY + 1,
      pixels: count,
    };
  };

  const left = regionForRange(minX, splitX);
  const right = regionForRange(splitX + 1, maxX);
  const valleyLimit = Math.max(1, Math.floor(maxColumnPixels * 0.25));
  if (columns[splitX] > valleyLimit) {
    const single = regionForRange(minX, maxX);
    return single ? [single] : [];
  }
  return left && right ? [left, right] : [];
}

/** 100%までは控えめ。300%で最大4.8pxまで確認・強調できる。 */
export function motionLabGazeRangePx(
  regions: readonly MotionLabEyeRegion[],
  strength: number,
): number {
  if (regions.length === 0 || !Number.isFinite(strength) || strength <= 0) return 0;
  const averageWidth = regions.reduce((sum, region) => sum + region.w, 0) / regions.length;
  const baseRange = clamp(averageWidth * 0.03, 0.45, 1.6);
  return clamp(baseRange * strength, 0, MOTION_LAB_GAZE_MAX_RANGE_PX);
}

/** Deterministic horizontal-only gaze; vertical motion is deliberately zero. */
export function motionLabHorizontalGazeAt(
  phase: number,
  regions: readonly MotionLabEyeRegion[],
  strength: number,
): { x: number; y: 0 } {
  return {
    x: Math.sin(Number.isFinite(phase) ? phase : 0) * motionLabGazeRangePx(regions, strength),
    y: 0,
  };
}

/** 虹彩全体を各目の中心で伸縮する「瞳の呼吸」。中間値は抑え、最大で±4%。 */
export function motionLabIrisBreathScale(
  timeMs: number,
  strength: number,
  periodMsOverride?: number,
): number {
  const normalized = clamp(Number.isFinite(strength) ? strength : 0, 0, 1);
  const amount = normalized * normalized;
  if (amount <= 0) return 1;
  const periodMs = periodMsOverride && periodMsOverride > 0 ? periodMsOverride : 5200;
  const phase = (Math.max(0, timeMs) / periodMs) * TAU + 0.65;
  return 1 + Math.sin(phase) * MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA * amount;
}

/**
 * 下まぶた側の濡れ反射。
 * 中間値は自然な薄さに保ち、最大値では縮小表示でも見失わない34〜52%。
 */
export function motionLabWetnessOpacity(
  timeMs: number,
  strength: number,
  periodMsOverride?: number,
): number {
  const normalized = clamp(Number.isFinite(strength) ? strength : 0, 0, 1);
  const amount = Math.pow(normalized, 2.2);
  if (amount <= 0) return 0;
  const periodMs = periodMsOverride && periodMsOverride > 0 ? periodMsOverride : 4600;
  const phase = (Math.max(0, timeMs) / periodMs) * TAU + 1.1;
  return amount * (0.43 + Math.sin(phase) * 0.09);
}

export function motionLabWetnessGeometry(
  region: Pick<MotionLabEyeRegion, "x" | "y" | "w" | "h">,
): MotionLabWetnessGeometry {
  return {
    centerX: region.x + region.w * 0.5,
    surfaceCenterY: region.y + region.h * 0.73,
    surfaceRadiusX: Math.max(2, region.w * 0.38),
    surfaceRadiusY: Math.max(1.5, region.h * 0.11),
    crescentCenterY: region.y + region.h * 0.68,
    crescentRadiusX: Math.max(2, region.w * 0.33),
    crescentRadiusY: Math.max(1.5, region.h * 0.16),
    crescentLineWidth: Math.max(1.8, region.h * 0.05),
  };
}

/**
 * 両眉を同じ表情のまま、待機中はごく小さく、発話中は少しだけ持ち上げる。
 * 眉形そのものを感情表現へ変えず、素材の存在感だけを足すための値を返す。
 */
export function motionLabBrowMotion(
  timeMs: number,
  voice: number,
  strength: number,
): MotionLabBrowMotionState {
  const normalized = clamp(Number.isFinite(strength) ? strength : 0, 0, 1.5);
  if (normalized <= 0) return { liftPx: 0, tiltDeg: 0, asymmetryDeg: 0 };
  const safeTime = Math.max(0, Number.isFinite(timeMs) ? timeMs : 0);
  const safeVoice = clamp(Number.isFinite(voice) ? voice : 0, 0, 1);
  const idle = 0.5 + 0.5 * Math.sin((safeTime / 3600) * TAU + 0.35);
  const liftPulse = 0.5 + 0.5 * Math.sin((safeTime / 820) * TAU + 1.2);
  // 上下とは周期・位相をずらし、眉全体が一枚の板のように動く印象を避ける。
  const tiltPulse = 0.5 + 0.5 * Math.sin((safeTime / 1240) * TAU + 1.9);
  const liftResponse = clamp(0.13 * idle + safeVoice * (0.42 + 0.45 * liftPulse), 0, 1);
  const tiltResponse = clamp(0.05 * idle + safeVoice * (0.22 + 0.44 * tiltPulse), 0, 1);
  const asymmetry = Math.sin((safeTime / 1750) * TAU + 0.55);
  return {
    liftPx: MOTION_LAB_BROW_MAX_LIFT_PX * normalized * liftResponse,
    tiltDeg: MOTION_LAB_BROW_MAX_TILT_DEG * normalized * tiltResponse,
    asymmetryDeg: MOTION_LAB_BROW_MAX_TILT_DEG * normalized * safeVoice * 0.18 * asymmetry,
  };
}

/** 画面左・右の眉へ、内側上がり＋わずかな左右差を適用する角度。 */
export function motionLabBrowRotationDeg(
  side: "left" | "right",
  motion: Pick<MotionLabBrowMotionState, "tiltDeg" | "asymmetryDeg">,
): number {
  return (side === "left" ? -motion.tiltDeg : motion.tiltDeg) + motion.asymmetryDeg;
}
