import {
  alphaBBox,
  chainAverage,
  chainFoldOffsets,
  clampDt,
  createArmSway,
  createChain,
  createHairStrandSpring,
  detectHairStrands,
  envelopeStep,
  noise1d,
  noise1dLoop,
  smoothDamp,
  snapAngularRateToPeriod,
  snapCyclesToPeriod,
  springStep,
  stepHairStrandSpring,
  stepChain,
  updateArmSway,
} from "../motionLabPhysics";
import {
  motionLabChestWarpBounds,
  motionLabChestWarpVerticalRadius,
  resolveMotionLabChestWarpRegion,
} from "./chestWarp";
import {
  detectMotionLabEyeRegions,
  MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA,
  MOTION_LAB_WETNESS_MAX_ALPHA,
  motionLabBrowMotion,
  motionLabBrowRotationDeg,
  motionLabHorizontalGazeAt,
  motionLabIrisBreathScale,
  motionLabWetnessGeometry,
  motionLabWetnessOpacity,
  type MotionLabEyeRegion,
} from "./eyeEffects";
import type {
  MotionLabImageSet,
  MotionLabLayerMode,
  MotionLabLayerTransform,
  MotionLabMethod,
  MotionLabMouthKey,
  MotionLabMouthRuntime,
  MotionLabChainWarpOptions,
  MotionLabPartsResult,
  MotionLabPhysicsState,
  MotionLabPreset,
  MotionLabRenderSettings,
  MotionLabTimelineEvent,
  MotionLabVerdict,
} from "./types";
import {
  MOTION_LAB_ARM_DEFAULTS,
  MOTION_LAB_BLINK_DEFAULTS,
  MOTION_LAB_CHEST_DEFAULTS,
  MOTION_LAB_DEFAULT_DRAW_ORDER,
  MOTION_LAB_DEPTH_DEFAULTS,
  MOTION_LAB_DURATION_MS,
  MOTION_LAB_EAR_TWITCH,
  MOTION_LAB_GAZE_DEFAULTS,
  MOTION_LAB_HAIR_SEGMENTS,
  MOTION_LAB_NOD_DEFAULTS,
  MOTION_LAB_PARALLAX_DEFAULTS,
  MOTION_LAB_PRESENCE_DEFAULTS,
  MOTION_LAB_PRESET_FACTORS,
  MOTION_LAB_SWAY_DEFAULTS,
  MOTION_LAB_TARGET_OPEN,
  MOTION_LAB_TIMELINE,
} from "./constants";
import { stepMotionLabBlink } from "./blinkState";
import {
  motionLabInitialEarTwitchWait,
  motionLabEarTwitchImpulse,
  motionLabNextEarTwitchWait,
} from "./earTwitch";

/**
 * パーツ描画順の解決: layer-order.json（Step4のレイヤー調整由来）があればそれを優先し、
 * 記載のない既定グループは既定順の相対位置へ補完する。個別の sway_* も
 * そのままの位置で描画し、旧形式の sways グループも受け付ける。
 * 無ければ従来の固定z順（armBehindBody で腕をbody背面へ）。
 */
export function drawMotionLabOrderedLayers(
  layerOrder: string[],
  armBehindBody: boolean,
  draws: Record<string, () => void>,
) {
  let order: string[];
  if (layerOrder.length > 0) {
    // 壊れたlayer-order.jsonに同じキーが複数回あっても二重描画しない。
    const seen = new Set<string>();
    order = [];
    for (const key of layerOrder) {
      if (!(key in draws) || seen.has(key)) continue;
      seen.add(key);
      order.push(key);
    }
    for (const key of MOTION_LAB_DEFAULT_DRAW_ORDER) {
      if (order.includes(key)) continue;
      // 既定順で直前にあるグループの直後へ挿入（見つからなければ末尾）
      const defaultIndex = MOTION_LAB_DEFAULT_DRAW_ORDER.indexOf(key);
      let insertAt = order.length;
      for (let i = defaultIndex - 1; i >= 0; i -= 1) {
        const at = order.indexOf(MOTION_LAB_DEFAULT_DRAW_ORDER[i]);
        if (at >= 0) {
          insertAt = at + 1;
          break;
        }
      }
      order.splice(insertAt, 0, key);
    }
  } else if (armBehindBody) {
    order = ["hair_back", "arm_l", "arm_r", "body", "chest", "sways", "eye", "mouth", "hair"];
  } else {
    order = [...MOTION_LAB_DEFAULT_DRAW_ORDER];
  }
  for (const key of order) draws[key]?.();
}
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function motionLabTimelineAt(
  elapsedMs: number,
  timeline: MotionLabTimelineEvent[] = MOTION_LAB_TIMELINE,
  durationMs: number = MOTION_LAB_DURATION_MS,
): { mouth: MotionLabMouthKey; energy: number; loopMs: number } {
  const loopMs = ((elapsedMs % durationMs) + durationMs) % durationMs;
  let current = timeline[0];
  for (const event of timeline) {
    if (event.timeMs <= loopMs) current = event;
  }
  return { mouth: current.mouth, energy: current.energy, loopMs };
}

export function pickMotionLabMouthFrame(frames: HTMLImageElement[] | undefined, openY: number): HTMLImageElement | null {
  if (!frames?.length) return null;
  const index = frames.length === 1 ? 0 : Math.round(clamp(openY, 0, 1) * (frames.length - 1));
  return frames[index] ?? frames[0] ?? null;
}

/**
 * 共通開度軸 openY → 母音別連番のフレーム比率。
 * 各母音の連番は「閉じ→その母音の完成形」で、狭母音（い等）の狭さは
 * 素材の最終フレーム自体に焼き込まれている。openY（い=0.5上限）をそのまま
 * スクラブに使うと狭さが二重掛けになり最終フレームへ到達できないため、
 * 母音別目標開度で正規化して定常時に必ず完成形（最終フレーム）へ届かせる。
 */
export function motionLabMouthFrameRatio(openY: number, key: MotionLabMouthKey): number {
  const fullOpen = MOTION_LAB_TARGET_OPEN[key];
  return fullOpen > 0 ? clamp(openY / fullOpen, 0, 1) : clamp(openY, 0, 1);
}

export function loadMotionLabImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("画像プレビューを読み込めませんでした"));
    image.src = src;
  });
}

export function validMotionLabPreset(value: unknown): MotionLabPreset {
  return value === "calm" || value === "lively" || value === "normal" ? value : "normal";
}

export function validMotionLabLayerMode(value: unknown): MotionLabLayerMode {
  return value === "simple" || value === "mesh" || value === "spring" ? value : "spring";
}

export function validMotionLabMethod(value: unknown): MotionLabMethod {
  return value === "baseline" || value === "bridge" || value === "smooth" ? value : "smooth";
}

export function validMotionLabVerdict(value: unknown): MotionLabVerdict {
  return value === "promising" || value === "hold" || value === "reject" || value === "undecided"
    ? value
    : "undecided";
}

export function drawMotionLabLayer(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement | HTMLCanvasElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  alpha = 1,
) {
  const pivotX = width * 0.5;
  const pivotY = height * 0.58;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.translate(pivotX + transform.x, pivotY + transform.y);
  ctx.rotate((transform.rotationDeg * Math.PI) / 180);
  if (transform.skewX) ctx.transform(1, 0, transform.skewX, 1, 0, 0);
  ctx.scale(transform.scaleX, transform.scaleY);
  ctx.drawImage(image, -pivotX, -pivotY, width, height);
  ctx.restore();
}

interface MotionLabRasterSource {
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
}

const motionLabRasterSourceCache = new WeakMap<HTMLImageElement, Map<string, MotionLabRasterSource>>();
const motionLabEyeRegionCache = new WeakMap<HTMLImageElement, Map<string, MotionLabEyeRegion[]>>();

function motionLabRasterSource(
  image: HTMLImageElement,
  width: number,
  height: number,
): MotionLabRasterSource | null {
  const cacheKey = `${width}x${height}`;
  let imageCache = motionLabRasterSourceCache.get(image);
  const cached = imageCache?.get(cacheKey);
  if (cached) return cached;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const sourceCtx = canvas.getContext("2d", { willReadFrequently: true });
  if (!sourceCtx) return null;
  sourceCtx.drawImage(image, 0, 0, width, height);
  const source: MotionLabRasterSource = {
    width,
    height,
    pixels: sourceCtx.getImageData(0, 0, width, height).data,
  };
  imageCache ??= new Map();
  imageCache.set(cacheKey, source);
  motionLabRasterSourceCache.set(image, imageCache);
  return source;
}

function motionLabEyeRegions(
  image: HTMLImageElement,
  width: number,
  height: number,
): MotionLabEyeRegion[] {
  const cacheKey = `${width}x${height}`;
  let imageCache = motionLabEyeRegionCache.get(image);
  const cached = imageCache?.get(cacheKey);
  if (cached) return cached;
  const source = motionLabRasterSource(image, width, height);
  const regions = source ? detectMotionLabEyeRegions(source.pixels, width, height) : [];
  imageCache ??= new Map();
  imageCache.set(cacheKey, regions);
  motionLabEyeRegionCache.set(image, imageCache);
  return regions;
}

function sampleMotionLabRaster(
  source: MotionLabRasterSource,
  sourceX: number,
  sourceY: number,
  output: Uint8ClampedArray,
  outputIndex: number,
) {
  // この変形はY方向だけなので、X補間は不要。縦2画素だけを混ぜて
  // ライブプレビューの画素処理量を抑える。
  const x = Math.round(clamp(sourceX, 0, source.width - 1));
  const y = clamp(sourceY, 0, source.height - 1);
  const y0 = Math.floor(y);
  const y1 = Math.min(source.height - 1, y0 + 1);
  const ty = y - y0;
  const weights = [1 - ty, ty];
  const indexes = [
    (y0 * source.width + x) * 4,
    (y1 * source.width + x) * 4,
  ];
  let alpha = 0;
  let red = 0;
  let green = 0;
  let blue = 0;
  for (let index = 0; index < 2; index += 1) {
    const sourceIndex = indexes[index];
    const weightedAlpha = (source.pixels[sourceIndex + 3] / 255) * weights[index];
    alpha += weightedAlpha;
    red += source.pixels[sourceIndex] * weightedAlpha;
    green += source.pixels[sourceIndex + 1] * weightedAlpha;
    blue += source.pixels[sourceIndex + 2] * weightedAlpha;
  }
  output[outputIndex] = alpha > 0 ? red / alpha : 0;
  output[outputIndex + 1] = alpha > 0 ? green / alpha : 0;
  output[outputIndex + 2] = alpha > 0 ? blue / alpha : 0;
  output[outputIndex + 3] = alpha * 255;
}

/**
 * bodyを一枚のまま局所変形する胸部追従。
 * chest.png は描画せず、存在する場合だけ範囲ガイドとして利用する。
 */
export function drawMotionLabChestWarp(
  ctx: CanvasRenderingContext2D,
  body: HTMLImageElement,
  guide: HTMLImageElement | null,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  offsetY: number,
  runtime: MotionLabMouthRuntime,
) {
  if (Math.abs(offsetY) < 0.02) {
    drawMotionLabLayer(ctx, body, width, height, transform);
    return;
  }

  const source = motionLabRasterSource(body, width, height);
  if (!source) {
    drawMotionLabLayer(ctx, body, width, height, transform);
    return;
  }

  let scratch = runtime.chestWarpScratch;
  if (!scratch || scratch.width !== width || scratch.height !== height) {
    scratch = document.createElement("canvas");
    scratch.width = width;
    scratch.height = height;
    runtime.chestWarpScratch = scratch;
  }
  const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
  if (!scratchCtx) {
    drawMotionLabLayer(ctx, body, width, height, transform);
    return;
  }
  scratchCtx.clearRect(0, 0, width, height);
  scratchCtx.drawImage(body, 0, 0, width, height);

  const region = resolveMotionLabChestWarpRegion(
    width,
    height,
    alphaBBox(body),
    guide ? alphaBBox(guide) : null,
  );
  const bounds = motionLabChestWarpBounds(width, height, region);
  if (bounds.w <= 0 || bounds.h <= 0) {
    drawMotionLabLayer(ctx, body, width, height, transform);
    return;
  }
  const warped = scratchCtx.getImageData(bounds.x, bounds.y, bounds.w, bounds.h);
  const xWeights = new Float32Array(bounds.w);
  const yWeights = new Float32Array(bounds.h);
  for (let localX = 0; localX < bounds.w; localX += 1) {
    const nx = (bounds.x + localX - region.centerX) / Math.max(1, region.radiusX);
    xWeights[localX] = Math.exp(-0.5 * nx * nx);
  }
  for (let localY = 0; localY < bounds.h; localY += 1) {
    const y = bounds.y + localY;
    const radiusY = motionLabChestWarpVerticalRadius(region, y);
    const ny = (y - region.centerY) / radiusY;
    yWeights[localY] = Math.exp(-0.5 * ny * ny);
  }
  for (let localY = 0; localY < bounds.h; localY += 1) {
    const y = bounds.y + localY;
    for (let localX = 0; localX < bounds.w; localX += 1) {
      const x = bounds.x + localX;
      const sourceY = y - offsetY * xWeights[localX] * yWeights[localY];
      sampleMotionLabRaster(source, x, sourceY, warped.data, (localY * bounds.w + localX) * 4);
    }
  }
  scratchCtx.putImageData(warped, bounds.x, bounds.y);
  drawMotionLabLayer(ctx, scratch, width, height, transform);
}

/**
 * B3メッシュ髪揺れのCanvas 2D描画: チェーン角の折れ線オフセットを
 * 縦ストリップに線形補間して適用（31-hair-mesh-b3.js のメッシュ変形と同じ数式）
 */
export function drawMotionLabChainWarp(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  options: MotionLabChainWarpOptions,
) {
  const pivotX = width * 0.5;
  const pivotY = height * 0.58;
  const stripCount = options.stripCount ?? 24;
  const alpha = options.alpha ?? 1;
  const segments = options.angles.length;
  const span = height * (1 - options.rootYRatio);
  const rows = chainFoldOffsets(options.angles, span);
  // 房ごと描画: この房のx範囲だけをスライスする（未指定なら全幅=従来B3）
  const sourceX = options.xRange?.x0 ?? 0;
  const sourceW = options.xRange ? options.xRange.x1 - options.xRange.x0 + 1 : width;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.translate(pivotX + transform.x, pivotY + transform.y);
  ctx.rotate((transform.rotationDeg * Math.PI) / 180);
  if (transform.skewX) ctx.transform(1, 0, transform.skewX, 1, 0, 0);
  ctx.scale(transform.scaleX, transform.scaleY);
  for (let index = 0; index < stripCount; index += 1) {
    const sourceY = Math.floor((height * index) / stripCount);
    const nextY = Math.floor((height * (index + 1)) / stripCount);
    const stripHeight = Math.max(1, nextY - sourceY);
    const centerYRatio = (sourceY + stripHeight * 0.5) / height;
    const tipRatio = clamp((centerYRatio - options.rootYRatio) / Math.max(0.001, 1 - options.rootYRatio), 0, 1);
    const pos = tipRatio * segments;
    const lower = Math.min(segments, Math.floor(pos));
    const upper = Math.min(segments, lower + 1);
    const frac = pos - lower;
    const dx = (rows[lower].dx + (rows[upper].dx - rows[lower].dx) * frac) * (options.offsetScale ?? 1);
    const dy = (rows[lower].dy + (rows[upper].dy - rows[lower].dy) * frac) * (options.offsetScale ?? 1);
    ctx.drawImage(
      image,
      sourceX,
      sourceY,
      sourceW,
      stripHeight,
      -pivotX + sourceX + dx,
      -pivotY + sourceY + dy,
      sourceW,
      stripHeight + 1,
    );
  }
  ctx.restore();
}

interface MotionLabMeshPoint {
  x: number;
  y: number;
}

/** Canvas 2D上で、元画像の三角形を変形後の三角形へテクスチャ付きで描く。 */
function drawMotionLabTexturedTriangle(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  drawWidth: number,
  drawHeight: number,
  source: [MotionLabMeshPoint, MotionLabMeshPoint, MotionLabMeshPoint],
  destination: [MotionLabMeshPoint, MotionLabMeshPoint, MotionLabMeshPoint],
) {
  const [s0, s1, s2] = source;
  const [d0, d1, d2] = destination;
  const determinant =
    s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y);
  if (Math.abs(determinant) < 1e-6) return;

  const a =
    (d0.x * (s1.y - s2.y) + d1.x * (s2.y - s0.y) + d2.x * (s0.y - s1.y)) /
    determinant;
  const b =
    (d0.y * (s1.y - s2.y) + d1.y * (s2.y - s0.y) + d2.y * (s0.y - s1.y)) /
    determinant;
  const c =
    (d0.x * (s2.x - s1.x) + d1.x * (s0.x - s2.x) + d2.x * (s1.x - s0.x)) /
    determinant;
  const d =
    (d0.y * (s2.x - s1.x) + d1.y * (s0.x - s2.x) + d2.y * (s1.x - s0.x)) /
    determinant;
  const e =
    (d0.x * (s1.x * s2.y - s2.x * s1.y) +
      d1.x * (s2.x * s0.y - s0.x * s2.y) +
      d2.x * (s0.x * s1.y - s1.x * s0.y)) /
    determinant;
  const f =
    (d0.y * (s1.x * s2.y - s2.x * s1.y) +
      d1.y * (s2.x * s0.y - s0.x * s2.y) +
      d2.y * (s0.x * s1.y - s1.x * s0.y)) /
    determinant;

  // 変形頂点は共有したまま、クリップだけ0.7px広げてAAの継ぎ目を隠す。
  const center = { x: (d0.x + d1.x + d2.x) / 3, y: (d0.y + d1.y + d2.y) / 3 };
  const expand = (point: MotionLabMeshPoint): MotionLabMeshPoint => {
    const vx = point.x - center.x;
    const vy = point.y - center.y;
    const length = Math.hypot(vx, vy);
    const scale = length > 1e-6 ? (length + 0.7) / length : 1;
    return { x: center.x + vx * scale, y: center.y + vy * scale };
  };
  const [clip0, clip1, clip2] = destination.map(expand) as [
    MotionLabMeshPoint,
    MotionLabMeshPoint,
    MotionLabMeshPoint,
  ];

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(clip0.x, clip0.y);
  ctx.lineTo(clip1.x, clip1.y);
  ctx.lineTo(clip2.x, clip2.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(a, b, c, d, e, f);
  // 各三角形の周辺だけを転送する。全画像を三角形数だけ再描画しないための軽量化。
  const displayLeft = Math.max(0, Math.min(s0.x, s1.x, s2.x) - 1);
  const displayTop = Math.max(0, Math.min(s0.y, s1.y, s2.y) - 1);
  const displayRight = Math.min(drawWidth, Math.max(s0.x, s1.x, s2.x) + 1);
  const displayBottom = Math.min(drawHeight, Math.max(s0.y, s1.y, s2.y) + 1);
  const displayWidth = displayRight - displayLeft;
  const displayHeight = displayBottom - displayTop;
  const naturalScaleX = (image.naturalWidth || drawWidth) / drawWidth;
  const naturalScaleY = (image.naturalHeight || drawHeight) / drawHeight;
  ctx.drawImage(
    image,
    displayLeft * naturalScaleX,
    displayTop * naturalScaleY,
    displayWidth * naturalScaleX,
    displayHeight * naturalScaleY,
    displayLeft,
    displayTop,
    displayWidth,
    displayHeight,
  );
  ctx.restore();
}

/**
 * ソフト房ブレンドワープ: 房を x範囲でハード分割せず、
 * 共有メッシュ頂点ごとに「房中心へのガウシアン重み」で複数房の変位をブレンドする。
 * 隣接セルが同じ変形頂点を参照するため、前髪のように上部が繋がった髪でも裂けない。
 */
export function drawMotionLabStrandBlendWarp(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  options: {
    rootYRatio: number;
    /** 各房の検出位置と、硬・柔2本のスカラーばねによる横変位 */
    strands: Array<{
      x: number;
      rootY: number;
      tipY: number;
      stiffDx: number;
      softDx: number;
    }>;
    /** 縦方向の目標メッシュ分割数（既存API互換） */
    stripCount?: number;
    /** 横方向の目標セル幅（既存API互換） */
    blockWidth?: number;
    /** 変位の倍率（揺れ幅スライダー用。1=既定） */
    offsetScale?: number;
    /** 根元から毛先へ変位を立ち上げる指数。長い後ろ髪は大きめにする。 */
    tipExponent?: number;
  },
) {
  const strands = options.strands;
  if (strands.length === 0) return;
  const pivotX = width * 0.5;
  const pivotY = height * 0.58;
  const requestedRows = options.stripCount ?? 20;
  const targetCellWidth = options.blockWidth ?? Math.max(16, Math.round(width / 48));
  const naturalWidth = Math.max(1, image.naturalWidth || width);
  const naturalHeight = Math.max(1, image.naturalHeight || height);
  const scaleX = width / naturalWidth;
  const scaleY = height / naturalHeight;
  const alpha = alphaBBox(image);
  // 透明境界の補間ピクセルを切らないよう、実画像bboxを1pxだけ広げる。
  const sourceLeft = Math.max(0, alpha.x - 1) * scaleX;
  const sourceTop = Math.max(0, alpha.y - 1) * scaleY;
  const sourceRight = Math.min(naturalWidth, alpha.x + alpha.w + 1) * scaleX;
  const sourceBottom = Math.min(naturalHeight, alpha.y + alpha.h + 1) * scaleY;
  const meshWidth = Math.max(1, sourceRight - sourceLeft);
  const meshHeight = Math.max(1, sourceBottom - sourceTop);
  // 三角形クリップは高コストなので、連続性を維持できる範囲で分割数に上限を置く。
  const columnCount = Math.max(2, Math.min(28, Math.ceil(meshWidth / targetCellWidth)));
  const rowCount = Math.max(2, Math.min(24, Math.round(requestedRows * (meshHeight / height))));
  const strandCenters = strands.map(strand => strand.x * scaleX);
  const strandRoots = strands.map(strand => Math.max(strand.rootY * scaleY, height * options.rootYRatio));
  const strandTips = strands.map((strand, index) => Math.max(strand.tipY * scaleY, strandRoots[index] + 1));
  // σ = 房間隔の中央値 × 0.6。1房ならブレンド不要で全幅追従。
  let sigma = width * 0.15;
  if (strands.length > 1) {
    const gaps = strandCenters
      .slice(1)
      .map((center, index) => center - strandCenters[index])
      .sort((a, b) => a - b);
    sigma = Math.max(meshWidth / columnCount, gaps[gaps.length >> 1] * 0.6);
  }

  const offsetScale = options.offsetScale ?? 1;
  const tipExponent = options.tipExponent ?? 1.8;
  const mesh: MotionLabMeshPoint[][] = [];
  for (let row = 0; row <= rowCount; row += 1) {
    const sourceY = sourceTop + (meshHeight * row) / rowCount;
    const meshRow: MotionLabMeshPoint[] = [];
    for (let column = 0; column <= columnCount; column += 1) {
      const sourceX = sourceLeft + (meshWidth * column) / columnCount;
      let totalWeight = 0;
      let weightedRootY = 0;
      let weightedTipY = 0;
      for (let strandIndex = 0; strandIndex < strands.length; strandIndex += 1) {
        const gaussianX = (sourceX - strandCenters[strandIndex]) / sigma;
        const weight = Math.exp(-gaussianX * gaussianX);
        totalWeight += weight;
        weightedRootY += weight * strandRoots[strandIndex];
        weightedTipY += weight * strandTips[strandIndex];
      }
      const safeWeight = Math.max(1e-6, totalWeight);
      const rootY = weightedRootY / safeWeight;
      const tipY = weightedTipY / safeWeight;
      const tipRatio = clamp((sourceY - rootY) / Math.max(1, tipY - rootY), 0, 1);
      const softMix = Math.pow(tipRatio, 1.2);
      let blendedDx = 0;
      for (let strandIndex = 0; strandIndex < strands.length; strandIndex += 1) {
        const strand = strands[strandIndex];
        const gaussianX = (sourceX - strandCenters[strandIndex]) / sigma;
        const weight = Math.exp(-gaussianX * gaussianX);
        const strandDx = strand.stiffDx + (strand.softDx - strand.stiffDx) * softMix;
        blendedDx += weight * strandDx;
      }
      const rawDx = (blendedDx / safeWeight) * Math.pow(tipRatio, tipExponent);
      const dx = rawDx * offsetScale;
      const dy = Math.abs(rawDx) * offsetScale * 0.12;
      meshRow.push({ x: -pivotX + sourceX + dx, y: -pivotY + sourceY + dy });
    }
    mesh.push(meshRow);
  }

  ctx.save();
  ctx.translate(pivotX + transform.x, pivotY + transform.y);
  ctx.rotate((transform.rotationDeg * Math.PI) / 180);
  if (transform.skewX) ctx.transform(1, 0, transform.skewX, 1, 0, 0);
  ctx.scale(transform.scaleX, transform.scaleY);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  for (let row = 0; row < rowCount; row += 1) {
    const sourceY0 = sourceTop + (meshHeight * row) / rowCount;
    const sourceY1 = sourceTop + (meshHeight * (row + 1)) / rowCount;
    for (let column = 0; column < columnCount; column += 1) {
      const sourceX0 = sourceLeft + (meshWidth * column) / columnCount;
      const sourceX1 = sourceLeft + (meshWidth * (column + 1)) / columnCount;
      const sourceTopLeft = { x: sourceX0, y: sourceY0 };
      const sourceTopRight = { x: sourceX1, y: sourceY0 };
      const sourceBottomLeft = { x: sourceX0, y: sourceY1 };
      const sourceBottomRight = { x: sourceX1, y: sourceY1 };
      const topLeft = mesh[row][column];
      const topRight = mesh[row][column + 1];
      const bottomLeft = mesh[row + 1][column];
      const bottomRight = mesh[row + 1][column + 1];
      drawMotionLabTexturedTriangle(
        ctx,
        image,
        width,
        height,
        [sourceTopLeft, sourceTopRight, sourceBottomRight],
        [topLeft, topRight, bottomRight],
      );
      drawMotionLabTexturedTriangle(
        ctx,
        image,
        width,
        height,
        [sourceTopLeft, sourceBottomRight, sourceBottomLeft],
        [topLeft, bottomRight, bottomLeft],
      );
    }
  }
  ctx.restore();
}

/**
 * PuruPuruPNGTuber `pyokopyokoHairShift` の周期・位相設計を変更・適応。
 * Upstream: rotejin/PuruPuruPNGTuber@9dc1e735 (Apache-2.0, Copyright 2026 masa)
 * Modified for PachiPakuGen: Canvas横ストリップ描画、振幅、発話連動、
 * 前後髪の扱いを変更。詳細は THIRD_PARTY_NOTICES.md を参照。
 * 位相が毛先(u=1)へ向かって進む複数sinの合成 = 髪を波が伝わって見える。
 */
export function drawMotionLabWaveWarp(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  options: {
    rootYRatio: number;
    timeMs: number;
    strength: number;
    seed: number;
    voice: number;
    stripCount?: number;
    /** 空間周波数倍率: 1=前髪の細かい波、小さいほど波長が長く「全体がたゆたう」大波（後ろ髪向け） */
    spatialFreq?: number;
    /** 時間倍率: 1=既定、小さいほどゆっくり */
    tempo?: number;
    /** ループ書き出し用: 各波成分の周波数をこの周期（秒）の整数分の一へ量子化する */
    loopPeriodSeconds?: number;
  },
) {
  const TAU = Math.PI * 2;
  const pivotX = width * 0.5;
  const pivotY = height * 0.58;
  const stripCount = options.stripCount ?? 24;
  const sf = options.spatialFreq ?? 1;
  const seconds = options.timeMs / 1000;
  // PuruPuruと同じBPM160基準。ループモードでは成分ごとの実周波数（cycles/s）を量子化する
  const beatPerSecond = (160 / 60) * (options.tempo ?? 1);
  const loopT = options.loopPeriodSeconds ?? 0;
  const componentPhase = (rate: number) => {
    const cyclesPerSecond = beatPerSecond * rate;
    return seconds * (loopT > 0 ? snapCyclesToPeriod(cyclesPerSecond, loopT) : cyclesPerSecond);
  };
  const px = (width / 1024) * options.strength;
  const voiceBoost = 1 + options.voice * 0.42;
  ctx.save();
  ctx.translate(pivotX + transform.x, pivotY + transform.y);
  ctx.rotate((transform.rotationDeg * Math.PI) / 180);
  if (transform.skewX) ctx.transform(1, 0, transform.skewX, 1, 0, 0);
  ctx.scale(transform.scaleX, transform.scaleY);
  for (let index = 0; index < stripCount; index += 1) {
    const sourceY = Math.floor((height * index) / stripCount);
    const nextY = Math.floor((height * (index + 1)) / stripCount);
    const stripHeight = Math.max(1, nextY - sourceY);
    const centerYRatio = (sourceY + stripHeight * 0.5) / height;
    const u = clamp((centerYRatio - options.rootYRatio) / Math.max(0.001, 1 - options.rootYRatio), 0, 1);
    // 根元固定マスク（maskFromY相当）: 根元0→毛先1へ滑らかに立ち上げ
    const mask = clamp(u * 1.25, 0, 1) * u;
    const idleDrift = Math.sin(TAU * (componentPhase(0.42) + u * 0.82 * sf + options.seed));
    const wave = Math.sin(TAU * (componentPhase(0.72) + u * 1.55 * sf + 0.16 + options.seed * 0.7));
    const slow = Math.sin(TAU * (componentPhase(0.5) - 0.255 + options.seed * 0.3));
    const idleFloat = Math.cos(TAU * (componentPhase(0.36) + u * 0.38 * sf + options.seed));
    const dx = mask * px * voiceBoost * (5.2 * idleDrift + 2.8 * wave + 3.4 * options.voice * slow);
    const dy = mask * px * (1.5 * idleFloat);
    ctx.drawImage(
      image,
      0,
      sourceY,
      width,
      stripHeight,
      -pivotX + dx,
      -pivotY + sourceY + dy,
      width,
      stripHeight + 1,
    );
  }
  ctx.restore();
}

/**
 * 視線ドリフト描画（§8.4）: 白目の不透明領域をステンシルにして虹彩をクリップ合成。
 * スクラッチキャンバス上で eyewhite → (source-atop) irides(gazeオフセット) を合成し、
 * eye相当の変換で本キャンバスへ描く。
 */
export function drawMotionLabGaze(
  ctx: CanvasRenderingContext2D,
  runtime: MotionLabMouthRuntime,
  eyewhite: HTMLImageElement,
  irides: HTMLImageElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  gazeX: number,
  gazeY: number,
  alpha = 1,
  irisScale = 1,
  wetnessOpacity = 0,
) {
  let scratch = runtime.gazeScratch;
  if (!scratch || scratch.width !== width || scratch.height !== height) {
    scratch = document.createElement("canvas");
    scratch.width = width;
    scratch.height = height;
    runtime.gazeScratch = scratch;
  }
  const scratchCtx = scratch.getContext("2d");
  if (!scratchCtx) return;
  scratchCtx.clearRect(0, 0, width, height);
  scratchCtx.globalCompositeOperation = "source-over";
  scratchCtx.drawImage(eyewhite, 0, 0, width, height);
  const regions = motionLabEyeRegions(irides, width, height);
  const sourceScaleX = irides.naturalWidth / Math.max(1, width);
  const sourceScaleY = irides.naturalHeight / Math.max(1, height);
  const safeScale = clamp(
    irisScale,
    1 - MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA,
    1 + MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA,
  );
  const drawIrises = (targetCtx: CanvasRenderingContext2D) => {
    if (regions.length === 0) {
      targetCtx.drawImage(irides, gazeX, gazeY, width, height);
      return;
    }
    for (const region of regions) {
      const padding = 2;
      const x = Math.max(0, region.x - padding);
      const y = Math.max(0, region.y - padding);
      const regionWidth = Math.min(width - x, region.w + padding * 2);
      const regionHeight = Math.min(height - y, region.h + padding * 2);
      const drawWidth = regionWidth * safeScale;
      const drawHeight = regionHeight * safeScale;
      targetCtx.drawImage(
        irides,
        x * sourceScaleX,
        y * sourceScaleY,
        regionWidth * sourceScaleX,
        regionHeight * sourceScaleY,
        x + gazeX - (drawWidth - regionWidth) * 0.5,
        y + gazeY - (drawHeight - regionHeight) * 0.5,
        drawWidth,
        drawHeight,
      );
    }
  };
  scratchCtx.globalCompositeOperation = "source-atop";
  drawIrises(scratchCtx);

  if (wetnessOpacity > 0 && regions.length > 0) {
    let wetnessScratch = runtime.eyeWetnessScratch;
    if (!wetnessScratch || wetnessScratch.width !== width || wetnessScratch.height !== height) {
      wetnessScratch = document.createElement("canvas");
      wetnessScratch.width = width;
      wetnessScratch.height = height;
      runtime.eyeWetnessScratch = wetnessScratch;
    }
    const wetnessCtx = wetnessScratch.getContext("2d");
    if (wetnessCtx) {
      wetnessCtx.clearRect(0, 0, width, height);
      wetnessCtx.globalCompositeOperation = "source-over";
      wetnessCtx.globalAlpha = clamp(wetnessOpacity, 0, MOTION_LAB_WETNESS_MAX_ALPHA);
      for (const region of regions) {
        const geometry = motionLabWetnessGeometry(region);
        const centerX = geometry.centerX + gazeX;
        const centerY = geometry.surfaceCenterY + gazeY;
        const radiusX = geometry.surfaceRadiusX;
        const radiusY = geometry.surfaceRadiusY;
        wetnessCtx.save();
        wetnessCtx.translate(centerX, centerY);
        wetnessCtx.scale(1, radiusY / radiusX);
        const gradient = wetnessCtx.createRadialGradient(0, 0, 0, 0, 0, radiusX);
        gradient.addColorStop(0, "rgba(255,255,255,0.88)");
        gradient.addColorStop(0.58, "rgba(232,249,255,0.48)");
        gradient.addColorStop(1, "rgba(255,255,255,0)");
        wetnessCtx.fillStyle = gradient;
        wetnessCtx.beginPath();
        wetnessCtx.arc(0, 0, radiusX, 0, Math.PI * 2);
        wetnessCtx.fill();
        wetnessCtx.restore();

        // A narrow, harder lower crescent remains legible after the full
        // character is scaled down. The soft surface alone collapses to one
        // translucent pixel in the normal preview size.
        wetnessCtx.save();
        wetnessCtx.strokeStyle = "rgba(238,252,255,0.96)";
        wetnessCtx.lineWidth = geometry.crescentLineWidth;
        wetnessCtx.lineCap = "round";
        wetnessCtx.beginPath();
        wetnessCtx.ellipse(
          centerX,
          geometry.crescentCenterY + gazeY,
          geometry.crescentRadiusX,
          geometry.crescentRadiusY,
          0,
          Math.PI * 0.16,
          Math.PI * 0.84,
        );
        wetnessCtx.stroke();
        wetnessCtx.restore();
      }
      // Mask after drawing so the reflection cannot bleed onto the white of the eye.
      wetnessCtx.globalAlpha = 1;
      wetnessCtx.globalCompositeOperation = "destination-in";
      drawIrises(wetnessCtx);
      wetnessCtx.globalCompositeOperation = "source-over";
      scratchCtx.globalCompositeOperation = "source-over";
      scratchCtx.drawImage(wetnessScratch, 0, 0);
    }
  }
  scratchCtx.globalCompositeOperation = "source-over";
  drawMotionLabLayer(ctx, scratch, width, height, transform, alpha);
}

/**
 * 独立した眉素材を左右ごとの中心でわずかに持ち上げる。
 * 眉が一つの連結領域として検出された場合は、回転させず上下移動だけに退避する。
 */
export function drawMotionLabBrows(
  ctx: CanvasRenderingContext2D,
  runtime: MotionLabMouthRuntime,
  eyebrow: HTMLImageElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  elapsedMs: number,
  voice: number,
  strength: number,
) {
  const motion = motionLabBrowMotion(elapsedMs, voice, strength);
  if (motion.liftPx <= 0.001 && motion.tiltDeg <= 0.001) {
    drawMotionLabLayer(ctx, eyebrow, width, height, transform);
    return;
  }
  const regions = motionLabEyeRegions(eyebrow, width, height);
  if (regions.length !== 2) {
    drawMotionLabLayer(ctx, eyebrow, width, height, {
      ...transform,
      y: transform.y - motion.liftPx,
    });
    return;
  }

  let scratch = runtime.browScratch;
  if (!scratch || scratch.width !== width || scratch.height !== height) {
    scratch = document.createElement("canvas");
    scratch.width = width;
    scratch.height = height;
    runtime.browScratch = scratch;
  }
  const scratchCtx = scratch.getContext("2d");
  if (!scratchCtx) {
    drawMotionLabLayer(ctx, eyebrow, width, height, {
      ...transform,
      y: transform.y - motion.liftPx,
    });
    return;
  }
  scratchCtx.clearRect(0, 0, width, height);
  const sourceScaleX = eyebrow.naturalWidth / Math.max(1, width);
  const sourceScaleY = eyebrow.naturalHeight / Math.max(1, height);
  for (let index = 0; index < regions.length; index += 1) {
    const region = regions[index];
    const padding = Math.max(2, Math.ceil(region.h * 0.18));
    const x = Math.max(0, region.x - padding);
    const y = Math.max(0, region.y - padding);
    const regionWidth = Math.min(width - x, region.w + padding * 2);
    const regionHeight = Math.min(height - y, region.h + padding * 2);
    const centerX = x + regionWidth * 0.5;
    const centerY = y + regionHeight * 0.5;
    const side = index === 0 ? "left" : "right";
    scratchCtx.save();
    scratchCtx.translate(centerX, centerY - motion.liftPx);
    // 左右反転した基本角度で内側を上げ、発話中だけ小さな左右差を足す。
    scratchCtx.rotate((motionLabBrowRotationDeg(side, motion) * Math.PI) / 180);
    scratchCtx.drawImage(
      eyebrow,
      x * sourceScaleX,
      y * sourceScaleY,
      regionWidth * sourceScaleX,
      regionHeight * sourceScaleY,
      -regionWidth * 0.5,
      -regionHeight * 0.5,
      regionWidth,
      regionHeight,
    );
    scratchCtx.restore();
  }
  drawMotionLabLayer(ctx, scratch, width, height, transform);
}

/** 腕・胸・揺れパーツ用: 指定ピボット中心の剛体回転＋縦オフセット描画（段階1/DOM相当） */
export function drawMotionLabPivotLayer(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
  base: MotionLabLayerTransform,
  pivot: { x: number; y: number },
  angleRad: number,
  offsetY: number,
) {
  ctx.save();
  ctx.translate(pivot.x + base.x, pivot.y + base.y + offsetY);
  ctx.rotate(angleRad);
  ctx.drawImage(image, -pivot.x, -pivot.y, width, height);
  ctx.restore();
}

export function createMotionLabPhysics(
  randomizePhase = MOTION_LAB_PRESENCE_DEFAULTS.randomizePhase,
): MotionLabPhysicsState {
  // 表示ごとの位相ランダム化（presence.randomizePhase）: 短い表示が毎回同じ動きに見える問題の回避
  const rand = (range: number) => (randomizePhase ? Math.random() * range : 0);
  return {
    breathPhase: rand(Math.PI * 2),
    swayPhase: rand(Math.PI * 2),
    prevRootX: 0,
    prevRootY: 0,
    rootVX: 0,
    rootVY: 0,
    noiseT: rand(100),
    hairChain: createChain(MOTION_LAB_HAIR_SEGMENTS, randomizePhase),
    hairSpring: { x: 0, v: 0 },
    hairBackSpring: { x: 0, v: 0 },
    arm: createArmSway(MOTION_LAB_ARM_DEFAULTS.segments, randomizePhase),
    chest: { x: 0, v: 0 },
    sways: new Map(),
    earTwitches: new Map(),
    strandSpringsBack: [],
    envOpen: 0,
    mouthVel: { v: 0 },
    speaking: false,
    blinkWait: randomizePhase ? 0.5 + Math.random() * 2 : 1.5,
    blinkPhase: "idle",
    blinkT: 0,
    headTurnT: rand(100),
    nod: { x: 0, v: 0 },
    gaze: { x: 0, y: 0 },
    gazeVelX: { v: 0 },
    gazeVelY: { v: 0 },
    // The gaze always starts from the centre; only its slow horizontal phase moves.
    gazeT: 0,
    highlightT: 0,
    highlight: { x: 0, y: 0 },
    highlightVelX: { v: 0 },
    highlightVelY: { v: 0 },
    strandSprings: [],
    pyoko: { x: 0, v: 0 },
    glanceWait: randomizePhase ? 1 + Math.random() * 2 : 2,
    glanceHead: 0,
    glanceHeadTarget: 0,
    glanceHeadVel: { v: 0 },
    glanceGaze: { x: 0, y: 0 },
  };
}

export function resetMotionLabRuntime(
  runtime: MotionLabMouthRuntime,
  entryBounce = MOTION_LAB_PRESENCE_DEFAULTS.entryBounce,
) {
  runtime.openY = 0;
  runtime.activeTarget = "closed";
  runtime.previousTarget = "closed";
  runtime.transitionStartMs = 0;
  runtime.lastMs = 0;
  runtime.browVoice = 0;
  runtime.physics = createMotionLabPhysics();
  // 登場撃力（presence.entryBounce）: 表示開始時に髪・肩・胸へ撃力を入れ「呼ばれた感」を出す
  if (entryBounce > 0) {
    runtime.physics.hairChain.omegas[0] += 0.9 * entryBounce;
    runtime.physics.arm.lift.v += MOTION_LAB_ARM_DEFAULTS.lift.bounce * 0.8 * entryBounce;
    runtime.physics.chest.v += 4 * entryBounce;
  }
}

export function prepareMotionLabCanvas(canvas: HTMLCanvasElement | null, width: number, height: number): CanvasRenderingContext2D | null {
  const ctx = canvas?.getContext("2d") ?? null;
  if (!canvas || !ctx) return null;
  canvas.width = width;
  canvas.height = height;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return ctx;
}

export function drawMotionLabScene(
  ctx: CanvasRenderingContext2D,
  parts: MotionLabPartsResult,
  images: MotionLabImageSet,
  runtime: MotionLabMouthRuntime,
  elapsedMs: number,
  settings: MotionLabRenderSettings,
) {
  const random = settings.random ?? Math.random;
  // ===== ループ書き出しモードの導出値 =====
  // 全駆動を loopSeconds の整数分の一周期へ量子化する。減衰バネはウォームアップで
  // 周期軌道へ収束するため、通常再生では決して戻らない「初期状態との一致」を
  // 待つ必要がなく、収束後の任意の1周期がそのままシームレスなループになる。
  const loopSeconds = settings.loopPeriodMs && settings.loopPeriodMs > 0
    ? settings.loopPeriodMs / 1000
    : 0;
  const loopMsNow = loopSeconds > 0
    ? ((elapsedMs % settings.loopPeriodMs!) + settings.loopPeriodMs!) % settings.loopPeriodMs!
    : 0;
  const liveInput = settings.liveInput?.() ?? null;
  const timelineTarget = motionLabTimelineAt(elapsedMs, settings.timeline, settings.timelineDurationMs);
  const target = liveInput
    ? {
      mouth: liveInput.mouth,
      energy: clamp(liveInput.energy, 0, 1),
      loopMs: timelineTarget.loopMs,
    }
    : timelineTarget;
  const dt = runtime.lastMs > 0 ? clampDt((elapsedMs - runtime.lastMs) / 1000) : 0;
  runtime.lastMs = elapsedMs;
  const ph = runtime.physics;

  if (runtime.activeTarget !== target.mouth) {
    runtime.previousTarget = runtime.activeTarget;
    runtime.activeTarget = target.mouth;
    runtime.transitionStartMs = elapsedMs;
  }

  const speaking = target.mouth !== "closed";
  const speechStarted = speaking && !ph.speaking;
  ph.speaking = speaking;
  // ループモード: まばたきを乱数タイマーではなくループ位相の固定マークで発火させる。
  // マークはシーム（ループ境界）を跨がない位置に置き、毎周同一のため境界では常にidle。
  // blinkWaitをidle中に毎フレーム上書きするだけで、blinkState本体は変更不要
  if (loopSeconds > 0 && ph.blinkPhase === "idle") {
    const blinkCount = Math.max(1, Math.round((loopSeconds * settings.blinkRate) / 6));
    const periodPerBlinkMs = settings.loopPeriodMs! / blinkCount;
    // 各区間の35%位置で開始（シーケンス全長≈550msが区間内に収まる）
    const markOffsetMs = periodPerBlinkMs * 0.35;
    const posInSegment = loopMsNow % periodPerBlinkMs;
    const untilMark = posInSegment <= markOffsetMs
      ? markOffsetMs - posInSegment
      : periodPerBlinkMs - posInSegment + markOffsetMs;
    ph.blinkWait = untilMark / 1000;
  }
  const blinkFrame = stepMotionLabBlink(
    ph,
    dt,
    settings.blinkEnabled && images.eyeFrames.length > 1,
    settings.blinkRate,
    MOTION_LAB_BLINK_DEFAULTS,
    random,
  );

  // ===== 口の開度: A4エンベロープ（attack/release）→ A1 SmoothDamp追従 =====
  // baselineレーンは attackMs=0/releaseMs=0/shapeSmoothing=0 で矩形駆動（従来相当）になる
  const targetOpenBase = MOTION_LAB_TARGET_OPEN[target.mouth];
  const targetOpen = liveInput
    ? clamp(liveInput.openness, 0, 1) * targetOpenBase
    : targetOpenBase * (1 - settings.restBias * (1 - target.energy));
  ph.envOpen = envelopeStep(ph.envOpen, targetOpen, settings.attackMs, settings.releaseMs, dt);
  const smoothTime = settings.shapeSmoothing * 0.15;
  runtime.openY = smoothTime < 0.005
    ? ph.envOpen
    : clamp(smoothDamp(runtime.openY, ph.envOpen, ph.mouthVel, smoothTime, dt), 0, 1);

  const width = parts.width;
  const height = parts.height;
  const preset = MOTION_LAB_PRESET_FACTORS[settings.preset];
  const voice = target.energy;
  // 口の段階切替を眉へ直結させず、少し遅れて上がり、ゆっくり戻す。
  runtime.browVoice ??= 0;
  runtime.browVoice = envelopeStep(runtime.browVoice, voice, 120, 220, dt);

  let bodyTransform: MotionLabLayerTransform;
  let hairFrontTransform: MotionLabLayerTransform;
  let hairBackTransform: MotionLabLayerTransform;
  let hairMeshAngles: ArrayLike<number> | null = null;
  type HairStrandRender = Array<{
    x: number;
    rootY: number;
    tipY: number;
    stiffDx: number;
    softDx: number;
  }>;
  /** 房ごと髪物理の描画リスト（検出位置と硬・柔2本のスカラーばね変位）。null=一枚チェーン */
  let hairStrandRender: HairStrandRender | null = null;
  let hairBackStrandRender: HairStrandRender | null = null;

  if (settings.layerMode === "simple") {
    // 基準レーン: 従来のB0相当（一体揺れ・絶対時刻sin）を維持
    const time = elapsedMs / 1000;
    const breath = Math.sin((time / 3.6) * Math.PI * 2);
    const sway = Math.sin(time * 1.35);
    bodyTransform = {
      x: 0,
      y: breath * 1.4 * settings.breathAmplitude,
      rotationDeg: sway * 0.35 * settings.bodySwayAmplitude,
      scaleX: 1,
      scaleY: 1,
    };
    hairFrontTransform = bodyTransform;
    hairBackTransform = bodyTransform;
  } else {
    // B0移植: 位相積分の呼吸・アイドル揺れ＋rootX/Y速度計測（B1/B3/腕/胸の駆動源）
    // ループモードでは両レートを書き出し周期の整数分の一へ量子化する
    // （推奨ループ長は3.6秒の倍数なので呼吸は無変化、体揺れのみ数%変わる）
    const breathRate = loopSeconds > 0
      ? snapAngularRateToPeriod((Math.PI * 2) / 3.6, loopSeconds)
      : (Math.PI * 2) / 3.6;
    const swayRate = loopSeconds > 0 ? snapAngularRateToPeriod(1.35, loopSeconds) : 1.35;
    ph.breathPhase += dt * breathRate;
    ph.swayPhase += dt * swayRate;
    if (ph.breathPhase > Math.PI * 200) ph.breathPhase -= Math.PI * 200;
    if (ph.swayPhase > Math.PI * 200) ph.swayPhase -= Math.PI * 200;
    const breath = Math.sin(ph.breathPhase);
    const sway = Math.sin(ph.swayPhase);
    const rootX = sway * 1.2 * settings.bodySwayAmplitude * preset.body;
    // 発話ぴょこバウンス（PuruPuru pyoko参考）: 定数下げではなくバネで「ぴょこん」と弾む
    springStep(ph.pyoko, -voice * settings.pyokoBounce, 90, 10, dt);
    const breathY = breath * 3.2 * settings.breathAmplitude * preset.breath;
    const rootY = breathY + ph.pyoko.x;
    // 頭・髪は胸の呼吸に少し遅れて追従する（-0.6位相の遅延呼吸）。
    const hairLagY = Math.sin(ph.breathPhase - 0.6) * 3.2 * settings.breathAmplitude * preset.breath + ph.pyoko.x;
    if (dt > 0) {
      ph.rootVX = (rootX - ph.prevRootX) / dt;
      ph.rootVY = (rootY - ph.prevRootY) / dt;
    }
    ph.prevRootX = rootX;
    ph.prevRootY = rootY;
    bodyTransform = {
      x: rootX,
      y: rootY,
      rotationDeg: sway * 1.05 * settings.bodySwayAmplitude * preset.body,
      scaleX: 1 + breath * 0.002 * settings.breathAmplitude,
      scaleY: 1 + breath * 0.006 * settings.breathAmplitude,
    };

    ph.noiseT += dt;
    if (settings.layerMode === "mesh") {
      // B3: 角度チェーン（毛先ほど低剛性・風=sin＋1Dノイズ・頭のX速度カップリング）
      const windAmp = settings.hairWind * preset.hair * settings.hairMotionStrength;
      const windSinRate = loopSeconds > 0 ? snapAngularRateToPeriod(1.7, loopSeconds) : 1.7;
      const windNoise = loopSeconds > 0
        ? noise1dLoop(ph.noiseT * 0.6, 0.6 * loopSeconds)
        : noise1d(ph.noiseT * 0.6);
      const wind = Math.sin(ph.noiseT * windSinRate) * windAmp + windNoise * windAmp * 0.6;
      const drive = clamp(-settings.hairDrive * settings.hairMotionStrength * ph.rootVX * 0.05, -0.2, 0.2);
      stepChain(ph.hairChain, drive + wind, settings.hairK, settings.hairC, dt, 0.5);
      hairMeshAngles = ph.hairChain.angles;
      // 毛先輪郭の実ピークごとに、硬・柔2本のスカラーばねを同じ目標へ追従させる。
      // 描画では検出したrootY/tipYを使い、根元から毛先へ柔らかい側の変位を混ぜる。
      const stepStrandSprings = (
        image: HTMLImageElement,
        springs: MotionLabPhysicsState["strandSprings"],
        driveScale: number,
        phaseSeed: number,
        // 後ろ髪の「大波」化: 剛性を下げてゆっくり大きく、風の時間も遅く
        kScale = 1,
        windTempo = 1,
      ): HairStrandRender | null => {
        const strands = detectHairStrands(image, MOTION_LAB_HAIR_SEGMENTS);
        if (strands.length <= 1) return null;
        if (springs.length !== strands.length) {
          springs.length = 0;
          for (let i = 0; i < strands.length; i += 1) {
            springs.push(createHairStrandSpring(i * 1.37 + phaseSeed));
          }
        }
        return strands.map((strand, index) => {
          const spring = springs[index];
          const strandRateA = loopSeconds > 0
            ? snapAngularRateToPeriod(windTempo * 1.7, loopSeconds)
            : windTempo * 1.7;
          const strandRateB = loopSeconds > 0
            ? snapAngularRateToPeriod(windTempo * 1.9, loopSeconds)
            : windTempo * 1.9;
          const strandWind =
            (Math.sin(ph.noiseT * strandRateA + spring.phase) * windAmp * 1.8 +
              Math.sin(ph.noiseT * strandRateB + spring.phase * 2.3) * windAmp) * 1.15;
          const span = Math.max(1, strand.tipY - strand.rootY);
          const target = clamp((drive + strandWind) * driveScale, -0.08, 0.08) * span;
          const displacement = stepHairStrandSpring(
            spring,
            target,
            dt,
            settings.hairK * kScale,
            settings.hairC,
            span * 0.09,
          );
          return { ...strand, ...displacement };
        });
      };
      if (settings.strandsEnabled && images.hair) {
        ph.strandSprings ??= [];
        hairStrandRender = stepStrandSprings(images.hair, ph.strandSprings, 1, 0);
      }
      // 後ろ髪も房分割対象（振幅は hairBackScale に従う）。
      // 大波特性: 剛性半分（固有振動数が低く、ゆっくり大きくたゆたう）＋風の時間0.55倍
      if (settings.strandsEnabled && images.hairBack) {
        ph.strandSpringsBack ??= [];
        hairBackStrandRender = stepStrandSprings(
          images.hairBack,
          ph.strandSpringsBack,
          settings.hairBackScale * 1.3,
          5.3,
          0.5,
          0.55,
        );
      }
      hairFrontTransform = { x: bodyTransform.x, y: hairLagY * 0.62, rotationDeg: 0, scaleX: 1, scaleY: 1 };
      // 後ろ髪はB1相当の回転で追従（参照実装: チェーン中間角×0.8）
      const backRot =
        ph.hairChain.angles[Math.floor(MOTION_LAB_HAIR_SEGMENTS / 2)] * 0.8 * settings.hairBackScale;
      hairBackTransform = {
        // 横シフト係数 30→18: 回転はそのまま、平行移動の「横滑り」感を抑える
        x: bodyTransform.x + backRot * 18,
        y: hairLagY * 0.42,
        rotationDeg: (backRot * 180) / Math.PI,
        scaleX: 1,
        scaleY: 1,
      };
    } else {
      // B1: スプリング遅延追従（頭が右へ動くと髪は左へ流れる速度カップリング）
      const maxAngle = 0.18;
      const drive = clamp(-settings.hairDrive * settings.hairMotionStrength * ph.rootVX, -maxAngle, maxAngle);
      const theta = clamp(springStep(ph.hairSpring, drive, settings.hairK, settings.hairC, dt), -maxAngle, maxAngle);
      // 後ろ髪: 柔らかく（低剛性=より遅延）・少し大きく
      const thetaB = clamp(
        springStep(ph.hairBackSpring, drive * 1.25, settings.hairK * 0.45, settings.hairC * 0.8, dt),
        -maxAngle * 1.4,
        maxAngle * 1.4,
      ) * settings.hairBackScale;
      hairFrontTransform = {
        x: bodyTransform.x + theta * 26 * preset.hair,
        y: hairLagY * 0.62,
        rotationDeg: (theta * 180) / Math.PI,
        scaleX: 1,
        scaleY: 1,
      };
      hairBackTransform = {
        // 横シフト係数 34→24: 後ろ髪の横滑りを抑える（回転追従は維持）
        x: bodyTransform.x + thetaB * 24 * preset.hair,
        y: hairLagY * 0.42,
        rotationDeg: (thetaB * 180) / Math.PI,
        scaleX: 1,
        scaleY: 1,
      };
    }
    // 「髪の揺れ」エフェクトOFF: バネ/波/房の揺れを全て無効化し、頭への追従
    // （遅延呼吸・パララックス・発話バウンス）だけ残す
    if (!settings.hairMotionEnabled) {
      hairFrontTransform = { x: bodyTransform.x, y: hairLagY * 0.62, rotationDeg: 0, scaleX: 1, scaleY: 1 };
      hairBackTransform = { x: bodyTransform.x, y: hairLagY * 0.42, rotationDeg: 0, scaleX: 1, scaleY: 1 };
      hairMeshAngles = null;
      hairStrandRender = null;
      hairBackStrandRender = null;
    }
    // 体の回転は最終段で全レイヤーへ継承する。
    // 髪の物理回転に体の傾きを加算 = 体の揺れに髪が必ずついてくる
    hairFrontTransform = {
      ...hairFrontTransform,
      rotationDeg: hairFrontTransform.rotationDeg + bodyTransform.rotationDeg,
    };
    hairBackTransform = {
      ...hairBackTransform,
      rotationDeg: hairBackTransform.rotationDeg + bodyTransform.rotationDeg,
    };
  }

  // ===== パララックス首振り（Anime2.5DRigの設計を参考に独自実装） =====
  // 駆動 = ノイズドリフト（headTurn）＋発話開始の頷きバネ（headNod）。
  // 各レイヤーへ depth × SHIFT_MAX の水平シフト＋シアーを適用（縦はシフトのみ）
  let eyeMouthTransform = bodyTransform;
  let parallaxArmDx = 0;
  let parallaxArmDy = 0;
  // Anime2.5DRig@d4882586 (MIT, Copyright 2026 hakoniwa) の更新間隔を参考に変更・適応。
  // ランダムグランス: 1.4〜4秒ごとに顔向き・視線の目標が
  // ふっと変わり、SmoothDampで滑らかに移行する（連続ノイズだけより「意図」が出る）
  if (settings.randomGlance && settings.glanceStrength > 0 && settings.layerMode !== "simple") {
    if (loopSeconds > 0) {
      // ループモード: 目標をループ位相のインデックスから決定的ハッシュで選ぶ。
      // 目標系列が毎周同一なら、SmoothDamp状態はウォームアップで周期軌道へ収束する
      const glanceCount = Math.max(1, Math.round(loopSeconds / 2.7));
      const index = Math.min(
        glanceCount - 1,
        Math.floor((loopMsNow / settings.loopPeriodMs!) * glanceCount),
      );
      const hash = Math.sin((index + 1) * 12.9898) * 43758.5453;
      ph.glanceHeadTarget = ((hash - Math.floor(hash)) * 2 - 1) * 0.45 * settings.glanceStrength;
    } else if (!blinkFrame.sequenceActive) {
      // 瞬き準備中は新しい視線目標を作らない。顔向きは維持し、目だけ中央へ戻す。
      ph.glanceWait -= dt;
      if (ph.glanceWait <= 0) {
        ph.glanceWait = 1.4 + random() * 2.6;
        ph.glanceHeadTarget = (random() * 2 - 1) * 0.45 * settings.glanceStrength;
      }
    }
    ph.glanceHead = smoothDamp(ph.glanceHead, ph.glanceHeadTarget, ph.glanceHeadVel, 0.5, dt);
  } else {
    ph.glanceHeadTarget = 0;
    ph.glanceGaze.x = 0;
    ph.glanceGaze.y = 0;
    ph.glanceHead = smoothDamp(ph.glanceHead, 0, ph.glanceHeadVel, 0.5, dt);
  }
  if (settings.layerMode !== "simple" && settings.parallaxScale > 0) {
    ph.headTurnT += dt * MOTION_LAB_PARALLAX_DEFAULTS.driftSpeed;
    const headDrift = loopSeconds > 0
      ? noise1dLoop(ph.headTurnT, MOTION_LAB_PARALLAX_DEFAULTS.driftSpeed * loopSeconds)
      : noise1d(ph.headTurnT);
    const headTurn = clamp(headDrift * 0.56 + ph.glanceHead, -1, 1) * settings.parallaxScale;
    if (speechStarted) ph.nod.v += MOTION_LAB_NOD_DEFAULTS.impulse;
    springStep(ph.nod, 0, MOTION_LAB_NOD_DEFAULTS.k, MOTION_LAB_NOD_DEFAULTS.c, dt);
    const nodPx =
      clamp(ph.nod.x, -MOTION_LAB_NOD_DEFAULTS.maxPx, MOTION_LAB_NOD_DEFAULTS.maxPx) *
      settings.parallaxScale;
    const shiftMax = width * MOTION_LAB_PARALLAX_DEFAULTS.shiftRatio;
    const applyParallax = (
      transform: MotionLabLayerTransform,
      depth: number,
    ): MotionLabLayerTransform => ({
      ...transform,
      x: transform.x + headTurn * depth * shiftMax,
      y: transform.y + nodPx * depth,
      // 顔の角度が一定に見えないよう、深度に応じた微回転も加える（傾き演出）
      rotationDeg: transform.rotationDeg + headTurn * depth * 4.5,
      skewX: (transform.skewX ?? 0) + headTurn * depth * MOTION_LAB_PARALLAX_DEFAULTS.shearMax,
    });
    hairBackTransform = applyParallax(hairBackTransform, MOTION_LAB_DEPTH_DEFAULTS.hair_back);
    hairFrontTransform = applyParallax(hairFrontTransform, MOTION_LAB_DEPTH_DEFAULTS.hair);
    eyeMouthTransform = applyParallax(bodyTransform, MOTION_LAB_DEPTH_DEFAULTS.eye);
    // body/chest は depth 0 = 基準面。腕は depth 0.1（ピボット描画側で平行移動のみ加算）
    parallaxArmDx = headTurn * MOTION_LAB_DEPTH_DEFAULTS.arm_l * shiftMax;
    parallaxArmDy = nodPx * MOTION_LAB_DEPTH_DEFAULTS.arm_l;
  }

  // ===== 腕揺れ＋肩の弾み（32-arm-sway.js移植） =====
  const animateParts = settings.layerMode !== "simple";
  let armOut: ReturnType<typeof updateArmSway> | null = null;
  if ((images.armL || images.armR) && settings.armEnabled && animateParts) {
    armOut = updateArmSway(
      ph.arm,
      {
        k: MOTION_LAB_ARM_DEFAULTS.k,
        c: MOTION_LAB_ARM_DEFAULTS.c,
        coupling: MOTION_LAB_ARM_DEFAULTS.coupling * settings.armSwayAmp,
        noise: MOTION_LAB_ARM_DEFAULTS.noise * settings.armSwayAmp,
        // 常時のゆっくりした振り子スイング（揺れ幅%で拡大。従来の体速度＋微ノイズだけでは体感が弱い）
        idleSwing: settings.armMaxAngle * 0.45 * settings.armSwayAmp,
        maxAngle: settings.armMaxAngle,
        liftEnabled: settings.liftEnabled,
        // 二次追従化: 一次バウンスは体1本、肩はそれに遅れて追従する。
        // 発話バウンス有効時は独自撃力を1/4に抑え、体のY速度カップリングを強めて
        // 「体が弾む→肩が遅れてついてくる」の連動にする
        liftCoupling: MOTION_LAB_ARM_DEFAULTS.lift.coupling * (settings.pyokoBounce > 0 ? 2.8 : 1) * settings.liftStrength,
        liftBounce: MOTION_LAB_ARM_DEFAULTS.lift.bounce * (settings.pyokoBounce > 0 ? 0.25 : 1) * settings.liftStrength,
        liftMax: MOTION_LAB_ARM_DEFAULTS.lift.max,
        loopPeriodSeconds: loopSeconds > 0 ? loopSeconds : undefined,
      },
      dt,
      bodyTransform.x,
      bodyTransform.y,
      speechStarted,
    );
  }

  // ===== 胸部追従: 体・衣服の局所ワープを駆動する強減衰バネ =====
  // 主駆動は呼吸に少し遅れた上下（視認できる胸部のやわらかさ）。
  // 体のY速度追従と発話時の撃力は二次成分。速度追従だけではサブピクセルになり見えない。
  let chestOffsetY = 0;
  if (animateParts && settings.chestMax > 0) {
    const pyokoActive = settings.pyokoBounce > 0;
    if (speechStarted) ph.chest.v += pyokoActive ? 6 : 10;
    const breathLag = Math.sin(ph.breathPhase - 0.5) * settings.chestMax * 0.65;
    const velocityFollow = clamp(-0.35 * ph.rootVY, -settings.chestMax, settings.chestMax);
    const chestNoise = pyokoActive
      ? 0
      : (loopSeconds > 0
        ? noise1dLoop(ph.noiseT * 0.8 + 13.7, 0.8 * loopSeconds)
        : noise1d(ph.noiseT * 0.8 + 13.7)) * settings.chestMax * 0.12;
    const driveY = clamp(
      breathLag + velocityFollow * 0.4 + chestNoise,
      -settings.chestMax,
      settings.chestMax,
    );
    springStep(ph.chest, driveY, MOTION_LAB_CHEST_DEFAULTS.k, MOTION_LAB_CHEST_DEFAULTS.c, dt);
    ph.chest.x = clamp(ph.chest.x, -settings.chestMax, settings.chestMax);
    chestOffsetY = ph.chest.x;
  }

  // 腕: 肩ピボット（不透明bbox上端中央を自動推定）の剛体回転＋lift（段階1/Canvas相当）
  const armBaseTransform: MotionLabLayerTransform = {
    ...bodyTransform,
    x: bodyTransform.x + parallaxArmDx,
    y: bodyTransform.y + parallaxArmDy,
  };
  const drawArm = (
    image: HTMLImageElement | null,
    out: { rigid: number; lift: number } | null,
    part: "arm_l" | "arm_r",
  ) => {
    if (!image) return;
    // 指などの切り出しオーバーレイも、親腕と完全に同じ肩ピボットを使う。
    // オーバーレイ自身の小さなbboxを基準にすると、回転時に腕からずれてしまう。
    const parentImage = part === "arm_l" ? images.armL : images.armR;
    const bbox = alphaBBox(parentImage ?? image);
    // 回転軸: エディタ指定 > bbox上端中央（=肩推定）＋armPivotRatio下方調整
    const pivot = settings.pivots[part] ?? {
      x: bbox.x + bbox.w / 2,
      y: bbox.y + bbox.h * settings.armPivotRatio,
    };
    let angle = (out?.rigid ?? 0) * (settings.swingScale[part] ?? 1);
    const range = settings.rangesDeg[part] ?? 0;
    if (range > 0) angle = clamp(angle, (-range * Math.PI) / 180, (range * Math.PI) / 180);
    // 体の傾きを腕へ継承する。
    angle += (bodyTransform.rotationDeg * Math.PI) / 180;
    drawMotionLabPivotLayer(ctx, image, width, height, armBaseTransform, pivot, angle, out?.lift ?? 0);
  };
  const drawHairBack = () => {
    if (!images.hairBack) return;
    if (settings.layerMode === "mesh" && hairBackStrandRender && !settings.hairWaveMode) {
      // 後ろ髪の房分割: 回転追従の代わりに房ごとソフトブレンドワープ
      const backRootYRatio = settings.pivots.hair_back
        ? clamp(settings.pivots.hair_back.y / height, 0, 0.9)
        : 0.08;
      drawMotionLabStrandBlendWarp(ctx, images.hairBack, width, height, { ...hairBackTransform, rotationDeg: bodyTransform.rotationDeg }, {
        rootYRatio: settings.pivots.hair_back ? backRootYRatio : 0,
        strands: hairBackStrandRender,
        offsetScale: settings.swingScale.hair_back ?? 1,
        tipExponent: 2.1,
      });
    } else if (settings.hairWaveMode && animateParts && settings.hairMotionEnabled) {
      // 波揺れ: 後ろ髪は前髪と位相をずらし、hairBackScale で振幅調整
      const backRootYRatio = settings.pivots.hair_back
        ? clamp(settings.pivots.hair_back.y / height, 0, 0.9)
        : 0.08;
      // 後ろ髪は「大波」: 波長を長く（×0.42）・ゆっくり（×0.6）・振幅は少し大きく。
      // 細かいプルプルではなく髪全体がゆるやかにたゆたう（実際の長い髪の低周波モード）
      drawMotionLabWaveWarp(ctx, images.hairBack, width, height, { ...hairBackTransform, rotationDeg: bodyTransform.rotationDeg }, {
        rootYRatio: backRootYRatio,
        timeMs: elapsedMs,
        strength: settings.hairWaveStrength * preset.hair * settings.hairBackScale * settings.hairMotionStrength * 1.6,
        seed: 0.82,
        voice,
        spatialFreq: 0.42,
        tempo: 0.6,
        loopPeriodSeconds: loopSeconds > 0 ? loopSeconds : undefined,
      });
    } else {
      drawWithOptionalPivot(images.hairBack, clampRotationDeg(hairBackTransform, "hair_back"), "hair_back");
    }
  };
  const drawBody = () => {
    drawMotionLabChestWarp(
      ctx,
      images.body,
      images.chest,
      width,
      height,
      bodyTransform,
      chestOffsetY,
      runtime,
    );
  };
  // 旧layer-order.jsonに残るchestキーを受ける。描画はbodyの局所ワープへ統合済み。
  const drawChest = () => {};

  // 汎用揺れパーツ sway_*: 腕と同系のチェーン物理（ピボット=bbox上端中央）。
  // 獣耳（sway_ear*）は頭に付いているパーツなので、前髪と同じ頭基準の変換
  // （呼吸遅延・パララックス首振り・頷き込み）に連動させ、その上でツイッチを乗せる
  const drawOneSway = (name: string, image: HTMLImageElement) => {
      const isEar = /(^|_)ears?(_|$)/i.test(name);
      const base = isEar ? hairFrontTransform : bodyTransform;
      // 土台の傾き（耳=頭、その他=体）を常に継承する
      let angle = (base.rotationDeg * Math.PI) / 180;
      let twitchOffsetY = 0;
      if (animateParts) {
        let chainState = ph.sways.get(name);
        if (!chainState) {
          chainState = createChain(MOTION_LAB_SWAY_DEFAULTS.segments);
          ph.sways.set(name, chainState);
        }
        chainState.t += dt;
        const noise = (loopSeconds > 0
          ? noise1dLoop(chainState.t * 0.5, 0.5 * loopSeconds)
          : noise1d(chainState.t * 0.5)) * MOTION_LAB_SWAY_DEFAULTS.noise;
        const drive = clamp(
          -MOTION_LAB_ARM_DEFAULTS.coupling * ph.rootVX * 0.05,
          -MOTION_LAB_SWAY_DEFAULTS.maxAngle,
          MOTION_LAB_SWAY_DEFAULTS.maxAngle,
        );
        // 獣耳ピコピコ（sway_ear*限定・オプション）。上下・傾き・二連の3方式を
        // 同じ決定的乱数列で駆動し、同じ設定なら毎回同じ動きを再現する。
        if (settings.earTwitch && isEar) {
          const earTwitchMode = settings.earTwitchMode ?? "double";
          let twitch = ph.earTwitches.get(name);
          if (!twitch) {
            twitch = {
              wait: motionLabInitialEarTwitchWait(random()),
              spring: { x: 0, v: 0 },
              queuedFollowUp: false,
              mode: earTwitchMode,
            };
            ph.earTwitches.set(name, twitch);
          }
          if (twitch.mode !== earTwitchMode) {
            twitch.mode = earTwitchMode;
            twitch.queuedFollowUp = false;
            twitch.wait = motionLabInitialEarTwitchWait(random());
          }
          if (loopSeconds > 0 && twitch.queuedFollowUp !== true) {
            // ループモード: 一次ツイッチをループ位相の固定マークで発火させる。
            // 区間の30%位置なら、二連の追撃＋バネ減衰（計約1.5秒）がシーム前に収まる。
            // 追撃（queuedFollowUp）はマーク相対の短い待ちなので上書きしない
            const twitchCount = Math.max(1, Math.round(loopSeconds / 7));
            const segmentMs = settings.loopPeriodMs! / twitchCount;
            const markMs = segmentMs * 0.3;
            const posMs = loopMsNow % segmentMs;
            const untilMark = posMs <= markMs ? markMs - posMs : segmentMs - posMs + markMs;
            twitch.wait = untilMark / 1000;
          }
          twitch.wait -= dt;
          if (twitch.wait <= 0) {
            const followUp = twitch.queuedFollowUp === true;
            const impulse = motionLabEarTwitchImpulse(
              earTwitchMode,
              settings.earTwitchScale,
              random() < 0.5 ? -1 : 1,
              followUp,
            );
            twitch.spring.v += impulse.bounceVelocity;
            chainState.omegas[0] += impulse.rotationVelocity;
            const queueFollowUp = earTwitchMode === "double" && !followUp;
            twitch.queuedFollowUp = queueFollowUp;
            twitch.wait = motionLabNextEarTwitchWait(earTwitchMode, queueFollowUp, random());
          }
          springStep(twitch.spring, 0, MOTION_LAB_EAR_TWITCH.k, MOTION_LAB_EAR_TWITCH.c, dt);
          twitch.spring.x = clamp(twitch.spring.x, -MOTION_LAB_EAR_TWITCH.maxPx, MOTION_LAB_EAR_TWITCH.maxPx);
          twitchOffsetY = twitch.spring.x;
        }
        stepChain(chainState, drive + noise, MOTION_LAB_SWAY_DEFAULTS.k, MOTION_LAB_SWAY_DEFAULTS.c, dt, MOTION_LAB_SWAY_DEFAULTS.maxAngle);
        // 土台の傾きはそのまま継承し、パーツ固有の揺れだけ倍率・可動域を適用する。
        let dynamicAngle = chainAverage(chainState) * (settings.swingScale[name] ?? 1);
        const rangeDeg = settings.rangesDeg[name] ?? 0;
        if (rangeDeg > 0) {
          const rangeRad = (rangeDeg * Math.PI) / 180;
          dynamicAngle = clamp(dynamicAngle, -rangeRad, rangeRad);
        }
        angle = dynamicAngle + (base.rotationDeg * Math.PI) / 180;
      }
      const bbox = alphaBBox(image);
      const pivot = settings.pivots[name] ?? { x: bbox.x + bbox.w / 2, y: bbox.y };
      drawMotionLabPivotLayer(ctx, image, width, height, base, pivot, angle, twitchOffsetY);
  };

  const swayEntries = Object.entries(images.sways)
    .sort(([left], [right]) => left.localeCompare(right));
  const explicitlyOrderedSways = new Set(
    parts.layerOrder.filter(name => name.startsWith("sway_") && name in images.sways),
  );
  const drawSways = () => {
    // 旧layer-order.jsonのswaysキー用。新形式で個別指定済みの画像は二重描画しない。
    for (const [name, image] of swayEntries) {
      if (!explicitlyOrderedSways.has(name)) drawOneSway(name, image);
    }
  };
  const individualSwayDraws: Record<string, () => void> = {};
  for (const [name, image] of swayEntries) {
    individualSwayDraws[name] = () => drawOneSway(name, image);
  }
  const linkedPartDraws: Record<string, () => void> = {};
  for (const [name, linked] of Object.entries(images.linkedParts ?? {})) {
    if (linked.parent === "arm_l") {
      linkedPartDraws[name] = () => drawArm(linked.image, armOut?.left ?? null, "arm_l");
    } else if (linked.parent === "arm_r") {
      linkedPartDraws[name] = () => drawArm(linked.image, armOut?.right ?? null, "arm_r");
    }
  }

  // ===== 視線ドリフト＋瞳クリップ（§8.4）: eyewhite < irides < highlight < eye連番 =====
  const drawEyeCluster = () => {
    const drawBrows = () => {
      if (!images.eyebrow) return;
      drawMotionLabBrows(
        ctx,
        runtime,
        images.eyebrow,
        width,
        height,
        eyeMouthTransform,
        elapsedMs,
        runtime.browVoice,
        settings.browEnabled ? settings.browStrength : 0,
      );
    };
    // HMRで旧ランタイムが残った場合も、再読込を要求せず新しい状態へ補完する。
    ph.highlight ??= { x: 0, y: 0 };
    ph.highlightVelX ??= { v: 0 };
    ph.highlightVelY ??= { v: 0 };
    const returningToCenter = blinkFrame.phase !== "idle";
    if (returningToCenter) ph.gazeT = 0;
    const centerSmoothTime = Math.max(0.035, MOTION_LAB_BLINK_DEFAULTS.centerMs / 3000);

    let gazeTargetX = 0;
    let gazeTargetY = 0;
    const gazeEnabled = animateParts
      && settings.gazeEnabled
      && settings.gazeStrength > 0
      && !!images.irides;
    if (!returningToCenter && gazeEnabled) {
      const gazeRate = Math.PI * 2 / MOTION_LAB_GAZE_DEFAULTS.periodSeconds;
      ph.gazeT += dt * (loopSeconds > 0 ? snapAngularRateToPeriod(gazeRate, loopSeconds) : gazeRate);
      const horizontalGaze = motionLabHorizontalGazeAt(
        ph.gazeT,
        motionLabEyeRegions(images.irides!, width, height),
        settings.gazeStrength,
      );
      gazeTargetX = clamp(horizontalGaze.x, -MOTION_LAB_GAZE_DEFAULTS.maxRangePx, MOTION_LAB_GAZE_DEFAULTS.maxRangePx);
      gazeTargetY = horizontalGaze.y;
    }
    const gazeSmoothTime = returningToCenter
      ? centerSmoothTime
      : MOTION_LAB_GAZE_DEFAULTS.smoothTime;
    if (!settings.gazeEnabled || settings.gazeStrength <= 0 || !images.irides) {
      ph.gazeT = 0;
      ph.gaze.x = 0;
      ph.gaze.y = 0;
      ph.gazeVelX.v = 0;
      ph.gazeVelY.v = 0;
    } else {
      ph.gaze.x = smoothDamp(ph.gaze.x, gazeTargetX, ph.gazeVelX, gazeSmoothTime, dt);
      ph.gaze.y = smoothDamp(ph.gaze.y, gazeTargetY, ph.gazeVelY, gazeSmoothTime, dt);
    }

    // A detached highlight must remain attached to the iris. Independent noise
    // made the eye colour appear to shimmer while returning to the blink frame.
    ph.highlight.x = ph.gaze.x;
    ph.highlight.y = ph.gaze.y;
    ph.highlightVelX.v = 0;
    ph.highlightVelY.v = 0;

    if (blinkFrame.rifeOwnsEye) {
      // 閉じ・開き本体では独立レイヤーを完全に止め、補間済みの目だけに描画権を渡す。
      ph.gaze.x = 0;
      ph.gaze.y = 0;
      ph.gazeVelX.v = 0;
      ph.gazeVelY.v = 0;
      ph.highlight.x = 0;
      ph.highlight.y = 0;
      ph.highlightVelX.v = 0;
      ph.highlightVelY.v = 0;
      const eyeIndex = Math.round(clamp(blinkFrame.rifeProgress, 0, 1) * (images.eyeFrames.length - 1));
      const eyeFrame = images.eyeFrames[eyeIndex] ?? images.eyeFrames[0];
      if (eyeFrame) drawMotionLabLayer(ctx, eyeFrame, width, height, eyeMouthTransform);
      drawBrows();
      return;
    }

    // 開眼RIFEフレームを常に下地にする。centering終盤では分離目を薄くし、
    // settlingでは同じ下地の上へ分離目を戻すため、虹彩の色差が瞬時に切り替わらない。
    const openEyeFrame = images.eyeFrames[0];
    if (openEyeFrame) drawMotionLabLayer(ctx, openEyeFrame, width, height, eyeMouthTransform);
    const dynamicEyeAlpha = clamp(blinkFrame.dynamicEyeAlpha, 0, 1);
    // ループモード: うるみ・瞳の呼吸の固定周期(4600/5200ms)を書き出し周期の整数分の一へ
    const wetnessPeriodMs = loopSeconds > 0
      ? settings.loopPeriodMs! / Math.max(1, Math.round(settings.loopPeriodMs! / 4600))
      : undefined;
    const irisBreathPeriodMs = loopSeconds > 0
      ? settings.loopPeriodMs! / Math.max(1, Math.round(settings.loopPeriodMs! / 5200))
      : undefined;
    const wetnessOpacity = settings.wetnessEnabled
      ? motionLabWetnessOpacity(elapsedMs, settings.wetnessStrength, wetnessPeriodMs)
      : 0;
    if (images.eyewhite && images.irides) {
      const rawIrisScale = settings.irisBreathEnabled
        ? motionLabIrisBreathScale(elapsedMs, settings.irisBreathStrength, irisBreathPeriodMs)
        : 1;
      const irisScale = 1 + (rawIrisScale - 1) * dynamicEyeAlpha;
      drawMotionLabGaze(
        ctx,
        runtime,
        images.eyewhite,
        images.irides,
        width,
        height,
        eyeMouthTransform,
        ph.gaze.x,
        ph.gaze.y,
        dynamicEyeAlpha,
        irisScale,
        wetnessOpacity,
      );
    }
    if (images.highlight) {
      drawMotionLabLayer(
        ctx,
        images.highlight,
        width,
        height,
        {
          ...eyeMouthTransform,
          x: eyeMouthTransform.x + ph.highlight.x,
          y: eyeMouthTransform.y + ph.highlight.y,
        },
        dynamicEyeAlpha,
      );
    }
    drawBrows();
  };

  const drawMouth = () => {
  const transitionMs = settings.mouthMethod === "baseline"
    ? 0
    : settings.mouthMethod === "bridge"
      ? Math.max(80, settings.crossfadeMs * (1 + settings.bridgeBias))
      : Math.max(0, settings.crossfadeMs);
  const blend = transitionMs > 0
    ? clamp((elapsedMs - runtime.transitionStartMs) / transitionMs, 0, 1)
    : 1;
  const previousFrames = images.mouths[runtime.previousTarget] ?? images.mouths.closed;
  const activeFrames = images.mouths[runtime.activeTarget] ?? images.mouths.closed;
  const neutralFrames = images.mouths.closed ?? previousFrames ?? activeFrames;
  const previousMouth = pickMotionLabMouthFrame(
    previousFrames,
    motionLabMouthFrameRatio(runtime.openY, runtime.previousTarget),
  );
  const activeMouth = pickMotionLabMouthFrame(
    activeFrames,
    motionLabMouthFrameRatio(runtime.openY, runtime.activeTarget),
  );
  const neutralMouth = pickMotionLabMouthFrame(neutralFrames, runtime.openY * (1 - settings.bridgeBias));
  if (settings.mouthMethod === "bridge" && transitionMs > 0 && blend < 1) {
    const easedBlend = blend * blend * (3 - 2 * blend);
    const bridgeAlpha = Math.sin(easedBlend * Math.PI) * settings.bridgeBias;
    if (previousMouth) drawMotionLabLayer(ctx, previousMouth, width, height, eyeMouthTransform, (1 - easedBlend) * (1 - bridgeAlpha));
    if (neutralMouth && bridgeAlpha > 0.01) drawMotionLabLayer(ctx, neutralMouth, width, height, eyeMouthTransform, bridgeAlpha);
    if (activeMouth) drawMotionLabLayer(ctx, activeMouth, width, height, eyeMouthTransform, easedBlend * (1 - bridgeAlpha));
  } else {
    if (previousMouth && blend < 1) drawMotionLabLayer(ctx, previousMouth, width, height, eyeMouthTransform, 1 - blend);
    if (activeMouth) drawMotionLabLayer(ctx, activeMouth, width, height, eyeMouthTransform, blend);
  }
  };

  // 回転軸上書き: hair/hair_back はワープの根元位置（rootYRatio）としても解釈する
  const hairRootYRatio = settings.pivots.hair
    ? clamp(settings.pivots.hair.y / height, 0, 0.9)
    : 0.16;
  const clampRotationDeg = (transform: MotionLabLayerTransform, part: string): MotionLabLayerTransform => {
    let rotationDeg = transform.rotationDeg * (settings.swingScale[part] ?? 1);
    const range = settings.rangesDeg[part] ?? 0;
    if (range > 0) rotationDeg = clamp(rotationDeg, -range, range);
    return rotationDeg === transform.rotationDeg ? transform : { ...transform, rotationDeg };
  };
  // 回転軸が指定されたパーツは、その点を中心とした回転として描く
  const drawWithOptionalPivot = (
    image: HTMLImageElement,
    transform: MotionLabLayerTransform,
    part: string,
  ) => {
    const pivot = settings.pivots[part];
    if (pivot) {
      const angleRad = (transform.rotationDeg * Math.PI) / 180;
      drawMotionLabPivotLayer(ctx, image, width, height, { ...transform, rotationDeg: 0 }, pivot, angleRad, 0);
    } else {
      drawMotionLabLayer(ctx, image, width, height, transform);
    }
  };

  const drawHair = () => {
  if (images.hair) {
    if (settings.hairWaveMode && animateParts && settings.hairMotionEnabled) {
      // ウェーブ式（進行波・根元固定）。体の傾きは維持して波だけ乗せる。
      drawMotionLabWaveWarp(ctx, images.hair, width, height, { ...hairFrontTransform, rotationDeg: bodyTransform.rotationDeg }, {
        rootYRatio: hairRootYRatio,
        timeMs: elapsedMs,
        strength: settings.hairWaveStrength * preset.hair * settings.hairMotionStrength,
        seed: 0.35,
        voice,
        loopPeriodSeconds: loopSeconds > 0 ? loopSeconds : undefined,
      });
    } else if (settings.layerMode === "mesh" && hairStrandRender) {
      // 房中心線へのガウシアン重みで複数のばね変位を滑らかに混合
      drawMotionLabStrandBlendWarp(ctx, images.hair, width, height, hairFrontTransform, {
        rootYRatio: settings.pivots.hair ? hairRootYRatio : 0,
        strands: hairStrandRender,
        offsetScale: settings.swingScale.hair ?? 1,
      });
    } else if (settings.layerMode === "mesh" && hairMeshAngles) {
      drawMotionLabChainWarp(ctx, images.hair, width, height, hairFrontTransform, {
        rootYRatio: hairRootYRatio,
        angles: hairMeshAngles,
        offsetScale: settings.swingScale.hair ?? 1,
      });
    } else {
      drawWithOptionalPivot(images.hair, clampRotationDeg(hairFrontTransform, "hair"), "hair");
    }
  }
  };

  ctx.clearRect(0, 0, width, height);
  drawMotionLabOrderedLayers(parts.layerOrder, settings.armBehindBody, {
    hair_back: drawHairBack,
    body: drawBody,
    chest: drawChest,
    arm_l: () => drawArm(images.armL, armOut?.left ?? null, "arm_l"),
    arm_r: () => drawArm(images.armR, armOut?.right ?? null, "arm_r"),
    sways: drawSways,
    ...individualSwayDraws,
    ...linkedPartDraws,
    eye: drawEyeCluster,
    mouth: drawMouth,
    hair: drawHair,
  });

  // 回転軸エディタ: 編集中パーツの回転軸マーカーを最前面に描く
  if (settings.pivotEditPart) {
    const part = settings.pivotEditPart;
    let pivot = settings.pivots[part] ?? null;
    if (!pivot) {
      const partImage =
        part === "arm_l" ? images.armL
        : part === "arm_r" ? images.armR
        : part === "hair" ? images.hair
        : part === "hair_back" ? images.hairBack
        : images.sways[part] ?? null;
      if (partImage) {
        const bbox = alphaBBox(partImage);
        pivot = part.startsWith("arm_")
          ? { x: bbox.x + bbox.w / 2, y: bbox.y + bbox.h * settings.armPivotRatio }
          : part in images.sways
            ? { x: bbox.x + bbox.w / 2, y: bbox.y }
          : { x: width / 2, y: height * (part === "hair_back" ? 0.08 : 0.16) };
      }
    }
    if (pivot) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 80, 140, 0.95)";
      ctx.lineWidth = Math.max(2, width / 400);
      const r = Math.max(10, width / 60);
      ctx.beginPath();
      ctx.arc(pivot.x, pivot.y, r, 0, Math.PI * 2);
      ctx.moveTo(pivot.x - r * 1.6, pivot.y);
      ctx.lineTo(pivot.x + r * 1.6, pivot.y);
      ctx.moveTo(pivot.x, pivot.y - r * 1.6);
      ctx.lineTo(pivot.x, pivot.y + r * 1.6);
      ctx.stroke();
      ctx.restore();
    }
  }
}
