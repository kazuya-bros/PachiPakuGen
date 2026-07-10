import {
  type ChainState,
  alphaBBox,
  chainAverage,
  chainFoldOffsets,
  clampDt,
  createArmSway,
  createChain,
  detectHairStrandCenters,
  envelopeStep,
  noise1d,
  smoothDamp,
  springStep,
  stepChain,
  updateArmSway,
} from "../motionLabPhysics";
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
  MOTION_LAB_HIGHLIGHT_DEFAULTS,
  MOTION_LAB_NOD_DEFAULTS,
  MOTION_LAB_PARALLAX_DEFAULTS,
  MOTION_LAB_PRESENCE_DEFAULTS,
  MOTION_LAB_PRESET_FACTORS,
  MOTION_LAB_SWAY_DEFAULTS,
  MOTION_LAB_TARGET_OPEN,
  MOTION_LAB_TIMELINE,
} from "./constants";

/**
 * グループ描画順の解決: layer-order.json（Step4のレイヤー調整由来）があればそれを優先し、
 * 記載のないグループ（sways等）は既定順の相対位置へ補完する。
 * 無ければ従来の固定z順（armBehindBody で腕をbody背面へ）。
 */
export function drawMotionLabOrderedLayers(
  layerOrder: string[],
  armBehindBody: boolean,
  draws: Record<string, () => void>,
) {
  let order: string[];
  if (layerOrder.length > 0) {
    order = layerOrder.filter(key => key in draws);
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

/**
 * 852話式ソフト房ブレンドワープ: 房を x範囲でハード分割せず、
 * 列ブロックごとに「房中心へのガウシアン重み」で複数房チェーンの変位をブレンドする。
 * 隣接ブロックの変位が連続的に変わるため、前髪のように上部が繋がった髪でも裂けない。
 * （Anime2.5DRig index.html の頂点重み L.sw = exp(-((x-S.x)/σ)^2) 正規化の Canvas2D 版）
 */
export function drawMotionLabStrandBlendWarp(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
  transform: MotionLabLayerTransform,
  options: {
    rootYRatio: number;
    /** anglesSoft がある場合、根元=angles（stiff）・毛先=anglesSoft を u^1.2 で混合（852話式二重バネ） */
    strands: Array<{ x: number; angles: ArrayLike<number>; anglesSoft?: ArrayLike<number> }>;
    stripCount?: number;
    blockWidth?: number;
    /** 変位の倍率（揺れ幅スライダー用。1=既定） */
    offsetScale?: number;
  },
) {
  const strands = options.strands;
  if (strands.length === 0) return;
  const pivotX = width * 0.5;
  const pivotY = height * 0.58;
  const stripCount = options.stripCount ?? 20;
  const blockWidth = options.blockWidth ?? Math.max(16, Math.round(width / 48));
  const span = height * (1 - options.rootYRatio);
  const rowsPerStrand = strands.map(strand => chainFoldOffsets(strand.angles, span));
  const rowsSoftPerStrand = strands.map(strand =>
    strand.anglesSoft ? chainFoldOffsets(strand.anglesSoft, span) : null,
  );
  const segments = strands[0].angles.length;
  // σ = 房間隔の中央値 × 0.6（852話実装と同係数）。1房ならブレンド不要で全幅追従。
  // 下限をブロック幅×2にして、隣接ブロック間の重み変化を必ず滑らかにする（縦縞防止）
  let sigma = width * 0.15;
  if (strands.length > 1) {
    const gaps = strands.slice(1).map((strand, i) => strand.x - strands[i].x).sort((a, b) => a - b);
    sigma = Math.max(blockWidth * 2, gaps[gaps.length >> 1] * 0.6);
  }
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
    const tipRatio = clamp((centerYRatio - options.rootYRatio) / Math.max(0.001, 1 - options.rootYRatio), 0, 1);
    const pos = tipRatio * segments;
    const lower = Math.min(segments, Math.floor(pos));
    const upper = Math.min(segments, lower + 1);
    const frac = pos - lower;
    // 852話式二重バネ混合: 根元はstiff（素早く追従）、毛先ほどsoft（ふわっと遅れる）
    const softMix = Math.pow(tipRatio, 1.2);
    for (let blockX = 0; blockX < width; blockX += blockWidth) {
      const blockW = Math.min(blockWidth, width - blockX);
      const centerX = blockX + blockW * 0.5;
      let totalWeight = 0;
      let dx = 0;
      let dy = 0;
      for (let s = 0; s < strands.length; s += 1) {
        const t = (centerX - strands[s].x) / sigma;
        const weight = Math.exp(-t * t);
        const rows = rowsPerStrand[s];
        let strandDx = rows[lower].dx + (rows[upper].dx - rows[lower].dx) * frac;
        let strandDy = rows[lower].dy + (rows[upper].dy - rows[lower].dy) * frac;
        const rowsSoft = rowsSoftPerStrand[s];
        if (rowsSoft) {
          const softDx = rowsSoft[lower].dx + (rowsSoft[upper].dx - rowsSoft[lower].dx) * frac;
          const softDy = rowsSoft[lower].dy + (rowsSoft[upper].dy - rowsSoft[lower].dy) * frac;
          strandDx += (softDx - strandDx) * softMix;
          strandDy += (softDy - strandDy) * softMix;
        }
        dx += weight * strandDx;
        dy += weight * strandDy;
        totalWeight += weight;
      }
      if (totalWeight > 1e-6) {
        dx = (dx / totalWeight) * (options.offsetScale ?? 1);
        // 縦変位は控えめに（852話実装は y += |dx|×0.12 程度。フル適用だと毛先が段差状に欠ける）
        dy = (dy / totalWeight) * (options.offsetScale ?? 1) * 0.35;
      } else {
        dx = 0;
        dy = 0;
      }
      ctx.drawImage(
        image,
        blockX,
        sourceY,
        blockW,
        stripHeight,
        -pivotX + blockX + dx,
        -pivotY + sourceY + dy,
        blockW + 0.5,
        stripHeight + 1,
      );
    }
  }
  ctx.restore();
}

/**
 * ろてじん式 波揺れワープ（PuruPuruPNGTuber pyokopyokoHairShift 参考）:
 * 位相が毛先(u=1)へ向かって進む複数sinの合成 = 髪を波が伝わって見える。
 * 根元(u<rootYRatio)は固定、発話energyでわずかにブースト。
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
  },
) {
  const TAU = Math.PI * 2;
  const pivotX = width * 0.5;
  const pivotY = height * 0.58;
  const stripCount = options.stripCount ?? 24;
  const sf = options.spatialFreq ?? 1;
  const beat = (options.timeMs / 1000) * (160 / 60) * (options.tempo ?? 1); // PuruPuruと同じBPM160基準
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
    const idleDrift = Math.sin(TAU * (beat * 0.42 + u * 0.82 * sf + options.seed));
    const wave = Math.sin(TAU * (beat * 0.72 + u * 1.55 * sf + 0.16 + options.seed * 0.7));
    const slow = Math.sin(TAU * (beat * 0.5 - 0.255 + options.seed * 0.3));
    const idleFloat = Math.cos(TAU * (beat * 0.36 + u * 0.38 * sf + options.seed));
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
  scratchCtx.globalCompositeOperation = "source-atop";
  scratchCtx.drawImage(irides, gazeX, gazeY, width, height);
  scratchCtx.globalCompositeOperation = "source-over";
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
    strandChainsBack: [],
    envOpen: 0,
    mouthVel: { v: 0 },
    speaking: false,
    blinkWait: randomizePhase ? 0.5 + Math.random() * 2 : 1.5,
    blinkT: -1,
    headTurnT: rand(100),
    nod: { x: 0, v: 0 },
    gaze: { x: 0, y: 0 },
    gazeVelX: { v: 0 },
    gazeVelY: { v: 0 },
    gazeT: rand(100),
    highlightT: rand(100),
    strandChains: [],
    strandChainsSoft: [],
    strandChainsBackSoft: [],
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
  runtime.physics = createMotionLabPhysics();
  // 登場撃力（presence.entryBounce）: 表示開始時に髪・肩・胸へ撃力を入れ「呼ばれた感」を出す
  if (entryBounce > 0) {
    runtime.physics.hairChain.omegas[0] += 0.9 * entryBounce;
    runtime.physics.arm.lift.v += MOTION_LAB_ARM_DEFAULTS.lift.bounce * 0.8 * entryBounce;
    runtime.physics.chest.v += 14 * entryBounce;
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
  const target = motionLabTimelineAt(elapsedMs, settings.timeline, settings.timelineDurationMs);
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

  // ===== 口の開度: A4エンベロープ（attack/release）→ A1 SmoothDamp追従 =====
  // baselineレーンは attackMs=0/releaseMs=0/shapeSmoothing=0 で矩形駆動（従来相当）になる
  const targetOpenBase = MOTION_LAB_TARGET_OPEN[target.mouth];
  const targetOpen = targetOpenBase * (1 - settings.restBias * (1 - target.energy));
  ph.envOpen = envelopeStep(ph.envOpen, targetOpen, settings.attackMs, settings.releaseMs, dt);
  const smoothTime = settings.shapeSmoothing * 0.15;
  runtime.openY = smoothTime < 0.005
    ? ph.envOpen
    : clamp(smoothDamp(runtime.openY, ph.envOpen, ph.mouthVel, smoothTime, dt), 0, 1);

  const width = parts.width;
  const height = parts.height;
  const preset = MOTION_LAB_PRESET_FACTORS[settings.preset];
  const voice = target.energy;

  let bodyTransform: MotionLabLayerTransform;
  let hairFrontTransform: MotionLabLayerTransform;
  let hairBackTransform: MotionLabLayerTransform;
  let hairMeshAngles: ArrayLike<number> | null = null;
  /** 房ごと髪物理の描画リスト（852話式ソフトブレンド用の房中心線＋stiff/softチェーン角。null=一枚チェーン） */
  let hairStrandRender: Array<{ x: number; angles: Float32Array; anglesSoft: Float32Array }> | null = null;
  let hairBackStrandRender: Array<{ x: number; angles: Float32Array; anglesSoft: Float32Array }> | null = null;

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
    ph.breathPhase += dt * ((Math.PI * 2) / 3.6);
    ph.swayPhase += dt * 1.35;
    if (ph.breathPhase > Math.PI * 200) ph.breathPhase -= Math.PI * 200;
    if (ph.swayPhase > Math.PI * 200) ph.swayPhase -= Math.PI * 200;
    const breath = Math.sin(ph.breathPhase);
    const sway = Math.sin(ph.swayPhase);
    const rootX = sway * 1.2 * settings.bodySwayAmplitude * preset.body;
    // 発話ぴょこバウンス（PuruPuru pyoko参考）: 定数下げではなくバネで「ぴょこん」と弾む
    springStep(ph.pyoko, -voice * settings.pyokoBounce, 90, 10, dt);
    const breathY = breath * 3.2 * settings.breathAmplitude * preset.breath;
    const rootY = breathY + ph.pyoko.x;
    // 852話式: 頭・髪は胸の呼吸に少し遅れて追従する（-0.6位相の遅延呼吸）
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
      const wind = Math.sin(ph.noiseT * 1.7) * windAmp + noise1d(ph.noiseT * 0.6) * windAmp * 0.6;
      const drive = clamp(-settings.hairDrive * settings.hairMotionStrength * ph.rootVX * 0.05, -0.2, 0.2);
      stepChain(ph.hairChain, drive + wind, settings.hairK, settings.hairC, dt, 0.5);
      hairMeshAngles = ph.hairChain.angles;
      // 房ごと髪物理（852話式・§8.1 #5）: 毛先輪郭ピークで房中心線を自動検出し、
      // 房ごとに独立チェーン＋風の位相ずらしで駆動（描画はガウシアン重みのソフトブレンド）。
      // 房ごとの位相を大きくずらし・風を強めて（×1.6）一枚チェーンとの差を体感しやすくする
      // 852話式二重バネ: stiff（硬く素早い、根元側）と soft（柔らかく遅い、毛先側）の
      // 2本のチェーンを同じ駆動で回し、描画時に u^1.2 で混合する
      const stepStrandChains = (
        image: HTMLImageElement,
        chains: ChainState[],
        chainsSoft: ChainState[],
        driveScale: number,
        phaseSeed: number,
        // 後ろ髪の「大波」化: 剛性を下げてゆっくり大きく、風の時間も遅く
        kScale = 1,
        windTempo = 1,
      ): Array<{ x: number; angles: Float32Array; anglesSoft: Float32Array }> | null => {
        const centers = detectHairStrandCenters(image, MOTION_LAB_HAIR_SEGMENTS);
        if (centers.length <= 1) return null;
        if (chains.length !== centers.length || chainsSoft.length !== centers.length) {
          chains.length = 0;
          chainsSoft.length = 0;
          for (let i = 0; i < centers.length; i += 1) {
            chains.push(createChain(MOTION_LAB_HAIR_SEGMENTS));
            chainsSoft.push(createChain(MOTION_LAB_HAIR_SEGMENTS));
          }
        }
        return centers.map((centerX, index) => {
          const chain = chains[index];
          const chainSoft = chainsSoft[index];
          chain.t += dt;
          chainSoft.t = chain.t;
          const strandWind =
            (Math.sin((ph.noiseT * windTempo + index * 1.7 + phaseSeed) * 1.7) * windAmp +
              noise1d(chain.t * 0.6 * windTempo + index * 29.3 + phaseSeed * 11) * windAmp) * 1.15;
          const target = (drive + strandWind) * driveScale;
          // stiff: k×2.2/c×1.4（本家 k70/c9 相当の比率）、soft: k×0.35/c×0.7（毛先のふわ遅れ・減衰は強めに）
          // 角度クランプ±0.28: 6段累積で過大な折れ（毛先の縦縞・欠け）を防ぐ
          stepChain(chain, target, settings.hairK * 2.2 * kScale, settings.hairC * 1.4, dt, 0.28);
          stepChain(chainSoft, target, settings.hairK * 0.35 * kScale, settings.hairC * 0.7, dt, 0.28);
          return { x: centerX, angles: chain.angles, anglesSoft: chainSoft.angles };
        });
      };
      if (settings.strandsEnabled && images.hair) {
        hairStrandRender = stepStrandChains(images.hair, ph.strandChains, ph.strandChainsSoft, 1, 0);
      }
      // 後ろ髪も房分割対象（振幅は hairBackScale に従う）。
      // 大波特性: 剛性半分（固有振動数が低く、ゆっくり大きくたゆたう）＋風の時間0.55倍
      if (settings.strandsEnabled && images.hairBack) {
        hairBackStrandRender = stepStrandChains(
          images.hairBack,
          ph.strandChainsBack,
          ph.strandChainsBackSoft,
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
    // 852話式: 体の回転は全レイヤーへ継承する（本家は最終段で全頂点に体回転を適用）。
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

  // ===== パララックス首振り（852話氏 Anime2.5DRig由来・§8.3） =====
  // 駆動 = ノイズドリフト（headTurn）＋発話開始の頷きバネ（headNod）。
  // 各レイヤーへ depth × SHIFT_MAX の水平シフト＋シアーを適用（縦はシフトのみ）
  let eyeMouthTransform = bodyTransform;
  let parallaxArmDx = 0;
  let parallaxArmDy = 0;
  // ランダムグランス（852話 auto.rand参考）: 1.4〜4秒ごとに顔向き・視線の目標が
  // ふっと変わり、SmoothDampで滑らかに移行する（連続ノイズだけより「意図」が出る）
  if (settings.randomGlance && settings.layerMode !== "simple") {
    ph.glanceWait -= dt;
    if (ph.glanceWait <= 0) {
      ph.glanceWait = 1.4 + Math.random() * 2.6;
      ph.glanceHeadTarget = (Math.random() * 2 - 1) * 0.45 * settings.glanceStrength;
      const glanceRange = width * MOTION_LAB_GAZE_DEFAULTS.rangeRatio * 2.2 * settings.glanceStrength;
      ph.glanceGaze.x = (Math.random() * 2 - 1) * glanceRange;
      ph.glanceGaze.y = (Math.random() * 2 - 1) * glanceRange * 0.5;
    }
    ph.glanceHead = smoothDamp(ph.glanceHead, ph.glanceHeadTarget, ph.glanceHeadVel, 0.5, dt);
  } else {
    ph.glanceHead = smoothDamp(ph.glanceHead, 0, ph.glanceHeadVel, 0.5, dt);
  }
  if (settings.layerMode !== "simple" && settings.parallaxScale > 0) {
    ph.headTurnT += dt * MOTION_LAB_PARALLAX_DEFAULTS.driftSpeed;
    const headTurn = clamp(noise1d(ph.headTurnT) * 0.56 + ph.glanceHead, -1, 1) * settings.parallaxScale;
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
        // 二次追従化（PuruPuru/852話式: 一次バウンスは体1本、肩はそれに遅れて追従）:
        // 発話バウンス有効時は独自撃力を1/4に抑え、体のY速度カップリングを強めて
        // 「体が弾む→肩が遅れてついてくる」の連動にする
        liftCoupling: MOTION_LAB_ARM_DEFAULTS.lift.coupling * (settings.pyokoBounce > 0 ? 2.8 : 1) * settings.liftStrength,
        liftBounce: MOTION_LAB_ARM_DEFAULTS.lift.bounce * (settings.pyokoBounce > 0 ? 0.25 : 1) * settings.liftStrength,
        liftMax: MOTION_LAB_ARM_DEFAULTS.lift.max,
      },
      dt,
      bodyTransform.x,
      bodyTransform.y,
      speechStarted,
    );
  }

  // ===== 胸揺れ: 縦バネ1本（低周波・強減衰）=====
  // 852話式（bustTgt=体の動き由来）に合わせ、独立発振ではなく体のY速度
  // （発話バウンス・呼吸を含む）への遅延追従=二次揺れとして駆動する。
  // 発話バウンス無効時のみ、視認用の弱い独自撃力・揺らぎでフォールバック
  let chestOffsetY = 0;
  if (images.chest && animateParts && settings.chestMax > 0) {
    const pyokoActive = settings.pyokoBounce > 0;
    if (speechStarted) ph.chest.v += pyokoActive ? 14 : 45;
    const chestNoise = pyokoActive
      ? 0
      : noise1d(ph.noiseT * 0.8 + 13.7) * settings.chestMax * 0.35;
    const driveY = clamp(-0.6 * ph.rootVY + chestNoise, -settings.chestMax, settings.chestMax);
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
    const bbox = alphaBBox(image);
    // 回転軸: エディタ指定 > bbox上端中央（=肩推定）＋armPivotRatio下方調整
    const pivot = settings.pivots[part] ?? {
      x: bbox.x + bbox.w / 2,
      y: bbox.y + bbox.h * settings.armPivotRatio,
    };
    let angle = (out?.rigid ?? 0) * (settings.swingScale[part] ?? 1);
    const range = settings.rangesDeg[part] ?? 0;
    if (range > 0) angle = clamp(angle, (-range * Math.PI) / 180, (range * Math.PI) / 180);
    // 体の傾きを継承（852話式: 体回転は全レイヤーに掛かる）
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
        rootYRatio: backRootYRatio,
        strands: hairBackStrandRender,
        offsetScale: settings.swingScale.hair_back ?? 1,
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
      });
    } else {
      drawWithOptionalPivot(images.hairBack, clampRotationDeg(hairBackTransform, "hair_back"), "hair_back");
    }
  };
  const drawBody = () => {
    drawMotionLabLayer(ctx, images.body, width, height, bodyTransform);
  };
  // 胸: body(0) と arm(1) の間 = 0.5相当（設計書§8.5）
  const drawChest = () => {
    if (!images.chest) return;
    drawMotionLabLayer(ctx, images.chest, width, height, {
      ...bodyTransform,
      y: bodyTransform.y + chestOffsetY,
    });
  };

  // 汎用揺れパーツ sway_*: 腕と同系のチェーン物理（ピボット=bbox上端中央）。
  // 獣耳（sway_ear*）は頭に付いているパーツなので、前髪と同じ頭基準の変換
  // （呼吸遅延・パララックス首振り・頷き込み）に連動させ、その上でツイッチを乗せる
  const drawSways = () => {
    for (const [name, image] of Object.entries(images.sways)) {
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
        const noise = noise1d(chainState.t * 0.5) * MOTION_LAB_SWAY_DEFAULTS.noise;
        const drive = clamp(
          -MOTION_LAB_ARM_DEFAULTS.coupling * ph.rootVX * 0.05,
          -MOTION_LAB_SWAY_DEFAULTS.maxAngle,
          MOTION_LAB_SWAY_DEFAULTS.maxAngle,
        );
        // 獣耳ピコピコ（sway_ear*限定・オプション）: 数秒ごとに縦バネへ撃力を入れて
        // 上下に「ピコッ」と跳ねさせる。短い間隔の連続ツイッチ（ピコピコ感）を確率で挟む
        if (settings.earTwitch && isEar) {
          let twitch = ph.earTwitches.get(name);
          if (!twitch) {
            twitch = { wait: 2 + Math.random() * 5, spring: { x: 0, v: 0 } };
            ph.earTwitches.set(name, twitch);
          }
          twitch.wait -= dt;
          if (twitch.wait <= 0) {
            twitch.spring.v -= MOTION_LAB_EAR_TWITCH.bounce * settings.earTwitchScale; // 上向きの撃力
            chainState.omegas[0] += (Math.random() < 0.5 ? 1 : -1) * MOTION_LAB_EAR_TWITCH.rotKick * settings.earTwitchScale;
            twitch.wait = Math.random() < 0.45
              ? MOTION_LAB_EAR_TWITCH.doubleMin + Math.random() * MOTION_LAB_EAR_TWITCH.doubleRange
              : MOTION_LAB_EAR_TWITCH.intervalMin + Math.random() * MOTION_LAB_EAR_TWITCH.intervalRange;
          }
          springStep(twitch.spring, 0, MOTION_LAB_EAR_TWITCH.k, MOTION_LAB_EAR_TWITCH.c, dt);
          twitch.spring.x = clamp(twitch.spring.x, -MOTION_LAB_EAR_TWITCH.maxPx, MOTION_LAB_EAR_TWITCH.maxPx);
          twitchOffsetY = twitch.spring.x;
        }
        stepChain(chainState, drive + noise, MOTION_LAB_SWAY_DEFAULTS.k, MOTION_LAB_SWAY_DEFAULTS.c, dt, MOTION_LAB_SWAY_DEFAULTS.maxAngle);
        // 土台の傾き（耳=頭、その他=体）＋チェーン揺れ（＋耳は縦ピコ）
        angle = chainAverage(chainState) + (base.rotationDeg * Math.PI) / 180;
      }
      const bbox = alphaBBox(image);
      drawMotionLabPivotLayer(ctx, image, width, height, base, { x: bbox.x + bbox.w / 2, y: bbox.y }, angle, twitchOffsetY);
    }
  };

  // ===== 視線ドリフト＋瞳クリップ（852話式・§8.4）: eyewhite < irides < highlight < eye連番 =====
  const drawEyeCluster = () => {
  if (images.eyewhite && images.irides) {
    let targetX = 0;
    let targetY = 0;
    if (animateParts && settings.gazeEnabled) {
      ph.gazeT += dt * MOTION_LAB_GAZE_DEFAULTS.driftSpeed;
      // 基本正面＋ごく小さな揺らぎ。発話中は正面へ復帰（話しかけている感・§8.6）
      if (!speaking) {
        const range = width * MOTION_LAB_GAZE_DEFAULTS.rangeRatio * settings.gazeStrength;
        targetX = noise1d(ph.gazeT) * range;
        targetY = noise1d(ph.gazeT + 53.7) * range * 0.6;
        // ランダムグランス: たまに視線がふっと別の場所へ（発話中は正面復帰を維持）
        if (settings.randomGlance) {
          targetX += ph.glanceGaze.x;
          targetY += ph.glanceGaze.y;
        }
      }
    }
    ph.gaze.x = smoothDamp(ph.gaze.x, targetX, ph.gazeVelX, MOTION_LAB_GAZE_DEFAULTS.smoothTime, dt);
    ph.gaze.y = smoothDamp(ph.gaze.y, targetY, ph.gazeVelY, MOTION_LAB_GAZE_DEFAULTS.smoothTime, dt);
    drawMotionLabGaze(ctx, runtime, images.eyewhite, images.irides, width, height, eyeMouthTransform, ph.gaze.x, ph.gaze.y);
  }
  if (images.highlight) {
    // ハイライトドリフト（±1〜2px・ろてじん氏の目元演出参考）
    let highlightX = 0;
    let highlightY = 0;
    if (animateParts) {
      ph.highlightT += dt * MOTION_LAB_HIGHLIGHT_DEFAULTS.speed;
      highlightX = noise1d(ph.highlightT) * MOTION_LAB_HIGHLIGHT_DEFAULTS.driftPx;
      highlightY = noise1d(ph.highlightT + 17.3) * MOTION_LAB_HIGHLIGHT_DEFAULTS.driftPx * 0.8;
    }
    drawMotionLabLayer(ctx, images.highlight, width, height, {
      ...eyeMouthTransform,
      x: eyeMouthTransform.x + highlightX,
      y: eyeMouthTransform.y + highlightY,
    });
  }

  // ===== 目: eye連番（frame 0=開き→最終=閉じ）＋自動瞬き（PuruPuru参考） =====
  if (images.eyeFrames.length > 0) {
    let blinkValue = 0;
    if (images.eyeFrames.length > 1 && settings.blinkEnabled) {
      const blink = MOTION_LAB_BLINK_DEFAULTS;
      if (ph.blinkT >= 0) {
        ph.blinkT += dt * 1000;
        if (ph.blinkT < blink.closeMs) {
          blinkValue = ph.blinkT / blink.closeMs;
        } else if (ph.blinkT < blink.closeMs + blink.openMs) {
          blinkValue = 1 - (ph.blinkT - blink.closeMs) / blink.openMs;
        } else {
          ph.blinkT = -1;
          // blinkRate=頻度倍率（2で間隔半分）
          ph.blinkWait =
            (blink.intervalMin + Math.random() * (blink.intervalMax - blink.intervalMin)) /
            Math.max(0.1, settings.blinkRate);
        }
      } else {
        ph.blinkWait -= dt;
        if (ph.blinkWait <= 0) ph.blinkT = 0;
      }
    }
    const eyeIndex = Math.round(clamp(blinkValue, 0, 1) * (images.eyeFrames.length - 1));
    const eyeFrame = images.eyeFrames[eyeIndex];
    if (eyeFrame) drawMotionLabLayer(ctx, eyeFrame, width, height, eyeMouthTransform);
  }
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
      // ろてじん式 波揺れ（進行波・根元固定）。体の傾きは維持して波だけ乗せる
      drawMotionLabWaveWarp(ctx, images.hair, width, height, { ...hairFrontTransform, rotationDeg: bodyTransform.rotationDeg }, {
        rootYRatio: hairRootYRatio,
        timeMs: elapsedMs,
        strength: settings.hairWaveStrength * preset.hair * settings.hairMotionStrength,
        seed: 0.35,
        voice,
      });
    } else if (settings.layerMode === "mesh" && hairStrandRender) {
      // 852話式ソフト房ブレンド: 房中心線へのガウシアン重みで複数チェーンを混合
      drawMotionLabStrandBlendWarp(ctx, images.hair, width, height, hairFrontTransform, {
        rootYRatio: hairRootYRatio,
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
        : null;
      if (partImage) {
        const bbox = alphaBBox(partImage);
        pivot = part.startsWith("arm_")
          ? { x: bbox.x + bbox.w / 2, y: bbox.y + bbox.h * settings.armPivotRatio }
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
