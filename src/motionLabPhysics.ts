/**
 * Motion Lab 物理エンジン — reference/animation-lab/js からの検証済み移植
 *
 * 移植元:
 * - 02-utils.js        smoothDamp / 1D Perlinノイズ / alphaBBox（肩ピボット自動推定）
 * - 30-body-methods.js B1スプリング遅延追従 / B3角度チェーン（springStep・サブステップ・クランプ）
 * - 32-arm-sway.js     腕チェーン＋肩の弾み（lift）
 * - 11-mora-player.js  A4開度エンベロープ（attack/release指数スムージング）
 *
 * 数値安定化（必須維持）: dt≤100msクランプ / dt>1/60でサブステップ2分割 / 角度・liftクランプ
 */

export interface SpringState {
  x: number;
  v: number;
}

export interface ChainState {
  angles: Float32Array;
  omegas: Float32Array;
  /** ノイズ・風の位相（表示ごとにランダム化 = presence.randomizePhase） */
  t: number;
}

export interface ArmSwayState {
  left: ChainState;
  right: ChainState;
  /** 肩の上下（両肩共通の1本のバネ、px） */
  lift: SpringState;
  prevRootX: number | null;
  prevRootY: number | null;
}

export interface ArmSwayParams {
  k: number;
  c: number;
  coupling: number;
  noise: number;
  maxAngle: number;
  /** アイドルスイング振幅（rad）: 体速度と無関係に常時ゆっくり振る駆動。0=無効 */
  idleSwing: number;
  liftEnabled: boolean;
  liftCoupling: number;
  liftBounce: number;
  liftMax: number;
}

export interface ArmSwayOutput {
  left: { rigid: number; lift: number };
  right: { rigid: number; lift: number };
}

const clamp = (v: number, min: number, max: number): number => (v < min ? min : v > max ? max : v);

/** dtの安全クランプ（タブ復帰などの巨大dtで発散させない） */
export function clampDt(dt: number): number {
  return clamp(dt, 0, 0.1);
}

/** Unity互換 SmoothDamp（臨界減衰バネによる滑らかな追従） */
export function smoothDamp(
  current: number,
  target: number,
  velRef: { v: number },
  smoothTime: number,
  dt: number,
): number {
  smoothTime = Math.max(0.0001, smoothTime);
  const omega = 2 / smoothTime;
  const x = omega * dt;
  const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const change = current - target;
  const temp = (velRef.v + omega * change) * dt;
  velRef.v = (velRef.v - omega * temp) * exp;
  let output = target + (change + temp) * exp;
  // オーバーシュート防止
  if (target - current > 0 === output > target) {
    output = target;
    velRef.v = dt > 0 ? (output - target) / dt : 0;
  }
  return output;
}

// 1D Perlin風グラデーションノイズ（-1..1）
const PERM = new Uint8Array(512);
(function seedPerm() {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) p[i] = i;
  let s = 1234567;
  for (let i = 255; i > 0; i -= 1) {
    s = (s * 16807) % 2147483647;
    const j = s % (i + 1);
    const t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
  for (let i = 0; i < 512; i += 1) PERM[i] = p[i & 255];
})();

export function noise1d(x: number): number {
  const i0 = Math.floor(x) & 255;
  const i1 = (i0 + 1) & 255;
  const f = x - Math.floor(x);
  const u = f * f * f * (f * (f * 6 - 15) + 10); // smootherstep
  const g0 = PERM[i0] / 127.5 - 1;
  const g1 = PERM[i1] / 127.5 - 1;
  const n0 = g0 * f;
  const n1 = g1 * (f - 1);
  return (n0 + (n1 - n0) * u) * 2;
}

/** バネ-ダンパー1本（半陰的オイラー＋サブステップ）。dtは事前にclampDt推奨 */
export function springStep(state: SpringState, target: number, k: number, c: number, dt: number): number {
  const dtc = clampDt(dt);
  const steps = dtc > 1 / 60 ? 2 : 1;
  const h = dtc / steps;
  for (let i = 0; i < steps; i += 1) {
    const a = -k * (state.x - target) - c * state.v;
    state.v += a * h;
    state.x += state.v * h;
  }
  return state.x;
}

export function createChain(segments: number, randomizePhase = true): ChainState {
  return {
    angles: new Float32Array(segments),
    omegas: new Float32Array(segments),
    t: randomizePhase ? Math.random() * 100 : 0,
  };
}

/**
 * B3角度チェーン1ステップ。
 * 根元行の目標=driveTarget、以降は親行に追従（1行分の伝播遅延が波を生む）。
 * 毛先ほど低剛性 k_i = k×(1 - 0.6×i/N)。
 */
export function stepChain(
  chain: ChainState,
  driveTarget: number,
  k: number,
  c: number,
  dt: number,
  maxAngle: number,
): void {
  const dtc = clampDt(dt);
  const steps = dtc > 1 / 60 ? 2 : 1;
  const h = dtc / steps;
  const { angles, omegas } = chain;
  const n = angles.length;
  for (let s = 0; s < steps; s += 1) {
    for (let i = 0; i < n; i += 1) {
      const target = i === 0 ? driveTarget : angles[i - 1];
      const ki = k * (1 - 0.6 * (i / n));
      const a = -ki * (angles[i] - target) - c * omegas[i];
      omegas[i] += a * h;
      angles[i] += omegas[i] * h;
      angles[i] = clamp(angles[i], -maxAngle, maxAngle);
    }
  }
}

/** チェーン角の平均（DOM/Canvas剛体回転フォールバック用） */
export function chainAverage(chain: ChainState): number {
  let sum = 0;
  for (let i = 0; i < chain.angles.length; i += 1) sum += chain.angles[i];
  return chain.angles.length > 0 ? sum / chain.angles.length : 0;
}

/**
 * チェーン角から各行の折れ線オフセットを計算（メッシュ描画用）。
 * 行0（根元）固定。行iの基準点は「行i-1の基準点＋累積回転した段ベクトル」。
 * @param spanHeight チェーンが占める縦の長さ（px）
 * @returns 各行の { dx, dy }（rows = angles.length + 1）
 */
export function chainFoldOffsets(
  angles: ArrayLike<number>,
  spanHeight: number,
): Array<{ dx: number; dy: number }> {
  const n = angles.length;
  const stepY = n > 0 ? spanHeight / n : spanHeight;
  const rows: Array<{ dx: number; dy: number }> = [{ dx: 0, dy: 0 }];
  let cum = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i += 1) {
    cum += angles[i];
    dx += Math.sin(cum) * stepY;
    dy += (Math.cos(cum) - 1) * stepY;
    rows.push({ dx, dy });
  }
  return rows;
}

/** A4開度エンベロープ: 立ち上がりattackMs / 立ち下がりreleaseMsの指数スムージング */
export function envelopeStep(
  current: number,
  target: number,
  attackMs: number,
  releaseMs: number,
  dt: number,
): number {
  const tauMs = target > current ? attackMs : releaseMs;
  if (tauMs <= 0) return target;
  const alpha = 1 - Math.exp(-dt / Math.max(0.001, tauMs / 1000));
  return current + (target - current) * alpha;
}

export function createArmSway(segments: number, randomizePhase = true): ArmSwayState {
  return {
    left: createChain(segments, randomizePhase),
    right: createChain(segments, randomizePhase),
    lift: { x: 0, v: 0 },
    prevRootX: null,
    prevRootY: null,
  };
}

/**
 * 腕揺れ＋肩の弾み（32-arm-sway.js ArmSway.update の移植）。
 * 駆動 = 体のX速度カップリング＋左右で位相をずらした常時微揺れノイズ。
 * lift = 体のY速度への遅延追従＋発話開始の撃力（両肩共通1本のバネ）。
 */
export function updateArmSway(
  state: ArmSwayState,
  params: ArmSwayParams,
  dt: number,
  rootX: number,
  rootY: number,
  speechStarted: boolean,
): ArmSwayOutput {
  const vx = state.prevRootX === null || dt <= 0 ? 0 : (rootX - state.prevRootX) / dt;
  const vy = state.prevRootY === null || dt <= 0 ? 0 : (rootY - state.prevRootY) / dt;
  state.prevRootX = rootX;
  state.prevRootY = rootY;

  // ===== 肩の弾み（縦方向の遅延追従＋発話バウンス） =====
  if (params.liftEnabled) {
    if (speechStarted) state.lift.v += params.liftBounce; // 撃力（px/s）
    const driveY = clamp(-params.liftCoupling * vy, -params.liftMax, params.liftMax);
    const kL = params.k * 0.8;
    const cL = params.c * 0.9;
    const hL = clampDt(dt);
    const a = -kL * (state.lift.x - driveY) - cL * state.lift.v;
    state.lift.v += a * hL;
    state.lift.x = clamp(state.lift.x + state.lift.v * hL, -params.liftMax, params.liftMax);
  } else {
    state.lift.x *= Math.max(0, 1 - dt * 8); // OFF時は滑らかにゼロへ
    state.lift.v = 0;
  }

  const dtc = clampDt(dt);
  const drive = clamp(-params.coupling * vx * 0.05, -params.maxAngle, params.maxAngle);

  const result = {} as Record<"left" | "right", { rigid: number; lift: number }>;
  for (const side of ["left", "right"] as const) {
    const chain = state[side];
    chain.t += dtc;
    // 左右で位相をずらした常時微揺れ（完全同期だと機械的に見える）
    const noise = noise1d(chain.t * 0.5 + (side === "left" ? 0 : 37)) * params.noise;
    // アイドルスイング: 振り子状のゆっくりした常時スイング（左右で位相ずらし）
    const idle = params.idleSwing > 0
      ? Math.sin(chain.t * 0.9 + (side === "left" ? 0 : 2.1)) * params.idleSwing
      : 0;
    stepChain(chain, drive + noise + idle, params.k, params.c, dtc, params.maxAngle);
    result[side] = { rigid: chainAverage(chain), lift: state.lift.x };
  }
  return result;
}

/**
 * 不透明ピクセルのバウンディングボックス（肩・揺れパーツのピボット自動推定用）。
 * 結果は画像要素単位でキャッシュ。
 */
export interface AlphaBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

const bboxCache = new WeakMap<HTMLImageElement, AlphaBBox>();

export function alphaBBox(image: HTMLImageElement): AlphaBBox {
  const cached = bboxCache.get(image);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const fallback: AlphaBBox = { x: 0, y: 0, w: canvas.width, h: canvas.height };
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return fallback;
  ctx.drawImage(image, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      if (data[(y * canvas.width + x) * 4 + 3] > 8) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  const box: AlphaBBox =
    maxX < 0 ? fallback : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
  bboxCache.set(image, box);
  return box;
}

// ===== 房ごと髪物理 =====

export interface HairStrand {
  /** 毛先輪郭のピーク位置（画像座標px） */
  x: number;
  /** ピーク列にある最上端・最下端の不透明画素（画像座標px） */
  rootY: number;
  tipY: number;
}

export interface HairStrandSpringState {
  stiff: SpringState;
  soft: SpringState;
  phase: number;
}

export interface HairStrandSpringOutput {
  stiffDx: number;
  softDx: number;
}

const strandCache = new WeakMap<HTMLImageElement, Map<number, HairStrand[]>>();

export function createHairStrandSpring(phase = 0): HairStrandSpringState {
  return {
    stiff: { x: 0, v: 0 },
    soft: { x: 0, v: 0 },
    phase,
  };
}

/**
 * Anime2.5DRigの房単位の二重バネを変更・適応。
 * Upstream: 852wa/Anime2.5DRig@d4882586 (MIT, Copyright 2026 hakoniwa)
 * Modified for PachiPakuGen: 透過PNG入力、変位制限、Canvas描画へ適合。
 * 1房を硬いバネと柔らかいバネの2本で追従させる。
 */
export function stepHairStrandSpring(
  state: HairStrandSpringState,
  target: number,
  dt: number,
  stiffK = 70,
  stiffC = 9,
  maxDisplacement = Number.POSITIVE_INFINITY,
): HairStrandSpringOutput {
  const safeK = Math.max(0.001, stiffK);
  const safeC = Math.max(0, stiffC);
  const safeLimit = Number.isFinite(maxDisplacement)
    ? Math.max(0, maxDisplacement)
    : Number.POSITIVE_INFINITY;
  springStep(state.stiff, target, safeK, safeC, dt);
  springStep(state.soft, target, safeK * (16 / 70), safeC * (1.3 / 9), dt);
  return {
    stiffDx: clamp(-(state.stiff.x - target) * 2.2, -safeLimit, safeLimit),
    softDx: clamp(-(state.soft.x - target) * 3, -safeLimit, safeLimit),
  };
}

/**
 * 毛先輪郭（各列の最下端不透明y）の局所ピーク列を返す。
 * 谷で区切った領域の中央ではなく、実際の輪郭ピークxを房中心として使う。
 * 検出できない場合は空配列（=一枚チェーンへフォールバック）。
 */
export function detectHairStrandCenters(image: HTMLImageElement, maxStrands = 6): number[] {
  const strands = detectHairStrands(image, maxStrands);
  return strands.length > 1 ? strands.map(strand => strand.x) : [];
}

/**
 * Anime2.5DRig `detectStrands` をTypeScriptへ変更・適応。
 * Upstream: 852wa/Anime2.5DRig@d4882586 (MIT, Copyright 2026 hakoniwa)
 * Modified for PachiPakuGen: alpha配列入力、キャッシュ、描画側との接続を変更。
 * 輪郭を41pxで平滑化し、突出量と最小距離でピークを絞る。
 */
export function detectHairStrandsFromAlpha(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  maxStrands = 6,
): HairStrand[] {
  if (width < 16 || height < 1 || maxStrands < 1 || alpha.length < width * height) return [];

  const top = new Int32Array(width).fill(-1);
  const bottom = new Float32Array(width);
  let minX = width;
  let maxX = -1;
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      if (alpha[y * width + x] > 16) {
        top[x] = y;
        break;
      }
    }
    if (top[x] < 0) continue;
    for (let y = height - 1; y >= 0; y -= 1) {
      if (alpha[y * width + x] > 16) {
        bottom[x] = y;
        break;
      }
    }
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
  }
  if (maxX < 0) return [];

  // 固定幅のbox smooth。端も同じ41で割ることで輪郭外を0として扱う。
  const kernelSize = 41;
  const halfKernel = 20;
  const prefix = new Float32Array(width + 1);
  for (let x = 0; x < width; x += 1) prefix[x + 1] = prefix[x] + bottom[x];
  const smoothed = new Float32Array(width);
  for (let x = 0; x < width; x += 1) {
    const from = Math.max(0, x - halfKernel);
    const to = Math.min(width - 1, x + halfKernel);
    smoothed[x] = (prefix[to + 1] - prefix[from]) / kernelSize;
  }

  const minSeparation = Math.max(30, Math.round((maxX - minX + 1) / (maxStrands * 1.6)));
  const candidates: Array<{ x: number; prominence: number }> = [];
  for (let x = 1; x < width - 1; x += 1) {
    if (!(smoothed[x] > smoothed[x - 1] && smoothed[x] >= smoothed[x + 1])) continue;
    let leftMin = smoothed[x];
    let rightMin = smoothed[x];
    for (let j = x - 1; j >= 0; j -= 1) {
      if (smoothed[j] > smoothed[x]) break;
      leftMin = Math.min(leftMin, smoothed[j]);
    }
    for (let j = x + 1; j < width; j += 1) {
      if (smoothed[j] > smoothed[x]) break;
      rightMin = Math.min(rightMin, smoothed[j]);
    }
    const prominence = smoothed[x] - Math.max(leftMin, rightMin);
    if (prominence >= 10 && top[x] >= 0) candidates.push({ x, prominence });
  }

  candidates.sort((a, b) => b.prominence - a.prominence);
  const selected: number[] = [];
  for (const candidate of candidates) {
    if (selected.every(x => Math.abs(x - candidate.x) >= minSeparation)) selected.push(candidate.x);
    if (selected.length >= maxStrands) break;
  }

  // 輪郭が滑らかでピークが足りない画像にも房物理を適用できるよう、内容のある列で補完する。
  const margin = Math.min(30, Math.max(0, Math.floor((maxX - minX) / 4)));
  for (let guard = 0; selected.length < maxStrands && guard < 50; guard += 1) {
    let best = -1;
    let bestDistance = -1;
    for (let sample = 0; sample < 40; sample += 1) {
      const start = minX + margin;
      const end = maxX - margin;
      const x = Math.round(start + ((end - start) * sample) / 39);
      if (x < 0 || x >= width || top[x] < 0 || selected.includes(x)) continue;
      const distance = selected.length === 0
        ? Number.MAX_SAFE_INTEGER - sample
        : Math.min(...selected.map(existing => Math.abs(x - existing)));
      if (distance > bestDistance) {
        bestDistance = distance;
        best = x;
      }
    }
    if (best < 0) break;
    selected.push(best);
  }

  selected.sort((a, b) => a - b);
  return selected
    .filter(x => top[x] >= 0)
    .map(x => ({ x, rootY: top[x], tipY: bottom[x] }));
}

export function detectHairStrands(image: HTMLImageElement, maxStrands = 6): HairStrand[] {
  const limit = Math.max(1, Math.floor(maxStrands));
  const cachedByLimit = strandCache.get(image);
  const cached = cachedByLimit?.get(limit);
  if (cached) return cached;

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx || canvas.width < 16 || canvas.height < 1) return [];
  ctx.drawImage(image, 0, 0);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const alpha = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < alpha.length; i += 1) alpha[i] = rgba[i * 4 + 3];
  const strands = detectHairStrandsFromAlpha(alpha, canvas.width, canvas.height, limit);
  const nextCache = cachedByLimit ?? new Map<number, HairStrand[]>();
  nextCache.set(limit, strands);
  strandCache.set(image, nextCache);
  return strands;
}
