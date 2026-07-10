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

// ===== 房ごと髪物理（B3強化。852話氏 Anime2.5DRig rigger.js の毛先輪郭ピーク検出を参考） =====

export interface HairStrand {
  /** 房のx範囲（画像座標px） */
  x0: number;
  x1: number;
}

const strandCache = new WeakMap<HTMLImageElement, HairStrand[]>();
const strandCenterCache = new WeakMap<HTMLImageElement, number[]>();

/**
 * 852話式の房中心線検出: 毛先輪郭（各列の最下端不透明y）の局所ピーク列を返す。
 * x範囲での分割（ハードスライス）ではなく、描画側でガウシアン重みブレンドに使う。
 * 検出できない場合は空配列（=一枚チェーンへフォールバック）。
 */
export function detectHairStrandCenters(image: HTMLImageElement, maxStrands = 6): number[] {
  const cached = strandCenterCache.get(image);
  if (cached) return cached;
  const strands = detectHairStrands(image, maxStrands);
  const bbox = alphaBBox(image);
  // detectHairStrands はピーク間の谷で区切った範囲を返すので、各範囲の中心を房中心とみなす。
  // 1房（=分割不能）のときは空を返してフォールバックさせる
  const centers = strands.length > 1 && bbox.w >= 16
    ? strands.map(strand => (strand.x0 + strand.x1) / 2)
    : [];
  strandCenterCache.set(image, centers);
  return centers;
}

/**
 * 髪画像の下端輪郭（各列で最も下の不透明ピクセル）の局所ピークから房を検出する。
 * ピーク=毛先が最も下へ伸びる列、境界=隣接ピーク間の谷。最大 maxStrands 房。
 * 検出できない場合は画像全幅の1房を返す（従来のB3一枚チェーンと等価）。
 */
export function detectHairStrands(image: HTMLImageElement, maxStrands = 6): HairStrand[] {
  const cached = strandCache.get(image);
  if (cached) return cached;

  const bbox = alphaBBox(image);
  const whole: HairStrand[] = [{ x0: bbox.x, x1: bbox.x + bbox.w - 1 }];
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx || bbox.w < 16) {
    strandCache.set(image, whole);
    return whole;
  }
  ctx.drawImage(image, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

  // 各列の毛先深さ（最下端の不透明y）。不透明ピクセルの無い列は -1
  const depth = new Float32Array(bbox.w).fill(-1);
  for (let ix = 0; ix < bbox.w; ix += 1) {
    const x = bbox.x + ix;
    for (let y = bbox.y + bbox.h - 1; y >= bbox.y; y -= 1) {
      if (data[(y * canvas.width + x) * 4 + 3] > 8) {
        depth[ix] = y;
        break;
      }
    }
  }

  // 平滑化（箱型フィルタ）で輪郭ノイズを除去
  const radius = Math.max(2, Math.round(bbox.w / 64));
  const smoothed = new Float32Array(bbox.w);
  for (let ix = 0; ix < bbox.w; ix += 1) {
    let sum = 0;
    let count = 0;
    for (let k = -radius; k <= radius; k += 1) {
      const j = ix + k;
      if (j >= 0 && j < bbox.w && depth[j] >= 0) {
        sum += depth[j];
        count += 1;
      }
    }
    smoothed[ix] = count > 0 ? sum / count : -1;
  }

  // 局所ピーク検出（毛先が下= y値が大きい方向の極大）。最小間隔 = 幅/12
  const minSeparation = Math.max(4, Math.round(bbox.w / 12));
  const peaks: number[] = [];
  for (let ix = 1; ix < bbox.w - 1; ix += 1) {
    if (smoothed[ix] < 0) continue;
    if (smoothed[ix] >= smoothed[ix - 1] && smoothed[ix] > smoothed[ix + 1]) {
      if (peaks.length > 0 && ix - peaks[peaks.length - 1] < minSeparation) {
        // 近すぎるピークは深い方を残す
        if (smoothed[ix] > smoothed[peaks[peaks.length - 1]]) peaks[peaks.length - 1] = ix;
      } else {
        peaks.push(ix);
      }
    }
  }
  // 深い順に maxStrands 個へ絞り、位置順へ戻す
  peaks.sort((a, b) => smoothed[b] - smoothed[a]);
  const selected = peaks.slice(0, maxStrands).sort((a, b) => a - b);
  if (selected.length < 2) {
    strandCache.set(image, whole);
    return whole;
  }

  // 房境界 = 隣接ピーク間で輪郭が最も浅い（yが小さい）列
  const strands: HairStrand[] = [];
  let start = 0;
  for (let i = 0; i < selected.length - 1; i += 1) {
    let valley = selected[i];
    let valleyDepth = Number.POSITIVE_INFINITY;
    for (let ix = selected[i]; ix <= selected[i + 1]; ix += 1) {
      const value = smoothed[ix] < 0 ? Number.POSITIVE_INFINITY : smoothed[ix];
      if (value < valleyDepth) {
        valleyDepth = value;
        valley = ix;
      }
    }
    strands.push({ x0: bbox.x + start, x1: bbox.x + valley });
    start = valley + 1;
  }
  strands.push({ x0: bbox.x + start, x1: bbox.x + bbox.w - 1 });

  strandCache.set(image, strands);
  return strands;
}
