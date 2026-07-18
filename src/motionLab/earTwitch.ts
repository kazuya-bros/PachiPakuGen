import type { MotionLabEarTwitchMode } from "./types";

export const MOTION_LAB_EAR_TWITCH = {
  bounce: 110,
  k: 260,
  c: 13,
  maxPx: 9,
  rotKick: 1.2,
  intervalMin: 3,
  intervalRange: 6,
  doubleMin: 0.12,
  doubleRange: 0.12,
  followUpScale: 0.72,
  initialMin: 0.55,
  initialRange: 0.65,
} as const;

export interface MotionLabEarTwitchImpulse {
  bounceVelocity: number;
  rotationVelocity: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** モードごとの単発インパルス。followUpはダブルの2打目を少し弱くする。 */
export function motionLabEarTwitchImpulse(
  mode: MotionLabEarTwitchMode,
  strength: number,
  direction: number,
  followUp = false,
): MotionLabEarTwitchImpulse {
  const amount = clamp(Number.isFinite(strength) ? strength : 0, 0, 2);
  const pulseScale = followUp ? MOTION_LAB_EAR_TWITCH.followUpScale : 1;
  const signedDirection = direction < 0 ? -1 : 1;
  return {
    bounceVelocity: mode === "tilt" ? 0 : -MOTION_LAB_EAR_TWITCH.bounce * amount * pulseScale,
    rotationVelocity: mode === "bounce"
      ? 0
      : signedDirection * MOTION_LAB_EAR_TWITCH.rotKick * amount * pulseScale,
  };
}

/** ダブルの1打目だけ短い待ち時間を返す。連打は必ず2回で止まる。 */
export function motionLabNextEarTwitchWait(
  mode: MotionLabEarTwitchMode,
  queueFollowUp: boolean,
  randomUnit: number,
): number {
  const unit = clamp(Number.isFinite(randomUnit) ? randomUnit : 0, 0, 1);
  if (mode === "double" && queueFollowUp) {
    return MOTION_LAB_EAR_TWITCH.doubleMin + unit * MOTION_LAB_EAR_TWITCH.doubleRange;
  }
  return MOTION_LAB_EAR_TWITCH.intervalMin + unit * MOTION_LAB_EAR_TWITCH.intervalRange;
}

/** 再生開始・種類変更後は、比較しやすいよう最初の一回だけ早めに見せる。 */
export function motionLabInitialEarTwitchWait(randomUnit: number): number {
  const unit = clamp(Number.isFinite(randomUnit) ? randomUnit : 0, 0, 1);
  return MOTION_LAB_EAR_TWITCH.initialMin + unit * MOTION_LAB_EAR_TWITCH.initialRange;
}
