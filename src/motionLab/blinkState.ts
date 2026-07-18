import type { MotionLabBlinkPhase } from "./types";

export interface MotionLabBlinkDurations {
  centerMs: number;
  closeMs: number;
  openMs: number;
  settleMs: number;
  intervalMin: number;
  intervalMax: number;
}

export interface MotionLabBlinkRuntimeState {
  blinkWait: number;
  blinkPhase: MotionLabBlinkPhase;
  blinkT: number;
}

export interface MotionLabBlinkFrame {
  phase: MotionLabBlinkPhase;
  rifeProgress: number;
  rifeOwnsEye: boolean;
  /**
   * 分離した白目・虹彩・ハイライトを重ねる割合。
   * 開眼RIFEフレーム自体は下地として残し、この上物だけを薄くすることで
   * 虹彩の色や輪郭が瞬き境界で急に切り替わるのを防ぐ。
   */
  dynamicEyeAlpha: number;
  sequenceActive: boolean;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function smoothstep01(value: number): number {
  const t = clamp01(value);
  return t * t * (3 - 2 * t);
}

function nextBlinkWait(
  durations: MotionLabBlinkDurations,
  rate: number,
  random: () => number,
): number {
  const min = Math.max(0, durations.intervalMin);
  const max = Math.max(min, durations.intervalMax);
  const unit = clamp01(random());
  return (min + unit * (max - min)) / Math.max(0.1, rate);
}

function phaseDuration(phase: MotionLabBlinkPhase, durations: MotionLabBlinkDurations): number {
  switch (phase) {
    case "centering": return Math.max(0, durations.centerMs);
    case "closing": return Math.max(0, durations.closeMs);
    case "opening": return Math.max(0, durations.openMs);
    case "settling": return Math.max(0, durations.settleMs);
    default: return 0;
  }
}

function enterNextPhase(
  state: MotionLabBlinkRuntimeState,
  durations: MotionLabBlinkDurations,
  rate: number,
  random: () => number,
) {
  switch (state.blinkPhase) {
    case "idle": state.blinkPhase = "centering"; break;
    case "centering": state.blinkPhase = "closing"; break;
    case "closing": state.blinkPhase = "opening"; break;
    case "opening": state.blinkPhase = "settling"; break;
    case "settling":
      state.blinkPhase = "idle";
      state.blinkWait = nextBlinkWait(durations, rate, random);
      break;
  }
  state.blinkT = 0;
}

/**
 * CanvasやReactに依存せず、瞬き一回分の描画所有権を進める。
 * centering中は動的な目を中央へ戻し、closingからsettlingまではRIFEへ描画を渡す。
 */
export function stepMotionLabBlink(
  state: MotionLabBlinkRuntimeState,
  deltaSeconds: number,
  enabled: boolean,
  rate: number,
  durations: MotionLabBlinkDurations,
  random: () => number = Math.random,
): MotionLabBlinkFrame {
  // 開発中のHMRで旧ランタイム（blinkPhase未保持）が残っても安全に移行する。
  if (!["idle", "centering", "closing", "opening", "settling"].includes(state.blinkPhase)) {
    state.blinkPhase = "idle";
    state.blinkT = 0;
  }
  if (!enabled) {
    state.blinkPhase = "idle";
    state.blinkT = 0;
    if (!Number.isFinite(state.blinkWait) || state.blinkWait <= 0) {
      state.blinkWait = nextBlinkWait(durations, rate, random);
    }
  } else {
    let remainingMs = Math.max(0, Number.isFinite(deltaSeconds) ? deltaSeconds * 1000 : 0);

    // 遅延フレームが複数の短いフェーズを跨いでも破綻させない。
    for (let guard = 0; guard < 12; guard += 1) {
      if (state.blinkPhase === "idle") {
        if (state.blinkWait > 0) {
          if (remainingMs <= 0) break;
          const waitMs = state.blinkWait * 1000;
          if (remainingMs < waitMs) {
            state.blinkWait = Math.max(0, state.blinkWait - remainingMs / 1000);
            remainingMs = 0;
            break;
          }
          remainingMs -= waitMs;
          state.blinkWait = 0;
        }
        enterNextPhase(state, durations, rate, random);
        if (remainingMs <= 0) break;
        continue;
      }

      const durationMs = phaseDuration(state.blinkPhase, durations);
      if (durationMs <= 0) {
        enterNextPhase(state, durations, rate, random);
        continue;
      }
      const phaseRemainingMs = Math.max(0, durationMs - state.blinkT);
      if (remainingMs < phaseRemainingMs) {
        state.blinkT += remainingMs;
        remainingMs = 0;
        break;
      }
      remainingMs -= phaseRemainingMs;
      enterNextPhase(state, durations, rate, random);
      if (remainingMs <= 0) break;
    }
  }

  let rifeProgress = 0;
  if (state.blinkPhase === "closing") {
    rifeProgress = clamp01(state.blinkT / Math.max(1, durations.closeMs));
  } else if (state.blinkPhase === "opening") {
    rifeProgress = 1 - clamp01(state.blinkT / Math.max(1, durations.openMs));
  }
  // RIFEが完全に目を所有するのは閉じ・開きの本体だけ。
  // centering終盤では分離目をフェードアウトし、settlingでは同じ開眼RIFE
  // フレームを下地に分離目をフェードバックするため、色差が段差にならない。
  let dynamicEyeAlpha = 1;
  if (state.blinkPhase === "centering") {
    const progress = clamp01(state.blinkT / Math.max(1, durations.centerMs));
    // Give the open procedural eye enough time to blend into the RIFE source.
    // A longer smooth fade hides small colour/antialiasing differences between
    // the separately rendered iris and the precomposited open-eye frame.
    const fadeProgress = (progress - 0.35) / 0.65;
    dynamicEyeAlpha = 1 - smoothstep01(fadeProgress);
  } else if (state.blinkPhase === "closing" || state.blinkPhase === "opening") {
    dynamicEyeAlpha = 0;
  } else if (state.blinkPhase === "settling") {
    const progress = clamp01(state.blinkT / Math.max(1, durations.settleMs));
    dynamicEyeAlpha = smoothstep01(progress);
  }
  const rifeOwnsEye = state.blinkPhase === "closing"
    || state.blinkPhase === "opening";
  return {
    phase: state.blinkPhase,
    rifeProgress,
    rifeOwnsEye,
    dynamicEyeAlpha,
    sequenceActive: state.blinkPhase !== "idle",
  };
}
