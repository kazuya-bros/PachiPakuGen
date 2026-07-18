import { useReducer } from "react";
import type {
  MotionLabEffectKey,
  MotionLabEarTwitchMode,
  MotionLabEngineFamily,
  MotionLabLayerMode,
  MotionLabManifest,
  MotionLabMethod,
  MotionLabPreset,
  MotionLabRenderSettings,
  MotionLabTimelineEvent,
} from "./types";
import {
  MOTION_LAB_ARM_DEFAULTS,
  MOTION_LAB_CHEST_DEFAULTS,
  MOTION_LAB_EFFECT_DEFAULTS,
  MOTION_LAB_EFFECT_DEFS,
  MOTION_LAB_EFFECT_SOLO_DEPS,
  MOTION_LAB_HAIR_DEFAULTS,
  MOTION_LAB_TEMPLATES,
} from "./constants";
import {
  clamp,
  validMotionLabLayerMode,
  validMotionLabMethod,
  validMotionLabPreset,
} from "./render";

/**
 * Motion Lab調整値の単一ステート。
 * 旧App.tsxのmotionLab* useState約60個を1オブジェクトへ束ねたもの。
 * reducerは必ず新オブジェクトを返す（rAF用useEffectの再購読保証）。
 */
export interface MotionLabSettings {
  method: MotionLabMethod;
  layerMode: MotionLabLayerMode;
  preset: MotionLabPreset;
  attackMs: number;
  releaseMs: number;
  crossfadeMs: number;
  restBias: number;
  shapeSmoothing: number;
  bridgeBias: number;
  breathAmplitude: number;
  bodySwayAmplitude: number;
  hairFrontDelay: number;
  hairBackDelay: number;
  hairK: number;
  hairC: number;
  hairWind: number;
  hairDrive: number;
  effects: Record<MotionLabEffectKey, boolean>;
  armMaxAngle: number;
  armSwayAmp: number;
  armPivotRatio: number;
  chestMax: number;
  hairBackScale: number;
  pyokoBounce: number;
  engineFamily: MotionLabEngineFamily;
  hairMotionStrength: number;
  glanceStrength: number;
  gazeStrength: number;
  irisBreathStrength: number;
  wetnessStrength: number;
  browStrength: number;
  blinkRate: number;
  liftStrength: number;
  earTwitchScale: number;
  earTwitchMode: MotionLabEarTwitchMode;
  templateName: string | null;
  hairEngine: "spring" | "wave";
  hairWaveStrength: number;
  pivots: Record<string, { x: number; y: number }>;
  rangesDeg: Record<string, number>;
  swingScale: Record<string, number>;
  intensity: number;
  parallaxScale: number;
  armBehindBody: boolean;
  strandsEnabled: boolean;
}

export const MOTION_LAB_DEFAULT_SETTINGS: MotionLabSettings = {
  method: "smooth",
  layerMode: "spring",
  preset: "normal",
  attackMs: 90,
  releaseMs: 160,
  crossfadeMs: 50,
  restBias: 0.25,
  shapeSmoothing: 0.65,
  bridgeBias: 0.45,
  breathAmplitude: 1,
  bodySwayAmplitude: 1,
  hairFrontDelay: 0.18,
  hairBackDelay: 0.28,
  hairK: MOTION_LAB_HAIR_DEFAULTS.k,
  hairC: MOTION_LAB_HAIR_DEFAULTS.c,
  hairWind: MOTION_LAB_HAIR_DEFAULTS.wind,
  hairDrive: MOTION_LAB_HAIR_DEFAULTS.drive,
  effects: { ...MOTION_LAB_EFFECT_DEFAULTS },
  armMaxAngle: MOTION_LAB_ARM_DEFAULTS.maxAngle,
  armSwayAmp: 1,
  armPivotRatio: 0,
  chestMax: MOTION_LAB_CHEST_DEFAULTS.max,
  hairBackScale: 0.6,
  pyokoBounce: 3,
  engineFamily: "springRig",
  hairMotionStrength: 1,
  glanceStrength: 1,
  gazeStrength: 1,
  irisBreathStrength: 0.5,
  wetnessStrength: 0.5,
  browStrength: 0.6,
  blinkRate: 1,
  liftStrength: 1,
  earTwitchScale: 1,
  earTwitchMode: "double",
  templateName: null,
  hairEngine: "spring",
  hairWaveStrength: 1,
  pivots: {},
  rangesDeg: {},
  swingScale: {},
  intensity: 1,
  parallaxScale: 1,
  armBehindBody: false,
  strandsEnabled: false,
};

export type MotionLabSettingsAction =
  | { type: "set"; patch: Partial<MotionLabSettings> }
  | { type: "setEffect"; key: MotionLabEffectKey; value: boolean }
  | { type: "soloEffect"; key: MotionLabEffectKey }
  | { type: "allEffects"; value: boolean }
  | { type: "applyEngineFamily"; family: MotionLabEngineFamily }
  | { type: "applyTemplate"; key: string }
  | { type: "applyIntensity"; value: number }
  | { type: "applyManifest"; manifest: MotionLabManifest };

function applyTemplate(state: MotionLabSettings, key: string): MotionLabSettings {
  const template = MOTION_LAB_TEMPLATES[key];
  if (!template) return state;
  return {
    ...state,
    templateName: key,
    engineFamily: template.engine,
    preset: template.preset,
    // テンプレは既存パラメータで強度を表現するため、エフェクト個別倍率は等倍へ戻す
    hairMotionStrength: 1,
    glanceStrength: 1,
    gazeStrength: 1,
    irisBreathStrength: 0.5,
    wetnessStrength: 0.5,
    browStrength: 0.6,
    blinkRate: 1,
    liftStrength: 1,
    earTwitchScale: 1,
    layerMode: template.layerMode,
    hairEngine: template.hairEngine,
    hairWaveStrength: template.hairWaveStrength,
    hairK: template.hairK,
    hairC: template.hairC,
    hairWind: template.hairWind,
    hairDrive: template.hairDrive,
    hairBackScale: template.hairBackScale,
    breathAmplitude: template.breath,
    bodySwayAmplitude: template.bodySway,
    pyokoBounce: template.pyokoBounce,
    parallaxScale: template.parallax,
    // テンプレは全エフェクトONを起点に、テンプレ固有のON/OFFだけ反映する
    effects: {
      ...MOTION_LAB_EFFECT_DEFAULTS,
      breath: true, bodySway: true, pyoko: true, hairMotion: true, hairBack: true,
      parallax: true, glance: template.randomGlance, gaze: true, blink: true,
      arm: true, lift: true, chest: true, earTwitch: template.earTwitch,
    },
    strandsEnabled: template.strands,
    armSwayAmp: template.armSwayAmp,
    armMaxAngle: template.armMaxAngle,
    chestMax: template.chestMax,
    intensity: 1,
    // 回転軸・可動域・揺れ幅の個別上書き（素材依存の調整）は維持する
  };
}

/** 現行の機能名へ正規化する。旧版が保存した個人名ベースの値は読み込み時だけ受け付ける。 */
export function normalizeMotionLabEngineFamily(value: unknown): MotionLabEngineFamily | null {
  if (value === "wave" || value === "rotejin") return "wave";
  if (value === "springRig" || value === "hachigoni") return "springRig";
  return null;
}

function applyManifest(state: MotionLabSettings, manifest: MotionLabManifest): MotionLabSettings {
  // New eye effects must not leak from the previously opened workspace when a
  // legacy manifest has no fields for them.
  const next = {
    ...state,
    templateName: null,
    intensity: 1,
    irisBreathStrength: MOTION_LAB_DEFAULT_SETTINGS.irisBreathStrength,
    wetnessStrength: MOTION_LAB_DEFAULT_SETTINGS.wetnessStrength,
    browStrength: MOTION_LAB_DEFAULT_SETTINGS.browStrength,
    effects: {
      ...state.effects,
      irisBreath: MOTION_LAB_EFFECT_DEFAULTS.irisBreath,
      wetness: MOTION_LAB_EFFECT_DEFAULTS.wetness,
      brow: MOTION_LAB_EFFECT_DEFAULTS.brow,
    },
  };
  const lip = manifest.methods?.lipTimelineSmoother;
  if (lip) {
    next.method = lip.method ? validMotionLabMethod(lip.method) : lip.enabled === false ? "baseline" : "smooth";
    if (typeof lip.attackMs === "number") next.attackMs = clamp(Math.round(lip.attackMs), 40, 180);
    if (typeof lip.releaseMs === "number") next.releaseMs = clamp(Math.round(lip.releaseMs), 80, 260);
    if (typeof lip.crossfadeMs === "number") next.crossfadeMs = clamp(Math.round(lip.crossfadeMs), 0, 120);
    if (typeof lip.restBias === "number") next.restBias = clamp(lip.restBias, 0, 1);
    if (typeof lip.shapeSmoothing === "number") next.shapeSmoothing = clamp(lip.shapeSmoothing, 0, 1);
    if (typeof lip.bridgeBias === "number") next.bridgeBias = clamp(lip.bridgeBias, 0, 0.85);
  }
  const spring = manifest.methods?.layeredSpring;
  if (spring) {
    if (spring.layerMode) next.layerMode = validMotionLabLayerMode(spring.layerMode);
    next.preset = validMotionLabPreset(spring.preset);
    if (typeof spring.breathAmplitude === "number") next.breathAmplitude = clamp(spring.breathAmplitude, 0, 1.6);
    if (typeof spring.bodySwayAmplitude === "number") next.bodySwayAmplitude = clamp(spring.bodySwayAmplitude, 0, 1.8);
    if (typeof spring.hairFrontDelay === "number") next.hairFrontDelay = clamp(spring.hairFrontDelay, 0, 0.6);
    if (typeof spring.hairBackDelay === "number") next.hairBackDelay = clamp(spring.hairBackDelay, 0, 0.8);
  }
  const physics = manifest.methods?.physicsLab;
  if (physics) {
    if (typeof physics.hairK === "number") next.hairK = clamp(physics.hairK, 10, 200);
    if (typeof physics.hairC === "number") next.hairC = clamp(physics.hairC, 1, 30);
    if (typeof physics.hairWind === "number") next.hairWind = clamp(physics.hairWind, 0, 0.06);
    if (typeof physics.hairDrive === "number") next.hairDrive = clamp(physics.hairDrive, 0, 0.2);
    if (typeof physics.armEnabled === "boolean") next.effects.arm = physics.armEnabled;
    if (typeof physics.armMaxAngle === "number") next.armMaxAngle = clamp(physics.armMaxAngle, 0, 0.6);
    if (typeof physics.armSwayAmp === "number") next.armSwayAmp = clamp(physics.armSwayAmp, 0, 3);
    if (typeof physics.armPivotRatio === "number") next.armPivotRatio = clamp(physics.armPivotRatio, 0, 0.6);
    if (typeof physics.liftEnabled === "boolean") next.effects.lift = physics.liftEnabled;
    if (typeof physics.chestMax === "number") next.chestMax = clamp(physics.chestMax, 0, 8);
    if (typeof physics.hairBackScale === "number") next.hairBackScale = clamp(physics.hairBackScale, 0, 1.5);
    if (typeof physics.earTwitch === "boolean") next.effects.earTwitch = physics.earTwitch;
    if (physics.earTwitchMode === "bounce" || physics.earTwitchMode === "tilt" || physics.earTwitchMode === "double") {
      next.earTwitchMode = physics.earTwitchMode;
    }
    if (physics.hairEngine === "spring" || physics.hairEngine === "wave") next.hairEngine = physics.hairEngine;
    if (typeof physics.pyokoBounce === "number") next.pyokoBounce = clamp(physics.pyokoBounce, 0, 12);
    if (typeof physics.randomGlance === "boolean") next.effects.glance = physics.randomGlance;
    const engineFamily = normalizeMotionLabEngineFamily(physics.engineFamily);
    if (engineFamily) {
      next.engineFamily = engineFamily;
      // 初期のマニフェストは方式だけを保存していたため、個別指定が無ければ髪方式も補完する。
      if (physics.hairEngine !== "spring" && physics.hairEngine !== "wave") {
        next.hairEngine = engineFamily === "wave" ? "wave" : "spring";
      }
    }
    if (typeof physics.hairMotionStrength === "number") next.hairMotionStrength = clamp(physics.hairMotionStrength, 0, 2);
    if (typeof physics.glanceStrength === "number") next.glanceStrength = clamp(physics.glanceStrength, 0, 2);
    if (typeof physics.gazeStrength === "number") next.gazeStrength = clamp(physics.gazeStrength, 0, 3);
    if (typeof physics.irisBreathStrength === "number") next.irisBreathStrength = clamp(physics.irisBreathStrength, 0, 1);
    if (typeof physics.wetnessStrength === "number") next.wetnessStrength = clamp(physics.wetnessStrength, 0, 1);
    if (typeof physics.browStrength === "number") next.browStrength = clamp(physics.browStrength, 0, 1.5);
    if (typeof physics.blinkRate === "number") next.blinkRate = clamp(physics.blinkRate, 0.3, 2.5);
    if (typeof physics.liftStrength === "number") next.liftStrength = clamp(physics.liftStrength, 0, 2);
    if (typeof physics.earTwitchScale === "number") next.earTwitchScale = clamp(physics.earTwitchScale, 0, 2);
    // v3: エフェクト一括ON/OFF（旧フィールドより後に適用して優先させる）
    if (physics.effects && typeof physics.effects === "object") {
      for (const def of MOTION_LAB_EFFECT_DEFS) {
        const value = physics.effects[def.key];
        if (typeof value === "boolean") next.effects[def.key] = value;
      }
    }
    if (typeof physics.hairWaveStrength === "number") next.hairWaveStrength = clamp(physics.hairWaveStrength, 0, 2);
    if (physics.pivots && typeof physics.pivots === "object") {
      const pivots: Record<string, { x: number; y: number }> = {};
      for (const [part, value] of Object.entries(physics.pivots)) {
        if (typeof value?.x === "number" && typeof value?.y === "number") {
          pivots[part] = { x: value.x, y: value.y };
        }
      }
      next.pivots = pivots;
    }
    if (physics.rangesDeg && typeof physics.rangesDeg === "object") {
      const ranges: Record<string, number> = {};
      for (const [part, value] of Object.entries(physics.rangesDeg)) {
        if (typeof value === "number") ranges[part] = clamp(value, 0, 90);
      }
      next.rangesDeg = ranges;
    }
    if (physics.swingScale && typeof physics.swingScale === "object") {
      const scales: Record<string, number> = {};
      for (const [part, value] of Object.entries(physics.swingScale)) {
        if (typeof value === "number") scales[part] = clamp(value, 0, 3);
      }
      next.swingScale = scales;
    }
    if (typeof physics.parallaxScale === "number") next.parallaxScale = clamp(physics.parallaxScale, 0, 1.5);
    if (typeof physics.strandsEnabled === "boolean") next.strandsEnabled = physics.strandsEnabled;
    if (typeof physics.armBehindBody === "boolean") next.armBehindBody = physics.armBehindBody;
  }
  return next;
}

export function motionLabSettingsReducer(
  state: MotionLabSettings,
  action: MotionLabSettingsAction,
): MotionLabSettings {
  switch (action.type) {
    case "set":
      return { ...state, ...action.patch };
    case "setEffect":
      return { ...state, effects: { ...state.effects, [action.key]: action.value } };
    case "soloEffect": {
      const effects = Object.fromEntries(
        MOTION_LAB_EFFECT_DEFS.map(def => [def.key, def.key === action.key]),
      ) as Record<MotionLabEffectKey, boolean>;
      for (const dep of MOTION_LAB_EFFECT_SOLO_DEPS[action.key] ?? []) effects[dep] = true;
      return { ...state, effects };
    }
    case "allEffects":
      return {
        ...state,
        effects: Object.fromEntries(
          MOTION_LAB_EFFECT_DEFS.map(def => [def.key, action.value]),
        ) as Record<MotionLabEffectKey, boolean>,
      };
    case "applyEngineFamily":
      return {
        ...state,
        engineFamily: action.family,
        hairEngine: action.family === "wave" ? "wave" : "spring",
        templateName:
          state.templateName && MOTION_LAB_TEMPLATES[state.templateName]?.engine === action.family
            ? state.templateName
            : null,
      };
    case "applyTemplate":
      return applyTemplate(state, action.key);
    case "applyIntensity": {
      const scale = clamp(Number.isFinite(action.value) ? action.value : 1, 0.5, 1.5);
      const template = state.templateName ? MOTION_LAB_TEMPLATES[state.templateName] : null;
      return {
        ...state,
        intensity: scale,
        breathAmplitude: clamp(Number(((template?.breath ?? 1) * scale).toFixed(3)), 0, 1.6),
        bodySwayAmplitude: clamp(Number(((template?.bodySway ?? 1) * scale).toFixed(3)), 0, 1.8),
        hairWind: clamp(Number(((template?.hairWind ?? MOTION_LAB_HAIR_DEFAULTS.wind) * scale).toFixed(4)), 0, 0.06),
        hairDrive: clamp(Number(((template?.hairDrive ?? MOTION_LAB_HAIR_DEFAULTS.drive) * scale).toFixed(3)), 0, 0.2),
        hairWaveStrength: clamp(Number(((template?.hairWaveStrength ?? 1) * scale).toFixed(3)), 0, 2),
        pyokoBounce: clamp(Number(((template?.pyokoBounce ?? 3) * scale).toFixed(2)), 0, 7),
        armMaxAngle: clamp(Number(((template?.armMaxAngle ?? MOTION_LAB_ARM_DEFAULTS.maxAngle) * scale).toFixed(3)), 0, 0.3),
        chestMax: clamp(Number(((template?.chestMax ?? MOTION_LAB_CHEST_DEFAULTS.max) * scale).toFixed(1)), 0, 8),
        parallaxScale: clamp(Number(((template?.parallax ?? 1) * scale).toFixed(3)), 0, 1.5),
      };
    }
    case "applyManifest":
      return applyManifest(state, action.manifest);
    default:
      return state;
  }
}

export function useMotionLabSettings() {
  return useReducer(motionLabSettingsReducer, MOTION_LAB_DEFAULT_SETTINGS);
}

/**
 * MotionLabSettings（永続値）→ drawMotionLabScene用のMotionLabRenderSettings（実行時値）。
 * エフェクトON/OFFのゲーティング（旧candidateレーンの組み立て）をここへ集約。
 */
export function toRenderSettings(
  settings: MotionLabSettings,
  extras: {
    pivotEditPart: string | null;
    timeline?: MotionLabTimelineEvent[];
    timelineDurationMs?: number;
    liveInput?: MotionLabRenderSettings["liveInput"];
  },
): MotionLabRenderSettings {
  const fx = settings.effects;
  return {
    mouthMethod: settings.method,
    layerMode: settings.layerMode,
    preset: settings.preset,
    attackMs: settings.attackMs,
    releaseMs: settings.releaseMs,
    crossfadeMs: settings.crossfadeMs,
    restBias: settings.restBias,
    shapeSmoothing: settings.shapeSmoothing,
    bridgeBias: settings.bridgeBias,
    breathAmplitude: fx.breath ? settings.breathAmplitude : 0,
    bodySwayAmplitude: fx.bodySway ? settings.bodySwayAmplitude : 0,
    hairK: settings.hairK,
    hairC: settings.hairC,
    hairWind: settings.hairWind,
    hairDrive: settings.hairDrive,
    // 肩の弾みだけONの場合も物理更新は必要（liftはarm経由で計算される）
    armEnabled: fx.arm || fx.lift,
    armMaxAngle: settings.armMaxAngle,
    armSwayAmp: fx.arm ? settings.armSwayAmp : 0,
    armPivotRatio: settings.armPivotRatio,
    liftEnabled: fx.lift,
    chestMax: fx.chest ? settings.chestMax : 0,
    hairBackScale: fx.hairBack ? settings.hairBackScale : 0,
    hairWaveMode: settings.hairEngine === "wave",
    hairWaveStrength: settings.hairWaveStrength,
    earTwitch: fx.earTwitch,
    earTwitchMode: settings.earTwitchMode,
    pyokoBounce: fx.pyoko ? settings.pyokoBounce : 0,
    randomGlance: fx.glance,
    hairMotionEnabled: fx.hairMotion,
    gazeEnabled: fx.gaze,
    irisBreathEnabled: fx.irisBreath,
    wetnessEnabled: fx.wetness,
    browEnabled: fx.brow,
    blinkEnabled: fx.blink,
    hairMotionStrength: settings.hairMotionStrength,
    glanceStrength: settings.glanceStrength,
    gazeStrength: settings.gazeStrength,
    irisBreathStrength: settings.irisBreathStrength,
    wetnessStrength: settings.wetnessStrength,
    browStrength: settings.browStrength,
    blinkRate: settings.blinkRate,
    liftStrength: settings.liftStrength,
    earTwitchScale: settings.earTwitchScale,
    parallaxScale: fx.parallax ? settings.parallaxScale : 0,
    pivots: settings.pivots,
    rangesDeg: settings.rangesDeg,
    swingScale: settings.swingScale,
    pivotEditPart: extras.pivotEditPart,
    strandsEnabled: settings.strandsEnabled,
    armBehindBody: settings.armBehindBody,
    timeline: extras.timeline,
    timelineDurationMs: extras.timelineDurationMs,
    liveInput: extras.liveInput,
  };
}
