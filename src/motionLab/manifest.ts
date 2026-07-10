import type { MotionLabManifest, SpritalkMotionProfile } from "./types";
import {
  MOTION_LAB_ARM_DEFAULTS,
  MOTION_LAB_CHEST_DEFAULTS,
  MOTION_LAB_DEFAULT_REVIEW_SCORES,
  MOTION_LAB_DEPTH_DEFAULTS,
  MOTION_LAB_GAZE_DEFAULTS,
  MOTION_LAB_HAIR_SEGMENTS,
  MOTION_LAB_HIGHLIGHT_DEFAULTS,
  MOTION_LAB_PARALLAX_DEFAULTS,
  MOTION_LAB_PRESENCE_DEFAULTS,
  MOTION_LAB_PRESET_FACTORS,
  MOTION_LAB_SWAY_DEFAULTS,
} from "./constants";
import { clamp } from "./render";
import type { MotionLabSettings } from "./useMotionLabSettings";

/**
 * 設定 → motion-preview-manifest.json（schema pachipakugen.motionPreview.v1）。
 * reviewブロックは互換維持のため固定既定値で出力を継続する（評価UIは製品版で廃止）。
 */
export function buildMotionLabManifest(
  settings: MotionLabSettings,
  sourceDir: string,
): MotionLabManifest {
  return {
    schema: "pachipakugen.motionPreview.v1",
    sourcePartsDir: sourceDir,
    createdAt: new Date().toISOString(),
    methods: {
      baseline: { enabled: true },
      lipTimelineSmoother: {
        enabled: settings.method !== "baseline",
        method: settings.method,
        attackMs: settings.attackMs,
        releaseMs: settings.releaseMs,
        crossfadeMs: settings.crossfadeMs,
        restBias: settings.restBias,
        shapeSmoothing: settings.shapeSmoothing,
        bridgeBias: settings.bridgeBias,
      },
      layeredSpring: {
        enabled: true,
        layerMode: settings.layerMode,
        preset: settings.preset,
        breathAmplitude: settings.breathAmplitude,
        bodySwayAmplitude: settings.bodySwayAmplitude,
        hairFrontDelay: settings.hairFrontDelay,
        hairBackDelay: settings.hairBackDelay,
      },
      physicsLab: {
        hairK: settings.hairK,
        hairC: settings.hairC,
        hairWind: settings.hairWind,
        hairDrive: settings.hairDrive,
        armEnabled: settings.effects.arm,
        armMaxAngle: settings.armMaxAngle,
        armSwayAmp: settings.armSwayAmp,
        armPivotRatio: settings.armPivotRatio,
        liftEnabled: settings.effects.lift,
        chestMax: settings.chestMax,
        hairBackScale: settings.hairBackScale,
        hairEngine: settings.hairEngine,
        hairWaveStrength: settings.hairWaveStrength,
        earTwitch: settings.effects.earTwitch,
        pyokoBounce: settings.pyokoBounce,
        randomGlance: settings.effects.glance,
        effects: settings.effects,
        engineFamily: settings.engineFamily,
        hairMotionStrength: settings.hairMotionStrength,
        glanceStrength: settings.glanceStrength,
        gazeStrength: settings.gazeStrength,
        blinkRate: settings.blinkRate,
        liftStrength: settings.liftStrength,
        earTwitchScale: settings.earTwitchScale,
        pivots: settings.pivots,
        rangesDeg: settings.rangesDeg,
        swingScale: settings.swingScale,
        parallaxScale: settings.parallaxScale,
        strandsEnabled: settings.strandsEnabled,
        armBehindBody: settings.armBehindBody,
      },
    },
    timeline: { type: "builtInVowelTest" },
    review: {
      verdict: "undecided",
      note: "",
      scores: { ...MOTION_LAB_DEFAULT_REVIEW_SCORES },
    },
  };
}

/**
 * 設定 → spritalk-motion-profile.json（schema spritalk.motionProfile.v2）。
 * SpriTalk側が読み込むアニメーション設定JSONの移植契約。
 */
export function buildSpritalkMotionProfile(
  settings: MotionLabSettings,
  sourceDir: string,
): SpritalkMotionProfile {
  const preset = MOTION_LAB_PRESET_FACTORS[settings.preset];
  const lipSyncRenderer = settings.method === "baseline"
    ? "directLayerSwitch"
    : settings.method === "bridge"
      ? "neutralBridgeOpacityBlend"
      : "smoothedFrameStepper";
  const layerRenderer = settings.layerMode === "mesh" ? "stripWarpExtension" : "existingProceduralAnimator";
  return {
    schema: "spritalk.motionProfile.v2",
    sourcePartsDir: sourceDir,
    createdAt: new Date().toISOString(),
    generatedBy: "PachiPakuGen Motion Lab",
    blink: {
      mode: "keepExisting",
    },
    lipSync: {
      method: settings.method,
      attackMs: settings.attackMs,
      releaseMs: settings.releaseMs,
      crossfadeMs: settings.crossfadeMs,
      restBias: settings.restBias,
      shapeSmoothing: settings.shapeSmoothing,
      bridgeBias: settings.bridgeBias,
    },
    layerMotion: {
      mode: settings.layerMode,
      preset: settings.preset,
      breathAmplitude: settings.breathAmplitude,
      bodySwayAmplitude: settings.bodySwayAmplitude,
      hairFrontDelayMs: Math.round(settings.hairFrontDelay * 1000),
      hairBackDelayMs: Math.round(settings.hairBackDelay * 1000),
    },
    spritalkProceduralAnimation: {
      breathing: {
        enabled: settings.breathAmplitude > 0,
        amplitude: Number((4.5 * settings.breathAmplitude * preset.breath).toFixed(2)),
        speed: 0.5,
      },
      idleSway: {
        enabled: settings.bodySwayAmplitude > 0,
        amplitudeX: Number((2.4 * settings.bodySwayAmplitude * preset.body).toFixed(2)),
        amplitudeY: Number((1.4 * settings.bodySwayAmplitude * preset.body).toFixed(2)),
        speed: 0.9,
        reduceOnSpeech: true,
      },
      hairSway: {
        enabled: settings.layerMode !== "simple",
        amplitude: Number((2.5 * settings.bodySwayAmplitude * preset.hair).toFixed(2)),
        speed: Number(clamp(0.95 - settings.hairFrontDelay, 0.3, 1.2).toFixed(2)),
        rotationAmount: Number((0.009 * preset.hair).toFixed(4)),
      },
      hairBackSway: {
        enabled: settings.layerMode !== "simple",
        amplitude: Number((2.1 * settings.bodySwayAmplitude * preset.hair).toFixed(2)),
        speed: Number(clamp(0.82 - settings.hairBackDelay, 0.25, 1.0).toFixed(2)),
        rotationAmount: Number((0.007 * preset.hair).toFixed(4)),
      },
    },
    // ===== v2 additive フィールド（docs/motion-lab-integration.md §1） =====
    physics: {
      hair: {
        mode: "chain",
        segments: MOTION_LAB_HAIR_SEGMENTS,
        k: settings.hairK,
        c: settings.hairC,
        wind: settings.hairWind,
        drive: settings.hairDrive,
        strands: settings.strandsEnabled,
      },
      arm: {
        enabled: settings.effects.arm,
        k: MOTION_LAB_ARM_DEFAULTS.k,
        c: MOTION_LAB_ARM_DEFAULTS.c,
        maxAngle: settings.armMaxAngle,
        coupling: MOTION_LAB_ARM_DEFAULTS.coupling,
        noise: MOTION_LAB_ARM_DEFAULTS.noise,
        lift: {
          enabled: settings.effects.lift,
          coupling: MOTION_LAB_ARM_DEFAULTS.lift.coupling,
          bounce: MOTION_LAB_ARM_DEFAULTS.lift.bounce,
          max: MOTION_LAB_ARM_DEFAULTS.lift.max,
        },
      },
      chest: {
        k: MOTION_LAB_CHEST_DEFAULTS.k,
        c: MOTION_LAB_CHEST_DEFAULTS.c,
        max: settings.chestMax,
      },
      sway: {
        k: MOTION_LAB_SWAY_DEFAULTS.k,
        c: MOTION_LAB_SWAY_DEFAULTS.c,
      },
      parallax: {
        shiftRatio: MOTION_LAB_PARALLAX_DEFAULTS.shiftRatio,
        shearMax: MOTION_LAB_PARALLAX_DEFAULTS.shearMax,
        scale: settings.parallaxScale,
      },
      gaze: {
        rangeRatio: MOTION_LAB_GAZE_DEFAULTS.rangeRatio,
        returnToFrontOnSpeech: true,
      },
      highlight: {
        driftPx: MOTION_LAB_HIGHLIGHT_DEFAULTS.driftPx,
      },
    },
    presence: { ...MOTION_LAB_PRESENCE_DEFAULTS },
    depth: { ...MOTION_LAB_DEPTH_DEFAULTS },
    motionScale: 1.0,
    bounceScale: 1.0,
    runtimeRequirements: {
      lipSyncRenderer,
      layerRenderer,
    },
    review: {
      verdict: "undecided",
      note: "",
      scores: { ...MOTION_LAB_DEFAULT_REVIEW_SCORES },
    },
  };
}
