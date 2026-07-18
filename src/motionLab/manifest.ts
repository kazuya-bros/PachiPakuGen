import type {
  MotionLabManifest,
  MotionLabTimelineEvent,
  SpritalkMotionProfile,
} from "./types";
import {
  MOTION_LAB_ARM_DEFAULTS,
  MOTION_LAB_BLINK_DEFAULTS,
  MOTION_LAB_CHEST_DEFAULTS,
  MOTION_LAB_DEFAULT_REVIEW_SCORES,
  MOTION_LAB_DEPTH_DEFAULTS,
  MOTION_LAB_EAR_TWITCH,
  MOTION_LAB_GAZE_DEFAULTS,
  MOTION_LAB_HAIR_SEGMENTS,
  MOTION_LAB_HIGHLIGHT_DEFAULTS,
  MOTION_LAB_PARALLAX_DEFAULTS,
  MOTION_LAB_PRESENCE_DEFAULTS,
  MOTION_LAB_PRESET_FACTORS,
  MOTION_LAB_SWAY_DEFAULTS,
} from "./constants";
import { clamp } from "./render";
import {
  MOTION_LAB_BROW_MAX_LIFT_PX,
  MOTION_LAB_BROW_MAX_TILT_DEG,
  MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA,
  MOTION_LAB_WETNESS_MAX_ALPHA,
  MOTION_LAB_WETNESS_MIN_ALPHA,
} from "./eyeEffects";
import type { MotionLabSettings } from "./useMotionLabSettings";

export type MotionLabSequenceDefinition =
  | { type: "builtInVowelTest" }
  | {
    type: "text";
    text: string;
    durationMs: number;
    events: MotionLabTimelineEvent[];
  };

export const BUILT_IN_MOTION_SEQUENCE: MotionLabSequenceDefinition = {
  type: "builtInVowelTest",
};

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record)
    .sort()
    .map(key => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
    .join(",")}}`;
}

function fingerprint(value: unknown): string {
  const source = stableStringify(value);
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < source.length; index += 1) {
    const code = source.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193);
    second = Math.imul(second ^ code, 0x85ebca6b);
  }
  const hex = (value: number) => (value >>> 0).toString(16).padStart(8, "0");
  return `motion-v1-${hex(first)}${hex(second)}-${source.length.toString(16)}`;
}

export function buildMotionContentFingerprints(
  settings: MotionLabSettings,
) {
  const rendererRevision = "motion-render.v5";
  const spritalk = fingerprint({ contract: "motionProfile.v3", rendererRevision, settings });
  return { spritalk };
}

/**
 * 設定 → motion-preview-manifest.json（schema pachipakugen.motionPreview.v1）。
 * reviewブロックは互換維持のため固定既定値で出力を継続する（評価UIは製品版で廃止）。
 */
export function buildMotionLabManifest(
  settings: MotionLabSettings,
  sourceDir: string,
  sequence: MotionLabSequenceDefinition = BUILT_IN_MOTION_SEQUENCE,
): MotionLabManifest {
  const contentFingerprints = buildMotionContentFingerprints(settings);
  return {
    schema: "pachipakugen.motionPreview.v1",
    sourcePartsDir: sourceDir,
    createdAt: new Date().toISOString(),
    contentFingerprints,
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
        earTwitchMode: settings.earTwitchMode,
        pyokoBounce: settings.pyokoBounce,
        randomGlance: settings.effects.glance,
        effects: settings.effects,
        engineFamily: settings.engineFamily,
        hairMotionStrength: settings.hairMotionStrength,
        glanceStrength: settings.glanceStrength,
        gazeStrength: settings.gazeStrength,
        irisBreathStrength: settings.irisBreathStrength,
        wetnessStrength: settings.wetnessStrength,
        browStrength: settings.browStrength,
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
    timeline: sequence,
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
  _sourceDir: string,
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
    // 素材と同じフォルダに置く自己完結プロファイル。絶対パスを埋め込まず、
    // フォルダごと移動しても選択したフォルダを基準に解決できるようにする。
    sourcePartsDir: ".",
    createdAt: new Date().toISOString(),
    generatedBy: "PachiPakuGen Motion Lab",
    contentFingerprint: buildMotionContentFingerprints(settings).spritalk,
    blink: {
      mode: "keepExisting",
      coordination: {
        strategy: "centerThenRife",
        centerMs: MOTION_LAB_BLINK_DEFAULTS.centerMs,
        closeMs: MOTION_LAB_BLINK_DEFAULTS.closeMs,
        openMs: MOTION_LAB_BLINK_DEFAULTS.openMs,
        settleMs: MOTION_LAB_BLINK_DEFAULTS.settleMs,
        suppressGazeDuringRife: true,
        suppressHighlightDuringRife: true,
        resumeDynamicEyeAfterSettle: true,
      },
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
        partOverrides: {
          pivots: settings.pivots,
          rangesDeg: settings.rangesDeg,
          swingScale: settings.swingScale,
        },
      },
      earTwitch: {
        enabled: settings.effects.earTwitch,
        mode: settings.earTwitchMode,
        strength: settings.effects.earTwitch ? settings.earTwitchScale : 0,
        bounceVelocity: MOTION_LAB_EAR_TWITCH.bounce,
        rotationVelocity: MOTION_LAB_EAR_TWITCH.rotKick,
        maxOffsetPx: MOTION_LAB_EAR_TWITCH.maxPx,
        intervalMin: MOTION_LAB_EAR_TWITCH.intervalMin,
        intervalMax: MOTION_LAB_EAR_TWITCH.intervalMin + MOTION_LAB_EAR_TWITCH.intervalRange,
        followUpScale: MOTION_LAB_EAR_TWITCH.followUpScale,
        followUpIntervalMin: MOTION_LAB_EAR_TWITCH.doubleMin,
        followUpIntervalMax: MOTION_LAB_EAR_TWITCH.doubleMin + MOTION_LAB_EAR_TWITCH.doubleRange,
      },
      parallax: {
        shiftRatio: MOTION_LAB_PARALLAX_DEFAULTS.shiftRatio,
        shearMax: MOTION_LAB_PARALLAX_DEFAULTS.shearMax,
        scale: settings.parallaxScale,
      },
      gaze: {
        enabled: settings.effects.gaze,
        strength: settings.effects.gaze ? clamp(settings.gazeStrength, 0, 3) : 0,
        rangeRatio: 0,
        maxRangePx: settings.effects.gaze
          ? Math.min(MOTION_LAB_GAZE_DEFAULTS.maxRangePx, 1.6 * clamp(settings.gazeStrength, 0, 3))
          : 0,
        periodSeconds: MOTION_LAB_GAZE_DEFAULTS.periodSeconds,
        returnToFrontOnSpeech: false,
      },
      highlight: {
        driftPx: MOTION_LAB_HIGHLIGHT_DEFAULTS.driftPx,
      },
      irisBreath: {
        maxScaleDelta: MOTION_LAB_IRIS_BREATH_MAX_SCALE_DELTA,
        strength: settings.effects.irisBreath ? clamp(settings.irisBreathStrength, 0, 1) : 0,
        responseCurve: "quadratic",
        periodMs: 5200,
      },
      wetness: {
        maxAlpha: MOTION_LAB_WETNESS_MAX_ALPHA,
        minAlpha: MOTION_LAB_WETNESS_MIN_ALPHA,
        strength: settings.effects.wetness ? clamp(settings.wetnessStrength, 0, 1) : 0,
        responseCurve: "power2.2",
        periodMs: 4600,
      },
      brow: {
        enabled: settings.effects.brow,
        strength: settings.effects.brow ? clamp(settings.browStrength, 0, 1.5) : 0,
        maxLiftPx: MOTION_LAB_BROW_MAX_LIFT_PX,
        maxTiltDeg: MOTION_LAB_BROW_MAX_TILT_DEG,
        response: "smoothedVoiceLiftAndAsymmetricTilt",
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
