import type { ArmSwayState, ChainState, SpringState } from "../motionLabPhysics";

export type MotionLabMouthKey = "closed" | "a" | "i" | "u" | "e" | "o";
export type MotionLabMethod = "baseline" | "smooth" | "bridge";
export type MotionLabLayerMode = "simple" | "spring" | "mesh";
export type MotionLabPreset = "calm" | "normal" | "lively";
export type MotionLabVerdict = "undecided" | "promising" | "hold" | "reject";
export type MotionLabReviewKey =
  | "mouthSmoothness"
  | "vowelReadability"
  | "bodyNaturalness"
  | "hairBodySeparation"
  | "settingSimplicity"
  | "migrationConfidence";

export interface MotionLabPartsResult {
  sourceDir: string;
  width: number;
  height: number;
  body: string;
  hair: string | null;
  hairBack: string | null;
  armL: string | null;
  armR: string | null;
  chest: string | null;
  sways: Record<string, string>;
  eyewhite: string | null;
  irides: string | null;
  highlight: string | null;
  eyeFrames: string[];
  mouths: Partial<Record<MotionLabMouthKey, string[]>>;
  /** layer-order.json 由来のグループ描画順（背面→前面）。無ければ空 */
  layerOrder: string[];
  missing: string[];
  warnings: string[];
}

export interface MotionLabImageSet {
  body: HTMLImageElement;
  hair: HTMLImageElement | null;
  hairBack: HTMLImageElement | null;
  armL: HTMLImageElement | null;
  armR: HTMLImageElement | null;
  chest: HTMLImageElement | null;
  sways: Record<string, HTMLImageElement>;
  /** 視線ドリフト用: 白目=クリップ領域、虹彩=ドリフト対象（§8.4） */
  eyewhite: HTMLImageElement | null;
  irides: HTMLImageElement | null;
  highlight: HTMLImageElement | null;
  /** eye/ 連番（frame 0=開き → 最終=閉じ。eyes-open→eyes-closedのRIFE出力順） */
  eyeFrames: HTMLImageElement[];
  mouths: Partial<Record<MotionLabMouthKey, HTMLImageElement[]>>;
}

/** バネ-ダンパー物理の実行時状態（レーンごとに独立保持） */
export interface MotionLabPhysicsState {
  breathPhase: number;
  swayPhase: number;
  prevRootX: number;
  prevRootY: number;
  rootVX: number;
  rootVY: number;
  noiseT: number;
  /** B3: 前髪の角度チェーン（後ろ髪は中間角×0.8で追従 = 参照実装どおり） */
  hairChain: ChainState;
  /** B1: spring モード用の遅延追従バネ */
  hairSpring: SpringState;
  hairBackSpring: SpringState;
  arm: ArmSwayState;
  chest: SpringState;
  sways: Map<string, ChainState>;
  /** 獣耳ピコピコ（sway_ear*）: 次ツイッチまでの残り秒＋縦跳ねバネ */
  earTwitches: Map<string, { wait: number; spring: SpringState }>;
  /** 後ろ髪の房ごとチェーン（前髪用は strandChains） */
  strandChainsBack: ChainState[];
  /** A4エンベロープ出力（A1のSmoothDamp前段） */
  envOpen: number;
  mouthVel: { v: number };
  speaking: boolean;
  /** 自動瞬き: 次の瞬きまでの残り秒 */
  blinkWait: number;
  /** 瞬き進行ms（-1=待機中） */
  blinkT: number;
  /** パララックス首振り: ヘッドターンのノイズ位相（§8.3） */
  headTurnT: number;
  /** 発話頷きバネ（px） */
  nod: SpringState;
  /** 視線ドリフト（px）＋SmoothDamp速度（§8.4） */
  gaze: { x: number; y: number };
  gazeVelX: { v: number };
  gazeVelY: { v: number };
  gazeT: number;
  /** ハイライトドリフトのノイズ位相 */
  highlightT: number;
  /** 房ごと髪物理: 房別チェーン（房分割OFF時は未使用） */
  strandChains: ChainState[];
  /** 房のsoftバネ（852話式 stiff+soft 二重バネ混合の柔らかい側） */
  strandChainsSoft: ChainState[];
  strandChainsBackSoft: ChainState[];
  /** 発話ぴょこバウンス（PuruPuru pyoko参考。voice→バネで縦に弾む、px） */
  pyoko: SpringState;
  /** ランダムグランス（852話 auto.rand参考）: 次の目標変更までの残り秒と現在バイアス */
  glanceWait: number;
  glanceHead: number;
  glanceHeadTarget: number;
  glanceHeadVel: { v: number };
  /** 視線グランスの目標オフセット（px。gaze側のSmoothDampが平滑化する） */
  glanceGaze: { x: number; y: number };
}

export interface MotionLabMouthRuntime {
  openY: number;
  activeTarget: MotionLabMouthKey;
  previousTarget: MotionLabMouthKey;
  transitionStartMs: number;
  lastMs: number;
  physics: MotionLabPhysicsState;
  /** 瞳クリップ合成用スクラッチキャンバス（レーン毎に保持） */
  gazeScratch?: HTMLCanvasElement;
}

export interface MotionLabLayerTransform {
  x: number;
  y: number;
  rotationDeg: number;
  scaleX: number;
  scaleY: number;
  /** パララックス首振りのシアー（§8.3: headTurn×depth×SHEAR_MAX） */
  skewX?: number;
}

export interface MotionLabChainWarpOptions {
  rootYRatio: number;
  angles: ArrayLike<number>;
  stripCount?: number;
  alpha?: number;
  /** 房ごと髪物理: この房のx範囲だけを描画（画像座標px） */
  xRange?: { x0: number; x1: number };
  /** 変位の倍率（揺れ幅スライダー用。1=既定） */
  offsetScale?: number;
}

export interface MotionLabRenderSettings {
  mouthMethod: MotionLabMethod;
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
  hairK: number;
  hairC: number;
  hairWind: number;
  hairDrive: number;
  armEnabled: boolean;
  armMaxAngle: number;
  /** 左右方向の腕揺れ幅倍率（1=既定。駆動ノイズ・体速度カップリングに乗算） */
  armSwayAmp: number;
  /** 腕の回転軸Y位置（0=不透明bbox上端〜1=下端。肩の位置補正用） */
  armPivotRatio: number;
  liftEnabled: boolean;
  chestMax: number;
  /** 後ろ髪の揺れ倍率（1=従来。揺れすぎ調整用） */
  hairBackScale: number;
  /** 髪の揺れ方式: false=バネ物理（B1/B3）、true=波揺れ（ろてじん式進行波） */
  hairWaveMode: boolean;
  /** 波揺れの強さ倍率（1=既定） */
  hairWaveStrength: number;
  /** sway_ear* パーツの獣耳ピコピコ揺れ */
  earTwitch: boolean;
  /** 発話ぴょこバウンス振幅（px。PuruPuru pyoko参考、バネ平滑） */
  pyokoBounce: number;
  /** ランダムグランス（852話 auto.rand参考）: 数秒ごとに顔向き・視線がふっと変わる */
  randomGlance: boolean;
  /** 髪の揺れ全般（バネ/波/房）。false=髪は頭に追従するだけ */
  hairMotionEnabled: boolean;
  /** 視線ドリフト（eyewhite/irides素材時）。false=瞳は正面固定 */
  gazeEnabled: boolean;
  /** 自動まばたき */
  blinkEnabled: boolean;
  /** エフェクト個別の強さ倍率（1=既定） */
  hairMotionStrength: number;
  glanceStrength: number;
  gazeStrength: number;
  /** まばたき頻度倍率（1=既定、2=2倍の頻度） */
  blinkRate: number;
  liftStrength: number;
  earTwitchScale: number;
  /** パーツごとの回転軸上書き（画像座標px）。未指定パーツは自動推定 */
  pivots: Record<string, { x: number; y: number }>;
  /** パーツごとの可動域（±度）。0または未指定=既定クランプのまま */
  rangesDeg: Record<string, number>;
  /** パーツごとの揺れ幅倍率（1=既定）。腕=振れ角、髪=ワープ変位/回転に乗算 */
  swingScale: Record<string, number>;
  /** 回転軸エディタで編集中のパーツ（マーカー描画用）。null=非表示 */
  pivotEditPart: string | null;
  /** パララックス首振りの強さ（0=無効、1=既定。§8.3） */
  parallaxScale: number;
  /** B3を房ごとチェーンに分割（852話式・§8.1 #5） */
  strandsEnabled: boolean;
  /** 腕を体の後ろに描く（素体で腕をbody背面に置いた素材向け。layer-order.json があればそちら優先） */
  armBehindBody: boolean;
  /** 口パクタイムライン（未指定なら内蔵の「あいうえお」テスト） */
  timeline?: MotionLabTimelineEvent[];
  timelineDurationMs?: number;
}

export interface MotionLabManifest {
  schema?: string;
  sourcePartsDir?: string;
  createdAt?: string;
  methods?: {
    baseline?: {
      enabled?: boolean;
    };
    lipTimelineSmoother?: {
      enabled?: boolean;
      method?: MotionLabMethod;
      attackMs?: number;
      releaseMs?: number;
      crossfadeMs?: number;
      restBias?: number;
      shapeSmoothing?: number;
      bridgeBias?: number;
    };
    layeredSpring?: {
      enabled?: boolean;
      layerMode?: MotionLabLayerMode;
      preset?: MotionLabPreset;
      breathAmplitude?: number;
      bodySwayAmplitude?: number;
      hairFrontDelay?: number;
      hairBackDelay?: number;
    };
    // v1に対するadditive追加: バネ-ダンパー物理パラメータ
    physicsLab?: {
      hairK?: number;
      hairC?: number;
      hairWind?: number;
      hairDrive?: number;
      armEnabled?: boolean;
      armMaxAngle?: number;
      armSwayAmp?: number;
      armPivotRatio?: number;
      liftEnabled?: boolean;
      chestMax?: number;
      hairBackScale?: number;
      hairEngine?: string;
      hairWaveStrength?: number;
      earTwitch?: boolean;
      pyokoBounce?: number;
      randomGlance?: boolean;
      /** エフェクト単位ON/OFF（v3 additive。旧フィールドより優先） */
      effects?: Record<string, boolean>;
      engineFamily?: string;
      hairMotionStrength?: number;
      glanceStrength?: number;
      gazeStrength?: number;
      blinkRate?: number;
      liftStrength?: number;
      earTwitchScale?: number;
      pivots?: Record<string, { x?: number; y?: number }>;
      rangesDeg?: Record<string, number>;
      swingScale?: Record<string, number>;
      parallaxScale?: number;
      strandsEnabled?: boolean;
      armBehindBody?: boolean;
    };
  };
  timeline?: { type?: string };
  review?: {
    verdict?: MotionLabVerdict;
    note?: string;
    scores?: Partial<Record<MotionLabReviewKey, number>>;
  };
}

export interface MotionLabManifestResult {
  path: string;
  manifest: MotionLabManifest;
}

export interface SpritalkMotionProfile {
  schema: string;
  sourcePartsDir: string;
  createdAt: string;
  generatedBy: string;
  blink: {
    mode: "keepExisting";
  };
  lipSync: {
    method: MotionLabMethod;
    attackMs: number;
    releaseMs: number;
    crossfadeMs: number;
    restBias: number;
    shapeSmoothing: number;
    bridgeBias: number;
  };
  layerMotion: {
    mode: MotionLabLayerMode;
    preset: MotionLabPreset;
    breathAmplitude: number;
    bodySwayAmplitude: number;
    hairFrontDelayMs: number;
    hairBackDelayMs: number;
  };
  spritalkProceduralAnimation: {
    breathing: {
      enabled: boolean;
      amplitude: number;
      speed: number;
    };
    idleSway: {
      enabled: boolean;
      amplitudeX: number;
      amplitudeY: number;
      speed: number;
      reduceOnSpeech: boolean;
    };
    hairSway: {
      enabled: boolean;
      amplitude: number;
      speed: number;
      rotationAmount: number;
    };
    hairBackSway: {
      enabled: boolean;
      amplitude: number;
      speed: number;
      rotationAmount: number;
    };
  };
  // v2 additive: 検証済みバネ物理パラメータ（docs/motion-lab-integration.md §1）
  physics: {
    hair: {
      mode: "chain";
      segments: number;
      k: number;
      c: number;
      wind: number;
      drive: number;
      /** 房ごとチェーン分割（852話式・最大6房自動検出） */
      strands: boolean;
    };
    arm: {
      enabled: boolean;
      k: number;
      c: number;
      maxAngle: number;
      coupling: number;
      noise: number;
      lift: { enabled: boolean; coupling: number; bounce: number; max: number };
    };
    chest: { k: number; c: number; max: number };
    sway: { k: number; c: number };
    /** パララックス首振り（§8.3。scale=0で無効） */
    parallax: { shiftRatio: number; shearMax: number; scale: number };
    /** 視線ドリフト（§8.4。eyewhite/irides読込時のみ有効） */
    gaze: { rangeRatio: number; returnToFrontOnSpeech: boolean };
    /** 目ハイライトの微小ドリフト（§8.1 #6） */
    highlight: { driftPx: number };
  };
  // v2 additive: SpriTalk特性連動（設計書§8.6）
  presence: {
    entryBounce: number;
    breathOnPause: number;
    randomizePhase: boolean;
  };
  // v2 additive: パララックス係数（設計書§8.3）
  depth: Record<string, number>;
  // v2 additive: 感情別倍率（感情フォルダごとのプロファイルで上書き）
  motionScale: number;
  bounceScale: number;
  runtimeRequirements: {
    lipSyncRenderer: "directLayerSwitch" | "smoothedFrameStepper" | "neutralBridgeOpacityBlend";
    layerRenderer: "existingProceduralAnimator" | "stripWarpExtension";
  };
  review: {
    verdict: MotionLabVerdict;
    note: string;
    scores: Record<MotionLabReviewKey, number>;
  };
}

export interface SpritalkMotionProfileResult {
  path: string;
  profile: SpritalkMotionProfile;
}
export type MotionLabTimelineEvent = { timeMs: number; mouth: MotionLabMouthKey; energy: number };
/** かんたん設定のエフェクト単位ON/OFF（「ソロ」で1効果だけを体感できる） */
export type MotionLabEffectKey =
  | "breath" | "bodySway" | "pyoko" | "hairMotion" | "hairBack" | "parallax"
  | "glance" | "gaze" | "blink" | "arm" | "lift" | "chest" | "earTwitch";
/** 方式（エンジン系統）: ろてじん式=波揺れ＋ぷるぷる弾み / 852話式=バネ・チェーンリグ */
export type MotionLabEngineFamily = "rotejin" | "hachigoni";

/** モーションテンプレート: 調整済みパラメータの一括適用セット（適用後に個別微調整可） */
export interface MotionLabTemplate {
  label: string;
  description: string;
  engine: MotionLabEngineFamily;
  preset: MotionLabPreset;
  layerMode: MotionLabLayerMode;
  hairEngine: "spring" | "wave";
  hairWaveStrength: number;
  hairK: number;
  hairC: number;
  hairWind: number;
  hairDrive: number;
  hairBackScale: number;
  breath: number;
  bodySway: number;
  pyokoBounce: number;
  parallax: number;
  randomGlance: boolean;
  strands: boolean;
  armSwayAmp: number;
  armMaxAngle: number;
  chestMax: number;
  earTwitch: boolean;
}
