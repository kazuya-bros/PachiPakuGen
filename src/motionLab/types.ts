import type { ArmSwayState, ChainState, HairStrandSpringState, SpringState } from "../motionLabPhysics";

export type MotionLabMouthKey = "closed" | "a" | "i" | "u" | "e" | "o";
export type MotionLabMethod = "baseline" | "smooth" | "bridge";
export type MotionLabLayerMode = "simple" | "spring" | "mesh";
export type MotionLabPreset = "calm" | "normal" | "lively";
export type MotionLabBlinkPhase = "idle" | "centering" | "closing" | "opening" | "settling";
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
  /** 腕と同じ変形へ追従し、描画順だけを独立させた切り出しパーツ */
  linkedParts: Record<string, MotionLabLinkedPartResult>;
  /** Independent eyebrow overlay. Null means legacy eye frames still own the brows. */
  eyebrow: string | null;
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

export interface MotionLabLinkedPartResult {
  parent: "arm_l" | "arm_r" | string;
  image: string;
}

export interface MotionLabImageSet {
  body: HTMLImageElement;
  hair: HTMLImageElement | null;
  hairBack: HTMLImageElement | null;
  armL: HTMLImageElement | null;
  armR: HTMLImageElement | null;
  chest: HTMLImageElement | null;
  sways: Record<string, HTMLImageElement>;
  /** linkedPartsの画像デコード後。parentの腕と同じ変形で描画する */
  linkedParts: Record<string, { parent: string; image: HTMLImageElement }>;
  /** Independent eyebrow overlay; absent for legacy baked-eye assets. */
  eyebrow: HTMLImageElement | null;
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
  earTwitches: Map<string, {
    wait: number;
    spring: SpringState;
    queuedFollowUp?: boolean;
    mode?: MotionLabEarTwitchMode;
  }>;
  /** 後ろ髪の房ごとの硬・柔2本バネ */
  strandSpringsBack: HairStrandSpringState[];
  /** A4エンベロープ出力（A1のSmoothDamp前段） */
  envOpen: number;
  mouthVel: { v: number };
  speaking: boolean;
  /** 自動瞬き: 次の瞬きまでの残り秒 */
  blinkWait: number;
  /** 瞬きの描画所有権を切り替える現在フェーズ */
  blinkPhase: MotionLabBlinkPhase;
  /** 現在の瞬きフェーズ内での経過ms */
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
  /** 瞬き前に中央へ滑らかに戻すためのハイライト表示位置 */
  highlight: { x: number; y: number };
  highlightVelX: { v: number };
  highlightVelY: { v: number };
  /** 前髪の房ごとの硬・柔2本バネ（房分割OFF時は未使用） */
  strandSprings: HairStrandSpringState[];
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
  /** うるみ反射を虹彩アルファだけへ限定するマスク用キャンバス */
  eyeWetnessScratch?: HTMLCanvasElement;
  /** Independent left/right eyebrow transform scratch canvas. */
  browScratch?: HTMLCanvasElement;
  /** 眉専用の発話エンベロープ。口より少し遅れて反応し、急な角度変化を防ぐ。 */
  browVoice: number;
  /** 胸部の局所ラスターワープ用スクラッチ（ランタイムごとに再利用） */
  chestWarpScratch?: HTMLCanvasElement;
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
  /** 髪の揺れ方式: false=バネ物理（B1/B3）、true=進行波によるウェーブ */
  hairWaveMode: boolean;
  /** 波揺れの強さ倍率（1=既定） */
  hairWaveStrength: number;
  /** sway_ear* パーツの獣耳ピコピコ揺れ */
  earTwitch: boolean;
  /** 獣耳ピコピコの動き方 */
  earTwitchMode: MotionLabEarTwitchMode;
  /** 発話ぴょこバウンス振幅（px。PuruPuru pyoko参考、バネ平滑） */
  pyokoBounce: number;
  /** ランダムグランス（852話 auto.rand参考）: 数秒ごとに顔向き・視線がふっと変わる */
  randomGlance: boolean;
  /** 髪の揺れ全般（バネ/波/房）。false=髪は頭に追従するだけ */
  hairMotionEnabled: boolean;
  /** 視線ドリフト（eyewhite/irides素材時）。false=瞳は正面固定 */
  gazeEnabled: boolean;
  /** 左右の虹彩を各中心でごく小さく伸縮する */
  irisBreathEnabled: boolean;
  /** 虹彩下辺に控えめな反射を重ねる */
  wetnessEnabled: boolean;
  /** Independent eyebrow micro motion. */
  browEnabled: boolean;
  /** 自動まばたき */
  blinkEnabled: boolean;
  /** エフェクト個別の強さ倍率（1=既定） */
  hairMotionStrength: number;
  glanceStrength: number;
  gazeStrength: number;
  irisBreathStrength: number;
  wetnessStrength: number;
  browStrength: number;
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
  /** B3を房ごとのスプリングへ分割（§8.1 #5） */
  strandsEnabled: boolean;
  /** 腕を体の後ろに描く（素体で腕をbody背面に置いた素材向け。layer-order.json があればそちら優先） */
  armBehindBody: boolean;
  /** 口パクタイムライン（未指定なら内蔵の「あいうえお」テスト） */
  timeline?: MotionLabTimelineEvent[];
  timelineDurationMs?: number;
  /** ライブ表示用の口入力。指定中はタイムラインより優先する。 */
  liveInput?: () => {
    mouth: MotionLabMouthKey;
    energy: number;
    openness: number;
  } | null;
  /** オフライン描画用の乱数源。未指定時は通常プレビューと同じ Math.random。 */
  random?: () => number;
}

export interface MotionLabManifest {
  schema?: string;
  sourcePartsDir?: string;
  createdAt?: string;
  /** 現在の調整値と再生内容から算出した、出力の鮮度判定用ID。 */
  contentFingerprints?: {
    spritalk?: string;
  };
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
      earTwitchMode?: MotionLabEarTwitchMode;
      pyokoBounce?: number;
      randomGlance?: boolean;
      /** エフェクト単位ON/OFF（v3 additive。旧フィールドより優先） */
      effects?: Record<string, boolean>;
      /** モーション方式。現行の保存値は wave / springRig（旧版の値は読込時に正規化）。 */
      engineFamily?: string;
      hairMotionStrength?: number;
      glanceStrength?: number;
      gazeStrength?: number;
      irisBreathStrength?: number;
      wetnessStrength?: number;
      browStrength?: number;
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
  timeline?: {
    type?: "builtInVowelTest" | "text" | string;
    text?: string;
    durationMs?: number;
    events?: MotionLabTimelineEvent[];
  };
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
  /** motion-preview-manifest.json と照合する出力鮮度ID。 */
  contentFingerprint: string;
  blink: {
    mode: "keepExisting";
    /** 頻度倍率。実際の待ち時間 = (intervalMin〜intervalMax秒) ÷ rate。1=既定 */
    rate: number;
    coordination: {
      strategy: "centerThenRife";
      centerMs: number;
      closeMs: number;
      openMs: number;
      settleMs: number;
      /** rate適用前の基準まばたき間隔（秒） */
      intervalMinSeconds: number;
      intervalMaxSeconds: number;
      suppressGazeDuringRife: boolean;
      suppressHighlightDuringRife: boolean;
      resumeDynamicEyeAfterSettle: boolean;
    };
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
    /** モーション方式。現行の保存値は wave / springRig（旧版の値は読込時に正規化）。 */
    engineFamily: MotionLabEngineFamily;
    hair: {
      mode: "chain";
      /** 髪の揺れ方式。waveの場合、layerModeがmeshでも波ワープが優先される（render.ts参照） */
      engine: "spring" | "wave";
      /** engine === "wave" の利便フィールド（両フィールドは常に整合） */
      waveMode: boolean;
      /** 波揺れの強さ倍率（1=既定。engineがwaveの時だけ意味を持つ） */
      waveStrength: number;
      /** 後ろ髪の揺れ倍率（1=従来） */
      backScale: number;
      /** 髪の揺れ全般の強さ倍率（1=既定） */
      motionStrength: number;
      segments: number;
      k: number;
      c: number;
      wind: number;
      drive: number;
      /** 房ごとのスプリング分割（最大6房自動検出） */
      strands: boolean;
    };
    arm: {
      enabled: boolean;
      k: number;
      c: number;
      maxAngle: number;
      /** 左右方向の腕揺れ幅倍率（1=既定） */
      swayAmp: number;
      /** 腕の回転軸Y位置（0=不透明bbox上端〜1=下端） */
      pivotRatio: number;
      /** 腕を体の後ろに描く */
      behindBody: boolean;
      coupling: number;
      noise: number;
      lift: {
        enabled: boolean;
        coupling: number;
        bounce: number;
        max: number;
        /** 肩リフトの強さ倍率（1=既定） */
        strength: number;
      };
    };
    chest: { k: number; c: number; max: number };
    /** 発話ぴょこバウンス（PuruPuru pyoko参考、バネ平滑） */
    pyoko: {
      enabled: boolean;
      /** 振幅（px） */
      amplitudePx: number;
    };
    /** ランダムグランス（852話 auto.rand参考）: 数秒ごとに顔向き・視線がふっと変わる */
    glance: {
      enabled: boolean;
      strength: number;
    };
    sway: {
      k: number;
      c: number;
      /** sway_ear* など、素材単位の回転軸・可動域・揺れ倍率。旧consumerは無視できる追加情報。 */
      partOverrides: {
        pivots: Record<string, { x: number; y: number }>;
        rangesDeg: Record<string, number>;
        swingScale: Record<string, number>;
      };
    };
    earTwitch: {
      enabled: boolean;
      mode: MotionLabEarTwitchMode;
      strength: number;
      bounceVelocity: number;
      rotationVelocity: number;
      maxOffsetPx: number;
      intervalMin: number;
      intervalMax: number;
      followUpScale: number;
      followUpIntervalMin: number;
      followUpIntervalMax: number;
    };
    /** パララックス首振り（§8.3。scale=0で無効） */
    parallax: { shiftRatio: number; shearMax: number; scale: number };
    /** 視線ドリフト（§8.4。eyewhite/irides読込時のみ有効） */
    gaze: {
      enabled: boolean;
      strength: number;
      /** 旧ランタイム互換。現行は虹彩実寸からpx上限を算出するため0。 */
      rangeRatio: number;
      maxRangePx: number;
      periodSeconds: number;
      returnToFrontOnSpeech: boolean;
    };
    /** 目ハイライトの微小ドリフト（§8.1 #6） */
    highlight: { driftPx: number };
    /** 虹彩全体の微小伸縮。左右を別中心で処理する */
    irisBreath: {
      maxScaleDelta: number;
      strength: number;
      responseCurve: "quadratic";
      periodMs: number;
    };
    /** 虹彩下辺へ加える控えめな水分反射 */
    wetness: {
      maxAlpha: number;
      minAlpha: number;
      strength: number;
      responseCurve: "power2.2";
      periodMs: number;
    };
    /** Optional independent eyebrow micro motion. */
    brow: {
      enabled: boolean;
      strength: number;
      maxLiftPx: number;
      maxTiltDeg: number;
      response: "idleAndVoice" | "smoothedVoiceLiftAndAsymmetricTilt";
    };
  };
  /**
   * v3 additive: エフェクト単位ON/OFFの生値（ソロ確認・個別トグルUIとの1:1対応用）。
   * physics以下の各enabled/strengthは既にこれを反映済みなので、通常の再生には不要。
   */
  effects: Record<MotionLabEffectKey, boolean>;
  // v2 additive: SpriTalk特性連動（設計書§8.6）
  presence: {
    entryBounce: number;
    breathOnPause: number;
    randomizePhase: boolean;
  };
  // v2 additive: パララックス係数（設計書§8.3）
  depth: Record<string, number>;
  /**
   * v3 additive: 「全体の強さ」スライダーの生値（0.5〜1.5）。
   * physics以下の各振幅・強さには既に反映済みなので、再生には不要（表示・ログ用）。
   * motionScale/bounceScaleはSpriTalk側が感情フォルダごとに独自に上書きする値のため、
   * ここに intensity を書き込むと二重適用になる。混同しないこと。
   */
  uiIntensity: number;
  // v2 additive: 感情別倍率（感情フォルダごとのプロファイルで上書き）
  motionScale: number;
  bounceScale: number;
  runtimeRequirements: {
    lipSyncRenderer: "directLayerSwitch" | "smoothedFrameStepper" | "neutralBridgeOpacityBlend";
    /** waveWarpRenderer: hairEngine===waveの波ワープ描画。mesh layerModeより優先される */
    layerRenderer: "existingProceduralAnimator" | "stripWarpExtension" | "waveWarpRenderer";
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
  | "glance" | "gaze" | "irisBreath" | "wetness" | "brow" | "blink" | "arm" | "lift" | "chest" | "earTwitch";
export type MotionLabEarTwitchMode = "bounce" | "tilt" | "double";
/** モーション方式: 波揺れ＋弾み / スプリングリグ */
export type MotionLabEngineFamily = "wave" | "springRig";

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
