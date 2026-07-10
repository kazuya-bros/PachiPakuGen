import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import "./App.css";


import {
  type SlotLoadResult,
  type LayerInfo,
  type MappingPreviewResult,
  type RenderCategoryResult,
  type ExportCorrectedLayerResult,
  type ImportCorrectionLayerResult,
  type OriginalImageResult,
  type MouthMaskPreviewResult,
  type CreateBaseResult,
  type CreateDiffResult,
  type ProgressPayload,
  type LayerPatch,
  type InterpPair,
  type MouthMaskSetting,
  type PreviewPan,
  type DiffPreview,
  type InterpStep,
  type BaseStep,
  type WorkspaceStep,
  type SeeThroughProfile,
  type SeeThroughOptionMode,
  type SeeThroughOptions,
  type SeeThroughRuntimeStatus,
  type SeeThroughProgress,
  type SeeThroughRunResult,
  type ExpressionWorkspaceResult,
  type WorkspaceGeneratedPartsStatus,
  type ExtractCodexGeneratedPartsResult,
  type PreviewCodexCompositeResult,
  type PreviewCodexRifeResult,
  type GenerateCodexRifeOutputResult,
  type SaveCodexBasePartsResult,
  type AdjustCodexExtractedPartsResult,
  type LoadCodexExpressionJobResult,
  DEFAULT_SEE_THROUGH_OPTIONS,
  RIFE_FRAME_MIN,
  RIFE_FRAME_MAX,
  RIFE_FRAME_RECOMMENDED,
  isNoisySeeThroughWarning,
  displaySeeThroughMessage,
  formatElapsed,
  CHEST_CUT_SENTINEL,
  workspacePreviewItemKey,
  workspacePreviewItemLabel,
  WORKSPACE_ADJUST_PART_KEYS,
  EYE_PAIRS,
  MOUTH_PAIRS_SINGLE,
  MOUTH_PAIRS_VOWELS,
} from "./workspace/types";
import {
  type MotionLabMouthKey,
  type MotionLabMethod,
  type MotionLabLayerMode,
  type MotionLabPreset,
  type MotionLabVerdict,
  type MotionLabReviewKey,
  type MotionLabPartsResult,
  type MotionLabImageSet,
  type MotionLabMouthRuntime,
  type MotionLabManifest,
  type MotionLabManifestResult,
  type SpritalkMotionProfile,
  type SpritalkMotionProfileResult,
  type MotionLabTimelineEvent,
  type MotionLabEffectKey,
  type MotionLabEngineFamily,
} from "./motionLab/types";
import {
  MOTION_LAB_MOUTH_KEYS,
  MOTION_LAB_VOWEL_KEYS,
  MOTION_LAB_MOUTH_LABELS,
  MOTION_LAB_DURATION_MS,
  MOTION_LAB_TIMELINE,
  motionLabTimelineFromText,
  MOTION_LAB_PRESET_FACTORS,
  MOTION_LAB_EFFECT_DEFS,
  MOTION_LAB_EFFECT_DEFAULTS,
  MOTION_LAB_EFFECT_SOLO_DEPS,
  MOTION_LAB_TEMPLATES,
  MOTION_LAB_HAIR_SEGMENTS,
  MOTION_LAB_HAIR_DEFAULTS,
  MOTION_LAB_ARM_DEFAULTS,
  MOTION_LAB_CHEST_DEFAULTS,
  MOTION_LAB_PARALLAX_DEFAULTS,
  MOTION_LAB_GAZE_DEFAULTS,
  MOTION_LAB_HIGHLIGHT_DEFAULTS,
  MOTION_LAB_SWAY_DEFAULTS,
  MOTION_LAB_PRESENCE_DEFAULTS,
  MOTION_LAB_DEPTH_DEFAULTS,
  MOTION_LAB_DEFAULT_REVIEW_SCORES,
  MOTION_LAB_REVIEW_LABELS,
} from "./motionLab/constants";
import {
  clamp,
  loadMotionLabImage,
  validMotionLabPreset,
  validMotionLabLayerMode,
  validMotionLabMethod,
  validMotionLabVerdict,
  createMotionLabPhysics,
  resetMotionLabRuntime,
  prepareMotionLabCanvas,
  drawMotionLabScene,
} from "./motionLab/render";
import { MotionTunePanel } from "./motionLab/MotionTunePanel";

type Mode = "select" | "workspace" | "motion_lab" | "base_input" | "hair_edit" | "base_edit" | "correction" | "interp";
type ThemeMode = "dark" | "light";

function App() {
  const [mode, setMode] = useState<Mode>("select");
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "dark";
    return window.localStorage.getItem("pachipakugen-theme") === "light" ? "light" : "dark";
  });

  // Shared
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("モードを選択してください");

  const [workspaceStep, setWorkspaceStep] = useState<WorkspaceStep>(1);
  const [expressionWorkspace, setExpressionWorkspace] = useState<ExpressionWorkspaceResult | null>(null);
  const [workspaceFiles, setWorkspaceFiles] = useState<Record<string, string>>({});
  const [workspaceImagePreviews, setWorkspaceImagePreviews] = useState<Record<string, string>>({});
  const [workspaceGeneratedStatus, setWorkspaceGeneratedStatus] = useState<WorkspaceGeneratedPartsStatus | null>(null);
  const [workspaceExtractResult, setWorkspaceExtractResult] = useState<ExtractCodexGeneratedPartsResult | null>(null);
  const [workspaceCompositePreview, setWorkspaceCompositePreview] = useState<PreviewCodexCompositeResult | null>(null);
  const [workspaceRifePreview, setWorkspaceRifePreview] = useState<PreviewCodexRifeResult | null>(null);
  const [workspaceSelectedPreviewPart, setWorkspaceSelectedPreviewPart] = useState<string>("base");
  const [workspacePreviewZoom, setWorkspacePreviewZoom] = useState(1);
  const [workspacePreviewPan, setWorkspacePreviewPan] = useState<PreviewPan>({ x: 0, y: 0 });
  const [workspacePreviewDragging, setWorkspacePreviewDragging] = useState(false);
  const workspacePreviewDrag = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startPan: PreviewPan;
    mode: "pan" | "part";
    startOffsetX: number;
    startOffsetY: number;
  } | null>(null);
  const [workspaceRifeResult, setWorkspaceRifeResult] = useState<GenerateCodexRifeOutputResult | null>(null);
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [seeThroughRuntime, setSeeThroughRuntime] = useState<SeeThroughRuntimeStatus | null>(null);
  const [seeThroughProgress, setSeeThroughProgress] = useState<SeeThroughProgress | null>(null);
  const [seeThroughStartedAt, setSeeThroughStartedAt] = useState<number | null>(null);
  const [seeThroughElapsedSeconds, setSeeThroughElapsedSeconds] = useState(0);
  const [seeThroughProfile, setSeeThroughProfile] = useState<SeeThroughProfile>("low-vram");
  const [seeThroughSplitParts, setSeeThroughSplitParts] = useState(true);
  const [seeThroughOptions, setSeeThroughOptions] = useState<SeeThroughOptions>(DEFAULT_SEE_THROUGH_OPTIONS);
  const [workspacePartOffsetX, setWorkspacePartOffsetX] = useState(0);
  const [workspacePartOffsetY, setWorkspacePartOffsetY] = useState(0);
  const [workspacePartScale, setWorkspacePartScale] = useState(100);
  /** Step5の調整対象: "all"=全パーツ一括 / それ以外=パーツ個別（例: "mouth-a"） */
  const [workspaceAdjustTarget, setWorkspaceAdjustTarget] = useState("all");
  /** Step5プレビュー上のドラッグでパーツを動かすモード */
  const [workspacePartDragMode, setWorkspacePartDragMode] = useState(false);

  // === Motion Preview Lab ===
  const [motionLabParts, setMotionLabParts] = useState<MotionLabPartsResult | null>(null);
  const [motionLabImages, setMotionLabImages] = useState<MotionLabImageSet | null>(null);
  const [motionLabImagesLoading, setMotionLabImagesLoading] = useState(false);
  const [motionLabPlaying, setMotionLabPlaying] = useState(true);
  const [motionLabMethod, setMotionLabMethod] = useState<MotionLabMethod>("smooth");
  const [motionLabLayerMode, setMotionLabLayerMode] = useState<MotionLabLayerMode>("spring");
  const [motionLabPreset, setMotionLabPreset] = useState<MotionLabPreset>("normal");
  const [motionLabAttackMs, setMotionLabAttackMs] = useState(90);
  const [motionLabReleaseMs, setMotionLabReleaseMs] = useState(160);
  const [motionLabCrossfadeMs, setMotionLabCrossfadeMs] = useState(50);
  const [motionLabRestBias, setMotionLabRestBias] = useState(0.25);
  const [motionLabShapeSmoothing, setMotionLabShapeSmoothing] = useState(0.65);
  const [motionLabBridgeBias, setMotionLabBridgeBias] = useState(0.45);
  const [motionLabBreathAmplitude, setMotionLabBreathAmplitude] = useState(1);
  const [motionLabBodySwayAmplitude, setMotionLabBodySwayAmplitude] = useState(1);
  const [motionLabHairFrontDelay, setMotionLabHairFrontDelay] = useState(0.18);
  const [motionLabHairBackDelay, setMotionLabHairBackDelay] = useState(0.28);
  // バネ-ダンパー物理パラメータ（検証済み既定値）
  const [motionLabHairK, setMotionLabHairK] = useState(MOTION_LAB_HAIR_DEFAULTS.k);
  const [motionLabHairC, setMotionLabHairC] = useState(MOTION_LAB_HAIR_DEFAULTS.c);
  const [motionLabHairWind, setMotionLabHairWind] = useState(MOTION_LAB_HAIR_DEFAULTS.wind);
  const [motionLabHairDrive, setMotionLabHairDrive] = useState(MOTION_LAB_HAIR_DEFAULTS.drive);
  // エフェクト単位のON/OFF（かんたん設定の体感リスト。ソロで1効果だけ確認できる）
  const [motionLabEffects, setMotionLabEffects] = useState<Record<MotionLabEffectKey, boolean>>({
    ...MOTION_LAB_EFFECT_DEFAULTS,
  });
  const setMotionLabEffect = (key: MotionLabEffectKey, value: boolean) =>
    setMotionLabEffects(prev => ({ ...prev, [key]: value }));
  const soloMotionLabEffect = (key: MotionLabEffectKey) => {
    const solo = Object.fromEntries(
      MOTION_LAB_EFFECT_DEFS.map(def => [def.key, def.key === key]),
    ) as Record<MotionLabEffectKey, boolean>;
    for (const dep of MOTION_LAB_EFFECT_SOLO_DEPS[key] ?? []) solo[dep] = true;
    setMotionLabEffects(solo);
  };
  const [motionLabArmMaxAngle, setMotionLabArmMaxAngle] = useState(MOTION_LAB_ARM_DEFAULTS.maxAngle);
  // 左右方向の腕揺れ幅倍率と回転軸位置（肩の弾みliftとは独立）
  const [motionLabArmSwayAmp, setMotionLabArmSwayAmp] = useState(1.0);
  const [motionLabArmPivotRatio, setMotionLabArmPivotRatio] = useState(0);
  const [motionLabChestMax, setMotionLabChestMax] = useState(MOTION_LAB_CHEST_DEFAULTS.max);
  // 後ろ髪の揺れ倍率（揺れすぎ調整。既定0.6=従来の6割）
  const [motionLabHairBackScale, setMotionLabHairBackScale] = useState(0.6);
  // 発話ぴょこバウンス（PuruPuru参考。0=無効）
  const [motionLabPyokoBounce, setMotionLabPyokoBounce] = useState(3);
  // 方式（エンジン系統）とエフェクト個別の強さ倍率
  const [motionLabEngineFamily, setMotionLabEngineFamily] = useState<MotionLabEngineFamily>("hachigoni");
  const [motionLabHairMotionStrength, setMotionLabHairMotionStrength] = useState(1.0);
  const [motionLabGlanceStrength, setMotionLabGlanceStrength] = useState(1.0);
  const [motionLabGazeStrength, setMotionLabGazeStrength] = useState(1.0);
  const [motionLabBlinkRate, setMotionLabBlinkRate] = useState(1.0);
  const [motionLabLiftStrength, setMotionLabLiftStrength] = useState(1.0);
  const [motionLabEarTwitchScale, setMotionLabEarTwitchScale] = useState(1.0);

  function applyMotionLabEngineFamily(family: MotionLabEngineFamily) {
    setMotionLabEngineFamily(family);
    setMotionLabHairEngine(family === "rotejin" ? "wave" : "spring");
    setMotionLabTemplateName(prev =>
      prev && MOTION_LAB_TEMPLATES[prev]?.engine === family ? prev : null,
    );
  }

  // 最後に適用したテンプレート名（表示用。スライダー個別調整でも保持される）
  const [motionLabTemplateName, setMotionLabTemplateName] = useState<string | null>(null);

  function applyMotionLabTemplate(key: string) {
    const template = MOTION_LAB_TEMPLATES[key];
    if (!template) return;
    setMotionLabTemplateName(key);
    setMotionLabEngineFamily(template.engine);
    setMotionLabPreset(template.preset);
    // テンプレは既存パラメータで強度を表現するため、エフェクト個別倍率は等倍へ戻す
    setMotionLabHairMotionStrength(1);
    setMotionLabGlanceStrength(1);
    setMotionLabGazeStrength(1);
    setMotionLabBlinkRate(1);
    setMotionLabLiftStrength(1);
    setMotionLabEarTwitchScale(1);
    setMotionLabLayerMode(template.layerMode);
    setMotionLabHairEngine(template.hairEngine);
    setMotionLabHairWaveStrength(template.hairWaveStrength);
    setMotionLabHairK(template.hairK);
    setMotionLabHairC(template.hairC);
    setMotionLabHairWind(template.hairWind);
    setMotionLabHairDrive(template.hairDrive);
    setMotionLabHairBackScale(template.hairBackScale);
    setMotionLabBreathAmplitude(template.breath);
    setMotionLabBodySwayAmplitude(template.bodySway);
    setMotionLabPyokoBounce(template.pyokoBounce);
    setMotionLabParallaxScale(template.parallax);
    // テンプレは全エフェクトONを起点に、テンプレ固有のON/OFFだけ反映する
    setMotionLabEffects({
      ...MOTION_LAB_EFFECT_DEFAULTS,
      breath: true, bodySway: true, pyoko: true, hairMotion: true, hairBack: true,
      parallax: true, glance: template.randomGlance, gaze: true, blink: true,
      arm: true, lift: true, chest: true, earTwitch: template.earTwitch,
    });
    setMotionLabStrandsEnabled(template.strands);
    setMotionLabArmSwayAmp(template.armSwayAmp);
    setMotionLabArmMaxAngle(template.armMaxAngle);
    setMotionLabChestMax(template.chestMax);
    setMotionLabIntensity(1.0);
    // 回転軸・可動域・揺れ幅の個別上書き（素材依存の調整）は維持する
  }
  // 髪の揺れ方式（spring=バネ物理B1/B3、wave=ろてじん式進行波）と波の強さ
  const [motionLabHairEngine, setMotionLabHairEngine] = useState<"spring" | "wave">("spring");
  const [motionLabHairWaveStrength, setMotionLabHairWaveStrength] = useState(1.0);
  // 回転軸・可動域エディタ: パーツごとの回転軸上書き（画像座標px）と可動域（±度、0=既定）
  const [motionLabPivots, setMotionLabPivots] = useState<Record<string, { x: number; y: number }>>({});
  const [motionLabRangesDeg, setMotionLabRangesDeg] = useState<Record<string, number>>({});
  const [motionLabSwingScale, setMotionLabSwingScale] = useState<Record<string, number>>({});
  const [motionLabPivotEditPart, setMotionLabPivotEditPart] = useState<string | null>(null);
  // かんたん設定の「動きの強さ」ノブ（詳細スライダー群への一括倍率。詳細側を個別に動かすと乖離してよい）
  const [motionLabIntensity, setMotionLabIntensity] = useState(1.0);
  const [motionLabParallaxScale, setMotionLabParallaxScale] = useState(1.0);
  const [motionLabArmBehindBody, setMotionLabArmBehindBody] = useState(false);
  const [motionLabStrandsEnabled, setMotionLabStrandsEnabled] = useState(false);
  const [motionLabText, setMotionLabText] = useState("こんにちは、あいうえお");
  const [motionLabCustomTimeline, setMotionLabCustomTimeline] = useState<{
    timeline: MotionLabTimelineEvent[];
    durationMs: number;
  } | null>(null);
  const [motionLabManifestPath, setMotionLabManifestPath] = useState("");
  const [motionLabProfilePath, setMotionLabProfilePath] = useState("");
  const [motionLabVerdict, setMotionLabVerdict] = useState<MotionLabVerdict>("undecided");
  const [motionLabReviewScores, setMotionLabReviewScores] = useState<Record<MotionLabReviewKey, number>>(MOTION_LAB_DEFAULT_REVIEW_SCORES);
  const [motionLabReviewNote, setMotionLabReviewNote] = useState("");
  const motionLabBaselineCanvasRef = useRef<HTMLCanvasElement>(null);
  const motionLabCandidateCanvasRef = useRef<HTMLCanvasElement>(null);
  const motionLabBaselineRuntimeRef = useRef<MotionLabMouthRuntime>({
    openY: 0,
    activeTarget: "closed",
    previousTarget: "closed",
    transitionStartMs: 0,
    lastMs: 0,
    physics: createMotionLabPhysics(),
  });
  const motionLabCandidateRuntimeRef = useRef<MotionLabMouthRuntime>({
    openY: 0,
    activeTarget: "closed",
    previousTarget: "closed",
    transitionStartMs: 0,
    lastMs: 0,
    physics: createMotionLabPhysics(),
  });

  useEffect(() => {
    window.localStorage.setItem("pachipakugen-theme", themeMode);
  }, [themeMode]);

  // === Input (shared) ===
  const [loadResult, setLoadResult] = useState<SlotLoadResult | null>(null);
  const [layerMapping, setLayerMapping] = useState<Record<string, string>>({});

  // === 素体モード ===
  const [mappingPreview, setMappingPreview] = useState<MappingPreviewResult | null>(null);
  // Step1: hair
  const [hairPreview, setHairPreview] = useState("");
  const [hairEnabledLayers, setHairEnabledLayers] = useState<Record<string, boolean>>({});
  const [hairLayerOrder, setHairLayerOrder] = useState<string[]>([]);
  const [hairBackPreview, setHairBackPreview] = useState("");
  const [hairBackEnabledLayers, setHairBackEnabledLayers] = useState<Record<string, boolean>>({});
  const [hairBackLayerOrder, setHairBackLayerOrder] = useState<string[]>([]);
  // Step2: body
  const [bodyPreview, setBodyPreview] = useState("");
  const [enabledLayers, setEnabledLayers] = useState<Record<string, boolean>>({});
  const [layerOrder, setLayerOrder] = useState<string[]>([]);
  const [layerPatches, setLayerPatches] = useState<LayerPatch[]>([]);
  const [layerOpacities, setLayerOpacities] = useState<Record<string, number>>({});
  const [selectedBodyLayer, setSelectedBodyLayer] = useState<string>("");
  const [overlapHighlightEnabled, setOverlapHighlightEnabled] = useState(false);
  const [baseResult, setBaseResult] = useState<CreateBaseResult | null>(null);
  const [correctionOutputPath, setCorrectionOutputPath] = useState("");
  const [baseStep, setBaseStep] = useState<BaseStep>(1);
  const [patchDraftSource, setPatchDraftSource] = useState("");
  const [patchTool, setPatchTool] = useState<"paint" | "erase">("paint");
  const [patchBrushSize, setPatchBrushSize] = useState(24);
  // 0=硬いエッジ（従来）、1=最大にぼかしたエッジ（境界のギザギザを緩和）
  const [patchBrushSoftness, setPatchBrushSoftness] = useState(0.5);
  const [sam3SelectMode, setSam3SelectMode] = useState(false);
  const [sam3Selecting, setSam3Selecting] = useState(false);
  // 胸を切出（852話式・素体調整の標準導線）: bodyから塗った範囲を chest.png として分離する
  const [chestMaskDataUrl, setChestMaskDataUrl] = useState<string | null>(null);
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number; size: number; visible: boolean }>({ x: 0, y: 0, size: 0, visible: false });

  // === フレーム補間モード (interp) ===
  const [diffTarget] = useState<"eye" | "mouth">("eye");
  const [frameCount, setFrameCount] = useState(8);
  const [outputPath, setOutputPath] = useState("");
  const [progress, setProgress] = useState({ current: 0, total: 0, pair_name: "" });
  const [completedDiffs, setCompletedDiffs] = useState<string[]>([]);
  const [diffPreviews, setDiffPreviews] = useState<DiffPreview[]>([]);
  const [diffPreviewTick, setDiffPreviewTick] = useState(0);
  const [diffPreviewZoom, setDiffPreviewZoom] = useState(1);
  const [interpPaths, setInterpPaths] = useState<Record<string, string>>({});
  const [interpOriginals, setInterpOriginals] = useState<Record<string, string>>({});
  const [interpGenerating, setInterpGenerating] = useState(false);
  const [interpStep, setInterpStep] = useState<InterpStep>(1);
  const [mouthMode] = useState<"single" | "vowels">("single");
  const [mouthMaskSettings, setMouthMaskSettings] = useState<Record<string, MouthMaskSetting>>({});
  const [mouthMaskPreviews, setMouthMaskPreviews] = useState<Record<string, string>>({});
  const [mouthMaskPreviewing, setMouthMaskPreviewing] = useState<Record<string, boolean>>({});
  const [mouthPreviewZooms, setMouthPreviewZooms] = useState<Record<string, number>>({});
  const [mouthPreviewPans, setMouthPreviewPans] = useState<Record<string, PreviewPan>>({});
  const [activePreviewKey, setActivePreviewKey] = useState("");
  const mouthMaskUpdateTimers = useRef<Record<string, number>>({});
  const mouthPreviewDrag = useRef<{ key: string; startX: number; startY: number; startPan: PreviewPan } | null>(null);

  // Zoom & pan (base_edit)
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });
  const previewRef = useRef<HTMLDivElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskDrawingRef = useRef(false);
  const layerPatchesRef = useRef(layerPatches);
  layerPatchesRef.current = layerPatches;
  const layerOpacitiesRef = useRef(layerOpacities);
  layerOpacitiesRef.current = layerOpacities;
  const opacityRenderTimer = useRef<number | null>(null);
  // 表示切替の連打をまとめて1回の再合成にするためのデバウンス
  const toggleRenderTimer = useRef<number | null>(null);
  const enabledLayersRef = useRef<Record<string, boolean>>({});
  // Drag reorder (generic - used for body, hair, hair_back)
  type DragTarget = "body" | "hair" | "hair_back";
  const dragState = useRef<{ target: DragTarget; idx: number; startY: number; currentIdx: number } | null>(null);
  const [draggedIdx, setDraggedIdx] = useState<number | null>(null);
  const [dragTarget, setDragTarget] = useState<DragTarget | null>(null);
  const layerOrderRef = useRef(layerOrder);
  layerOrderRef.current = layerOrder;
  const hairLayerOrderRef = useRef(hairLayerOrder);
  hairLayerOrderRef.current = hairLayerOrder;
  const hairBackLayerOrderRef = useRef(hairBackLayerOrder);
  hairBackLayerOrderRef.current = hairBackLayerOrder;

  useEffect(() => {
    const unlisten = listen<ProgressPayload>("generation-progress", (event) => {
      setProgress({ current: event.payload.current, total: event.payload.total, pair_name: event.payload.pair_name });
    });
    const unlistenSeeThrough = listen<SeeThroughProgress>("see-through-progress", (event) => {
      setSeeThroughProgress(event.payload);
      if (!isNoisySeeThroughWarning(event.payload.message)) {
        setStatus(event.payload.message);
      }
    });
    return () => {
      unlisten.then(fn => fn());
      unlistenSeeThrough.then(fn => fn());
      if (opacityRenderTimer.current !== null) {
        window.clearTimeout(opacityRenderTimer.current);
      }
      for (const timer of Object.values(mouthMaskUpdateTimers.current)) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  useEffect(() => {
    if (!workspaceBusy || !seeThroughStartedAt) {
      setSeeThroughElapsedSeconds(0);
      return;
    }
    const updateElapsed = () => {
      setSeeThroughElapsedSeconds(Math.max(0, Math.floor((Date.now() - seeThroughStartedAt) / 1000)));
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [workspaceBusy, seeThroughStartedAt]);

  useEffect(() => {
    if (diffPreviews.length === 0) return;
    const timer = window.setInterval(() => {
      setDiffPreviewTick(prev => prev + 1);
    }, 160);
    return () => window.clearInterval(timer);
  }, [diffPreviews.length]);

  useEffect(() => {
    let cancelled = false;
    if (!motionLabParts) {
      setMotionLabImages(null);
      setMotionLabImagesLoading(false);
      return;
    }
    const parts = motionLabParts;

    setMotionLabImages(null);
    setMotionLabImagesLoading(true);
    async function loadImages() {
      try {
        const mouthEntries = await Promise.all(
          MOTION_LAB_MOUTH_KEYS.map(async (key) => {
            const sources = parts.mouths[key] ?? [];
            const images = await Promise.all(sources.map(loadMotionLabImage));
            return [key, images] as const;
          }),
        );
        const swayEntries = await Promise.all(
          Object.entries(parts.sways).map(async ([name, source]) => [name, await loadMotionLabImage(source)] as const),
        );
        const nextImages: MotionLabImageSet = {
          body: await loadMotionLabImage(parts.body),
          hair: parts.hair ? await loadMotionLabImage(parts.hair) : null,
          hairBack: parts.hairBack ? await loadMotionLabImage(parts.hairBack) : null,
          armL: parts.armL ? await loadMotionLabImage(parts.armL) : null,
          armR: parts.armR ? await loadMotionLabImage(parts.armR) : null,
          chest: parts.chest ? await loadMotionLabImage(parts.chest) : null,
          sways: Object.fromEntries(swayEntries),
          eyewhite: parts.eyewhite ? await loadMotionLabImage(parts.eyewhite) : null,
          irides: parts.irides ? await loadMotionLabImage(parts.irides) : null,
          highlight: parts.highlight ? await loadMotionLabImage(parts.highlight) : null,
          eyeFrames: await Promise.all(parts.eyeFrames.map(loadMotionLabImage)),
          mouths: Object.fromEntries(mouthEntries) as Partial<Record<MotionLabMouthKey, HTMLImageElement[]>>,
        };
        if (!cancelled) {
          setMotionLabImages(nextImages);
          setMotionLabImagesLoading(false);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : String(loadError));
          setMotionLabImagesLoading(false);
        }
      }
    }
    void loadImages();

    return () => {
      cancelled = true;
    };
  }, [motionLabParts]);

  // 素材読込時に物理をリセット（登場撃力＋位相ランダム化）。
  // スライダー変更ではリセットしない = 調整中もポーズが飛ばない
  useEffect(() => {
    if (!motionLabImages) return;
    resetMotionLabRuntime(motionLabBaselineRuntimeRef.current);
    resetMotionLabRuntime(motionLabCandidateRuntimeRef.current);
  }, [motionLabImages]);

  useEffect(() => {
    if (mode !== "motion_lab" || !motionLabParts || !motionLabImages) return;
    const baselineCtx = prepareMotionLabCanvas(motionLabBaselineCanvasRef.current, motionLabParts.width, motionLabParts.height);
    const candidateCtx = prepareMotionLabCanvas(motionLabCandidateCanvasRef.current, motionLabParts.width, motionLabParts.height);
    if (!baselineCtx || !candidateCtx) return;

    const baselineRuntime = motionLabBaselineRuntimeRef.current;
    const candidateRuntime = motionLabCandidateRuntimeRef.current;

    const startedAt = performance.now();
    let animationId = 0;

    const draw = (now: number) => {
      const elapsedMs = motionLabPlaying ? now - startedAt : 0;
      drawMotionLabScene(baselineCtx, motionLabParts, motionLabImages, baselineRuntime, elapsedMs, {
        mouthMethod: "baseline",
        layerMode: "simple",
        preset: "calm",
        attackMs: 0,
        releaseMs: 0,
        crossfadeMs: 0,
        restBias: 0,
        shapeSmoothing: 0,
        bridgeBias: 0,
        breathAmplitude: motionLabBreathAmplitude,
        bodySwayAmplitude: motionLabBodySwayAmplitude,
        hairK: MOTION_LAB_HAIR_DEFAULTS.k,
        hairC: MOTION_LAB_HAIR_DEFAULTS.c,
        hairWind: MOTION_LAB_HAIR_DEFAULTS.wind,
        hairDrive: MOTION_LAB_HAIR_DEFAULTS.drive,
        armEnabled: false,
        armMaxAngle: MOTION_LAB_ARM_DEFAULTS.maxAngle,
        armSwayAmp: 1,
        armPivotRatio: 0,
        liftEnabled: false,
        chestMax: 0,
        hairBackScale: 1,
        hairWaveMode: false,
        hairWaveStrength: 1,
        earTwitch: false,
        pyokoBounce: 1.4,
        randomGlance: false,
        hairMotionEnabled: true,
        gazeEnabled: true,
        blinkEnabled: true,
        hairMotionStrength: 1,
        glanceStrength: 1,
        gazeStrength: 1,
        blinkRate: 1,
        liftStrength: 1,
        earTwitchScale: 1,
        pivots: {},
        rangesDeg: {},
        swingScale: {},
        pivotEditPart: null,
        parallaxScale: 0,
        strandsEnabled: false,
        armBehindBody: motionLabArmBehindBody,
        timeline: motionLabCustomTimeline?.timeline,
        timelineDurationMs: motionLabCustomTimeline?.durationMs,
      });
      const fx = motionLabEffects;
      drawMotionLabScene(candidateCtx, motionLabParts, motionLabImages, candidateRuntime, elapsedMs, {
        mouthMethod: motionLabMethod,
        layerMode: motionLabLayerMode,
        preset: motionLabPreset,
        attackMs: motionLabAttackMs,
        releaseMs: motionLabReleaseMs,
        crossfadeMs: motionLabCrossfadeMs,
        restBias: motionLabRestBias,
        shapeSmoothing: motionLabShapeSmoothing,
        bridgeBias: motionLabBridgeBias,
        breathAmplitude: fx.breath ? motionLabBreathAmplitude : 0,
        bodySwayAmplitude: fx.bodySway ? motionLabBodySwayAmplitude : 0,
        hairK: motionLabHairK,
        hairC: motionLabHairC,
        hairWind: motionLabHairWind,
        hairDrive: motionLabHairDrive,
        // 肩の弾みだけONの場合も物理更新は必要（liftはarm経由で計算される）
        armEnabled: fx.arm || fx.lift,
        armMaxAngle: motionLabArmMaxAngle,
        armSwayAmp: fx.arm ? motionLabArmSwayAmp : 0,
        armPivotRatio: motionLabArmPivotRatio,
        liftEnabled: fx.lift,
        chestMax: fx.chest ? motionLabChestMax : 0,
        hairBackScale: fx.hairBack ? motionLabHairBackScale : 0,
        hairWaveMode: motionLabHairEngine === "wave",
        hairWaveStrength: motionLabHairWaveStrength,
        earTwitch: fx.earTwitch,
        pyokoBounce: fx.pyoko ? motionLabPyokoBounce : 0,
        randomGlance: fx.glance,
        hairMotionEnabled: fx.hairMotion,
        gazeEnabled: fx.gaze,
        blinkEnabled: fx.blink,
        hairMotionStrength: motionLabHairMotionStrength,
        glanceStrength: motionLabGlanceStrength,
        gazeStrength: motionLabGazeStrength,
        blinkRate: motionLabBlinkRate,
        liftStrength: motionLabLiftStrength,
        earTwitchScale: motionLabEarTwitchScale,
        parallaxScale: fx.parallax ? motionLabParallaxScale : 0,
        pivots: motionLabPivots,
        rangesDeg: motionLabRangesDeg,
        swingScale: motionLabSwingScale,
        pivotEditPart: motionLabPivotEditPart,
        strandsEnabled: motionLabStrandsEnabled,
        armBehindBody: motionLabArmBehindBody,
        timeline: motionLabCustomTimeline?.timeline,
        timelineDurationMs: motionLabCustomTimeline?.durationMs,
      });

      if (motionLabPlaying) {
        animationId = window.requestAnimationFrame(draw);
      }
    };

    animationId = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(animationId);
  }, [
    mode,
    motionLabParts,
    motionLabImages,
    motionLabPlaying,
    motionLabMethod,
    motionLabLayerMode,
    motionLabPreset,
    motionLabAttackMs,
    motionLabReleaseMs,
    motionLabCrossfadeMs,
    motionLabRestBias,
    motionLabShapeSmoothing,
    motionLabBridgeBias,
    motionLabBreathAmplitude,
    motionLabBodySwayAmplitude,
    motionLabHairK,
    motionLabHairC,
    motionLabHairWind,
    motionLabHairDrive,
    motionLabEffects,
    motionLabArmMaxAngle,
    motionLabArmSwayAmp,
    motionLabArmPivotRatio,
    motionLabChestMax,
    motionLabHairBackScale,
    motionLabPyokoBounce,
    motionLabHairEngine,
    motionLabHairWaveStrength,
    motionLabHairMotionStrength,
    motionLabGlanceStrength,
    motionLabGazeStrength,
    motionLabBlinkRate,
    motionLabLiftStrength,
    motionLabEarTwitchScale,
    motionLabPivots,
    motionLabRangesDeg,
    motionLabSwingScale,
    motionLabPivotEditPart,
    motionLabParallaxScale,
    motionLabStrandsEnabled,
    motionLabArmBehindBody,
    motionLabCustomTimeline,
  ]);

  // --- Category rendering ---
  async function renderCategory(
    order: string[],
    enabled: Record<string, boolean>,
    patches: LayerPatch[] = [],
    opacities: Record<string, number> = {},
  ): Promise<string> {
    const active = [...order.filter(name => enabled[name] !== false)].reverse();
    try {
      const result = await invoke<RenderCategoryResult>("render_category", {
        mappingJson: JSON.stringify(layerMapping), target: "body", enabledLayers: active, layerPatches: patches, layerOpacities: opacities,
        overlapHighlight: false,
      });
      return result.preview;
    } catch (e) { console.error("render error:", e); return ""; }
  }

  async function renderHair(order: string[], enabled: Record<string, boolean>) {
    setHairPreview(await renderCategory(order, enabled, [], {}));
  }

  async function renderHairBack(order: string[], enabled: Record<string, boolean>) {
    setHairBackPreview(await renderCategory(order, enabled, [], {}));
  }

  // --- Body rendering ---
  async function renderBody(
    order: string[],
    enabled: Record<string, boolean>,
    patches: LayerPatch[] = layerPatches,
    opacities: Record<string, number> = layerOpacities,
    overlapHighlight = overlapHighlightEnabled,
  ) {
    const active = [...order.filter(name => enabled[name] !== false)].reverse();
    try {
      const result = await invoke<RenderCategoryResult>("render_category", {
        mappingJson: JSON.stringify(layerMapping), target: "body", enabledLayers: active, layerPatches: patches, layerOpacities: opacities,
        overlapHighlight,
      });
      setBodyPreview(result.preview);
    } catch (e) { console.error("render error:", e); }
  }

  async function handleLayerToggle(name: string, checked: boolean) {
    // 連打時は最後の状態だけ合成する（毎クリック全レイヤー再合成すると重い）
    const base = toggleRenderTimer.current !== null ? enabledLayersRef.current : enabledLayers;
    const newEnabled = { ...base, [name]: checked };
    enabledLayersRef.current = newEnabled;
    setEnabledLayers(newEnabled);
    if (toggleRenderTimer.current !== null) {
      window.clearTimeout(toggleRenderTimer.current);
    }
    toggleRenderTimer.current = window.setTimeout(() => {
      toggleRenderTimer.current = null;
      void renderBody(layerOrderRef.current, enabledLayersRef.current);
    }, 160);
  }

  async function setAllLayerVisibility(visible: boolean) {
    const nextEnabled: Record<string, boolean> = { ...enabledLayers };
    for (const name of layerOrder) {
      nextEnabled[name] = visible;
    }
    setEnabledLayers(nextEnabled);
    await renderBody(layerOrder, nextEnabled);
  }

  function updateLayerOpacityDraft(name: string, opacity: number) {
    const next = { ...layerOpacitiesRef.current, [name]: Math.max(0, Math.min(1, opacity)) };
    layerOpacitiesRef.current = next;
    setLayerOpacities(next);
    if (opacityRenderTimer.current !== null) {
      window.clearTimeout(opacityRenderTimer.current);
    }
    opacityRenderTimer.current = window.setTimeout(() => {
      opacityRenderTimer.current = null;
      void commitLayerOpacity();
    }, 120);
  }

  async function commitLayerOpacity() {
    if (opacityRenderTimer.current !== null) {
      window.clearTimeout(opacityRenderTimer.current);
      opacityRenderTimer.current = null;
    }
    await renderBody(layerOrderRef.current, enabledLayers, layerPatchesRef.current, layerOpacitiesRef.current);
  }

  async function setAllBodyOpacities(opacity: number) {
    const value = Math.max(0, Math.min(1, opacity));
    const next: Record<string, number> = {};
    for (const name of layerOrderRef.current) next[name] = value;
    layerOpacitiesRef.current = next;
    setLayerOpacities(next);
    await renderBody(layerOrderRef.current, enabledLayers, layerPatchesRef.current, next);
  }

  async function setSelectedBodyOpacity(opacity: number) {
    if (!selectedBodyLayer) return;
    const value = Math.max(0, Math.min(1, opacity));
    const next = { ...layerOpacitiesRef.current, [selectedBodyLayer]: value };
    layerOpacitiesRef.current = next;
    setLayerOpacities(next);
    await renderBody(layerOrderRef.current, enabledLayers, layerPatchesRef.current, next);
  }

  async function toggleOverlapHighlight() {
    const next = !overlapHighlightEnabled;
    setOverlapHighlightEnabled(next);
    const active = [...layerOrderRef.current.filter(name => enabledLayers[name] !== false)].reverse();
    try {
      const result = await invoke<RenderCategoryResult>("render_category", {
        mappingJson: JSON.stringify(layerMapping),
        target: "body",
        enabledLayers: active,
        layerPatches: layerPatchesRef.current,
        layerOpacities: layerOpacitiesRef.current,
        overlapHighlight: next,
      });
      setBodyPreview(result.preview);
    } catch (e) { console.error("render error:", e); }
  }

  function createDefaultOpacities(order: string[]): Record<string, number> {
    const next: Record<string, number> = {};
    for (const name of order) next[name] = 0.5;
    return next;
  }

  // Drag handlers (generic)
  function getOrderRef(t: DragTarget) {
    return t === "body" ? layerOrderRef : t === "hair" ? hairLayerOrderRef : hairBackLayerOrderRef;
  }
  function onDragPointerDown(e: React.PointerEvent, idx: number, target: DragTarget = "body") {
    e.preventDefault(); (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragState.current = { target, idx, startY: e.clientY, currentIdx: idx };
    setDraggedIdx(idx); setDragTarget(target);
  }
  function onDragPointerMove(e: React.PointerEvent) {
    if (!dragState.current) return;
    const ref = getOrderRef(dragState.current.target);
    const newIdx = Math.max(0, Math.min(ref.current.length - 1,
      dragState.current.idx + Math.round((e.clientY - dragState.current.startY) / 50)));
    if (newIdx !== dragState.current.currentIdx) {
      const newOrder = [...ref.current];
      const [item] = newOrder.splice(dragState.current.currentIdx, 1);
      newOrder.splice(newIdx, 0, item);
      const t = dragState.current.target;
      if (t === "body") setLayerOrder(newOrder);
      else if (t === "hair") setHairLayerOrder(newOrder);
      else setHairBackLayerOrder(newOrder);
      dragState.current.currentIdx = newIdx; setDraggedIdx(newIdx);
    }
  }
  async function onDragPointerUp() {
    if (!dragState.current) return;
    const t = dragState.current.target;
    dragState.current = null; setDraggedIdx(null); setDragTarget(null);
    if (t === "body") await renderBody(layerOrderRef.current, enabledLayers);
    else if (t === "hair") await renderHair(hairLayerOrderRef.current, hairEnabledLayers);
    else await renderHairBack(hairBackLayerOrderRef.current, hairBackEnabledLayers);
  }

  // Zoom
  function handleWheel(e: React.WheelEvent) { e.preventDefault(); setZoom(prev => Math.max(0.1, Math.min(10, prev * (e.deltaY > 0 ? 0.9 : 1.1)))); }
  function handleMouseDown(e: React.MouseEvent) { if (e.button === 0) { setIsPanning(true); panStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }; } }
  function handleMouseMove(e: React.MouseEvent) { if (isPanning) setPan({ x: e.clientX - panStart.current.x, y: e.clientY - panStart.current.y }); }
  function handleMouseUp() { setIsPanning(false); }
  function resetZoom() { setZoom(1); setPan({ x: 0, y: 0 }); }
  function resetWorkspacePreviewZoom() {
    setWorkspacePreviewZoom(1);
    setWorkspacePreviewPan({ x: 0, y: 0 });
  }
  function handleWorkspacePreviewWheel(e: React.WheelEvent) {
    e.preventDefault();
    setWorkspacePreviewZoom(prev => Math.max(0.25, Math.min(8, prev * (e.deltaY > 0 ? 0.9 : 1.1))));
  }
  function handleWorkspacePreviewPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    // Step5でパーツ移動モードかつ個別パーツ選択中は、ドラッグ=パーツのXY補正
    const partDrag = workspaceStep === 5 && workspacePartDragMode && workspaceAdjustTarget !== "all";
    workspacePreviewDrag.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      startPan: workspacePreviewPan,
      mode: partDrag ? "part" : "pan",
      startOffsetX: workspacePartOffsetX,
      startOffsetY: workspacePartOffsetY,
    };
    setWorkspacePreviewDragging(true);
  }
  function handleWorkspacePreviewPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    if (drag.mode === "part") {
      // 画面座標→画像座標（ズーム補正）でパーツのXY補正値を更新
      const scale = Math.max(0.01, workspacePreviewZoom);
      setWorkspacePartOffsetX(Math.round(drag.startOffsetX + (e.clientX - drag.startX) / scale));
      setWorkspacePartOffsetY(Math.round(drag.startOffsetY + (e.clientY - drag.startY) / scale));
      return;
    }
    setWorkspacePreviewPan({
      x: drag.startPan.x + e.clientX - drag.startX,
      y: drag.startPan.y + e.clientY - drag.startY,
    });
  }
  function handleWorkspacePreviewPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    workspacePreviewDrag.current = null;
    setWorkspacePreviewDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    // パーツ移動ドラッグ終了時は自動で適用→プレビュー更新
    // （setStateの反映を待たず、イベント座標から最終値を直接計算して渡す）
    if (drag.mode === "part") {
      const scale = Math.max(0.01, workspacePreviewZoom);
      void applyWorkspacePartAdjustment({
        x: Math.round(drag.startOffsetX + (e.clientX - drag.startX) / scale),
        y: Math.round(drag.startOffsetY + (e.clientY - drag.startY) / scale),
      });
    }
  }
  function handleWorkspacePreviewLostPointerCapture(e: React.PointerEvent<HTMLDivElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    workspacePreviewDrag.current = null;
    setWorkspacePreviewDragging(false);
  }

  // === Step 1: Input ===
  async function loadPsd() {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "PSD", extensions: ["psd"] }] });
    if (!file) return;
    setLoading(true); setStatus("PSD読み込み中...");
    try {
      const result = await invoke<SlotLoadResult>("load_slot", { path: file });
      setLoadResult(result);
      if (Object.keys(layerMapping).length === 0) {
        const m: Record<string, string> = {};
        for (const l of result.adjustable_layers) m[l.name] = l.default_target;
        setLayerMapping(m);
      }
      setStatus(`PSD読み込み完了 (${result.detected_layers.length}レイヤー)`);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  async function proceedToHairEdit() {
    setLoading(true); setStatus("Hairレイヤー準備中...");
    try {
      const preview = await invoke<MappingPreviewResult>("get_mapping_preview", { mappingJson: JSON.stringify(layerMapping) });
      setMappingPreview(preview);
      await openHairEditorWithPreview(preview);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  async function openHairEditorWithPreview(preview: MappingPreviewResult) {
    const hairCat = preview.categories.find(c => c.target === "hair");
    if (hairCat) {
      const order = hairCat.layers.map(l => l.name);
      setHairLayerOrder(order);
      const en: Record<string, boolean> = {};
      for (const l of hairCat.layers) en[l.name] = true;
      setHairEnabledLayers(en);
      await renderHair(order, en);
    }
    const hairBackCat = preview.categories.find(c => c.target === "hair_back");
    if (hairBackCat) {
      const order = hairBackCat.layers.map(l => l.name);
      setHairBackLayerOrder(order);
      const en: Record<string, boolean> = {};
      for (const l of hairBackCat.layers) en[l.name] = true;
      setHairBackEnabledLayers(en);
      await renderHairBack(order, en);
    }
    resetZoom(); setBaseStep(2); setMode("hair_edit"); setStatus("Step 2/4: Hairレイヤーを確認");
  }

  async function openUnifiedBaseEditorWithPreview(preview: MappingPreviewResult) {
    const freeCat = preview.categories.find(c => c.target === "free");
    const layers = freeCat?.layers ?? preview.categories.flatMap(c => c.layers);
    const seen = new Set<string>();
    const order = recommendedUnifiedLayerOrder(layers
      .map(layer => layer.name)
      .filter(name => {
        if (seen.has(name)) return false;
        seen.add(name);
        return true;
      }));
    const enabled: Record<string, boolean> = {};
    for (const name of order) enabled[name] = true;
    const opacities = createDefaultOpacities(order);
    setLayerOrder(order);
    setLayerPatches([]);
    setPatchDraftSource("");
    setSelectedBodyLayer(order[0] ?? "");
    setEnabledLayers(enabled);
    setLayerOpacities(opacities);
    await renderBody(order, enabled, [], opacities);
    resetZoom();
    setBaseStep(3);
    setMode("base_edit");
    setStatus("Step 4: 全レイヤーを元の順序で調整");
  }

  function baseLayerName(name: string) {
    return name
      .toLowerCase()
      .replace(/\s+/g, "_")
      .replace(/[-_](l|r)$/u, "");
  }

  function recommendedLayerPriority(name: string) {
    const base = baseLayerName(name);
    if (base === "front_hair") return 10;
    if (base === "headwear" || base === "earwear" || base === "eyewear") return 20;
    if (base === "eyebrow") return 30;
    if (base === "eyelash") return 35;
    if (base === "irides") return 40;
    if (base === "eyewhite") return 45;
    if (base === "mouth") return 50;
    if (base === "nose") return 55;
    if (base === "face") return 60;
    if (base === "neckwear") return 65;
    if (base === "topwear" || base === "handwear" || base === "objects") return 70;
    if (base === "bottomwear" || base === "legwear" || base === "footwear") return 75;
    if (base === "ears" || base === "wings" || base === "tail") return 78;
    if (base === "neck") return 80;
    if (base === "back_hair") return 90;
    return 85;
  }

  function recommendedUnifiedLayerOrder(order: string[]) {
    return order
      .map((name, index) => ({ name, index, priority: recommendedLayerPriority(name) }))
      .sort((a, b) => a.priority - b.priority || a.index - b.index)
      .map(item => item.name);
  }

  async function applyRecommendedLayerOrder() {
    const nextOrder = [...layerOrderRef.current]
      .map((name, index) => {
        const patch = layerPatchesRef.current.find(item => item.id === name);
        const sourceName = patch?.sourceLayer ?? name;
        return { name, index, priority: recommendedLayerPriority(sourceName) };
      })
      .sort((a, b) => a.priority - b.priority || a.index - b.index)
      .map(item => item.name);
    setLayerOrder(nextOrder);
    await renderBody(nextOrder, enabledLayers, layerPatchesRef.current, layerOpacitiesRef.current);
  }

  async function proceedToBodyEdit() {
    const bodyCat = mappingPreview?.categories.find(c => c.target === "body");
    if (bodyCat) {
      const order = bodyCat.layers.map(l => l.name);
      setLayerOrder(order);
      setLayerPatches([]);
      setPatchDraftSource("");
      setSelectedBodyLayer(order[0] ?? "");
      const en: Record<string, boolean> = {};
      for (const l of bodyCat.layers) en[l.name] = true;
      const opacities = createDefaultOpacities(order);
      setEnabledLayers(en);
      setLayerOpacities(opacities);
      await renderBody(order, en, [], opacities);
    }
    resetZoom(); setBaseStep(3); setMode("base_edit"); setStatus("Step 3/4: Bodyレイヤーを調整して出力");
  }

  async function loadCorrectionPsd() {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "PSD", extensions: ["psd"] }] });
    if (!file) return;
    setLoading(true); setStatus("See-Through補正: PSD読み込み中...");
    try {
      const result = await invoke<SlotLoadResult>("load_slot", { path: file });
      setLoadResult(result);
      const preview = await invoke<MappingPreviewResult>("get_all_layers_preview");
      setMappingPreview(preview);
      const freeCat = preview.categories.find(c => c.target === "free");
      const order = freeCat?.layers.map(l => l.name) ?? [];
      const enabled: Record<string, boolean> = {};
      const opacities: Record<string, number> = {};
      for (const name of order) {
        enabled[name] = true;
        opacities[name] = 1;
      }
      setLayerOrder(order);
      setEnabledLayers(enabled);
      setLayerPatches([]);
      setLayerOpacities(opacities);
      setSelectedBodyLayer(order[0] ?? "");
      setPatchDraftSource("");
      setOverlapHighlightEnabled(false);
      setCorrectionOutputPath("");
      await renderBody(order, enabled, [], opacities, false);
      resetZoom();
      setStatus(`See-Through補正: ${result.detected_layers.length}レイヤーを読み込みました`);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  async function addCorrectionLayerImage() {
    setError("");
    const file = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp"] }],
    });
    if (!file) return;

    const fileName = String(file).split(/[\\/]/).pop() ?? "layer.png";
    const defaultName = fileName.replace(/\.[^.]+$/, "").trim() || "layer";
    const inputName = window.prompt("追加するレイヤー名", defaultName);
    if (!inputName) return;

    setLoading(true);
    setStatus("See-Through補正: 追加レイヤー読み込み中...");
    try {
      const imported = await invoke<ImportCorrectionLayerResult>("import_correction_layer", {
        path: file,
        layerName: inputName,
      });
      const preview = await invoke<MappingPreviewResult>("get_all_layers_preview");
      setMappingPreview(preview);

      const freeCat = preview.categories.find(c => c.target === "free");
      const availableNames = freeCat?.layers.map(l => l.name) ?? [];
      const previousOrder = layerOrder.filter(name => availableNames.includes(name));
      const missing = availableNames.filter(name => !previousOrder.includes(name));
      const nextOrder = imported.layer_name
        ? [imported.layer_name, ...previousOrder.filter(name => name !== imported.layer_name), ...missing.filter(name => name !== imported.layer_name)]
        : [...previousOrder, ...missing];
      const nextEnabled = { ...enabledLayers, [imported.layer_name]: true };
      const nextOpacities = { ...layerOpacities, [imported.layer_name]: 1 };

      setLayerOrder(nextOrder);
      setEnabledLayers(nextEnabled);
      setLayerOpacities(nextOpacities);
      setSelectedBodyLayer(imported.layer_name);
      await renderBody(nextOrder, nextEnabled, layerPatches, nextOpacities);
      setStatus(`See-Through補正: ${imported.layer_name} を追加しました`);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  async function handleExportCorrection() {
    setError("");
    const output = await save({
      defaultPath: "corrected.png",
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    if (!output) return;
    setLoading(true); setStatus("See-Through補正: PNG出力中...");
    try {
      const activeOrder = layerOrder.filter(name => enabledLayers[name] !== false).reverse();
      const result = await invoke<ExportCorrectedLayerResult>("export_corrected_layer", {
        outputPath: output,
        enabledLayers: activeOrder,
        layerPatches,
        layerOpacities,
      });
      setCorrectionOutputPath(result.output_path);
      setStatus(`See-Through補正完了: ${result.output_path}`);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  // === Step 3A: base_edit - create base ===
  async function handleCreateBase() {
    setError("");
    const workspacePath = expressionWorkspace?.workPath ?? "";
    if (workspacePath) {
      setLoading(true);
      setStatus("素体を作成しています...");
      try {
        // 素体合成は必ず source の分解結果（source.psd）を対象にする。
        // 直前にどの画像を分解したかというグローバル状態に依存させない
        // （eyes-closed分解が残って素体の目が閉じる問題の根治）
        await invoke("load_codex_source_see_through", { jobPath: workspacePath });
        const activeOrder = layerOrder.filter(name => enabledLayers[name] !== false);
        const result = await invoke<CreateBaseResult>("create_base", {
          mappingJson: JSON.stringify(layerMapping),
          // 素体の目を元画像の画素そのまま（exact_eye経路）にするため source を渡す
          originalImagePath: workspaceFiles.source ?? "",
          baseEyeSlot: "eye_open",
          baseMouthSlot: "mouth_closed",
          bodyLayerOrder: activeOrder,
          bodyLayerPatches: layerPatches,
          hairLayerOrder: [] as string[],
          hairBackLayerOrder: [] as string[],
          outputPath: "",
          chestMaskPng: chestMaskDataUrl,
        });
        await invoke<SaveCodexBasePartsResult>("save_codex_base_parts", {
          jobPath: workspacePath,
        });
        await refreshWorkspaceCompositePreview().catch(() => null);
        setBaseResult(result);
        setBaseStep(4);
        await setWorkspaceStepAndPersist(5);
        setMode("workspace");
        setStatus("素体を作業フォルダに保存しました。差分位置調整へ進めます。");
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
      return;
    }
    const dir = await open({ multiple: false, directory: true, title: "素体の出力先フォルダを選択" });
    if (!dir) return;
    setLoading(true); setStatus("素体を作成中...");
    try {
      const activeOrder = layerOrder.filter(name => enabledLayers[name] !== false);
      const activeHairOrder = hairLayerOrder.filter(name => hairEnabledLayers[name] !== false);
      const activeHairBackOrder = hairBackLayerOrder.filter(name => hairBackEnabledLayers[name] !== false);
      const result = await invoke<CreateBaseResult>("create_base", {
        mappingJson: JSON.stringify(layerMapping), originalImagePath: "",
        baseEyeSlot: "eye_open", baseMouthSlot: "mouth_closed",
        bodyLayerOrder: activeOrder, bodyLayerPatches: layerPatches, hairLayerOrder: activeHairOrder,
        hairBackLayerOrder: activeHairBackOrder, outputPath: dir,
        chestMaskPng: chestMaskDataUrl,
      });
      setBaseResult(result);
      setBaseStep(4);
      setStatus(`素体作成完了: ${result.file_count}ファイル出力`);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  // === Step 3B: interp ===
  function resetMouthMaskForSlot(slotKey: string) {
    const affected = visiblePairs.filter(pair =>
      pair.closed.key === slotKey || pair.open.key === slotKey
    );
    if (affected.length === 0) return;
    const affectedOpenKeys = new Set(affected.map(pair => pair.open.key));
    setMouthMaskPreviews(prev => {
      const next = { ...prev };
      for (const key of affectedOpenKeys) delete next[key];
      return next;
    });
    setMouthMaskPreviewing(prev => {
      const next = { ...prev };
      for (const key of affectedOpenKeys) delete next[key];
      return next;
    });
    setMouthPreviewZooms(prev => {
      const next = { ...prev };
      for (const key of affectedOpenKeys) next[key] = 1;
      return next;
    });
    setMouthPreviewPans(prev => {
      const next = { ...prev };
      for (const key of affectedOpenKeys) next[key] = { x: 0, y: 0 };
      return next;
    });
    setActivePreviewKey(prev => affected.some(pair => pair.name === prev || pair.open.key === prev) ? "" : prev);
  }

  async function pickInterpPsd(slot: string) {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "PSD", extensions: ["psd"] }] });
    if (file) {
      setInterpPaths(prev => ({ ...prev, [slot]: file }));
      setCompletedDiffs([]);
      setDiffPreviews([]);
      setDiffPreviewTick(0);
      setDiffPreviewZoom(1);
      setInterpStep(1);
      resetMouthMaskForSlot(slot);
    }
  }

  async function pickInterpOriginal(slotKey: string) {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "画像", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (file) {
      setInterpOriginals(prev => ({ ...prev, [slotKey]: file }));
      setCompletedDiffs([]);
      setDiffPreviews([]);
      setDiffPreviewTick(0);
      setDiffPreviewZoom(1);
      resetMouthMaskForSlot(slotKey);
    }
  }

  function getMouthMaskSetting(slotKey: string): MouthMaskSetting {
    return mouthMaskSettings[slotKey] ?? { dilate: 15, blur: 0 };
  }

  async function previewMouthMask(pair: InterpPair) {
    setError("");
    const openOriginal = interpOriginals[pair.open.key];
    const closedPsd = interpPaths[pair.closed.key];
    if (!openOriginal || !closedPsd) return;
    const setting = getMouthMaskSetting(pair.open.key);

    setMouthMaskPreviewing(prev => ({ ...prev, [pair.open.key]: true }));
    try {
      // Set the canvas size from the closed PSD before resizing the original image for SAM3.
      await invoke<SlotLoadResult>("load_slot", { path: closedPsd });
      setStatus(`${pair.label}: SAM3口マスク確認中...`);
      const result = await invoke<OriginalImageResult>("load_original_image", {
        path: openOriginal,
        mouthMaskDilateRadius: setting.dilate,
        mouthMaskBlurRadius: setting.blur,
      });
      if (result.mouth_preview) {
        setMouthMaskPreviews(prev => ({ ...prev, [pair.open.key]: result.mouth_preview! }));
        setActivePreviewKey(pair.name);
        setMouthPreviewZooms(prev => ({ ...prev, [pair.open.key]: 1 }));
        setMouthPreviewPans(prev => ({ ...prev, [pair.open.key]: { x: 0, y: 0 } }));
        setStatus(`${pair.label}: 口マスクを確認しました`);
      } else {
        setError(`${pair.label}: 口マスクを取得できませんでした`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setMouthMaskPreviewing(prev => ({ ...prev, [pair.open.key]: false }));
    }
  }

  function updateMouthMaskSetting(slotKey: string, patch: Partial<MouthMaskSetting>) {
    const current = getMouthMaskSetting(slotKey);
    const next = { ...current, ...patch };
    setMouthMaskSettings(prev => ({ ...prev, [slotKey]: next }));
    setCompletedDiffs([]);
    setDiffPreviews([]);
    setDiffPreviewTick(0);
    setDiffPreviewZoom(1);

    if (!mouthMaskPreviews[slotKey] || !interpOriginals[slotKey]) return;
    if (mouthMaskUpdateTimers.current[slotKey]) {
      window.clearTimeout(mouthMaskUpdateTimers.current[slotKey]);
    }
    mouthMaskUpdateTimers.current[slotKey] = window.setTimeout(() => {
      delete mouthMaskUpdateTimers.current[slotKey];
      void refreshMouthMaskPreview(slotKey, next);
    }, 80);
  }

  async function refreshMouthMaskPreview(slotKey: string, setting: MouthMaskSetting = getMouthMaskSetting(slotKey)) {
    const original = interpOriginals[slotKey];
    if (!original) return;
    setMouthMaskPreviewing(prev => ({ ...prev, [slotKey]: true }));
    try {
      const result = await invoke<MouthMaskPreviewResult>("update_mouth_mask_preview", {
        path: original,
        mouthMaskDilateRadius: setting.dilate,
        mouthMaskBlurRadius: setting.blur,
      });
      setMouthMaskPreviews(prev => ({ ...prev, [slotKey]: result.mouth_preview }));
      setActivePreviewKey(prev => prev || visiblePairs.find(pair => pair.open.key === slotKey)?.name || "");
    } catch (e) {
      setError(String(e));
    } finally {
      setMouthMaskPreviewing(prev => ({ ...prev, [slotKey]: false }));
    }
  }

  function setMouthPreviewZoom(slotKey: string, zoom: number) {
    setMouthPreviewZooms(prev => {
      return { ...prev, [slotKey]: Math.max(1, Math.min(5, zoom)) };
    });
  }

  function onMouthPreviewPointerDown(e: React.PointerEvent, slotKey: string) {
    if (!mouthMaskPreviews[slotKey]) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    mouthPreviewDrag.current = {
      key: slotKey,
      startX: e.clientX,
      startY: e.clientY,
      startPan: mouthPreviewPans[slotKey] ?? { x: 0, y: 0 },
    };
  }

  function onMouthPreviewPointerMove(e: React.PointerEvent) {
    const drag = mouthPreviewDrag.current;
    if (!drag) return;
    setMouthPreviewPans(prev => ({
      ...prev,
      [drag.key]: {
        x: drag.startPan.x + e.clientX - drag.startX,
        y: drag.startPan.y + e.clientY - drag.startY,
      },
    }));
  }

  function onMouthPreviewPointerUp() {
    mouthPreviewDrag.current = null;
  }

  function resetMouthPreviewView(slotKey: string) {
    setMouthPreviewZooms(prev => ({ ...prev, [slotKey]: 1 }));
    setMouthPreviewPans(prev => ({ ...prev, [slotKey]: { x: 0, y: 0 } }));
  }

  // Pairs for current interp mode
  const visiblePairs = diffTarget === "eye"
    ? EYE_PAIRS
    : (mouthMode === "single" ? MOUTH_PAIRS_SINGLE : MOUTH_PAIRS_VOWELS);

  function pingPongFrameIndex(tick: number, frameLength: number) {
    if (frameLength <= 1) return 0;
    const cycle = frameLength * 2 - 2;
    const pos = tick % cycle;
    return pos < frameLength ? pos : cycle - pos;
  }

  function isPairReady(pair: typeof EYE_PAIRS[0]) {
    const hasPsds = !!interpPaths[pair.closed.key] && !!interpPaths[pair.open.key];
    if (diffTarget === "eye") return hasPsds;
    return hasPsds && !!interpOriginals[pair.closed.key] && !!interpOriginals[pair.open.key];
  }

  const canGenerate = visiblePairs.some(isPairReady);
  const anyMouthMaskPreviewing = Object.values(mouthMaskPreviewing).some(Boolean);
  const readyPairsForMask = visiblePairs.filter(pair => isPairReady(pair));
  const missingMouthMaskCount = readyPairsForMask.filter(pair => !mouthMaskPreviews[pair.open.key]).length;
  const allReadyPairsHaveMasks = diffTarget !== "mouth" || (readyPairsForMask.length > 0 && readyPairsForMask.every(pair => !!mouthMaskPreviews[pair.open.key]));
  const canOpenInterpStep = (step: InterpStep) => {
    if (step === 1) return true;
    if (step === 2) return canGenerate;
    if (step === 3) return diffTarget === "eye" ? canGenerate : allReadyPairsHaveMasks;
    return diffPreviews.length > 0;
  };
  function goInterpStep(step: InterpStep) {
    if (canOpenInterpStep(step)) setInterpStep(step);
  }
  const activeMaskPair = visiblePairs.find(pair => pair.name === activePreviewKey || pair.open.key === activePreviewKey)
    ?? visiblePairs.find(pair => !!mouthMaskPreviews[pair.open.key])
    ?? visiblePairs[0];
  const activeMaskSlot = activeMaskPair?.open.key ?? "";
  const activeMaskSetting = activeMaskSlot ? getMouthMaskSetting(activeMaskSlot) : { dilate: 15, blur: 0 };
  const activeMaskPreview = activeMaskSlot ? mouthMaskPreviews[activeMaskSlot] : "";
  const activeMaskZoom = activeMaskSlot ? (mouthPreviewZooms[activeMaskSlot] ?? 1) : 1;
  const activeMaskPan = activeMaskSlot ? (mouthPreviewPans[activeMaskSlot] ?? { x: 0, y: 0 }) : { x: 0, y: 0 };
  const activeDiffPreview = diffPreviews.find(preview => preview.pairName === activePreviewKey) ?? diffPreviews[0];
  const hasInterpWork = Object.keys(interpPaths).length > 0
    || Object.keys(interpOriginals).length > 0
    || Object.keys(mouthMaskPreviews).length > 0
    || diffPreviews.length > 0
    || completedDiffs.length > 0;

  async function handleGenerateAll() {
    setError("");
    const dir = outputPath || await open({ multiple: false, directory: true, title: "出力先フォルダを選択" });
    if (!dir) return;
    setOutputPath(dir);
    setInterpGenerating(true);
    setCompletedDiffs([]);
    setDiffPreviews([]);
    setDiffPreviewTick(0);

    try {
      const readyPairs = visiblePairs.filter(isPairReady);
      const mappingJson = JSON.stringify(layerMapping);

      for (const pair of readyPairs) {
        // Load the "closed" PSD as base (frame_001 = closed)
        setStatus(`${pair.label}: ベース読み込み中...`);
        const result = await invoke<SlotLoadResult>("load_slot", { path: interpPaths[pair.closed.key] });
        if (Object.keys(layerMapping).length === 0) {
          const m: Record<string, string> = {};
          for (const l of result.adjustable_layers) m[l.name] = l.default_target;
          setLayerMapping(m);
        }

        const pairDiffType = pair.name.startsWith("mouth") ? "mouth" : "eye";

        // Load original image for SAM3 mouth extraction (use "open" original - open mouth is easier to detect)
        const openOriginal = interpOriginals[pair.open.key];
        if (pairDiffType === "mouth" && openOriginal) {
          const setting = getMouthMaskSetting(pair.open.key);
          if (mouthMaskPreviews[pair.open.key]) {
            setStatus(`${pair.label}: 口マスク設定を適用中...`);
            await invoke<MouthMaskPreviewResult>("update_mouth_mask_preview", {
              path: openOriginal,
              mouthMaskDilateRadius: setting.dilate,
              mouthMaskBlurRadius: setting.blur,
            });
          } else {
            setStatus(`${pair.label}: SAM3で口を検出中...`);
            await invoke<OriginalImageResult>("load_original_image", {
              path: openOriginal,
              mouthMaskDilateRadius: setting.dilate,
              mouthMaskBlurRadius: setting.blur,
            });
          }
        }

        // Determine base slot names
        const baseSlotEye = pairDiffType === "eye" ? "eye_closed" : "eye_closed";
        const baseSlotMouth = pairDiffType === "mouth" ? "mouth_closed" : "mouth_closed";

        await invoke<CreateBaseResult>("create_base", {
          mappingJson,
          originalImagePath: pairDiffType === "mouth" ? (interpOriginals[pair.closed.key] || "") : "",
          baseEyeSlot: baseSlotEye, baseMouthSlot: baseSlotMouth,
          bodyLayerOrder: [] as string[], bodyLayerPatches: [], hairLayerOrder: [] as string[],
          hairBackLayerOrder: [] as string[], outputPath: "",
          chestMaskPng: null,
        });

        // Generate diff: open PSD against closed base (frame_N = open)
        setStatus(`${pair.label}: フレーム生成中...`);
        setProgress({ current: 0, total: frameCount, pair_name: pair.label });

        const diffResult = await invoke<CreateDiffResult>("create_diff", {
          path: interpPaths[pair.open.key],
          diffType: pairDiffType,
          slotName: pair.name,
          frameCount,
          outputPath: dir,
          originalImagePath: pairDiffType === "mouth" ? (interpOriginals[pair.open.key] || "") : "",
        });
        setCompletedDiffs(prev => [...prev, diffResult.pair_name]);
        setDiffPreviews(prev => [...prev, {
          pairName: diffResult.pair_name,
          label: pair.label,
          frames: diffResult.previews.length > 0 ? diffResult.previews : [diffResult.preview],
        }]);
        setActivePreviewKey(diffResult.pair_name);
      }

      setInterpStep(4);
      setStatus(`生成完了: ${readyPairs.length}パーツ`);
    } catch (e) { setError(String(e)); }
    finally { setInterpGenerating(false); }
  }

  async function openOutputFolder() {
    if (outputPath) try { await revealItemInDir(outputPath); } catch (_) {}
  }

  async function handleCreateMouthMasks() {
    setError("");
    const targets = visiblePairs.filter(pair => isPairReady(pair) && !mouthMaskPreviews[pair.open.key]);
    if (targets.length === 0) return;
    setInterpGenerating(true);
    try {
      for (const pair of targets) {
        await previewMouthMask(pair);
      }
      setInterpStep(2);
    } finally {
      setInterpGenerating(false);
    }
  }

  function discardMouthMask(pair: typeof EYE_PAIRS[0]) {
    if (!mouthMaskPreviews[pair.open.key]) return;
    const ok = window.confirm(`${pair.label} の口マスクを破棄します。生成済みプレビューもクリアされます。続けますか？`);
    if (!ok) return;

    setMouthMaskPreviews(prev => {
      const next = { ...prev };
      delete next[pair.open.key];
      return next;
    });
    setMouthPreviewZooms(prev => {
      const next = { ...prev };
      delete next[pair.open.key];
      return next;
    });
    setMouthPreviewPans(prev => {
      const next = { ...prev };
      delete next[pair.open.key];
      return next;
    });
    setCompletedDiffs([]);
    setDiffPreviews([]);
    setDiffPreviewTick(0);
    setDiffPreviewZoom(1);
    setActivePreviewKey(pair.name);
    setStatus(`${pair.label}: 口マスクを破棄しました`);
  }

  async function startExpressionWorkspace(kind: "new" | "resume") {
    setError("");
    const selected = await open({
      multiple: false,
      directory: true,
      title: kind === "new" ? "作業フォルダを選択" : "既存の作業フォルダを選択",
    });
    const workPath = typeof selected === "string" ? selected : null;
    if (!workPath) return;
    setWorkspaceBusy(true);
    try {
      const workspace = await invoke<ExpressionWorkspaceResult>(
        kind === "new" ? "create_expression_workspace" : "load_expression_workspace",
        { workPath },
      );
      setExpressionWorkspace(workspace);
      const nextFiles = {
        ...(workspace.project.sourceImagePath ? { source: workspace.project.sourceImagePath } : {}),
        ...(workspace.project.referenceImagePath ? { reference: workspace.project.referenceImagePath } : {}),
      };
      setWorkspaceFiles(nextFiles);
      await loadWorkspaceImagePreviews(nextFiles);
      setWorkspaceGeneratedStatus(null);
      setWorkspaceExtractResult(null);
      setWorkspaceCompositePreview(null);
      setWorkspaceRifeResult(null);
      setWorkspaceRifePreview(null);
      setWorkspaceStep(Math.min(Math.max(workspace.project.currentStep || 1, 1), 7) as WorkspaceStep);
      setMode("workspace");
      setStatus(`作業フォルダ: ${workspace.workPath}`);
      if (kind === "resume") {
        await restoreWorkspaceProgress(workspace);
      }
      await refreshWorkspaceSeeThroughStatus();
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function loadMotionLabPartsFromDir(dir: string) {
    setError("");
    setLoading(true);
    try {
      const result = await invoke<MotionLabPartsResult>("load_motion_lab_parts", { dir });
      setMotionLabParts(result);
      setMotionLabPlaying(true);
      setMotionLabManifestPath("");
      setMotionLabProfilePath("");
      setStatus(`Motion Preview Lab: ${result.sourceDir}`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoading(false);
    }
  }

  async function pickMotionLabPartsDir() {
    const selected = await open({
      multiple: false,
      directory: true,
      title: "04_spritalk_parts フォルダを選択",
    });
    const dir = typeof selected === "string" ? selected : null;
    if (!dir) return;
    await loadMotionLabPartsFromDir(dir);
  }

  async function loadCurrentWorkspaceMotionLabParts() {
    const dir = workspaceRifeResult?.outputPath || expressionWorkspace?.spritalkPartsPath;
    if (!dir) return;
    await loadMotionLabPartsFromDir(dir);
  }

  // エフェクト行のインラインスライダー定義（チェックON時のみ表示される強さ調整）
  const motionLabPercentFormat = (value: number) => `${Math.round(value * 100)}%`;
  const motionLabEffectSliders: Partial<Record<MotionLabEffectKey, {
    value: number;
    min: number;
    max: number;
    step: number;
    set: (value: number) => void;
    format: (value: number) => string;
  }>> = {
    breath: { value: motionLabBreathAmplitude, min: 0, max: 1.6, step: 0.05, set: setMotionLabBreathAmplitude, format: motionLabPercentFormat },
    bodySway: { value: motionLabBodySwayAmplitude, min: 0, max: 1.8, step: 0.05, set: setMotionLabBodySwayAmplitude, format: motionLabPercentFormat },
    pyoko: { value: motionLabPyokoBounce, min: 0, max: 12, step: 0.5, set: setMotionLabPyokoBounce, format: value => `${value.toFixed(1)}px` },
    hairMotion: { value: motionLabHairMotionStrength, min: 0, max: 2, step: 0.05, set: setMotionLabHairMotionStrength, format: motionLabPercentFormat },
    hairBack: { value: motionLabHairBackScale, min: 0, max: 1.5, step: 0.05, set: setMotionLabHairBackScale, format: motionLabPercentFormat },
    parallax: { value: motionLabParallaxScale, min: 0, max: 1.5, step: 0.05, set: setMotionLabParallaxScale, format: motionLabPercentFormat },
    glance: { value: motionLabGlanceStrength, min: 0, max: 2, step: 0.05, set: setMotionLabGlanceStrength, format: motionLabPercentFormat },
    gaze: { value: motionLabGazeStrength, min: 0, max: 2, step: 0.05, set: setMotionLabGazeStrength, format: motionLabPercentFormat },
    blink: { value: motionLabBlinkRate, min: 0.3, max: 2.5, step: 0.05, set: setMotionLabBlinkRate, format: value => `×${value.toFixed(2)}` },
    arm: { value: motionLabArmSwayAmp, min: 0, max: 3, step: 0.1, set: setMotionLabArmSwayAmp, format: motionLabPercentFormat },
    lift: { value: motionLabLiftStrength, min: 0, max: 2, step: 0.05, set: setMotionLabLiftStrength, format: motionLabPercentFormat },
    chest: { value: motionLabChestMax, min: 0, max: 12, step: 0.5, set: setMotionLabChestMax, format: value => `${value.toFixed(1)}px` },
    earTwitch: { value: motionLabEarTwitchScale, min: 0, max: 2, step: 0.05, set: setMotionLabEarTwitchScale, format: motionLabPercentFormat },
  };

  function buildMotionLabManifest(): MotionLabManifest | null {
    if (!motionLabParts) return null;
    return {
      schema: "pachipakugen.motionPreview.v1",
      sourcePartsDir: motionLabParts.sourceDir,
      createdAt: new Date().toISOString(),
      methods: {
        baseline: { enabled: true },
        lipTimelineSmoother: {
          enabled: motionLabMethod !== "baseline",
          method: motionLabMethod,
          attackMs: motionLabAttackMs,
          releaseMs: motionLabReleaseMs,
          crossfadeMs: motionLabCrossfadeMs,
          restBias: motionLabRestBias,
          shapeSmoothing: motionLabShapeSmoothing,
          bridgeBias: motionLabBridgeBias,
        },
        layeredSpring: {
          enabled: true,
          layerMode: motionLabLayerMode,
          preset: motionLabPreset,
          breathAmplitude: motionLabBreathAmplitude,
          bodySwayAmplitude: motionLabBodySwayAmplitude,
          hairFrontDelay: motionLabHairFrontDelay,
          hairBackDelay: motionLabHairBackDelay,
        },
        physicsLab: {
          hairK: motionLabHairK,
          hairC: motionLabHairC,
          hairWind: motionLabHairWind,
          hairDrive: motionLabHairDrive,
          armEnabled: motionLabEffects.arm,
          armMaxAngle: motionLabArmMaxAngle,
          armSwayAmp: motionLabArmSwayAmp,
          armPivotRatio: motionLabArmPivotRatio,
          liftEnabled: motionLabEffects.lift,
          chestMax: motionLabChestMax,
          hairBackScale: motionLabHairBackScale,
          hairEngine: motionLabHairEngine,
          hairWaveStrength: motionLabHairWaveStrength,
          earTwitch: motionLabEffects.earTwitch,
          pyokoBounce: motionLabPyokoBounce,
          randomGlance: motionLabEffects.glance,
          effects: motionLabEffects,
          engineFamily: motionLabEngineFamily,
          hairMotionStrength: motionLabHairMotionStrength,
          glanceStrength: motionLabGlanceStrength,
          gazeStrength: motionLabGazeStrength,
          blinkRate: motionLabBlinkRate,
          liftStrength: motionLabLiftStrength,
          earTwitchScale: motionLabEarTwitchScale,
          pivots: motionLabPivots,
          rangesDeg: motionLabRangesDeg,
          swingScale: motionLabSwingScale,
          parallaxScale: motionLabParallaxScale,
          strandsEnabled: motionLabStrandsEnabled,
          armBehindBody: motionLabArmBehindBody,
        },
      },
      timeline: { type: "builtInVowelTest" },
      review: {
        verdict: motionLabVerdict,
        note: motionLabReviewNote,
        scores: motionLabReviewScores,
      },
    };
  }

  function buildSpritalkMotionProfile(): SpritalkMotionProfile | null {
    if (!motionLabParts) return null;
    const preset = MOTION_LAB_PRESET_FACTORS[motionLabPreset];
    const lipSyncRenderer = motionLabMethod === "baseline"
      ? "directLayerSwitch"
      : motionLabMethod === "bridge"
        ? "neutralBridgeOpacityBlend"
        : "smoothedFrameStepper";
    const layerRenderer = motionLabLayerMode === "mesh" ? "stripWarpExtension" : "existingProceduralAnimator";
    return {
      schema: "spritalk.motionProfile.v2",
      sourcePartsDir: motionLabParts.sourceDir,
      createdAt: new Date().toISOString(),
      generatedBy: "PachiPakuGen Motion Lab",
      blink: {
        mode: "keepExisting",
      },
      lipSync: {
        method: motionLabMethod,
        attackMs: motionLabAttackMs,
        releaseMs: motionLabReleaseMs,
        crossfadeMs: motionLabCrossfadeMs,
        restBias: motionLabRestBias,
        shapeSmoothing: motionLabShapeSmoothing,
        bridgeBias: motionLabBridgeBias,
      },
      layerMotion: {
        mode: motionLabLayerMode,
        preset: motionLabPreset,
        breathAmplitude: motionLabBreathAmplitude,
        bodySwayAmplitude: motionLabBodySwayAmplitude,
        hairFrontDelayMs: Math.round(motionLabHairFrontDelay * 1000),
        hairBackDelayMs: Math.round(motionLabHairBackDelay * 1000),
      },
      spritalkProceduralAnimation: {
        breathing: {
          enabled: motionLabBreathAmplitude > 0,
          amplitude: Number((4.5 * motionLabBreathAmplitude * preset.breath).toFixed(2)),
          speed: 0.5,
        },
        idleSway: {
          enabled: motionLabBodySwayAmplitude > 0,
          amplitudeX: Number((2.4 * motionLabBodySwayAmplitude * preset.body).toFixed(2)),
          amplitudeY: Number((1.4 * motionLabBodySwayAmplitude * preset.body).toFixed(2)),
          speed: 0.9,
          reduceOnSpeech: true,
        },
        hairSway: {
          enabled: motionLabLayerMode !== "simple",
          amplitude: Number((2.5 * motionLabBodySwayAmplitude * preset.hair).toFixed(2)),
          speed: Number(clamp(0.95 - motionLabHairFrontDelay, 0.3, 1.2).toFixed(2)),
          rotationAmount: Number((0.009 * preset.hair).toFixed(4)),
        },
        hairBackSway: {
          enabled: motionLabLayerMode !== "simple",
          amplitude: Number((2.1 * motionLabBodySwayAmplitude * preset.hair).toFixed(2)),
          speed: Number(clamp(0.82 - motionLabHairBackDelay, 0.25, 1.0).toFixed(2)),
          rotationAmount: Number((0.007 * preset.hair).toFixed(4)),
        },
      },
      // ===== v2 additive フィールド（docs/motion-lab-integration.md §1） =====
      physics: {
        hair: {
          mode: "chain",
          segments: MOTION_LAB_HAIR_SEGMENTS,
          k: motionLabHairK,
          c: motionLabHairC,
          wind: motionLabHairWind,
          drive: motionLabHairDrive,
          strands: motionLabStrandsEnabled,
        },
        arm: {
          enabled: motionLabEffects.arm,
          k: MOTION_LAB_ARM_DEFAULTS.k,
          c: MOTION_LAB_ARM_DEFAULTS.c,
          maxAngle: motionLabArmMaxAngle,
          coupling: MOTION_LAB_ARM_DEFAULTS.coupling,
          noise: MOTION_LAB_ARM_DEFAULTS.noise,
          lift: {
            enabled: motionLabEffects.lift,
            coupling: MOTION_LAB_ARM_DEFAULTS.lift.coupling,
            bounce: MOTION_LAB_ARM_DEFAULTS.lift.bounce,
            max: MOTION_LAB_ARM_DEFAULTS.lift.max,
          },
        },
        chest: {
          k: MOTION_LAB_CHEST_DEFAULTS.k,
          c: MOTION_LAB_CHEST_DEFAULTS.c,
          max: motionLabChestMax,
        },
        sway: {
          k: MOTION_LAB_SWAY_DEFAULTS.k,
          c: MOTION_LAB_SWAY_DEFAULTS.c,
        },
        parallax: {
          shiftRatio: MOTION_LAB_PARALLAX_DEFAULTS.shiftRatio,
          shearMax: MOTION_LAB_PARALLAX_DEFAULTS.shearMax,
          scale: motionLabParallaxScale,
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
        verdict: motionLabVerdict,
        note: motionLabReviewNote,
        scores: motionLabReviewScores,
      },
    };
  }

  function applyMotionLabManifest(manifest: MotionLabManifest) {
    const lip = manifest.methods?.lipTimelineSmoother;
    const spring = manifest.methods?.layeredSpring;
    if (lip) {
      setMotionLabMethod(lip.method ? validMotionLabMethod(lip.method) : lip.enabled === false ? "baseline" : "smooth");
      if (typeof lip.attackMs === "number") setMotionLabAttackMs(clamp(Math.round(lip.attackMs), 40, 180));
      if (typeof lip.releaseMs === "number") setMotionLabReleaseMs(clamp(Math.round(lip.releaseMs), 80, 260));
      if (typeof lip.crossfadeMs === "number") setMotionLabCrossfadeMs(clamp(Math.round(lip.crossfadeMs), 0, 120));
      if (typeof lip.restBias === "number") setMotionLabRestBias(clamp(lip.restBias, 0, 1));
      if (typeof lip.shapeSmoothing === "number") setMotionLabShapeSmoothing(clamp(lip.shapeSmoothing, 0, 1));
      if (typeof lip.bridgeBias === "number") setMotionLabBridgeBias(clamp(lip.bridgeBias, 0, 0.85));
    }
    if (spring) {
      if (spring.layerMode) setMotionLabLayerMode(validMotionLabLayerMode(spring.layerMode));
      setMotionLabPreset(validMotionLabPreset(spring.preset));
      if (typeof spring.breathAmplitude === "number") setMotionLabBreathAmplitude(clamp(spring.breathAmplitude, 0, 1.6));
      if (typeof spring.bodySwayAmplitude === "number") setMotionLabBodySwayAmplitude(clamp(spring.bodySwayAmplitude, 0, 1.8));
      if (typeof spring.hairFrontDelay === "number") setMotionLabHairFrontDelay(clamp(spring.hairFrontDelay, 0, 0.6));
      if (typeof spring.hairBackDelay === "number") setMotionLabHairBackDelay(clamp(spring.hairBackDelay, 0, 0.8));
    }
    const physics = manifest.methods?.physicsLab;
    if (physics) {
      if (typeof physics.hairK === "number") setMotionLabHairK(clamp(physics.hairK, 10, 200));
      if (typeof physics.hairC === "number") setMotionLabHairC(clamp(physics.hairC, 1, 30));
      if (typeof physics.hairWind === "number") setMotionLabHairWind(clamp(physics.hairWind, 0, 0.06));
      if (typeof physics.hairDrive === "number") setMotionLabHairDrive(clamp(physics.hairDrive, 0, 0.2));
      if (typeof physics.armEnabled === "boolean") setMotionLabEffect("arm", physics.armEnabled);
      if (typeof physics.armMaxAngle === "number") setMotionLabArmMaxAngle(clamp(physics.armMaxAngle, 0, 0.6));
      if (typeof physics.armSwayAmp === "number") setMotionLabArmSwayAmp(clamp(physics.armSwayAmp, 0, 3));
      if (typeof physics.armPivotRatio === "number") setMotionLabArmPivotRatio(clamp(physics.armPivotRatio, 0, 0.6));
      if (typeof physics.liftEnabled === "boolean") setMotionLabEffect("lift", physics.liftEnabled);
      if (typeof physics.chestMax === "number") setMotionLabChestMax(clamp(physics.chestMax, 0, 8));
      if (typeof physics.hairBackScale === "number") setMotionLabHairBackScale(clamp(physics.hairBackScale, 0, 1.5));
      if (typeof physics.earTwitch === "boolean") setMotionLabEffect("earTwitch", physics.earTwitch);
      if (physics.hairEngine === "spring" || physics.hairEngine === "wave") setMotionLabHairEngine(physics.hairEngine);
      if (typeof physics.pyokoBounce === "number") setMotionLabPyokoBounce(clamp(physics.pyokoBounce, 0, 12));
      if (typeof physics.randomGlance === "boolean") setMotionLabEffect("glance", physics.randomGlance);
      if (physics.engineFamily === "rotejin" || physics.engineFamily === "hachigoni") {
        setMotionLabEngineFamily(physics.engineFamily);
      }
      if (typeof physics.hairMotionStrength === "number") setMotionLabHairMotionStrength(clamp(physics.hairMotionStrength, 0, 2));
      if (typeof physics.glanceStrength === "number") setMotionLabGlanceStrength(clamp(physics.glanceStrength, 0, 2));
      if (typeof physics.gazeStrength === "number") setMotionLabGazeStrength(clamp(physics.gazeStrength, 0, 2));
      if (typeof physics.blinkRate === "number") setMotionLabBlinkRate(clamp(physics.blinkRate, 0.3, 2.5));
      if (typeof physics.liftStrength === "number") setMotionLabLiftStrength(clamp(physics.liftStrength, 0, 2));
      if (typeof physics.earTwitchScale === "number") setMotionLabEarTwitchScale(clamp(physics.earTwitchScale, 0, 2));
      // v3: エフェクト一括ON/OFF（旧フィールドより後に適用して優先させる）
      if (physics.effects && typeof physics.effects === "object") {
        setMotionLabEffects(prev => {
          const next = { ...prev };
          for (const def of MOTION_LAB_EFFECT_DEFS) {
            const value = physics.effects?.[def.key];
            if (typeof value === "boolean") next[def.key] = value;
          }
          return next;
        });
      }
      if (typeof physics.hairWaveStrength === "number") setMotionLabHairWaveStrength(clamp(physics.hairWaveStrength, 0, 2));
      if (physics.pivots && typeof physics.pivots === "object") {
        const pivots: Record<string, { x: number; y: number }> = {};
        for (const [part, value] of Object.entries(physics.pivots)) {
          if (typeof value?.x === "number" && typeof value?.y === "number") {
            pivots[part] = { x: value.x, y: value.y };
          }
        }
        setMotionLabPivots(pivots);
      }
      if (physics.rangesDeg && typeof physics.rangesDeg === "object") {
        const ranges: Record<string, number> = {};
        for (const [part, value] of Object.entries(physics.rangesDeg)) {
          if (typeof value === "number") ranges[part] = clamp(value, 0, 90);
        }
        setMotionLabRangesDeg(ranges);
      }
      if (physics.swingScale && typeof physics.swingScale === "object") {
        const scales: Record<string, number> = {};
        for (const [part, value] of Object.entries(physics.swingScale)) {
          if (typeof value === "number") scales[part] = clamp(value, 0, 3);
        }
        setMotionLabSwingScale(scales);
      }
      if (typeof physics.parallaxScale === "number") setMotionLabParallaxScale(clamp(physics.parallaxScale, 0, 1.5));
      if (typeof physics.strandsEnabled === "boolean") setMotionLabStrandsEnabled(physics.strandsEnabled);
      if (typeof physics.armBehindBody === "boolean") setMotionLabArmBehindBody(physics.armBehindBody);
    }
    if (manifest.review) {
      setMotionLabVerdict(validMotionLabVerdict(manifest.review.verdict));
      setMotionLabReviewNote(typeof manifest.review.note === "string" ? manifest.review.note : "");
      const scores = manifest.review.scores ?? {};
      setMotionLabReviewScores({
        mouthSmoothness: typeof scores.mouthSmoothness === "number" ? clamp(Math.round(scores.mouthSmoothness), 1, 5) : MOTION_LAB_DEFAULT_REVIEW_SCORES.mouthSmoothness,
        vowelReadability: typeof scores.vowelReadability === "number" ? clamp(Math.round(scores.vowelReadability), 1, 5) : MOTION_LAB_DEFAULT_REVIEW_SCORES.vowelReadability,
        bodyNaturalness: typeof scores.bodyNaturalness === "number" ? clamp(Math.round(scores.bodyNaturalness), 1, 5) : MOTION_LAB_DEFAULT_REVIEW_SCORES.bodyNaturalness,
        hairBodySeparation: typeof scores.hairBodySeparation === "number" ? clamp(Math.round(scores.hairBodySeparation), 1, 5) : MOTION_LAB_DEFAULT_REVIEW_SCORES.hairBodySeparation,
        settingSimplicity: typeof scores.settingSimplicity === "number" ? clamp(Math.round(scores.settingSimplicity), 1, 5) : MOTION_LAB_DEFAULT_REVIEW_SCORES.settingSimplicity,
        migrationConfidence: typeof scores.migrationConfidence === "number" ? clamp(Math.round(scores.migrationConfidence), 1, 5) : MOTION_LAB_DEFAULT_REVIEW_SCORES.migrationConfidence,
      });
    }
  }

  async function saveMotionLabManifest() {
    if (!motionLabParts) return;
    const manifest = buildMotionLabManifest();
    if (!manifest) return;
    setError("");
    setLoading(true);
    try {
      const result = await invoke<MotionLabManifestResult>("save_motion_lab_manifest", {
        request: { sourceDir: motionLabParts.sourceDir, manifest },
      });
      setMotionLabManifestPath(result.path);
      setStatus(`Motion Lab設定を保存しました: ${result.path}`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoading(false);
    }
  }

  async function loadMotionLabManifest() {
    if (!motionLabParts) return;
    setError("");
    setLoading(true);
    try {
      const result = await invoke<MotionLabManifestResult>("load_motion_lab_manifest", {
        sourceDir: motionLabParts.sourceDir,
      });
      applyMotionLabManifest(result.manifest);
      setMotionLabManifestPath(result.path);
      setStatus(`Motion Lab設定を読み込みました: ${result.path}`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoading(false);
    }
  }

  async function saveSpritalkMotionProfile() {
    if (!motionLabParts) return;
    const profile = buildSpritalkMotionProfile();
    if (!profile) return;
    setError("");
    setLoading(true);
    try {
      const result = await invoke<SpritalkMotionProfileResult>("save_spritalk_motion_profile", {
        request: { sourceDir: motionLabParts.sourceDir, profile },
      });
      setMotionLabProfilePath(result.path);
      setStatus(`SpriTalk motion profileを保存しました: ${result.path}`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoading(false);
    }
  }

  async function persistWorkspaceStep(step: WorkspaceStep, workspaceOverride?: ExpressionWorkspaceResult) {
    const workspace = workspaceOverride ?? expressionWorkspace;
    if (!workspace) return;
    try {
      const updated = await invoke<ExpressionWorkspaceResult>("update_expression_workspace_step", {
        workPath: workspace.workPath,
        currentStep: step,
      });
      setExpressionWorkspace(updated);
    } catch {
      // UI progress can continue even if project.json could not be updated.
    }
  }

  async function setWorkspaceStepAndPersist(step: WorkspaceStep) {
    setWorkspaceStep(step);
    await persistWorkspaceStep(step);
  }

  async function restoreWorkspaceProgress(workspace: ExpressionWorkspaceResult) {
    let restoredStep = Math.min(Math.max(workspace.project.currentStep || 1, 1), 7) as WorkspaceStep;
    try {
      const generated = await invoke<WorkspaceGeneratedPartsStatus>("inspect_workspace_generated_parts", {
        workPath: workspace.workPath,
      });
      setWorkspaceGeneratedStatus(generated);
      if (generated.ready && restoredStep < 3) restoredStep = 3;
    } catch {
      // Step 1 workspaces may not have a source image or request files yet.
    }

    try {
      const loaded = await invoke<LoadCodexExpressionJobResult>("load_codex_expression_job", {
        jobPath: workspace.workPath,
      });
      setWorkspaceGeneratedStatus(loaded.generatedParts);
      setWorkspaceExtractResult(loaded.extractedParts);
      setWorkspaceRifeResult(loaded.rifeOutput);
      if (loaded.extractedParts) {
        await refreshWorkspaceCompositePreview(workspace);
      }
      if (loaded.rifeOutput) {
        await refreshWorkspaceRifePreview(workspace).catch(() => null);
      }
      // project.jsonが7を保持していれば7で復帰。rifeOutputがあるのに6未満なら6へ引き上げ。
      // rifeOutputが無いのに7が保存されていた場合（出力を削除した等）は6へ落とす
      if (loaded.rifeOutput) {
        restoredStep = Math.max(restoredStep, 6) as WorkspaceStep;
      } else {
        if (restoredStep > 6) restoredStep = 6;
        if (loaded.extractedParts) restoredStep = Math.max(restoredStep, 5) as WorkspaceStep;
        else if (loaded.generatedParts.ready) restoredStep = Math.max(restoredStep, 3) as WorkspaceStep;
      }
    } catch {
      // Not all workspaces are valid Codex jobs until Step 2 has been prepared.
    }

    setWorkspaceStep(restoredStep);
    await persistWorkspaceStep(restoredStep, workspace);
  }

  async function refreshWorkspaceCompositePreview(workspaceOverride?: ExpressionWorkspaceResult) {
    const workspace = workspaceOverride ?? expressionWorkspace;
    if (!workspace) return;
    const preview = await invoke<PreviewCodexCompositeResult>("preview_codex_composite", {
      jobPath: workspace.workPath,
      profile: "auto",
    });
    setWorkspaceCompositePreview(preview);
    setWorkspaceSelectedPreviewPart(prev => (
      prev === "base" || preview.previews.some(item => item.part === prev) ? prev : "base"
    ));
  }

  async function refreshWorkspaceRifePreview(workspaceOverride?: ExpressionWorkspaceResult) {
    const workspace = workspaceOverride ?? expressionWorkspace;
    if (!workspace) return;
    const preview = await invoke<PreviewCodexRifeResult>("preview_codex_rife_outputs", {
      jobPath: workspace.workPath,
    });
    setWorkspaceRifePreview(preview);
    setWorkspaceSelectedPreviewPart(prev => (
      prev === "base" || preview.previews.some(item => workspacePreviewItemKey(item) === prev)
        ? prev
        : preview.previews[0] ? workspacePreviewItemKey(preview.previews[0]) : "base"
    ));
  }

  async function updateWorkspaceCompositePreview() {
    setError("");
    setWorkspaceBusy(true);
    try {
      await refreshWorkspaceCompositePreview();
      setStatus("合成プレビューを更新しました");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function openWorkspaceBaseAdjustment() {
    if (!expressionWorkspace || !workspaceFiles.source) {
      setError("立ち絵を読み込んでから素体調整を開いてください");
      return;
    }
    setError("");
    setWorkspaceBusy(true);
    try {
      const base = await invoke<SeeThroughRunResult>("load_codex_source_see_through", {
        jobPath: expressionWorkspace.workPath,
      });
      const defaultMapping = Object.fromEntries(base.slotLoad.adjustable_layers.map(layer => [layer.name, layer.default_target]));
      // 腕は原則分離出力（handwear-l/-r検出時は既定でarm_l/arm_rへ。bodyに戻したい場合はStep4のチェックをOFF）
      for (const name of base.slotLoad.detected_layers) {
        if (/^handwear[-_][lr]$/i.test(name)) defaultMapping[name] = /[-_]r$/i.test(name) ? "arm_r" : "arm_l";
      }
      setLoadResult(base.slotLoad);
      setLayerMapping(defaultMapping);
      const allPreview = await invoke<MappingPreviewResult>("get_all_layers_preview");
      setMappingPreview(allPreview);
      await openUnifiedBaseEditorWithPreview(allPreview);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function loadWorkspaceImagePreviews(files: Record<string, string>) {
    const entries = Object.entries(files).filter(([, path]) => !!path);
    if (entries.length === 0) {
      setWorkspaceImagePreviews({});
      return;
    }
    const next: Record<string, string> = {};
    await Promise.all(entries.map(async ([key, path]) => {
      try {
        next[key] = await invoke<string>("load_expression_source_preview", { path });
      } catch {
        // Keep the path visible even when thumbnail decoding fails.
      }
    }));
    setWorkspaceImagePreviews(next);
  }

  async function pickWorkspaceImage(key: "source" | "reference") {
    const file = await open({ multiple: false, directory: false, filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (!file || typeof file !== "string") return;
    setWorkspaceFiles(prev => {
      const next = { ...prev, [key]: file };
      void loadWorkspaceImagePreviews(next);
      return next;
    });
  }

  async function prepareWorkspaceCodexRequest() {
    if (!expressionWorkspace || !workspaceFiles.source) {
      setError("先に作業フォルダと立ち絵を選択してください");
      return;
    }
    setError("");
    setWorkspaceBusy(true);
    try {
      const status = await invoke<WorkspaceGeneratedPartsStatus>("prepare_workspace_codex_request", {
        request: {
          workPath: expressionWorkspace.workPath,
          sourceImagePath: workspaceFiles.source,
          referenceImagePath: workspaceFiles.reference || null,
          prompt: "Keep character identity, pose, hair, clothes, lighting, and background unchanged. Edit only the requested eyes or mouth.",
          mouthCorner: "neutral",
          mouthSize: "normal",
        },
      });
      setWorkspaceGeneratedStatus(status);
      await setWorkspaceStepAndPersist(2);
      setStatus("Codex依頼ファイルを作成しました");
      await revealItemInDir(status.requestPath).catch(() => {});
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function inspectWorkspaceGeneratedParts() {
    if (!expressionWorkspace) return;
    setError("");
    setWorkspaceBusy(true);
    try {
      const status = await invoke<WorkspaceGeneratedPartsStatus>("inspect_workspace_generated_parts", { workPath: expressionWorkspace.workPath });
      setWorkspaceGeneratedStatus(status);
      if (status.ready) await setWorkspaceStepAndPersist(3);
      setStatus(status.ready ? "Codex成果物が揃いました" : "Codex成果物に不足があります");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function refreshWorkspaceSeeThroughStatus() {
    setError("");
    setWorkspaceBusy(true);
    setSeeThroughProgress({ stage: "status", percent: 0, message: "See-Through環境を確認しています" });
    setStatus("See-Through環境を確認しています");
    try {
      const runtime = await invoke<SeeThroughRuntimeStatus>("get_see_through_runtime_status");
      setSeeThroughRuntime(runtime);
      setStatus(runtime.message);
      setSeeThroughProgress({ stage: "status", percent: 100, message: runtime.message });
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function prepareWorkspaceSeeThroughRuntime() {
    setError("");
    setWorkspaceBusy(true);
    setSeeThroughProgress({ stage: "prepare", percent: 0, message: "See-Throughを準備しています" });
    try {
      const runtime = await invoke<SeeThroughRuntimeStatus>("prepare_see_through_runtime");
      setSeeThroughRuntime(runtime);
      setStatus(runtime.message);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function runWorkspaceSeeThroughBatch() {
    if (!expressionWorkspace || !workspaceFiles.source || !workspaceGeneratedStatus?.ready) {
      setError("立ち絵とCodex成果物を先に揃えてください");
      return;
    }
    if (!seeThroughRuntime?.ready) {
      setError("See-Throughの初回セットアップを先に完了してください");
      return;
    }
    setError("");
    setWorkspaceBusy(true);
    setSeeThroughStartedAt(Date.now());
    setSeeThroughElapsedSeconds(0);
    setSeeThroughProgress({ stage: "inference", percent: 0, message: "See-Through一括分解中" });
    try {
      const base = await invoke<SeeThroughRunResult>("run_see_through", {
        sourcePath: workspaceFiles.source,
        profile: seeThroughProfile,
        splitParts: seeThroughSplitParts,
        options: seeThroughOptions,
      });
      await invoke<string>("cache_codex_source_see_through", {
        jobPath: expressionWorkspace.workPath,
        psdPath: base.psdPath,
      });
      const defaultMapping = Object.fromEntries(base.slotLoad.adjustable_layers.map(layer => [layer.name, layer.default_target]));
      // 腕は原則分離出力（handwear-l/-r検出時は既定でarm_l/arm_rへ。bodyに戻したい場合はStep4のチェックをOFF）
      for (const name of base.slotLoad.detected_layers) {
        if (/^handwear[-_][lr]$/i.test(name)) defaultMapping[name] = /[-_]r$/i.test(name) ? "arm_r" : "arm_l";
      }
      setLoadResult(base.slotLoad);
      setLayerMapping(defaultMapping);
      setMappingPreview(base.mappingPreview);
      setSeeThroughProgress({ stage: "inference", percent: 50, message: "Codex成果物をSee-Throughで分解しています" });
      const extracted = await invoke<ExtractCodexGeneratedPartsResult>("extract_codex_generated_parts", {
        jobPath: expressionWorkspace.workPath,
        profile: seeThroughProfile,
        splitParts: seeThroughSplitParts,
        options: seeThroughOptions,
      });
      setWorkspaceExtractResult(extracted);
      setWorkspaceCompositePreview(null);
      setWorkspaceRifeResult(null);
      setWorkspaceRifePreview(null);
      await setWorkspaceStepAndPersist(4);
      const allPreview = await invoke<MappingPreviewResult>("get_all_layers_preview");
      setMappingPreview(allPreview);
      await openUnifiedBaseEditorWithPreview(allPreview);
      setSeeThroughProgress({ stage: "complete", percent: 100, message: "See-Through一括分解が完了しました" });
      setStatus(`See-Through一括分解が完了しました: ${extracted.extractedParts.length}パーツ`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
      setSeeThroughStartedAt(null);
    }
  }

  async function applyWorkspacePartAdjustment(offsetOverride?: { x: number; y: number }) {
    if (!expressionWorkspace || !workspaceExtractResult) return;
    setWorkspaceBusy(true);
    try {
      const result = await invoke<AdjustCodexExtractedPartsResult>("adjust_codex_extracted_parts", {
        request: {
          jobPath: expressionWorkspace.workPath,
          offsetX: offsetOverride?.x ?? workspacePartOffsetX,
          offsetY: offsetOverride?.y ?? workspacePartOffsetY,
          scalePercent: workspacePartScale,
          // "all" 以外はそのパーツだけ調整（パーツごとのズレに個別対応）
          part: workspaceAdjustTarget === "all" ? null : workspaceAdjustTarget,
        },
      });
      setWorkspaceExtractResult({
        extractedPartsPath: result.extractedPartsPath,
        // 個別適用時は対象1件しか返らないため、一覧は維持する
        extractedParts: workspaceAdjustTarget === "all" ? result.adjustedParts : workspaceExtractResult.extractedParts,
        warnings: workspaceExtractResult.warnings,
      });
      setWorkspaceRifeResult(null);
      setWorkspaceRifePreview(null);
      await refreshWorkspaceCompositePreview().catch(() => null);
      await setWorkspaceStepAndPersist(5);
      setStatus("差分位置を調整しました");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function generateWorkspaceRifeOutputs() {
    if (!expressionWorkspace || !workspaceExtractResult) {
      setError("先にSee-Through一括分解を完了してください");
      return;
    }
    setWorkspaceBusy(true);
    try {
      await invoke<SaveCodexBasePartsResult>("save_codex_base_parts", {
        jobPath: expressionWorkspace.workPath,
      }).catch(() => null);
      const result = await invoke<GenerateCodexRifeOutputResult>("generate_codex_rife_outputs", {
        jobPath: expressionWorkspace.workPath,
        frameCount,
        profile: "auto",
      });
      setWorkspaceRifeResult(result);
      await refreshWorkspaceRifePreview().catch(() => null);
      // RIFE完了後は自動でSTEP7（モーション調整）へ進む
      await setWorkspaceStepAndPersist(7);
      setStatus(`SpriTalk用フォルダへ出力しました: ${result.outputPath}。モーション調整に進みます`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  function returnToModeSelect() {
    if (mode === "interp" && interpStep !== 4 && hasInterpWork) {
      const ok = window.confirm("現在のフレーム補間作業を終了してモード選択へ戻ります。出力、口マスク、生成プレビューはこの画面から離れると失われます。戻りますか？");
      if (!ok) return;
    }
    setMode("select");
    setStatus("モードを選択してください");
    resetZoom();
  }

  function returnFromInterpInput() {
    if (hasInterpWork) {
      const ok = window.confirm("現在の入力内容を破棄して戻ります。続けますか？");
      if (!ok) return;
    }
    setMode("select");
    setStatus("モードを選択してください");
    resetZoom();
  }

  function returnFromBaseFlow() {
    if (expressionWorkspace) {
      setMode("workspace");
      setWorkspaceStep(4);
      setStatus("素体調整に戻りました。");
      resetZoom();
      return;
    }
    setMode("select");
    setBaseStep(1);
    setStatus("モードを選択してください");
    resetZoom();
  }

  function goBaseStep(step: BaseStep) {
    if (step === 1) {
      setBaseStep(1);
      setMode("base_input");
      return;
    }
    if (step === 2 && mappingPreview) {
      setBaseStep(2);
      setMode("hair_edit");
      return;
    }
    if (step === 3 && bodyPreview) {
      setBaseStep(3);
      setMode("base_edit");
      return;
    }
    if (step === 4 && baseResult) {
      setBaseStep(4);
      setMode("base_edit");
    }
  }

  function baseStepEnabled(step: BaseStep) {
    if (step === 1) return true;
    if (step === 2) return !!mappingPreview;
    if (step === 3) return !!bodyPreview;
    return !!baseResult;
  }

  const bodyCategory = mappingPreview?.categories.find(c => c.target === (mode === "correction" || !!expressionWorkspace ? "free" : "body"));
  const selectedLayerOpacity = selectedBodyLayer ? (layerOpacities[selectedBodyLayer] ?? 1) : 1;

  function getBodyOrderItem(name: string): (LayerInfo & { isPatch?: boolean }) | null {
    const patch = layerPatches.find(p => p.id === name);
    if (patch) {
      const source = bodyCategory?.layers.find(l => l.name === patch.sourceLayer);
      if (!source) return null;
      return { ...source, name: patch.name, thumbnail: patch.thumbnail ?? source.thumbnail, isPatch: true };
    }
    return bodyCategory?.layers.find(l => l.name === name) ?? null;
  }

  function createPatchThumbnail(sourceLayerName: string, maskCanvas: HTMLCanvasElement): Promise<string | undefined> {
    const source = bodyCategory?.layers.find(l => l.name === sourceLayerName);
    if (!source) return Promise.resolve(undefined);

    return new Promise(resolve => {
      const sourceImg = new Image();
      sourceImg.onload = () => {
        const thumbSize = 120;
        const canvas = document.createElement("canvas");
        canvas.width = thumbSize;
        canvas.height = thumbSize;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(undefined);
          return;
        }

        const scale = Math.min(thumbSize / maskCanvas.width, thumbSize / maskCanvas.height);
        const drawW = maskCanvas.width * scale;
        const drawH = maskCanvas.height * scale;
        const dx = (thumbSize - drawW) / 2;
        const dy = (thumbSize - drawH) / 2;

        ctx.clearRect(0, 0, thumbSize, thumbSize);
        ctx.drawImage(sourceImg, dx, dy, drawW, drawH);
        ctx.globalCompositeOperation = "destination-in";
        ctx.drawImage(maskCanvas, dx, dy, drawW, drawH);
        ctx.globalCompositeOperation = "source-over";
        resolve(canvas.toDataURL("image/png"));
      };
      sourceImg.onerror = () => resolve(undefined);
      sourceImg.src = source.thumbnail;
    });
  }

  function getPreviewImageMetrics() {
    const viewport = previewRef.current;
    if (!viewport || !loadResult) return null;
    const viewportRect = viewport.getBoundingClientRect();
    const fitScale = Math.min(viewportRect.width / loadResult.canvas_width, viewportRect.height / loadResult.canvas_height);
    const displayScale = fitScale * zoom;
    const width = loadResult.canvas_width * displayScale;
    const height = loadResult.canvas_height * displayScale;
    return {
      left: (viewportRect.width - width) / 2 + pan.x,
      top: (viewportRect.height - height) / 2 + pan.y,
      width,
      height,
      displayScale,
    };
  }

  function previewImageStyle(): React.CSSProperties | undefined {
    const metrics = getPreviewImageMetrics();
    if (!metrics || !loadResult) return undefined;
    return {
      left: metrics.left,
      top: metrics.top,
      width: metrics.width,
      height: metrics.height,
    };
  }

  function brushCursorStyle(): React.CSSProperties | undefined {
    const metrics = getPreviewImageMetrics();
    if (!metrics || !brushCursor.visible) return undefined;
    return {
      left: metrics.left + brushCursor.x - brushCursor.size / 2,
      top: metrics.top + brushCursor.y - brushCursor.size / 2,
      width: brushCursor.size,
      height: brushCursor.size,
    };
  }

  function initPatchMask(sourceLayer: string) {
    setPatchDraftSource(sourceLayer);
    setSelectedBodyLayer(sourceLayer);
    requestAnimationFrame(() => {
      const canvas = maskCanvasRef.current;
      if (!canvas || !loadResult) return;
      canvas.width = loadResult.canvas_width;
      canvas.height = loadResult.canvas_height;
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    });
  }

  function initChestCut() {
    setSelectedBodyLayer("");
    initPatchMask(CHEST_CUT_SENTINEL);
  }

  function clearPatchMask() {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawPatchMask(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = maskCanvasRef.current;
    const metrics = getPreviewImageMetrics();
    if (!canvas || !metrics) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / metrics.displayScale;
    const y = (e.clientY - rect.top) / metrics.displayScale;
    setBrushCursor({ x: x * metrics.displayScale, y: y * metrics.displayScale, size: patchBrushSize * metrics.displayScale, visible: true });
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const radius = patchBrushSize / 2;
    ctx.save();
    ctx.globalCompositeOperation = patchTool === "erase" ? "destination-out" : "source-over";
    if (patchBrushSoftness <= 0.01) {
      // 硬いエッジ（従来どおり）
      ctx.fillStyle = "rgba(233, 69, 96, 0.55)";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // ぼかしブラシ: 中心から不透明、外周へガウス的に減衰する放射グラデーション。
      // 切出境界のギザギザ（ジャギー）はエッジのアルファ中間値で自然に馴染む
      const innerStop = Math.max(0, 1 - patchBrushSoftness);
      const gradient = ctx.createRadialGradient(x, y, radius * innerStop, x, y, radius);
      gradient.addColorStop(0, "rgba(233, 69, 96, 0.85)");
      gradient.addColorStop(1, "rgba(233, 69, 96, 0)");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  async function sam3SelectAtPoint(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = maskCanvasRef.current;
    const metrics = getPreviewImageMetrics();
    if (!canvas || !metrics || !bodyPreview || sam3Selecting) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / metrics.displayScale;
    const y = (e.clientY - rect.top) / metrics.displayScale;
    setSam3Selecting(true);
    try {
      const result = await invoke<{ maskPng: string }>("sam3_select_region", {
        imageDataUrl: bodyPreview,
        points: [[x, y]],
      });
      const img = await loadMotionLabImage(result.maskPng);
      const ctx = canvas.getContext("2d");
      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setSam3Selecting(false);
    }
  }

  function onPatchMaskPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!patchDraftSource) return;
    e.preventDefault();
    e.stopPropagation();
    if (sam3SelectMode) {
      void sam3SelectAtPoint(e);
      return;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    maskDrawingRef.current = true;
    drawPatchMask(e);
  }

  function onPatchMaskPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = maskCanvasRef.current;
    const metrics = getPreviewImageMetrics();
    if (canvas && metrics) {
      const rect = canvas.getBoundingClientRect();
      setBrushCursor({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        size: patchBrushSize * metrics.displayScale,
        visible: true,
      });
    }
    if (!maskDrawingRef.current) return;
    e.preventDefault();
    e.stopPropagation();
    drawPatchMask(e);
  }

  function onPatchMaskPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!maskDrawingRef.current) return;
    e.preventDefault();
    e.stopPropagation();
    maskDrawingRef.current = false;
  }

  function onPatchMaskPointerLeave() {
    setBrushCursor(prev => ({ ...prev, visible: false }));
    maskDrawingRef.current = false;
  }

  async function commitPatchMask() {
    const canvas = maskCanvasRef.current;
    if (!canvas || !patchDraftSource) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let hasMask = false;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] > 0) { hasMask = true; break; }
    }
    if (!hasMask) {
      setError("切り出す範囲を塗ってください");
      return;
    }

    if (patchDraftSource === CHEST_CUT_SENTINEL) {
      // 胸を切出（852話式導線）: bodyから塗った範囲を chest.png として分離する。
      // 実際の分離処理は create_base 実行時（保存して戻る）にまとめて行う
      setChestMaskDataUrl(canvas.toDataURL("image/png"));
      setPatchDraftSource("");
      clearPatchMask();
      setStatus("胸の切出範囲を設定しました。「保存して戻る」で chest.png に反映されます。");
      return;
    }

    const patchIndex = layerPatchesRef.current.filter(p => p.sourceLayer === patchDraftSource).length + 1;
    const thumbnail = await createPatchThumbnail(patchDraftSource, canvas);
    const patch: LayerPatch = {
      id: `patch_${Date.now()}`,
      name: `${patchDraftSource}_patch_${patchIndex}`,
      sourceLayer: patchDraftSource,
      maskPng: canvas.toDataURL("image/png"),
      cutSource: true,
      thumbnail,
    };
    const nextPatches = [...layerPatchesRef.current, patch];
    const sourceIdx = layerOrderRef.current.indexOf(patchDraftSource);
    const nextOrder = [...layerOrderRef.current];
    nextOrder.splice(sourceIdx >= 0 ? sourceIdx : 0, 0, patch.id);
    const nextEnabled = { ...enabledLayers, [patch.id]: true };
    const nextOpacities = { ...layerOpacitiesRef.current, [patch.id]: mode === "correction" ? 1 : 0.5 };

    setLayerPatches(nextPatches);
    setLayerOrder(nextOrder);
    setEnabledLayers(nextEnabled);
    setLayerOpacities(nextOpacities);
    setSelectedBodyLayer(patch.id);
    setPatchDraftSource("");
    clearPatchMask();
    await renderBody(nextOrder, nextEnabled, nextPatches, nextOpacities);
  }

  async function removeLayerPatch(patchId: string) {
    const nextPatches = layerPatchesRef.current.filter(p => p.id !== patchId);
    const nextOrder = layerOrderRef.current.filter(name => name !== patchId);
    const nextEnabled = { ...enabledLayers };
    const nextOpacities = { ...layerOpacitiesRef.current };
    delete nextEnabled[patchId];
    delete nextOpacities[patchId];
    setLayerPatches(nextPatches);
    setLayerOrder(nextOrder);
    setEnabledLayers(nextEnabled);
    setLayerOpacities(nextOpacities);
    if (selectedBodyLayer === patchId) setSelectedBodyLayer("");
    await renderBody(nextOrder, nextEnabled, nextPatches, nextOpacities);
  }

  function renderWorkspaceMode() {
    const workspace = expressionWorkspace;
    const steps = [
      [1, "画像選択", "立ち絵と参照画像"],
      [2, "Codex依頼", "依頼作成と成果物確認"],
      [3, "See-Through", "一括分解"],
      [4, "素体調整", "レイヤー確認"],
      [5, "差分位置調整", "合成確認"],
      [6, "RIFE補完", "SpriTalk出力"],
      [7, "モーション調整", "揺れ・口パクの仕上げ"],
    ] as Array<[WorkspaceStep, string, string]>;
    const canOpenWorkspaceStep = (step: WorkspaceStep) => {
      if (workspaceBusy) return false;
      if (step === 1) return true;
      if (step === 2) return !!workspaceFiles.source;
      if (step === 3) return !!workspaceGeneratedStatus?.ready;
      if (step === 4) return !!mappingPreview || !!workspaceFiles.source;
      if (step === 5) return !!workspaceExtractResult && !!workspaceCompositePreview?.basePreview;
      if (step === 6) return !!workspaceExtractResult && !!workspaceCompositePreview?.basePreview;
      return !!workspaceRifeResult;
    };
    const canAdvanceWorkspaceStep = () => {
      if (workspaceBusy) return false;
      if (workspaceStep === 1) return !!workspaceFiles.source;
      if (workspaceStep === 2) return !!workspaceGeneratedStatus?.ready;
      if (workspaceStep === 3) return !!mappingPreview;
      if (workspaceStep === 4) return !!workspaceCompositePreview?.basePreview;
      if (workspaceStep === 5) return !!workspaceExtractResult && !!workspaceCompositePreview?.basePreview;
      if (workspaceStep === 6) return !!workspaceRifeResult;
      return false;
    };
    const seeThroughRunning = workspaceBusy && !!seeThroughProgress && ["prepare", "inference", "load"].includes(seeThroughProgress.stage);
    const activePreviewSet = workspaceStep >= 6 && workspaceRifePreview ? workspaceRifePreview : workspaceCompositePreview;
    const selectedPreview = workspaceSelectedPreviewPart === "base"
      ? null
      : activePreviewSet?.previews.find(item => workspacePreviewItemKey(item) === workspaceSelectedPreviewPart) ?? null;
    const mainPreviewImage = selectedPreview?.preview ?? activePreviewSet?.basePreview ?? "";
    const mainPreviewLabel = selectedPreview ? workspacePreviewItemLabel(selectedPreview) : "base";
    const goPreviousWorkspaceStep = () => {
      if (workspaceStep === 1) {
        returnToModeSelect();
        return;
      }
      void setWorkspaceStepAndPersist((workspaceStep - 1) as WorkspaceStep);
    };
    const goNextWorkspaceStep = () => {
      if (workspaceStep >= 7 || !canAdvanceWorkspaceStep()) return;
      void setWorkspaceStepAndPersist((workspaceStep + 1) as WorkspaceStep);
    };
    if (!workspace) return null;

    return (
      <div className="workspace-flow-screen">
        <nav className="workspace-flow-stepper" aria-label="表情セット作成ステップ">
          {steps.map(([step, label, note]) => (
            <button
              key={step}
              className={`workspace-flow-step${workspaceStep === step ? " active" : ""}${workspaceStep > step ? " done" : ""}`}
              disabled={!canOpenWorkspaceStep(step)}
              onClick={() => void setWorkspaceStepAndPersist(step)}
            >
              <b>{step}</b>
              <span><strong>{label}</strong><small>{note}</small></span>
            </button>
          ))}
        </nav>

        <div
          className={`workspace-flow-layout${workspaceStep === 1 ? " step-one" : ""}${workspaceStep === 2 || workspaceStep === 3 ? " single-panel" : ""}`}
          style={workspaceStep === 7 ? { display: "none" } : undefined}
        >
          <section className="workspace-flow-panel">
            {workspaceStep === 1 && (
              <div className="workspace-step-one">
                <div className="workspace-panel-heading compact">
                  <span>STEP 1</span>
                  <h3>立ち絵と参照画像を選択</h3>
                  <p>立ち絵は必須です。参照画像は目や口の中の色、質感をCodex生成で合わせたい場合だけ使います。</p>
                </div>
                <div className="workspace-image-picker-grid">
                  <button className={`workspace-image-picker${workspaceFiles.source ? " ready" : ""}`} onClick={() => void pickWorkspaceImage("source")}>
                    <span>立ち絵</span>
                    {workspaceImagePreviews.source ? <img src={workspaceImagePreviews.source} alt="立ち絵プレビュー" /> : <strong>選択</strong>}
                    <small>{workspaceFiles.source ? "選択済み" : "必須: 表情セットの元画像"}</small>
                  </button>
                  <button className={`workspace-image-picker${workspaceFiles.reference ? " ready" : ""}`} onClick={() => void pickWorkspaceImage("reference")}>
                    <span>参照画像</span>
                    {workspaceImagePreviews.reference ? <img src={workspaceImagePreviews.reference} alt="参照画像プレビュー" /> : <strong>選択</strong>}
                    <small>{workspaceFiles.reference ? "選択済み" : "任意: 目や口内の色参照"}</small>
                  </button>
                </div>
              </div>
            )}

            {workspaceStep === 2 && (
              <>
                <div className="workspace-panel-heading">
                  <span>STEP 2</span>
                  <h3>Codex依頼を作成</h3>
                  <p>依頼ファイルを作成し、Codexが生成した7枚を <code>02_generated_parts</code> に配置してから確認します。</p>
                </div>
                <div className="workspace-codex-steps">
                  <section className="workspace-codex-card">
                    <div>
                      <span>1</span>
                      <strong>依頼ファイル作成</strong>
                      <p>Codexへ渡す説明書と元画像を <code>01_codex_request</code> に作成します。</p>
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-secondary" onClick={() => revealItemInDir(workspace.codexRequestPath).catch(() => {})}>依頼フォルダを開く</button>
                      <button className="btn btn-primary" disabled={workspaceBusy || !workspaceFiles.source} onClick={() => void prepareWorkspaceCodexRequest()}>依頼ファイルを作成</button>
                    </div>
                  </section>
                  <section className="workspace-codex-card">
                    <div>
                      <span>2</span>
                      <strong>成果物を確認</strong>
                      <p>Codexが作成した素材を <code>02_generated_parts</code> に配置して、必要ファイルが揃ったか確認します。</p>
                      <div className="workspace-status-card compact">
                        <strong>{workspaceGeneratedStatus?.ready ? "成果物は揃っています" : "成果物待ちです"}</strong>
                        <span>必要: {workspaceGeneratedStatus?.expectedParts.length ?? 7} / 配置済み: {workspaceGeneratedStatus?.presentParts.length ?? 0}</span>
                        {!!workspaceGeneratedStatus?.missingParts.length && <small>不足: {workspaceGeneratedStatus.missingParts.join(", ")}</small>}
                        {!!workspaceGeneratedStatus?.sizeMismatches.length && <small>サイズ不一致: {workspaceGeneratedStatus.sizeMismatches.join(", ")}</small>}
                      </div>
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-primary" disabled={workspaceBusy} onClick={() => void inspectWorkspaceGeneratedParts()}>成果物を確認</button>
                    </div>
                  </section>
                </div>
              </>
            )}

            {workspaceStep === 3 && (
              <>
                <div className="workspace-step3-header">
                  <div className="workspace-panel-heading">
                    <span>STEP 3</span>
                    <h3>See-Throughを一括実行</h3>
                    <p>立ち絵とCodex成果物をSee-Throughで分解します。</p>
                  </div>
                  <div className="workspace-step3-env">
                    <div className="workspace-action-row">
                      <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => void refreshWorkspaceSeeThroughStatus()}>環境を再確認</button>
                      {seeThroughRuntime?.ready ? (
                        <span className="workspace-action-done">セットアップ済み</span>
                      ) : (
                        <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => void prepareWorkspaceSeeThroughRuntime()}>初回セットアップ</button>
                      )}
                    </div>
                  </div>
                </div>
                <div className="workspace-seethrough-options">
                  <div className="workspace-option-card">
                    <span>実行プロファイル</span>
                    <div className="workspace-segmented">
                      <button className={seeThroughProfile === "low-vram" ? "active" : ""} disabled={workspaceBusy} onClick={() => setSeeThroughProfile("low-vram")}>省VRAM</button>
                      <button className={seeThroughProfile === "standard" ? "active" : ""} disabled={workspaceBusy} onClick={() => setSeeThroughProfile("standard")}>高VRAM</button>
                    </div>
                  </div>
                  <label className="workspace-toggle-option workspace-option-card">
                    <input type="checkbox" checked={seeThroughSplitParts} disabled={workspaceBusy} onChange={(event) => setSeeThroughSplitParts(event.target.checked)} />
                    <span>左右パーツ分解</span>
                    <small>目や耳などを左右レイヤーに分けます。</small>
                  </label>
                  <div className="workspace-option-card workspace-option-card-wide">
                    <div className="workspace-option-header">
                      <span>See-Throughパラメータ</span>
                      <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => setSeeThroughOptions(DEFAULT_SEE_THROUGH_OPTIONS)}>標準値に戻す</button>
                    </div>
                    <div className="workspace-option-grid">
                      <label><span>Seed</span><input type="number" value={seeThroughOptions.seed} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, seed: Number(event.target.value) })} /></label>
                      <label><span>LayerDiff解像度</span><input type="number" min={256} max={4096} step={64} value={seeThroughOptions.resolution} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, resolution: Number(event.target.value) })} /></label>
                      <label><span>Depth解像度</span><input type="number" min={-1} max={4096} step={64} value={seeThroughOptions.resolutionDepth} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, resolutionDepth: Number(event.target.value) })} /></label>
                      <label><span>LayerDiff step</span><input type="number" min={1} max={150} value={seeThroughOptions.inferenceSteps} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, inferenceSteps: Number(event.target.value) })} /></label>
                      <label><span>Depth step</span><input type="number" min={-1} max={150} value={seeThroughOptions.inferenceStepsDepth} disabled={workspaceBusy || seeThroughProfile === "low-vram"} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, inferenceStepsDepth: Number(event.target.value) })} /></label>
                      <label><span>Group offload</span><select value={seeThroughOptions.groupOffload} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, groupOffload: event.target.value as SeeThroughOptionMode })}><option value="default">標準</option><option value="on">有効</option><option value="off">無効</option></select></label>
                      <label><span>CPU offload</span><select value={seeThroughOptions.cpuOffload} disabled={workspaceBusy || seeThroughProfile === "standard"} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, cpuOffload: event.target.value as SeeThroughOptionMode })}><option value="default">標準</option><option value="on">有効</option><option value="off">無効</option></select></label>
                    </div>
                  </div>
                </div>
                <div className="workspace-start-seethrough">
                  <div><strong>分解処理を開始</strong><p>立ち絵、閉じ目、閉じ口、あいうえお口の素材をまとめて分解します。</p></div>
                  <button className="btn btn-primary" disabled={workspaceBusy || !seeThroughRuntime?.ready || !workspaceGeneratedStatus?.ready} onClick={() => void runWorkspaceSeeThroughBatch()}>{seeThroughRunning ? "分解処理中..." : "一括分解を開始"}</button>
                  {seeThroughRunning && (
                    <div className="workspace-running-inline">
                      <span>{displaySeeThroughMessage(seeThroughProgress)}</span>
                      <small>経過: {formatElapsed(seeThroughElapsedSeconds)}</small>
                      <div className="workspace-progress-bar" aria-label="See-Through進捗"><div style={{ width: `${Math.max(3, Math.min(100, seeThroughProgress.percent))}%` }} /></div>
                    </div>
                  )}
                </div>
              </>
            )}

            {workspaceStep === 4 && (
              <>
                <div className="workspace-panel-heading"><span>STEP 4</span><h3>素体を調整</h3><p>分解済みレイヤーの前後関係を確認し、SpriTalkに渡す素体パーツを作成します。</p></div>
                <div className="workspace-action-row"><button className="btn btn-primary" disabled={workspaceBusy} onClick={() => void openWorkspaceBaseAdjustment()}>素体調整を開く</button></div>
              </>
            )}

            {workspaceStep === 5 && (
              <>
                <div className="workspace-panel-heading"><span>STEP 5</span><h3>差分位置を調整</h3><p>パーツを選んで個別にX/Y/Scale補正できます。「パーツ移動」ON中はプレビューを直接ドラッグして動かせます（離すと適用）。</p></div>
                <div className="workspace-action-row">
                  <div className="workspace-adjust-grid">
                    <label><span>対象</span>
                      <select value={workspaceAdjustTarget} onChange={e => setWorkspaceAdjustTarget(e.target.value)}>
                        <option value="all">全パーツ一括</option>
                        {WORKSPACE_ADJUST_PART_KEYS.map(part => <option key={part} value={part}>{part}</option>)}
                      </select>
                    </label>
                    <label><span>X offset</span><input type="number" value={workspacePartOffsetX} onChange={e => setWorkspacePartOffsetX(Number(e.target.value))} /></label>
                    <label><span>Y offset</span><input type="number" value={workspacePartOffsetY} onChange={e => setWorkspacePartOffsetY(Number(e.target.value))} /></label>
                    <label><span>Scale %</span><input type="number" min={50} max={150} value={workspacePartScale} onChange={e => setWorkspacePartScale(Number(e.target.value))} /></label>
                  </div>
                  <button
                    className={`btn ${workspacePartDragMode ? "btn-primary" : "btn-secondary"}`}
                    disabled={workspaceBusy || !workspaceExtractResult || workspaceAdjustTarget === "all"}
                    title="ON中はプレビューのドラッグで選択パーツを移動（画面パンは無効化）"
                    onClick={() => setWorkspacePartDragMode(prev => !prev)}
                  >
                    パーツ移動 {workspacePartDragMode ? "ON" : "OFF"}
                  </button>
                  <button className="btn btn-secondary" disabled={workspaceBusy || !workspaceExtractResult} onClick={() => void updateWorkspaceCompositePreview()}>合成プレビューを更新</button>
                  <button className="btn btn-primary" disabled={workspaceBusy || !workspaceExtractResult} onClick={() => void applyWorkspacePartAdjustment()}>位置調整を適用</button>
                </div>
                <div className="motion-lab-note">
                  補正値はパーツごとに元画像基準の絶対値で保存されます（adjustment.json v2）。
                  eyes-open は元画像由来のため補正対象外です。下のサムネイルをクリックすると対象パーツが切り替わります。
                </div>
              </>
            )}

            {workspaceStep === 6 && (
              <>
                <div className="workspace-panel-heading"><span>STEP 6</span><h3>RIFE補完してSpriTalk用に出力</h3><p>RIFEフレーム、素体、抽出差分を <code>04_spritalk_parts</code> にまとめます。</p></div>
                <div className="workspace-rife-panel">
                  <div className="workspace-frame-slider">
                    <label><span>補間枚数</span><strong>{frameCount}枚</strong></label>
                    <input type="range" min={RIFE_FRAME_MIN} max={RIFE_FRAME_MAX} value={frameCount} disabled={workspaceBusy} onChange={e => setFrameCount(Number(e.target.value))} />
                    <small>SpriTalkでは {RIFE_FRAME_RECOMMENDED} 枚が扱いやすい目安です。</small>
                  </div>
                  <button className="btn btn-primary" disabled={workspaceBusy || !workspaceExtractResult || !workspaceCompositePreview?.basePreview} onClick={() => void generateWorkspaceRifeOutputs()}>{workspaceBusy ? "RIFE補完中..." : "RIFE補完を開始"}</button>
                </div>
              </>
            )}
          </section>

          {workspaceStep >= 4 && workspaceStep <= 6 && (
            <aside className="workspace-flow-preview">
              <div className="preview-card-heading"><span>PREVIEW</span><strong>{mainPreviewLabel}</strong></div>
              <div className="workspace-preview-stage">
                {mainPreviewImage ? (
                  <>
                    <div className={`workspace-preview-pan${workspacePreviewDragging ? " dragging" : ""}`} onWheel={handleWorkspacePreviewWheel} onPointerDown={handleWorkspacePreviewPointerDown} onPointerMove={handleWorkspacePreviewPointerMove} onPointerUp={handleWorkspacePreviewPointerUp} onPointerCancel={handleWorkspacePreviewPointerUp} onLostPointerCapture={handleWorkspacePreviewLostPointerCapture}>
                      <img src={mainPreviewImage} alt={`${mainPreviewLabel} プレビュー`} draggable={false} style={{ transform: `translate(${workspacePreviewPan.x}px, ${workspacePreviewPan.y}px) scale(${workspacePreviewZoom})` }} />
                    </div>
                    <div className="workspace-preview-zoom-controls">
                      <button className="btn-zoom" type="button" onClick={() => setWorkspacePreviewZoom(prev => Math.min(8, prev * 1.25))}>+</button>
                      <span className="zoom-level">{Math.round(workspacePreviewZoom * 100)}%</span>
                      <button className="btn-zoom" type="button" onClick={() => setWorkspacePreviewZoom(prev => Math.max(0.25, prev * 0.8))}>-</button>
                      <button className="btn-zoom btn-zoom-reset" type="button" onClick={resetWorkspacePreviewZoom}>リセット</button>
                    </div>
                  </>
                ) : <span>{workspaceStep >= 5 ? "素体作成後にプレビューできます" : "この工程では画像プレビューはありません"}</span>}
              </div>
              {activePreviewSet?.previews.length ? (
                <div className="workspace-preview-list">
                  <button type="button" className={workspaceSelectedPreviewPart === "base" ? "active" : ""} onClick={() => setWorkspaceSelectedPreviewPart("base")}><span>base</span><img src={activePreviewSet.basePreview} alt="base プレビュー" /></button>
                  {activePreviewSet.previews.map(item => (
                    <button type="button" key={workspacePreviewItemKey(item)} className={workspaceSelectedPreviewPart === workspacePreviewItemKey(item) ? "active" : ""} onClick={() => {
                      const key = workspacePreviewItemKey(item);
                      setWorkspaceSelectedPreviewPart(key);
                      // Step5ではサムネイル選択＝調整対象の切替（直観的な個別調整）
                      if (workspaceStep === 5 && WORKSPACE_ADJUST_PART_KEYS.includes(key)) {
                        setWorkspaceAdjustTarget(key);
                      }
                    }}>
                      <span>{workspacePreviewItemLabel(item)}</span><img src={item.preview} alt={`${workspacePreviewItemLabel(item)} preview`} />
                    </button>
                  ))}
                </div>
              ) : null}
            </aside>
          )}
        </div>

        {workspaceRifeResult && (
          <div className="workspace-motion-tune" style={workspaceStep === 7 ? undefined : { display: "none" }}>
            <MotionTunePanel
              partsDir={workspaceRifeResult.outputPath || workspace.spritalkPartsPath}
              active={workspaceStep === 7}
              onNotify={setStatus}
              onError={setError}
            />
          </div>
        )}

        <div className="workspace-bottom-nav">
          <button className="btn btn-secondary" onClick={goPreviousWorkspaceStep}>戻る</button>
          {workspaceStep >= 7 ? (
            <button className="btn btn-primary" disabled={workspaceBusy} onClick={() => revealItemInDir(workspaceRifeResult?.outputPath || workspace.spritalkPartsPath).catch(() => {})}>出力フォルダを開く</button>
          ) : (
            <button className="btn btn-primary" disabled={!canAdvanceWorkspaceStep()} onClick={goNextWorkspaceStep}>次へ</button>
          )}
        </div>
        <div className="workspace-log-console" aria-live="polite">
          <div className="workspace-log-title">LOG</div>
          <div className="workspace-log-lines">
            <div><span>step</span>{workspaceStep}/7 {steps[workspaceStep - 1]?.[1]}</div>
            {workspaceBusy && <div><span>run</span>処理中...</div>}
            {seeThroughProgress && <div><span>see-through</span>{seeThroughProgress.message}</div>}
            {progress.total > 0 && <div><span>rife</span>{progress.pair_name} {progress.current}/{progress.total}</div>}
            <div><span>{error ? "error" : "status"}</span>{error || status}</div>
          </div>
        </div>
      </div>
    );
  }
  function renderMotionLabMode() {
    const availableMouths = MOTION_LAB_VOWEL_KEYS.filter(key => (motionLabParts?.mouths[key]?.length ?? 0) > 0);
    const sourceLabel = motionLabParts?.sourceDir ?? "未選択";
    const canUseWorkspaceOutput = !!(workspaceRifeResult?.outputPath || expressionWorkspace?.spritalkPartsPath);
    const layerSummary = motionLabLayerMode === "mesh" ? "mesh strip" : motionLabLayerMode === "spring" ? "layer spring" : "simple";
    const mouthSummary = motionLabMethod === "bridge" ? "bridge" : motionLabMethod === "smooth" ? "smooth" : "direct";
    const candidateLabel = `${mouthSummary} + ${layerSummary}`;
    const smoothingSummary = motionLabMethod === "smooth"
      ? `${motionLabAttackMs}/${motionLabReleaseMs}ms + ${motionLabCrossfadeMs}ms`
      : motionLabMethod === "bridge"
        ? `${motionLabAttackMs}/${motionLabReleaseMs}ms + bridge ${Math.round(motionLabBridgeBias * 100)}%`
      : "mouth target direct";
    const renderRange = (
      label: string,
      value: number,
      min: number,
      max: number,
      step: number,
      onChange: (next: number) => void,
      suffix = "",
    ) => (
      <label className="motion-lab-range">
        <span>{label}<b>{value}{suffix}</b></span>
        <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      </label>
    );
    const renderReviewRange = (key: MotionLabReviewKey) => (
      <label className="motion-lab-review-range">
        <span>{MOTION_LAB_REVIEW_LABELS[key]}<b>{motionLabReviewScores[key]}/5</b></span>
        <input
          type="range"
          min={1}
          max={5}
          step={1}
          value={motionLabReviewScores[key]}
          onChange={(event) => setMotionLabReviewScores(prev => ({ ...prev, [key]: Number(event.target.value) }))}
        />
      </label>
    );

    return (
      <main className="motion-lab-screen">
        <section className="motion-lab-control-panel">
          <div className="motion-lab-heading">
            <span>MOTION LAB</span>
            <h2>モーション比較</h2>
          </div>

          <div className="motion-lab-source">
            <button className="btn btn-primary" disabled={loading} onClick={() => void pickMotionLabPartsDir()}>
              素材フォルダを選択
            </button>
            {canUseWorkspaceOutput && (
              <button className="btn btn-secondary" disabled={loading} onClick={() => void loadCurrentWorkspaceMotionLabParts()}>
                現在の出力を読む
              </button>
            )}
            <strong title={sourceLabel}>{sourceLabel}</strong>
          </div>

          <div className="motion-lab-manifest-actions">
            <button className="btn btn-secondary" disabled={loading || !motionLabParts} onClick={() => void loadMotionLabManifest()}>
              設定読込
            </button>
            <button className="btn btn-primary" disabled={loading || !motionLabParts} onClick={() => void saveMotionLabManifest()}>
              設定保存
            </button>
            <small title={motionLabManifestPath}>{motionLabManifestPath || "motion-preview-manifest.json"}</small>
            <button className="btn btn-secondary" disabled={loading || !motionLabParts} onClick={() => void saveSpritalkMotionProfile()}>
              SpriTalk用出力
            </button>
            <small title={motionLabProfilePath}>{motionLabProfilePath || "spritalk-motion-profile.json"}</small>
          </div>

          {motionLabParts && (
            <div className="motion-lab-status-grid">
              <span><b>{motionLabParts.width}x{motionLabParts.height}</b><small>canvas</small></span>
              <span><b>{availableMouths.map(key => MOTION_LAB_MOUTH_LABELS[key]).join(" ") || "-"}</b><small>mouth</small></span>
              <span><b>{motionLabParts.eyeFrames.length}</b><small>eye frames</small></span>
              <span><b>{motionLabParts.hair ? "front" : "-"} / {motionLabParts.hairBack ? "back" : "-"}</b><small>hair</small></span>
              <span><b>{motionLabParts.armL ? "L" : "-"}/{motionLabParts.armR ? "R" : "-"} {motionLabParts.chest ? "胸" : ""}</b><small>arm / chest</small></span>
              <span><b>{Object.keys(motionLabParts.sways).length}</b><small>sway parts</small></span>
              <span><b>{motionLabParts.eyewhite && motionLabParts.irides ? "視線" : "-"} {motionLabParts.highlight ? "HL" : ""}</b><small>gaze / highlight</small></span>
            </div>
          )}

          {motionLabParts?.warnings.length ? <div className="motion-lab-note">{motionLabParts.warnings.join(" / ")}</div> : null}
          {motionLabParts?.missing.length ? <div className="motion-lab-note warning">不足: {motionLabParts.missing.join(", ")}</div> : null}

          <div className="motion-lab-section motion-lab-simple">
            <div className="motion-lab-section-title">
              <strong>かんたん設定</strong>
              <div className="motion-lab-segmented three">
                <button className={motionLabPreset === "calm" ? "active" : ""} onClick={() => setMotionLabPreset("calm")}>おとなしめ</button>
                <button className={motionLabPreset === "normal" ? "active" : ""} onClick={() => setMotionLabPreset("normal")}>ふつう</button>
                <button className={motionLabPreset === "lively" ? "active" : ""} onClick={() => setMotionLabPreset("lively")}>元気</button>
              </div>
            </div>
            <div className="motion-lab-section-title">
              <strong>方式</strong>
              <div className="motion-lab-segmented">
                <button
                  className={motionLabEngineFamily === "rotejin" ? "active" : ""}
                  title="PuruPuruPNGTuber系: 進行波の髪揺れ＋ぷるぷるした弾み"
                  onClick={() => applyMotionLabEngineFamily("rotejin")}
                >ろてじん式（波・ぷるぷる）</button>
                <button
                  className={motionLabEngineFamily === "hachigoni" ? "active" : ""}
                  title="Anime2.5DRig系: バネ・チェーンの髪物理＋パララックス首振り"
                  onClick={() => applyMotionLabEngineFamily("hachigoni")}
                >852話式（バネ・リグ）</button>
              </div>
            </div>
            <div className="motion-lab-section-title">
              <strong>テンプレート</strong>
              <div className="motion-lab-segmented">
                {Object.entries(MOTION_LAB_TEMPLATES)
                  .filter(([, template]) => template.engine === motionLabEngineFamily)
                  .map(([key, template]) => (
                    <button
                      key={key}
                      className={motionLabTemplateName === key ? "active" : ""}
                      title={template.description}
                      onClick={() => applyMotionLabTemplate(key)}
                    >{template.label}</button>
                  ))}
              </div>
            </div>
            {motionLabTemplateName && (
              <div className="motion-lab-note">
                {MOTION_LAB_TEMPLATES[motionLabTemplateName].description}。適用後も各スライダーで微調整できます。
              </div>
            )}
            {renderRange("動きの強さ", Math.round(motionLabIntensity * 100), 50, 150, 5, (value) => {
              const scale = value / 100;
              setMotionLabIntensity(scale);
              setMotionLabBreathAmplitude(scale);
              setMotionLabBodySwayAmplitude(scale);
              setMotionLabHairWind(Number((MOTION_LAB_HAIR_DEFAULTS.wind * scale).toFixed(4)));
              setMotionLabHairDrive(Number((MOTION_LAB_HAIR_DEFAULTS.drive * scale).toFixed(3)));
              setMotionLabArmMaxAngle(clamp(Number((MOTION_LAB_ARM_DEFAULTS.maxAngle * scale).toFixed(3)), 0, 0.3));
              setMotionLabChestMax(clamp(Number((MOTION_LAB_CHEST_DEFAULTS.max * scale).toFixed(1)), 0, 8));
              setMotionLabParallaxScale(clamp(scale, 0, 1.5));
            }, "%")}
            <div className="motion-lab-section-title">
              <strong>エフェクト</strong>
              <div className="motion-lab-segmented">
                <button onClick={() => setMotionLabEffects({ ...MOTION_LAB_EFFECT_DEFAULTS, earTwitch: motionLabEffects.earTwitch })}>すべてON</button>
                <button onClick={() => setMotionLabEffects(Object.fromEntries(MOTION_LAB_EFFECT_DEFS.map(def => [def.key, false])) as Record<MotionLabEffectKey, boolean>)}>すべてOFF</button>
              </div>
            </div>
            <div className="motion-lab-effect-list">
              {MOTION_LAB_EFFECT_DEFS.filter(def => {
                if (def.key === "arm" || def.key === "lift") return !!(motionLabParts?.armL || motionLabParts?.armR);
                if (def.key === "chest") return !!motionLabParts?.chest;
                if (def.key === "gaze") return !!(motionLabParts?.eyewhite && motionLabParts?.irides);
                if (def.key === "earTwitch") return Object.keys(motionLabParts?.sways ?? {}).some(name => /(^|_)ears?(_|$)/i.test(name));
                if (def.key === "hairBack") return !!motionLabParts?.hairBack;
                if (def.key === "blink") return (motionLabParts?.eyeFrames.length ?? 0) > 1;
                return true;
              }).map(def => (
                <div key={def.key} className="motion-lab-effect-row" title={def.hint}>
                  <label>
                    <input
                      type="checkbox"
                      checked={motionLabEffects[def.key]}
                      onChange={(e) => setMotionLabEffect(def.key, e.target.checked)}
                    />
                    <span>{def.label}</span>
                  </label>
                  {motionLabEffects[def.key] && motionLabEffectSliders[def.key] ? (
                    <>
                      <input
                        type="range"
                        className="motion-lab-effect-slider"
                        min={motionLabEffectSliders[def.key]!.min}
                        max={motionLabEffectSliders[def.key]!.max}
                        step={motionLabEffectSliders[def.key]!.step}
                        value={motionLabEffectSliders[def.key]!.value}
                        onChange={(e) => motionLabEffectSliders[def.key]!.set(Number(e.target.value))}
                      />
                      <small className="motion-lab-effect-value">
                        {motionLabEffectSliders[def.key]!.format(motionLabEffectSliders[def.key]!.value)}
                      </small>
                    </>
                  ) : (
                    <span className="motion-lab-effect-slider-spacer" />
                  )}
                  <button
                    className="motion-lab-effect-solo"
                    title={`この効果だけONにして単体で体感する: ${def.hint}`}
                    onClick={() => soloMotionLabEffect(def.key)}
                  >ソロ</button>
                </div>
              ))}
            </div>
            <div className="motion-lab-note">
              各エフェクトを個別にON/OFFできます。「ソロ」でその効果だけを再生し、1つずつ体感できます（依存する効果は自動でONになります）。
            </div>
            <div className="motion-lab-simple-toggles">
              {(motionLabParts?.layerOrder?.length ?? 0) === 0 && (
                <label><input type="checkbox" checked={motionLabArmBehindBody} onChange={(e) => setMotionLabArmBehindBody(e.target.checked)} />腕を体の後ろ</label>
              )}
            </div>
            {(motionLabParts?.layerOrder?.length ?? 0) > 0 && (
              <div className="motion-lab-note">
                レイヤー順は素材の layer-order.json（Step4のレイヤー調整で保存した並び）に従っています: {motionLabParts?.layerOrder.join(" → ")}
              </div>
            )}
            <div className="motion-lab-note">
              「動きの強さ」は呼吸・体・髪風・腕振幅・胸振幅・首振りをまとめて変更するおまかせノブです（詳細値は下の「詳細パラメータ」で個別調整できます）。
              プリセット（おとなしめ/ふつう/元気）は揺れ全体の基礎倍率で、「動きの強さ」と重ねて効きます。
            </div>
          </div>

          <details className="motion-lab-advanced">
            <summary>詳細パラメータ</summary>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>口パク</strong>
              <div className="motion-lab-segmented three">
                <button className={motionLabMethod === "baseline" ? "active" : ""} onClick={() => setMotionLabMethod("baseline")}>直接</button>
                <button className={motionLabMethod === "smooth" ? "active" : ""} onClick={() => setMotionLabMethod("smooth")}>スムーズ</button>
                <button className={motionLabMethod === "bridge" ? "active" : ""} onClick={() => setMotionLabMethod("bridge")}>ブリッジ</button>
              </div>
            </div>
            {renderRange("attack", motionLabAttackMs, 40, 180, 5, setMotionLabAttackMs, "ms")}
            {renderRange("release", motionLabReleaseMs, 80, 260, 5, setMotionLabReleaseMs, "ms")}
            {renderRange("crossfade", motionLabCrossfadeMs, 0, 120, 5, setMotionLabCrossfadeMs, "ms")}
            {renderRange("rest", Math.round(motionLabRestBias * 100), 0, 100, 1, value => setMotionLabRestBias(value / 100), "%")}
            {renderRange("smooth", Math.round(motionLabShapeSmoothing * 100), 0, 100, 1, value => setMotionLabShapeSmoothing(value / 100), "%")}
            {renderRange("bridge", Math.round(motionLabBridgeBias * 100), 0, 85, 1, value => setMotionLabBridgeBias(value / 100), "%")}
            <div className="motion-lab-text-row">
              <input
                type="text"
                value={motionLabText}
                onChange={(e) => setMotionLabText(e.target.value)}
                placeholder="ひらがな・カタカナで入力（例: こんにちは）"
              />
              <button className="btn btn-secondary" disabled={!motionLabParts} onClick={() => {
                setMotionLabCustomTimeline(motionLabTimelineFromText(motionLabText));
                resetMotionLabRuntime(motionLabBaselineRuntimeRef.current);
                resetMotionLabRuntime(motionLabCandidateRuntimeRef.current);
                setMotionLabPlaying(false);
                window.setTimeout(() => setMotionLabPlaying(true), 0);
              }}>テキスト再生</button>
              <button className="btn btn-secondary" disabled={!motionLabCustomTimeline} onClick={() => setMotionLabCustomTimeline(null)}>内蔵あいうえお</button>
            </div>
            <div className="motion-lab-note">
              直接=A1調音結合（母音間で閉じずレイヤー即差替え・採用決定）/ スムーズ=A0改善 / ブリッジ=A2クロスフェード（見送り案）。
              attack/release=A4開度エンベロープ（uLipSync系のSmoothDamp・ADSRの知見。SpriTalk 4方式比較で採用）。
              rest=無声時の閉じ具合、smooth=開度追従の滑らかさ（A1のsmoothTime相当）
            </div>
          </div>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>髪・身体</strong>
              <div className="motion-lab-segmented three">
                <button className={motionLabPreset === "calm" ? "active" : ""} onClick={() => setMotionLabPreset("calm")}>calm</button>
                <button className={motionLabPreset === "normal" ? "active" : ""} onClick={() => setMotionLabPreset("normal")}>normal</button>
                <button className={motionLabPreset === "lively" ? "active" : ""} onClick={() => setMotionLabPreset("lively")}>lively</button>
              </div>
            </div>
            <div className="motion-lab-segmented">
              <button className={motionLabLayerMode === "spring" ? "active" : ""} onClick={() => setMotionLabLayerMode("spring")}>spring</button>
              <button className={motionLabLayerMode === "mesh" ? "active" : ""} onClick={() => setMotionLabLayerMode("mesh")}>mesh</button>
            </div>
            <div className="motion-lab-segmented">
              <button className={motionLabHairEngine === "spring" ? "active" : ""} onClick={() => setMotionLabHairEngine("spring")}>バネ物理</button>
              <button className={motionLabHairEngine === "wave" ? "active" : ""} onClick={() => setMotionLabHairEngine("wave")}>波揺れ（ろてじん式）</button>
            </div>
            {motionLabHairEngine === "wave" &&
              renderRange("波の強さ", Math.round(motionLabHairWaveStrength * 100), 0, 200, 5, value => setMotionLabHairWaveStrength(value / 100), "%")}
            {motionLabLayerMode === "mesh" && (
              <div className="motion-lab-segmented">
                <button className={!motionLabStrandsEnabled ? "active" : ""} onClick={() => setMotionLabStrandsEnabled(false)}>一枚チェーン</button>
                <button className={motionLabStrandsEnabled ? "active" : ""} onClick={() => setMotionLabStrandsEnabled(true)}>房分割</button>
              </div>
            )}
            {renderRange("breath", Math.round(motionLabBreathAmplitude * 100), 0, 160, 1, value => setMotionLabBreathAmplitude(value / 100), "%")}
            {renderRange("body sway", Math.round(motionLabBodySwayAmplitude * 100), 0, 180, 1, value => setMotionLabBodySwayAmplitude(value / 100), "%")}
            {renderRange("発話バウンス", Number(motionLabPyokoBounce.toFixed(1)), 0, 12, 0.5, setMotionLabPyokoBounce, "px")}
            {renderRange("柔らかさ k", motionLabHairK, 10, 200, 5, setMotionLabHairK)}
            {renderRange("収まり c", motionLabHairC, 1, 30, 1, setMotionLabHairC)}
            {renderRange("風 wind", Number((motionLabHairWind * 1000).toFixed(0)), 0, 60, 2, value => setMotionLabHairWind(value / 1000), "‰")}
            {renderRange("体追従 drive", Number((motionLabHairDrive * 100).toFixed(0)), 0, 20, 1, value => setMotionLabHairDrive(value / 100), "%")}
            {renderRange("後ろ髪の揺れ", Math.round(motionLabHairBackScale * 100), 0, 150, 5, value => setMotionLabHairBackScale(value / 100), "%")}
            <div className="motion-lab-note">
              プリセットは揺れ振幅の倍率: calm=呼吸0.65/体0.55/髪0.55、normal=1.0、lively=呼吸1.25/体1.35/髪1.45。
              spring=B1遅延追従バネ（ろてじん氏 PuruPuruPNGTuber由来の「頭に遅れて髪が追従」原理）/
              mesh=B3角度チェーン（B1を縦6分割に多段化・Live2D物理系。SpriTalk比較で採用決定）。
              房分割=毛先輪郭ピークから房中心線を自動検出し、房ごとの独立チェーンをガウシアン重みでソフトブレンド（852話氏の本家方式）。前髪・後ろ髪の両方に適用され、裂けずに房ごとの位相差が出ます。
              k=硬さ（低いほど遅延大）、c=揺り戻しの収まり、wind=常時微風（sin＋1Dノイズ）、drive=体の動きへの反応量。
              まばたきは自動（PuruPuru参考: 閉90ms/開130ms・2〜10秒間隔）
            </div>
          </div>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>首振り・視線</strong>
            </div>
            {renderRange("パララックス", Math.round(motionLabParallaxScale * 100), 0, 150, 5, value => setMotionLabParallaxScale(value / 100), "%")}
            <div className="motion-lab-note">
              パララックス首振り=レイヤー深度×水平シフト＋シアー（852話氏 Anime2.5DRig由来・§8.3）。
              駆動はノイズドリフト＋発話開始の頷きバネ。0%で無効。
              深度: 後髪-0.6 / 体0 / 腕0.1 / 目口0.35 / 前髪0.8。
              視線ドリフトは eyewhite.png + irides.png 読込時のみ自動有効
              （基本正面・発話中は正面復帰=SpriTalk特性§8.6）。highlight.png は±1.5pxドリフト（PuruPuru目元演出参考）
            </div>
          </div>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>回転軸・可動域</strong>
            </div>
            <div className="motion-lab-segmented">
              {([
                ["hair", "前髪", !!motionLabParts?.hair],
                ["hair_back", "後ろ髪", !!motionLabParts?.hairBack],
                ["arm_l", "左腕", !!motionLabParts?.armL],
                ["arm_r", "右腕", !!motionLabParts?.armR],
              ] as Array<[string, string, boolean]>).filter(([, , exists]) => exists).map(([part, label]) => (
                <button
                  key={part}
                  className={motionLabPivotEditPart === part ? "active" : ""}
                  onClick={() => setMotionLabPivotEditPart(prev => (prev === part ? null : part))}
                >{label}</button>
              ))}
            </div>
            {motionLabPivotEditPart && (
              <>
                {renderRange(
                  "可動域",
                  motionLabRangesDeg[motionLabPivotEditPart] ?? 0,
                  0,
                  90,
                  1,
                  value => setMotionLabRangesDeg(prev => ({ ...prev, [motionLabPivotEditPart]: value })),
                  "°",
                )}
                {renderRange(
                  "揺れ幅",
                  Math.round((motionLabSwingScale[motionLabPivotEditPart] ?? 1) * 100),
                  0,
                  300,
                  10,
                  value => setMotionLabSwingScale(prev => ({ ...prev, [motionLabPivotEditPart]: value / 100 })),
                  "%",
                )}
                <div className="motion-lab-text-row">
                  <small>
                    回転軸: {motionLabPivots[motionLabPivotEditPart]
                      ? `${motionLabPivots[motionLabPivotEditPart].x}, ${motionLabPivots[motionLabPivotEditPart].y}`
                      : "自動推定"}（候補プレビューをクリックで指定）
                  </small>
                  {motionLabPivots[motionLabPivotEditPart] && (
                    <button className="btn btn-secondary" onClick={() => setMotionLabPivots(prev => {
                      const next = { ...prev };
                      delete next[motionLabPivotEditPart];
                      return next;
                    })}>自動に戻す</button>
                  )}
                </div>
              </>
            )}
            <div className="motion-lab-note">
              パーツを選ぶと候補プレビューに回転軸マーカー（＋印）が出ます。プレビューをクリックすると回転軸を移動できます。
              可動域=回転角の上限（±度、0=制限なし・既定の物理クランプのみ）。
              揺れ幅=このパーツだけの振れ倍率（腕=振れ角、髪=ワープ変位/回転に乗算。物理クランプ後に掛かるので確実に効きます）。
              前髪・後ろ髪は mesh / 波揺れ時は回転軸のY位置が「揺れの根元」（それより上は固定）として効きます。
            </div>
          </div>

          {(motionLabParts?.armL || motionLabParts?.armR) ? (
            <div className="motion-lab-section">
              <div className="motion-lab-section-title">
                <strong>腕揺れ</strong>
              </div>
              {renderRange("最大角", Number((motionLabArmMaxAngle * 100).toFixed(0)), 0, 60, 1, value => setMotionLabArmMaxAngle(value / 100), "×0.01rad")}
              {renderRange("揺れ幅", Math.round(motionLabArmSwayAmp * 100), 0, 300, 10, value => setMotionLabArmSwayAmp(value / 100), "%")}
              {renderRange("回転軸位置", Math.round(motionLabArmPivotRatio * 100), 0, 60, 2, value => setMotionLabArmPivotRatio(value / 100), "%")}
              <div className="motion-lab-note">
                ON/OFFは「かんたん設定 › エフェクト」の腕揺れ・肩の弾みで切り替え。
                B3と同じ角度チェーン3分割をarm_l/arm_rに適用（ろてじん氏の遅延追従原理の応用）。
                揺れ幅=左右方向の駆動量（常時微揺れ＋体速度カップリング）の倍率。最大角=振れ角の上限。
                回転軸位置=肩ピボットのY位置（0%=不透明部の上端中央を自動推定、下げるほど回転中心が下がり振りが穏やかに見える）。
              </div>
            </div>
          ) : null}

          {motionLabParts?.chest ? (
            <div className="motion-lab-section">
              <div className="motion-lab-section-title">
                <strong>胸揺れ</strong>
              </div>
              {renderRange("揺れ幅", Number(motionLabChestMax.toFixed(1)), 0, 20, 0.5, setMotionLabChestMax, "px")}
              <div className="motion-lab-note">
                縦バネ1本・低周波強減衰（852話氏 Anime2.5DRig由来）。呼吸のY速度＋発話開始の二次撃力（肩より遅れて小さく）で駆動
              </div>
            </div>
          ) : null}

          </details>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>評価</strong>
              <select value={motionLabVerdict} onChange={(event) => setMotionLabVerdict(event.target.value as MotionLabVerdict)}>
                <option value="undecided">未判断</option>
                <option value="promising">有力</option>
                <option value="hold">保留</option>
                <option value="reject">除外</option>
              </select>
            </div>
            {renderReviewRange("mouthSmoothness")}
            {renderReviewRange("vowelReadability")}
            {renderReviewRange("bodyNaturalness")}
            {renderReviewRange("hairBodySeparation")}
            {renderReviewRange("settingSimplicity")}
            {renderReviewRange("migrationConfidence")}
            <textarea
              className="motion-lab-review-note"
              value={motionLabReviewNote}
              onChange={(event) => setMotionLabReviewNote(event.target.value)}
              rows={3}
              placeholder="採用/保留理由"
            />
          </div>
        </section>

        <section className="motion-lab-preview-panel">
          <div className="motion-lab-preview-toolbar">
            <button className="btn btn-secondary" disabled={!motionLabParts} onClick={() => setMotionLabPlaying(prev => !prev)}>
              {motionLabPlaying ? "停止" : "再生"}
            </button>
            <button className="btn btn-secondary" disabled={!motionLabParts} onClick={() => {
              // 物理状態を作り直し、登場撃力＋位相ランダム化を再現（presence検証用）
              resetMotionLabRuntime(motionLabBaselineRuntimeRef.current);
              resetMotionLabRuntime(motionLabCandidateRuntimeRef.current);
              setMotionLabPlaying(false);
              window.setTimeout(() => setMotionLabPlaying(true), 0);
            }}>
              リセット
            </button>
          </div>

          <div className="motion-lab-stage">
            {motionLabParts ? (
              <>
                <div className="motion-lab-stage-comparison">
                  <div className="motion-lab-preview-lane">
                    <div className="motion-lab-lane-label"><strong>基準</strong><span>直切替 / 一体揺れ</span></div>
                    <canvas ref={motionLabBaselineCanvasRef} />
                  </div>
                  <div className="motion-lab-preview-lane">
                    <div className="motion-lab-lane-label"><strong>候補</strong><span>{candidateLabel}</span></div>
                    <canvas
                      ref={motionLabCandidateCanvasRef}
                      style={motionLabPivotEditPart ? { cursor: "crosshair" } : undefined}
                      onClick={(e) => {
                        if (!motionLabPivotEditPart || !motionLabParts) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        if (rect.width <= 0 || rect.height <= 0) return;
                        const x = ((e.clientX - rect.left) / rect.width) * motionLabParts.width;
                        const y = ((e.clientY - rect.top) / rect.height) * motionLabParts.height;
                        setMotionLabPivots(prev => ({
                          ...prev,
                          [motionLabPivotEditPart]: { x: Math.round(x), y: Math.round(y) },
                        }));
                      }}
                    />
                  </div>
                </div>
                {motionLabImagesLoading && <span className="motion-lab-placeholder">画像読込中...</span>}
              </>
            ) : (
              <span className="motion-lab-placeholder">04_spritalk_parts</span>
            )}
          </div>

          <div className="motion-lab-mouth-strip">
            {MOTION_LAB_MOUTH_KEYS.map(key => {
              const count = motionLabParts?.mouths[key]?.length ?? 0;
              return (
                <span key={key} className={count > 0 ? "ready" : ""}>
                  <b>{MOTION_LAB_MOUTH_LABELS[key]}</b>
                  <small>{count}</small>
                </span>
              );
            })}
          </div>
          <div className="motion-lab-metric-strip">
            <span><b>timeline</b><small>{(motionLabCustomTimeline?.timeline ?? MOTION_LAB_TIMELINE).length} events / {motionLabCustomTimeline?.durationMs ?? MOTION_LAB_DURATION_MS}ms{motionLabCustomTimeline ? " (text)" : ""}</small></span>
            <span><b>mouth</b><small>{smoothingSummary}</small></span>
            <span><b>layer</b><small>{motionLabLayerMode}, {motionLabPreset}, k{motionLabHairK}/c{motionLabHairC} wind{(motionLabHairWind * 1000).toFixed(0)}‰</small></span>
            <span><b>eye</b><small>{motionLabParts?.eyeFrames.length ? `auto blink 90/130ms (${motionLabParts.eyeFrames.length}f)` : "eye連番なし"}</small></span>
          </div>
        </section>
      </main>
    );
  }

  void [
    hairPreview, hairBackPreview, correctionOutputPath, baseStep, dragTarget,
    frameCount, setFrameCount, outputPath, completedDiffs, diffPreviewTick, diffPreviewZoom, setDiffPreviewZoom,
    interpPaths, interpOriginals, interpGenerating, interpStep, mouthMode, mouthMaskSettings, mouthMaskPreviews,
    mouthMaskPreviewing, mouthPreviewZooms, mouthPreviewPans, activePreviewKey, canGenerate, anyMouthMaskPreviewing,
    readyPairsForMask, missingMouthMaskCount, allReadyPairsHaveMasks, canOpenInterpStep, goInterpStep, activeMaskPair,
    activeMaskSlot, activeMaskSetting, activeMaskPreview, activeMaskZoom, activeMaskPan, activeDiffPreview, hasInterpWork,
    pingPongFrameIndex, pickInterpPsd, pickInterpOriginal, previewMouthMask, updateMouthMaskSetting, refreshMouthMaskPreview,
    setMouthPreviewZoom, onMouthPreviewPointerDown, onMouthPreviewPointerMove, onMouthPreviewPointerUp,
    resetMouthPreviewView, handleGenerateAll, openOutputFolder, handleCreateMouthMasks, discardMouthMask, returnFromInterpInput,
    loadPsd, proceedToHairEdit, openHairEditorWithPreview, proceedToBodyEdit, loadCorrectionPsd,
    setAllBodyOpacities, returnFromBaseFlow, goBaseStep, baseStepEnabled,
  ];

  function renderModeSelect() {
    return (
      <main className="mode-select-screen">
        <section className="workspace-start-panel">
          <button className="primary-workflow-card" onClick={() => void startExpressionWorkspace("new")}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">NEW WORKSPACE</span>
              <strong>はじめから</strong>
              <p>作業フォルダを選択して、画像選択からSpriTalk用出力まで順番に進めます。</p>
            </div>
            <span className="primary-workflow-cta">作業フォルダを選択</span>
          </button>
          <button className="primary-workflow-card secondary" onClick={() => void startExpressionWorkspace("resume")}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">RESUME</span>
              <strong>つづきから</strong>
              <p>既存の作業フォルダを選択して、保存済みの工程から再開します。</p>
            </div>
            <span className="primary-workflow-cta">既存フォルダを開く</span>
          </button>
          <button className="primary-workflow-card secondary motion-lab-entry" onClick={() => { setMode("motion_lab"); setStatus("Motion Preview Lab"); }}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">MOTION LAB</span>
              <strong>モーション比較</strong>
              <p>作成済みのSpriTalk素材で、口パク補正と髪・身体の揺れを試します。</p>
            </div>
            <span className="primary-workflow-cta">比較を開く</span>
          </button>
        </section>
      </main>
    );
  }

  function renderSimpleInput(title: string, description: string, action: () => void | Promise<void>) {
    return (
      <main className="mode-select-screen base-input-screen">
        <div className="input-card-center">
          <h2 className="input-card-title">{title}</h2>
          <p className="input-hint">{description}</p>
          <div className="file-input-row-large">
            <button className="btn btn-primary" onClick={() => void action()} disabled={loading}>PSDを選択</button>
            <span className="slot-path-inline-large">{loadResult ? `${loadResult.detected_layers.length}レイヤー 読み込み済み` : "未選択"}</span>
          </div>
          {loading && <div className="progress-bar indeterminate" style={{ marginTop: 12 }}><div className="fill" /></div>}
          <div className="step-nav-actions base-input-actions">
            <button className="btn btn-secondary" onClick={returnToModeSelect}>戻る</button>
          </div>
        </div>
      </main>
    );
  }

  function renderLayerEditor() {
    // 腕分離オプション: See-Throughのhandwear(-l/-r)レイヤーを arm_l/arm_r として
    // 独立出力する（Motion Lab / SpriTalkの腕揺れ用）。OFFなら従来どおりbodyへ統合
    const armSplitLayers = layerOrder.filter(name => /^handwear([-_][lr])?$/i.test(name));
    const armSplitActive = armSplitLayers.some(name => (layerMapping[name] ?? "").startsWith("arm_"));
    const toggleArmSplit = (enable: boolean) => {
      setLayerMapping(prev => {
        const next = { ...prev };
        for (const name of armSplitLayers) {
          if (enable) {
            next[name] = /[-_]r$/i.test(name) ? "arm_r" : "arm_l";
          } else {
            delete next[name];
          }
        }
        return next;
      });
    };
    // 獣耳分離オプション: ears(-l/-r) / headwear（犬耳・獣耳がheadwear扱いのキャラ用）を
    // sway_ear* として独立出力（Motion Lab の獣耳ピコピコ・SpriTalk の汎用揺れパーツ用）
    const earSplitLayers = layerOrder.filter(name => /^(ears|headwear)([-_][lr])?$/i.test(name));
    const earSplitActive = earSplitLayers.some(name => (layerMapping[name] ?? "").startsWith("sway_"));
    const toggleEarSplit = (enable: boolean) => {
      setLayerMapping(prev => {
        const next = { ...prev };
        for (const name of earSplitLayers) {
          if (enable) {
            // 左右サフィックス無し（headwear等の一枚もの）は sway_ear として出力
            next[name] = /[-_]r$/i.test(name) ? "sway_ear_r" : /[-_]l$/i.test(name) ? "sway_ear_l" : "sway_ear";
          } else {
            delete next[name];
          }
        }
        return next;
      });
    };
    return (
      <main className="panel-right base-edit-panel">
        <div className="preview-and-layers">
          <div
            className="preview-viewport"
            ref={previewRef}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {bodyPreview ? (
              <img src={bodyPreview} alt="合成プレビュー" className="preview-img" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: isPanning ? "grabbing" : "grab" }} draggable={false} />
            ) : <span className="placeholder">合成プレビュー</span>}
            {bodyPreview && patchDraftSource && (
              <canvas
                ref={maskCanvasRef}
                className="patch-mask-canvas"
                style={previewImageStyle()}
                onPointerDown={onPatchMaskPointerDown}
                onPointerMove={onPatchMaskPointerMove}
                onPointerUp={onPatchMaskPointerUp}
                onPointerCancel={onPatchMaskPointerUp}
                onPointerLeave={onPatchMaskPointerLeave}
              />
            )}
            {bodyPreview && patchDraftSource && brushCursor.visible && <div className="patch-brush-cursor" style={brushCursorStyle()} />}
          </div>

          {bodyCategory && layerOrder.length > 0 && (
            <div className="layer-sidebar" onPointerMove={onDragPointerMove} onPointerUp={onDragPointerUp}>
              <div className="layer-sidebar-title-row">
                <div>
                  <div className="layer-sidebar-header">レイヤー順序</div>
                  <div className="layer-sidebar-hint">上が手前</div>
                </div>
                {mode === "correction" && <button className="btn-layer-add" onClick={() => void addCorrectionLayerImage()} disabled={loading}>PNG追加</button>}
              </div>
              <div className="layer-bulk-row">
                <button className="btn-layer-bulk" onClick={() => void setAllLayerVisibility(true)}>全ON</button>
                <button className="btn-layer-bulk" onClick={() => void setAllLayerVisibility(false)}>全OFF</button>
                <button className="btn-layer-bulk" onClick={() => void setAllBodyOpacities(1)}>全100%</button>
                {!!expressionWorkspace && <button className="btn-layer-bulk" onClick={() => void applyRecommendedLayerOrder()}>推奨順</button>}
              </div>
              {armSplitLayers.length > 0 && (
                <label className="layer-arm-split-toggle" title="腕レイヤー（handwear-l/-r）を arm_l.png / arm_r.png として分離出力します。SpriTalk・Motion Labの腕揺れ用">
                  <input type="checkbox" checked={armSplitActive} onChange={(e) => toggleArmSplit(e.target.checked)} />
                  <span>腕を分離出力（arm_l / arm_r）</span>
                </label>
              )}
              {earSplitLayers.length > 0 && (
                <label className="layer-arm-split-toggle" title="獣耳レイヤー（ears-l/-r または headwear）を sway_ear*.png として分離出力します。Motion Labの獣耳ピコピコ・揺れパーツ用。犬耳がheadwear扱いのキャラにも対応">
                  <input type="checkbox" checked={earSplitActive} onChange={(e) => toggleEarSplit(e.target.checked)} />
                  <span>獣耳を分離出力（sway_ear*）</span>
                </label>
              )}
              <div className="layer-bulk-row">
                <button
                  className={`btn-layer-bulk${chestMaskDataUrl ? " active" : ""}`}
                  title="See-Throughにはchestレイヤーが無いため、bodyから胸部を塗って手動で切り出します（852話式・Motion Labの胸揺れ用）"
                  onClick={initChestCut}
                >
                  胸を切出{chestMaskDataUrl ? "済み" : ""}
                </button>
                {chestMaskDataUrl && (
                  <button className="btn-layer-bulk" onClick={() => setChestMaskDataUrl(null)}>胸切出を取消</button>
                )}
              </div>
              <div className="layer-sidebar-list">
                {layerOrder.map((name, idx) => {
                  const layer = getBodyOrderItem(name);
                  if (!layer) return null;
                  const patch = layerPatches.find(p => p.id === name);
                  const sourceName = patch?.sourceLayer ?? name;
                  const opacity = layerOpacities[name] ?? 1;
                  return (
                    <div key={name} className={`layer-sidebar-item${draggedIdx === idx ? " dragging" : ""}${selectedBodyLayer === name ? " selected" : ""}${layer.isPatch ? " patch" : ""}`} onClick={() => setSelectedBodyLayer(name)}>
                      <span className="drag-handle" onPointerDown={(e) => onDragPointerDown(e, idx)}>☰</span>
                      <input type="checkbox" checked={enabledLayers[name] !== false} onChange={(e) => void handleLayerToggle(name, e.target.checked)} />
                      <img src={layer.thumbnail} alt={layer.name} className="layer-sidebar-thumb" />
                      <span className="layer-sidebar-name">{layer.name}</span>
                      {opacity < 1 && <span className="layer-offset-badge">{Math.round(opacity * 100)}%</span>}
                      {layer.isPatch ? (
                        <button className="btn-layer-mini" onClick={(e) => { e.stopPropagation(); void removeLayerPatch(name); }}>削除</button>
                      ) : (
                        <button className="btn-layer-mini" onClick={(e) => { e.stopPropagation(); initPatchMask(sourceName); }}>切出</button>
                      )}
                    </div>
                  );
                })}
              </div>
              {patchDraftSource && (
                <div className="layer-adjust-panel patch-panel">
                  <div className="layer-adjust-title">
                    {patchDraftSource === CHEST_CUT_SENTINEL ? "胸を切出" : `切り出し作成: ${patchDraftSource}`}
                  </div>
                  <div className="layer-adjust-values">
                    {patchDraftSource === CHEST_CUT_SENTINEL
                      ? "bodyのプレビュー上で胸部を塗ってください。保存時に chest.png として分離されます。"
                      : "塗った範囲を別レイヤーとして切り出し、元レイヤーから抜きます。"}
                  </div>
                  <div className="patch-tool-row">
                    <button className={`btn-nudge ${patchTool === "paint" ? "active" : ""}`} onClick={() => setPatchTool("paint")}>塗る</button>
                    <button className={`btn-nudge ${patchTool === "erase" ? "active" : ""}`} onClick={() => setPatchTool("erase")}>消す</button>
                  </div>
                  <label className="patch-brush-size-label">
                    <span>ブラシサイズ {patchBrushSize}px</span>
                    <input type="range" min={4} max={96} value={patchBrushSize} onChange={(e) => setPatchBrushSize(Number(e.target.value))} />
                  </label>
                  <label className="patch-brush-size-label">
                    <span>ぼかし {Math.round(patchBrushSoftness * 100)}%</span>
                    <input type="range" min={0} max={100} value={Math.round(patchBrushSoftness * 100)} onChange={(e) => setPatchBrushSoftness(Number(e.target.value) / 100)} />
                  </label>
                  <div className="patch-tool-row">
                    <button className="btn-nudge btn-nudge-reset" onClick={clearPatchMask}>クリア</button>
                    <button className="btn-nudge btn-nudge-reset" onClick={() => { setPatchDraftSource(""); setSam3SelectMode(false); }}>取消</button>
                    <button className="btn-nudge btn-nudge-reset" onClick={commitPatchMask}>追加</button>
                  </div>
                </div>
              )}
              {selectedBodyLayer && (
                <div className="layer-adjust-panel opacity-panel">
                  <div className="layer-adjust-title">表示透明度: {getBodyOrderItem(selectedBodyLayer)?.name ?? selectedBodyLayer}</div>
                  <div className="layer-adjust-values">{Math.round(selectedLayerOpacity * 100)}%</div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(selectedLayerOpacity * 100)}
                    onChange={(e) => updateLayerOpacityDraft(selectedBodyLayer, Number(e.target.value) / 100)}
                    onPointerUp={() => void commitLayerOpacity()}
                    onMouseUp={() => void commitLayerOpacity()}
                    onTouchEnd={() => void commitLayerOpacity()}
                    onBlur={() => void commitLayerOpacity()}
                  />
                  <div className="opacity-preset-row">
                    <button className="btn-nudge btn-nudge-reset" onClick={() => void setSelectedBodyOpacity(0)}>このレイヤー0%</button>
                    <button className="btn-nudge btn-nudge-reset" onClick={() => void setSelectedBodyOpacity(1)}>このレイヤー100%</button>
                    <button className={`btn-nudge btn-nudge-reset opacity-highlight-toggle${overlapHighlightEnabled ? " active" : ""}`} onClick={() => void toggleOverlapHighlight()}>
                      重なり表示 {overlapHighlightEnabled ? "ON" : "OFF"}
                    </button>
                  </div>
                </div>
              )}
              <div className="layer-sidebar-hint">下が奥</div>
            </div>
          )}
        </div>
        {bodyPreview && (
          <div className="zoom-controls">
            <button className="btn-zoom" onClick={() => setZoom(prev => Math.min(10, prev * 1.3))}>+</button>
            <span className="zoom-level">{Math.round(zoom * 100)}%</span>
            <button className="btn-zoom" onClick={() => setZoom(prev => Math.max(0.1, prev * 0.7))}>-</button>
            <button className="btn-zoom btn-zoom-reset" onClick={resetZoom}>リセット</button>
          </div>
        )}
        <div className="base-flow-controls editing bottom-actions">
          <div className="interp-action-panel base-flow-action body-action">
            <div>
              <div className="action-panel-title">{mode === "correction" ? "See-Through補正" : "素体レイヤー調整"}</div>
              <div className="action-panel-hint">レイヤー順、ON/OFF、切り出し、透明度を確認して保存します。</div>
            </div>
            <div className="step-nav-actions base-output-actions">
              <button className="btn btn-secondary" onClick={returnToModeSelect}>戻る</button>
              {mode === "correction" ? (
                <button className="btn btn-primary" onClick={handleExportCorrection} disabled={loading || !bodyPreview}>{loading ? "保存中..." : "PNG保存"}</button>
              ) : (
                <button className="btn btn-primary" onClick={handleCreateBase} disabled={loading || !bodyPreview}>{loading ? "出力中..." : "保存して戻る"}</button>
              )}
            </div>
          </div>
        </div>
      </main>
    );
  }

  function renderMainContent() {
    if (mode === "select") return renderModeSelect();
    if (mode === "workspace") return renderWorkspaceMode();
    if (mode === "motion_lab") return renderMotionLabMode();
    if (mode === "base_input") return renderSimpleInput("素体出力", "See-Throughで分解したPSDを読み込みます。", loadPsd);
    if (mode === "hair_edit") return renderSimpleInput("Hairレイヤー確認", "この旧画面はワークスペースフローへ統合中です。", proceedToBodyEdit);
    if (mode === "base_edit") return renderLayerEditor();
    if (mode === "correction") return mappingPreview ? renderLayerEditor() : renderSimpleInput("See-Through補正", "補正したいPSDを読み込みます。", loadCorrectionPsd);
    if (mode === "interp") return renderSimpleInput("フレーム補間", "旧補間モードはワークスペースフローへ統合中です。", returnFromInterpInput);
    return null;
  }

  return (
    <div className={`app theme-${themeMode}`}>
      <div className="app-header">
        <h1>PachiPakuGen</h1>
        <span className="version">v0.3.0</span>
        <div className="app-header-actions">
          <button className="btn btn-secondary theme-toggle-button" onClick={() => setThemeMode(prev => prev === "dark" ? "light" : "dark")}>
            {themeMode === "dark" ? "ライトmode" : "ダークmode"}
          </button>
          {mode !== "select" && <button className="btn btn-secondary" onClick={returnToModeSelect}>制作ホーム</button>}
        </div>
      </div>

      {(mode === "base_input" || mode === "hair_edit" || mode === "base_edit") && (
        <div className="interp-header base-flow-header">
          <h2 className="interp-header-title">{expressionWorkspace ? "Step 4 素体調整" : "素体出力"}</h2>
          <div className="interp-header-note">分解済みレイヤーを確認し、作業フォルダへ保存します。</div>
        </div>
      )}

      {mode === "correction" && (
        <div className="interp-header base-flow-header">
          <h2 className="interp-header-title">See-Through補正</h2>
          <div className="interp-header-note">See-Throughの分解結果を手動で切り出し、PNG保存します。</div>
        </div>
      )}

      <div className={`main-content${mode === "base_input" || mode === "hair_edit" || mode === "base_edit" || mode === "correction" ? " base-main-content" : ""}`}>
        {renderMainContent()}
      </div>

      {mode !== "workspace" && (
        <div className="status-bar">
          {error && <span className="error-msg">{error}</span>}
          {!error && status}
        </div>
      )}
    </div>
  );
}

export default App;
