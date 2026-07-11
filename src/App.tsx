import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
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
  sanitizeSeeThroughLogMessage,
  formatElapsed,
  CHEST_CUT_SENTINEL,
  workspacePreviewItemKey,
  workspacePreviewItemLabel,
  WORKSPACE_ADJUST_PART_KEYS,
  DEFAULT_PART_ADJUSTMENT,
  EYE_PAIRS,
  MOUTH_PAIRS_SINGLE,
  MOUTH_PAIRS_VOWELS,
} from "./workspace/types";
import { loadMotionLabImage } from "./motionLab/render";
import { MotionTunePanel } from "./motionLab/MotionTunePanel";

type Mode = "select" | "workspace" | "base_input" | "hair_edit" | "base_edit" | "correction" | "interp";
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
  /** STEP3実行中の工程表示（工程 n/N: ラベル）。null=一括分解以外の処理中 */
  const [seeThroughPhase, setSeeThroughPhase] = useState<{ index: number; total: number; label: string } | null>(null);
  const [seeThroughProfile, setSeeThroughProfile] = useState<SeeThroughProfile>("low-vram");
  /** 検出済みGPU一覧と選択（null=最大VRAMを自動選択） */
  const [seeThroughGpus, setSeeThroughGpus] = useState<Array<{ index: number; name: string; memoryMb: number }>>([]);
  const [seeThroughGpuIndex, setSeeThroughGpuIndex] = useState<number | null>(null);
  /** ユーザーが手動でプロファイルを選んだら、環境確認による推奨自動選択を止める */
  const seeThroughProfileTouched = useRef(false);
  const applyRecommendedSeeThroughProfile = (runtime: SeeThroughRuntimeStatus) => {
    if (seeThroughProfileTouched.current) return;
    if (runtime.recommendedProfile === "low-vram" || runtime.recommendedProfile === "standard") {
      setSeeThroughProfile(runtime.recommendedProfile);
    }
  };
  const [seeThroughSplitParts, setSeeThroughSplitParts] = useState(true);
  const [seeThroughOptions, setSeeThroughOptions] = useState<SeeThroughOptions>(DEFAULT_SEE_THROUGH_OPTIONS);
  const [workspacePartOffsetX, setWorkspacePartOffsetX] = useState(0);
  const [workspacePartOffsetY, setWorkspacePartOffsetY] = useState(0);
  const [workspacePartScale, setWorkspacePartScale] = useState(100);
  /** Step5の調整対象: "all"=全パーツ一括 / それ以外=パーツ個別（例: "mouth-a"） */
  const [workspaceAdjustTarget, setWorkspaceAdjustTarget] = useState("all");
  /** Step5プレビュー上のドラッグでパーツを動かすモード */
  const [workspacePartDragMode, setWorkspacePartDragMode] = useState(false);
  /** Step5で位置補正を適用済みのパーツ（サムネイルのバッジ表示用） */
  const [workspaceAdjustedParts, setWorkspaceAdjustedParts] = useState<Record<string, boolean>>({});
  /** 直近に「読み込んだ/適用した」値のベースライン。これと一致する間は変更なしとみなし自動適用しない
   * （対象パーツ切替時に値を復元しても誤って再適用しないため。値の一致だけでなくtargetも見る） */
  const workspaceAdjustBaseline = useRef({ target: "all", offsetX: 0, offsetY: 0, scalePercent: 100 });
  const workspaceAdjustDebounceTimer = useRef<number | null>(null);
  /** 完了フィードバック用トースト（3秒で自動消滅） */
  const [toast, setToast] = useState("");
  const toastTimer = useRef<number | null>(null);
  const showToast = (message: string) => {
    setToast(message);
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(""), 3000);
  };
  /** 追記式の作業ログ（最新50件） */
  const [workspaceLogs, setWorkspaceLogs] = useState<Array<{ time: string; level: "info" | "error"; text: string }>>([]);
  const pushWorkspaceLog = (level: "info" | "error", text: string) => {
    if (!text) return;
    setWorkspaceLogs(prev => {
      if (prev.length > 0 && prev[prev.length - 1].text === text) return prev;
      return [...prev.slice(-49), { time: new Date().toLocaleTimeString("ja-JP", { hour12: false }), level, text }];
    });
  };
  useEffect(() => { pushWorkspaceLog("info", status); /* eslint-disable-line react-hooks/exhaustive-deps */ }, [status]);
  useEffect(() => { pushWorkspaceLog("error", error); /* eslint-disable-line react-hooks/exhaustive-deps */ }, [error]);

  useEffect(() => {
    window.localStorage.setItem("pachipakugen-theme", themeMode);
  }, [themeMode]);

  // STEP2: Codex成果物の自動確認（依頼作成後〜揃うまで5秒間隔でポーリング）
  useEffect(() => {
    if (mode !== "workspace" || workspaceStep !== 2) return;
    if (!expressionWorkspace || !workspaceGeneratedStatus || workspaceGeneratedStatus.ready) return;
    const workPath = expressionWorkspace.workPath;
    const timer = window.setInterval(async () => {
      try {
        const generated = await invoke<WorkspaceGeneratedPartsStatus>("inspect_workspace_generated_parts", { workPath });
        setWorkspaceGeneratedStatus(generated);
        if (generated.ready) {
          showToast("Codex成果物が揃いました");
          setStatus("Codex成果物が揃いました。STEP3へ進めます");
        }
      } catch {
        // 依頼作成前のフォルダ状態では失敗してよい
      }
    }, 5000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, workspaceStep, expressionWorkspace, workspaceGeneratedStatus]);

  // STEP5: 「パーツ移動」ON中は矢印キーで±1px（Shiftで±10px）微調整できる
  useEffect(() => {
    if (mode !== "workspace" || workspaceStep !== 5 || !workspacePartDragMode) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const delta = event.shiftKey ? 10 : 1;
      if (event.key === "ArrowLeft") setWorkspacePartOffsetX(v => v - delta);
      else if (event.key === "ArrowRight") setWorkspacePartOffsetX(v => v + delta);
      else if (event.key === "ArrowUp") setWorkspacePartOffsetY(v => v - delta);
      else if (event.key === "ArrowDown") setWorkspacePartOffsetY(v => v + delta);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, workspaceStep, workspacePartDragMode]);

  // STEP5: オフセット/スケールが読み込み済みベースラインから変化したら、ドラッグ中でなければ
  // 少し待って自動適用する。「対象パーツ切替による値の読み込み」はベースライン比較で除外され、
  // 誤って別パーツへ上書き適用してしまう事故を防ぐ
  useEffect(() => {
    if (mode !== "workspace" || workspaceStep !== 5) return;
    const baseline = workspaceAdjustBaseline.current;
    const unchanged = baseline.target === workspaceAdjustTarget
      && baseline.offsetX === workspacePartOffsetX
      && baseline.offsetY === workspacePartOffsetY
      && baseline.scalePercent === workspacePartScale;
    if (unchanged || workspacePreviewDragging || !workspaceExtractResult) return;
    if (workspaceAdjustDebounceTimer.current !== null) window.clearTimeout(workspaceAdjustDebounceTimer.current);
    workspaceAdjustDebounceTimer.current = window.setTimeout(() => {
      workspaceAdjustDebounceTimer.current = null;
      void applyWorkspacePartAdjustment();
    }, 250);
    return () => {
      if (workspaceAdjustDebounceTimer.current !== null) {
        window.clearTimeout(workspaceAdjustDebounceTimer.current);
        workspaceAdjustDebounceTimer.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspacePartOffsetX, workspacePartOffsetY, workspacePartScale, workspaceAdjustTarget]);

  /** STEP5: 「対象」切替時、そのパーツの実際の現在値をフィールドへ読み込む（前の対象の値が残るのを防ぐ） */
  function loadWorkspacePartAdjustmentFields(target: string) {
    const adjustment = target === "all" ? undefined : workspaceExtractResult?.partAdjustments[target];
    const values = adjustment ?? DEFAULT_PART_ADJUSTMENT;
    workspaceAdjustBaseline.current = { target, ...values };
    setWorkspacePartOffsetX(values.offsetX);
    setWorkspacePartOffsetY(values.offsetY);
    setWorkspacePartScale(values.scalePercent);
  }

  // STEP1: 画像ファイルのドラッグ&ドロップ対応（1枚目=立ち絵、2枚目=参照画像）
  useEffect(() => {
    if (mode !== "workspace" || workspaceStep !== 1) return;
    let unlisten: (() => void) | null = null;
    let disposed = false;
    void getCurrentWebview().onDragDropEvent(event => {
      if (event.payload.type !== "drop") return;
      const paths = event.payload.paths.filter(path => /\.(png|jpe?g|webp)$/i.test(path));
      if (paths.length === 0) return;
      setWorkspaceFiles(prev => {
        const next = { ...prev };
        const [first, second] = paths;
        if (!next.source) {
          next.source = first;
          if (second) next.reference = second;
        } else if (!next.reference) {
          next.reference = first;
        } else {
          next.source = first;
        }
        void loadWorkspaceImagePreviews(next);
        return next;
      });
      setStatus("ドロップした画像を設定しました");
    }).then(fn => {
      if (disposed) fn();
      else unlisten = fn;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, workspaceStep]);

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
        // tqdm等の生ログ断片はステータス/ログ履歴に流さず短文へ整形。無意味な断片は捨てる
        const cleaned = sanitizeSeeThroughLogMessage(event.payload.message);
        if (cleaned) setStatus(cleaned);
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
    if (outputPath) try { await openPath(outputPath); } catch (_) {}
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
      const message = String(cause);
      if (kind === "resume" && message.includes("project.json が見つかりません")) {
        setError("選択したフォルダはPachiPakuGenの作業フォルダではありません（project.json がありません）。「はじめから」で作成した作業フォルダを選んでください");
      } else {
        setError(message);
      }
    } finally {
      setWorkspaceBusy(false);
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
      // プレビュー再構築は保存済みbase_partsからのみ行う（推論は走らない）。
      // base_parts未保存の作業ではここで失敗し、STEP4からのやり直しに誘導する
      let compositeReady = false;
      if (loaded.extractedParts) {
        compositeReady = await refreshWorkspaceCompositePreview(workspace)
          .then(() => true)
          .catch(() => false);
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
        if (loaded.extractedParts) {
          restoredStep = compositeReady
            ? (Math.max(restoredStep, 5) as WorkspaceStep)
            : (Math.min(Math.max(restoredStep, 4), 4) as WorkspaceStep);
        } else if (loaded.generatedParts.ready) {
          restoredStep = Math.max(restoredStep, 3) as WorkspaceStep;
        }
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

  function clearWorkspaceImage(key: "source" | "reference") {
    setWorkspaceFiles(prev => {
      const next = { ...prev };
      delete next[key];
      void loadWorkspaceImagePreviews(next);
      return next;
    });
  }

  function workspaceFileName(path: string | undefined): string {
    if (!path) return "";
    return path.split(/[\\/]/).pop() ?? path;
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
      setStatus("Codex依頼ファイルを作成しました。依頼フォルダの内容をCodexへ渡してください");
      showToast("依頼ファイルを作成しました");
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
      // 揃っていても自動でSTEP3へは進めない（進むのはユーザーの「次へ」だけ）
      if (status.ready) showToast("Codex成果物が揃いました");
      setStatus(status.ready ? "Codex成果物が揃いました。「次へ」でSee-Throughに進めます" : `Codex成果物に不足があります（残り${status.missingParts.length}ファイル）`);
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
      applyRecommendedSeeThroughProfile(runtime);
      await invoke<Array<{ index: number; name: string; memoryMb: number }>>("list_see_through_gpus")
        .then(setSeeThroughGpus)
        .catch(() => setSeeThroughGpus([]));
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
      applyRecommendedSeeThroughProfile(runtime);
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
    setSeeThroughPhase({ index: 1, total: 3, label: "立ち絵を分解しています" });
    try {
      const base = await invoke<SeeThroughRunResult>("run_see_through", {
        sourcePath: workspaceFiles.source,
        profile: seeThroughProfile,
        splitParts: seeThroughSplitParts,
        options: seeThroughOptions,
      });
      // 左右分解が素材依存で失敗した場合、バックエンドが左右分解なしで自動リトライ済み。
      // 以降の工程（表情素材の分解）も左右分解なしに揃え、ユーザーへ報告する
      let effectiveSplitParts = seeThroughSplitParts;
      if (base.splitPartsFallback) {
        effectiveSplitParts = false;
        setSeeThroughSplitParts(false);
        showToast("左右パーツ分解に失敗したため、分解なしで続行しています");
        setStatus("左右パーツ分解がこの素材では失敗したため、左右分解なしで処理を続行しました（目・耳は左右一体のレイヤーになります）");
      }
      if (base.oomRetryNote) {
        showToast(`GPUエラーのため設定を変更して続行しています（${base.oomRetryNote}）`);
        pushWorkspaceLog("info", `警告: GPUエラーのため自動リトライしました（${base.oomRetryNote}）`);
        // 高VRAM（非量子化）へ切り替わった場合はUIのプロファイル表示も揃える
        if (base.oomRetryNote.includes("高VRAM") || base.oomRetryNote.includes("standard")) {
          seeThroughProfileTouched.current = true;
          setSeeThroughProfile("standard");
        }
      }
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
      setSeeThroughPhase({ index: 2, total: 3, label: "表情素材（Codex成果物）を分解しています" });
      const extracted = await invoke<ExtractCodexGeneratedPartsResult>("extract_codex_generated_parts", {
        jobPath: expressionWorkspace.workPath,
        profile: seeThroughProfile,
        splitParts: effectiveSplitParts,
        options: seeThroughOptions,
      });
      // 左右分解フォールバック等の警告は作業ログへ残す（トーストでは流れてしまうため）
      for (const warning of extracted.warnings) {
        pushWorkspaceLog("info", `警告: ${warning}`);
      }
      if (extracted.warnings.some(warning => warning.includes("左右分解なしで処理しました"))) {
        setSeeThroughSplitParts(false);
        showToast("左右パーツ分解に失敗したため、分解なしで続行しました");
      }
      if (extracted.warnings.some(warning => warning.includes("GPUエラーのため自動リトライしました"))) {
        showToast("GPUエラーのため設定を変更して続行しました");
      }
      // 期待する全表情（目開閉＋口6種）のうち、抽出できず欠落したものを明示する
      const EXPECTED_EXPRESSION_PARTS = ["eyes-open", "eyes-closed", "mouth-closed", "mouth-a", "mouth-i", "mouth-u", "mouth-e", "mouth-o"];
      const missingExpressions = EXPECTED_EXPRESSION_PARTS.filter(part => !extracted.extractedParts.includes(part));
      if (missingExpressions.length > 0) {
        const label = missingExpressions.join("・");
        pushWorkspaceLog("error", `表情が不足: ${label} を抽出できませんでした`);
        setError(`一部の表情（${label}）を元画像から抽出できませんでした。これらはまばたきや口パクに使われます。\n\n対処: STEP2の「配置フォルダを開く」で該当画像を確認し、目や口がはっきり分かる素材に差し替えて（またはCodexで作り直して）STEP3を再実行してください。左右パーツ分解をOFFにすると改善する場合もあります。`);
      }
      setWorkspaceExtractResult(extracted);
      setWorkspaceCompositePreview(null);
      setWorkspaceRifeResult(null);
      setWorkspaceRifePreview(null);
      await setWorkspaceStepAndPersist(4);
      setSeeThroughPhase({ index: 3, total: 3, label: "プレビューを準備しています" });
      const allPreview = await invoke<MappingPreviewResult>("get_all_layers_preview");
      setMappingPreview(allPreview);
      await openUnifiedBaseEditorWithPreview(allPreview);
      setSeeThroughProgress({ stage: "complete", percent: 100, message: "See-Through一括分解が完了しました" });
      setStatus(`See-Through一括分解が完了しました: ${extracted.extractedParts.length}パーツ`);
      showToast("一括分解が完了しました");
    } catch (cause) {
      const message = String(cause);
      // 左右分割（耳・目などのL/R分離）のインデックスエラーはVRAM不足ではなく
      // 素材依存の検出失敗。回避方法を明示する
      if (seeThroughSplitParts && /(tag_lr_split|part_lr_split|lr_split|IndexError)/.test(message)) {
        setError(`左右パーツ分解の処理（耳などの左右分割）で失敗しました。このキャラクターでは「左右パーツ分解」をOFFにして再実行すると回避できます（VRAM不足ではありません）。\n\n詳細: ${message}`);
      } else {
        setError(message);
      }
    } finally {
      setWorkspaceBusy(false);
      setSeeThroughStartedAt(null);
      setSeeThroughPhase(null);
    }
  }

  async function applyWorkspacePartAdjustment(offsetOverride?: { x: number; y: number }) {
    if (!expressionWorkspace || !workspaceExtractResult) return;
    setWorkspaceBusy(true);
    try {
      const appliedOffsetX = offsetOverride?.x ?? workspacePartOffsetX;
      const appliedOffsetY = offsetOverride?.y ?? workspacePartOffsetY;
      const result = await invoke<AdjustCodexExtractedPartsResult>("adjust_codex_extracted_parts", {
        request: {
          jobPath: expressionWorkspace.workPath,
          offsetX: appliedOffsetX,
          offsetY: appliedOffsetY,
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
        partAdjustments: result.partAdjustments,
      });
      // 適用済みの値をベースラインとして記録（同じ値のままでは再適用が走らないように）
      workspaceAdjustBaseline.current = {
        target: workspaceAdjustTarget,
        offsetX: appliedOffsetX,
        offsetY: appliedOffsetY,
        scalePercent: workspacePartScale,
      };
      setWorkspaceRifeResult(null);
      setWorkspaceRifePreview(null);
      // 調整済みバッジ: バックエンドが実際に適用したパーツを記録
      // （一括適用はeyes-open対象外のため、静的リストではなく戻り値を使う）
      setWorkspaceAdjustedParts(prev => {
        const next = { ...prev };
        for (const part of result.adjustedParts) next[part] = true;
        return next;
      });
      await refreshWorkspaceCompositePreview().catch(() => null);
      await setWorkspaceStepAndPersist(5);
      setStatus("差分位置を調整しました");
      showToast("位置調整を適用しました");
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
      showToast("RIFE補完が完了しました");
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
    // 「次へ」が押せない理由の提示（無効ボタンの理由が分からない問題への対応）
    const nextStepBlockReason = (): string | null => {
      if (workspaceStep >= 7 || canAdvanceWorkspaceStep()) return null;
      if (workspaceBusy) return "処理が完了するまでお待ちください";
      switch (workspaceStep) {
        case 1: return "立ち絵を選択すると次へ進めます";
        case 2: {
          const missing = workspaceGeneratedStatus?.missingParts.length ?? 0;
          return missing > 0
            ? `Codex成果物が揃うと進めます（残り${missing}ファイル）`
            : "「依頼ファイルを作成」してCodexの成果物を配置してください";
        }
        case 3: return "「一括分解を開始」が完了すると次へ進めます";
        case 4: return "「素体調整を開く」で調整を保存すると次へ進めます";
        case 5: return "差分の抽出結果と合成プレビューが揃うと次へ進めます";
        case 6: return "「RIFE補完を開始」が完了すると自動でモーション調整へ進みます";
        default: return null;
      }
    };
    const nextStepLabel = workspaceStep < 7 ? steps[workspaceStep]?.[1] ?? "" : "";
    if (!workspace) return null;

    return (
      <div className="workspace-flow-screen">
        <nav className="workspace-flow-stepper" aria-label="表情セット作成ステップ">
          {steps.map(([step, label, note]) => (
            <button
              key={step}
              className={`workspace-flow-step${workspaceStep === step ? " active" : ""}${workspaceStep > step ? " done" : ""}`}
              disabled={!canOpenWorkspaceStep(step)}
              aria-current={workspaceStep === step ? "step" : undefined}
              title={!canOpenWorkspaceStep(step) && !workspaceBusy ? "前の工程を完了すると開けます" : workspaceStep > step ? "クリックでこの工程へ戻れます" : undefined}
              onClick={() => void setWorkspaceStepAndPersist(step)}
            >
              <b>{workspaceBusy && workspaceStep === step ? <span className="step-spinner" aria-label="処理中" /> : workspaceStep > step ? "✓" : step}</b>
              <span><strong>{label}</strong><small>{note}</small></span>
            </button>
          ))}
        </nav>

        <div
          className={`workspace-flow-layout${workspaceStep === 1 ? " step-one" : ""}${workspaceStep >= 2 && workspaceStep <= 4 ? " single-panel" : ""}`}
          style={workspaceStep === 7 ? { display: "none" } : undefined}
        >
          <section className="workspace-flow-panel">
            {workspaceStep === 1 && (
              <div className="workspace-step-one">
                <div className="workspace-panel-heading">
                  <span>STEP 1</span>
                  <h3>立ち絵と参照画像を選択</h3>
                  <p>立ち絵は必須です。参照画像は任意で、使うと目や口の中の色・質感が元絵に近づきます。</p>
                </div>
                <div className="workspace-image-picker-grid">
                  <div className="workspace-image-picker-cell">
                    <button className={`workspace-image-picker${workspaceFiles.source ? " ready" : ""}`} onClick={() => void pickWorkspaceImage("source")}>
                      <span>立ち絵（必須）</span>
                      {workspaceImagePreviews.source ? (
                        <img src={workspaceImagePreviews.source} alt="立ち絵プレビュー" />
                      ) : (
                        <strong className="workspace-picker-empty">
                          <b>クリックして立ち絵を選択</b>
                          <em>表情セットの元になる1枚です。<br />この画面へのドラッグ&ドロップでも設定できます。</em>
                        </strong>
                      )}
                      <small title={workspaceFiles.source}>{workspaceFiles.source ? workspaceFileName(workspaceFiles.source) : "PNG / JPG / WebP"}</small>
                    </button>
                    {workspaceFiles.source && (
                      <button className="workspace-image-clear" title="立ち絵の選択を解除" onClick={() => clearWorkspaceImage("source")}>✕ クリア</button>
                    )}
                  </div>
                  <div className="workspace-image-picker-cell">
                    <button className={`workspace-image-picker${workspaceFiles.reference ? " ready" : ""}`} onClick={() => void pickWorkspaceImage("reference")}>
                      <span>参照画像（任意）</span>
                      {workspaceImagePreviews.reference ? (
                        <img src={workspaceImagePreviews.reference} alt="参照画像プレビュー" />
                      ) : (
                        <strong className="workspace-picker-empty">
                          <b>クリックして参照画像を選択</b>
                          <em>同じキャラクターの別ポーズや口を開けた絵があると、<br />Codex生成の目・口内の色や質感が元絵に近づきます。<br />なければ空のままで進めます。</em>
                        </strong>
                      )}
                      <small title={workspaceFiles.reference}>{workspaceFiles.reference ? workspaceFileName(workspaceFiles.reference) : "設定しない場合はそのまま次へ"}</small>
                    </button>
                    {workspaceFiles.reference && (
                      <button className="workspace-image-clear" title="参照画像の選択を解除" onClick={() => clearWorkspaceImage("reference")}>✕ クリア</button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {workspaceStep === 2 && (() => {
              // ①依頼作成 → ②Codexで生成（このアプリの外の作業）→ ③配置確認 の現在地
              const codexPhase = workspaceGeneratedStatus?.ready ? 3 : workspaceGeneratedStatus ? 2 : 1;
              return (
              <>
                <div className="workspace-panel-heading">
                  <span>STEP 2</span>
                  <h3>Codex依頼を作成</h3>
                  <p>この工程の画像生成はアプリの外（Codex）で行います。</p>
                </div>
                <div className="workspace-codex-steps">
                  <section className={`workspace-codex-card${codexPhase === 1 ? " current" : ""}`}>
                    <div>
                      <span>1</span>
                      <strong>依頼ファイルを作成 {codexPhase === 1 && <em className="workspace-phase-badge">いまここ</em>}{codexPhase > 1 && <em className="workspace-phase-done">✓ 済み</em>}</strong>
                      <p>PachiPakuGenがCodex向けの指示書と元画像を <code>01_codex_request</code> に書き出します。</p>
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-primary" disabled={workspaceBusy || !workspaceFiles.source} onClick={() => void prepareWorkspaceCodexRequest()}>依頼ファイルを作成</button>
                    </div>
                  </section>
                  <section className={`workspace-codex-card${codexPhase === 2 ? " current" : ""}`}>
                    <div>
                      <span>2</span>
                      <strong>Codexで生成（アプリ外） {codexPhase === 2 && <em className="workspace-phase-badge">いまここ</em>}{codexPhase > 2 && <em className="workspace-phase-done">✓ 済み</em>}</strong>
                      <p><code>01_codex_request</code> を<b>フォルダごと</b>Codexへ渡してください（指示は同梱済み）。生成物は通常Codexが <code>02_generated_parts</code> へ自動配置します。</p>
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-secondary" onClick={() => openPath(workspace.codexRequestPath).catch(() => {})}>依頼フォルダを開く</button>
                      <button className="btn btn-secondary" onClick={() => openPath(workspace.generatedPartsPath).catch(() => {})}>配置フォルダを開く</button>
                    </div>
                  </section>
                  <section className={`workspace-codex-card${codexPhase === 3 ? " current" : ""}`}>
                    <div>
                      <span>3</span>
                      <strong>配置を確認 {codexPhase === 3 ? <em className="workspace-phase-done">✓ 揃いました</em> : codexPhase === 2 ? <em className="workspace-phase-badge">自動確認中</em> : null}</strong>
                      <p>
                        {workspaceGeneratedStatus?.ready
                          ? "すべて揃いました。「次へ」で進んでください。"
                          : codexPhase === 2
                            ? `5秒ごとに自動確認中（${workspaceGeneratedStatus?.presentParts.length ?? 0}/${workspaceGeneratedStatus?.expectedParts.length ?? 7}）`
                            : "依頼ファイル作成後に一覧が表示されます。"}
                      </p>
                      {workspaceGeneratedStatus && (
                        <div className="workspace-parts-checklist">
                          {workspaceGeneratedStatus.expectedParts.map(part => {
                            const present = workspaceGeneratedStatus.presentParts.includes(part);
                            const mismatch = workspaceGeneratedStatus.sizeMismatches.includes(part);
                            return (
                              <span key={part} className={mismatch ? "mismatch" : present ? "present" : "missing"} title={mismatch ? "サイズが立ち絵と一致していません。立ち絵と同じ縦横サイズで再生成してください" : present ? "配置済み" : "未配置"}>
                                <b>{mismatch ? "⚠" : present ? "✓" : "・"}</b>{part}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-primary" disabled={workspaceBusy} onClick={() => void inspectWorkspaceGeneratedParts()}>いますぐ確認</button>
                    </div>
                  </section>
                </div>
              </>
              );
            })()}

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
                      <button
                        className={seeThroughProfile === "low-vram" ? "active" : ""}
                        disabled={workspaceBusy}
                        title="目安: VRAM 8GB級でも動くようモデルを退避しながら実行します（低速）"
                        onClick={() => { seeThroughProfileTouched.current = true; setSeeThroughProfile("low-vram"); }}
                      >
                        省VRAM{seeThroughRuntime?.recommendedProfile === "low-vram" && <em className="workspace-recommend-badge">推奨</em>}
                      </button>
                      <button
                        className={seeThroughProfile === "standard" ? "active" : ""}
                        disabled={workspaceBusy}
                        title="目安: VRAM 16GB以上のGPU向け。退避なしで最速です"
                        onClick={() => { seeThroughProfileTouched.current = true; setSeeThroughProfile("standard"); }}
                      >
                        高VRAM{seeThroughRuntime?.recommendedProfile === "standard" && <em className="workspace-recommend-badge">推奨</em>}
                      </button>
                    </div>
                    <small className="workspace-gpu-info">目安: 省VRAM=8GB級でも可（低速） / 高VRAM=16GB以上（最速）</small>
                    {seeThroughGpus.length > 0 ? (
                      <label className="workspace-gpu-select">
                        <span>使用GPU</span>
                        <select
                          value={seeThroughGpuIndex ?? "auto"}
                          disabled={workspaceBusy}
                          onChange={(event) => {
                            const value = event.target.value === "auto" ? null : Number(event.target.value);
                            setSeeThroughGpuIndex(value);
                            void invoke("set_see_through_gpu", { gpuIndex: value })
                              .then(() => invoke<SeeThroughRuntimeStatus>("get_see_through_runtime_status"))
                              .then(runtime => { setSeeThroughRuntime(runtime); applyRecommendedSeeThroughProfile(runtime); })
                              .catch(cause => setError(String(cause)));
                          }}
                        >
                          <option value="auto">自動（最大VRAMのGPU）</option>
                          {seeThroughGpus.map(gpu => (
                            <option key={gpu.index} value={gpu.index}>
                              GPU{gpu.index}: {gpu.name}（{Math.round(gpu.memoryMb / 1024)}GB）
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : (
                      seeThroughRuntime?.gpuName && (
                        <small className="workspace-gpu-info">
                          GPU: {seeThroughRuntime.gpuName}
                          {seeThroughRuntime.gpuMemoryMb ? ` (${Math.round(seeThroughRuntime.gpuMemoryMb / 1024)}GB)` : ""}
                        </small>
                      )
                    )}
                  </div>
                  <label className="workspace-toggle-option workspace-option-card">
                    <input type="checkbox" checked={seeThroughSplitParts} disabled={workspaceBusy} onChange={(event) => setSeeThroughSplitParts(event.target.checked)} />
                    <span>左右パーツ分解</span>
                    <small>目や耳などを左右レイヤーに分けます。</small>
                  </label>
                  <details className="workspace-option-card workspace-option-card-wide workspace-advanced-options">
                    <summary>
                      <span>詳細設定（通常は変更不要）</span>
                    </summary>
                    <div className="workspace-option-header">
                      <span>See-Throughパラメータ</span>
                      <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => setSeeThroughOptions(DEFAULT_SEE_THROUGH_OPTIONS)}>標準値に戻す</button>
                    </div>
                    <div className="workspace-option-grid">
                      <label title="生成の乱数シード。同じ値なら同じ分解結果になります。分解結果が気に入らない時に値を変えて再実行してください"><span>Seed <i className="workspace-info-mark">?</i></span><input type="number" value={seeThroughOptions.seed} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, seed: Number(event.target.value) })} /></label>
                      <label title="レイヤー分解の処理解像度。大きいほど輪郭が精細になりますがVRAM消費と処理時間が増えます"><span>LayerDiff解像度 <i className="workspace-info-mark">?</i></span><input type="number" min={256} max={4096} step={64} value={seeThroughOptions.resolution} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, resolution: Number(event.target.value) })} /></label>
                      <label title="深度（前後関係）推定の処理解像度。-1で自動。レイヤー前後の判定が怪しい時に上げます"><span>Depth解像度 <i className="workspace-info-mark">?</i></span><input type="number" min={-1} max={4096} step={64} value={seeThroughOptions.resolutionDepth} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, resolutionDepth: Number(event.target.value) })} /></label>
                      <label title="レイヤー分解の推論ステップ数。多いほど品質が上がりますが遅くなります（既定30）"><span>LayerDiff step <i className="workspace-info-mark">?</i></span><input type="number" min={1} max={150} value={seeThroughOptions.inferenceSteps} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, inferenceSteps: Number(event.target.value) })} /></label>
                      <label title="深度推定のステップ数。-1で自動。省VRAMプロファイルでは固定のため変更できません"><span>Depth step <i className="workspace-info-mark">?</i></span><input type="number" min={-1} max={150} value={seeThroughOptions.inferenceStepsDepth} disabled={workspaceBusy || seeThroughProfile === "low-vram"} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, inferenceStepsDepth: Number(event.target.value) })} /></label>
                      <label title="モデルをブロック単位でCPUメモリへ退避してVRAMを節約します（少し低速）。「自動」はプロファイルの既定動作に任せます。VRAM不足エラーが出る時に有効化"><span>Group offload <i className="workspace-info-mark">?</i></span><select value={seeThroughOptions.groupOffload} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, groupOffload: event.target.value as SeeThroughOptionMode })}><option value="default">自動（推奨）</option><option value="on">有効</option><option value="off">無効</option></select></label>
                      <label title="モデル全体をCPUへ退避する最も強い省VRAM設定（大きく低速）。「自動」はプロファイルの既定動作に任せます。高VRAMプロファイルでは使いません"><span>CPU offload <i className="workspace-info-mark">?</i></span><select value={seeThroughOptions.cpuOffload} disabled={workspaceBusy || seeThroughProfile === "standard"} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, cpuOffload: event.target.value as SeeThroughOptionMode })}><option value="default">自動（推奨）</option><option value="on">有効</option><option value="off">無効</option></select></label>
                    </div>
                  </details>
                </div>
                <div className="workspace-start-seethrough">
                  <div><strong>分解処理を開始</strong><p>立ち絵、閉じ目、閉じ口、あいうえお口の素材をまとめて分解します。</p></div>
                  <button className="btn btn-primary" disabled={workspaceBusy || !seeThroughRuntime?.ready || !workspaceGeneratedStatus?.ready} onClick={() => void runWorkspaceSeeThroughBatch()}>{seeThroughRunning ? "分解処理中..." : "一括分解を開始"}</button>
                </div>
              </>
            )}

            {workspaceStep === 4 && (
              <>
                <div className="workspace-panel-heading"><span>STEP 4</span><h3>素体を調整</h3><p>分解済みレイヤーの前後関係を確認し、SpriTalkに渡す素体パーツを作成します。</p></div>
                {workspaceCompositePreview?.basePreview ? (
                  <div className="workspace-status-card">
                    <strong>✓ 素体は保存済みです</strong>
                    <span>直したい箇所があれば「再調整する」で調整画面を開き直せます。問題なければ「次へ」で差分位置調整に進んでください。</span>
                  </div>
                ) : (
                  <div className="workspace-status-card">
                    <strong>「素体調整を開く」で専用の調整画面が開きます</strong>
                    <ul className="workspace-feature-list">
                      <li>レイヤーの表示/非表示と前後関係（描画順）の入れ替え</li>
                      <li>腕・獣耳の分離出力（腕揺れ・耳ピコピコ用）</li>
                      <li>胸の切り出し（揺れ物理用の chest.png 分離）</li>
                      <li>ブラシによるレイヤーの継ぎ足し・削り（SAM3補助つき）</li>
                    </ul>
                    <span>調整画面で「保存して戻る」と自動でSTEP5（差分位置調整）へ進みます。</span>
                  </div>
                )}
                <div className="workspace-action-row">
                  <button className="btn btn-primary" disabled={workspaceBusy} onClick={() => void openWorkspaceBaseAdjustment()}>
                    {workspaceBusy ? "読み込み中..." : workspaceCompositePreview?.basePreview ? "再調整する" : "素体調整を開く"}
                  </button>
                </div>
              </>
            )}

            {workspaceStep === 5 && (
              <>
                <div className="workspace-panel-heading"><span>STEP 5</span><h3>差分位置を調整</h3><p>右のサムネイルか「対象」でパーツを選ぶと、そのパーツの現在の補正値が表示されます。プレビューを直接ドラッグ、矢印ボタン/矢印キー（Shiftで±10）、数値入力のいずれでも動かせ、少し待つと自動で保存されます。</p></div>
                <div className="workspace-action-row">
                  <div className="workspace-adjust-grid">
                    <label><span>対象</span>
                      <select
                        value={workspaceAdjustTarget}
                        onChange={e => {
                          setWorkspaceAdjustTarget(e.target.value);
                          // パーツ個別を選んだら即ドラッグで動かせるように自動ON（一括ではOFF）
                          setWorkspacePartDragMode(e.target.value !== "all");
                          // 前の対象の値が残らないよう、選んだパーツの実際の現在値を読み込む
                          loadWorkspacePartAdjustmentFields(e.target.value);
                        }}
                      >
                        <option value="all">全パーツ一括</option>
                        {WORKSPACE_ADJUST_PART_KEYS.map(part => <option key={part} value={part}>{part}{workspaceAdjustedParts[part] ? " ✓" : ""}</option>)}
                      </select>
                    </label>
                    <label><span>X offset</span>
                      <span className="workspace-nudge-row">
                        <button type="button" title="-10" onClick={() => setWorkspacePartOffsetX(v => v - 10)}>≪</button>
                        <button type="button" title="-1" onClick={() => setWorkspacePartOffsetX(v => v - 1)}>‹</button>
                        <input type="number" value={workspacePartOffsetX} onChange={e => setWorkspacePartOffsetX(Number(e.target.value))} />
                        <button type="button" title="+1" onClick={() => setWorkspacePartOffsetX(v => v + 1)}>›</button>
                        <button type="button" title="+10" onClick={() => setWorkspacePartOffsetX(v => v + 10)}>≫</button>
                      </span>
                    </label>
                    <label><span>Y offset</span>
                      <span className="workspace-nudge-row">
                        <button type="button" title="-10" onClick={() => setWorkspacePartOffsetY(v => v - 10)}>≪</button>
                        <button type="button" title="-1" onClick={() => setWorkspacePartOffsetY(v => v - 1)}>‹</button>
                        <input type="number" value={workspacePartOffsetY} onChange={e => setWorkspacePartOffsetY(Number(e.target.value))} />
                        <button type="button" title="+1" onClick={() => setWorkspacePartOffsetY(v => v + 1)}>›</button>
                        <button type="button" title="+10" onClick={() => setWorkspacePartOffsetY(v => v + 10)}>≫</button>
                      </span>
                    </label>
                    <label><span>Scale {workspacePartScale}%</span>
                      <span className="workspace-scale-row">
                        <input type="range" min={50} max={150} value={workspacePartScale} onChange={e => setWorkspacePartScale(Number(e.target.value))} />
                        <input type="number" min={50} max={150} value={workspacePartScale} onChange={e => setWorkspacePartScale(Number(e.target.value))} />
                      </span>
                    </label>
                  </div>
                  <button
                    className={`btn ${workspacePartDragMode ? "btn-primary" : "btn-secondary"}`}
                    disabled={workspaceBusy || !workspaceExtractResult || workspaceAdjustTarget === "all"}
                    title="ON中はプレビューのドラッグで選択パーツを移動（画面パンは無効化）"
                    onClick={() => setWorkspacePartDragMode(prev => !prev)}
                  >
                    パーツ移動 {workspacePartDragMode ? "ON" : "OFF"}
                  </button>
                  <button
                    className="btn btn-secondary"
                    disabled={workspaceBusy}
                    title="X/Y/Scaleを 0 / 0 / 100% に戻します（少し待つと自動で適用されます）"
                    onClick={() => { setWorkspacePartOffsetX(0); setWorkspacePartOffsetY(0); setWorkspacePartScale(100); }}
                  >0に戻す</button>
                  <button className="btn btn-secondary" disabled={workspaceBusy || !workspaceExtractResult} onClick={() => void updateWorkspaceCompositePreview()}>合成プレビューを更新</button>
                  <button className="btn btn-primary" disabled={workspaceBusy || !workspaceExtractResult} title="待たずにすぐ適用します（通常は自動で適用されるため押さなくてもOK）" onClick={() => void applyWorkspacePartAdjustment()}>今すぐ適用</button>
                </div>
                <div className="motion-lab-note">
                  自動整列が効くので通常は微調整のみでOKです。変更（ドラッグ・矢印・数値入力）は自動で保存されます。補正値はパーツごとに元画像基準の絶対値で保存され（adjustment.json v2）、対象を切り替えるとそのパーツの現在値が表示されます。
                  base は目・口を乗せない素体（のっぺらぼう）で、eyes-open は素体の目そのもの（他フレームと共通の平常時の目）なので調整対象外です。ズレうるのは閉じ目・口の差分だけです。「パーツ移動」ON中は矢印キーでも±1px（Shiftで±10px）動かせます。
                </div>
              </>
            )}

            {workspaceStep === 6 && (
              <>
                <div className="workspace-panel-heading"><span>STEP 6</span><h3>RIFE補完してSpriTalk用に出力</h3><p>RIFEフレーム、素体、抽出差分を <code>04_spritalk_parts</code> にまとめます。完了すると自動でモーション調整（STEP7）へ進みます。</p></div>
                <div className="workspace-rife-panel">
                  <div className="workspace-frame-slider">
                    <label><span>補間枚数</span><strong>{frameCount}枚</strong></label>
                    <input type="range" min={RIFE_FRAME_MIN} max={RIFE_FRAME_MAX} value={frameCount} disabled={workspaceBusy} onChange={e => setFrameCount(Number(e.target.value))} />
                    <small>多いほど口パク・まばたきが滑らかになりますが、ファイル数と生成時間が増えます。SpriTalkでは {RIFE_FRAME_RECOMMENDED} 枚が扱いやすい目安です。</small>
                  </div>
                  <button className="btn btn-primary" disabled={workspaceBusy || !workspaceExtractResult || !workspaceCompositePreview?.basePreview} onClick={() => void generateWorkspaceRifeOutputs()}>{workspaceBusy ? "RIFE補完中..." : "RIFE補完を開始"}</button>
                  {!workspaceBusy && workspaceRifeResult && (
                    <div className="workspace-status-card">
                      <strong>✓ RIFE補完は完了しています</strong>
                      <span>出力先: <code>{workspaceRifeResult.outputPath}</code></span>
                      <span>{workspaceRifeResult.directories.length}ディレクトリ / 各{workspaceRifeResult.frameCount}フレーム</span>
                      <span>「次へ: モーション調整」で揺れ・口パクの仕上げに進めます。枚数を変えて再生成することもできます。</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </section>

          {workspaceStep >= 5 && workspaceStep <= 6 && (
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
                      // Step5ではサムネイル選択＝調整対象の切替＋即ドラッグ可能に。
                      // 前の対象の値が残らないよう、選んだパーツの実際の現在値を読み込む
                      if (workspaceStep === 5 && WORKSPACE_ADJUST_PART_KEYS.includes(key)) {
                        setWorkspaceAdjustTarget(key);
                        setWorkspacePartDragMode(true);
                        loadWorkspacePartAdjustmentFields(key);
                      }
                    }}>
                      <span>
                        {workspacePreviewItemLabel(item)}
                        {workspaceStep === 5 && workspaceAdjustedParts[workspacePreviewItemKey(item)] && <em className="workspace-adjusted-badge">調整済</em>}
                      </span>
                      <img src={item.preview} alt={`${workspacePreviewItemLabel(item)} preview`} />
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
          <button className="btn btn-secondary" onClick={goPreviousWorkspaceStep}>
            {workspaceStep === 1 ? "制作ホームへ戻る" : `戻る: ${steps[workspaceStep - 2]?.[1] ?? ""}`}
          </button>
          {/* 戻る/次への間の空きスペースに実行中の進捗を表示（パネル側のスクロールを増やさない） */}
          <div className="workspace-nav-progress" aria-live="polite">
            {seeThroughRunning && (
              <>
                <span>
                  {seeThroughPhase
                    ? <><b>工程 {seeThroughPhase.index}/{seeThroughPhase.total}</b> {seeThroughPhase.label}</>
                    : <>{sanitizeSeeThroughLogMessage(displaySeeThroughMessage(seeThroughProgress)) || "処理を継続しています"}</>}
                  <small>・経過 {formatElapsed(seeThroughElapsedSeconds)}</small>
                </span>
                <div className="workspace-progress-bar indeterminate" aria-label="See-Through進捗"><div /></div>
              </>
            )}
            {!seeThroughRunning && workspaceBusy && progress.total > 0 && (
              <>
                <span><b>RIFE補完中</b> {progress.pair_name} {progress.current}/{progress.total}</span>
                <div className="workspace-progress-bar" aria-label="RIFE進捗"><div style={{ width: `${Math.max(3, Math.min(100, (progress.current / Math.max(1, progress.total)) * 100))}%` }} /></div>
              </>
            )}
          </div>
          <div className="workspace-next-area">
            {nextStepBlockReason() && <small className="workspace-next-hint">{nextStepBlockReason()}</small>}
            {workspaceStep >= 7 ? (
              <button className="btn btn-primary" disabled={workspaceBusy} onClick={() => openPath(workspaceRifeResult?.outputPath || workspace.spritalkPartsPath).catch(() => {})}>出力フォルダを開く</button>
            ) : (
              <button
                className="btn btn-primary"
                disabled={!canAdvanceWorkspaceStep()}
                title={nextStepBlockReason() ?? undefined}
                onClick={goNextWorkspaceStep}
              >次へ: {nextStepLabel}</button>
            )}
          </div>
        </div>
        <div className="workspace-log-console" aria-live="polite">
          <details className="workspace-log-history">
            <summary>
              <span className="workspace-log-title">LOG</span>
              <span className="workspace-log-latest">
                <span>step {workspaceStep}/7 {steps[workspaceStep - 1]?.[1]}</span>
                {workspaceBusy && <span className="running">処理中...</span>}
                {seeThroughProgress && workspaceBusy && <span>{sanitizeSeeThroughLogMessage(displaySeeThroughMessage(seeThroughProgress)) || "処理を継続しています"}</span>}
                {progress.total > 0 && workspaceBusy && <span>RIFE {progress.pair_name} {progress.current}/{progress.total}</span>}
                <span className={error ? "log-error" : ""}>{error || status}</span>
              </span>
            </summary>
            <div className="workspace-log-lines">
              {workspaceLogs.length === 0 && <div><span>-</span>まだログはありません</div>}
              {[...workspaceLogs].reverse().map((entry, index) => (
                <div key={`${entry.time}-${index}`} className={entry.level === "error" ? "log-error" : ""}>
                  <span>{entry.time}</span>{entry.text}
                </div>
              ))}
            </div>
          </details>
        </div>
      </div>
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
              <p>空のフォルダを作業フォルダとして選び、7つのSTEPを順に進めてSpriTalk用の素材一式を出力します。</p>
            </div>
            <span className="primary-workflow-cta">作業フォルダを選択</span>
          </button>
          <button className="primary-workflow-card secondary" onClick={() => void startExpressionWorkspace("resume")}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">RESUME</span>
              <strong>つづきから</strong>
              <p>「はじめから」で使った作業フォルダ（<code>project.json</code> のあるフォルダ）を選ぶと、前回の工程から再開します。</p>
            </div>
            <span className="primary-workflow-cta">作業フォルダを開く</span>
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
              <div className="layer-bulk-section">
                <small>表示</small>
                <div className="layer-bulk-body">
                  <button className="btn-layer-bulk" onClick={() => void setAllLayerVisibility(true)}>全ON</button>
                  <button className="btn-layer-bulk" onClick={() => void setAllLayerVisibility(false)}>全OFF</button>
                </div>
              </div>
              <div className="layer-bulk-section">
                <small>透明度</small>
                <div className="layer-bulk-body">
                  <button className="btn-layer-bulk" onClick={() => void setAllBodyOpacities(1)}>全100%</button>
                  <button className="btn-layer-bulk" onClick={() => void setAllBodyOpacities(0.5)}>全50%</button>
                  <button className="btn-layer-bulk" onClick={() => void setAllBodyOpacities(0)}>全0%</button>
                </div>
              </div>
              {!!expressionWorkspace && (
                <div className="layer-bulk-section">
                  <small>並び順</small>
                  <div className="layer-bulk-body">
                    <button className="btn-layer-bulk" title="SpriTalk向けの標準的な前後関係（後ろ髪→体→目口→前髪）に並べ直します" onClick={() => void applyRecommendedLayerOrder()}>推奨順に並べ直す</button>
                  </div>
                </div>
              )}
              <div className="layer-bulk-section">
                <small>揺れ用の分離</small>
                <div className="layer-bulk-body">
                  {armSplitLayers.length > 0 && (
                    <label className="layer-arm-split-toggle" title="腕レイヤー（handwear-l/-r）を arm_l.png / arm_r.png として分離出力します。腕揺れ用">
                      <input type="checkbox" checked={armSplitActive} onChange={(e) => toggleArmSplit(e.target.checked)} />
                      <span>腕</span>
                    </label>
                  )}
                  {earSplitLayers.length > 0 && (
                    <label className="layer-arm-split-toggle" title="獣耳レイヤー（ears-l/-r または headwear）を sway_ear*.png として分離出力します。獣耳ピコピコ用">
                      <input type="checkbox" checked={earSplitActive} onChange={(e) => toggleEarSplit(e.target.checked)} />
                      <span>獣耳</span>
                    </label>
                  )}
                  <button
                    className={`btn-layer-bulk${chestMaskDataUrl ? " active" : ""}`}
                    title="See-Throughにはchestレイヤーが無いため、bodyから胸部を塗って手動で切り出します（胸揺れ用）"
                    onClick={initChestCut}
                  >
                    胸を切出{chestMaskDataUrl ? "済" : ""}
                  </button>
                  {chestMaskDataUrl && (
                    <button className="btn-layer-bulk" onClick={() => setChestMaskDataUrl(null)}>取消</button>
                  )}
                </div>
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
              <div className="action-panel-hint">
                {mode !== "correction" && expressionWorkspace
                  ? "レイヤー順、ON/OFF、切り出しを確認して保存します。「保存して戻る」とSTEP5（差分位置調整）へ進みます。"
                  : "レイヤー順、ON/OFF、切り出し、透明度を確認して保存します。"}
              </div>
            </div>
            <div className="step-nav-actions base-output-actions">
              <button
                className="btn btn-secondary"
                title={expressionWorkspace ? "保存せずにSTEP4へ戻ります" : undefined}
                onClick={mode !== "correction" && expressionWorkspace ? returnFromBaseFlow : returnToModeSelect}
              >{mode !== "correction" && expressionWorkspace ? "STEP4へ戻る（保存しない）" : "戻る"}</button>
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
          {status}
        </div>
      )}

      {error && (
        <div className="workspace-error-banner" role="alert">
          <strong>エラー</strong>
          <span>{error}</span>
          <button type="button" aria-label="エラーを閉じる" onClick={() => setError("")}>✕</button>
        </div>
      )}
      {toast && <div className="workspace-toast" role="status">{toast}</div>}
    </div>
  );
}

export default App;
