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
