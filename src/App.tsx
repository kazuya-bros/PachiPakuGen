import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import "./App.css";

// --- Types ---
interface AdjustableLayer { name: string; thumbnail: string; default_target: string; }
interface SlotLoadResult { detected_layers: string[]; adjustable_layers: AdjustableLayer[]; canvas_width: number; canvas_height: number; source_type: string; }
interface LayerBounds { x: number; y: number; width: number; height: number; }
interface LayerInfo { name: string; thumbnail: string; bounds: LayerBounds; }
interface CategoryPreview { target: string; label: string; preview: string; layer_names: string[]; layers: LayerInfo[]; }
interface MappingPreviewResult { categories: CategoryPreview[]; composite_preview: string; }
interface RenderCategoryResult { preview: string; }
interface ExportCorrectedLayerResult { output_path: string; }
interface ImportCorrectionLayerResult { layer_name: string; }
interface OriginalImageResult { original_preview: string; mouth_preview: string | null; }
interface MouthMaskPreviewResult { mouth_preview: string; }
interface CreateBaseResult { output_path: string; composite_preview: string; base_eye_slot: string; base_mouth_slot: string; file_count: number; }
interface CreateDiffResult { output_path: string; pair_name: string; frame_count: number; preview: string; previews: string[]; }
interface ProgressPayload { current: number; total: number; pair_name: string; }
interface LayerPatch { id: string; name: string; sourceLayer: string; maskPng: string; cutSource: boolean; thumbnail?: string; }
type InterpPair = { name: string; label: string; closed: { key: string; label: string }; open: { key: string; label: string }; required: boolean };
type MouthMaskSetting = { dilate: number; blur: number };
type PreviewPan = { x: number; y: number };
type DiffPreview = { pairName: string; label: string; frames: string[] };
type InterpStep = 1 | 2 | 3 | 4;
type BaseStep = 1 | 2 | 3 | 4;

// Each RIFE pair: closed PSD ↔ open PSD (open = base)
const EYE_PAIRS = [
  { name: "eye", label: "まばたき（目）",
    closed: { key: "eye_closed", label: "閉じる" },
    open: { key: "eye_open", label: "開く" },
    required: true },
];

const MOUTH_PAIRS_SINGLE = [
  { name: "mouth", label: "口パク",
    closed: { key: "mouth_closed", label: "閉じる" },
    open: { key: "mouth_open", label: "開く" },
    required: true },
];

const MOUTH_PAIRS_VOWELS = [
  { name: "mouth_a", label: "口パク（あ）",
    closed: { key: "mouth_a_closed", label: "閉じる" },
    open: { key: "mouth_a_open", label: "開く" },
    required: false },
  { name: "mouth_i", label: "口パク（い）",
    closed: { key: "mouth_i_closed", label: "閉じる" },
    open: { key: "mouth_i_open", label: "開く" },
    required: false },
  { name: "mouth_u", label: "口パク（う）",
    closed: { key: "mouth_u_closed", label: "閉じる" },
    open: { key: "mouth_u_open", label: "開く" },
    required: false },
  { name: "mouth_e", label: "口パク（え）",
    closed: { key: "mouth_e_closed", label: "閉じる" },
    open: { key: "mouth_e_open", label: "開く" },
    required: false },
  { name: "mouth_o", label: "口パク（お）",
    closed: { key: "mouth_o_closed", label: "閉じる" },
    open: { key: "mouth_o_open", label: "開く" },
    required: false },
];

type Mode = "select" | "base_input" | "hair_edit" | "base_edit" | "correction" | "interp";

function App() {
  const [mode, setMode] = useState<Mode>("select");

  // Shared
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("モードを選択してください");

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
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number; size: number; visible: boolean }>({ x: 0, y: 0, size: 0, visible: false });

  // === フレーム補間モード (interp) ===
  const [diffTarget, setDiffTarget] = useState<"eye" | "mouth">("eye");
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
  const [mouthMode, setMouthMode] = useState<"single" | "vowels">("single");
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
  // Drag reorder (generic — used for body, hair, hair_back)
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
    return () => {
      unlisten.then(fn => fn());
      if (opacityRenderTimer.current !== null) {
        window.clearTimeout(opacityRenderTimer.current);
      }
      for (const timer of Object.values(mouthMaskUpdateTimers.current)) {
        window.clearTimeout(timer);
      }
    };
  }, []);

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
    const newEnabled = { ...enabledLayers, [name]: checked };
    setEnabledLayers(newEnabled);
    await renderBody(layerOrder, newEnabled);
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
      // Init hair layers
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
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
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

  function startCorrectionMode() {
    setMode("correction");
    setLoadResult(null);
    setMappingPreview(null);
    setBodyPreview("");
    setLayerOrder([]);
    setEnabledLayers({});
    setLayerPatches([]);
    setLayerOpacities({});
    setSelectedBodyLayer("");
    setPatchDraftSource("");
    setCorrectionOutputPath("");
    resetZoom();
    setStatus("See-Through補正用のPSDを読み込んでください");
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

  function proceedToInterp(target: "eye" | "mouth", selectedMouthMode: "single" | "vowels" = "single") {
    setDiffTarget(target);
    setInterpPaths({});
    setInterpOriginals({});
    setMouthMaskSettings({});
    setMouthMaskPreviews({});
    setMouthMaskPreviewing({});
    setMouthPreviewZooms({});
    setMouthPreviewPans({});
    setCompletedDiffs([]);
    setDiffPreviews([]);
    setDiffPreviewTick(0);
    setDiffPreviewZoom(1);
    setOutputPath("");
    setProgress({ current: 0, total: 0, pair_name: "" });
    setActivePreviewKey("");
    setInterpStep(1);
    if (target === "mouth") setMouthMode(selectedMouthMode);
    setMode("interp");
    setStatus(target === "mouth" ? "差分PSD+元画像を設定してください" : "差分PSDを設定してください");
  }

  // === Step 3A: base_edit — create base ===
  async function handleCreateBase() {
    setError("");
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

        // Load original image for SAM3 mouth extraction (use "open" original — open mouth is easier to detect)
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

  function returnToModeSelect() {
    if (mode === "interp" && interpStep !== 4 && hasInterpWork) {
      const ok = window.confirm("現在のフレーム補間作業を終了してモード選択へ戻ります。入力、口マスク、生成プレビューはこの画面から離れると失われます。戻りますか？");
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

  const bodyCategory = mappingPreview?.categories.find(c => c.target === (mode === "correction" ? "free" : "body"));
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
    ctx.save();
    ctx.globalCompositeOperation = patchTool === "erase" ? "destination-out" : "source-over";
    ctx.fillStyle = "rgba(233, 69, 96, 0.55)";
    ctx.beginPath();
    ctx.arc(x, y, patchBrushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function onPatchMaskPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!patchDraftSource) return;
    e.preventDefault();
    e.stopPropagation();
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
      setError("パッチにする範囲を塗ってください");
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

  return (
    <div className="app">
      <div className="app-header">
        <h1>PachiPakuGen</h1>
        <span className="version">v0.3.0</span>
      </div>

      {(mode === "base_input" || mode === "hair_edit" || mode === "base_edit") && (
        <div className="interp-header base-flow-header">
          <h2 className="interp-header-title">素体出力</h2>
          <div className="interp-header-note">See-Throughで作成したPSDから body / hair / hair_back を出力します</div>
        </div>
      )}

      {mode === "correction" && (
        <div className="interp-header base-flow-header">
          <h2 className="interp-header-title">See-Through補正</h2>
          <div className="interp-header-note">See-Throughの分類結果を手動で切り出し、指定ファイル名でPNG保存します</div>
        </div>
      )}

      <div className={`main-content${mode === "base_input" || mode === "hair_edit" || mode === "base_edit" || mode === "correction" ? " base-main-content" : ""}`}>
        {/* ===== Mode Select ===== */}
        {mode === "select" && (
          <div className="mode-select-screen">
            <div className="output-select-buttons">
              <button className="btn-output" onClick={() => { setBaseStep(1); setMode("base_input"); setStatus("PSDを読み込んでください"); }}>
                <span className="btn-output-title">素体出力</span>
                <span className="btn-output-desc">See-Through PSD</span>
                <span className="btn-output-desc">→ body / hair / hair_back</span>
              </button>
              <button className="btn-output" onClick={startCorrectionMode}>
                <span className="btn-output-title">See-Through補正</span>
                <span className="btn-output-desc">分類済みPSDを手動調整</span>
                <span className="btn-output-desc">→ 指定PNG保存</span>
              </button>
              <button className="btn-output" onClick={() => proceedToInterp("eye")}>
                <span className="btn-output-title">まばたき</span>
                <span className="btn-output-desc">表情差分PSD × 2</span>
                <span className="btn-output-desc">→ RIFE中間フレーム</span>
              </button>
              <button className="btn-output" onClick={() => proceedToInterp("mouth", "single")}>
                <span className="btn-output-title">口パク mouthのみ</span>
                <span className="btn-output-desc">表情差分PSD × 2</span>
                <span className="btn-output-desc">→ mouth単品</span>
              </button>
              <button className="btn-output" onClick={() => proceedToInterp("mouth", "vowels")}>
                <span className="btn-output-title">口パク あ〜お</span>
                <span className="btn-output-desc">母音別PSD</span>
                <span className="btn-output-desc">→ mouth_a〜o</span>
              </button>
            </div>
          </div>
        )}

        {(mode === "base_input" || mode === "hair_edit" || mode === "base_edit") && (
          <div className={`base-flow-controls${mode === "base_input" ? " input" : " editing"}`}>
            <div className="interp-workflow-rail">
              <button className={`workflow-step ${baseStep === 1 ? "active" : ""}${loadResult ? " done" : ""}`}
                onClick={() => goBaseStep(1)}>1 入力</button>
              <button className={`workflow-step ${baseStep === 2 ? "active" : ""}${hairPreview || hairBackPreview ? " done" : ""}`}
                disabled={!baseStepEnabled(2)} onClick={() => goBaseStep(2)}>2 Hair</button>
              <button className={`workflow-step ${baseStep === 3 ? "active" : ""}${bodyPreview ? " done" : ""}`}
                disabled={!baseStepEnabled(3)} onClick={() => goBaseStep(3)}>3 Body</button>
              <button className={`workflow-step ${baseStep === 4 ? "active" : ""}${baseResult ? " done" : ""}`}
                disabled={!baseStepEnabled(4)} onClick={() => goBaseStep(4)}>4 確認</button>
            </div>
          </div>
        )}

        {mode === "correction" && !mappingPreview && (
          <div className="mode-select-screen base-input-screen">
            <div className="input-card-center">
              <h2 className="input-card-title">See-Through補正 — ファイル読み込み</h2>
              <div className="file-input-row-large">
                <span className="file-input-label-large">PSD:</span>
                <button className="btn btn-primary" onClick={loadCorrectionPsd} disabled={loading}>選択</button>
                <span className="slot-path-inline-large">{loadResult ? `${loadResult.detected_layers.length}レイヤー ✓` : "未選択"}</span>
              </div>
              <p className="input-hint">See-Throughで分解したPSDを読み込み、任意のレイヤーを手動で補正します。</p>
              {loading && (
                <div className="progress-bar indeterminate" style={{ marginTop: 12 }}><div className="fill" /></div>
              )}
              <div className="step-nav-actions base-input-actions">
                <button className="btn btn-secondary" onClick={() => { setMode("select"); setStatus("モードを選択してください"); }}>前へ</button>
                <button className="btn btn-primary" disabled={!mappingPreview || loading} onClick={() => {}}>
                  次へ
                </button>
              </div>
            </div>
          </div>
        )}

        {mode === "base_input" && (
          <div className="mode-select-screen base-input-screen">
            <div className="input-card-center">
              <h2 className="input-card-title">素体出力 — ファイル読み込み</h2>
              <div className="file-input-row-large">
                <span className="file-input-label-large">PSD:</span>
                <button className="btn btn-primary" onClick={loadPsd} disabled={loading}>選択</button>
                <span className="slot-path-inline-large">{loadResult ? `${loadResult.detected_layers.length}レイヤー ✓` : "未選択"}</span>
              </div>
              <p className="input-hint">See-Throughで分解したPSDファイル</p>
              {loading && (
                <div className="progress-bar indeterminate" style={{ marginTop: 12 }}><div className="fill" /></div>
              )}
              <div className="step-nav-actions base-input-actions">
                <button className="btn btn-secondary" onClick={returnFromBaseFlow}>前へ</button>
                <button className="btn btn-primary"
                  disabled={!loadResult || loading}
                  onClick={proceedToHairEdit}>
                  次へ
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ===== Step 1/2: Hair Edit ===== */}
        {mode === "hair_edit" && (
          <div className="panel-right base-edit-panel">
            <div className="preview-and-layers">
              <div className="preview-viewport" ref={previewRef}
                onWheel={handleWheel} onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
                {(hairPreview || hairBackPreview) ? (<>
                  {hairBackPreview && <img src={hairBackPreview} alt="Hair Back" className="preview-img"
                    style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, position: "absolute" }}
                    draggable={false} />}
                  {hairPreview && <img src={hairPreview} alt="Hair" className="preview-img"
                    style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: isPanning ? "grabbing" : "grab" }}
                    draggable={false} />}
                </>) : (
                  <span className="placeholder">Hairプレビュー</span>
                )}
              </div>

              <div className="layer-sidebar" onPointerMove={onDragPointerMove} onPointerUp={onDragPointerUp}>
                {/* Hair (前髪) */}
                {(() => {
                  const hairCat = mappingPreview?.categories.find(c => c.target === "hair");
                  return hairCat && hairLayerOrder.length > 0 ? (
                    <div className="layer-sidebar-section">
                      <div className="layer-sidebar-header">Hair (前髪)</div>
                      <div className="layer-sidebar-hint">上が手前</div>
                      <div className="layer-sidebar-list">
                        {hairLayerOrder.map((name, idx) => {
                          const layer = hairCat.layers.find(l => l.name === name);
                          if (!layer) return null;
                          return (
                            <div key={layer.name} className={`layer-sidebar-item${dragTarget === "hair" && draggedIdx === idx ? " dragging" : ""}`}>
                              <span className="drag-handle" onPointerDown={(e) => onDragPointerDown(e, idx, "hair")}>☰</span>
                              <input type="checkbox" checked={hairEnabledLayers[layer.name] !== false}
                                onChange={(e) => {
                                  const en = { ...hairEnabledLayers, [layer.name]: e.target.checked };
                                  setHairEnabledLayers(en);
                                  renderHair(hairLayerOrder, en);
                                }} />
                              <img src={layer.thumbnail} alt={layer.name} className="layer-sidebar-thumb" />
                              <span className="layer-sidebar-name">{layer.name}</span>
                            </div>
                          );
                        })}
                      </div>
                      <div className="layer-sidebar-hint">下が奥</div>
                    </div>
                  ) : null;
                })()}

                {/* Hair Back (後ろ髪) */}
                {(() => {
                  const hairBackCat = mappingPreview?.categories.find(c => c.target === "hair_back");
                  return hairBackCat && hairBackLayerOrder.length > 0 ? (
                    <div className="layer-sidebar-section">
                      <div className="layer-sidebar-header">Hair Back (後ろ髪)</div>
                      <div className="layer-sidebar-hint">上が手前</div>
                      <div className="layer-sidebar-list">
                        {hairBackLayerOrder.map((name, idx) => {
                          const layer = hairBackCat.layers.find(l => l.name === name);
                          if (!layer) return null;
                          return (
                            <div key={layer.name} className={`layer-sidebar-item${dragTarget === "hair_back" && draggedIdx === idx ? " dragging" : ""}`}>
                              <span className="drag-handle" onPointerDown={(e) => onDragPointerDown(e, idx, "hair_back")}>☰</span>
                              <input type="checkbox" checked={hairBackEnabledLayers[layer.name] !== false}
                                onChange={(e) => {
                                  const en = { ...hairBackEnabledLayers, [layer.name]: e.target.checked };
                                  setHairBackEnabledLayers(en);
                                  renderHairBack(hairBackLayerOrder, en);
                                }} />
                              <img src={layer.thumbnail} alt={layer.name} className="layer-sidebar-thumb" />
                              <span className="layer-sidebar-name">{layer.name}</span>
                            </div>
                          );
                        })}
                      </div>
                      <div className="layer-sidebar-hint">下が奥</div>
                    </div>
                  ) : null;
                })()}
              </div>
            </div>
            <div className="base-flow-controls editing bottom-actions">
              <div className="interp-action-panel base-flow-action">
                <div>
                  <div className="action-panel-title">Hairレイヤー確認</div>
                  <div className="action-panel-hint">hair / hair_back の並び替えとON/OFFを確認してください。</div>
                </div>
                <div className="step-nav-actions">
                  <button className="btn btn-secondary" onClick={() => goBaseStep(1)}>前へ</button>
                  <button className="btn btn-primary" onClick={proceedToBodyEdit}>次へ</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===== フレーム補間: diff PSD input ===== */}
        {mode === "interp" && (
          <div className="interp-screen">
            {/* Fixed header */}
            <div className="interp-header">
              <h2 className="interp-header-title">
                {diffTarget === "eye" ? "まばたき フレーム補間" : "口パク フレーム補間"}
              </h2>
              <div className="interp-header-note">
                {diffTarget === "mouth"
                  ? "See-Throughで作成したPSDと、See-Throughに渡した元画像を使用します"
                  : "See-Throughで作成した閉じ/開きPSDを使用します"}
              </div>
            </div>

            {/* Scrollable content */}
            <div className="interp-body">
              <div className="interp-workbench">
                <div className="interp-control-pane">
                  <div className="interp-workflow-rail">
                    <button className={`workflow-step ${interpStep === 1 ? "active" : ""}${visiblePairs.some(pair => !!interpPaths[pair.closed.key] && !!interpPaths[pair.open.key]) ? " done" : ""}`}
                      onClick={() => goInterpStep(1)}>1 入力</button>
                    {diffTarget === "mouth" && (
                      <button className={`workflow-step ${interpStep === 2 ? "active" : ""}${visiblePairs.some(pair => !!mouthMaskPreviews[pair.open.key]) ? " done" : ""}`}
                        disabled={!canOpenInterpStep(2)} onClick={() => goInterpStep(2)}>2 口マスク</button>
                    )}
                    <button className={`workflow-step ${interpStep === 3 || interpGenerating ? "active" : ""}${completedDiffs.length > 0 ? " done" : ""}`}
                      disabled={!canOpenInterpStep(3)} onClick={() => goInterpStep(3)}>
                      {diffTarget === "mouth" ? "3 生成" : "2 生成"}
                    </button>
                    <button className={`workflow-step ${interpStep === 4 ? "active" : ""}${diffPreviews.length > 0 ? " done" : ""}`}
                      disabled={!canOpenInterpStep(4)} onClick={() => goInterpStep(4)}>
                      {diffTarget === "mouth" ? "4 確認" : "3 確認"}
                    </button>
                  </div>
                  {interpStep === 1 && (
                  <div className={`interp-pairs-list${diffTarget === "mouth" && mouthMode === "vowels" ? " vowels" : ""}`}>
                {visiblePairs.map(pair => {
                  return (
                  <div key={pair.name} className="interp-pair-block">
                    <div className="interp-section-label">
                      {pair.label}
                    </div>
                    <div className="interp-pair-row">
                      <div className="interp-pair-item">
                        <div className="interp-pair-item-row">
                          <button className="btn btn-sm" onClick={() => pickInterpPsd(pair.closed.key)} disabled={interpGenerating}>
                            {pair.closed.label} PSD
                          </button>
                          {interpPaths[pair.closed.key]
                            ? <span className="interp-file-ok">✓</span>
                            : <span className="interp-file-empty">未選択</span>}
                        </div>
                        <div className="interp-pair-item-row">
                          {diffTarget === "mouth" && (<>
                            <button className="btn btn-sm btn-secondary" onClick={() => pickInterpOriginal(pair.closed.key)} disabled={interpGenerating}>
                              閉じ元画像
                            </button>
                            {interpOriginals[pair.closed.key]
                              ? <span className="interp-file-ok">✓</span>
                              : <span className="interp-file-empty">未選択</span>}
                          </>)}
                        </div>
                      </div>
                      <span className="interp-pair-arrow">↔</span>
                      <div className="interp-pair-item">
                        <div className="interp-pair-item-row">
                          <button className="btn btn-sm" onClick={() => pickInterpPsd(pair.open.key)} disabled={interpGenerating}>
                            {pair.open.label} PSD
                          </button>
                          {interpPaths[pair.open.key]
                            ? <span className="interp-file-ok">✓</span>
                            : <span className="interp-file-empty">未選択</span>}
                        </div>
                        <div className="interp-pair-item-row">
                          {diffTarget === "mouth" && (<>
                            <button className="btn btn-sm btn-secondary" onClick={() => pickInterpOriginal(pair.open.key)} disabled={interpGenerating}>
                              開き元画像
                            </button>
                            {interpOriginals[pair.open.key]
                              ? <span className="interp-file-ok">✓</span>
                              : <span className="interp-file-empty">未選択</span>}
                          </>)}
                        </div>
                      </div>
                    </div>
                    {diffTarget === "mouth" && (
                      <div className={`pair-status-line ${mouthMaskPreviews[pair.open.key] ? "done" : ""}`}>
                        {mouthMaskPreviews[pair.open.key] ? "口マスク作成済み" : "入力後に口マスクを作成"}
                      </div>
                    )}
                  </div>
                  );
                })}
                  </div>
                  )}
                  {interpStep === 2 && diffTarget === "mouth" && (
                    <div className="interp-step-panel">
                      <div className="action-panel-title">口マスク</div>
                      <div className="action-panel-hint">入力済みペアの口マスクを作成します。作成後は右側で余白とぼかしを調整できます。</div>
                      <div className="mask-target-list">
                        {visiblePairs.map(pair => {
                          const ready = isPairReady(pair);
                          const done = !!mouthMaskPreviews[pair.open.key];
                          return (
                            <div
                              key={pair.name}
                              className={`mask-target-item${activeMaskPair?.name === pair.name ? " active" : ""}${done ? " done" : ""}${!ready ? " disabled" : ""}`}
                              onClick={() => {
                                if (ready) setActivePreviewKey(pair.name);
                              }}
                            >
                              <span>{pair.label}</span>
                              <div className="mask-target-status">
                                <strong>{done ? "作成済み" : ready ? "未作成" : "入力待ち"}</strong>
                                {done && (
                                  <button
                                    className="mask-discard-button"
                                    title="口マスクを破棄"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      discardMouthMask(pair);
                                    }}
                                  >
                                    ×
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  <div className="interp-action-panel">
                    {interpStep === 1 ? (
                      <>
                        <div className="action-panel-title">入力</div>
                        <div className="action-panel-hint">
                          {diffTarget === "mouth"
                            ? (mouthMode === "vowels"
                              ? "閉じ/開きのPSDと元画像を最低1組設定してください。"
                              : "閉じ/開きのPSDと元画像を設定してください。")
                            : "閉じ/開きPSDを設定してください。"}
                        </div>
                        <div className="step-nav-actions">
                          <button className="btn btn-secondary" onClick={returnFromInterpInput}>前へ</button>
                          <button className="btn btn-primary" disabled={!canOpenInterpStep(diffTarget === "mouth" ? 2 : 3)}
                            onClick={() => goInterpStep(diffTarget === "mouth" ? 2 : 3)}>
                            次へ
                          </button>
                        </div>
                      </>
                    ) : interpStep === 2 && diffTarget === "mouth" ? (
                      <>
                        <div className="action-panel-title">口マスク作成</div>
                        <div className="action-panel-hint">
                          {allReadyPairsHaveMasks ? "右側でマスクを確認し、必要なら余白とぼかしを調整してください。作り直す場合は作成済み行の×で破棄できます。" : "SAM3で口マスクを作成します。結果は右側に表示されます。"}
                        </div>
                        <button className="btn btn-primary" style={{ padding: "10px 40px", fontSize: "1rem" }}
                          disabled={missingMouthMaskCount === 0 || interpGenerating || anyMouthMaskPreviewing}
                          onClick={handleCreateMouthMasks}>
                          {interpGenerating || anyMouthMaskPreviewing ? "作成中..." : "口マスク作成"}
                        </button>
                        <div className="step-nav-actions">
                          <button className="btn btn-secondary" onClick={() => goInterpStep(1)}>前へ</button>
                          <button className="btn btn-primary" disabled={!allReadyPairsHaveMasks} onClick={() => goInterpStep(3)}>
                            次へ
                          </button>
                        </div>
                      </>
                    ) : interpStep === 3 ? (
                      <>
                        <div className="interp-footer-row">
                          <span className="interp-footer-label">フレーム数:</span>
                          <input type="range" min={2} max={16} value={frameCount}
                            onChange={(e) => setFrameCount(Number(e.target.value))} style={{ flex: 1 }} />
                          <span className="frame-count-value">{frameCount}</span>
                        </div>
                        {interpGenerating && (
                          <div className="progress-bar" style={{ marginTop: 4 }}><div className="fill" style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }} /></div>
                        )}
                        <div className="interp-footer-actions generate-actions">
                          {completedDiffs.length > 0 && (
                            <div className="completed-list">
                              {completedDiffs.map(d => <span key={d} className="completed-badge">{d}</span>)}
                            </div>
                          )}
                          <button className="btn btn-primary" style={{ padding: "10px 40px", fontSize: "1rem" }}
                            disabled={!canGenerate || interpGenerating}
                            onClick={handleGenerateAll}>
                            {interpGenerating ? "生成中..." : "中間フレーム作成"}
                          </button>
                          <div className="step-nav-actions">
                            <button className="btn btn-secondary" onClick={() => goInterpStep(diffTarget === "mouth" ? 2 : 1)}>前へ</button>
                            <button className="btn btn-primary" disabled={diffPreviews.length === 0} onClick={() => goInterpStep(4)}>次へ</button>
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="action-panel-title">確認</div>
                        <div className="action-panel-hint">右側のプレビューで生成結果を確認してください。</div>
                        <div className="interp-footer-actions confirm-actions">
                          {completedDiffs.length > 0 && outputPath && (
                            <button className="btn btn-open-folder" onClick={openOutputFolder}>出力フォルダを開く</button>
                          )}
                          <div className="step-nav-actions">
                            <button className="btn btn-secondary" onClick={() => goInterpStep(3)}>前へ</button>
                            <button className="btn btn-secondary" onClick={returnToModeSelect}>モード選択へ</button>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
                <div className="interp-preview-pane">
                  <div className="preview-pane-header">
                    <span>{diffPreviews.length > 0 ? "生成プレビュー" : diffTarget === "mouth" ? "口マスクプレビュー" : "生成プレビュー"}</span>
                    <small>{diffPreviews.length > 0 ? "差分部分を拡大再生" : diffTarget === "mouth" ? "口マスク作成後にここへ表示" : "生成後にここへ表示"}</small>
                  </div>
                  {diffPreviews.length > 0 ? (
                    <>
                      {diffPreviews.length > 1 && (
                        <div className="preview-tabs">
                          {diffPreviews.map(preview => (
                            <button
                              key={preview.pairName}
                              className={`preview-tab${(activeDiffPreview?.pairName ?? "") === preview.pairName ? " active" : ""}`}
                              onClick={() => setActivePreviewKey(preview.pairName)}
                            >
                              {preview.label.replace("口パク（", "").replace("）", "")}
                            </button>
                          ))}
                        </div>
                      )}
                      {activeDiffPreview ? (() => {
                        const frameIndex = pingPongFrameIndex(diffPreviewTick, activeDiffPreview.frames.length);
                        const frame = activeDiffPreview.frames[frameIndex];
                        return (
                          <div className="diff-preview-card large">
                            <div className="diff-preview-title">{activeDiffPreview.label}</div>
                            <div className="diff-preview-stage">
                              <img
                                src={frame}
                                alt={`${activeDiffPreview.label} preview`}
                                style={{ transform: `scale(${diffPreviewZoom})` }}
                                draggable={false}
                              />
                            </div>
                            <div className="diff-preview-meta">
                              {(frameIndex + 1).toString().padStart(3, "0")} / {activeDiffPreview.frames.length}
                            </div>
                            <div className="mouth-preview-zoom-controls">
                              <button className="btn-zoom" onClick={() => setDiffPreviewZoom(prev => Math.max(1, prev - 0.25))}>-</button>
                              <input type="range" min={1} max={5} step={0.05} value={diffPreviewZoom}
                                onChange={(e) => setDiffPreviewZoom(Number(e.target.value))} />
                              <button className="btn-zoom" onClick={() => setDiffPreviewZoom(prev => Math.min(5, prev + 0.25))}>+</button>
                              <span className="zoom-level">{Math.round(diffPreviewZoom * 100)}%</span>
                            </div>
                          </div>
                        );
                      })() : null}
                    </>
                  ) : diffTarget === "mouth" && activeMaskPreview ? (
                    <>
                      {visiblePairs.filter(pair => mouthMaskPreviews[pair.open.key]).length > 1 && (
                        <div className="preview-tabs">
                          {visiblePairs.filter(pair => mouthMaskPreviews[pair.open.key]).map(pair => (
                            <button
                              key={pair.name}
                              className={`preview-tab${activeMaskPair?.name === pair.name ? " active" : ""}`}
                              onClick={() => setActivePreviewKey(pair.name)}
                            >
                              {pair.label.replace("口パク（", "").replace("）", "")}
                            </button>
                          ))}
                        </div>
                      )}
                      <div className="mouth-mask-panel preview-side">
                        <div className="mouth-mask-control-grid">
                          <label className="mouth-mask-slider">
                            <span>余白</span>
                            <input type="range" min={0} max={40} value={activeMaskSetting.dilate}
                              onChange={(e) => updateMouthMaskSetting(activeMaskSlot, { dilate: Number(e.target.value) })} />
                            <strong>{activeMaskSetting.dilate}px</strong>
                          </label>
                          <label className="mouth-mask-slider">
                            <span>ぼかし</span>
                            <input type="range" min={0} max={16} value={activeMaskSetting.blur}
                              onChange={(e) => updateMouthMaskSetting(activeMaskSlot, { blur: Number(e.target.value) })} />
                            <strong>{activeMaskSetting.blur}px</strong>
                          </label>
                        </div>
                        <div
                          className="mouth-mask-preview-viewport right-preview"
                          onPointerDown={(e) => onMouthPreviewPointerDown(e, activeMaskSlot)}
                          onPointerMove={onMouthPreviewPointerMove}
                          onPointerUp={onMouthPreviewPointerUp}
                          onPointerCancel={onMouthPreviewPointerUp}
                        >
                          <img
                            src={activeMaskPreview}
                            alt={`${activeMaskPair?.label ?? "口"} 口マスク`}
                            className="mouth-mask-preview"
                            style={{ transform: `translate(${activeMaskPan.x}px, ${activeMaskPan.y}px) scale(${activeMaskZoom})` }}
                            draggable={false}
                          />
                          <button className="mouth-preview-reset" onClick={(e) => { e.stopPropagation(); resetMouthPreviewView(activeMaskSlot); }}>
                            リセット
                          </button>
                        </div>
                        <div className="mouth-preview-zoom-controls">
                          <button className="btn-zoom" onClick={() => setMouthPreviewZoom(activeMaskSlot, activeMaskZoom - 0.25)}>-</button>
                          <input type="range" min={1} max={5} step={0.05} value={activeMaskZoom}
                            onChange={(e) => setMouthPreviewZoom(activeMaskSlot, Number(e.target.value))} />
                          <button className="btn-zoom" onClick={() => setMouthPreviewZoom(activeMaskSlot, activeMaskZoom + 0.25)}>+</button>
                          <span className="zoom-level">{Math.round(activeMaskZoom * 100)}%</span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="diff-preview-empty">
                      {diffTarget === "mouth" ? "口マスク作成後、ここにマスク確認プレビューを表示します" : "RIFE生成後、目の差分だけを拡大して再生します"}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===== Step 3A: base_edit / correction — Right Panel ===== */}
        {(mode === "base_edit" || (mode === "correction" && !!mappingPreview)) && <div className="panel-right base-edit-panel">
          <div className="preview-and-layers">
            <div className="preview-viewport" ref={previewRef}
              onWheel={handleWheel} onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
              {bodyPreview ? (
                <img src={bodyPreview} alt="Body" className="preview-img"
                  style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: isPanning ? "grabbing" : "grab" }}
                  draggable={false} />
              ) : (
                <span className="placeholder">{mode === "correction" ? "補正プレビュー" : "Bodyプレビュー"}</span>
              )}
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
              {bodyPreview && patchDraftSource && brushCursor.visible && (
                <div
                  className="patch-brush-cursor"
                  style={brushCursorStyle()}
                />
              )}
            </div>

            {bodyCategory && layerOrder.length > 0 && (
              <div className="layer-sidebar" onPointerMove={onDragPointerMove} onPointerUp={onDragPointerUp}>
                <div className="layer-sidebar-title-row">
                  <div>
                    <div className="layer-sidebar-header">レイヤー順序</div>
                    <div className="layer-sidebar-hint">上が手前</div>
                  </div>
                  {mode === "correction" && (
                    <button className="btn-layer-add" onClick={() => void addCorrectionLayerImage()} disabled={loading}>
                      PNG追加
                    </button>
                  )}
                </div>
                <div className="layer-bulk-row">
                  <button className="btn-layer-bulk" onClick={() => void setAllLayerVisibility(true)}>全ON</button>
                  <button className="btn-layer-bulk" onClick={() => void setAllLayerVisibility(false)}>全OFF</button>
                </div>
                <div className="layer-sidebar-list">
                  {layerOrder.map((name, idx) => {
                    const layer = getBodyOrderItem(name);
                    if (!layer) return null;
                    const patch = layerPatches.find(p => p.id === name);
                    const sourceName = patch?.sourceLayer ?? name;
                    const opacity = layerOpacities[name] ?? 1;
                    return (
                      <div
                        key={name}
                        className={`layer-sidebar-item${draggedIdx === idx ? " dragging" : ""}${selectedBodyLayer === name ? " selected" : ""}${layer.isPatch ? " patch" : ""}`}
                        onClick={() => setSelectedBodyLayer(name)}
                      >
                        <span className="drag-handle" onPointerDown={(e) => onDragPointerDown(e, idx)}>☰</span>
                        <input type="checkbox" checked={enabledLayers[name] !== false}
                          onChange={(e) => handleLayerToggle(name, e.target.checked)} />
                        <img src={layer.thumbnail} alt={layer.name} className="layer-sidebar-thumb" />
                        <span className="layer-sidebar-name">{layer.name}</span>
                        {opacity < 1 && (
                          <span className="layer-offset-badge">{Math.round(opacity * 100)}%</span>
                        )}
                        {layer.isPatch ? (
                          <button className="btn-layer-mini" onClick={(e) => { e.stopPropagation(); void removeLayerPatch(name); }}>
                            削除
                          </button>
                        ) : (
                          <button className="btn-layer-mini" onClick={(e) => { e.stopPropagation(); initPatchMask(sourceName); }}>
                            切出
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
                {patchDraftSource && (
                  <div className="layer-adjust-panel patch-panel">
                    <div className="layer-adjust-title">パッチ作成: {patchDraftSource}</div>
                    <div className="layer-adjust-values">塗った範囲を別レイヤーとして切り出し、元レイヤーから抜きます</div>
                    <div className="patch-tool-row">
                      <button className={`btn-nudge ${patchTool === "paint" ? "active" : ""}`} onClick={() => setPatchTool("paint")}>塗る</button>
                      <button className={`btn-nudge ${patchTool === "erase" ? "active" : ""}`} onClick={() => setPatchTool("erase")}>消す</button>
                    </div>
                    <input
                      type="range"
                      min={4}
                      max={96}
                      value={patchBrushSize}
                      onChange={(e) => setPatchBrushSize(Number(e.target.value))}
                    />
                    <div className="patch-tool-row">
                      <button className="btn-nudge btn-nudge-reset" onClick={clearPatchMask}>クリア</button>
                      <button className="btn-nudge btn-nudge-reset" onClick={() => setPatchDraftSource("")}>取消</button>
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
                    <div className="opacity-preset-row layer-only">
                      <button className="btn-nudge btn-nudge-reset" onClick={() => void setSelectedBodyOpacity(0)}>このレイヤー0%</button>
                      <button className="btn-nudge btn-nudge-reset" onClick={() => void setSelectedBodyOpacity(0.5)}>このレイヤー50%</button>
                      <button className="btn-nudge btn-nudge-reset" onClick={() => void setSelectedBodyOpacity(1)}>このレイヤー100%</button>
                    </div>
                    <div className="opacity-preset-row">
                      <button className="btn-nudge btn-nudge-reset" onClick={() => void setAllBodyOpacities(0)}>全て0%</button>
                      <button className="btn-nudge btn-nudge-reset" onClick={() => void setAllBodyOpacities(0.5)}>全て50%</button>
                      <button className="btn-nudge btn-nudge-reset" onClick={() => void setAllBodyOpacities(1)}>全て100%</button>
                    </div>
                    <button
                      className={`btn-nudge btn-nudge-reset opacity-highlight-toggle${overlapHighlightEnabled ? " active" : ""}`}
                      onClick={() => void toggleOverlapHighlight()}
                    >
                      重なり表示 {overlapHighlightEnabled ? "ON" : "OFF"}
                    </button>
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
            {mode === "correction" ? (
              <div className="interp-action-panel base-flow-action body-action">
                <div>
                  <div className="action-panel-title">{correctionOutputPath ? "出力結果確認" : "See-Through補正"}</div>
                  <div className="action-panel-hint">
                    {correctionOutputPath
                      ? "PNG保存が完了しました。出力先を開いて結果を確認できます。"
                      : "レイヤーのON/OFF、並び替え、切り出しを調整してPNG保存します。"}
                  </div>
                </div>
                {correctionOutputPath ? (
                  <div className="confirm-actions">
                    <button className="btn btn-open-folder" onClick={() => revealItemInDir(correctionOutputPath).catch(() => {})}>出力フォルダを開く</button>
                    <div className="step-nav-actions">
                      <button className="btn btn-primary" onClick={handleExportCorrection} disabled={loading || !bodyPreview}>
                        {loading ? "保存中..." : "再保存"}
                      </button>
                      <button className="btn btn-secondary" onClick={returnToModeSelect}>モード選択へ</button>
                    </div>
                  </div>
                ) : (
                  <div className="step-nav-actions base-output-actions">
                    <button className="btn btn-secondary" onClick={() => { setMode("select"); setStatus("モードを選択してください"); }}>前へ</button>
                    <button className="btn btn-primary" onClick={handleExportCorrection} disabled={loading || !bodyPreview}>
                      {loading ? "保存中..." : "PNG保存"}
                    </button>
                  </div>
                )}
              </div>
            ) : baseStep !== 4 ? (
              <div className="interp-action-panel base-flow-action body-action">
                <div>
                  <div className="action-panel-title">Bodyレイヤー調整</div>
                  <div className="action-panel-hint">レイヤー順、ON/OFF、切り出し、透明度を調整してから出力します。</div>
                </div>
                <div className="step-nav-actions base-output-actions">
                  <button className="btn btn-secondary" onClick={() => goBaseStep(2)}>前へ</button>
                  <button className="btn btn-primary" onClick={handleCreateBase} disabled={loading}>
                    {loading ? "出力中..." : "出力して確認へ"}
                  </button>
                </div>
              </div>
            ) : (
              <div className="interp-action-panel base-flow-action">
                <div>
                  <div className="action-panel-title">確認</div>
                  <div className="action-panel-hint">素体出力が完了しました。出力先を開いて結果を確認できます。</div>
                </div>
                <div className="confirm-actions">
                  {baseResult && <button className="btn btn-open-folder" onClick={() => revealItemInDir(baseResult.output_path).catch(() => {})}>出力フォルダを開く</button>}
                  <div className="step-nav-actions">
                    <button className="btn btn-secondary" onClick={() => setBaseStep(3)}>前へ</button>
                    <button className="btn btn-secondary" onClick={returnToModeSelect}>モード選択へ</button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>}

      </div>

      <div className="status-bar">
        {error && <span className="error-msg">{error}</span>}
        {!error && status}
      </div>
    </div>
  );
}

export default App;
