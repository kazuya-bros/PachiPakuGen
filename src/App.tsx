import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import "./App.css";

// --- Types ---
interface AdjustableLayer { name: string; thumbnail: string; default_target: string; }
interface SlotLoadResult { detected_layers: string[]; adjustable_layers: AdjustableLayer[]; canvas_width: number; canvas_height: number; source_type: string; }
interface LayerInfo { name: string; thumbnail: string; }
interface CategoryPreview { target: string; label: string; preview: string; layer_names: string[]; layers: LayerInfo[]; }
interface MappingPreviewResult { categories: CategoryPreview[]; composite_preview: string; }
interface RenderCategoryResult { preview: string; }
interface CreateBaseResult { output_path: string; composite_preview: string; base_eye_slot: string; base_mouth_slot: string; file_count: number; }
interface CreateDiffResult { output_path: string; pair_name: string; frame_count: number; preview: string; }
interface ProgressPayload { current: number; total: number; pair_name: string; }

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

type Mode = "select" | "base_input" | "hair_edit" | "base_edit" | "interp";

function App() {
  const [mode, setMode] = useState<Mode>("select");

  // Shared
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("モードを選択してください");

  // === Input (shared) ===
  const [loadResult, setLoadResult] = useState<SlotLoadResult | null>(null);
  const [layerMapping, setLayerMapping] = useState<Record<string, string>>({});
  const [originalImagePath, setOriginalImagePath] = useState("");

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
  const [baseResult, setBaseResult] = useState<CreateBaseResult | null>(null);

  // === フレーム補間モード (interp) ===
  const [diffTarget, setDiffTarget] = useState<"eye" | "mouth">("eye");
  const [frameCount, setFrameCount] = useState(4);
  const [outputPath, setOutputPath] = useState("");
  const [progress, setProgress] = useState({ current: 0, total: 0, pair_name: "" });
  const [completedDiffs, setCompletedDiffs] = useState<string[]>([]);
  const [interpPaths, setInterpPaths] = useState<Record<string, string>>({});
  const [interpOriginals, setInterpOriginals] = useState<Record<string, string>>({});
  const [interpGenerating, setInterpGenerating] = useState(false);
  const [mouthMode, setMouthMode] = useState<"single" | "vowels">("vowels");

  // Zoom & pan (base_edit)
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });
  const previewRef = useRef<HTMLDivElement>(null);

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
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // --- Category rendering ---
  async function renderCategory(order: string[], enabled: Record<string, boolean>): Promise<string> {
    const active = [...order.filter(name => enabled[name] !== false)].reverse();
    try {
      const result = await invoke<RenderCategoryResult>("render_category", {
        mappingJson: JSON.stringify(layerMapping), target: "body", enabledLayers: active,
      });
      return result.preview;
    } catch (e) { console.error("render error:", e); return ""; }
  }

  async function renderHair(order: string[], enabled: Record<string, boolean>) {
    setHairPreview(await renderCategory(order, enabled));
  }

  async function renderHairBack(order: string[], enabled: Record<string, boolean>) {
    setHairBackPreview(await renderCategory(order, enabled));
  }

  // --- Body rendering ---
  async function renderBody(order: string[], enabled: Record<string, boolean>) {
    const active = [...order.filter(name => enabled[name] !== false)].reverse();
    try {
      const result = await invoke<RenderCategoryResult>("render_category", {
        mappingJson: JSON.stringify(layerMapping), target: "body", enabledLayers: active,
      });
      setBodyPreview(result.preview);
    } catch (e) { console.error("render error:", e); }
  }

  async function handleLayerToggle(name: string, checked: boolean) {
    const newEnabled = { ...enabledLayers, [name]: checked };
    setEnabledLayers(newEnabled);
    await renderBody(layerOrder, newEnabled);
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

  async function pickOriginalImage() {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "画像", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (file) setOriginalImagePath(file);
  }

  async function proceedToHairEdit() {
    setLoading(true); setStatus("SAM3で首を検出中...");
    try {
      await invoke<string>("load_original_image", { path: originalImagePath });
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
      resetZoom(); setMode("hair_edit"); setStatus("Step 1/2: Hairレイヤーを確認");
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  async function proceedToBodyEdit() {
    const bodyCat = mappingPreview?.categories.find(c => c.target === "body");
    if (bodyCat) {
      const order = bodyCat.layers.map(l => l.name);
      setLayerOrder(order);
      const en: Record<string, boolean> = {};
      for (const l of bodyCat.layers) en[l.name] = true;
      setEnabledLayers(en);
      await renderBody(order, en);
    }
    resetZoom(); setMode("base_edit"); setStatus("Step 2/2: Bodyレイヤーを確認して出力");
  }

  function proceedToInterp(target: "eye" | "mouth") {
    setDiffTarget(target);
    setCompletedDiffs([]);
    setMode("interp");
    setStatus("差分PSD+元画像を設定してください");
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
        mappingJson: JSON.stringify(layerMapping), originalImagePath,
        baseEyeSlot: "eye_open", baseMouthSlot: "mouth_closed",
        bodyLayerOrder: activeOrder, hairLayerOrder: activeHairOrder,
        hairBackLayerOrder: activeHairBackOrder, outputPath: dir,
      });
      setBaseResult(result);
      setStatus(`素体作成完了: ${result.file_count}ファイル出力`);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  // === Step 3B: interp ===
  async function pickInterpPsd(slot: string) {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "PSD", extensions: ["psd"] }] });
    if (file) setInterpPaths(prev => ({ ...prev, [slot]: file }));
  }

  async function pickInterpOriginal(slotKey: string) {
    setError("");
    const file = await open({ multiple: false, directory: false, filters: [{ name: "画像", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (file) setInterpOriginals(prev => ({ ...prev, [slotKey]: file }));
  }

  // Pairs for current interp mode
  const visiblePairs = diffTarget === "eye"
    ? EYE_PAIRS
    : (mouthMode === "single" ? MOUTH_PAIRS_SINGLE : MOUTH_PAIRS_VOWELS);

  function isPairReady(pair: typeof EYE_PAIRS[0]) {
    return !!interpPaths[pair.closed.key] && !!interpPaths[pair.open.key]
      && !!interpOriginals[pair.closed.key] && !!interpOriginals[pair.open.key];
  }

  const canGenerate = visiblePairs.some(isPairReady);

  async function handleGenerateAll() {
    setError("");
    const dir = outputPath || await open({ multiple: false, directory: true, title: "出力先フォルダを選択" });
    if (!dir) return;
    setOutputPath(dir);
    setInterpGenerating(true);
    setCompletedDiffs([]);

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

        // Load original image for SAM3 (use "open" original — open mouth is easier to detect)
        const openOriginal = interpOriginals[pair.open.key];
        if (openOriginal) {
          setStatus(`${pair.label}: SAM3で口・首を検出中...`);
          await invoke<string>("load_original_image", { path: openOriginal });
        }

        // Determine base slot names
        const pairDiffType = pair.name.startsWith("mouth") ? "mouth" : "eye";
        const baseSlotEye = pairDiffType === "eye" ? "eye_closed" : "eye_closed";
        const baseSlotMouth = pairDiffType === "mouth" ? "mouth_closed" : "mouth_closed";

        await invoke<CreateBaseResult>("create_base", {
          mappingJson, originalImagePath: interpOriginals[pair.closed.key] || "",
          baseEyeSlot: baseSlotEye, baseMouthSlot: baseSlotMouth,
          bodyLayerOrder: [] as string[], hairLayerOrder: [] as string[],
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
          originalImagePath: interpOriginals[pair.open.key] || "",
        });
        setCompletedDiffs(prev => [...prev, diffResult.pair_name]);
      }

      setStatus(`生成完了: ${readyPairs.length}パーツ`);
    } catch (e) { setError(String(e)); }
    finally { setInterpGenerating(false); }
  }

  async function openOutputFolder() {
    if (outputPath) try { await revealItemInDir(outputPath); } catch (_) {}
  }

  const bodyCategory = mappingPreview?.categories.find(c => c.target === "body");

  return (
    <div className="app">
      <div className="app-header">
        <h1>PachiPakuGen</h1>
        <span className="version">v0.2.0</span>
      </div>

      {/* Top bar for hair_edit mode */}
      {mode === "hair_edit" && (
        <div className="top-bar">
          <button className="btn-nav" onClick={() => setMode("base_input")}>&larr; 戻る</button>
          <span className="top-bar-title">Step 1/2: Hairレイヤー確認</span>
          <button className="btn btn-primary" onClick={proceedToBodyEdit}>
            次へ → Body編集
          </button>
        </div>
      )}

      {/* Top bar for base_edit mode */}
      {mode === "base_edit" && (
        <div className="top-bar">
          <button className="btn-nav" onClick={() => setMode("hair_edit")}>&larr; 戻る</button>
          <span className="top-bar-title">Step 2/2: Bodyレイヤー確認 — 並び替え・ON/OFFで調整</span>
          <button className="btn btn-primary" onClick={handleCreateBase} disabled={loading}>
            {loading ? "作成中..." : "素体出力"}
          </button>
          {baseResult && (
            <div className="top-bar-done">
              <span className="check-mark-inline">✓</span> {baseResult.file_count}ファイル出力
              <button className="btn btn-open-folder btn-sm" onClick={() => { if (baseResult) revealItemInDir(baseResult.output_path).catch(() => {}); }}>開く</button>
            </div>
          )}
        </div>
      )}

      <div className="main-content">
        {/* ===== Mode Select (3 buttons) ===== */}
        {mode === "select" && (
          <div className="mode-select-screen">
            <div className="output-select-buttons">
              <button className="btn-output" onClick={() => { setMode("base_input"); setStatus("PSDと元画像を読み込んでください"); }}>
                <span className="btn-output-title">素体出力</span>
                <span className="btn-output-desc">See-Through PSD + 元画像</span>
                <span className="btn-output-desc">→ body / hair / hair_back</span>
              </button>
              <button className="btn-output" onClick={() => proceedToInterp("eye")}>
                <span className="btn-output-title">まばたき</span>
                <span className="btn-output-desc">表情差分PSD × 2</span>
                <span className="btn-output-desc">→ RIFE中間フレーム</span>
              </button>
              <button className="btn-output" onClick={() => proceedToInterp("mouth")}>
                <span className="btn-output-title">口パク</span>
                <span className="btn-output-desc">表情差分PSD × 2</span>
                <span className="btn-output-desc">→ RIFE中間フレーム</span>
              </button>
            </div>
          </div>
        )}

        {/* ===== 素体: File Input ===== */}
        {mode === "base_input" && (
          <div className="mode-select-screen">
            <div className="input-card-center">
              <h2 className="input-card-title">素体出力 — ファイル読み込み</h2>
              <div className="file-input-row-large">
                <span className="file-input-label-large">PSD:</span>
                <button className="btn btn-primary" onClick={loadPsd} disabled={loading}>選択</button>
                <span className="slot-path-inline-large">{loadResult ? `${loadResult.detected_layers.length}レイヤー ✓` : "未選択"}</span>
              </div>
              <p className="input-hint">See-Throughで分解したPSDファイル</p>
              <div className="file-input-row-large">
                <span className="file-input-label-large">元画像:</span>
                <button className="btn btn-primary" onClick={pickOriginalImage} disabled={loading}>選択</button>
                <span className="slot-path-inline-large">{originalImagePath ? `${originalImagePath.split(/[/\\]/).pop()} ✓` : "未選択"}</span>
              </div>
              <p className="input-hint">See-Throughに入力した元のイラスト画像</p>
              {loading && (
                <div className="progress-bar indeterminate" style={{ marginTop: 12 }}><div className="fill" /></div>
              )}
              <button className="btn btn-primary btn-full" style={{ marginTop: 20, padding: "12px 16px", fontSize: "1rem" }}
                disabled={!loadResult || !originalImagePath || loading}
                onClick={proceedToHairEdit}>
                次へ → Hair編集
              </button>
              <button className="btn-nav" style={{ marginTop: 12 }} onClick={() => setMode("select")}>&larr; モード選択に戻る</button>
            </div>
          </div>
        )}

        {/* ===== Step 1/2: Hair Edit ===== */}
        {mode === "hair_edit" && (
          <div className="panel-right">
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
          </div>
        )}

        {/* ===== フレーム補間: diff PSD input ===== */}
        {mode === "interp" && (
          <div className="interp-screen">
            {/* Fixed header */}
            <div className="interp-header">
              <button className="btn-nav" onClick={() => setMode("select")}>&larr; 戻る</button>
              <h2 className="interp-header-title">
                {diffTarget === "eye" ? "まばたき フレーム補間" : "口パク フレーム補間"}
              </h2>
              {diffTarget === "mouth" && (
                <div className="interp-header-right">
                  <span className="interp-header-label">モード:</span>
                  <select className="interp-mode-select" value={mouthMode}
                    onChange={(e) => setMouthMode(e.target.value as "single" | "vowels")}>
                    <option value="single">mouth のみ</option>
                    <option value="vowels">あ〜お（母音別）</option>
                  </select>
                </div>
              )}
            </div>

            {/* Scrollable content */}
            <div className="interp-body">
              <div className="interp-pairs-list">
                {visiblePairs.map(pair => (
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
                          <button className="btn btn-sm btn-secondary" onClick={() => pickInterpOriginal(pair.closed.key)} disabled={interpGenerating}>
                            元画像
                          </button>
                          {interpOriginals[pair.closed.key]
                            ? <span className="interp-file-ok">✓</span>
                            : <span className="interp-file-empty">未選択</span>}
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
                          <button className="btn btn-sm btn-secondary" onClick={() => pickInterpOriginal(pair.open.key)} disabled={interpGenerating}>
                            元画像
                          </button>
                          {interpOriginals[pair.open.key]
                            ? <span className="interp-file-ok">✓</span>
                            : <span className="interp-file-empty">未選択</span>}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Fixed footer */}
            <div className="interp-footer">
              <div className="interp-footer-row">
                <span className="interp-footer-label">フレーム数:</span>
                <input type="range" min={2} max={16} value={frameCount}
                  onChange={(e) => setFrameCount(Number(e.target.value))} style={{ flex: 1, maxWidth: 300 }} />
                <span className="frame-count-value">{frameCount}</span>
              </div>
              {interpGenerating && (
                <div className="progress-bar" style={{ marginTop: 4 }}><div className="fill" style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }} /></div>
              )}
              <div className="interp-footer-actions">
                {completedDiffs.length > 0 && (
                  <div className="completed-list">
                    {completedDiffs.map(d => <span key={d} className="completed-badge">{d}</span>)}
                  </div>
                )}
                <button className="btn btn-primary" style={{ padding: "10px 40px", fontSize: "1rem" }}
                  disabled={!canGenerate || interpGenerating}
                  onClick={handleGenerateAll}>
                  {interpGenerating ? "生成中..." : "一括生成"}
                </button>
                {completedDiffs.length > 0 && outputPath && (
                  <button className="btn btn-open-folder" onClick={openOutputFolder}>出力フォルダを開く</button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ===== Step 3A: base_edit — Right Panel ===== */}
        {mode === "base_edit" && <div className="panel-right">
          <div className="preview-and-layers">
            <div className="preview-viewport" ref={previewRef}
              onWheel={handleWheel} onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
              {bodyPreview ? (
                <img src={bodyPreview} alt="Body" className="preview-img"
                  style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: isPanning ? "grabbing" : "grab" }}
                  draggable={false} />
              ) : (
                <span className="placeholder">Bodyプレビュー</span>
              )}
            </div>

            {bodyCategory && layerOrder.length > 0 && (
              <div className="layer-sidebar" onPointerMove={onDragPointerMove} onPointerUp={onDragPointerUp}>
                <div className="layer-sidebar-header">レイヤー順序</div>
                <div className="layer-sidebar-hint">上が手前</div>
                <div className="layer-sidebar-list">
                  {layerOrder.map((name, idx) => {
                    const layer = bodyCategory.layers.find(l => l.name === name);
                    if (!layer) return null;
                    return (
                      <div key={layer.name} className={`layer-sidebar-item${draggedIdx === idx ? " dragging" : ""}`}>
                        <span className="drag-handle" onPointerDown={(e) => onDragPointerDown(e, idx)}>☰</span>
                        <input type="checkbox" checked={enabledLayers[layer.name] !== false}
                          onChange={(e) => handleLayerToggle(layer.name, e.target.checked)} />
                        <img src={layer.thumbnail} alt={layer.name} className="layer-sidebar-thumb" />
                        <span className="layer-sidebar-name">{layer.name}</span>
                      </div>
                    );
                  })}
                </div>
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
