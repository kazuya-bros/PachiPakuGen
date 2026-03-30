import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import BrushCanvas, { BrushTarget } from "./BrushCanvas";
import "./App.css";

// --- Types ---

interface PartPreview {
  preview: string;
  width: number;
  height: number;
  has_alpha: boolean;
}

interface LoadPartsResult {
  canvas_width: number;
  canvas_height: number;
  part_count: number;
}

interface ModelAvailability {
  unet: boolean;
  sam3: boolean;
}

interface MaskInfo {
  eye_pixels: number;
  mouth_pixels: number;
  eyebrow_pixels: number;
  has_eye: boolean;
  has_mouth: boolean;
  has_eyebrow: boolean;
  overlay_preview: string;
}

interface ApplyMasksResult {
  pair_count: number;
  composite_preview: string;
}

interface PairResult {
  name: string;
  frame_count: number;
  previews: string[];
}

interface GenerationResult {
  total_frames: number;
  pair_results: PairResult[];
}

interface ProgressPayload {
  current: number;
  total: number;
  pair_name: string;
  ratio: number;
}

interface ExportResult {
  output_path: string;
  total_files: number;
  pair_dirs: string[];
}

// --- Part definitions ---

type PartKey =
  | "body" | "hair"
  | "eye_open" | "eye_closed"
  | "mouth_closed" | "mouth_a" | "mouth_i" | "mouth_u" | "mouth_e" | "mouth_o";

const PART_LABELS: Record<PartKey, string> = {
  body: "素体(body)",
  hair: "髪(hair)",
  eye_open: "目_開",
  eye_closed: "目_閉",
  mouth_closed: "口_閉",
  mouth_a: "口_あ",
  mouth_i: "口_い",
  mouth_u: "口_う",
  mouth_e: "口_え",
  mouth_o: "口_お",
};

const REQUIRED_PARTS: PartKey[] = ["body", "hair", "eye_open", "eye_closed", "mouth_closed"];
const MOUTH_VOWELS: PartKey[] = ["mouth_a", "mouth_i", "mouth_u", "mouth_e", "mouth_o"];

function App() {
  // Step management (4 steps)
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3 | 4>(1);

  // Step 1: Part loading
  const [partPaths, setPartPaths] = useState<Record<string, string>>({});
  const [partThumbs, setPartThumbs] = useState<Record<string, string>>({});
  const [partsResult, setPartsResult] = useState<LoadPartsResult | null>(null);

  // Step 2: Segmentation + Brush
  const [modelAvail, setModelAvail] = useState<ModelAvailability | null>(null);
  const [segModel] = useState("sam3");
  const [segTargets, setSegTargets] = useState({ eye: true, mouth: true, eyebrow: true });
  const [segLoading, setSegLoading] = useState(false);
  const [maskInfo, setMaskInfo] = useState<MaskInfo | null>(null);
  const [overlayPreview, setOverlayPreview] = useState("");
  const [brushTarget, setBrushTarget] = useState<BrushTarget | null>(null);
  const [brushRefreshKey, setBrushRefreshKey] = useState(0);
  const [pairsResult, setPairsResult] = useState<ApplyMasksResult | null>(null);

  // Step 3: Generation
  const [frameCount, setFrameCount] = useState(4);
  const [genLoading, setGenLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, pair_name: "" });
  const [genResult, setGenResult] = useState<GenerationResult | null>(null);

  // Step 4: Preview & Export
  const [eyeFrame, setEyeFrame] = useState(0);
  const [mouthPair, setMouthPair] = useState("");
  const [mouthFrame, setMouthFrame] = useState(0);
  const [compositePreview, setCompositePreview] = useState("");
  const previewDebounce = useRef<number | null>(null);

  // Global
  const [error, setError] = useState("");
  const [status, setStatus] = useState("準備完了");

  // Export done banner
  const [exportDone, setExportDone] = useState<{
    totalFiles: number;
    pairDirs: string[];
    outputPath: string;
  } | null>(null);

  // Listen for generation progress
  useEffect(() => {
    const unlisten = listen<ProgressPayload>("generation-progress", (event) => {
      setProgress({
        current: event.payload.current,
        total: event.payload.total,
        pair_name: event.payload.pair_name,
      });
      setStatus(`生成中: ${event.payload.pair_name} (${event.payload.current}/${event.payload.total})`);
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Check model availability when entering step 2
  useEffect(() => {
    if (currentStep === 2 && partsResult && !modelAvail) {
      invoke<ModelAvailability>("check_model_availability").then(setModelAvail).catch(() => {});
    }
  }, [currentStep, partsResult, modelAvail]);

  // --- Step 1: Part loading ---
  async function pickPart(key: PartKey) {
    setError("");
    const file = await open({
      multiple: false,
      filters: [{ name: "透過PNG", extensions: ["png"] }],
    });
    if (!file) return;
    setPartPaths(prev => ({ ...prev, [key]: file }));

    try {
      const sp = await invoke<PartPreview>("load_part_preview", { path: file });
      setPartThumbs(prev => ({ ...prev, [key]: sp.preview }));
      if (!sp.has_alpha && key !== "body") {
        setError(`警告: ${file.split(/[/\\]/).pop()} にアルファチャンネルがありません`);
      }
    } catch (_) { /* ignore */ }
  }

  const hasRequired = REQUIRED_PARTS.every(k => !!partPaths[k]);
  const hasMouthVowel = MOUTH_VOWELS.some(k => !!partPaths[k]);
  const canLoadParts = hasRequired && hasMouthVowel;

  const [partsLoading, setPartsLoading] = useState(false);

  async function loadAllParts() {
    setError("");
    setPartsLoading(true);
    setStatus("パーツを読み込み中...");
    try {
      const partsJson = JSON.stringify(partPaths);
      const result = await invoke<LoadPartsResult>("load_parts", { partsJson });
      setPartsResult(result);
      setMaskInfo(null);
      setOverlayPreview("");
      setPairsResult(null);
      setGenResult(null);
      setModelAvail(null);
      setStatus(`読み込み完了: ${result.canvas_width}x${result.canvas_height} / ${result.part_count}パーツ`);
      setCurrentStep(2);
    } catch (e) {
      setError(String(e));
    } finally {
      setPartsLoading(false);
    }
  }

  // --- Step 2: Segmentation + Brush ---
  async function runSegmentation() {
    setError("");
    setSegLoading(true);
    setMaskInfo(null);
    setOverlayPreview("");
    setBrushTarget(null);
    const targetStr = [
      segTargets.eye ? "eye" : "",
      segTargets.mouth ? "mouth" : "",
      segTargets.eyebrow ? "eyebrow" : "",
    ].filter(Boolean).join(",");
    setStatus("セグメンテーション実行中 (SAM3)...");
    try {
      const result = await invoke<MaskInfo>("run_segmentation_cmd", {
        model: segModel,
        targets: targetStr,
      });
      setMaskInfo(result);
      setOverlayPreview(result.overlay_preview);
      const info = [];
      if (result.has_eye) info.push(`目: ${result.eye_pixels}px`);
      if (result.has_mouth) info.push(`口: ${result.mouth_pixels}px`);
      if (result.has_eyebrow) info.push(`眉: ${result.eyebrow_pixels}px`);
      setStatus(`セグメンテーション完了: ${info.join(" / ")}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setSegLoading(false);
    }
  }

  function handleBrushApplied(result: { overlay_preview: string }) {
    setOverlayPreview(result.overlay_preview);
  }

  async function applyMasksAndBuildPairs() {
    setError("");
    setStatus("マスク適用中...");
    try {
      const result = await invoke<ApplyMasksResult>("apply_masks_and_build_pairs");
      setPairsResult(result);
      setGenResult(null);
      setStatus(`マスク適用完了: ${result.pair_count}ペア`);
      setCurrentStep(3);
    } catch (e) {
      setError(String(e));
    }
  }

  // --- Step 3: Generation ---
  async function generateFrames() {
    setError("");
    setGenLoading(true);
    setGenResult(null);
    setProgress({ current: 0, total: 0, pair_name: "" });
    setStatus("RIFEモデルを初期化中...");
    try {
      const result = await invoke<GenerationResult>("generate_frames", { frameCount });
      setGenResult(result);
      setEyeFrame(0);
      setMouthFrame(0);
      // Always set mouthPair to the first available mouth pair
      const firstMouth = result.pair_results.find(p => p.name.startsWith("mouth_"));
      setMouthPair(firstMouth ? firstMouth.name : "");
      setStatus(`生成完了: ${result.pair_results.length}ペア × ${frameCount}フレーム`);
      setCurrentStep(4);
    } catch (e) {
      setError(String(e));
    } finally {
      setGenLoading(false);
    }
  }

  // --- Step 4: Composite preview (debounced) ---
  useEffect(() => {
    if (!genResult) return;
    if (previewDebounce.current) clearTimeout(previewDebounce.current);
    previewDebounce.current = window.setTimeout(async () => {
      try {
        const preview = await invoke<string>("get_composite_preview", {
          eyeFrame,
          mouthPair,
          mouthFrame,
        });
        setCompositePreview(preview);
      } catch (e) {
        console.error("Preview error:", e);
        setError(String(e));
      }
    }, 50);
    return () => {
      if (previewDebounce.current) clearTimeout(previewDebounce.current);
    };
  }, [eyeFrame, mouthPair, mouthFrame, genResult]);

  // --- Export ---
  async function exportFrames() {
    setError("");
    setExportDone(null);
    const dir = await save({
      title: "出力先フォルダを選択",
      defaultPath: "PachiPakuGen_output",
    });
    if (!dir) return;
    setStatus("エクスポート中...");
    try {
      const result = await invoke<ExportResult>("export_frames", { outputPath: dir });
      setStatus(`${result.total_files}ファイルを出力完了`);
      setExportDone({
        totalFiles: result.total_files,
        pairDirs: result.pair_dirs,
        outputPath: result.output_path,
      });
    } catch (e) {
      setError(String(e));
    }
  }

  async function openExportFolder() {
    if (!exportDone) return;
    try {
      await revealItemInDir(exportDone.outputPath);
    } catch (e) {
      console.error("Failed to open folder:", e);
    }
  }

  // --- Derived ---
  const availableMouthPairs = genResult
    ? genResult.pair_results.filter(p => p.name.startsWith("mouth_"))
    : [];
  const maxFrame = frameCount - 1;
  const stepLabels = ["パーツ入力", "セグメント", "フレーム生成", "プレビュー"];

  return (
    <div className="app">
      <div className="app-header">
        <h1>PachiPakuGen</h1>
        <span className="version">v0.1.0</span>
      </div>

      {/* Step Indicator */}
      <div className="step-indicator">
        {[1, 2, 3, 4].map((step, i) => (
          <div key={step} style={{ display: "flex", alignItems: "center" }}>
            <div
              className={`step-item${currentStep === step ? " active" : ""}${
                (step === 1 && !!partsResult) ||
                (step === 2 && !!pairsResult) ||
                (step === 3 && !!genResult)
                  ? " completed" : ""
              }`}
              onClick={() => setCurrentStep(step as 1 | 2 | 3 | 4)}
            >
              <div className="step-dot">{step}</div>
              <span className="step-label">{stepLabels[i]}</span>
            </div>
            {i < 3 && (
              <div className={`step-connector${
                (i === 0 && !!partsResult) ||
                (i === 1 && !!pairsResult) ||
                (i === 2 && !!genResult)
                  ? " done" : ""
              }`} />
            )}
          </div>
        ))}
      </div>

      <div className="main-content">
        <div className="panel-left">

          {/* ===== STEP 1: Part Loading ===== */}
          {currentStep === 1 && (
            <>
              <div className="card">
                <h2>必須パーツ</h2>
                <div className="parts-grid">
                  {REQUIRED_PARTS.map(key => (
                    <div key={key} className="drop-zone drop-zone-part" onClick={() => pickPart(key)}>
                      {partThumbs[key] ? (
                        <img src={partThumbs[key]} alt={PART_LABELS[key]} className="part-thumb" />
                      ) : (
                        <span className="label">
                          {partPaths[key] ? partPaths[key].split(/[/\\]/).pop() : "クリック"}
                        </span>
                      )}
                      <span className="label part-label">{PART_LABELS[key]}</span>
                      {partPaths[key] && <span className="check-mark">✓</span>}
                    </div>
                  ))}
                </div>
              </div>

              <div className="card">
                <h2>口パーツ（1つ以上必須）</h2>
                <div className="parts-grid">
                  {MOUTH_VOWELS.map(key => (
                    <div key={key} className="drop-zone drop-zone-part" onClick={() => pickPart(key)}>
                      {partThumbs[key] ? (
                        <img src={partThumbs[key]} alt={PART_LABELS[key]} className="part-thumb" />
                      ) : (
                        <span className="label">
                          {partPaths[key] ? partPaths[key].split(/[/\\]/).pop() : "クリック"}
                        </span>
                      )}
                      <span className="label part-label">{PART_LABELS[key]}</span>
                      {partPaths[key] && <span className="check-mark">✓</span>}
                    </div>
                  ))}
                </div>
              </div>

              <button
                className="btn btn-primary btn-full"
                style={{ marginTop: 8 }}
                disabled={!canLoadParts || partsLoading}
                onClick={loadAllParts}
              >
                {partsLoading ? "読み込み中..." : "読み込み → セグメントへ"}
              </button>
            </>
          )}

          {/* ===== STEP 2: Segmentation + Brush ===== */}
          {currentStep === 2 && (
            <>
              <div className="card">
                <h2>セグメンテーション (SAM3)</h2>

                {!modelAvail?.sam3 && (
                  <div className="error-msg" style={{ marginBottom: 8 }}>
                    SAM3モデル(sam3.pt)が見つかりません。models/ フォルダに配置してください。
                  </div>
                )}

                {/* Target checkboxes */}
                <div className="seg-targets">
                  <label>
                    <input type="checkbox" checked={segTargets.eye}
                      onChange={(e) => setSegTargets(prev => ({ ...prev, eye: e.target.checked }))}
                    /> 目
                  </label>
                  <label>
                    <input type="checkbox" checked={segTargets.mouth}
                      onChange={(e) => setSegTargets(prev => ({ ...prev, mouth: e.target.checked }))}
                    /> 口
                  </label>
                  <label>
                    <input type="checkbox" checked={segTargets.eyebrow}
                      onChange={(e) => setSegTargets(prev => ({ ...prev, eyebrow: e.target.checked }))}
                    /> 眉毛
                  </label>
                </div>

                <button
                  className="btn btn-primary btn-full"
                  onClick={runSegmentation}
                  disabled={segLoading || !partsResult || !modelAvail?.sam3}
                >
                  {segLoading ? "検出中..." : "セグメンテーション実行"}
                </button>

                {maskInfo && (
                  <div className="mask-info">
                    {maskInfo.has_eye && <div>目: {maskInfo.eye_pixels.toLocaleString()} px</div>}
                    {maskInfo.has_mouth && <div>口: {maskInfo.mouth_pixels.toLocaleString()} px</div>}
                    {maskInfo.has_eyebrow && <div>眉: {maskInfo.eyebrow_pixels.toLocaleString()} px</div>}
                  </div>
                )}
              </div>

              {/* Brush editing section */}
              {maskInfo && (
                <div className="card">
                  <h2>ブラシ補正</h2>
                  <div className="brush-targets">
                    {maskInfo.has_eye && (
                      <button
                        className={`btn-brush-target${brushTarget === "eye" ? " active" : ""}`}
                        onClick={() => { setBrushTarget(brushTarget === "eye" ? null : "eye"); setBrushRefreshKey(k => k + 1); }}
                      >目</button>
                    )}
                    {maskInfo.has_mouth && (
                      <button
                        className={`btn-brush-target${brushTarget === "mouth" ? " active" : ""}`}
                        onClick={() => { setBrushTarget(brushTarget === "mouth" ? null : "mouth"); setBrushRefreshKey(k => k + 1); }}
                      >口</button>
                    )}
                    {maskInfo.has_eyebrow && (
                      <button
                        className={`btn-brush-target${brushTarget === "eyebrow" ? " active" : ""}`}
                        onClick={() => { setBrushTarget(brushTarget === "eyebrow" ? null : "eyebrow"); setBrushRefreshKey(k => k + 1); }}
                      >眉</button>
                    )}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", marginTop: 4 }}>
                    ターゲットを選んでブラシで補正（左: 追加 / 右: 消去）
                  </div>
                </div>
              )}

              {/* Apply masks and move to step 3 */}
              {maskInfo && (
                <button
                  className="btn btn-primary btn-full"
                  style={{ marginTop: 0 }}
                  onClick={applyMasksAndBuildPairs}
                >
                  マスク確定 → フレーム生成へ
                </button>
              )}

              <div className="step-nav">
                <button className="btn-nav" onClick={() => setCurrentStep(1)}>
                  &larr; 戻る
                </button>
                <button
                  className="btn-nav"
                  disabled={!pairsResult}
                  onClick={() => setCurrentStep(3)}
                >
                  次へ &rarr;
                </button>
              </div>
            </>
          )}

          {/* ===== STEP 3: Generation ===== */}
          {currentStep === 3 && (
            <>
              <div className="card">
                <h2>パーツ確認</h2>
                {pairsResult && (
                  <div className="parts-info">
                    <div>キャンバス: {partsResult?.canvas_width}x{partsResult?.canvas_height}</div>
                    <div>RIFEペア数: {pairsResult.pair_count}</div>
                  </div>
                )}
              </div>

              <div className="card">
                <h2>フレーム生成</h2>
                <div className="slider-row">
                  <span style={{ fontSize: "0.8rem" }}>フレーム数:</span>
                  <input
                    type="range" min={2} max={16}
                    value={frameCount}
                    onChange={(e) => setFrameCount(Number(e.target.value))}
                  />
                  <span className="value">{frameCount}</span>
                </div>
                <button
                  className="btn btn-primary btn-full"
                  style={{ marginTop: 8 }}
                  onClick={generateFrames}
                  disabled={!pairsResult || genLoading}
                >
                  {genLoading ? "生成中..." : "全ペア一括生成"}
                </button>
                {genLoading && (
                  <>
                    <div className="progress-bar">
                      <div
                        className="fill"
                        style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }}
                      />
                    </div>
                    <div className="progress-text">
                      {progress.pair_name} {progress.current}/{progress.total}
                    </div>
                  </>
                )}
              </div>

              <div className="step-nav">
                <button className="btn-nav" onClick={() => setCurrentStep(2)}>
                  &larr; 戻る
                </button>
                <button
                  className="btn-nav"
                  disabled={!genResult}
                  onClick={() => setCurrentStep(4)}
                >
                  次へ &rarr;
                </button>
              </div>
            </>
          )}

          {/* ===== STEP 4: Preview & Export ===== */}
          {currentStep === 4 && (
            <>
              <div className="card">
                <h2>プレビュー操作</h2>

                {/* Eye slider */}
                <div className="slider-section">
                  <span className="slider-section-label">目（開↔閉）</span>
                  <div className="slider-row">
                    <span style={{ fontSize: "0.7rem" }}>開</span>
                    <input
                      type="range" min={0} max={maxFrame}
                      value={eyeFrame}
                      onChange={(e) => setEyeFrame(Number(e.target.value))}
                    />
                    <span style={{ fontSize: "0.7rem" }}>閉</span>
                    <span className="value">{eyeFrame}</span>
                  </div>
                </div>

                {/* Mouth pair selector + slider */}
                <div className="slider-section">
                  <div className="mouth-selector">
                    <span className="slider-section-label">口</span>
                    <div className="mouth-buttons">
                      {availableMouthPairs.map(p => (
                        <button
                          key={p.name}
                          className={`btn btn-sm${mouthPair === p.name ? " btn-active" : ""}`}
                          onClick={() => { setMouthPair(p.name); setMouthFrame(0); }}
                        >
                          {p.name.replace("mouth_", "")}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="slider-row">
                    <span style={{ fontSize: "0.7rem" }}>閉</span>
                    <input
                      type="range" min={0} max={maxFrame}
                      value={mouthFrame}
                      onChange={(e) => setMouthFrame(Number(e.target.value))}
                    />
                    <span style={{ fontSize: "0.7rem" }}>開</span>
                    <span className="value">{mouthFrame}</span>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>エクスポート</h2>
                {exportDone ? (
                  <div className="export-done-banner">
                    <div className="export-done-icon">✓</div>
                    <div className="export-done-text">
                      <div className="export-done-title">エクスポート完了</div>
                      <div className="export-done-detail">
                        {exportDone.totalFiles}ファイル出力
                        <span className="export-done-pairs">
                          ({exportDone.pairDirs.join(", ")})
                        </span>
                      </div>
                    </div>
                    <button
                      className="btn btn-open-folder"
                      onClick={openExportFolder}
                    >
                      フォルダを開く
                    </button>
                    <button
                      className="btn btn-export-again"
                      onClick={() => setExportDone(null)}
                    >
                      再出力
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="export-info">
                      パーツ別透過PNG連番として出力します
                    </div>
                    <button
                      className="btn btn-primary btn-full"
                      style={{ marginTop: 8 }}
                      onClick={exportFrames}
                      disabled={!genResult}
                    >
                      エクスポート
                    </button>
                  </>
                )}
              </div>

              <div className="step-nav">
                <button className="btn-nav" onClick={() => setCurrentStep(3)}>
                  &larr; 戻る
                </button>
                <button className="btn-nav" onClick={() => setCurrentStep(1)}>
                  最初から
                </button>
              </div>
            </>
          )}
        </div>

        {/* ===== Right Panel: Preview ===== */}
        <div className="panel-right">
          {/* Step 1: Placeholder */}
          {currentStep === 1 && (
            <div className="preview-area">
              <span className="placeholder">
                必須パーツ + 口パーツ（1つ以上）を読み込んでください
              </span>
            </div>
          )}

          {/* Step 2: Segmentation overlay or brush canvas */}
          {currentStep === 2 && !brushTarget && overlayPreview && (
            <div className="preview-area">
              <img src={overlayPreview} alt="セグメントオーバーレイ" />
            </div>
          )}
          {currentStep === 2 && !brushTarget && !overlayPreview && (
            <div className="preview-area">
              <span className="placeholder">
                セグメンテーションを実行してマスクを生成してください
              </span>
            </div>
          )}
          {currentStep === 2 && brushTarget && partsResult && (
            <BrushCanvas
              maskWidth={partsResult.canvas_width}
              maskHeight={partsResult.canvas_height}
              target={brushTarget}
              onBrushApplied={handleBrushApplied}
              refreshKey={brushRefreshKey}
            />
          )}

          {/* Step 3: Composite preview */}
          {currentStep === 3 && pairsResult && (
            <div className="preview-area">
              <img src={pairsResult.composite_preview} alt="合成プレビュー" />
            </div>
          )}
          {currentStep === 3 && !pairsResult && (
            <div className="preview-area">
              <span className="placeholder">
                マスクを確定してください
              </span>
            </div>
          )}

          {/* Step 4: Live composite preview */}
          {currentStep === 4 && (
            <div className="preview-area">
              {compositePreview ? (
                <img src={compositePreview} alt="合成プレビュー" />
              ) : (
                <span className="placeholder">
                  スライダーを操作してプレビュー
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="status-bar">
        {error && <span className="error-msg">{error}</span>}
        {!error && status}
      </div>
    </div>
  );
}

export default App;
