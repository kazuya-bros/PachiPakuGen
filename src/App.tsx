import { useState, useEffect, useRef } from "react";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import "./App.css";


import {
  type SlotLoadResult,
  type LayerInfo,
  type MappingPreviewResult,
  type RenderCategoryResult,
  type CreateBaseResult,
  type ProgressPayload,
  type LayerPatch,
  type PreviewPan,
  type WorkspaceStep,
  type WorkspaceMouthCornerMode,
  type SeeThroughProfile,
  type SeeThroughOptionMode,
  type SeeThroughOptions,
  type SeeThroughRuntimeStatus,
  type SeeThroughProgress,
  type SeeThroughRunResult,
  type SeeThroughLayerProbeResult,
  type ExpressionWorkspaceResult,
  type WorkspaceGeneratedPartsStatus,
  type ExtractCodexGeneratedPartsResult,
  type PreviewCodexCompositeResult,
  type GenerateCodexRifeOutputResult,
  type SaveCodexBasePartsResult,
  type AdjustCodexExtractedPartsResult,
  type LoadCodexExpressionJobResult,
  type PartAdjustment,
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
} from "./workspace/types";
import { loadMotionLabImage } from "./motionLab/render";
import { MotionTunePanel } from "./motionLab/MotionTunePanel";
import { MotionLiveView } from "./motionLab/MotionLiveView";
import type { MotionLabManifestResult, SpritalkMotionProfileResult } from "./motionLab/types";
import {
  buildWorkspacePhaseModel,
  persistenceCommandAfterWorkspaceEdit,
  restoredWorkspaceStep,
  workspaceArtifactReadiness,
} from "./workspace/phaseModel";
import "./WorkspaceFlow.css";

type Mode = "select" | "workspace" | "live";
type ThemeMode = "dark" | "light";
type WorkspaceInlineEditor = "base" | "position" | "motion" | null;
type LiveOrigin = "select" | "workspace";
type WorkspaceBaseEditorSnapshot = {
  signature: string;
  layerMapping: Record<string, string>;
  bodyPreview: string;
  enabledLayers: Record<string, boolean>;
  layerOrder: string[];
  layerPatches: LayerPatch[];
  layerOpacities: Record<string, number>;
  chestMaskDataUrl: string | null;
};
type WorkspacePartBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};
type WorkspacePositionVisual = {
  part: string;
  baseImage: HTMLImageElement;
  partImage: HTMLImageElement;
  companionImage: HTMLImageElement | null;
  companionPart: string | null;
  alphaMask: Uint8Array;
  bounds: WorkspacePartBounds;
  width: number;
  height: number;
};
type WorkspacePartAdjustmentDrafts = Record<string, PartAdjustment>;

function sameWorkspacePartAdjustment(left: PartAdjustment | undefined, right: PartAdjustment | undefined) {
  return left?.offsetX === right?.offsetX
    && left?.offsetY === right?.offsetY
    && left?.scalePercent === right?.scalePercent;
}

function cloneWorkspacePartAdjustment(adjustment?: PartAdjustment): PartAdjustment {
  const source = adjustment ?? DEFAULT_PART_ADJUSTMENT;
  return {
    offsetX: source.offsetX,
    offsetY: source.offsetY,
    scalePercent: source.scalePercent,
  };
}

function normalizeWorkspacePartAdjustment(adjustment: PartAdjustment): PartAdjustment {
  const finiteOr = (value: number, fallback: number) => Number.isFinite(value) ? Math.round(value) : fallback;
  return {
    offsetX: finiteOr(adjustment.offsetX, 0),
    offsetY: finiteOr(adjustment.offsetY, 0),
    scalePercent: Math.min(150, Math.max(50, finiteOr(adjustment.scalePercent, 100))),
  };
}

function createWorkspacePartAdjustmentDrafts(
  adjustments: Record<string, PartAdjustment> = {},
): WorkspacePartAdjustmentDrafts {
  return Object.fromEntries(WORKSPACE_ADJUST_PART_KEYS.map(part => [
    part,
    cloneWorkspacePartAdjustment(adjustments[part]),
  ]));
}

function cloneWorkspacePartAdjustmentDrafts(
  drafts: WorkspacePartAdjustmentDrafts,
): WorkspacePartAdjustmentDrafts {
  return Object.fromEntries(WORKSPACE_ADJUST_PART_KEYS.map(part => [
    part,
    cloneWorkspacePartAdjustment(drafts[part]),
  ]));
}

function workspacePartAdjustmentDraftsDiffer(
  left: WorkspacePartAdjustmentDrafts,
  right: WorkspacePartAdjustmentDrafts,
) {
  return WORKSPACE_ADJUST_PART_KEYS.some(part => !sameWorkspacePartAdjustment(left[part], right[part]));
}

function isDefaultWorkspacePartAdjustment(adjustment?: PartAdjustment) {
  return sameWorkspacePartAdjustment(adjustment, DEFAULT_PART_ADJUSTMENT);
}

function workspaceChildPath(root: string, ...segments: string[]) {
  const separator = root.includes("\\") ? "\\" : "/";
  return [root.replace(/[\\/]+$/, ""), ...segments.map(segment => segment.replace(/^[\\/]+|[\\/]+$/g, ""))]
    .join(separator);
}

function workspaceCompanionPart(part: string) {
  if (part.startsWith("eyes-")) return "mouth-closed";
  if (part.startsWith("mouth-")) return "eyes-open";
  return null;
}

function analyzeWorkspacePartImage(image: HTMLImageElement) {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("差分パーツの当たり判定を準備できませんでした");
  context.drawImage(image, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  const alphaMask = new Uint8Array(width * height);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let index = 0; index < alphaMask.length; index += 1) {
    const alpha = pixels[index * 4 + 3];
    alphaMask[index] = alpha;
    if (alpha <= 8) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (maxX < minX || maxY < minY) throw new Error("差分パーツに表示できる画素がありません");
  return {
    alphaMask,
    bounds: { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 },
    width,
    height,
  };
}

function workspacePartGeometry(
  visual: WorkspacePositionVisual,
  offsetX: number,
  offsetY: number,
  scalePercent: number,
) {
  const { scaledWidth, scaledHeight, originX, originY } = workspacePartCanvasTransform(
    visual.width,
    visual.height,
    offsetX,
    offsetY,
    scalePercent,
  );
  const scaleX = scaledWidth / visual.width;
  const scaleY = scaledHeight / visual.height;
  return {
    scaledWidth,
    scaledHeight,
    originX,
    originY,
    scaleX,
    scaleY,
    bounds: {
      x: originX + visual.bounds.x * scaleX,
      y: originY + visual.bounds.y * scaleY,
      width: visual.bounds.width * scaleX,
      height: visual.bounds.height * scaleY,
    },
  };
}

function workspacePartCanvasTransform(
  width: number,
  height: number,
  offsetX: number,
  offsetY: number,
  scalePercent: number,
) {
  const scaledWidth = Math.max(1, Math.floor((width * scalePercent) / 100));
  const scaledHeight = Math.max(1, Math.floor((height * scalePercent) / 100));
  return {
    scaledWidth,
    scaledHeight,
    originX: Math.floor((width - scaledWidth) / 2) + offsetX,
    originY: Math.floor((height - scaledHeight) / 2) + offsetY,
  };
}

function WorkspaceToolbarIcon({ name }: { name: "sun" | "moon" | "home" | "folder" }) {
  if (name === "sun") {
    return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41" /></svg>;
  }
  if (name === "moon") {
    return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20.3 15.7A8.5 8.5 0 0 1 8.3 3.7 8.5 8.5 0 1 0 20.3 15.7Z" /></svg>;
  }
  if (name === "home") {
    return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m3 11 9-8 9 8" /><path d="M5 10v10h14V10M9 20v-6h6v6" /></svg>;
  }
  return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 6.5h7l2 2h9v10.5H3z" /><path d="M3 9h18" /></svg>;
}

const WORKSPACE_MOUTH_CORNER_OPTIONS: Array<{
  value: WorkspaceMouthCornerMode;
  label: string;
  description: string;
}> = [
  { value: "source", label: "元画像に合わせる", description: "元画像の口角の向き・強さ・自然な左右差を、各口形へ引き継ぎます。" },
  { value: "up", label: "少し上げる（楽しい・嬉しい）", description: "口角だけを控えめに上げます。目・眉・頬は変えません。" },
  { value: "flat", label: "普通・ニュートラル", description: "口角を意図的に上げ下げせず、自然な普通の口元にします。" },
  { value: "down", label: "少し下げる（不満・怒り）", description: "口角だけを控えめに下げます。目・眉・頬は変えません。" },
];

function App() {
  const [mode, setMode] = useState<Mode>("select");
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "dark";
    return window.localStorage.getItem("pachipakugen-theme") === "light" ? "light" : "dark";
  });

  // Shared
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("作業を選択してください");

  const [workspaceStep, setWorkspaceStep] = useState<WorkspaceStep>(1);
  const [expressionWorkspace, setExpressionWorkspace] = useState<ExpressionWorkspaceResult | null>(null);
  const activeWorkspacePath = useRef<string | null>(null);
  const [workspaceFiles, setWorkspaceFiles] = useState<Record<string, string>>({});
  const [workspaceImagePreviews, setWorkspaceImagePreviews] = useState<Record<string, string>>({});
  const [workspaceCodexPrompt, setWorkspaceCodexPrompt] = useState("");
  const [workspaceMouthCorner, setWorkspaceMouthCorner] = useState<WorkspaceMouthCornerMode>("flat");
  const [workspaceAssetPreparationMethod, setWorkspaceAssetPreparationMethod] = useState<"codex" | "image-ai" | "manual">("codex");
  const [workspaceCodexRequestDirty, setWorkspaceCodexRequestDirty] = useState(false);
  const [workspaceGeneratedStatus, setWorkspaceGeneratedStatus] = useState<WorkspaceGeneratedPartsStatus | null>(null);
  const workspaceGeneratedPollRequestId = useRef(0);
  const [workspaceExtractResult, setWorkspaceExtractResult] = useState<ExtractCodexGeneratedPartsResult | null>(null);
  const [workspaceCompositePreview, setWorkspaceCompositePreview] = useState<PreviewCodexCompositeResult | null>(null);
  const [workspaceOverviewPreviewPart, setWorkspaceOverviewPreviewPart] = useState<string>("eyes-open");
  const [workspaceOverviewPreviewLoading, setWorkspaceOverviewPreviewLoading] = useState(false);
  const workspaceOverviewPreviewLoadingRef = useRef(false);
  const [workspaceSelectedPreviewPart, setWorkspaceSelectedPreviewPart] = useState<string>("eyes-open");
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
    sourcePerClientX?: number;
    sourcePerClientY?: number;
  } | null>(null);
  const workspacePositionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [workspacePositionVisual, setWorkspacePositionVisual] = useState<WorkspacePositionVisual | null>(null);
  const [workspacePartHovered, setWorkspacePartHovered] = useState(false);
  const [workspacePartSaving, setWorkspacePartSaving] = useState(false);
  const workspacePositionVisualLoadId = useRef(0);
  const [workspacePartDrafts, setWorkspacePartDrafts] = useState<WorkspacePartAdjustmentDrafts>(
    () => createWorkspacePartAdjustmentDrafts(),
  );
  const workspacePartDraftsRef = useRef<WorkspacePartAdjustmentDrafts>(createWorkspacePartAdjustmentDrafts());
  const workspacePartEditorBaseline = useRef<WorkspacePartAdjustmentDrafts>(createWorkspacePartAdjustmentDrafts());
  const workspacePartPersistedDuringEditor = useRef(false);
  const [workspaceRifeResult, setWorkspaceRifeResult] = useState<GenerateCodexRifeOutputResult | null>(null);
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [workspaceInlineEditor, setWorkspaceInlineEditor] = useState<WorkspaceInlineEditor>(null);
  const [workspaceEditorPreparing, setWorkspaceEditorPreparing] = useState(false);
  const [motionProfileReady, setMotionProfileReady] = useState(false);
  const [motionEditorDirty, setMotionEditorDirty] = useState(false);
  const [motionDraftSaveRequestId, setMotionDraftSaveRequestId] = useState(0);
  const [motionEditorDraftSaveBusy, setMotionEditorDraftSaveBusy] = useState(false);
  const [motionExportRequestId, setMotionExportRequestId] = useState(0);
  const [motionEditorExportReady, setMotionEditorExportReady] = useState(false);
  const [motionEditorExportBusy, setMotionEditorExportBusy] = useState(false);
  const [motionPostSaveDestination, setMotionPostSaveDestination] = useState<"overview" | "live">("overview");
  const [livePartsDir, setLivePartsDir] = useState<string | null>(null);
  const [liveOrigin, setLiveOrigin] = useState<LiveOrigin>("select");
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
  /** profile/GPU切替時に古い状態確認レスポンスがready表示を上書きしないための世代番号 */
  const seeThroughStatusRequestId = useRef(0);
  /** 明示キャンセル後に子プロセス終了エラーを通常の失敗として表示しないための印 */
  const seeThroughCancelRequested = useRef(false);
  const applyRecommendedSeeThroughProfile = (runtime: SeeThroughRuntimeStatus) => {
    if (seeThroughProfileTouched.current) return;
    if (runtime.recommendedProfile === "low-vram" || runtime.recommendedProfile === "standard") {
      setSeeThroughProfile(runtime.recommendedProfile);
    }
  };
  const requestedSeeThroughProfile = (): SeeThroughProfile | "auto" =>
    seeThroughProfileTouched.current ? seeThroughProfile : "auto";
  /** See-Through（Python環境+モデル本体、数〜十数GB）のインストール先 */
  const [seeThroughInstallLocation, setSeeThroughInstallLocation] = useState<{ path: string; isDefault: boolean } | null>(null);
  /** HuggingFaceトークン設定状態（値そのものはフロントへ返さない） */
  const [hfTokenStatus, setHfTokenStatus] = useState<{ configured: boolean } | null>(null);
  const [hfTokenInput, setHfTokenInput] = useState("");
  const [seeThroughModelDownloadLaunching, setSeeThroughModelDownloadLaunching] = useState(false);
  const [seeThroughSplitParts, setSeeThroughSplitParts] = useState(true);
  const [seeThroughOptions, setSeeThroughOptions] = useState<SeeThroughOptions>(DEFAULT_SEE_THROUGH_OPTIONS);
  const [seeThroughLayerProbe, setSeeThroughLayerProbe] = useState<SeeThroughLayerProbeResult | null>(null);
  const [seeThroughLayerProbeRunning, setSeeThroughLayerProbeRunning] = useState(false);
  /** 抽出ガチャのサムネイルをクリックした時の拡大表示対象 */
  const [seeThroughProbeZoom, setSeeThroughProbeZoom] = useState<{ name: string; thumbnail: string } | null>(null);
  useEffect(() => {
    if (!seeThroughProbeZoom) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSeeThroughProbeZoom(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [seeThroughProbeZoom]);
  const [workspacePartOffsetX, setWorkspacePartOffsetX] = useState(0);
  const [workspacePartOffsetY, setWorkspacePartOffsetY] = useState(0);
  const [workspacePartScale, setWorkspacePartScale] = useState(100);
  /** Step5で現在選択している目・口パーツ */
  const [workspaceAdjustTarget, setWorkspaceAdjustTarget] = useState("eyes-open");
  /** Step5プレビュー上のドラッグでパーツを動かすモード */
  const [workspacePartDragMode, setWorkspacePartDragMode] = useState(true);
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

  // STEP2: 表情素材の自動確認（作成ガイド出力後〜揃うまで5秒間隔でポーリング）
  useEffect(() => {
    if (mode !== "workspace" || workspaceStep !== 2 || workspaceBusy) return;
    if (!expressionWorkspace || !workspaceGeneratedStatus || workspaceGeneratedStatus.ready) return;
    const workPath = expressionWorkspace.workPath;
    const pollRequestId = ++workspaceGeneratedPollRequestId.current;
    let checking = false;
    const checkGeneratedParts = async () => {
      if (checking) return;
      checking = true;
      try {
        const generated = await invoke<WorkspaceGeneratedPartsStatus>("inspect_workspace_generated_parts", { workPath });
        if (workspaceGeneratedPollRequestId.current !== pollRequestId || activeWorkspacePath.current !== workPath) return;
        setWorkspaceGeneratedStatus(generated);
        if (generated.downstreamStale) {
          const synced = await reloadWorkspaceAfterMutationFailure(workPath);
          if (workspaceGeneratedPollRequestId.current !== pollRequestId || activeWorkspacePath.current !== workPath) return;
          if (!synced) {
            setError("表情素材の変更を検知しましたが、作業状態を再読込できませんでした。作業フォルダを開き直してください");
            return;
          }
          setStatus("表情素材の変更を検知しました。STEP3から再処理してください");
          showToast("表情素材の変更を検知しました");
          return;
        }
        if (generated.ready) {
          showToast("表情素材が揃いました");
          setStatus("表情素材が揃いました。STEP3へ進めます");
        }
      } catch {
        // 依頼作成前のフォルダ状態では失敗してよい
      } finally {
        checking = false;
      }
    };
    const timer = window.setInterval(() => void checkGeneratedParts(), 5000);
    return () => {
      workspaceGeneratedPollRequestId.current += 1;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, workspaceStep, workspaceBusy, expressionWorkspace, workspaceGeneratedStatus]);

  function setWorkspacePartDraft(part: string, adjustment: PartAdjustment, syncFields = true) {
    const normalized = normalizeWorkspacePartAdjustment(adjustment);
    const nextDrafts = {
      ...workspacePartDraftsRef.current,
      [part]: normalized,
    };
    workspacePartDraftsRef.current = nextDrafts;
    setWorkspacePartDrafts(nextDrafts);
    if (syncFields && part === workspaceAdjustTarget) {
      setWorkspacePartOffsetX(normalized.offsetX);
      setWorkspacePartOffsetY(normalized.offsetY);
      setWorkspacePartScale(normalized.scalePercent);
    }
  }

  function updateSelectedWorkspacePartDraft(patch: Partial<PartAdjustment>) {
    const current = workspacePartDraftsRef.current[workspaceAdjustTarget] ?? DEFAULT_PART_ADJUSTMENT;
    setWorkspacePartDraft(workspaceAdjustTarget, { ...current, ...patch });
  }

  /** STEP5: 選択したパーツの下書き値をフィールドへ投影する。パーツ切替では保存しない。 */
  function loadWorkspacePartAdjustmentFields(target: string) {
    const values = workspacePartDraftsRef.current[target] ?? DEFAULT_PART_ADJUSTMENT;
    setWorkspacePartOffsetX(values.offsetX);
    setWorkspacePartOffsetY(values.offsetY);
    setWorkspacePartScale(values.scalePercent);
  }

  function selectWorkspacePositionPart(part: string) {
    if (!WORKSPACE_ADJUST_PART_KEYS.includes(part)) return;
    if (part === workspaceAdjustTarget && part === workspaceSelectedPreviewPart) return;
    setWorkspaceSelectedPreviewPart(part);
    setWorkspaceAdjustTarget(part);
    setWorkspacePartDragMode(true);
    loadWorkspacePartAdjustmentFields(part);
  }

  // STEP5: 矢印キーは選択中パーツの下書きだけを±1px（Shiftで±10px）動かす。
  useEffect(() => {
    if (mode !== "workspace"
      || workspaceStep !== 5
      || workspaceInlineEditor !== "position"
      || !workspacePartDragMode
      || workspaceBusy) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const delta = event.shiftKey ? 10 : 1;
      const current = workspacePartDraftsRef.current[workspaceAdjustTarget] ?? DEFAULT_PART_ADJUSTMENT;
      if (event.key === "ArrowLeft") setWorkspacePartDraft(workspaceAdjustTarget, { ...current, offsetX: current.offsetX - delta });
      else if (event.key === "ArrowRight") setWorkspacePartDraft(workspaceAdjustTarget, { ...current, offsetX: current.offsetX + delta });
      else if (event.key === "ArrowUp") setWorkspacePartDraft(workspaceAdjustTarget, { ...current, offsetY: current.offsetY - delta });
      else if (event.key === "ArrowDown") setWorkspacePartDraft(workspaceAdjustTarget, { ...current, offsetY: current.offsetY + delta });
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, workspaceStep, workspaceInlineEditor, workspacePartDragMode, workspaceAdjustTarget, workspaceBusy]);

  useEffect(() => {
    if (workspaceInlineEditor !== "position" || WORKSPACE_ADJUST_PART_KEYS.includes(workspaceAdjustTarget)) return;
    const available = workspaceCompositePreview?.previews.map(item => item.part) ?? [];
    const fallback = available.includes("eyes-open")
      ? "eyes-open"
      : available.find(part => WORKSPACE_ADJUST_PART_KEYS.includes(part));
    if (!fallback) return;
    setWorkspaceSelectedPreviewPart(fallback);
    setWorkspaceAdjustTarget(fallback);
    setWorkspacePartDragMode(true);
    loadWorkspacePartAdjustmentFields(fallback);
  }, [workspaceInlineEditor, workspaceAdjustTarget, workspaceCompositePreview, workspaceExtractResult]);

  useEffect(() => {
    const loadId = ++workspacePositionVisualLoadId.current;
    const basePreview = workspaceCompositePreview?.basePreview;
    const extractedPath = workspaceExtractResult?.extractedPartsPath;
    const editable = mode === "workspace"
      && workspaceStep === 5
      && workspaceInlineEditor === "position"
      && workspaceAdjustTarget !== "all"
      && WORKSPACE_ADJUST_PART_KEYS.includes(workspaceAdjustTarget);
    if (!editable || !basePreview || !extractedPath) {
      setWorkspacePositionVisual(null);
      setWorkspacePartHovered(false);
      return;
    }

    void (async () => {
      try {
        const loadOriginalPartPreview = async (part: string) => {
          const originalPath = workspaceChildPath(
            extractedPath,
            "original_extracted_parts",
            `${part}.png`,
          );
          try {
            return await invoke<string>("load_expression_source_preview", { path: originalPath });
          } catch {
            // 初回調整前はoriginal_extracted_partsが無い。現在PNGはまだ原本と同じなので利用できる。
            return invoke<string>("load_expression_source_preview", {
              path: workspaceChildPath(extractedPath, `${part}.png`),
            });
          }
        };
        const partPreview = await loadOriginalPartPreview(workspaceAdjustTarget);
        const companionPart = workspaceCompanionPart(workspaceAdjustTarget);
        const companionPreview = companionPart
          ? await loadOriginalPartPreview(companionPart).catch(() => null)
          : null;
        const [baseImage, partImage, companionImage] = await Promise.all([
          loadMotionLabImage(basePreview),
          loadMotionLabImage(partPreview),
          companionPreview ? loadMotionLabImage(companionPreview) : Promise.resolve(null),
        ]);
        const analyzed = analyzeWorkspacePartImage(partImage);
        if (loadId !== workspacePositionVisualLoadId.current) return;
        setWorkspacePositionVisual({
          part: workspaceAdjustTarget,
          baseImage,
          partImage,
          companionImage,
          companionPart,
          ...analyzed,
        });
        setWorkspacePartHovered(false);
      } catch {
        if (loadId === workspacePositionVisualLoadId.current) {
          // 合成済みプレビューへフォールバックする。位置調整自体は数値・矢印で継続できる。
          setWorkspacePositionVisual(null);
          setWorkspacePartHovered(false);
        }
      }
    })();
  }, [
    mode,
    workspaceStep,
    workspaceInlineEditor,
    workspaceAdjustTarget,
    workspaceExtractResult?.extractedPartsPath,
    workspaceCompositePreview?.basePreview,
  ]);

  useEffect(() => {
    const canvas = workspacePositionCanvasRef.current;
    const visual = workspacePositionVisual;
    if (!canvas || !visual || visual.part !== workspaceAdjustTarget) return;
    if (canvas.width !== visual.width) canvas.width = visual.width;
    if (canvas.height !== visual.height) canvas.height = visual.height;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, visual.width, visual.height);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(visual.baseImage, 0, 0, visual.width, visual.height);
    const geometry = workspacePartGeometry(
      visual,
      workspacePartOffsetX,
      workspacePartOffsetY,
      workspacePartScale,
    );
    const drawSelectedPart = () => context.drawImage(
      visual.partImage,
      geometry.originX,
      geometry.originY,
      geometry.scaledWidth,
      geometry.scaledHeight,
    );
    const drawCompanion = () => {
      if (!visual.companionImage || !visual.companionPart) return;
      const adjustment = workspacePartDrafts[visual.companionPart] ?? DEFAULT_PART_ADJUSTMENT;
      const companionGeometry = workspacePartCanvasTransform(
        visual.width,
        visual.height,
        adjustment.offsetX,
        adjustment.offsetY,
        adjustment.scalePercent,
      );
      context.drawImage(
        visual.companionImage,
        companionGeometry.originX,
        companionGeometry.originY,
        companionGeometry.scaledWidth,
        companionGeometry.scaledHeight,
      );
    };
    // 実際の合成順と同じく目→口。相方は表示のみで、当たり判定は選択中パーツに限定する。
    if (visual.part.startsWith("eyes-")) {
      drawSelectedPart();
      drawCompanion();
    } else {
      drawCompanion();
      drawSelectedPart();
    }

    // 小さい閉じ口でも見失わないよう、実画素の周囲に画面上最低32pxの操作枠を表示する。
    if (workspacePartDragMode) {
      const canvasRect = canvas.getBoundingClientRect();
      const displayScale = Math.max(0.001, canvasRect.width / visual.width);
      const padding = 8 / displayScale;
      const minimumSize = 32 / displayScale;
      const hitWidth = Math.max(geometry.bounds.width + padding * 2, minimumSize);
      const hitHeight = Math.max(geometry.bounds.height + padding * 2, minimumSize);
      const hitX = geometry.bounds.x + geometry.bounds.width / 2 - hitWidth / 2;
      const hitY = geometry.bounds.y + geometry.bounds.height / 2 - hitHeight / 2;
      const draggingPart = workspacePreviewDragging && workspacePreviewDrag.current?.mode === "part";
      context.save();
      context.lineWidth = (workspacePartHovered || draggingPart ? 2.5 : 1.5) / displayScale;
      context.strokeStyle = workspacePartHovered || draggingPart ? "#1677ff" : "rgba(22, 119, 255, 0.72)";
      context.fillStyle = workspacePartHovered || draggingPart
        ? "rgba(22, 119, 255, 0.10)"
        : "rgba(22, 119, 255, 0.04)";
      context.setLineDash([6 / displayScale, 4 / displayScale]);
      context.fillRect(hitX, hitY, hitWidth, hitHeight);
      context.strokeRect(hitX, hitY, hitWidth, hitHeight);
      context.restore();
    }
  }, [
    workspacePositionVisual,
    workspaceAdjustTarget,
    workspacePartOffsetX,
    workspacePartOffsetY,
    workspacePartScale,
    workspacePartDrafts,
    workspacePartDragMode,
    workspacePartHovered,
    workspacePreviewDragging,
    workspacePreviewZoom,
    workspacePreviewPan,
  ]);

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
      setWorkspaceCodexRequestDirty(true);
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

  // === STEP 4: 素体編集 ===
  const [mappingPreview, setMappingPreview] = useState<MappingPreviewResult | null>(null);
  const [bodyPreview, setBodyPreview] = useState("");
  const [enabledLayers, setEnabledLayers] = useState<Record<string, boolean>>({});
  const [layerOrder, setLayerOrder] = useState<string[]>([]);
  const [layerPatches, setLayerPatches] = useState<LayerPatch[]>([]);
  const [layerOpacities, setLayerOpacities] = useState<Record<string, number>>({});
  const [selectedBodyLayer, setSelectedBodyLayer] = useState<string>("");
  const [overlapHighlightEnabled, setOverlapHighlightEnabled] = useState(false);
  const [patchDraftSource, setPatchDraftSource] = useState("");
  const [patchTool, setPatchTool] = useState<"paint" | "erase">("paint");
  const [patchBrushSize, setPatchBrushSize] = useState(24);
  // 0=硬いエッジ（従来）、1=最大にぼかしたエッジ（境界のギザギザを緩和）
  const [patchBrushSoftness, setPatchBrushSoftness] = useState(0.5);
  // 胸部範囲ガイド: Motion Labの局所ワープ位置と旧形式chest.pngの互換出力に使う
  const [chestMaskDataUrl, setChestMaskDataUrl] = useState<string | null>(null);
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number; size: number; visible: boolean }>({ x: 0, y: 0, size: 0, visible: false });

  const [frameCount, setFrameCount] = useState(8);
  const [progress, setProgress] = useState({ current: 0, total: 0, pair_name: "" });

  // Zoom & pan for the unified STEP 4 editor
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
  const baseEditorBaselineRef = useRef<WorkspaceBaseEditorSnapshot | null>(null);
  const opacityRenderTimer = useRef<number | null>(null);
  // 表示切替の連打をまとめて1回の再合成にするためのデバウンス
  const toggleRenderTimer = useRef<number | null>(null);
  const enabledLayersRef = useRef<Record<string, boolean>>({});
  const dragState = useRef<{ idx: number; startY: number; currentIdx: number } | null>(null);

  useEffect(() => {
    if (workspaceInlineEditor === "base" && !workspaceEditorPreparing && baseEditorBaselineRef.current === null) {
      baseEditorBaselineRef.current = createWorkspaceBaseEditorSnapshot();
    } else if (workspaceInlineEditor !== "base") {
      baseEditorBaselineRef.current = null;
    }
    // 編集開始時の一度だけ基準値を記録し、編集操作では更新しない。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceInlineEditor, workspaceEditorPreparing]);
  const [draggedIdx, setDraggedIdx] = useState<number | null>(null);
  const layerOrderRef = useRef(layerOrder);
  layerOrderRef.current = layerOrder;

  // 旧版では ears* と headwear を同時に獣耳として扱っていたため、髪飾りまで
  // sway_ear* に保存されている場合がある。明示的な ears* が存在する素材では
  // headwear は髪飾りとして扱い、過去の誤分類だけを安全に hair へ戻す。
  useEffect(() => {
    if (!layerOrder.some(name => /^ears([-_][lr])?$/i.test(name))) return;
    const misclassifiedHeadwear = layerOrder.filter(name => (
      /^headwear([-_][lr])?$/i.test(name)
      && (layerMapping[name] ?? "").startsWith("sway_")
    ));
    if (misclassifiedHeadwear.length === 0) return;
    setLayerMapping(prev => {
      const next = { ...prev };
      for (const name of misclassifiedHeadwear) {
        if ((next[name] ?? "").startsWith("sway_")) next[name] = "hair";
      }
      return next;
    });
  }, [layerOrder, layerMapping]);

  useEffect(() => {
    if (!isTauri()) return;
    const unlisten = listen<ProgressPayload>("generation-progress", (event) => {
      setProgress({ current: event.payload.current, total: event.payload.total, pair_name: event.payload.pair_name });
    });
    const unlistenSeeThrough = listen<SeeThroughProgress>("see-through-progress", (event) => {
      setSeeThroughProgress(event.payload);
      if (event.payload.stage === "model-download-complete" || event.payload.stage === "model-download-failed") {
        setSeeThroughRuntime(current => current ? { ...current, modelDownloadBusy: false } : current);
      }
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

  function onDragPointerDown(e: React.PointerEvent, idx: number) {
    e.preventDefault(); (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragState.current = { idx, startY: e.clientY, currentIdx: idx };
    setDraggedIdx(idx);
  }
  function onDragPointerMove(e: React.PointerEvent) {
    if (!dragState.current) return;
    const newIdx = Math.max(0, Math.min(layerOrderRef.current.length - 1,
      dragState.current.idx + Math.round((e.clientY - dragState.current.startY) / 50)));
    if (newIdx !== dragState.current.currentIdx) {
      const newOrder = [...layerOrderRef.current];
      const [item] = newOrder.splice(dragState.current.currentIdx, 1);
      newOrder.splice(newIdx, 0, item);
      setLayerOrder(newOrder);
      dragState.current.currentIdx = newIdx; setDraggedIdx(newIdx);
    }
  }
  async function onDragPointerUp() {
    if (!dragState.current) return;
    dragState.current = null;
    setDraggedIdx(null);
    await renderBody(layerOrderRef.current, enabledLayers);
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
    workspacePreviewDrag.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      startPan: workspacePreviewPan,
      mode: "pan",
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
      updateSelectedWorkspacePartDraft({
        offsetX: Math.round(drag.startOffsetX + (e.clientX - drag.startX) / scale),
        offsetY: Math.round(drag.startOffsetY + (e.clientY - drag.startY) / scale),
      });
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
    if (drag.mode === "part") {
      const scale = Math.max(0.01, workspacePreviewZoom);
      updateSelectedWorkspacePartDraft({
        offsetX: Math.round(drag.startOffsetX + (e.clientX - drag.startX) / scale),
        offsetY: Math.round(drag.startOffsetY + (e.clientY - drag.startY) / scale),
      });
    }
  }
  function handleWorkspacePreviewLostPointerCapture(e: React.PointerEvent<HTMLDivElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    workspacePreviewDrag.current = null;
    setWorkspacePreviewDragging(false);
  }

  function workspacePositionHitTest(clientX: number, clientY: number) {
    const canvas = workspacePositionCanvasRef.current;
    const visual = workspacePositionVisual;
    if (!canvas || !visual || visual.part !== workspaceAdjustTarget || !workspacePartDragMode) {
      return { hit: false, sourcePerClientX: 1, sourcePerClientY: 1 };
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return { hit: false, sourcePerClientX: 1, sourcePerClientY: 1 };
    }
    const canvasX = (clientX - rect.left) * visual.width / rect.width;
    const canvasY = (clientY - rect.top) * visual.height / rect.height;
    const geometry = workspacePartGeometry(
      visual,
      workspacePartOffsetX,
      workspacePartOffsetY,
      workspacePartScale,
    );
    const sourceX = (canvasX - geometry.originX) / geometry.scaleX;
    const sourceY = (canvasY - geometry.originY) / geometry.scaleY;
    // アルファ形状を画面上12pxだけ膨張。35%表示の閉じ口も掴めるが、顔や胴体は反応しない。
    const radiusX = Math.max(1, (12 * visual.width / rect.width) / geometry.scaleX);
    const radiusY = Math.max(1, (12 * visual.height / rect.height) / geometry.scaleY);
    const bounds = visual.bounds;
    if (sourceX < bounds.x - radiusX
      || sourceX > bounds.x + bounds.width + radiusX
      || sourceY < bounds.y - radiusY
      || sourceY > bounds.y + bounds.height + radiusY) {
      return {
        hit: false,
        sourcePerClientX: visual.width / rect.width,
        sourcePerClientY: visual.height / rect.height,
      };
    }

    const minY = Math.max(0, Math.floor(sourceY - radiusY));
    const maxY = Math.min(visual.height - 1, Math.ceil(sourceY + radiusY));
    let hit = false;
    for (let y = minY; y <= maxY && !hit; y += 1) {
      const normalizedY = (y - sourceY) / radiusY;
      const horizontalRadius = radiusX * Math.sqrt(Math.max(0, 1 - normalizedY * normalizedY));
      const minX = Math.max(0, Math.floor(sourceX - horizontalRadius));
      const maxX = Math.min(visual.width - 1, Math.ceil(sourceX + horizontalRadius));
      for (let x = minX; x <= maxX; x += 1) {
        if (visual.alphaMask[y * visual.width + x] > 8) {
          hit = true;
          break;
        }
      }
    }
    return {
      hit,
      sourcePerClientX: visual.width / rect.width,
      sourcePerClientY: visual.height / rect.height,
    };
  }

  function handleWorkspacePositionPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (e.button !== 0) return;
    const hit = workspacePositionHitTest(e.clientX, e.clientY);
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    workspacePreviewDrag.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      startPan: workspacePreviewPan,
      mode: hit.hit ? "part" : "pan",
      startOffsetX: workspacePartOffsetX,
      startOffsetY: workspacePartOffsetY,
      sourcePerClientX: hit.sourcePerClientX,
      sourcePerClientY: hit.sourcePerClientY,
    };
    setWorkspacePartHovered(hit.hit);
    setWorkspacePreviewDragging(true);
  }

  function handleWorkspacePositionPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) {
      const hit = workspacePositionHitTest(e.clientX, e.clientY).hit;
      setWorkspacePartHovered(previous => previous === hit ? previous : hit);
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    if (drag.mode === "part") {
      updateSelectedWorkspacePartDraft({
        offsetX: Math.round(
          drag.startOffsetX + (e.clientX - drag.startX) * (drag.sourcePerClientX ?? 1),
        ),
        offsetY: Math.round(
          drag.startOffsetY + (e.clientY - drag.startY) * (drag.sourcePerClientY ?? 1),
        ),
      });
      return;
    }
    setWorkspacePreviewPan({
      x: drag.startPan.x + e.clientX - drag.startX,
      y: drag.startPan.y + e.clientY - drag.startY,
    });
  }

  function handleWorkspacePositionPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.stopPropagation();
    workspacePreviewDrag.current = null;
    setWorkspacePreviewDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (drag.mode === "part") {
      const finalOffset = {
        x: Math.round(drag.startOffsetX + (e.clientX - drag.startX) * (drag.sourcePerClientX ?? 1)),
        y: Math.round(drag.startOffsetY + (e.clientY - drag.startY) * (drag.sourcePerClientY ?? 1)),
      };
      updateSelectedWorkspacePartDraft({ offsetX: finalOffset.x, offsetY: finalOffset.y });
    }
  }

  function handleWorkspacePositionLostPointerCapture(e: React.PointerEvent<HTMLCanvasElement>) {
    const drag = workspacePreviewDrag.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    workspacePreviewDrag.current = null;
    setWorkspacePreviewDragging(false);
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

  // === STEP 4: 素体を保存 ===
  async function handleCreateBase() {
    setError("");
    const workspacePath = expressionWorkspace?.workPath;
    if (!workspacePath) {
      setError("作業フォルダが読み込まれていません");
      return;
    }

    setLoading(true);
    setWorkspaceBusy(true);
    setStatus("素体を作成しています...");
    try {
      // 素体合成は必ず source の分解結果（source.psd）を対象にする。
      // 直前にどの画像を分解したかというグローバル状態に依存させない
      // （eyes-closed分解が残って素体の目が閉じる問題の根治）
      await invoke("load_codex_source_see_through", { jobPath: workspacePath });
      const activeOrder = layerOrder.filter(name => enabledLayers[name] !== false);
      await invoke<CreateBaseResult>("create_base", {
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
      await invoke<SaveCodexBasePartsResult>("save_codex_base_parts", { jobPath: workspacePath });
      // 素体画像とlayer-order.jsonが更新された時点で、同じ出力フォルダを参照する
      // 既存のRIFE/Motion Lab結果は古い。再生成されるまで下流を無効化する。
      setWorkspaceRifeResult(null);
      setMotionProfileReady(false);
      await refreshWorkspaceCompositePreview();
      await setWorkspaceStepAfterEdit(5);
      setWorkspaceInlineEditor("position");
      setStatus("素体を作業フォルダに保存しました。差分位置調整へ進めます。");
    } catch (e) {
      await reloadWorkspaceAfterMutationFailure(workspacePath);
      setError(String(e));
    } finally {
      setLoading(false);
      setWorkspaceBusy(false);
    }
  }
  function resetWorkspaceBaseEditorState() {
    setLoadResult(null);
    setLayerMapping({});
    setMappingPreview(null);
    setBodyPreview("");
    setEnabledLayers({});
    enabledLayersRef.current = {};
    setLayerOrder([]);
    layerOrderRef.current = [];
    setLayerPatches([]);
    layerPatchesRef.current = [];
    setLayerOpacities({});
    layerOpacitiesRef.current = {};
    setSelectedBodyLayer("");
    setOverlapHighlightEnabled(false);
    setChestMaskDataUrl(null);
    baseEditorBaselineRef.current = null;
  }

  async function loadExpressionWorkspaceAtPath(workPath: string, kind: "new" | "resume") {
    activeWorkspacePath.current = workPath;
    setWorkspaceBusy(true);
    try {
      const workspace = await invoke<ExpressionWorkspaceResult>(
        kind === "new" ? "create_expression_workspace" : "load_expression_workspace",
        { workPath },
      );
      if (activeWorkspacePath.current !== workPath) return null;
      setExpressionWorkspace(workspace);
      const nextFiles = {
        ...(workspace.project.sourceImagePath ? { source: workspace.project.sourceImagePath } : {}),
        ...(workspace.project.referenceImagePath ? { reference: workspace.project.referenceImagePath } : {}),
      };
      setWorkspaceFiles(nextFiles);
      setWorkspaceCodexPrompt(workspace.project.codexPrompt ?? "");
      setWorkspaceMouthCorner(workspace.project.mouthCorner ?? "flat");
      setWorkspaceCodexRequestDirty(false);
      await loadWorkspaceImagePreviews(nextFiles, workPath);
      if (activeWorkspacePath.current !== workPath) return null;
      setWorkspaceGeneratedStatus(null);
      setWorkspaceExtractResult(null);
      setWorkspaceCompositePreview(null);
      setWorkspaceOverviewPreviewPart("eyes-open");
      setWorkspaceOverviewPreviewLoading(false);
      workspaceOverviewPreviewLoadingRef.current = false;
      setWorkspaceSelectedPreviewPart("eyes-open");
      setWorkspaceAdjustTarget("eyes-open");
      setWorkspacePartDragMode(true);
      setWorkspacePartOffsetX(0);
      setWorkspacePartOffsetY(0);
      setWorkspacePartScale(100);
      const emptyPartDrafts = createWorkspacePartAdjustmentDrafts();
      workspacePartDraftsRef.current = emptyPartDrafts;
      workspacePartEditorBaseline.current = cloneWorkspacePartAdjustmentDrafts(emptyPartDrafts);
      setWorkspacePartDrafts(emptyPartDrafts);
      setWorkspacePositionVisual(null);
      setWorkspacePartHovered(false);
      setWorkspacePartSaving(false);
      workspacePartPersistedDuringEditor.current = false;
      setWorkspaceRifeResult(null);
      setWorkspaceInlineEditor(null);
      setWorkspaceEditorPreparing(false);
      setMotionProfileReady(false);
      setMotionEditorDirty(false);
      setMotionDraftSaveRequestId(0);
      setMotionEditorDraftSaveBusy(false);
      setMotionExportRequestId(0);
      setMotionPostSaveDestination("overview");
      resetWorkspaceBaseEditorState();
      setWorkspaceStep(Math.min(Math.max(workspace.project.currentStep || 1, 1), 7) as WorkspaceStep);
      setMode("workspace");
      setStatus(`作業フォルダ: ${workspace.workPath}`);
      if (kind === "resume") {
        await restoreWorkspaceProgress(workspace);
      }
      if (activeWorkspacePath.current !== workPath) return null;
      if (workspace.project.currentStep <= 3) await refreshWorkspaceSeeThroughStatus();
      return workspace;
    } catch (cause) {
      if (activeWorkspacePath.current !== workPath) return null;
      const message = String(cause);
      if (kind === "resume" && message.includes("project.json が見つかりません")) {
        setError("選択したフォルダはPachiPakuGenの作業フォルダではありません（project.json がありません）。「はじめから」で作成した作業フォルダを選んでください");
      } else {
        setError(message);
      }
      return null;
    } finally {
      if (activeWorkspacePath.current === workPath) setWorkspaceBusy(false);
    }
  }

  async function startExpressionWorkspace(kind: "new" | "resume") {
    if (workspaceBusy) return;
    setError("");
    setWorkspaceBusy(true);
    try {
      const selected = await open({
        multiple: false,
        directory: true,
        title: kind === "new" ? "作業フォルダを選択" : "既存の作業フォルダを選択",
      });
      const workPath = typeof selected === "string" ? selected : null;
      if (!workPath) return;
      await loadExpressionWorkspaceAtPath(workPath, kind);
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function startLiveDisplayFromMenu() {
    if (workspaceBusy) return;
    setError("");
    setWorkspaceBusy(true);
    try {
      const selected = await open({
        multiple: false,
        directory: true,
        title: "ライブ表示に使う作業フォルダを選択",
      });
      const workPath = typeof selected === "string" ? selected : null;
      if (!workPath) return;
      const workspace = await invoke<ExpressionWorkspaceResult>("load_expression_workspace", { workPath });
      if (workspace.project.currentStep < 7) {
        throw new Error("この作業フォルダは再編集後のフレーム生成が未完了です。STEP6でRIFE補完を再実行してください。");
      }
      await invoke("load_motion_lab_parts", { dir: workspace.spritalkPartsPath });
      setLiveOrigin("select");
      setLivePartsDir(workspace.spritalkPartsPath);
      setMode("live");
      setStatus(`ライブ表示: ${workspace.workPath}`);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message.includes("body.png")
        ? "この作業フォルダにはライブ表示できる素材がありません。先にSTEP6まで完了してください。"
        : message);
    } finally {
      setWorkspaceBusy(false);
    }
  }

  function openLiveDisplayFromWorkspace() {
    if (!expressionWorkspace) return;
    const partsDir = workspaceRifeResult?.outputPath || expressionWorkspace.spritalkPartsPath;
    setLiveOrigin("workspace");
    setLivePartsDir(partsDir);
    setWorkspaceInlineEditor(null);
    setMode("live");
    setStatus("マイクを選択してライブ表示を開始できます");
  }

  function requestLiveDisplayFromMotionEditor() {
    if (motionEditorDraftSaveBusy || motionEditorExportBusy) return;
    if (motionEditorDirty) {
      setError("");
      setMotionPostSaveDestination("live");
      setMotionEditorDraftSaveBusy(true);
      setStatus("変更を保存してライブ表示を準備しています...");
      setMotionDraftSaveRequestId(previous => previous + 1);
      return;
    }
    openLiveDisplayFromWorkspace();
  }

  function returnFromLiveDisplay() {
    setLivePartsDir(null);
    if (liveOrigin === "workspace" && expressionWorkspace) {
      setMode("workspace");
      setWorkspaceStep(7);
      setStatus("モーション仕上げ");
      return;
    }
    setMode("select");
    setStatus("モードを選択してください");
  }

  async function persistWorkspaceStep(step: WorkspaceStep, workspaceOverride?: ExpressionWorkspaceResult) {
    const workspace = workspaceOverride ?? expressionWorkspace;
    if (!workspace) throw new Error("作業フォルダが読み込まれていません");
    const updated = await invoke<ExpressionWorkspaceResult>("update_expression_workspace_step", {
      workPath: workspace.workPath,
      currentStep: step,
    });
    if (activeWorkspacePath.current === workspace.workPath) {
      setExpressionWorkspace(updated);
    }
  }

  async function setWorkspaceStepAndPersist(step: WorkspaceStep) {
    await persistWorkspaceStep(step);
    setWorkspaceStep(step);
  }

  // STEP4/5で実際に保存した時だけ使う明示的な編集チェックポイント更新。
  // 通常の工程更新は前進専用のままにし、完了済み案件の再編集時だけ下流成果物を
  // 無効扱いにするためproject.jsonを5/6へ巻き戻す。
  async function setWorkspaceStepAfterEdit(step: WorkspaceStep) {
    const workspace = expressionWorkspace;
    if (!workspace) throw new Error("作業フォルダが読み込まれていません");
    const command = persistenceCommandAfterWorkspaceEdit(workspace.project.currentStep, step);
    const updated = await invoke<ExpressionWorkspaceResult>(command, {
      workPath: workspace.workPath,
      currentStep: step,
    });
    if (activeWorkspacePath.current === workspace.workPath) {
      setExpressionWorkspace(updated);
      setWorkspaceStep(step);
    }
  }

  /**
   * 画像を書き換えるバックエンド処理は、最初の書き込み前にproject.jsonを安全な
   * checkpointへ戻す。途中で失敗した場合も同じ画面に古い下流成果物を残さないよう、
   * ディスク上のcheckpointを正本として作業状態を読み直す。
   */
  async function reloadWorkspaceAfterMutationFailure(workPath: string): Promise<boolean> {
    try {
      const workspace = await invoke<ExpressionWorkspaceResult>("load_expression_workspace", { workPath });
      if (activeWorkspacePath.current !== workPath) return false;
      const checkpoint = Math.min(Math.max(workspace.project.currentStep || 1, 1), 7) as WorkspaceStep;
      setExpressionWorkspace(workspace);
      setWorkspaceStep(checkpoint);
      if (checkpoint <= 2) setWorkspaceGeneratedStatus(null);
      if (checkpoint < 4) {
        setWorkspaceExtractResult(null);
        resetWorkspaceBaseEditorState();
      }
      if (checkpoint < 5) setWorkspaceCompositePreview(null);
      if (checkpoint < 7) {
        setWorkspaceRifeResult(null);
        setMotionProfileReady(false);
      }
      await restoreWorkspaceProgress(workspace);
      return activeWorkspacePath.current === workPath;
    } catch {
      // 元の処理エラーを優先する。再同期に失敗してもエラー内容は上書きしない。
      return false;
    }
  }

  async function refreshMotionOutputStatus(
    workspace: ExpressionWorkspaceResult,
    rifeOutput?: GenerateCodexRifeOutputResult | null,
  ) {
    const sourceDir = rifeOutput?.outputPath || workspace.spritalkPartsPath;
    if (!rifeOutput && !workspaceRifeResult) {
      if (activeWorkspacePath.current === workspace.workPath) {
        setMotionProfileReady(false);
      }
      return;
    }
    const [manifestResult, profileResult] = await Promise.allSettled([
      invoke<MotionLabManifestResult>("load_motion_lab_manifest", { sourceDir }),
      invoke<SpritalkMotionProfileResult>("load_spritalk_motion_profile", { sourceDir }),
    ]);
    if (activeWorkspacePath.current !== workspace.workPath) return;
    const fingerprints = manifestResult.status === "fulfilled"
      ? manifestResult.value.manifest.contentFingerprints
      : undefined;
    const profileFresh = profileResult.status === "fulfilled"
      && !!fingerprints?.spritalk
      && profileResult.value.profile.contentFingerprint === fingerprints.spritalk;
    setMotionProfileReady(profileFresh);
  }

  async function restoreWorkspaceProgress(workspace: ExpressionWorkspaceResult) {
    if (activeWorkspacePath.current !== workspace.workPath) return;
    let restoredStep = Math.min(Math.max(workspace.project.currentStep || 1, 1), 7) as WorkspaceStep;
    let workspaceGeneratedReady = false;
    try {
      const generated = await invoke<WorkspaceGeneratedPartsStatus>("inspect_workspace_generated_parts", {
        workPath: workspace.workPath,
      });
      if (activeWorkspacePath.current !== workspace.workPath) return;
      setWorkspaceGeneratedStatus(generated);
      workspaceGeneratedReady = generated.ready;
      if (generated.downstreamStale) {
        setWorkspaceExtractResult(null);
        setWorkspaceCompositePreview(null);
        setWorkspaceRifeResult(null);
        setMotionProfileReady(false);
        setWorkspaceStep(3);
        try {
          const updated = await invoke<ExpressionWorkspaceResult>("load_expression_workspace", {
            workPath: workspace.workPath,
          });
          if (activeWorkspacePath.current !== workspace.workPath) return;
          setExpressionWorkspace(updated);
          await restoreWorkspaceProgress(updated);
        } catch (cause) {
          setError(`表情素材の変更を検知しましたが、作業状態を再読込できませんでした: ${String(cause)}`);
        }
        return;
      }
    } catch {
      // Step 1 workspaces may not have a source image or request files yet.
    }

    try {
      const loaded = await invoke<LoadCodexExpressionJobResult>("load_codex_expression_job", {
        jobPath: workspace.workPath,
      });
      if (activeWorkspacePath.current !== workspace.workPath) return;
      // project.jsonを正本にする。再編集前の物理ファイルが残っていても、対応する
      // checkpointへ再到達するまではReact stateへ復活させない。
      const validExtractedParts = workspace.project.currentStep >= 4 ? loaded.extractedParts : null;
      const validRifeOutput = workspace.project.currentStep >= 7 ? loaded.rifeOutput : null;
      setWorkspaceExtractResult(validExtractedParts);
      setWorkspaceRifeResult(validRifeOutput);
      if (validRifeOutput) setFrameCount(validRifeOutput.frameCount);
      // プレビュー再構築は保存済みbase_partsからのみ行う（推論は走らない）。
      // base_parts未保存の作業ではここで失敗し、STEP4からのやり直しに誘導する
      let compositeReady = false;
      if (workspace.project.currentStep >= 5 && validExtractedParts) {
        compositeReady = await refreshWorkspaceCompositePreview(workspace, true)
          .then(() => activeWorkspacePath.current === workspace.workPath)
          .catch(() => false);
      } else {
        setWorkspaceCompositePreview(null);
      }
      if (activeWorkspacePath.current !== workspace.workPath) return;
      if (validRifeOutput) {
        await refreshMotionOutputStatus(workspace, validRifeOutput);
      } else {
        setMotionProfileReady(false);
      }
      // checkpointより上へ物理ファイルだけで昇格させない。対応成果物が消えている時だけ
      // 安全側へ下げる。完成済みRIFEが有効なら、後から上流素材を移動していてもSTEP7を維持する。
      restoredStep = restoredWorkspaceStep(restoredStep, {
        generatedReady: workspaceGeneratedReady,
        extractedReady: !!validExtractedParts,
        compositeReady,
        rifeReady: !!validRifeOutput,
      });
    } catch {
      // Not all workspaces are valid Codex jobs until Step 2 has been prepared.
    }

    if (activeWorkspacePath.current === workspace.workPath) {
      if (restoredStep < workspace.project.currentStep) {
        const updated = await invoke<ExpressionWorkspaceResult>("regress_expression_workspace_step", {
          workPath: workspace.workPath,
          currentStep: restoredStep,
        });
        if (activeWorkspacePath.current !== workspace.workPath) return;
        setExpressionWorkspace(updated);
      }
      setWorkspaceStep(restoredStep);
    }
  }

  async function refreshWorkspaceCompositePreview(
    workspaceOverride?: ExpressionWorkspaceResult,
    baseOnly = false,
    syncPositionSelection = true,
  ) {
    const workspace = workspaceOverride ?? expressionWorkspace;
    if (!workspace) return;
    const preview = await invoke<PreviewCodexCompositeResult>("preview_codex_composite", {
      jobPath: workspace.workPath,
      profile: baseOnly ? "base-only" : "auto",
    });
    if (activeWorkspacePath.current !== workspace.workPath) return;
    setWorkspaceCompositePreview(preview);
    if (!baseOnly && preview.previews.length > 0) {
      const available = preview.previews.map(item => item.part);
      setWorkspaceOverviewPreviewPart(previous => (
        available.includes(previous)
          ? previous
          : (available.includes("eyes-open") ? "eyes-open" : available[0])
      ));
      if (syncPositionSelection) {
        const nextPart = available.includes(workspaceSelectedPreviewPart)
          && WORKSPACE_ADJUST_PART_KEYS.includes(workspaceSelectedPreviewPart)
          ? workspaceSelectedPreviewPart
          : (available.includes("eyes-open") ? "eyes-open" : available.find(part => WORKSPACE_ADJUST_PART_KEYS.includes(part)));
        if (nextPart) {
          setWorkspaceSelectedPreviewPart(nextPart);
          setWorkspaceAdjustTarget(nextPart);
          setWorkspacePartDragMode(true);
        }
      }
    }
    return preview;
  }

  async function loadWorkspaceOverviewPreviews() {
    if (!expressionWorkspace || workspaceBusy || workspaceOverviewPreviewLoadingRef.current || !workspaceCompositePreview?.basePreview) return;
    setError("");
    workspaceOverviewPreviewLoadingRef.current = true;
    setWorkspaceOverviewPreviewLoading(true);
    setWorkspaceBusy(true);
    try {
      const preview = await refreshWorkspaceCompositePreview(expressionWorkspace, false, false);
      if (!preview?.previews.length) throw new Error("表示できる表情プレビューがありません");
      setStatus("表情プレビューを読み込みました");
    } catch (cause) {
      setError(`表情プレビューを読み込めませんでした: ${String(cause)}`);
    } finally {
      workspaceOverviewPreviewLoadingRef.current = false;
      setWorkspaceOverviewPreviewLoading(false);
      setWorkspaceBusy(false);
    }
  }

  async function openWorkspaceBaseAdjustment(
    workspaceOverride?: ExpressionWorkspaceResult | null,
    sourceOverride?: string | null,
  ): Promise<boolean> {
    const workspace = workspaceOverride ?? expressionWorkspace;
    const source = sourceOverride ?? workspaceFiles.source;
    if (!workspace || !source) {
      setError("立ち絵を読み込んでから素体調整を開いてください");
      return false;
    }
    setError("");
    setWorkspaceBusy(true);
    try {
      const base = await invoke<SeeThroughRunResult>("load_codex_source_see_through", {
        jobPath: workspace.workPath,
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
      return true;
    } catch (cause) {
      setError(String(cause));
      return false;
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function startInlineBaseEditor() {
    if (!expressionWorkspace) return;
    setError("");
    setWorkspaceInlineEditor("base");
    setWorkspaceEditorPreparing(true);
    try {
      // 同じセッションでの再編集なら、保存前の並び順や表示状態を維持する。
      if (!mappingPreview || !bodyPreview || !loadResult) {
        const opened = await openWorkspaceBaseAdjustment();
        if (!opened) {
          setWorkspaceInlineEditor(null);
          return;
        }
      }
      setStatus("素体のレイヤー順・表示・分離を調整します");
    } finally {
      setWorkspaceEditorPreparing(false);
    }
  }

  async function startInlinePositionEditor() {
    if (!expressionWorkspace || !workspaceExtractResult || !workspaceCompositePreview?.basePreview) return;
    setError("");
    const initialDrafts = createWorkspacePartAdjustmentDrafts(workspaceExtractResult.partAdjustments);
    workspacePartDraftsRef.current = initialDrafts;
    workspacePartEditorBaseline.current = cloneWorkspacePartAdjustmentDrafts(initialDrafts);
    setWorkspacePartDrafts(initialDrafts);
    workspacePartPersistedDuringEditor.current = false;
    setWorkspacePartSaving(false);
    setWorkspaceInlineEditor("position");
    setWorkspaceEditorPreparing(true);
    try {
      // 起動時はbase-onlyで高速復元する。表情サムネイルは編集開始時だけ展開する。
      const preview = workspaceCompositePreview.previews.length === 0
        ? await refreshWorkspaceCompositePreview(expressionWorkspace, false)
        : workspaceCompositePreview;
      const available = preview?.previews.map(item => item.part) ?? [];
      const target = available.includes(workspaceAdjustTarget)
        && WORKSPACE_ADJUST_PART_KEYS.includes(workspaceAdjustTarget)
        ? workspaceAdjustTarget
        : (available.includes("eyes-open") ? "eyes-open" : available.find(part => WORKSPACE_ADJUST_PART_KEYS.includes(part)));
      if (!target) throw new Error("位置調整できる目・口パーツがありません");
      setWorkspaceSelectedPreviewPart(target);
      setWorkspaceAdjustTarget(target);
      setWorkspacePartDragMode(true);
      loadWorkspacePartAdjustmentFields(target);
      setStatus("青い枠の目・口をドラッグして位置を調整できます");
    } catch (cause) {
      setError(String(cause));
      setWorkspaceInlineEditor(null);
    } finally {
      setWorkspaceEditorPreparing(false);
    }
  }

  function startInlineMotionEditor() {
    if (!workspaceRifeResult) return;
    setError("");
    setMotionEditorDirty(false);
    setMotionDraftSaveRequestId(0);
    setMotionEditorDraftSaveBusy(false);
    setMotionExportRequestId(0);
    setMotionPostSaveDestination("overview");
    setMotionEditorExportReady(false);
    setMotionEditorExportBusy(false);
    setWorkspaceInlineEditor("motion");
    setStatus("モーション素材を読み込んでいます...");
  }

  function handleInlineMotionNotify(message: string) {
    setStatus(message);
    showToast(message);
    if (message.startsWith("SpriTalk用アニメーション設定を出力しました:")) {
      setMotionProfileReady(true);
      setMotionEditorDirty(false);
    }
  }

  function handleMotionExportStateChange(state: { ready: boolean; busy: boolean }) {
    setMotionEditorExportReady(state.ready);
    setMotionEditorExportBusy(state.busy);
  }

  function handleInlineMotionDirtyChange(dirty: boolean, scope: "settings" | "sequence" = "settings") {
    setMotionEditorDirty(dirty);
    if (!dirty) return;
    // 調整値が変わった時点で、既存の書き出しは現在値と一致しなくなる。
    if (scope === "settings") setMotionProfileReady(false);
  }

  function handleInlineMotionDraftSaved() {
    const destination = motionPostSaveDestination;
    setMotionEditorDirty(false);
    setMotionDraftSaveRequestId(0);
    setMotionEditorDraftSaveBusy(false);
    setMotionExportRequestId(0);
    setMotionPostSaveDestination("overview");
    setMotionEditorExportReady(false);
    setMotionEditorExportBusy(false);
    setWorkspaceInlineEditor(null);
    if (destination === "live") {
      openLiveDisplayFromWorkspace();
    } else {
      setStatus("変更を保存して編集概要へ戻りました");
    }
  }

  function handleInlineMotionExported(path: string) {
    setMotionProfileReady(true);
    setMotionEditorDirty(false);
    setMotionDraftSaveRequestId(0);
    setMotionEditorDraftSaveBusy(false);
    setMotionExportRequestId(0);
    setMotionPostSaveDestination("overview");
    setMotionEditorExportReady(false);
    setMotionEditorExportBusy(false);
    setWorkspaceInlineEditor(null);
    setStatus(`SpriTalk向けアニメーションを作成しました: ${path}`);
  }

  async function loadWorkspaceImagePreviews(
    files: Record<string, string>,
    expectedWorkPath = activeWorkspacePath.current,
  ) {
    const entries = Object.entries(files).filter(([, path]) => !!path);
    if (entries.length === 0) {
      if (activeWorkspacePath.current === expectedWorkPath) {
        setWorkspaceImagePreviews({});
      }
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
    if (activeWorkspacePath.current === expectedWorkPath) {
      setWorkspaceImagePreviews(next);
    }
  }

  async function pickWorkspaceImage(key: "source" | "reference") {
    const file = await open({ multiple: false, directory: false, filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (!file || typeof file !== "string") return;
    setWorkspaceFiles(prev => {
      const next = { ...prev, [key]: file };
      void loadWorkspaceImagePreviews(next);
      return next;
    });
    setWorkspaceCodexRequestDirty(true);
  }

  function clearWorkspaceImage(key: "source" | "reference") {
    setWorkspaceFiles(prev => {
      const next = { ...prev };
      delete next[key];
      void loadWorkspaceImagePreviews(next);
      return next;
    });
    setWorkspaceCodexRequestDirty(true);
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
    workspaceGeneratedPollRequestId.current += 1;
    setWorkspaceBusy(true);
    try {
      const status = await invoke<WorkspaceGeneratedPartsStatus>("prepare_workspace_codex_request", {
        request: {
          workPath: expressionWorkspace.workPath,
          sourceImagePath: workspaceFiles.source,
          referenceImagePath: workspaceFiles.reference || null,
          prompt: workspaceCodexPrompt,
          mouthCorner: workspaceMouthCorner,
          mouthSize: "normal",
        },
      });
      setWorkspaceGeneratedStatus(status);
      setWorkspaceCodexRequestDirty(false);
      // 作成ガイドを更新した時点で、旧入力から作られた下流結果は画面上でも無効化する。
      setWorkspaceExtractResult(null);
      setWorkspaceCompositePreview(null);
      setWorkspaceRifeResult(null);
      setMotionProfileReady(false);
      await setWorkspaceStepAndPersist(2);
      setStatus("作成ガイドを出力しました。選んだ方法で7枚の表情素材を用意してください");
      showToast("依頼ファイルを作成しました");
    } catch (cause) {
      await reloadWorkspaceAfterMutationFailure(expressionWorkspace.workPath);
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function inspectWorkspaceGeneratedParts() {
    if (!expressionWorkspace) return;
    setError("");
    workspaceGeneratedPollRequestId.current += 1;
    setWorkspaceBusy(true);
    try {
      const status = await invoke<WorkspaceGeneratedPartsStatus>("inspect_workspace_generated_parts", { workPath: expressionWorkspace.workPath });
      setWorkspaceGeneratedStatus(status);
      if (status.downstreamStale) {
        const synced = await reloadWorkspaceAfterMutationFailure(expressionWorkspace.workPath);
        if (!synced) {
          setError("表情素材の変更を検知しましたが、作業状態を再読込できませんでした。作業フォルダを開き直してください");
          setStatus("作業状態の再読込が必要です");
          return;
        }
        setStatus("表情素材の変更を検知しました。STEP3から再処理してください");
        showToast("表情素材の変更を検知しました");
        return;
      }
      // 揃っていても自動でSTEP3へは進めない（進むのはユーザーの「次へ」だけ）
      if (status.ready) showToast("表情素材が揃いました");
      setStatus(status.ready ? "表情素材が揃いました。「次へ」でSee-Throughに進めます" : `表情素材に不足があります（残り${status.missingParts.length}ファイル）`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function refreshWorkspaceSeeThroughStatus(
    profile: SeeThroughProfile | "auto" = requestedSeeThroughProfile(),
  ) {
    const requestId = ++seeThroughStatusRequestId.current;
    setError("");
    setSeeThroughRuntime(null);
    setWorkspaceBusy(true);
    setSeeThroughProgress({ stage: "status", percent: 0, message: "See-Through環境を確認しています" });
    setStatus("See-Through環境を確認しています");
    try {
      const runtime = await invoke<SeeThroughRuntimeStatus>("get_see_through_runtime_status", { profile });
      if (requestId !== seeThroughStatusRequestId.current) return;
      setSeeThroughRuntime(runtime);
      applyRecommendedSeeThroughProfile(runtime);
      await invoke<Array<{ index: number; name: string; memoryMb: number }>>("list_see_through_gpus")
        .then(setSeeThroughGpus)
        .catch(() => setSeeThroughGpus([]));
      await invoke<{ path: string; isDefault: boolean }>("get_see_through_install_location")
        .then(setSeeThroughInstallLocation)
        .catch(() => setSeeThroughInstallLocation(null));
      await invoke<{ configured: boolean }>("get_hf_token_status")
        .then(setHfTokenStatus)
        .catch(() => setHfTokenStatus(null));
      if (requestId !== seeThroughStatusRequestId.current) return;
      setStatus(runtime.message);
      setSeeThroughProgress({ stage: "status", percent: 100, message: runtime.message });
    } catch (cause) {
      if (requestId === seeThroughStatusRequestId.current) setError(String(cause));
    } finally {
      if (requestId === seeThroughStatusRequestId.current) setWorkspaceBusy(false);
    }
  }

  async function changeSeeThroughGpuSelection(gpuIndex: number | null) {
    const requestId = ++seeThroughStatusRequestId.current;
    setError("");
    setSeeThroughRuntime(null);
    setWorkspaceBusy(true);
    setSeeThroughGpuIndex(gpuIndex);
    try {
      await invoke("set_see_through_gpu", { gpuIndex });
      const runtime = await invoke<SeeThroughRuntimeStatus>("get_see_through_runtime_status", {
        profile: requestedSeeThroughProfile(),
      });
      if (requestId !== seeThroughStatusRequestId.current) return;
      setSeeThroughRuntime(runtime);
      applyRecommendedSeeThroughProfile(runtime);
      setStatus(runtime.message);
    } catch (cause) {
      if (requestId === seeThroughStatusRequestId.current) setError(String(cause));
    } finally {
      if (requestId === seeThroughStatusRequestId.current) setWorkspaceBusy(false);
    }
  }

  async function changeSeeThroughInstallLocation() {
    const selected = await open({ multiple: false, directory: true, title: "See-Throughのインストール先フォルダを選択" });
    const dir = typeof selected === "string" ? selected : null;
    if (!dir) return;
    setError("");
    setWorkspaceBusy(true);
    try {
      const location = await invoke<{ path: string; isDefault: boolean }>("set_see_through_install_location", { path: dir });
      setSeeThroughInstallLocation(location);
      setSeeThroughRuntime(null);
      showToast("インストール先を変更しました。この場所で初回セットアップが必要です");
      setStatus(`インストール先を変更しました: ${location.path}`);
      await refreshWorkspaceSeeThroughStatus();
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function resetSeeThroughInstallLocation() {
    setError("");
    setWorkspaceBusy(true);
    try {
      const location = await invoke<{ path: string; isDefault: boolean }>("set_see_through_install_location", { path: null });
      setSeeThroughInstallLocation(location);
      setSeeThroughRuntime(null);
      showToast("インストール先を既定に戻しました");
      await refreshWorkspaceSeeThroughStatus();
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function saveHfToken() {
    if (!hfTokenInput.trim()) return;
    setError("");
    setWorkspaceBusy(true);
    try {
      const status = await invoke<{ configured: boolean }>("save_hf_token", { token: hfTokenInput });
      setHfTokenStatus(status);
      setHfTokenInput("");
      showToast("HuggingFaceトークンを保存しました");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function deleteHfToken() {
    setError("");
    setWorkspaceBusy(true);
    try {
      const status = await invoke<{ configured: boolean }>("delete_hf_token");
      setHfTokenStatus(status);
      showToast("HuggingFaceトークンを削除しました");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function prepareWorkspaceSeeThroughRuntime() {
    ++seeThroughStatusRequestId.current;
    seeThroughCancelRequested.current = false;
    setError("");
    setSeeThroughRuntime(null);
    setWorkspaceBusy(true);
    setSeeThroughStartedAt(Date.now());
    setSeeThroughElapsedSeconds(0);
    setSeeThroughPhase(null);
    setSeeThroughProgress({ stage: "prepare", percent: 0, message: "See-Throughを準備しています" });
    try {
      const runtime = await invoke<SeeThroughRuntimeStatus>("prepare_see_through_runtime", {
        profile: requestedSeeThroughProfile(),
      });
      setSeeThroughRuntime(runtime);
      applyRecommendedSeeThroughProfile(runtime);
      setStatus(runtime.message);
    } catch (cause) {
      if (seeThroughCancelRequested.current) {
        setError("");
        setStatus("ランタイムのセットアップを中止しました。もう一度実行できます");
        setSeeThroughProgress({ stage: "cancelled", percent: 0, message: "ランタイムのセットアップを中止しました" });
      } else {
        setError(String(cause));
      }
    } finally {
      setWorkspaceBusy(false);
      setSeeThroughStartedAt(null);
      setSeeThroughPhase(null);
      seeThroughCancelRequested.current = false;
    }
  }

  async function cancelWorkspaceSeeThroughSetup() {
    seeThroughCancelRequested.current = true;
    setError("");
    setStatus("See-Throughセットアップを中止しています");
    try {
      const cancelled = await invoke<boolean>("cancel_see_through");
      if (!cancelled) {
        seeThroughCancelRequested.current = false;
        setError("停止対象のSee-Throughプロセスが見つかりませんでした");
      }
    } catch (cause) {
      seeThroughCancelRequested.current = false;
      setError(String(cause));
    }
  }

  async function startWorkspaceSeeThroughModelDownload() {
    setError("");
    setSeeThroughModelDownloadLaunching(true);
    try {
      const launch = await invoke<{ started: boolean; message: string }>("start_see_through_model_download", {
        profile: requestedSeeThroughProfile(),
      });
      setStatus(launch.started ? `${launch.message}。完了までコンソールを開いたままにしてください` : launch.message);
      if (launch.started) {
        setSeeThroughProgress({ stage: "model-download-external", percent: 0, message: launch.message });
        setSeeThroughRuntime(current => current ? { ...current, modelDownloadBusy: true } : current);
      } else {
        void refreshWorkspaceSeeThroughStatus();
      }
    } catch (cause) {
      setError(String(cause));
    } finally {
      setSeeThroughModelDownloadLaunching(false);
    }
  }

  async function runWorkspaceSeeThroughBatch() {
    if (!expressionWorkspace || !workspaceFiles.source || !workspaceGeneratedStatus?.ready) {
      setError("立ち絵と7枚の表情素材を先に揃えてください");
      return;
    }
    if (!seeThroughRuntime?.ready || seeThroughRuntime.selectedProfile !== seeThroughProfile) {
      setError(seeThroughRuntime?.message ?? "See-Throughの初回セットアップを先に完了してください");
      return;
    }
    setError("");
    workspaceGeneratedPollRequestId.current += 1;
    setWorkspaceBusy(true);
    setSeeThroughStartedAt(Date.now());
    setSeeThroughElapsedSeconds(0);
    setSeeThroughProgress({ stage: "inference", percent: 0, message: "See-Through一括分解中" });
    setSeeThroughPhase({ index: 1, total: 3, label: "立ち絵を分解しています" });
    // 同じ案件で再分解した場合も、旧レイヤー順や切り出しを次のSTEP4へ持ち越さない。
    resetWorkspaceBaseEditorState();
    try {
      // ユーザーが手動でプロファイルを選んでいなければ "auto" を渡し、実行時点のGPU検出
      // による判定（VRAM 16GB以上ならstandard等）をバックエンドに委ねる。
      // フロント側stateの初期値（"low-vram"）をそのまま送ると、環境確認がまだ反映されて
      // いない場面（再起動直後・ステップ再訪問等）で意図せず低速な量子化経路に落ちるため
      const base = await invoke<SeeThroughRunResult>("run_see_through", {
        sourcePath: workspaceFiles.source,
        profile: seeThroughProfileTouched.current ? seeThroughProfile : "auto",
        splitParts: seeThroughSplitParts,
        options: seeThroughOptions,
      });
      // 実際に使われたプロファイルをUIへ反映（自動判定の結果を可視化）
      setSeeThroughProfile(base.selectedProfile as SeeThroughProfile);
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
        showToast(`推論エラーのため設定を変更して続行しています（${base.oomRetryNote}）`);
        pushWorkspaceLog("info", `警告: 推論エラーのため自動リトライしました（${base.oomRetryNote}）`);
        // 自動リトライで実際に成功したprofileを次回以降も使い、autoが失敗側へ戻るのを防ぐ
        seeThroughProfileTouched.current = true;
        setSeeThroughProfile(base.selectedProfile as SeeThroughProfile);
      }
      const effectiveSeeThroughOptions = base.effectiveOptions ?? seeThroughOptions;
      if (base.effectiveOptions) setSeeThroughOptions(base.effectiveOptions);
      const runtimeAfterInference = await invoke<SeeThroughRuntimeStatus>("get_see_through_runtime_status", {
        profile: base.selectedProfile,
      });
      ++seeThroughStatusRequestId.current;
      setSeeThroughRuntime(runtimeAfterInference);
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
      setSeeThroughProgress({ stage: "inference", percent: 50, message: "表情素材をSee-Throughで分解しています" });
      setSeeThroughPhase({ index: 2, total: 3, label: "表情素材を分解しています" });
      const extracted = await invoke<ExtractCodexGeneratedPartsResult>("extract_codex_generated_parts", {
        jobPath: expressionWorkspace.workPath,
        // 1回目の呼び出しで確定した実際のプロファイルをそのまま使う（"auto"を再解決しない）
        profile: base.selectedProfile,
        splitParts: effectiveSplitParts,
        options: effectiveSeeThroughOptions,
      });
      seeThroughProfileTouched.current = true;
      setSeeThroughProfile(extracted.selectedProfile as SeeThroughProfile);
      if (extracted.effectiveOptions) setSeeThroughOptions(extracted.effectiveOptions);
      setSeeThroughSplitParts(extracted.splitParts);
      // 左右分解フォールバック等の警告は作業ログへ残す（トーストでは流れてしまうため）
      for (const warning of extracted.warnings) {
        pushWorkspaceLog("info", `警告: ${warning}`);
      }
      if (extracted.warnings.some(warning => warning.includes("左右分解なしで処理しました"))) {
        setSeeThroughSplitParts(false);
        showToast("左右パーツ分解に失敗したため、分解なしで続行しました");
      }
      if (extracted.warnings.some(warning => warning.includes("推論エラーのため自動リトライしました"))) {
        showToast("推論エラーのため設定を変更して続行しました");
      }
      // 期待する全表情（目開閉＋口6種）のうち、抽出できず欠落したものを明示する
      const EXPECTED_EXPRESSION_PARTS = ["eyes-open", "eyes-closed", "mouth-closed", "mouth-a", "mouth-i", "mouth-u", "mouth-e", "mouth-o"];
      const missingExpressions = EXPECTED_EXPRESSION_PARTS.filter(part => !extracted.extractedParts.includes(part));
      if (missingExpressions.length > 0) {
        const label = missingExpressions.join("・");
        pushWorkspaceLog("error", `表情が不足: ${label} を抽出できませんでした`);
        setError(`一部の表情（${label}）を元画像から抽出できませんでした。これらはまばたきや口パクに使われます。\n\n対処: STEP2の「配置フォルダを開く」で該当画像を確認し、目や口がはっきり分かる素材へ差し替えてSTEP3を再実行してください。画像編集AIや手作業で作り直しても構いません。左右パーツ分解をOFFにすると改善する場合もあります。`);
      }
      setWorkspaceExtractResult(extracted);
      setWorkspaceCompositePreview(null);
      setWorkspaceRifeResult(null);
      setMotionProfileReady(false);
      if (missingExpressions.length > 0) {
        await setWorkspaceStepAndPersist(3);
        setSeeThroughProgress({ stage: "error", percent: 100, message: "必要な表情素材の抽出に失敗しました" });
        setStatus("不足している表情素材を確認し、STEP2から差し替えてください");
        return;
      }
      await setWorkspaceStepAndPersist(4);
      setSeeThroughPhase({ index: 3, total: 3, label: "プレビューを準備しています" });
      const allPreview = await invoke<MappingPreviewResult>("get_all_layers_preview");
      setMappingPreview(allPreview);
      setSeeThroughProgress({ stage: "complete", percent: 100, message: "See-Through一括分解が完了しました" });
      setStatus(`See-Through一括分解が完了しました: ${extracted.extractedParts.length}パーツ`);
      showToast("一括分解が完了しました");
    } catch (cause) {
      const message = String(cause);
      await reloadWorkspaceAfterMutationFailure(expressionWorkspace.workPath);
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

  // 立ち絵1枚をLayerDiff（レイヤー分解）だけで処理し、獣耳・眼鏡などSeed依存で
  // 抽出が不安定なパーツの取れ具合をサムネイルで確認する。深度推定・PSD組立・
  // 表情素材7枚の分解をすべて省くため、一括分解より大幅に短時間で終わる。
  // Seedを変えて結果が気に入るまで繰り返す（「ガチャ」）用途を想定
  async function probeWorkspaceSeeThroughLayers(randomizeSeed: boolean) {
    if (!expressionWorkspace || !workspaceFiles.source) {
      setError("立ち絵を先に選択してください");
      return;
    }
    if (!seeThroughRuntime?.ready || seeThroughRuntime.selectedProfile !== seeThroughProfile) {
      setError(seeThroughRuntime?.message ?? "See-Throughの初回セットアップを先に完了してください");
      return;
    }
    setError("");
    let options = seeThroughOptions;
    if (randomizeSeed) {
      options = { ...seeThroughOptions, seed: Math.floor(Math.random() * 2_147_483_647) };
      setSeeThroughOptions(options);
    }
    setWorkspaceBusy(true);
    setSeeThroughLayerProbeRunning(true);
    setSeeThroughProbeZoom(null);
    try {
      const result = await invoke<SeeThroughLayerProbeResult>("probe_see_through_layers", {
        sourcePath: workspaceFiles.source,
        profile: seeThroughProfileTouched.current ? seeThroughProfile : "auto",
        options,
      });
      setSeeThroughProfile(result.selectedProfile as SeeThroughProfile);
      seeThroughProfileTouched.current = true;
      setSeeThroughLayerProbe(result);
      const interesting = result.layers.filter(layer => /^(ears|earwear|headwear|eyewear)$/.test(layer.name));
      showToast(interesting.length > 0
        ? `${interesting.map(layer => layer.name).join("・")} を抽出しました`
        : "獣耳・眼鏡系レイヤーは抽出されませんでした");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
      setSeeThroughLayerProbeRunning(false);
    }
  }

  async function persistWorkspacePartDrafts(draftSnapshot: WorkspacePartAdjustmentDrafts) {
    if (!expressionWorkspace || !workspaceExtractResult) {
      throw new Error("差分パーツが読み込まれていません");
    }
    const workspacePath = expressionWorkspace.workPath;
    const changedParts = WORKSPACE_ADJUST_PART_KEYS.filter(part => !sameWorkspacePartAdjustment(
      workspacePartEditorBaseline.current[part],
      draftSnapshot[part],
    ));
    if (changedParts.length === 0) return;

    const result = await invoke<AdjustCodexExtractedPartsResult>("adjust_codex_extracted_parts_batch", {
      request: {
        jobPath: workspacePath,
        adjustments: changedParts.map(part => ({
          part,
          ...(draftSnapshot[part] ?? DEFAULT_PART_ADJUSTMENT),
        })),
      },
    });
    if (activeWorkspacePath.current !== workspacePath) {
      throw new Error("作業フォルダが切り替わったため保存を中止しました");
    }
    // バッチ全体が成功した後だけ画面状態を更新する。失敗時はバックエンドが画像を
    // ロールバックし、呼び出し元がproject.jsonの安全なcheckpointを再読込する。
    setWorkspaceRifeResult(null);
    setMotionProfileReady(false);
    setWorkspaceExtractResult({
      ...workspaceExtractResult,
      extractedPartsPath: result.extractedPartsPath,
      partAdjustments: result.partAdjustments,
    });
    workspacePartEditorBaseline.current = cloneWorkspacePartAdjustmentDrafts(draftSnapshot);
    workspacePartPersistedDuringEditor.current = true;
  }

  async function generateWorkspaceRifeOutputs() {
    if (!expressionWorkspace || !workspaceExtractResult) {
      setError("先にSee-Through一括分解を完了してください");
      return;
    }
    setError("");
    // 出力先は毎回同じため、古いMotionTunePanelを先にアンマウントしてキャッシュを捨てる。
    // 成功後に結果を再設定すると、新しいPNGとlayer-order.jsonを必ず読み直す。
    setWorkspaceInlineEditor(null);
    setWorkspaceRifeResult(null);
    setMotionProfileReady(false);
    setMotionEditorDirty(false);
    setMotionDraftSaveRequestId(0);
    setMotionEditorDraftSaveBusy(false);
    setMotionExportRequestId(0);
    setMotionPostSaveDestination("overview");
    setWorkspaceBusy(true);
    try {
      // STEP4で保存済みのbase_partsを正本として使う。
      // ここで再保存すると、再起動後など揮発メモリが空の作業を再開した際に失敗する。
      // RIFE側で保存済みbody.pngの存在確認と読み込みを行うため、事前保存は不要。
      const result = await invoke<GenerateCodexRifeOutputResult>("generate_codex_rife_outputs", {
        jobPath: expressionWorkspace.workPath,
        frameCount,
        profile: "auto",
      });
      setWorkspaceRifeResult(result);
      // RIFE完了後は自動でSTEP7（モーション調整）へ進む
      await setWorkspaceStepAndPersist(7);
      setStatus(`SpriTalk用フォルダへ出力しました: ${result.outputPath}。モーション調整に進みます`);
      showToast("RIFE補完が完了しました");
    } catch (cause) {
      await reloadWorkspaceAfterMutationFailure(expressionWorkspace.workPath);
      setError(String(cause));
    } finally {
      setWorkspaceBusy(false);
    }
  }

  function returnToModeSelect() {
    setWorkspaceInlineEditor(null);
    setWorkspaceEditorPreparing(false);
    setMotionEditorDirty(false);
    setMotionDraftSaveRequestId(0);
    setMotionEditorDraftSaveBusy(false);
    setMode("select");
    setStatus("作業を選択してください");
    resetZoom();
  }

  const bodyCategory = mappingPreview?.categories.find(c => c.target === "free");
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
      setError("切り出す範囲を塗ってください");
      return;
    }

    if (patchDraftSource === CHEST_CUT_SENTINEL) {
      // 胸部範囲ガイドを保存する。body自体から画素は抜かない。
      // 互換用chest.pngの作成はcreate_base実行時（保存して戻る）にまとめて行う。
      setChestMaskDataUrl(canvas.toDataURL("image/png"));
      setPatchDraftSource("");
      clearPatchMask();
      setStatus("胸部の変形範囲を設定しました。「保存して戻る」で反映されます。");
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
    const nextOpacities = { ...layerOpacitiesRef.current, [patch.id]: 0.5 };

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

  function renderPartPositionEditor() {
    const activePreviewSet = workspaceCompositePreview;
    const positionPreviews = activePreviewSet?.previews.filter(item => (
      WORKSPACE_ADJUST_PART_KEYS.includes(workspacePreviewItemKey(item))
    )) ?? [];
    const selectedPreview = positionPreviews.find(
      item => workspacePreviewItemKey(item) === workspaceSelectedPreviewPart,
    ) ?? null;
    const mainPreviewImage = selectedPreview?.preview ?? "";
    const mainPreviewLabel = selectedPreview ? workspacePreviewItemLabel(selectedPreview) : "目・口を選択";
    const directEditReady = !!selectedPreview
      && workspacePartDragMode
      && !workspaceBusy
      && workspacePositionVisual?.part === workspaceAdjustTarget
      && workspaceSelectedPreviewPart === workspaceAdjustTarget;

    return (
      <div className="part-position-editor">
        <section className="part-position-parts">
          <div className="workspace-panel-heading">
            <span>PARTS</span>
            <h3>目・口パーツ</h3>
            <p>調整するパーツを選びます。変更は「編集を完了」でまとめて保存されます。</p>
          </div>
          <div className="workspace-position-part-list">
            {positionPreviews.map(item => {
              const key = workspacePreviewItemKey(item);
              const adjusted = !isDefaultWorkspacePartAdjustment(workspacePartDrafts[key]);
              return (
                <button
                  type="button"
                  key={key}
                  className={workspaceSelectedPreviewPart === key ? "active" : ""}
                  aria-pressed={workspaceSelectedPreviewPart === key}
                  disabled={workspaceBusy}
                  onClick={() => selectWorkspacePositionPart(key)}
                >
                  <img src={item.preview} alt="" />
                  <span>
                    <strong>{workspacePreviewItemLabel(item)}</strong>
                    <small>{key.startsWith("eyes-") ? "目パーツ" : "口パーツ"}</small>
                  </span>
                  {adjusted && <em className="workspace-adjusted-badge">調整済</em>}
                </button>
              );
            })}
            {positionPreviews.length === 0 && <p className="workspace-position-list-empty">調整できるパーツがありません</p>}
          </div>
        </section>

        <aside className="workspace-hub-preview part-position-preview">
          <div className="preview-card-heading"><span>PREVIEW</span><strong>{mainPreviewLabel}</strong></div>
          <div className="workspace-preview-stage">
            {mainPreviewImage ? (
              <>
                <div
                  className={`workspace-preview-pan${directEditReady ? " position-direct" : ""}${workspacePreviewDragging ? " dragging" : ""}`}
                  onWheel={handleWorkspacePreviewWheel}
                  onPointerDown={directEditReady ? undefined : handleWorkspacePreviewPointerDown}
                  onPointerMove={directEditReady ? undefined : handleWorkspacePreviewPointerMove}
                  onPointerUp={directEditReady ? undefined : handleWorkspacePreviewPointerUp}
                  onPointerCancel={directEditReady ? undefined : handleWorkspacePreviewPointerUp}
                  onLostPointerCapture={directEditReady ? undefined : handleWorkspacePreviewLostPointerCapture}
                >
                  {directEditReady && workspacePositionVisual ? (
                    <canvas
                      ref={workspacePositionCanvasRef}
                      className={`workspace-position-canvas${workspacePartHovered ? " part-hover" : ""}${workspacePreviewDragging ? ` dragging-${workspacePreviewDrag.current?.mode ?? "pan"}` : ""}`}
                      width={workspacePositionVisual.width}
                      height={workspacePositionVisual.height}
                      aria-label={`${mainPreviewLabel} の目・口をドラッグして位置調整`}
                      onPointerDown={handleWorkspacePositionPointerDown}
                      onPointerMove={handleWorkspacePositionPointerMove}
                      onPointerUp={handleWorkspacePositionPointerUp}
                      onPointerCancel={handleWorkspacePositionPointerUp}
                      onPointerLeave={() => {
                        if (!workspacePreviewDragging) setWorkspacePartHovered(false);
                      }}
                      onLostPointerCapture={handleWorkspacePositionLostPointerCapture}
                      style={{ transform: `translate(${workspacePreviewPan.x}px, ${workspacePreviewPan.y}px) scale(${workspacePreviewZoom})` }}
                    />
                  ) : (
                    <img
                      src={mainPreviewImage}
                      alt={`${mainPreviewLabel} プレビュー`}
                      draggable={false}
                      style={{ transform: `translate(${workspacePreviewPan.x}px, ${workspacePreviewPan.y}px) scale(${workspacePreviewZoom})` }}
                    />
                  )}
                </div>
                <div className="workspace-preview-zoom-controls">
                  <button className="btn-zoom" type="button" onClick={() => setWorkspacePreviewZoom(previous => Math.min(8, previous * 1.25))}>+</button>
                  <span className="zoom-level">{Math.round(workspacePreviewZoom * 100)}%</span>
                  <button className="btn-zoom" type="button" onClick={() => setWorkspacePreviewZoom(previous => Math.max(0.25, previous * 0.8))}>-</button>
                  <button className="btn-zoom btn-zoom-reset" type="button" onClick={resetWorkspacePreviewZoom}>リセット</button>
                </div>
              </>
            ) : <span>素体構成を保存するとプレビューできます</span>}
          </div>
          <div className="workspace-position-toolbar" aria-label="選択パーツの位置と大きさ">
            <label><span>X offset</span>
              <span className="workspace-nudge-row">
                <button type="button" title="Xを10px左へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetX: workspacePartOffsetX - 10 })}>≪</button>
                <button type="button" title="Xを1px左へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetX: workspacePartOffsetX - 1 })}>‹</button>
                <input type="number" value={workspacePartOffsetX} disabled={workspaceBusy} onChange={event => updateSelectedWorkspacePartDraft({ offsetX: Number(event.target.value) })} />
                <button type="button" title="Xを1px右へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetX: workspacePartOffsetX + 1 })}>›</button>
                <button type="button" title="Xを10px右へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetX: workspacePartOffsetX + 10 })}>≫</button>
              </span>
            </label>
            <label><span>Y offset</span>
              <span className="workspace-nudge-row">
                <button type="button" title="Yを10px上へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetY: workspacePartOffsetY - 10 })}>≪</button>
                <button type="button" title="Yを1px上へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetY: workspacePartOffsetY - 1 })}>‹</button>
                <input type="number" value={workspacePartOffsetY} disabled={workspaceBusy} onChange={event => updateSelectedWorkspacePartDraft({ offsetY: Number(event.target.value) })} />
                <button type="button" title="Yを1px下へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetY: workspacePartOffsetY + 1 })}>›</button>
                <button type="button" title="Yを10px下へ" disabled={workspaceBusy} onClick={() => updateSelectedWorkspacePartDraft({ offsetY: workspacePartOffsetY + 10 })}>≫</button>
              </span>
            </label>
            <label><span>Scale {workspacePartScale}%</span>
              <span className="workspace-scale-row">
                <input type="range" min={50} max={150} value={workspacePartScale} disabled={workspaceBusy} onChange={event => updateSelectedWorkspacePartDraft({ scalePercent: Number(event.target.value) })} />
                <input type="number" min={50} max={150} value={workspacePartScale} disabled={workspaceBusy} onChange={event => updateSelectedWorkspacePartDraft({ scalePercent: Number(event.target.value) })} />
              </span>
            </label>
            <button
              type="button"
              className="btn btn-secondary workspace-position-reset"
              data-action-tone="edit"
              disabled={workspaceBusy || isDefaultWorkspacePartAdjustment(workspacePartDrafts[workspaceAdjustTarget])}
              onClick={() => setWorkspacePartDraft(workspaceAdjustTarget, DEFAULT_PART_ADJUSTMENT)}
            >0に戻す</button>
          </div>
        </aside>
      </div>
    );
  }

  function renderWorkspaceMode() {
    const workspace = expressionWorkspace;
    const steps = [
      [1, "画像選択", "立ち絵と参照画像"],
      [2, "表情素材", "Codex・画像編集AI・手作業"],
      [3, "See-Through", "一括分解"],
      [4, "素体調整", "レイヤー確認"],
      [5, "差分位置調整", "合成確認"],
      [6, "RIFE補完", "SpriTalk出力"],
      [7, "モーション調整", "揺れ・口パクの仕上げ"],
    ] as Array<[WorkspaceStep, string, string]>;
    if (!workspace) return null;

    const expectedExpressionParts = [
      "eyes-open", "eyes-closed", "mouth-closed", "mouth-a", "mouth-i", "mouth-u", "mouth-e", "mouth-o",
    ];
    const step2FilesPresent = !!workspaceGeneratedStatus
      && workspaceGeneratedStatus.expectedParts.length > 0
      && workspaceGeneratedStatus.missingParts.length === 0;
    const step3ArtifactsPresent = !!workspaceExtractResult
      && expectedExpressionParts.every(part => workspaceExtractResult.extractedParts.includes(part));
    const {
      step2Ready,
      step3Complete,
      step4Complete,
      step5Complete: step5Confirmed,
      step6Complete: step6Current,
    } = workspaceArtifactReadiness({
      currentStep: workspace.project.currentStep,
      requestDirty: workspaceCodexRequestDirty,
      generatedReady: !!workspaceGeneratedStatus?.ready,
      extractedComplete: step3ArtifactsPresent,
      basePreviewReady: !!workspaceCompositePreview?.basePreview,
      rifeCurrent: !!workspaceRifeResult && workspaceRifeResult.frameCount === frameCount,
    });
    const step3HasMissingParts = !!workspaceExtractResult && !step3ArtifactsPresent;
    const step2HasInvalidSize = (workspaceGeneratedStatus?.sizeMismatches.length ?? 0) > 0;
    const phaseModel = buildWorkspacePhaseModel({
      currentStep: workspaceStep,
      finished: {
        1: !!workspaceFiles.source,
        2: step2FilesPresent,
        3: step3Complete,
        4: step4Complete,
        5: step5Confirmed,
        6: !!workspaceRifeResult,
        7: motionProfileReady,
      },
      runningStep: workspaceBusy && (
        (workspaceStep === 3 && seeThroughPhase !== null)
        || (workspaceStep === 6 && progress.total > 0)
      ) ? workspaceStep : null,
      stale: {
        2: workspaceCodexRequestDirty || (workspaceGeneratedStatus?.staleParts?.length ?? 0) > 0,
        6: !!workspaceRifeResult && !step6Current,
      },
      error: {
        2: step2HasInvalidSize,
        3: step3HasMissingParts,
      },
    });
    const currentStepModel = phaseModel.substeps.find(item => item.step === workspaceStep);
    const canOpenWorkspaceStep = (step: WorkspaceStep) => {
      if (workspaceBusy || workspaceEditorPreparing) return false;
      if (step === 1) return true;
      if (step === 2) return !!workspaceFiles.source;
      if (step === 3) return step2Ready;
      if (step === 4) return step3Complete;
      if (step === 5) return step3Complete && step4Complete;
      if (step === 6) return step5Confirmed;
      return step6Current;
    };
    const canAdvanceWorkspaceStep = () => {
      if (workspaceBusy) return false;
      if (workspaceStep === 1) return !!workspaceFiles.source;
      if (workspaceStep === 2) return step2Ready;
      if (workspaceStep === 3) return step3Complete;
      if (workspaceStep === 4) return step4Complete;
      if (workspaceStep === 5) return step5Confirmed;
      if (workspaceStep === 6) return step6Current;
      return false;
    };
    const runtimeMatchesSelectedProfile = seeThroughRuntime?.selectedProfile === seeThroughProfile;
    const selectedProfileReady = !!seeThroughRuntime?.ready && runtimeMatchesSelectedProfile;
    const seeThroughRunning = workspaceBusy && !!seeThroughProgress && ["prepare", "inference", "load"].includes(seeThroughProgress.stage);
    // 概要画面では目・口を合成した8表情だけを表示する。目口の無いbaseは内部の
    // 合成下地として保持し、UIには出さない。RIFE全フレームはSTEP7編集開始まで読まない。
    const overviewPreviewItems = workspaceCompositePreview?.previews.filter(item => item.part !== "base") ?? [];
    const selectedOverviewPreview = overviewPreviewItems.find(item => item.part === workspaceOverviewPreviewPart)
      ?? overviewPreviewItems.find(item => item.part === "eyes-open")
      ?? overviewPreviewItems[0]
      ?? null;
    const sourcePreviewBeforeBaseSave = workspaceStep === 4 && !step4Complete
      ? (workspaceImagePreviews.source ?? "")
      : "";
    const mainPreviewImage = selectedOverviewPreview?.preview ?? sourcePreviewBeforeBaseSave;
    const mainPreviewLabel = selectedOverviewPreview
      ? workspacePreviewItemLabel(selectedOverviewPreview)
      : sourcePreviewBeforeBaseSave ? "元画像" : "表情プレビュー";
    const selectWorkspaceStep = (step: WorkspaceStep) => {
      if (step === workspaceStep || workspaceInlineEditor) return;
      setError("");
      setWorkspaceStep(step);
      if (step === 2 && workspaceGeneratedStatus && !workspaceCodexRequestDirty) {
        void inspectWorkspaceGeneratedParts();
      }
    };
    const goPreviousWorkspaceStep = () => {
      if (workspaceInlineEditor) {
        void closeInlineEditor();
        return;
      }
      if (workspaceStep === 1) {
        returnToModeSelect();
        return;
      }
      // 閲覧のために戻るだけでは、project.jsonの到達済みチェックポイントを巻き戻さない。
      // 再生成・再保存を実行した場合は各処理側が適切なSTEPを永続化する。
      const previousStep = (workspaceStep - 1) as WorkspaceStep;
      setWorkspaceStep(previousStep);
      if (previousStep === 2 && workspaceGeneratedStatus && !workspaceCodexRequestDirty) {
        void inspectWorkspaceGeneratedParts();
      }
    };
    // 「次へ」が押せない理由の提示（無効ボタンの理由が分からない問題への対応）
    const nextStepBlockReason = (): string | null => {
      if (workspaceStep >= 7 || canAdvanceWorkspaceStep()) return null;
      if (workspaceBusy) return "処理が完了するまでお待ちください";
      switch (workspaceStep) {
        case 1: return "立ち絵を選択すると次へ進めます";
        case 2: {
          if (workspaceCodexRequestDirty) return "素材設定を作成ガイドへ反映すると進めます";
          const pending = (workspaceGeneratedStatus?.missingParts.length ?? 0)
            + (workspaceGeneratedStatus?.staleParts?.length ?? 0);
          return pending > 0
            ? `表情素材が揃うと進めます（未生成・再生成が残り${pending}ファイル）`
            : "作成ガイドを出力し、7枚の表情素材を配置してください";
        }
        case 3: return step3HasMissingParts ? "不足している表情素材をSTEP2で差し替えて再実行してください" : "「一括分解を開始」が完了すると次へ進めます";
        case 4: return "「編集を開始」で素体を保存すると次へ進めます";
        case 5: return "「編集を開始」で差分位置を確認し、「編集を完了」を押すと進めます";
        case 6: return null;
        default: return null;
      }
    };
    const phaseStatusLabel = error
      ? error
      : workspaceBusy ? "処理を実行しています"
        : status;

    const renderPrimaryAction = () => {
      if (workspaceStep === 1) {
        return workspaceFiles.source ? (
          <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="forward" onClick={() => setWorkspaceStep(2)}>次へ: 表情素材 <span className="workspace-nav-icon" aria-hidden="true">→</span></button>
        ) : (
          <button className="btn btn-secondary" data-action-tone="edit" onClick={() => void pickWorkspaceImage("source")}>立ち絵を選択</button>
        );
      }
      if (workspaceStep === 2) {
        if (!workspaceGeneratedStatus || workspaceCodexRequestDirty || !workspaceGeneratedStatus.ready) {
          return null;
        }
        return <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="forward" disabled={workspaceBusy} onClick={() => setWorkspaceStep(3)}>次へ: 一括分解 <span className="workspace-nav-icon" aria-hidden="true">→</span></button>;
      }
      if (workspaceStep === 3) {
        return step3Complete ? (
          <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="forward" disabled={workspaceBusy} onClick={() => setWorkspaceStep(4)}>次へ: パーツ編集 <span className="workspace-nav-icon" aria-hidden="true">→</span></button>
        ) : null;
      }
      if (workspaceStep === 4) {
        if (workspaceInlineEditor === "base") {
          return <button className="btn btn-primary" disabled={loading || workspaceBusy || !bodyPreview} onClick={() => void handleCreateBase()}>{loading ? "保存中..." : "保存して次へ"}</button>;
        }
        if (!step4Complete) {
          return null;
        }
        return <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="forward" disabled={workspaceBusy || workspaceOverviewPreviewLoading} onClick={() => setWorkspaceStep(5)}>次へ: 差分位置 <span className="workspace-nav-icon" aria-hidden="true">→</span></button>;
      }
      if (workspaceStep === 5) {
        if (workspaceInlineEditor === "position") {
          return <button className="btn btn-primary" disabled={workspaceBusy || workspaceEditorPreparing} onClick={() => void finishInlinePartEditor()}>{workspacePartSaving ? "変更を保存中..." : "編集を完了して次へ"}</button>;
        }
        if (!step5Confirmed) {
          return null;
        }
        return <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="forward" disabled={workspaceBusy || workspaceOverviewPreviewLoading} onClick={() => setWorkspaceStep(6)}>次へ: フレーム生成 <span className="workspace-nav-icon" aria-hidden="true">→</span></button>;
      }
      if (workspaceStep === 6) {
        return step6Current ? (
          <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="forward" disabled={workspaceBusy || workspaceOverviewPreviewLoading} onClick={() => setWorkspaceStep(7)}>次へ: モーション仕上げ <span className="workspace-nav-icon" aria-hidden="true">→</span></button>
        ) : null;
      }
      if (workspaceInlineEditor === "motion") {
        return (
          <div className="workspace-motion-export-actions">
            <button
              className="btn btn-secondary"
              data-action-tone="edit"
              disabled={workspaceBusy || workspaceEditorPreparing || motionEditorDraftSaveBusy || motionEditorExportBusy || !motionEditorExportReady}
              onClick={requestLiveDisplayFromMotionEditor}
            >ライブ表示へ</button>
            <button
              className="btn btn-primary"
              disabled={workspaceBusy || workspaceEditorPreparing || motionEditorDraftSaveBusy || motionEditorExportBusy || !motionEditorExportReady}
              onClick={() => {
                setMotionEditorExportBusy(true);
                setMotionExportRequestId(previous => previous + 1);
              }}
            >
              {motionEditorExportBusy ? "SpriTalk向け素材を保存中..." : "SpriTalk向けに保存"}
            </button>
          </div>
        );
      }
      return motionProfileReady ? (
        <button className="btn btn-secondary" data-action-tone="navigate" disabled={workspaceBusy || workspaceOverviewPreviewLoading} onClick={() => openPath(workspaceRifeResult?.outputPath || workspace.spritalkPartsPath).catch(() => {})}>素材フォルダを開く</button>
      ) : null;
    };

    return (
      <div className="workspace-hub">
        <div className="workspace-commandbar">
          <nav className="workspace-phase-stepper" aria-label="表情セット作成の7工程">
            {steps.map(([step, label, note]) => {
              const stepModel = phaseModel.substeps.find(item => item.step === step);
              const stepStatus = stepModel?.status ?? "locked";
              const isActive = workspaceStep === step;
              return (
                <button
                  key={step}
                  type="button"
                  data-step={step}
                  className={`workspace-phase-step ${stepStatus}${isActive ? " active" : ""}`}
                  disabled={!canOpenWorkspaceStep(step) || (!!workspaceInlineEditor && !isActive)}
                  aria-current={isActive ? "step" : undefined}
                  title={`STEP ${step}: ${label} — ${note}`}
                  onClick={() => selectWorkspaceStep(step)}
                >
                  <span>
                    <small className="workspace-phase-step-label">STEP {step}</small>
                    <strong>{label}</strong>
                  </span>
                </button>
              );
            })}
          </nav>
          <div className="workspace-toolbar" aria-label="ワークスペース操作">
            <button
              type="button"
              className="workspace-icon-button"
              title={themeMode === "dark" ? "ライトモードへ切り替え" : "ダークモードへ切り替え"}
              aria-label={themeMode === "dark" ? "ライトモードへ切り替え" : "ダークモードへ切り替え"}
              onClick={() => setThemeMode(previous => previous === "dark" ? "light" : "dark")}
            >
              <WorkspaceToolbarIcon name={themeMode === "dark" ? "sun" : "moon"} />
            </button>
            <button
              type="button"
              className="workspace-icon-button"
              title="制作ホームへ戻る"
              aria-label="制作ホームへ戻る"
              disabled={workspaceBusy || workspaceEditorPreparing || !!workspaceInlineEditor}
              onClick={returnToModeSelect}
            >
              <WorkspaceToolbarIcon name="home" />
            </button>
            <button
              type="button"
              className="workspace-icon-button"
              title={`作業フォルダを開く: ${workspace.workPath}`}
              aria-label="作業フォルダを開く"
              onClick={() => openPath(workspace.workPath).catch(() => {})}
            >
              <WorkspaceToolbarIcon name="folder" />
            </button>
          </div>
        </div>

        <div
          className={`workspace-hub-body${workspaceStep <= 3 || workspaceInlineEditor ? " single-panel" : ""}`}
        >
          <section className="workspace-phase-card workspace-flow-panel">
            <div className={`workspace-phase-content${workspaceInlineEditor ? " editing" : ""}`}>
            {workspaceStep === 1 && (
              <div className="workspace-step-one">
                <div className="workspace-panel-heading workspace-step-heading">
                  <span>STEP 1 / 7</span>
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
                          <em>同じキャラクターの別ポーズや口を開けた絵があると、<br />作成する目・口内の色や質感を元絵に近づけやすくなります。<br />なければ空のままで進めます。</em>
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
              // ①作成ガイド出力 → ②任意の方法で素材作成 → ③配置確認 の現在地
              const codexPhase = workspaceCodexRequestDirty ? 1 : workspaceGeneratedStatus?.ready ? 3 : workspaceGeneratedStatus ? 2 : 1;
              const mouthCornerDescription = WORKSPACE_MOUTH_CORNER_OPTIONS.find(option => option.value === workspaceMouthCorner)?.description ?? "";
              return (
              <div className="workspace-step-two">
                <div className="workspace-panel-heading workspace-step-heading">
                  <span>STEP 2 / 7</span>
                  <h3>表情素材を用意</h3>
                  <p>Codex向けに最適化した作成ガイドを使いますが、画像編集AIや手作業で用意した素材でも進められます。</p>
                </div>
                <div className="workspace-material-routes" role="group" aria-label="表情素材の用意方法">
                  <button
                    type="button"
                    className={workspaceAssetPreparationMethod === "codex" ? "active" : ""}
                    aria-pressed={workspaceAssetPreparationMethod === "codex"}
                    onClick={() => setWorkspaceAssetPreparationMethod("codex")}
                  >
                    <span>おすすめ</span>
                    <strong>Codexでまとめて作る</strong>
                    <small>作成ガイドと元画像をフォルダごと渡します。</small>
                  </button>
                  <button
                    type="button"
                    className={workspaceAssetPreparationMethod === "image-ai" ? "active" : ""}
                    aria-pressed={workspaceAssetPreparationMethod === "image-ai"}
                    onClick={() => setWorkspaceAssetPreparationMethod("image-ai")}
                  >
                    <span>画像編集AI</span>
                    <strong>Nano Bananaなどで作る</strong>
                    <small>7枚を個別に保存し、指定名で配置します。</small>
                  </button>
                  <button
                    type="button"
                    className={workspaceAssetPreparationMethod === "manual" ? "active" : ""}
                    aria-pressed={workspaceAssetPreparationMethod === "manual"}
                    onClick={() => setWorkspaceAssetPreparationMethod("manual")}
                  >
                    <span>手作業</span>
                    <strong>自分で用意する</strong>
                    <small>画像編集ソフト等で同じ仕様の7枚を作ります。</small>
                  </button>
                </div>
                <div className="workspace-material-route-advice" role="status">
                  {workspaceAssetPreparationMethod === "codex" && (
                    <><strong>Codexを使う場合</strong><span><code>01_codex_request</code> をフォルダごと渡せば、指示と保存先をまとめて伝えられます。</span></>
                  )}
                  {workspaceAssetPreparationMethod === "image-ai" && (
                    <><strong>画像編集AIを使う場合</strong><span>作成ガイドの共通ルールを使い、完成画像を指定ファイル名で <code>02_generated_parts</code> に保存してください。立ち絵とサイズが違っても縦横比が近ければ自動で合わせます。</span></>
                  )}
                  {workspaceAssetPreparationMethod === "manual" && (
                    <><strong>手作業で用意する場合</strong><span>透過は不要です。閉じ口・あいうえお・閉じ目の7枚を <code>02_generated_parts</code> に保存してください。立ち絵とサイズが違っても縦横比が近ければ自動で合わせます。</span></>
                  )}
                </div>
                <div className="workspace-mouth-corner-setting">
                  <label>
                    <span>口角の向き</span>
                    <select
                      aria-label="生成する口画像の口角"
                      value={workspaceMouthCorner}
                      disabled={workspaceBusy}
                      onChange={(event) => {
                        setWorkspaceMouthCorner(event.target.value as WorkspaceMouthCornerMode);
                        setWorkspaceCodexRequestDirty(true);
                      }}
                    >
                      {WORKSPACE_MOUTH_CORNER_OPTIONS.map(option => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                  <p>{mouthCornerDescription}<small>閉じ口＋あいうえおの6枚に共通適用。閉じ目画像の口は元画像のまま維持します。</small></p>
                </div>
                <details className="advanced-prompt">
                  <summary>素材作成への追加指示（任意）{workspaceCodexRequestDirty ? " — 作成ガイドへ未反映" : ""}</summary>
                  <textarea
                    aria-label="表情素材への追加指示"
                    value={workspaceCodexPrompt}
                    maxLength={2000}
                    disabled={workspaceBusy}
                    placeholder="例: このキャラクター固有の口内色を維持する。口以外の表情は元画像から変えない。"
                    onChange={(event) => {
                      setWorkspaceCodexPrompt(event.target.value);
                      setWorkspaceCodexRequestDirty(true);
                    }}
                  />
                  <p className="workspace-codex-prompt-note">
                    入力内容は作成ガイドの「追加指示」に入ります。共通の変更範囲や口形仕様と矛盾する場合は共通ルールが優先されます。
                    <span>{workspaceCodexPrompt.length}/2000</span>
                  </p>
                </details>
                <div className="workspace-codex-steps">
                  <section className={`workspace-codex-card${codexPhase === 1 ? " current" : ""}`}>
                    <div>
                      <span>1</span>
                      <strong>作成ガイドを出力 {codexPhase === 1 && <em className="workspace-phase-badge">いまここ</em>}{codexPhase > 1 && <em className="workspace-phase-done">✓ 済み</em>}</strong>
                      <p>PachiPakuGenが共通仕様・Codex向け指示・元画像を <code>01_codex_request</code> に書き出します。</p>
                      {(!workspaceGeneratedStatus || workspaceCodexRequestDirty) && (
                        <div className="workspace-action-row">
                          <button className="btn btn-primary" disabled={workspaceBusy || !workspaceFiles.source} onClick={() => void prepareWorkspaceCodexRequest()}>{workspaceGeneratedStatus ? "作成ガイドを更新" : "作成ガイドを出力"}</button>
                        </div>
                      )}
                    </div>
                  </section>
                  <section className={`workspace-codex-card${codexPhase === 2 ? " current" : ""}`}>
                    <div>
                      <span>2</span>
                      <strong>表情素材を作成（アプリ外） {codexPhase === 2 && <em className="workspace-phase-badge">いまここ</em>}{codexPhase > 2 && <em className="workspace-phase-done">✓ 済み</em>}</strong>
                      <p>
                        {workspaceAssetPreparationMethod === "codex"
                          ? <><code>01_codex_request</code> を<b>フォルダごと</b>Codexへ渡してください。生成物は通常 <code>02_generated_parts</code> へ自動配置されます。</>
                          : workspaceAssetPreparationMethod === "image-ai"
                            ? <>画像編集AIで7枚を作成し、ダウンロード後に指定名で <code>02_generated_parts</code> へ配置してください。</>
                            : <>作成ガイドを確認して7枚を編集し、指定名で <code>02_generated_parts</code> へ配置してください。</>}
                      </p>
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-secondary" data-action-tone="navigate" onClick={() => openPath(workspace.codexRequestPath).catch(() => {})}>作成ガイドを開く</button>
                      <button className="btn btn-secondary" data-action-tone="navigate" onClick={() => openPath(workspace.generatedPartsPath).catch(() => {})}>配置フォルダを開く</button>
                    </div>
                  </section>
                  <section className={`workspace-codex-card${codexPhase === 3 ? " current" : ""}`}>
                    <div>
                      <span>3</span>
                      <strong>配置を確認 {codexPhase === 3 ? <em className="workspace-phase-done">✓ 揃いました</em> : codexPhase === 2 ? <em className="workspace-phase-badge">自動確認中</em> : null}</strong>
                      <p>
                        {workspaceCodexRequestDirty
                          ? "口角設定または追加指示を反映するため、作成ガイドを更新してください。"
                          : workspaceGeneratedStatus?.ready
                          ? "すべて揃いました。「次へ」で進んでください。"
                          : codexPhase === 2
                            ? `5秒ごとに自動確認中（${workspaceGeneratedStatus?.presentParts.length ?? 0}/${workspaceGeneratedStatus?.expectedParts.length ?? 7}）`
                            : "作成ガイド出力後に一覧が表示されます。"}
                      </p>
                      {workspaceGeneratedStatus && (
                        <div className="workspace-parts-checklist">
                          {workspaceGeneratedStatus.expectedParts.map(part => {
                            const present = workspaceGeneratedStatus.presentParts.includes(part);
                            const stale = workspaceGeneratedStatus.staleParts?.includes(part) ?? false;
                            const mismatch = workspaceGeneratedStatus.sizeMismatches.some(item => item.startsWith(`${part}.png:`));
                            const autoFit = workspaceGeneratedStatus.autoFitParts?.includes(part) ?? false;
                            return (
                              <span key={part} className={mismatch ? "mismatch" : stale ? "stale" : present ? "present" : "missing"} title={mismatch ? "縦横比が立ち絵と大きく異なります。立ち絵に近い縦横比で再生成してください" : stale ? "依頼書の設定変更後に再生成してください" : autoFit ? "サイズが立ち絵と異なるため、分解時に自動で立ち絵サイズへ合わせます" : present ? "配置済み" : "未配置"}>
                                <b>{mismatch ? "⚠" : stale ? "↻" : autoFit ? "⤢" : present ? "✓" : "・"}</b>{part}
                              </span>
                            );
                          })}
                        </div>
                      )}
                      {workspaceGeneratedStatus && !workspaceGeneratedStatus.ready && !workspaceCodexRequestDirty && (
                        <div className="workspace-action-row">
                          <button className="btn btn-secondary" data-action-tone="edit" disabled={workspaceBusy} onClick={() => void inspectWorkspaceGeneratedParts()}>配置を再確認</button>
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </div>
              );
            })()}

            {workspaceStep === 3 && (
              <>
                <div className="workspace-step3-header">
                  <div className="workspace-panel-heading workspace-step-heading">
                    <span>STEP 3 / 7</span>
                    <h3>See-Throughを一括実行</h3>
                    <p>立ち絵と用意した表情素材をSee-Throughで分解します。</p>
                  </div>
                  <div className="workspace-step3-env">
                    <div className="workspace-action-row">
                      <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => void refreshWorkspaceSeeThroughStatus()}>環境を再確認</button>
                      {selectedProfileReady ? (
                        <span className="workspace-action-done">
                          セットアップ済み（{seeThroughRuntime.selectedProfile === "low-vram" ? "省VRAM" : "高VRAM"}）
                        </span>
                      ) : null}
                      {workspaceBusy && seeThroughProgress?.stage === "prepare" && (
                        <button className="btn btn-secondary" onClick={() => void cancelWorkspaceSeeThroughSetup()}>
                          ランタイム構築を中止
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                <div className="workspace-step3-main">
                  <div className="workspace-setup-steps" aria-label="See-Throughの事前準備">
                    <div className={`workspace-process-card${!seeThroughRuntime?.runtimeReady ? " primary" : ""}`}>
                      <div>
                        <span>準備 1/2</span>
                        <strong>ランタイム初期セットアップ</strong>
                        <p>See-Through本体、専用Python環境、CUDA依存関係だけをアプリ内で準備します。大容量モデルはまだ取得しません。</p>
                      </div>
                      <div className="workspace-action-row">
                        {seeThroughRuntime?.runtimeReady ? (
                          <span className="workspace-action-done">準備済み</span>
                        ) : (
                          <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => void prepareWorkspaceSeeThroughRuntime()}>
                            ランタイムをセットアップ
                          </button>
                        )}
                      </div>
                    </div>
                    <div className={`workspace-process-card${seeThroughRuntime?.runtimeReady && !selectedProfileReady ? " primary" : ""}`}>
                      <div>
                        <span>準備 2/2</span>
                        <strong>モデルを事前ダウンロード</strong>
                        <p>
                          選択中の{seeThroughProfile === "low-vram" ? "省VRAM" : "高VRAM"}モデル
                          （約{seeThroughProfile === "low-vram" ? "5.7" : "13.4"}GB）を別コンソールで取得します。実バイト進捗を確認でき、ウィンドウを閉じても再実行時に途中から再開します。
                        </p>
                      </div>
                      <div className="workspace-action-row">
                        {selectedProfileReady ? (
                          <span className="workspace-action-done">モデル準備済み</span>
                        ) : seeThroughRuntime?.modelDownloadBusy ? (
                          <span className="workspace-action-done">別コンソールで取得中</span>
                        ) : (
                          <button
                            className="btn btn-secondary"
                            disabled={workspaceBusy || seeThroughModelDownloadLaunching || !seeThroughRuntime?.runtimeReady}
                            onClick={() => void startWorkspaceSeeThroughModelDownload()}
                          >
                            {seeThroughModelDownloadLaunching ? "コンソールを起動中..." : "モデルDL用コンソールを開く"}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                  {seeThroughRuntime && !selectedProfileReady && (
                    <div className={`motion-lab-note${seeThroughRuntime.runtimeReady ? " workspace-setup-error" : ""}`} role={seeThroughRuntime.runtimeReady ? "alert" : undefined}>
                      <strong>{seeThroughRuntime.runtimeReady ? "モデル未準備:" : "ランタイム未準備:"}</strong> {seeThroughRuntime.message}
                    </div>
                  )}
                  {hfTokenStatus?.configured === false && !selectedProfileReady && (
                    <div className="motion-lab-note" role="status">
                      別コンソールでの大容量モデル取得ではHuggingFaceトークンの設定を推奨します。匿名取得はレート制限で遅くなるため、下の「詳細設定」で無料のreadトークンを保存してから開始してください。
                    </div>
                  )}
                  <div className="workspace-seethrough-options">
                  <div className="workspace-option-card">
                    <span>実行プロファイル</span>
                    <div className="workspace-segmented">
                      <button
                        className={seeThroughProfile === "low-vram" ? "active" : ""}
                        disabled={workspaceBusy || !!seeThroughRuntime?.modelDownloadBusy}
                        title="目安: VRAM 8GB級でも動くようモデルを退避しながら実行します（低速）"
                        onClick={() => {
                          seeThroughProfileTouched.current = true;
                          setSeeThroughProfile("low-vram");
                          void refreshWorkspaceSeeThroughStatus("low-vram");
                        }}
                      >
                        省VRAM{seeThroughRuntime?.recommendedProfile === "low-vram" && <em className="workspace-recommend-badge">推奨</em>}
                      </button>
                      <button
                        className={seeThroughProfile === "standard" ? "active" : ""}
                        disabled={workspaceBusy || !!seeThroughRuntime?.modelDownloadBusy}
                        title="目安: VRAM 16GB以上のGPU向け。退避なしで最速です"
                        onClick={() => {
                          seeThroughProfileTouched.current = true;
                          setSeeThroughProfile("standard");
                          void refreshWorkspaceSeeThroughStatus("standard");
                        }}
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
                          disabled={workspaceBusy || !!seeThroughRuntime?.modelDownloadBusy}
                          onChange={(event) => {
                            const value = event.target.value === "auto" ? null : Number(event.target.value);
                            void changeSeeThroughGpuSelection(value);
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
                      <label title="深度（前後関係）推定の処理解像度。-1で自動、それ以外は256〜4096の64倍数。レイヤー前後の判定が怪しい時に上げます"><span>Depth解像度 <i className="workspace-info-mark">?</i></span><input type="number" min={-1} max={4096} step={1} value={seeThroughOptions.resolutionDepth} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, resolutionDepth: Number(event.target.value) })} /></label>
                      <label title="レイヤー分解の推論ステップ数。多いほど品質が上がりますが遅くなります（既定30）"><span>LayerDiff step <i className="workspace-info-mark">?</i></span><input type="number" min={1} max={150} value={seeThroughOptions.inferenceSteps} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, inferenceSteps: Number(event.target.value) })} /></label>
                      <label title="深度推定のステップ数。-1で自動。省VRAMプロファイルでは固定のため変更できません"><span>Depth step <i className="workspace-info-mark">?</i></span><input type="number" min={-1} max={150} value={seeThroughOptions.inferenceStepsDepth} disabled={workspaceBusy || seeThroughProfile === "low-vram"} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, inferenceStepsDepth: Number(event.target.value) })} /></label>
                      <label title="モデルをブロック単位でCPUメモリへ退避してVRAMを節約します（少し低速）。「自動」はプロファイルの既定動作に任せます。VRAM不足エラーが出る時に有効化"><span>Group offload <i className="workspace-info-mark">?</i></span><select value={seeThroughOptions.groupOffload} disabled={workspaceBusy} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, groupOffload: event.target.value as SeeThroughOptionMode })}><option value="default">自動（推奨）</option><option value="on">有効</option><option value="off">無効</option></select></label>
                      <label title="大型UNetをCPUへ退避する互換設定（低速）。本家のカスタムVAEはGPUに残るため、処理条件によってはピークVRAMが減らないことがあります。有効時はGroup offloadより優先します。通常は自動を推奨します"><span>CPU offload <i className="workspace-info-mark">?</i></span><select value={seeThroughOptions.cpuOffload} disabled={workspaceBusy || seeThroughProfile === "standard"} onChange={(event) => setSeeThroughOptions({ ...seeThroughOptions, cpuOffload: event.target.value as SeeThroughOptionMode })}><option value="default">自動（推奨）</option><option value="on">有効（互換設定）</option><option value="off">無効</option></select></label>
                    </div>
                    {/* インストール先とHFトークンはモデルセットアップ時の設定。
                        セットアップ完了後は変更する場面がないため、未セットアップ時のみ表示する */}
                    {!selectedProfileReady && (
                      <>
                        <div className="workspace-option-header">
                          <span>インストール先（Python環境+選択モデル、省VRAM約14GB / 高VRAM約22GB）</span>
                        </div>
                        <div className="workspace-install-location">
                          <small title={seeThroughInstallLocation?.path} className="workspace-install-location-path">
                            {seeThroughInstallLocation?.path ?? "取得中..."}
                            {seeThroughInstallLocation?.isDefault && <em className="workspace-recommend-badge">既定</em>}
                          </small>
                          <button className="btn btn-secondary" disabled={workspaceBusy || !!seeThroughRuntime?.modelDownloadBusy} onClick={() => void changeSeeThroughInstallLocation()}>変更...</button>
                          {seeThroughInstallLocation && !seeThroughInstallLocation.isDefault && (
                            <button className="btn btn-secondary" disabled={workspaceBusy || !!seeThroughRuntime?.modelDownloadBusy} onClick={() => void resetSeeThroughInstallLocation()}>既定に戻す</button>
                          )}
                        </div>
                        <div className="motion-lab-note">
                          そのままで良ければ変更不要です。C:ドライブの空き容量が少ない場合に、大きな空きのあるドライブへ変更できます。変更すると新しい場所でランタイム構築とモデル事前ダウンロードがそれぞれ必要です。
                        </div>
                        <div className="workspace-option-header">
                          <span>HuggingFaceトークン（初回モデルDLで推奨・任意）</span>
                        </div>
                        <div className="workspace-install-location">
                          {hfTokenStatus?.configured ? (
                            <>
                              <small className="workspace-install-location-path">設定済み（huggingface.co/settings/tokens で発行したトークン）</small>
                              <button className="btn btn-secondary" disabled={workspaceBusy} onClick={() => void deleteHfToken()}>削除</button>
                            </>
                          ) : (
                            <>
                              <input
                                type="password"
                                value={hfTokenInput}
                                disabled={workspaceBusy}
                                placeholder="hf_ から始まるトークンを貼り付け（使わない場合は空のまま）"
                                onChange={(event) => setHfTokenInput(event.target.value)}
                              />
                              <button className="btn btn-secondary" disabled={workspaceBusy || !hfTokenInput.trim()} onClick={() => void saveHfToken()}>保存</button>
                            </>
                          )}
                        </div>
                        <div className="motion-lab-note">
                          設定しなくても進めます（スキップ可）。匿名取得はHuggingFace側のレート制限で低速になるため、huggingface.co/settings/tokens で無料のreadトークンを発行して貼り付けると高速になります。保存済みトークンはコンソールの環境変数にだけ渡され、コマンド行や画面には表示されません。
                        </div>
                      </>
                    )}
                  </details>
                  </div>
                  <div className="workspace-process-card workspace-probe-card">
                    <div>
                      <strong>抽出ガチャ（獣耳・眼鏡などの高速確認）</strong>
                      <p>立ち絵1枚をレイヤー分解だけで処理し（深度推定・PSD組立・表情素材の分解を省略）、獣耳や眼鏡などSeedによって取れたり取れなかったりするパーツの抽出結果をサムネイルで確認します。良い結果が出たら、そのSeedのまま一括分解を開始してください。</p>
                      {seeThroughLayerProbe && (
                        <div className="workspace-probe-result">
                          <small>Seed {seeThroughOptions.seed} で採れたパーツ:</small>
                          <div className="workspace-probe-grid">
                            {seeThroughLayerProbe.layers.map(layer => (
                              <figure
                                key={layer.name}
                                className={/^(ears|earwear|headwear|eyewear)$/.test(layer.name) ? "probe-layer-highlight" : ""}
                                role="button"
                                tabIndex={0}
                                title="クリックで拡大表示"
                                onClick={() => setSeeThroughProbeZoom({ name: layer.name, thumbnail: layer.thumbnail })}
                                onKeyDown={(event) => {
                                  if (event.key === "Enter" || event.key === " ") {
                                    event.preventDefault();
                                    setSeeThroughProbeZoom({ name: layer.name, thumbnail: layer.thumbnail });
                                  }
                                }}
                              >
                                <img src={layer.thumbnail} alt={layer.name} loading="lazy" />
                                <figcaption>{layer.name}</figcaption>
                              </figure>
                            ))}
                          </div>
                          {!seeThroughLayerProbe.layers.some(layer => /^(ears|earwear|headwear|eyewear)$/.test(layer.name)) && (
                            <div className="motion-lab-note" role="status">獣耳・眼鏡系のレイヤーは抽出されませんでした。「Seedを変えて確認」で引き直せます。何度も出ない場合は、詳細設定でLayerDiff解像度やstep数を上げると取れやすくなることがあります。</div>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="workspace-action-row">
                      <button className="btn btn-secondary" disabled={workspaceBusy || !selectedProfileReady || !workspaceFiles.source} onClick={() => void probeWorkspaceSeeThroughLayers(false)}>
                        {seeThroughLayerProbeRunning ? "確認中..." : "このSeedで確認"}
                      </button>
                      <button className="btn btn-secondary" disabled={workspaceBusy || !selectedProfileReady || !workspaceFiles.source} onClick={() => void probeWorkspaceSeeThroughLayers(true)}>
                        Seedを変えて確認
                      </button>
                    </div>
                  </div>
                  {seeThroughProbeZoom && (
                    <div
                      className="workspace-probe-zoom-overlay"
                      role="dialog"
                      aria-label={`${seeThroughProbeZoom.name} の拡大表示`}
                      onClick={() => setSeeThroughProbeZoom(null)}
                    >
                      <figure onClick={(event) => event.stopPropagation()}>
                        <img src={seeThroughProbeZoom.thumbnail} alt={seeThroughProbeZoom.name} />
                        <figcaption>
                          <strong>{seeThroughProbeZoom.name}</strong>
                          <button className="btn btn-secondary" onClick={() => setSeeThroughProbeZoom(null)}>閉じる (Esc)</button>
                        </figcaption>
                      </figure>
                    </div>
                  )}
                  <div className="workspace-start-seethrough">
                    <div>
                      <strong>{step3Complete ? "✓ 一括分解は完了しています" : "分解処理を開始"}</strong>
                      <p>{step3Complete ? "問題なければ下部の「次へ」でパーツ編集へ進めます。設定を変えて作り直すこともできます。" : "立ち絵、閉じ目、閉じ口、あいうえお口の素材をまとめて分解します。"}</p>
                    </div>
                    <button className="btn btn-primary" disabled={workspaceBusy || !selectedProfileReady || !workspaceGeneratedStatus?.ready} onClick={() => void runWorkspaceSeeThroughBatch()}>{seeThroughRunning ? "分解処理中..." : step3Complete ? "一括分解を再実行" : "一括分解を開始"}</button>
                  </div>
                </div>
              </>
            )}

            {workspaceStep === 4 && (
              workspaceInlineEditor === "base" ? (
                workspaceEditorPreparing ? (
                  <div className="workspace-editor-loading" role="status" aria-live="polite">
                    <span className="step-spinner" aria-hidden="true" />
                    <strong>素体編集を準備しています</strong>
                    <small>{status}</small>
                  </div>
                ) : (
                  <div className="workspace-inline-editor base-editor-inline">{renderLayerEditor()}</div>
                )
              ) : (
                <div className="workspace-studio-overview">
                  <div className="workspace-studio-overview-main copy-only">
                    <div className="workspace-panel-heading workspace-step-heading">
                      <span>STEP 4 / 7</span>
                      <h3>素体のレイヤー構成を調整</h3>
                      <p>レイヤー順・表示・腕や獣耳の分離・切り出しを、中央の編集領域でまとめて確認します。データは「編集を開始」を押してから読み込みます。</p>
                      <button className="btn btn-secondary workspace-wide-action" data-action-tone="edit" disabled={workspaceBusy || workspaceOverviewPreviewLoading || workspaceEditorPreparing} onClick={() => void startInlineBaseEditor()}>{workspaceEditorPreparing ? "準備中..." : step4Complete ? "再編集する" : "編集を開始"}</button>
                    </div>
                  </div>
                  <div className={`workspace-status-card${step4Complete ? " complete" : ""}`}>
                    <strong>{step4Complete ? "✓ 素体は保存済みです" : "素体編集はまだ完了していません"}</strong>
                    <span>{step4Complete ? "問題なければ下部の「次へ」で差分位置へ進めます。" : "上部の「編集を開始」から調整し、保存してください。"}</span>
                  </div>
                </div>
              )
            )}

            {workspaceStep === 5 && (
              workspaceInlineEditor === "position" ? (
                workspaceEditorPreparing ? (
                  <div className="workspace-editor-loading" role="status" aria-live="polite">
                    <span className="step-spinner" aria-hidden="true" />
                    <strong>表情プレビューを準備しています</strong>
                    <small>この編集を開いた時だけ、全表情を合成して読み込みます。</small>
                  </div>
                ) : (
                  <div className="workspace-inline-editor position-editor-inline">{renderPartPositionEditor()}</div>
                )
              ) : (
                <div className="workspace-studio-overview">
                  <div className="workspace-studio-overview-main copy-only">
                    <div className="workspace-panel-heading workspace-step-heading">
                      <span>STEP 5 / 7</span>
                      <h3>差分パーツの位置を確認</h3>
                      <p>自動整列後の閉じ目と口を確認し、ずれがあるパーツだけ微調整します。全表情の合成は「編集を開始」を押した時だけ行います。</p>
                      <button className="btn btn-secondary workspace-wide-action" data-action-tone="edit" disabled={workspaceBusy || workspaceOverviewPreviewLoading || workspaceEditorPreparing || !step4Complete} onClick={() => void startInlinePositionEditor()}>{workspaceEditorPreparing ? "準備中..." : step5Confirmed ? "再編集する" : "編集を開始"}</button>
                    </div>
                  </div>
                  <div className={`workspace-status-card${step5Confirmed ? " complete" : ""}`}>
                    <strong>{step5Confirmed ? "✓ 差分位置は確認済みです" : "差分位置の確認が必要です"}</strong>
                    <span>{step5Confirmed ? "問題なければ下部の「次へ」でRIFE補完へ進めます。" : "上部の「編集を開始」から全表情を確認してください。"}</span>
                  </div>
                </div>
              )
            )}

            {workspaceStep === 6 && (
              <>
                <div className="workspace-panel-heading workspace-step-heading"><span>STEP 6 / 7</span><h3>RIFE補完してSpriTalk用に出力</h3><p>RIFEフレーム、素体、抽出差分を <code>04_spritalk_parts</code> にまとめます。完了するとモーション仕上げへ進めます。</p></div>
                <div className="workspace-rife-panel">
                  <div className="workspace-frame-slider">
                    <label><span>補間枚数</span><strong>{frameCount}枚</strong></label>
                    <input type="range" min={RIFE_FRAME_MIN} max={RIFE_FRAME_MAX} value={frameCount} disabled={workspaceBusy} onChange={e => setFrameCount(Number(e.target.value))} />
                    <small>多いほど口パク・まばたきが滑らかになりますが、ファイル数と生成時間が増えます。SpriTalkでは {RIFE_FRAME_RECOMMENDED} 枚が扱いやすい目安です。</small>
                  </div>
                  {!step6Current && (
                    <button
                      className="btn btn-primary workspace-wide-action"
                      disabled={workspaceBusy || workspaceOverviewPreviewLoading || !step5Confirmed}
                      onClick={() => void generateWorkspaceRifeOutputs()}
                    >
                      {workspaceBusy ? "RIFE補完中..." : workspaceRifeResult ? "RIFE補完を再実行" : "RIFE補完を開始"}
                    </button>
                  )}
                  {!workspaceBusy && step6Current && workspaceRifeResult && (
                    <div className="workspace-status-card complete">
                      <strong>✓ RIFE補完は完了しています</strong>
                      <span>出力先: <code>{workspaceRifeResult.outputPath}</code></span>
                      <span>{workspaceRifeResult.directories.length}ディレクトリ / 各{workspaceRifeResult.frameCount}フレーム</span>
                      <span>「次へ: モーション仕上げ」で揺れ・口パクの調整へ進めます。枚数を変えて再生成することもできます。</span>
                    </div>
                  )}
                </div>
              </>
            )}
            {workspaceStep === 7 && (
              workspaceInlineEditor === "motion" ? (
                <div className="workspace-inline-editor motion-editor-inline workspace-motion-tune">
                  <MotionTunePanel
                    key={workspaceRifeResult?.outputPath || workspace.spritalkPartsPath}
                    partsDir={workspaceRifeResult?.outputPath || workspace.spritalkPartsPath}
                    active
                    onNotify={handleInlineMotionNotify}
                    onError={message => {
                      setError(message);
                      setMotionEditorDraftSaveBusy(false);
                      setMotionEditorExportBusy(false);
                    }}
                    onDirtyChange={handleInlineMotionDirtyChange}
                    draftSaveRequestId={motionDraftSaveRequestId}
                    onDraftSaveStateChange={setMotionEditorDraftSaveBusy}
                    onDraftSaved={handleInlineMotionDraftSaved}
                    exportRequestId={motionExportRequestId}
                    onExportStateChange={handleMotionExportStateChange}
                    onExported={handleInlineMotionExported}
                  />
                </div>
              ) : (
                <div className="workspace-studio-overview motion-studio-overview">
                  <div className="workspace-studio-overview-main copy-only">
                    <div className="workspace-panel-heading workspace-step-heading">
                      <span>STEP 7 / 7</span>
                      <h3>モーションを仕上げる</h3>
                      <p>揺れ・口パクを調整し、SpriTalk向け画像素材とPachiPakuGen用モーション設定を保存するか、そのままマイク連動のライブ表示で使用します。</p>
                      <div className="workspace-step7-action-stack">
                        <button
                          className="btn btn-secondary workspace-wide-action"
                          data-action-tone="edit"
                          disabled={workspaceBusy || workspaceOverviewPreviewLoading || !workspaceRifeResult}
                          onClick={startInlineMotionEditor}
                        >
                          {motionProfileReady ? "再編集する" : "編集を開始"}
                        </button>
                        <button
                          className="btn btn-secondary workspace-wide-action"
                          data-action-tone="navigate"
                          disabled={workspaceBusy || workspaceOverviewPreviewLoading || !workspaceRifeResult}
                          onClick={openLiveDisplayFromWorkspace}
                        >ライブ表示へ</button>
                      </div>
                    </div>
                  </div>
                  <div className={`workspace-status-card${motionProfileReady ? " complete" : ""}`}>
                    <strong>{motionProfileReady ? "✓ 素材とモーション設定を保存しました" : "モーション設定はまだ保存されていません"}</strong>
                    <span>PachiPakuGen用モーション設定: {motionProfileReady ? "保存済み" : "未保存"}</span>
                    <span className="workspace-spritalk-handoff-note">
                      {motionProfileReady
                        ? "SpriTalkへ基本画像を取り込むときは次のフォルダを使います。モーション設定v2はPachiPakuGenのライブ表示・将来連携用で、現行SpriTalkでは読み込みません。"
                        : "編集画面で「SpriTalk向けに保存」すると、基本画像とPachiPakuGen用モーション設定を同じフォルダへまとめます。"}
                    </span>
                    {motionProfileReady && (
                      <div className="workspace-spritalk-handoff">
                        <code title={workspaceRifeResult?.outputPath || workspace.spritalkPartsPath}>
                          {workspaceRifeResult?.outputPath || workspace.spritalkPartsPath}
                        </code>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          data-action-tone="navigate"
                          onClick={() => openPath(workspaceRifeResult?.outputPath || workspace.spritalkPartsPath).catch(() => {})}
                        >素材フォルダを開く</button>
                      </div>
                    )}
                    <span>ライブ表示は保存の有無にかかわらず、現在のモーション設定を読み込んで使用できます。</span>
                  </div>
                </div>
              )
            )}
            </div>
          </section>

          {!workspaceInlineEditor && workspaceStep >= 4 && (
            <aside className="workspace-hub-preview workspace-flow-preview">
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
                ) : (
                  <div className="workspace-expression-preview-empty">
                    {workspaceOverviewPreviewLoading ? (
                      <>
                        <span className="step-spinner" aria-hidden="true" />
                        <strong>表情プレビューを読み込んでいます</strong>
                      </>
                    ) : step4Complete ? (
                      <>
                        <strong>目・口を合成した表情を確認できます</strong>
                        <button className="btn btn-secondary" data-action-tone="edit" type="button" disabled={workspaceBusy || workspaceEditorPreparing} onClick={() => void loadWorkspaceOverviewPreviews()}>
                          表情プレビューを読み込む
                        </button>
                      </>
                    ) : (
                      <strong>素体編集を保存すると表情を確認できます</strong>
                    )}
                  </div>
                )}
              </div>
              {overviewPreviewItems.length > 0 ? (
                <div className="workspace-preview-list workspace-expression-preview-list" aria-label="表示する表情">
                  {overviewPreviewItems.map(item => {
                    const key = workspacePreviewItemKey(item);
                    const active = selectedOverviewPreview?.part === key;
                    return (
                      <button
                        type="button"
                        key={key}
                        className={active ? "active" : ""}
                        aria-pressed={active}
                        onClick={() => {
                          setWorkspaceOverviewPreviewPart(key);
                          resetWorkspacePreviewZoom();
                        }}
                      >
                        <span>{workspacePreviewItemLabel(item)}</span>
                        <img src={item.preview} alt="" />
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="workspace-expression-preview-hint">
                  {sourcePreviewBeforeBaseSave
                    ? "保存後は、目・口を合成した8表情から選択できます。"
                    : "目・口を合成した表情だけを表示します。"}
                </div>
              )}
            </aside>
          )}
        </div>

        <footer className="workspace-hub-footer">
          <button className="btn btn-secondary" data-action-tone="navigate" data-nav-direction="back" disabled={workspaceBusy || workspaceEditorPreparing || motionEditorDraftSaveBusy || motionEditorExportBusy} onClick={goPreviousWorkspaceStep}>
            <span className="workspace-nav-icon" aria-hidden="true">←</span>
            {workspaceInlineEditor ? "編集概要へ戻る" : workspaceStep === 1 ? "制作ホームへ戻る" : `戻る: ${steps[workspaceStep - 2]?.[1] ?? ""}`}
          </button>
          <div className={`workspace-hub-status${error ? " error" : workspaceBusy || workspaceEditorPreparing || motionEditorDraftSaveBusy || motionEditorExportBusy ? " running" : currentStepModel?.status === "complete" ? " complete" : ""}`} aria-live="polite">
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
            {!seeThroughRunning && !(workspaceBusy && progress.total > 0) && (
              <span>{motionEditorDraftSaveBusy
                ? "変更を保存しています..."
                : motionEditorExportBusy
                  ? "SpriTalk向け素材を保存しています..."
                  : phaseStatusLabel}</span>
            )}
          </div>
          <div className="workspace-next-area">
            {nextStepBlockReason() && <small className="workspace-next-hint">{nextStepBlockReason()}</small>}
            {renderPrimaryAction()}
          </div>
          <details className="workspace-hub-log workspace-log-history" aria-live="polite">
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
        </footer>
      </div>
    );
  }
  function renderModeSelect() {
    return (
      <main className="mode-select-screen">
        <section className="workspace-start-panel">
          <button className="primary-workflow-card" disabled={workspaceBusy} onClick={() => void startExpressionWorkspace("new")}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">NEW WORKSPACE</span>
              <strong>はじめから</strong>
              <p>空のフォルダを作業フォルダとして選び、7つのSTEPでSpriTalk向けの表情・モーション素材を作成します。</p>
            </div>
            <span className="primary-workflow-cta">作業フォルダを選択</span>
          </button>
          <button className="primary-workflow-card secondary" disabled={workspaceBusy} onClick={() => void startExpressionWorkspace("resume")}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">RESUME</span>
              <strong>つづきから</strong>
              <p>「はじめから」で使った作業フォルダ（<code>project.json</code> のあるフォルダ）を選ぶと、前回の工程から再開します。</p>
            </div>
            <span className="primary-workflow-cta">作業フォルダを開く</span>
          </button>
          <button className="primary-workflow-card live-entry" disabled={workspaceBusy} onClick={() => void startLiveDisplayFromMenu()}>
            <div className="primary-workflow-copy">
              <span className="workflow-kicker">LIVE DISPLAY</span>
              <strong>ライブ表示</strong>
              <p>完成した作業フォルダを読み込み、マイクに反応するキャラクターをOBS向けに表示します。</p>
            </div>
            <span className="primary-workflow-cta">作業フォルダを選択</span>
          </button>
        </section>
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
    // 獣耳分離オプション: 明示的な ears(-l/-r) を優先する。ears* が無い素材でのみ
    // headwear（犬耳・獣耳がheadwear扱いのキャラ）をフォールバックとして使う。
    const explicitEarSplitLayers = layerOrder.filter(name => /^ears([-_][lr])?$/i.test(name));
    const headwearEarFallbackLayers = layerOrder.filter(name => /^headwear([-_][lr])?$/i.test(name));
    const earSplitUsesHeadwearFallback = explicitEarSplitLayers.length === 0;
    const earSplitLayers = earSplitUsesHeadwearFallback ? headwearEarFallbackLayers : explicitEarSplitLayers;
    const earSplitActive = earSplitLayers.some(name => (layerMapping[name] ?? "").startsWith("sway_"));
    const toggleEarSplit = (enable: boolean) => {
      setLayerMapping(prev => {
        const next = { ...prev };
        if (!earSplitUsesHeadwearFallback) {
          // 旧版で獣耳と一緒に sway_* へ入った髪飾りを、切替操作時にも確実に戻す。
          for (const name of headwearEarFallbackLayers) {
            if ((next[name] ?? "").startsWith("sway_")) next[name] = "hair";
          }
        }
        for (const name of earSplitLayers) {
          if (enable) {
            // 左右サフィックス無し（ears/headwearの一枚もの）は sway_ear として出力
            next[name] = /[-_]r$/i.test(name) ? "sway_ear_r" : /[-_]l$/i.test(name) ? "sway_ear_l" : "sway_ear";
          } else {
            delete next[name];
          }
        }
        return next;
      });
    };
    return (
      <div className="panel-right base-edit-panel">
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
                    <label className="layer-arm-split-toggle" title="腕レイヤー（handwear-l/-r）を分離出力します。レイヤー一覧で topwear より上なら体の前、下なら体の後ろに描画されます。">
                      <input type="checkbox" checked={armSplitActive} onChange={(e) => toggleArmSplit(e.target.checked)} />
                      <span>腕</span>
                    </label>
                  )}
                  {earSplitLayers.length > 0 && (
                    <label
                      className="layer-arm-split-toggle"
                      title={earSplitUsesHeadwearFallback
                        ? "earsレイヤーが無いため、headwearを sway_ear*.png として分離出力します。獣耳ピコピコ用"
                        : "獣耳レイヤー（ears-l/-r）を sway_ear*.png として分離出力します。獣耳ピコピコ用"}
                    >
                      <input type="checkbox" checked={earSplitActive} onChange={(e) => toggleEarSplit(e.target.checked)} />
                      <span>獣耳</span>
                    </label>
                  )}
                  <button
                    className={`btn-layer-bulk${chestMaskDataUrl ? " active" : ""}`}
                    title="胸部追従の変形範囲をbody上で指定します。未指定でも上半身から自動推定します"
                    onClick={initChestCut}
                  >
                    胸部範囲{chestMaskDataUrl ? "設定済" : ""}
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
                    {patchDraftSource === CHEST_CUT_SENTINEL ? "胸部の変形範囲" : `切り出し作成: ${patchDraftSource}`}
                  </div>
                  <div className="layer-adjust-values">
                    {patchDraftSource === CHEST_CUT_SENTINEL
                      ? "bodyのプレビュー上で、呼吸に合わせてやわらかく変形させる胸部を塗ってください。"
                      : "塗った範囲を別レイヤーとして切り出します。腕から切り出したパーツは腕の動きに追従したまま、一覧の前後関係を保持します。"}
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
      </div>
    );
  }

  function renderMainContent() {
    if (mode === "select") return renderModeSelect();
    if (mode === "workspace") return renderWorkspaceMode();
    if (mode === "live") return (
      <MotionLiveView
        partsDir={livePartsDir}
        onBack={returnFromLiveDisplay}
        onError={setError}
        onNotify={setStatus}
      />
    );
    return null;
  }

  function hasPendingPartAdjustment() {
    return workspacePartAdjustmentDraftsDiffer(
      workspacePartEditorBaseline.current,
      workspacePartDraftsRef.current,
    );
  }

  async function finishInlinePartEditor() {
    if (!expressionWorkspace || !workspaceCompositePreview?.basePreview) return;
    const draftSnapshot = cloneWorkspacePartAdjustmentDrafts(workspacePartDraftsRef.current);
    const editorChanged = workspacePartAdjustmentDraftsDiffer(
      workspacePartEditorBaseline.current,
      draftSnapshot,
    );
    const reachedStep = expressionWorkspace.project.currentStep;
    setError("");
    setWorkspaceBusy(true);
    setWorkspacePartSaving(editorChanged);
    try {
      if (editorChanged) await persistWorkspacePartDrafts(draftSnapshot);
      if (editorChanged || workspacePartPersistedDuringEditor.current) {
        await refreshWorkspaceCompositePreview();
      }
      // 完了済み案件を確認しただけならSTEP7と既存RIFEを維持する。
      // 初回確認または実際に位置を変更した場合だけSTEP6へ戻して再生成を要求する。
      if (editorChanged || workspacePartPersistedDuringEditor.current || reachedStep < 6) {
        await setWorkspaceStepAfterEdit(6);
      } else {
        setWorkspaceStep(Math.min(Math.max(reachedStep, 6), 7) as WorkspaceStep);
      }
      workspacePartEditorBaseline.current = cloneWorkspacePartAdjustmentDrafts(draftSnapshot);
      workspacePartPersistedDuringEditor.current = false;
      setWorkspaceInlineEditor(null);
      setStatus("パーツ編集を完了しました。フレーム生成へ進めます");
    } catch (cause) {
      await reloadWorkspaceAfterMutationFailure(expressionWorkspace.workPath);
      setError(String(cause));
    } finally {
      setWorkspacePartSaving(false);
      setWorkspaceBusy(false);
    }
  }

  function workspaceBaseEditorSignature() {
    const sortedRecord = <T,>(record: Record<string, T>) => Object.fromEntries(
      Object.entries(record).sort(([left], [right]) => left.localeCompare(right)),
    );
    return JSON.stringify({
      mapping: sortedRecord(layerMapping),
      order: layerOrder,
      enabled: sortedRecord(enabledLayers),
      opacities: sortedRecord(layerOpacities),
      patches: layerPatches.map(patch => ({
        id: patch.id,
        name: patch.name,
        sourceLayer: patch.sourceLayer,
        cutSource: patch.cutSource,
      })),
      chestMask: chestMaskDataUrl
        ? `${chestMaskDataUrl.length}:${chestMaskDataUrl.slice(-64)}`
        : "",
    });
  }

  function createWorkspaceBaseEditorSnapshot(): WorkspaceBaseEditorSnapshot {
    return {
      signature: workspaceBaseEditorSignature(),
      layerMapping: { ...layerMapping },
      bodyPreview,
      enabledLayers: { ...enabledLayers },
      layerOrder: [...layerOrder],
      layerPatches: layerPatches.map(patch => ({ ...patch })),
      layerOpacities: { ...layerOpacities },
      chestMaskDataUrl,
    };
  }

  function restoreWorkspaceBaseEditorSnapshot(snapshot: WorkspaceBaseEditorSnapshot) {
    setLayerMapping(snapshot.layerMapping);
    setBodyPreview(snapshot.bodyPreview);
    setEnabledLayers(snapshot.enabledLayers);
    enabledLayersRef.current = snapshot.enabledLayers;
    setLayerOrder(snapshot.layerOrder);
    layerOrderRef.current = snapshot.layerOrder;
    setLayerPatches(snapshot.layerPatches);
    layerPatchesRef.current = snapshot.layerPatches;
    setLayerOpacities(snapshot.layerOpacities);
    layerOpacitiesRef.current = snapshot.layerOpacities;
    setChestMaskDataUrl(snapshot.chestMaskDataUrl);
  }

  async function closeInlineEditor() {
    if (workspaceBusy) return;
    if (workspaceInlineEditor === "base"
      && baseEditorBaselineRef.current !== null
      && baseEditorBaselineRef.current.signature !== workspaceBaseEditorSignature()) {
      const leave = window.confirm("素体編集の変更はまだ保存されていません。変更を破棄して概要へ戻りますか？");
      if (!leave) return;
      restoreWorkspaceBaseEditorSnapshot(baseEditorBaselineRef.current);
    }
    if (workspaceInlineEditor === "position") {
      if (hasPendingPartAdjustment()) {
        const leave = window.confirm("差分位置の変更はまだ保存されていません。変更を破棄して概要へ戻りますか？");
        if (!leave) return;
      }
      const restoredDrafts = cloneWorkspacePartAdjustmentDrafts(workspacePartEditorBaseline.current);
      workspacePartDraftsRef.current = restoredDrafts;
      setWorkspacePartDrafts(restoredDrafts);
      loadWorkspacePartAdjustmentFields(workspaceAdjustTarget);
      if (workspacePartPersistedDuringEditor.current) {
        setWorkspaceBusy(true);
        try {
          await refreshWorkspaceCompositePreview();
          workspacePartPersistedDuringEditor.current = false;
        } catch (cause) {
          setError(String(cause));
          return;
        } finally {
          setWorkspaceBusy(false);
        }
      }
    }
    if (workspaceInlineEditor === "motion" && motionEditorDirty) {
      const saveAndLeave = window.confirm("変更を保存して編集概要へ戻りますか？");
      if (!saveAndLeave) return;
      setError("");
      setMotionPostSaveDestination("overview");
      setMotionEditorDraftSaveBusy(true);
      setStatus("変更を保存しています...");
      setMotionDraftSaveRequestId(previous => previous + 1);
      return;
    }
    setWorkspaceInlineEditor(null);
    setMotionEditorDirty(false);
    setMotionDraftSaveRequestId(0);
    setMotionEditorDraftSaveBusy(false);
    setMotionExportRequestId(0);
    setMotionPostSaveDestination("overview");
    setMotionEditorExportReady(false);
    setMotionEditorExportBusy(false);
    setStatus("");
  }

  return (
    <div className={`app theme-${themeMode}`}>
      {mode !== "workspace" && (
        <div className="app-header app-header-minimal">
          <div className="app-header-actions">
            <button
              type="button"
              className="app-icon-button"
              title={themeMode === "dark" ? "ライトモードへ切り替え" : "ダークモードへ切り替え"}
              aria-label={themeMode === "dark" ? "ライトモードへ切り替え" : "ダークモードへ切り替え"}
              onClick={() => setThemeMode(prev => prev === "dark" ? "light" : "dark")}
            >
              <WorkspaceToolbarIcon name={themeMode === "dark" ? "sun" : "moon"} />
            </button>
            {mode !== "select" && (
              <button type="button" className="app-icon-button" title="制作ホームへ戻る" aria-label="制作ホームへ戻る" onClick={returnToModeSelect}>
                <WorkspaceToolbarIcon name="home" />
              </button>
            )}
          </div>
        </div>
      )}

      <div className="main-content">
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
