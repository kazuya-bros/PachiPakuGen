// --- Types ---
export interface AdjustableLayer { name: string; thumbnail: string; default_target: string; }
export interface SlotLoadResult { detected_layers: string[]; adjustable_layers: AdjustableLayer[]; canvas_width: number; canvas_height: number; source_type: string; }
export interface LayerBounds { x: number; y: number; width: number; height: number; }
export interface LayerInfo { name: string; thumbnail: string; bounds: LayerBounds; }
export interface CategoryPreview { target: string; label: string; preview: string; layer_names: string[]; layers: LayerInfo[]; }
export interface MappingPreviewResult { categories: CategoryPreview[]; composite_preview: string; }
export interface RenderCategoryResult { preview: string; }
export interface CreateBaseResult { output_path: string; composite_preview: string; base_eye_slot: string; base_mouth_slot: string; file_count: number; }
export interface ProgressPayload { current: number; total: number; pair_name: string; }
export interface LayerPatch { id: string; name: string; sourceLayer: string; maskPng: string; cutSource: boolean; thumbnail?: string; }
export type PreviewPan = { x: number; y: number };
export type WorkspaceStep = 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type WorkspaceMouthCornerMode = "source" | "up" | "flat" | "down";
export type SeeThroughProfile = "low-vram" | "standard";
export type SeeThroughOptionMode = "default" | "on" | "off";

export interface SeeThroughOptions {
  seed: number;
  resolution: number;
  resolutionDepth: number;
  inferenceSteps: number;
  inferenceStepsDepth: number;
  groupOffload: SeeThroughOptionMode;
  cpuOffload: SeeThroughOptionMode;
}

export const DEFAULT_SEE_THROUGH_OPTIONS: SeeThroughOptions = {
  seed: 42,
  resolution: 1280,
  resolutionDepth: 768,
  inferenceSteps: 30,
  inferenceStepsDepth: -1,
  groupOffload: "default",
  cpuOffload: "default",
};

export const RIFE_FRAME_MIN = 2;
export const RIFE_FRAME_MAX = 16;
export const RIFE_FRAME_RECOMMENDED = "4〜8";

export interface SeeThroughRuntimeStatus {
  ready: boolean;
  runtimeReady: boolean;
  busy: boolean;
  modelDownloadBusy: boolean;
  runtimeRoot: string;
  repoPath: string;
  pythonPath: string;
  pinnedCommit: string;
  installedCommit: string | null;
  gpuIndex: number | null;
  gpuName: string | null;
  gpuMemoryMb: number | null;
  recommendedProfile: string;
  selectedProfile: string;
  message: string;
}
export interface SeeThroughProgress { stage: string; percent: number; message: string; }
export interface SeeThroughRunResult {
  psdPath: string;
  outputDir: string;
  selectedProfile: string;
  /** 自動回復を含め、実際に成功した設定 */
  effectiveOptions: SeeThroughOptions | null;
  slotLoad: SlotLoadResult;
  mappingPreview: MappingPreviewResult;
  /** 左右パーツ分解に失敗し、左右分解なしで自動リトライされた場合にtrue */
  splitPartsFallback: boolean;
  /** GPU/ネイティブ推論エラーで自動リトライされた場合、その内容の説明文 */
  oomRetryNote: string | null;
}
export interface SeeThroughLayerProbeLayer {
  name: string;
  thumbnail: string;
  opaquePixels: number;
}
export interface SeeThroughLayerProbeResult {
  selectedProfile: string;
  layers: SeeThroughLayerProbeLayer[];
}
export interface WorkspaceProject {
  version: number;
  createdAt: number;
  updatedAt: number;
  currentStep: number;
  sourceImagePath: string | null;
  referenceImagePath: string | null;
  codexPrompt: string | null;
  /** 旧バックエンド／旧project.jsonでは未定義のため、UI側でflatへフォールバックする。 */
  mouthCorner?: WorkspaceMouthCornerMode;
}
export interface ExpressionWorkspaceResult {
  workPath: string;
  projectPath: string;
  codexRequestPath: string;
  generatedPartsPath: string;
  seeThroughPath: string;
  spritalkPartsPath: string;
  project: WorkspaceProject;
}
export interface WorkspaceGeneratedPartsStatus {
  requestPath: string;
  generatedPartsPath: string;
  expectedParts: string[];
  presentParts: string[];
  missingParts: string[];
  /** 口角設定導入前のバックエンド応答やHMR中の保持状態では未定義。 */
  staleParts?: string[];
  sizeMismatches: string[];
  /** 立ち絵とサイズが異なるが自動リサイズで吸収するパーツ（旧バックエンド応答では未定義） */
  autoFitParts?: string[];
  /** 完了後に生成素材の差し替えを検知し、下流工程をSTEP3へ戻した場合にtrue。 */
  downstreamStale: boolean;
  ready: boolean;
}
export interface PartAdjustment {
  offsetX: number;
  offsetY: number;
  scalePercent: number;
}

export const DEFAULT_PART_ADJUSTMENT: PartAdjustment = { offsetX: 0, offsetY: 0, scalePercent: 100 };

export interface ExtractCodexGeneratedPartsResult {
  extractedPartsPath: string;
  extractedParts: string[];
  warnings: string[];
  selectedProfile: string;
  effectiveOptions: SeeThroughOptions | null;
  splitParts: boolean;
  /** パーツごとの現在の位置補正値。STEP5でパーツ切替時に実際の値を表示するために使う */
  partAdjustments: Record<string, PartAdjustment>;
}

export function isNoisySeeThroughWarning(message: string): boolean {
  return message.includes("HF_TOKEN")
    || message.includes("unauthenticated requests to the HF Hub")
    || message.includes("local_dir_use_symlinks");
}

export function displaySeeThroughMessage(progress: SeeThroughProgress | null): string {
  if (!progress) return "See-Throughを実行しています";
  if (isNoisySeeThroughWarning(progress.message)) {
    return progress.stage === "inference" ? "モデル取得または分解処理を継続しています" : "See-Through処理を継続しています";
  }
  if (progress.message.trim()) return progress.message;
  return "See-Through処理を継続しています";
}

/**
 * tqdm進捗バー・ANSIエスケープ等の生ログノイズを除去し、表示向けの短い一文へ整形する。
 * 表示する価値のない断片（カーソル制御・Pythonの途中行など）は "" を返す＝表示しない。
 */
export function sanitizeSeeThroughLogMessage(message: string): string {
  // ANSIエスケープ（ESC[A 等のカーソル制御）と、ESCが欠落した "[A" 断片を除去
  let cleaned = message
    .replace(/\[[0-9;]*[A-Za-z]/g, " ")
    .replace(/(^|\s)\[[A-Z](?=\s|$)/g, " ");
  const modelDownload = cleaned.match(/^Model download (\d+\/\d+): (.+) \(([^)]+)\)$/i);
  if (modelDownload) {
    return `モデル取得 ${modelDownload[1]}: ${modelDownload[2]} (${modelDownload[3]})`;
  }
  const preparingModel = cleaned.match(/^Preparing required See-Through model: (.+)$/i);
  if (preparingModel) return `必須モデルを準備しています: ${preparingModel[1]}`;
  const repairingModel = cleaned.match(/^Repairing incomplete model file: (.+)$/i);
  if (repairingModel) return `不完全なモデルファイルを再取得しています: ${repairingModel[1]}`;
  if (/Loading pipeline components|Loading weights|Loading checkpoint/i.test(cleaned)) {
    return "モデルを読み込んでいます";
  }
  // 初回や cache が空の時はモデルをHuggingFaceからDLするため、この状態が長く続くことがある（正常）
  if (/hf_hub_download|Downloading|\.safetensors|resolve\/main/i.test(cleaned)) {
    return "モデルをダウンロードしています（初回は時間がかかります）";
  }
  if (/running layerdiff/i.test(cleaned)) return "レイヤーを分解しています";
  if (/running marigold/i.test(cleaned)) return "深度を推定しています";
  // tqdmの進捗バー（\r更新、例: "45%|████▌ | 2.62G/5.83G [02:15<02:45, 18.2MB/s]"）は
  // 単に除去すると「本当に進んでいるか」が画面から分からなくなるため、内容を抽出して
  // 表示する（ダウンロード/処理が生きていることを可視化する）
  const tqdmMatch = cleaned.match(
    /(\d+)%\|[^|]*\|\s*([^\s/]+)\/([^\s[]+)\s*\[([^<]+)<([^,\]]+),?\s*([^\]]*)\]/,
  );
  if (tqdmMatch) {
    const [, percent, current, total, elapsed, remaining, speed] = tqdmMatch;
    const remainingPart = remaining.trim() && remaining.trim() !== "?" ? `・残り${remaining.trim()}` : "";
    const speedPart = speed.trim() ? `・${speed.trim()}` : "";
    return `処理中 ${percent}%（${current}/${total}・経過${elapsed.trim()}${remainingPart}${speedPart}）`;
  }
  cleaned = cleaned
    // 上記でマッチしなかった残りのtqdm断片（不完全な行等）を除去
    .replace(/\d+%\|[^|]*\|\s*\d+\/\d+\s*(\[[^\]]*\])?/g, " ")
    .replace(/\[\d+:\d+<[^\]]*\]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  // 意味のない断片（Pythonソースの途中行・罫線など）は表示しない
  if (cleaned.length < 4) return "";
  if (/^(return |File |Traceback|self\.|args\.|\^|~)/.test(cleaned)) return "";
  return cleaned;
}

export function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return minutes > 0 ? `${minutes}分${rest.toString().padStart(2, "0")}秒` : `${rest}秒`;
}
export interface CodexCompositePreviewItem {
  part: string;
  preview: string;
}
export interface PreviewCodexCompositeResult {
  basePreview: string;
  previews: CodexCompositePreviewItem[];
}

// 胸切出モードの patchDraftSource センチネル値（特定のPSDレイヤーに紐付かないため）
export const CHEST_CUT_SENTINEL = "__chest__";

export function workspacePreviewItemKey(item: CodexCompositePreviewItem): string {
  return item.part;
}

export function workspacePreviewItemLabel(item: CodexCompositePreviewItem): string {
  return item.part;
}

// Step5でパーツ個別調整できる対象。平常時のeyes-openも口パク全体の基準になるため調整可能。
export const WORKSPACE_ADJUST_PART_KEYS = [
  "eyes-open",
  "eyes-closed",
  "mouth-closed",
  "mouth-a",
  "mouth-i",
  "mouth-u",
  "mouth-e",
  "mouth-o",
];

export interface GenerateCodexRifeOutputResult {
  outputPath: string;
  directories: string[];
  frameCount: number;
}
export interface SaveCodexBasePartsResult {
  basePartsPath: string;
  savedParts: string[];
}
export interface AdjustCodexExtractedPartsResult {
  extractedPartsPath: string;
  adjustedParts: string[];
  offsetX: number;
  offsetY: number;
  scalePercent: number;
  /** 適用後のパーツごとの位置補正値（全パーツ分） */
  partAdjustments: Record<string, PartAdjustment>;
}
export interface PrepareCodexExpressionJobResult {
  jobPath: string;
  sourcePath: string;
  referencePath: string | null;
  requestPath: string;
  handoffPath: string;
  generatedPartsPath: string;
  expectedParts: string[];
  missingParts: string[];
}
export interface InspectCodexGeneratedPartsResult {
  generatedPartsPath: string;
  expectedParts: string[];
  presentParts: string[];
  missingParts: string[];
  sizeMismatches: string[];
  ready: boolean;
}
export interface LoadCodexExpressionJobResult {
  job: PrepareCodexExpressionJobResult;
  generatedParts: InspectCodexGeneratedPartsResult;
  extractedParts: ExtractCodexGeneratedPartsResult | null;
  rifeOutput: GenerateCodexRifeOutputResult | null;
  resumeStep: number;
}
