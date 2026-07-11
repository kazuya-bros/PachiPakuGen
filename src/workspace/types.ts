// --- Types ---
export interface AdjustableLayer { name: string; thumbnail: string; default_target: string; }
export interface SlotLoadResult { detected_layers: string[]; adjustable_layers: AdjustableLayer[]; canvas_width: number; canvas_height: number; source_type: string; }
export interface LayerBounds { x: number; y: number; width: number; height: number; }
export interface LayerInfo { name: string; thumbnail: string; bounds: LayerBounds; }
export interface CategoryPreview { target: string; label: string; preview: string; layer_names: string[]; layers: LayerInfo[]; }
export interface MappingPreviewResult { categories: CategoryPreview[]; composite_preview: string; }
export interface RenderCategoryResult { preview: string; }
export interface ExportCorrectedLayerResult { output_path: string; }
export interface ImportCorrectionLayerResult { layer_name: string; }
export interface OriginalImageResult { original_preview: string; mouth_preview: string | null; }
export interface MouthMaskPreviewResult { mouth_preview: string; }
export interface CreateBaseResult { output_path: string; composite_preview: string; base_eye_slot: string; base_mouth_slot: string; file_count: number; }
export interface CreateDiffResult { output_path: string; pair_name: string; frame_count: number; preview: string; previews: string[]; }
export interface ProgressPayload { current: number; total: number; pair_name: string; }
export interface LayerPatch { id: string; name: string; sourceLayer: string; maskPng: string; cutSource: boolean; thumbnail?: string; }
export type InterpPair = { name: string; label: string; closed: { key: string; label: string }; open: { key: string; label: string }; required: boolean };
export type MouthMaskSetting = { dilate: number; blur: number };
export type PreviewPan = { x: number; y: number };
export type DiffPreview = { pairName: string; label: string; frames: string[] };
export type InterpStep = 1 | 2 | 3 | 4;
export type BaseStep = 1 | 2 | 3 | 4;
export type WorkspaceStep = 1 | 2 | 3 | 4 | 5 | 6 | 7;
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
  busy: boolean;
  runtimeRoot: string;
  repoPath: string;
  pythonPath: string;
  pinnedCommit: string;
  installedCommit: string | null;
  gpuIndex: number | null;
  gpuName: string | null;
  gpuMemoryMb: number | null;
  recommendedProfile: string;
  message: string;
}
export interface SeeThroughProgress { stage: string; percent: number; message: string; }
export interface SeeThroughRunResult {
  psdPath: string;
  outputDir: string;
  selectedProfile: string;
  slotLoad: SlotLoadResult;
  mappingPreview: MappingPreviewResult;
  /** 左右パーツ分解に失敗し、左右分解なしで自動リトライされた場合にtrue */
  splitPartsFallback: boolean;
  /** GPU VRAM不足で自動リトライされた場合、その内容の説明文 */
  oomRetryNote: string | null;
}
export interface WorkspaceProject {
  version: number;
  createdAt: number;
  updatedAt: number;
  currentStep: number;
  sourceImagePath: string | null;
  referenceImagePath: string | null;
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
  handoffPath: string;
  generatedPartsPath: string;
  expectedParts: string[];
  presentParts: string[];
  missingParts: string[];
  sizeMismatches: string[];
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
  if (/Loading pipeline components|Loading weights|Loading checkpoint/i.test(cleaned)) {
    return "モデルを読み込んでいます";
  }
  if (/running marigold/i.test(cleaned)) return "深度を推定しています";
  cleaned = cleaned
    // "100%|██████| 5/5 [00:02<00:00, 1.59it/s]" のようなtqdm断片を除去
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
export interface CodexRifeFramePreviewItem {
  part: string;
  frameIndex: number;
  frameCount: number;
  preview: string;
}
export interface PreviewCodexRifeResult {
  basePreview: string;
  previews: CodexRifeFramePreviewItem[];
}
export type WorkspacePreviewItem = CodexCompositePreviewItem | CodexRifeFramePreviewItem;

export function isRifePreviewItem(item: WorkspacePreviewItem): item is CodexRifeFramePreviewItem {
  return "frameIndex" in item;
}

// 胸切出モードの patchDraftSource センチネル値（特定のPSDレイヤーに紐付かないため）
export const CHEST_CUT_SENTINEL = "__chest__";

export function workspacePreviewItemKey(item: WorkspacePreviewItem): string {
  return isRifePreviewItem(item) ? `${item.part}:${item.frameIndex}` : item.part;
}

export function workspacePreviewItemLabel(item: WorkspacePreviewItem): string {
  return isRifePreviewItem(item) ? `${item.part} ${item.frameIndex}/${item.frameCount}` : item.part;
}

// Step5でパーツ個別調整できる対象。
// eyes-open はsource由来で通常はズレないため「全パーツ一括」の対象外だが、
// See-Throughの目レイヤー検出が甘い素材向けに個別選択でのみ調整できる
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
export interface LoadCodexExpressionJobResult {
  job: PrepareCodexExpressionJobResult;
  generatedParts: WorkspaceGeneratedPartsStatus;
  extractedParts: ExtractCodexGeneratedPartsResult | null;
  rifeOutput: GenerateCodexRifeOutputResult | null;
  resumeStep: number;
}

// Each RIFE pair: closed PSD -> open PSD (open = base)
export const EYE_PAIRS = [
  { name: "eye", label: "まばたき(目)",
    closed: { key: "eye_closed", label: "閉じる" },
    open: { key: "eye_open", label: "開く" },
    required: true },
];

export const MOUTH_PAIRS_SINGLE = [
  { name: "mouth", label: "口パク",
    closed: { key: "mouth_closed", label: "閉じる" },
    open: { key: "mouth_open", label: "開く" },
    required: true },
];

export const MOUTH_PAIRS_VOWELS = [
  { name: "mouth_a", label: "口パク(あ)",
    closed: { key: "mouth_a_closed", label: "閉じる" },
    open: { key: "mouth_a_open", label: "開く" },
    required: false },
  { name: "mouth_i", label: "口パク(い)",
    closed: { key: "mouth_i_closed", label: "閉じる" },
    open: { key: "mouth_i_open", label: "開く" },
    required: false },
  { name: "mouth_u", label: "口パク(う)",
    closed: { key: "mouth_u_closed", label: "閉じる" },
    open: { key: "mouth_u_open", label: "開く" },
    required: false },
  { name: "mouth_e", label: "口パク(え)",
    closed: { key: "mouth_e_closed", label: "閉じる" },
    open: { key: "mouth_e_open", label: "開く" },
    required: false },
  { name: "mouth_o", label: "口パク(お)",
    closed: { key: "mouth_o_closed", label: "閉じる" },
    open: { key: "mouth_o_open", label: "開く" },
    required: false },
];
