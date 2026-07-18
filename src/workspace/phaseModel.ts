import type { WorkspaceStep } from "./types";

/** ユーザーに見せる工程。内部の7 STEPは WorkspaceStep として維持する。 */
export type WorkspacePhaseId = "materials" | "parts" | "frames" | "motion";

export type WorkspacePhaseKind = "flow" | "editor";

export type WorkspaceProgressStatus =
  | "locked"
  | "ready"
  | "active"
  | "running"
  | "complete"
  | "stale"
  | "error";

/**
 * STEPごとの成果物・異常フラグ。
 * 未指定のSTEPは false として扱うため、UIは把握している状態だけを渡せる。
 */
export type WorkspaceStepFlags = Readonly<Partial<Record<WorkspaceStep, boolean>>>;

export interface WorkspacePhaseDefinition {
  readonly id: WorkspacePhaseId;
  readonly label: string;
  readonly kind: WorkspacePhaseKind;
}

export interface WorkspacePhaseModelInput {
  /** 現在UIに表示している内部STEP。 */
  readonly currentStep: WorkspaceStep;
  /**
   * 各STEPの成果物が存在するか。
   * stale / error / running はこの値より優先され、validな完了とは区別される。
   */
  readonly finished: WorkspaceStepFlags;
  /** 同時に実行できる処理は1 STEPだけ、という現行ワークスペースの前提。 */
  readonly runningStep?: WorkspaceStep | null;
  /** 明示したSTEP以降に既存成果物があれば、自動的に stale が伝播する。 */
  readonly stale?: WorkspaceStepFlags;
  /** エラー表示をSTEPへ紐付ける。複数指定も可能。 */
  readonly error?: WorkspaceStepFlags;
}

export interface WorkspaceSubstepModel {
  readonly step: WorkspaceStep;
  readonly phaseId: WorkspacePhaseId;
  readonly status: WorkspaceProgressStatus;
  /** statusがcompleteでも、現在表示中のSTEPかどうかを別に表現できる。 */
  readonly isCurrent: boolean;
  /** 入力された成果物の有無。staleな成果物も true のまま保持する。 */
  readonly isFinished: boolean;
  /** 先行STEPがすべてvalidに完了しているか。 */
  readonly prerequisitesComplete: boolean;
  /** stale / running / errorを含まないvalidな完了。 */
  readonly isComplete: boolean;
}

export interface WorkspacePhaseModelItem extends WorkspacePhaseDefinition {
  readonly status: WorkspaceProgressStatus;
  readonly isActive: boolean;
  readonly steps: readonly WorkspaceSubstepModel[];
  readonly completedCount: number;
  readonly totalCount: number;
}

export interface WorkspacePhaseModel {
  readonly currentStep: WorkspaceStep;
  readonly activePhaseId: WorkspacePhaseId;
  readonly substeps: readonly WorkspaceSubstepModel[];
  readonly phases: readonly WorkspacePhaseModelItem[];
}

export const WORKSPACE_STEPS = [1, 2, 3, 4, 5, 6, 7] as const;

export type WorkspaceStepPersistenceCommand =
  | "update_expression_workspace_step"
  | "regress_expression_workspace_step";

/**
 * 通常進行は前進専用、再編集保存だけは明示的な巻き戻しを選ぶ。
 * 画面を閲覧しただけではこの関数を呼ばない。
 */
export function persistenceCommandAfterWorkspaceEdit(
  reachedStep: number,
  editedCheckpoint: WorkspaceStep,
): WorkspaceStepPersistenceCommand {
  return reachedStep > editedCheckpoint
    ? "regress_expression_workspace_step"
    : "update_expression_workspace_step";
}

export interface WorkspaceArtifactReadinessInput {
  readonly currentStep: number;
  readonly requestDirty: boolean;
  readonly generatedReady: boolean;
  readonly extractedComplete: boolean;
  readonly basePreviewReady: boolean;
  readonly rifeCurrent: boolean;
}

export interface WorkspaceArtifactReadiness {
  readonly step2Ready: boolean;
  readonly step3Complete: boolean;
  readonly step4Complete: boolean;
  readonly step5Complete: boolean;
  readonly step6Complete: boolean;
}

/** project.jsonのcheckpointを正本として、残存ファイルだけで後工程を復活させない。 */
export function workspaceArtifactReadiness(
  input: WorkspaceArtifactReadinessInput,
): WorkspaceArtifactReadiness {
  const step2Ready = input.generatedReady && !input.requestDirty;
  const step3Complete = step2Ready && input.currentStep >= 4 && input.extractedComplete;
  const step4Complete = step3Complete && input.currentStep >= 5 && input.basePreviewReady;
  const step5Complete = step4Complete && input.currentStep >= 6;
  const step6Complete = step5Complete && input.currentStep >= 7 && input.rifeCurrent;
  return { step2Ready, step3Complete, step4Complete, step5Complete, step6Complete };
}

export interface WorkspaceResumeArtifacts {
  readonly generatedReady: boolean;
  readonly extractedReady: boolean;
  readonly compositeReady: boolean;
  readonly rifeReady: boolean;
}

/**
 * 再開時はcheckpointより上へ成果物の存在だけで昇格させない。
 * checkpointが指す成果物を失った場合だけ、再作成できる工程へ安全側に下げる。
 */
export function restoredWorkspaceStep(
  checkpoint: WorkspaceStep,
  artifacts: WorkspaceResumeArtifacts,
): WorkspaceStep {
  if (checkpoint === 7 && artifacts.rifeReady) return 7;
  let restored = checkpoint === 7 ? 6 : checkpoint;
  if (restored >= 5 && !artifacts.compositeReady) restored = 4;
  if (restored >= 4 && !artifacts.extractedReady) restored = 3;
  if (restored >= 3 && !artifacts.generatedReady) restored = 2;
  return restored as WorkspaceStep;
}

export const WORKSPACE_PHASE_ORDER = [
  "materials",
  "parts",
  "frames",
  "motion",
] as const satisfies readonly WorkspacePhaseId[];

export const WORKSPACE_PHASE_DEFINITIONS = {
  materials: { id: "materials", label: "素材準備", kind: "flow" },
  parts: { id: "parts", label: "パーツ編集", kind: "editor" },
  frames: { id: "frames", label: "フレーム生成", kind: "flow" },
  motion: { id: "motion", label: "モーション仕上げ", kind: "editor" },
} as const satisfies Readonly<Record<WorkspacePhaseId, WorkspacePhaseDefinition>>;

/**
 * Record<WorkspaceStep, ...> に satisfies させることで、内部STEPが増減した際に
 * 対応漏れをTypeScriptのコンパイルエラーとして検出する。
 */
const PHASE_BY_STEP = {
  1: "materials",
  2: "materials",
  3: "materials",
  4: "parts",
  5: "parts",
  6: "frames",
  7: "motion",
} as const satisfies Readonly<Record<WorkspaceStep, WorkspacePhaseId>>;

export function phaseForStep(step: WorkspaceStep): WorkspacePhaseId {
  return PHASE_BY_STEP[step];
}

export function activePhaseId(currentStep: WorkspaceStep): WorkspacePhaseId {
  return phaseForStep(currentStep);
}

function hasFlag(flags: WorkspaceStepFlags | undefined, step: WorkspaceStep): boolean {
  return flags?.[step] === true;
}

function statusForSubstep(
  input: WorkspacePhaseModelInput,
  step: WorkspaceStep,
  prerequisitesComplete: boolean,
): WorkspaceProgressStatus {
  const isFinished = hasFlag(input.finished, step);

  if (hasFlag(input.error, step)) return "error";
  if (input.runningStep === step) return "running";
  if (hasFlag(input.stale, step) || (isFinished && !prerequisitesComplete)) return "stale";
  if (isFinished) return "complete";
  if (!prerequisitesComplete) return "locked";
  if (input.currentStep === step) return "active";
  return "ready";
}

function statusForPhase(
  steps: readonly WorkspaceSubstepModel[],
  isActive: boolean,
): WorkspaceProgressStatus {
  if (steps.some(({ status }) => status === "error")) return "error";
  if (steps.some(({ status }) => status === "running")) return "running";
  if (steps.some(({ status }) => status === "stale")) return "stale";
  if (steps.every(({ status }) => status === "complete")) return "complete";
  if (isActive) return "active";
  if (steps.some(({ status }) => status === "ready")) return "ready";
  return "locked";
}

/**
 * 7 STEPの成果物状態を、4フェーズと各サブステップの表示状態へ変換する純粋関数。
 *
 * 状態の優先順位は error > running > stale > complete > active > ready > locked。
 * 先行STEPがvalidでなくなった場合、既存の後続成果物はstale、未作成の後続STEPは
 * lockedになるため、呼び出し側でstaleを全工程へ重複指定する必要はない。
 */
export function buildWorkspacePhaseModel(
  input: WorkspacePhaseModelInput,
): WorkspacePhaseModel {
  const substeps: WorkspaceSubstepModel[] = [];
  let prerequisitesComplete = true;

  for (const step of WORKSPACE_STEPS) {
    const status = statusForSubstep(input, step, prerequisitesComplete);
    const isComplete = status === "complete";

    substeps.push({
      step,
      phaseId: phaseForStep(step),
      status,
      isCurrent: input.currentStep === step,
      isFinished: hasFlag(input.finished, step),
      prerequisitesComplete,
      isComplete,
    });

    prerequisitesComplete = prerequisitesComplete && isComplete;
  }

  const selectedPhaseId = activePhaseId(input.currentStep);
  const phases = WORKSPACE_PHASE_ORDER.map((id): WorkspacePhaseModelItem => {
    const steps = substeps.filter(({ phaseId }) => phaseId === id);
    const isActive = id === selectedPhaseId;

    return {
      ...WORKSPACE_PHASE_DEFINITIONS[id],
      status: statusForPhase(steps, isActive),
      isActive,
      steps,
      completedCount: steps.filter(({ isComplete }) => isComplete).length,
      totalCount: steps.length,
    };
  });

  return {
    currentStep: input.currentStep,
    activePhaseId: selectedPhaseId,
    substeps,
    phases,
  };
}
