import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  MotionLabEffectKey,
  MotionLabImageSet,
  MotionLabManifestResult,
  MotionLabMouthKey,
  MotionLabMouthRuntime,
  MotionLabPartsResult,
  MotionLabTimelineEvent,
  SpritalkMotionProfileResult,
} from "./types";
import {
  MOTION_LAB_EFFECT_DEFS,
  MOTION_LAB_MOUTH_KEYS,
  MOTION_LAB_MOUTH_LABELS,
  MOTION_LAB_TEMPLATE_LAYOUT,
  MOTION_LAB_TEMPLATES,
  MOTION_LAB_VOWEL_KEYS,
  motionLabTimelineFromText,
  motionLabVowelLoopTimeline,
} from "./constants";
import {
  createMotionLabPhysics,
  drawMotionLabScene,
  loadMotionLabImage,
  prepareMotionLabCanvas,
  resetMotionLabRuntime,
} from "./render";
import {
  BUILT_IN_MOTION_SEQUENCE,
  buildMotionLabManifest,
  buildSpritalkMotionProfile,
  type MotionLabSequenceDefinition,
} from "./manifest";
import { toRenderSettings, useMotionLabSettings } from "./useMotionLabSettings";

export interface MotionTunePanelProps {
  /** 読み込む素材フォルダ（04_spritalk_parts / rife_output）。null=未選択 */
  partsDir: string | null;
  /** false=非表示中（display:none）。rAF描画を止めて調整stateは保持する */
  active?: boolean;
  /** ステータス通知（親のステータスバーへ） */
  onNotify?: (message: string) => void;
  /** エラー通知（親のエラーバナーへ） */
  onError?: (message: string) => void;
  /** 保存対象の調整値が変更されたかを親へ通知 */
  onDirtyChange?: (dirty: boolean, scope?: "settings" | "sequence") => void;
  /** 親フッターからSpriTalk向け出力を要求する連番 */
  exportRequestId?: number;
  /** 親の戻る操作から途中設定の保存を要求する連番 */
  draftSaveRequestId?: number;
  /** 出力操作の準備・処理状態を親へ通知 */
  onExportStateChange?: (state: { ready: boolean; busy: boolean }) => void;
  /** 戻る前の途中保存状態を親へ通知 */
  onDraftSaveStateChange?: (busy: boolean) => void;
  /** 戻る前の途中保存完了を親へ通知 */
  onDraftSaved?: () => void;
  /** SpriTalk向け出力の完了を親へ通知 */
  onExported?: (path: string) => void;
}

function createRuntime(): MotionLabMouthRuntime {
  return {
    openY: 0,
    activeTarget: "closed",
    previousTarget: "closed",
    transitionStartMs: 0,
    lastMs: 0,
    browVoice: 0,
    physics: createMotionLabPhysics(),
  };
}

const EAR_SWAY_NAME_PATTERN = /(^|_)ears?(_|$)/i;

function earSwayLabel(name: string, index: number, total: number): string {
  if (/(?:_|-)l$/i.test(name)) return "左獣耳";
  if (/(?:_|-)r$/i.test(name)) return "右獣耳";
  return total === 1 ? "獣耳" : `獣耳 ${index + 1}`;
}

/**
 * STEP7「モーション調整」パネル。
 * 旧Motion Preview Lab（2レーン比較実験画面）を製品向けの1レーン調整画面へ再構成したもの。
 * 素材の読込・物理プレビュー・設定の保存/読込・SpriTalk用設定JSONの出力まで自己完結する。
 */
export function MotionTunePanel({
  partsDir,
  active = true,
  onNotify,
  onError,
  onDirtyChange,
  exportRequestId = 0,
  draftSaveRequestId = 0,
  onExportStateChange,
  onDraftSaveStateChange,
  onDraftSaved,
  onExported,
}: MotionTunePanelProps) {
  const [parts, setParts] = useState<MotionLabPartsResult | null>(null);
  const [images, setImages] = useState<MotionLabImageSet | null>(null);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [text, setText] = useState("こんにちは、あいうえお");
  const [customTimeline, setCustomTimeline] = useState<{
    timeline: MotionLabTimelineEvent[];
    durationMs: number;
    text: string;
  } | null>(null);
  const [pivotEditPart, setPivotEditPart] = useState<string | null>(null);
  const [previewZoom, setPreviewZoom] = useState(1);
  const [previewPan, setPreviewPan] = useState({ x: 0, y: 0 });
  const [previewPanning, setPreviewPanning] = useState(false);
  /** ループ素材書き出しの設定。周期は呼吸(3.6s)と内蔵あいうえお(3600ms)の公倍数 */
  const [loopExportConfig, setLoopExportConfig] = useState({
    periodMs: 14400,
    // "silent"=口パクなし（閉じたまま）。それ以外は該当母音のみを開閉ループする
    mouth: "a" as "silent" | MotionLabMouthKey,
    frames: true,
    apng: true,
    gif: false,
  });
  const [loopExportProgress, setLoopExportProgress] = useState<{
    phase: string;
    current: number;
    total: number;
    /** 所要フレーム数が未確定な段階（形式の組み立て中など）。進捗バーを不確定表示にする */
    indeterminate?: boolean;
  } | null>(null);
  const loopExportCancelRef = useRef(false);
  /** フッターの「書き出す」を押した時に、SpriTalk保存/ループ素材のどちらかを選ぶモーダル */
  const [exportChoiceOpen, setExportChoiceOpen] = useState(false);
  const [loopExportPanelOpen, setLoopExportPanelOpen] = useState(false);
  const [spritalkSaving, setSpritalkSaving] = useState(false);

  useEffect(() => {
    if (!exportChoiceOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) setExportChoiceOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [exportChoiceOpen, busy]);
  const [, setManifestPath] = useState("");
  const [settings, settingsDispatch] = useMotionLabSettings();
  const previewStageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewPanDragRef = useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startPanX: number;
    startPanY: number;
  } | null>(null);
  const runtimeRef = useRef<MotionLabMouthRuntime>(createRuntime());
  const lastExportRequestRef = useRef(0);
  const lastDraftSaveRequestRef = useRef(0);
  const latestSettingsRef = useRef(settings);
  const manifestWriteChainRef = useRef<Promise<void>>(Promise.resolve());

  const notify = (message: string) => onNotify?.(message);
  const fail = (cause: unknown) => onError?.(cause instanceof Error ? cause.message : String(cause));
  const dispatch = (action: Parameters<typeof settingsDispatch>[0]) => {
    settingsDispatch(action);
    onDirtyChange?.(true, "settings");
  };

  function currentSequence(): MotionLabSequenceDefinition {
    return customTimeline
      ? {
        type: "text",
        text: customTimeline.text,
        durationMs: customTimeline.durationMs,
        events: customTimeline.timeline,
      }
      : BUILT_IN_MOTION_SEQUENCE;
  }

  function enqueueManifestWrite(
    sourceDir: string,
    manifest: ReturnType<typeof buildMotionLabManifest>,
  ) {
    const queued = manifestWriteChainRef.current.then(
      () => invoke<MotionLabManifestResult>("save_motion_lab_manifest", {
        request: { sourceDir, manifest },
      }),
      () => invoke<MotionLabManifestResult>("save_motion_lab_manifest", {
        request: { sourceDir, manifest },
      }),
    );
    manifestWriteChainRef.current = queued.then(() => undefined, () => undefined);
    return queued;
  }

  async function loadPartsFromDir(dir: string) {
    setBusy(true);
    onExportStateChange?.({ ready: false, busy: true });
    try {
      const result = await invoke<MotionLabPartsResult>("load_motion_lab_parts", { dir });
      setPlaying(true);
      setManifestPath("");
      // 保存済み設定（motion-preview-manifest.json）があれば自動復元（つづきから対応）
      try {
        const manifest = await invoke<MotionLabManifestResult>("load_motion_lab_manifest", {
          sourceDir: result.sourceDir,
        });
        settingsDispatch({ type: "applyManifest", manifest: manifest.manifest });
        const savedTimeline = manifest.manifest.timeline;
        if (
          savedTimeline?.type === "text"
          && typeof savedTimeline.text === "string"
          && Number.isFinite(savedTimeline.durationMs)
          && (savedTimeline.durationMs ?? 0) > 0
          && Array.isArray(savedTimeline.events)
          && savedTimeline.events.length > 0
        ) {
          setText(savedTimeline.text);
          setCustomTimeline({
            durationMs: savedTimeline.durationMs!,
            timeline: savedTimeline.events,
            text: savedTimeline.text,
          });
        } else {
          setCustomTimeline(null);
        }
        setManifestPath(manifest.path);
        notify(`モーション素材と保存済み設定を読み込みました: ${result.sourceDir}`);
      } catch {
        notify(`モーション素材を読み込みました: ${result.sourceDir}`);
      }
      // 設定復元が終わってから素材を公開し、画像デコード完了通知との競合を防ぐ。
      setParts(result);
      onDirtyChange?.(false);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
      // SpriTalk出力は素材の読込完了後にだけ許可する。
      onExportStateChange?.({ ready: false, busy: false });
    }
  }

  // partsDir変化時の自動ロード（同じフォルダを読込済みならスキップ = 調整中に設定が飛ばない）
  useEffect(() => {
    if (!partsDir) return;
    if (parts?.sourceDir === partsDir) return;
    void loadPartsFromDir(partsDir);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [partsDir]);

  useEffect(() => {
    latestSettingsRef.current = settings;
  }, [settings]);

  // 素材 → HTMLImageElement デコード
  useEffect(() => {
    let cancelled = false;
    if (!parts) {
      setImages(null);
      setImagesLoading(false);
      onExportStateChange?.({ ready: false, busy: false });
      return;
    }
    const current = parts;
    setImages(null);
    setImagesLoading(true);
    async function loadImages() {
      try {
        const mouthEntries = await Promise.all(
          MOTION_LAB_MOUTH_KEYS.map(async (key) => {
            const sources = current.mouths[key] ?? [];
            const loaded = await Promise.all(sources.map(loadMotionLabImage));
            return [key, loaded] as const;
          }),
        );
        const swayEntries = await Promise.all(
          Object.entries(current.sways).map(async ([name, source]) => [name, await loadMotionLabImage(source)] as const),
        );
        const linkedPartEntries = await Promise.all(
          Object.entries(current.linkedParts ?? {}).map(async ([name, linked]) => [
            name,
            { parent: linked.parent, image: await loadMotionLabImage(linked.image) },
          ] as const),
        );
        const nextImages: MotionLabImageSet = {
          body: await loadMotionLabImage(current.body),
          hair: current.hair ? await loadMotionLabImage(current.hair) : null,
          hairBack: current.hairBack ? await loadMotionLabImage(current.hairBack) : null,
          armL: current.armL ? await loadMotionLabImage(current.armL) : null,
          armR: current.armR ? await loadMotionLabImage(current.armR) : null,
          chest: current.chest ? await loadMotionLabImage(current.chest) : null,
          sways: Object.fromEntries(swayEntries),
          linkedParts: Object.fromEntries(linkedPartEntries),
          eyebrow: current.eyebrow ? await loadMotionLabImage(current.eyebrow) : null,
          eyewhite: current.eyewhite ? await loadMotionLabImage(current.eyewhite) : null,
          irides: current.irides ? await loadMotionLabImage(current.irides) : null,
          highlight: current.highlight ? await loadMotionLabImage(current.highlight) : null,
          eyeFrames: await Promise.all(current.eyeFrames.map(loadMotionLabImage)),
          mouths: Object.fromEntries(mouthEntries) as Partial<Record<MotionLabMouthKey, HTMLImageElement[]>>,
        };
        if (!cancelled) {
          setImages(nextImages);
          setImagesLoading(false);
          onExportStateChange?.({ ready: true, busy: false });
        }
      } catch (cause) {
        if (!cancelled) {
          fail(cause);
          setImagesLoading(false);
          onExportStateChange?.({ ready: false, busy: false });
        }
      }
    }
    void loadImages();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parts]);

  // 素材読込時に物理をリセット（登場撃力＋位相ランダム化）。設定変更ではリセットしない
  useEffect(() => {
    if (!images) return;
    resetMotionLabRuntime(runtimeRef.current);
  }, [images]);

  // 素材が変わったときは、前の素材で使った表示位置を持ち越さない。
  useEffect(() => {
    setPreviewZoom(1);
    setPreviewPan({ x: 0, y: 0 });
    setPreviewPanning(false);
    previewPanDragRef.current = null;
  }, [parts]);

  // 描画ループ（1レーン）。非表示中（active=false）は止める
  useEffect(() => {
    if (!active || !parts || !images) return;
    const ctx = prepareMotionLabCanvas(canvasRef.current, parts.width, parts.height);
    if (!ctx) return;
    const runtime = runtimeRef.current;
    const startedAt = performance.now();
    let animationId = 0;
    const renderSettings = toRenderSettings(settings, {
      pivotEditPart,
      timeline: customTimeline?.timeline,
      timelineDurationMs: customTimeline?.durationMs,
    });
    const draw = (now: number) => {
      const elapsedMs = playing ? now - startedAt : 0;
      drawMotionLabScene(ctx, parts, images, runtime, elapsedMs, renderSettings);
      if (playing) {
        animationId = window.requestAnimationFrame(draw);
      }
    };
    animationId = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(animationId);
  }, [active, parts, images, playing, settings, pivotEditPart, customTimeline]);

  function restartPlayback() {
    resetMotionLabRuntime(runtimeRef.current);
    setPlaying(false);
    window.setTimeout(() => setPlaying(true), 0);
  }

  function clampPreviewPan(next: { x: number; y: number }, zoom: number) {
    const stage = previewStageRef.current;
    const canvas = canvasRef.current;
    if (!stage || !canvas) return { x: 0, y: 0 };

    // ステージの約35%分は「はみ出し」を許容し、ズーム1でもある程度パンできるようにする
    const slackX = stage.clientWidth * 0.35;
    const slackY = stage.clientHeight * 0.35;
    const contentW = canvas.offsetWidth * zoom;
    const contentH = canvas.offsetHeight * zoom;
    const maxX = Math.max(slackX, (contentW - stage.clientWidth) / 2 + slackX);
    const maxY = Math.max(slackY, (contentH - stage.clientHeight) / 2 + slackY);
    return {
      x: Math.max(-maxX, Math.min(maxX, next.x)),
      y: Math.max(-maxY, Math.min(maxY, next.y)),
    };
  }

  function changePreviewZoom(nextValue: number, anchor?: { clientX: number; clientY: number }) {
    const nextZoom = Math.max(1, Math.min(4, Math.round(nextValue * 4) / 4));
    if (nextZoom === previewZoom) return;

    const stage = previewStageRef.current;
    const ratio = nextZoom / previewZoom;
    let anchorX = 0;
    let anchorY = 0;
    if (stage && anchor) {
      const rect = stage.getBoundingClientRect();
      anchorX = anchor.clientX - (rect.left + rect.width / 2);
      anchorY = anchor.clientY - (rect.top + rect.height / 2);
    }
    const nextPan = clampPreviewPan({
      x: anchorX - (anchorX - previewPan.x) * ratio,
      y: anchorY - (anchorY - previewPan.y) * ratio,
    }, nextZoom);
    setPreviewZoom(nextZoom);
    setPreviewPan(nextPan);
  }

  function resetPreviewView() {
    setPreviewZoom(1);
    setPreviewPan({ x: 0, y: 0 });
    setPreviewPanning(false);
    previewPanDragRef.current = null;
  }

  async function saveMotionDraft() {
    if (!parts) {
      onDraftSaveStateChange?.(false);
      fail("保存できるモーション素材が読み込まれていません");
      return;
    }
    onDraftSaveStateChange?.(true);
    try {
      const manifest = buildMotionLabManifest(
        latestSettingsRef.current,
        parts.sourceDir,
        currentSequence(),
      );
      const result = await enqueueManifestWrite(parts.sourceDir, manifest);
      setManifestPath(result.path);
      onDirtyChange?.(false);
      notify("変更を保存しました");
      onDraftSaved?.();
    } catch (cause) {
      fail(cause);
    } finally {
      onDraftSaveStateChange?.(false);
    }
  }

  // ===== ループ素材書き出し =====
  // 全駆動をループ長の整数分の一へ量子化（render.tsのloopPeriodMs）し、
  // 減衰バネが周期軌道へ収束するまで2周ウォームアップしてから1周期分を記録する。
  // 通常再生と違い「最初の状態に戻る瞬間」を待つ必要がない
  const LOOP_EXPORT_FPS = 30;

  /** 決定的PRNG（mulberry32）。ループ書き出しの毎回同じ結果を保証する */
  function createSeededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  async function exportMotionLoop() {
    if (!parts || !images) return;
    const config = loopExportConfig;
    if (!config.frames && !config.apng && !config.gif) {
      onError?.("出力形式（PNG連番 / APNG / AGIF）を1つ以上選択してください");
      return;
    }
    loopExportCancelRef.current = false;
    setBusy(true);
    const exportSettings = latestSettingsRef.current;
    try {
      const periodMs = config.periodMs;
      const framesPerLoop = Math.round((periodMs / 1000) * LOOP_EXPORT_FPS);
      const warmupFrames = framesPerLoop * 2;
      const frameMs = 1000 / LOOP_EXPORT_FPS;
      const vowelLoop = config.mouth !== "silent" ? motionLabVowelLoopTimeline(config.mouth) : null;
      const timeline: MotionLabTimelineEvent[] = vowelLoop?.timeline ?? [{ timeMs: 0, mouth: "closed", energy: 0 }];
      const timelineDurationMs = vowelLoop?.durationMs ?? periodMs;
      const renderSettings = {
        ...toRenderSettings(exportSettings, {
          pivotEditPart: null,
          timeline,
          timelineDurationMs,
        }),
        loopPeriodMs: periodMs,
        random: createSeededRandom(0x9e3779b9),
      };
      // 位相ゼロ・登場撃力なしの専用ランタイム（プレビューの状態に影響させない）
      const runtime: MotionLabMouthRuntime = {
        openY: 0,
        activeTarget: "closed",
        previousTarget: "closed",
        transitionStartMs: 0,
        lastMs: 0,
        browVoice: 0,
        physics: createMotionLabPhysics(false),
      };
      // 素材の実寸そのままで書き出す（縮小はOBS等の利用側に任せる）
      const fullCanvas = document.createElement("canvas");
      fullCanvas.width = parts.width;
      fullCanvas.height = parts.height;
      const fullCtx = fullCanvas.getContext("2d");
      if (!fullCtx) throw new Error("書き出し用キャンバスを作成できません");
      fullCtx.imageSmoothingEnabled = true;
      fullCtx.imageSmoothingQuality = "high";

      // ウォームアップ: 描画のみ（記録なし）。UIを固めないよう定期的にyieldする
      for (let frame = 0; frame < warmupFrames; frame += 1) {
        if (loopExportCancelRef.current) throw new Error("書き出しをキャンセルしました");
        drawMotionLabScene(fullCtx, parts, images, runtime, frame * frameMs, renderSettings);
        if (frame % 30 === 29) {
          setLoopExportProgress({ phase: "収束待ち", current: frame + 1, total: warmupFrames });
          await new Promise(resolve => window.setTimeout(resolve, 0));
        }
      }

      // 記録: ウォームアップ直後の1周期（開始時点でループ位相はちょうど0）
      for (let frame = 0; frame < framesPerLoop; frame += 1) {
        if (loopExportCancelRef.current) throw new Error("書き出しをキャンセルしました");
        const elapsedMs = (warmupFrames + frame) * frameMs;
        drawMotionLabScene(fullCtx, parts, images, runtime, elapsedMs, renderSettings);
        const blob = await new Promise<Blob>((resolve, reject) => {
          fullCanvas.toBlob(
            value => (value ? resolve(value) : reject(new Error("フレームのPNG化に失敗しました"))),
            "image/png",
          );
        });
        const pngBase64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(reader.error ?? new Error("フレームの読み出しに失敗しました"));
          reader.readAsDataURL(blob);
        });
        await invoke("save_motion_loop_frame", {
          request: { sourceDir: parts.sourceDir, frameIndex: frame, pngBase64 },
        });
        setLoopExportProgress({ phase: "フレーム保存", current: frame + 1, total: framesPerLoop });
      }

      setLoopExportProgress({ phase: "書き出し形式を組み立て中", current: 0, total: 1, indeterminate: true });
      const result = await invoke<{
        exportDir: string;
        framesDir: string | null;
        apngPath: string | null;
        gifPath: string | null;
        frameCount: number;
        fps: number;
      }>("finalize_motion_loop_export", {
        request: {
          sourceDir: parts.sourceDir,
          fps: LOOP_EXPORT_FPS,
          frameCount: framesPerLoop,
          makeApng: config.apng,
          makeGif: config.gif,
          keepFrames: config.frames,
        },
      });
      notify(`ループ素材を書き出しました（${framesPerLoop}フレーム/${LOOP_EXPORT_FPS}fps）: ${result.exportDir}`);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
      setLoopExportProgress(null);
    }
  }

  async function exportSpritalkProfile(): Promise<boolean> {
    if (!parts) return false;
    let exportedPath = "";
    let succeeded = false;
    const exportSettings = latestSettingsRef.current;
    setBusy(true);
    onExportStateChange?.({ ready: false, busy: true });
    try {
      // 設定JSONと一緒にmanifestも保存し、次回のつづきから復元と内容を一致させる
      const manifest = buildMotionLabManifest(exportSettings, parts.sourceDir, currentSequence());
      const manifestResult = await enqueueManifestWrite(parts.sourceDir, manifest);
      setManifestPath(manifestResult.path);
      const profile = buildSpritalkMotionProfile(exportSettings, parts.sourceDir);
      const result = await invoke<SpritalkMotionProfileResult>("save_spritalk_motion_profile", {
        request: { sourceDir: parts.sourceDir, profile },
      });
      exportedPath = result.path;
      succeeded = true;
      onDirtyChange?.(false);
      notify(`SpriTalk用アニメーション設定を出力しました: ${result.path}`);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
      onExportStateChange?.({ ready: true, busy: false });
    }
    if (exportedPath) onExported?.(exportedPath);
    return succeeded;
  }

  useEffect(() => {
    if (draftSaveRequestId <= 0 || draftSaveRequestId === lastDraftSaveRequestRef.current) return;
    lastDraftSaveRequestRef.current = draftSaveRequestId;
    void saveMotionDraft();
    // requestId更新時点のparts・settingsを保存する。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftSaveRequestId]);

  useEffect(() => {
    if (exportRequestId <= 0 || exportRequestId === lastExportRequestRef.current) return;
    lastExportRequestRef.current = exportRequestId;
    // フッターの「書き出す」は直接保存せず、SpriTalk保存/ループ素材書き出しの
    // どちらを行うか選ぶモーダルを開く。
    setExportChoiceOpen(true);
  }, [exportRequestId]);

  async function chooseSpritalkExport() {
    setSpritalkSaving(true);
    try {
      const succeeded = await exportSpritalkProfile();
      if (succeeded) setExportChoiceOpen(false);
    } finally {
      setSpritalkSaving(false);
    }
  }

  const percentFormat = (value: number) => `${Math.round(value * 100)}%`;
  const effectSliders: Partial<Record<MotionLabEffectKey, {
    value: number;
    min: number;
    max: number;
    step: number;
    set: (value: number) => void;
    format: (value: number) => string;
  }>> = {
    breath: { value: settings.breathAmplitude, min: 0, max: 1.6, step: 0.05, set: v => dispatch({ type: "set", patch: { breathAmplitude: v } }), format: percentFormat },
    bodySway: { value: settings.bodySwayAmplitude, min: 0, max: 1.8, step: 0.05, set: v => dispatch({ type: "set", patch: { bodySwayAmplitude: v } }), format: percentFormat },
    pyoko: { value: settings.pyokoBounce, min: 0, max: 12, step: 0.5, set: v => dispatch({ type: "set", patch: { pyokoBounce: v } }), format: value => `${value.toFixed(1)}px` },
    hairMotion: { value: settings.hairMotionStrength, min: 0, max: 2, step: 0.05, set: v => dispatch({ type: "set", patch: { hairMotionStrength: v } }), format: percentFormat },
    hairBack: { value: settings.hairBackScale, min: 0, max: 1.5, step: 0.05, set: v => dispatch({ type: "set", patch: { hairBackScale: v } }), format: percentFormat },
    parallax: { value: settings.parallaxScale, min: 0, max: 1.5, step: 0.05, set: v => dispatch({ type: "set", patch: { parallaxScale: v } }), format: percentFormat },
    glance: { value: settings.glanceStrength, min: 0, max: 2, step: 0.05, set: v => dispatch({ type: "set", patch: { glanceStrength: v } }), format: percentFormat },
    gaze: { value: settings.gazeStrength, min: 0, max: 3, step: 0.05, set: v => dispatch({ type: "set", patch: { gazeStrength: v } }), format: percentFormat },
    irisBreath: { value: settings.irisBreathStrength, min: 0, max: 1, step: 0.05, set: v => dispatch({ type: "set", patch: { irisBreathStrength: v } }), format: percentFormat },
    wetness: { value: settings.wetnessStrength, min: 0, max: 1, step: 0.05, set: v => dispatch({ type: "set", patch: { wetnessStrength: v } }), format: percentFormat },
    brow: { value: settings.browStrength, min: 0, max: 1.5, step: 0.05, set: v => dispatch({ type: "set", patch: { browStrength: v } }), format: percentFormat },
    blink: { value: settings.blinkRate, min: 0.3, max: 2.5, step: 0.05, set: v => dispatch({ type: "set", patch: { blinkRate: v } }), format: value => `×${value.toFixed(2)}` },
    arm: { value: settings.armSwayAmp, min: 0, max: 3, step: 0.1, set: v => dispatch({ type: "set", patch: { armSwayAmp: v } }), format: percentFormat },
    lift: { value: settings.liftStrength, min: 0, max: 2, step: 0.05, set: v => dispatch({ type: "set", patch: { liftStrength: v } }), format: percentFormat },
    chest: { value: settings.chestMax, min: 0, max: 12, step: 0.5, set: v => dispatch({ type: "set", patch: { chestMax: v } }), format: value => `${value.toFixed(1)}px` },
    earTwitch: { value: settings.earTwitchScale, min: 0, max: 2, step: 0.05, set: v => dispatch({ type: "set", patch: { earTwitchScale: v } }), format: percentFormat },
  };

  const renderRange = (
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    onChange: (next: number) => void,
    suffix = "",
    hint = "",
  ) => (
    <label className="motion-lab-range" title={hint || undefined}>
      <span>{label}<b>{value}{suffix}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );

  const missingBody = !parts;
  const earSwayParts = Object.keys(parts?.sways ?? {})
    .filter(name => EAR_SWAY_NAME_PATTERN.test(name))
    .sort((left, right) => left.localeCompare(right));
  const pivotPartOptions: Array<[string, string, boolean]> = [
    ["hair", "前髪", !!parts?.hair],
    ["hair_back", "後ろ髪", !!parts?.hairBack],
    ["arm_l", "左腕", !!parts?.armL],
    ["arm_r", "右腕", !!parts?.armR],
    ...earSwayParts.map((name, index) => [
      name,
      earSwayLabel(name, index, earSwayParts.length),
      true,
    ] as [string, string, boolean]),
  ];
  const templateRows = MOTION_LAB_TEMPLATE_LAYOUT[settings.engineFamily];
  const selectedTemplate = settings.templateName
    ? MOTION_LAB_TEMPLATES[settings.templateName]
    : null;

  return (
    <>
      <div className="motion-tune">
      <section className="motion-lab-preview-panel motion-tune-preview">
        <div className="motion-lab-preview-toolbar">
          <button className="btn btn-secondary" data-action-tone="edit" disabled={missingBody} onClick={() => setPlaying(prev => !prev)}>
            {playing ? "停止" : "再生"}
          </button>
          <button className="btn btn-secondary" data-action-tone="edit" disabled={missingBody} onClick={restartPlayback}>
            最初から
          </button>
          <div className="motion-lab-text-row motion-tune-text">
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              maxLength={80}
              placeholder="ひらがな・カタカナで入力（例: こんにちは)"
            />
            <button className="btn btn-secondary" data-action-tone="edit" disabled={missingBody} onClick={() => {
              setCustomTimeline({ ...motionLabTimelineFromText(text), text });
              onDirtyChange?.(true, "sequence");
              restartPlayback();
            }}>テキスト再生</button>
            {customTimeline && (
              <button className="btn btn-secondary" data-action-tone="edit" onClick={() => {
                setCustomTimeline(null);
                onDirtyChange?.(true, "sequence");
              }}>内蔵あいうえお</button>
            )}
          </div>
        </div>

        <div
          ref={previewStageRef}
          className="motion-lab-stage motion-tune-stage"
          onWheel={(event) => {
            if (!images) return;
            event.preventDefault();
            changePreviewZoom(previewZoom + (event.deltaY < 0 ? 0.25 : -0.25), {
              clientX: event.clientX,
              clientY: event.clientY,
            });
          }}
        >
          {parts ? (
            <>
              <canvas
                ref={canvasRef}
                className={pivotEditPart
                  ? "is-pivot-editing"
                  : previewPanning ? "is-panning" : "is-pannable"}
                style={{
                  transform: `translate3d(${previewPan.x}px, ${previewPan.y}px, 0) scale(${previewZoom})`,
                }}
                onPointerDown={(event) => {
                  if (pivotEditPart || event.button !== 0) return;
                  previewPanDragRef.current = {
                    pointerId: event.pointerId,
                    startClientX: event.clientX,
                    startClientY: event.clientY,
                    startPanX: previewPan.x,
                    startPanY: previewPan.y,
                  };
                  event.currentTarget.setPointerCapture(event.pointerId);
                  setPreviewPanning(true);
                  event.preventDefault();
                }}
                onPointerMove={(event) => {
                  const drag = previewPanDragRef.current;
                  if (!drag || drag.pointerId !== event.pointerId || pivotEditPart) return;
                  setPreviewPan(clampPreviewPan({
                    x: drag.startPanX + event.clientX - drag.startClientX,
                    y: drag.startPanY + event.clientY - drag.startClientY,
                  }, previewZoom));
                }}
                onPointerUp={(event) => {
                  if (previewPanDragRef.current?.pointerId !== event.pointerId) return;
                  previewPanDragRef.current = null;
                  setPreviewPanning(false);
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                  }
                }}
                onPointerCancel={() => {
                  previewPanDragRef.current = null;
                  setPreviewPanning(false);
                }}
                onLostPointerCapture={() => {
                  previewPanDragRef.current = null;
                  setPreviewPanning(false);
                }}
                onClick={(e) => {
                  if (!pivotEditPart || !parts) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (rect.width <= 0 || rect.height <= 0) return;
                  const x = ((e.clientX - rect.left) / rect.width) * parts.width;
                  const y = ((e.clientY - rect.top) / rect.height) * parts.height;
                  dispatch({
                    type: "set",
                    patch: { pivots: { ...settings.pivots, [pivotEditPart]: { x: Math.round(x), y: Math.round(y) } } },
                  });
                }}
              />
              {imagesLoading && <span className="motion-lab-placeholder">画像読込中...</span>}
              <div
                className="motion-tune-zoom-controls"
                title="ホイールで拡大・縮小。ドラッグでプレビューを移動できます"
                onWheel={(event) => event.stopPropagation()}
              >
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-action-tone="edit"
                  aria-label="プレビューを拡大"
                  disabled={!images || previewZoom >= 4}
                  onClick={() => changePreviewZoom(previewZoom + 0.25)}
                >＋</button>
                <span aria-live="polite">{Math.round(previewZoom * 100)}%</span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-action-tone="edit"
                  aria-label="プレビューを縮小"
                  disabled={!images || previewZoom <= 1}
                  onClick={() => changePreviewZoom(previewZoom - 0.25)}
                >－</button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  data-action-tone="edit"
                  disabled={!images || (previewZoom === 1 && previewPan.x === 0 && previewPan.y === 0)}
                  onClick={resetPreviewView}
                >リセット</button>
              </div>
            </>
          ) : (
            <span className="motion-lab-placeholder">
              {busy ? "素材を読み込んでいます..." : "RIFE補完の出力（04_spritalk_parts）を読み込むとプレビューが表示されます"}
            </span>
          )}
        </div>

      </section>

      <section className="motion-lab-control-panel motion-tune-controls">
        {parts?.missing.length ? (
          <div className="motion-lab-note warning">不足素材: {parts.missing.join(", ")}</div>
        ) : null}
        {parts?.warnings.length ? <div className="motion-lab-note">{parts.warnings.join(" / ")}</div> : null}
        {parts && (!parts.eyewhite || !parts.irides) ? (
          <div className="motion-lab-note warning">
            瞳の動きは未準備です。STEP 4を再編集して保存すると、RIFEを再生成せず利用できます。
          </div>
        ) : null}

        <div className="motion-lab-section motion-lab-simple">
          <div className="motion-lab-section-title">
            <strong>方式</strong>
            <div className="motion-lab-segmented motion-lab-engine-family">
              <button
                className={settings.engineFamily === "wave" ? "active" : ""}
                title="波のように揺れる髪と、弾む体の動きを組み合わせます"
                onClick={() => dispatch({ type: "applyEngineFamily", family: "wave" })}
              >ウェーブ式</button>
              <button
                className={settings.engineFamily === "springRig" ? "active" : ""}
                title="バネ物理の髪と、奥行きのある首振りを組み合わせます"
                onClick={() => dispatch({ type: "applyEngineFamily", family: "springRig" })}
              >スプリング式</button>
            </div>
          </div>
          <div className="motion-lab-section-title">
            <strong>テンプレート</strong>
            <div className="motion-lab-template-matrix">
              <div className="motion-lab-template-axis" aria-hidden="true">
                <small>動きの性格</small>
                <small>小さめ</small>
                <small>大きめ</small>
              </div>
              {templateRows.map(row => (
                <div className="motion-lab-template-row" key={row.label}>
                  <span>{row.label}</span>
                  {(["small", "large"] as const).map(size => {
                    const key = row[size];
                    const template = MOTION_LAB_TEMPLATES[key];
                    const sizeLabel = size === "small" ? "小さめ" : "大きめ";
                    return (
                      <button
                        key={key}
                        className={settings.templateName === key ? "active" : ""}
                        aria-label={`${row.label}、動き${sizeLabel}: ${template.label}`}
                        title={`${row.label}・動き${sizeLabel}: ${template.description}`}
                        onClick={() => dispatch({ type: "applyTemplate", key })}
                      >{template.label}</button>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
          {selectedTemplate && (
            <div className="motion-lab-note">
              {selectedTemplate.description}。適用後も各項目で微調整できます。
            </div>
          )}
          {renderRange("動きの強さ", Math.round(settings.intensity * 100), 50, 150, 5, (value) => {
            dispatch({ type: "applyIntensity", value: value / 100 });
          }, "%")}
          <div className="motion-lab-simple-subsection">
            <div className="motion-lab-section-title">
              <strong>口パク</strong>
              <div className="motion-lab-segmented three">
                <button className={settings.method === "baseline" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { method: "baseline" } })}>直接切替</button>
                <button className={settings.method === "smooth" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { method: "smooth" } })}>なめらか</button>
                <button className={settings.method === "bridge" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { method: "bridge" } })}>閉じ口経由</button>
              </div>
            </div>
            {renderRange("口を開く反応時間", settings.attackMs, 40, 180, 5, v => dispatch({ type: "set", patch: { attackMs: v } }), "ms", "口が開き始めるまでの追従時間。短いほど素早く開きます。")}
            {renderRange("口を閉じる反応時間", settings.releaseMs, 80, 260, 5, v => dispatch({ type: "set", patch: { releaseMs: v } }), "ms", "口が閉じ始めるまでの追従時間。短いほど素早く閉じます。")}
            {settings.method !== "baseline" && renderRange("母音の切替時間", settings.crossfadeMs, 0, 120, 5, v => dispatch({ type: "set", patch: { crossfadeMs: v } }), "ms", "母音画像を切り替える時間。短いほど切替がはっきりし、長いほど滑らかです。")}
            {renderRange("弱い発音の開き抑制", Math.round(settings.restBias * 100), 0, 100, 1, v => dispatch({ type: "set", patch: { restBias: v / 100 } }), "%", "弱い発音で口をどれだけ閉じ気味にするか。大きいほど小さな声では開きにくくなります。")}
            {renderRange("開き量のなめらかさ", Math.round(settings.shapeSmoothing * 100), 0, 100, 1, v => dispatch({ type: "set", patch: { shapeSmoothing: v / 100 } }), "%", "口の開き量の変化をならす強さ。大きいほど滑らかですが、反応は穏やかになります。")}
            {settings.method === "bridge" && renderRange("閉じ口の経由量", Math.round(settings.bridgeBias * 100), 0, 85, 1, v => dispatch({ type: "set", patch: { bridgeBias: v / 100 } }), "%", "母音同士の切替で閉じ口を経由する量。大きいほど一度閉じる動きが強くなります。")}
          </div>
          <div className="motion-lab-section-title">
            <strong>エフェクト</strong>
            <div className="motion-lab-segmented">
              <button onClick={() => dispatch({ type: "allEffects", value: true })}>すべてON</button>
              <button onClick={() => dispatch({ type: "allEffects", value: false })}>すべてOFF</button>
            </div>
          </div>
          <div className="motion-lab-effect-list">
            {MOTION_LAB_EFFECT_DEFS.filter(def => {
              if (def.key === "arm" || def.key === "lift") return !!(parts?.armL || parts?.armR);
              if (def.key === "chest") return !!parts?.body;
              if (def.key === "gaze") return !!(parts?.eyewhite && parts?.irides);
              if (def.key === "irisBreath" || def.key === "wetness") return !!(parts?.eyewhite && parts?.irides);
              if (def.key === "brow") return !!parts?.eyebrow;
              if (def.key === "earTwitch") return earSwayParts.length > 0;
              if (def.key === "hairBack") return !!parts?.hairBack;
              if (def.key === "blink") return (parts?.eyeFrames.length ?? 0) > 1;
              return true;
            }).map(def => (
              <div key={def.key} className="motion-lab-effect-row" title={def.hint}>
                <label>
                  <input
                    type="checkbox"
                    checked={settings.effects[def.key]}
                    onChange={(e) => dispatch({ type: "setEffect", key: def.key, value: e.target.checked })}
                  />
                  <span>{def.label}</span>
                </label>
                {settings.effects[def.key] && effectSliders[def.key] ? (
                  <>
                    <input
                      type="range"
                      className="motion-lab-effect-slider"
                      min={effectSliders[def.key]!.min}
                      max={effectSliders[def.key]!.max}
                      step={effectSliders[def.key]!.step}
                      value={effectSliders[def.key]!.value}
                      onChange={(e) => effectSliders[def.key]!.set(Number(e.target.value))}
                    />
                    <small className="motion-lab-effect-value">
                      {effectSliders[def.key]!.format(effectSliders[def.key]!.value)}
                    </small>
                  </>
                ) : (
                  <span className="motion-lab-effect-slider-spacer" />
                )}
                <button
                  className="motion-lab-effect-solo"
                  title={`この効果だけONにして単体で体感する: ${def.hint}`}
                  onClick={() => dispatch({ type: "soloEffect", key: def.key })}
                >ソロ</button>
              </div>
            ))}
          </div>
          {(parts?.layerOrder?.length ?? 0) === 0 && (
            <div className="motion-lab-simple-toggles">
              <label>
                <input
                  type="checkbox"
                  checked={settings.armBehindBody}
                  onChange={(e) => dispatch({ type: "set", patch: { armBehindBody: e.target.checked } })}
                />
                腕を体の後ろ
              </label>
            </div>
          )}
        </div>

        <details className="motion-lab-advanced">
          <summary>詳細調整（髪・回転軸・獣耳・腕）</summary>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>髪・身体</strong>
              <div className="motion-lab-segmented three">
                <button className={settings.preset === "calm" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { preset: "calm" } })}>おとなしめ</button>
                <button className={settings.preset === "normal" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { preset: "normal" } })}>ふつう</button>
                <button className={settings.preset === "lively" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { preset: "lively" } })}>元気</button>
              </div>
            </div>
            <div className="motion-lab-segmented">
              <button className={settings.layerMode === "spring" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { layerMode: "spring" } })}>レイヤー全体</button>
              <button className={settings.layerMode === "mesh" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { layerMode: "mesh" } })}>毛先をしならせる</button>
            </div>
            <div className="motion-lab-segmented">
              <button className={settings.hairEngine === "spring" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { hairEngine: "spring" } })}>バネ物理</button>
              <button className={settings.hairEngine === "wave" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { hairEngine: "wave" } })}>波揺れ</button>
            </div>
            {settings.hairEngine === "wave" &&
              renderRange("波の強さ", Math.round(settings.hairWaveStrength * 100), 0, 200, 5, v => dispatch({ type: "set", patch: { hairWaveStrength: v / 100 } }), "%")}
            {settings.layerMode === "mesh" && (
              <div className="motion-lab-segmented">
                <button className={!settings.strandsEnabled ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { strandsEnabled: false } })}>一枚で揺らす</button>
                <button className={settings.strandsEnabled ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { strandsEnabled: true } })}>房ごとに揺らす</button>
              </div>
            )}
            {renderRange("バネの硬さ", settings.hairK, 10, 200, 5, v => dispatch({ type: "set", patch: { hairK: v } }))}
            {renderRange("揺れの収まり", settings.hairC, 1, 30, 1, v => dispatch({ type: "set", patch: { hairC: v } }))}
            {renderRange("風の強さ", Number((settings.hairWind * 1000).toFixed(0)), 0, 60, 2, v => dispatch({ type: "set", patch: { hairWind: v / 1000 } }), "‰")}
            {renderRange("体への追従", Number((settings.hairDrive * 100).toFixed(0)), 0, 20, 1, v => dispatch({ type: "set", patch: { hairDrive: v / 100 } }), "%")}
            <div className="motion-lab-note">
              バネの硬さは戻ろうとする力、揺れの収まりは余韻の短さです。体への追従を上げると、体の動きに合わせて髪が大きく動きます。
            </div>
          </div>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>回転軸・可動域</strong>
            </div>
            <div className="motion-lab-segmented">
              {pivotPartOptions.filter(([, , exists]) => exists).map(([part, label]) => (
                <button
                  key={part}
                  className={pivotEditPart === part ? "active" : ""}
                  onClick={() => setPivotEditPart(prev => (prev === part ? null : part))}
                >{label}</button>
              ))}
            </div>
            {pivotEditPart && (
              <>
                {renderRange(
                  "可動域",
                  settings.rangesDeg[pivotEditPart] ?? 0,
                  0,
                  90,
                  1,
                  v => dispatch({ type: "set", patch: { rangesDeg: { ...settings.rangesDeg, [pivotEditPart]: v } } }),
                  "°",
                )}
                {renderRange(
                  "揺れ幅",
                  Math.round((settings.swingScale[pivotEditPart] ?? 1) * 100),
                  0,
                  300,
                  10,
                  v => dispatch({ type: "set", patch: { swingScale: { ...settings.swingScale, [pivotEditPart]: v / 100 } } }),
                  "%",
                )}
                <div className="motion-lab-text-row motion-lab-pivot-status">
                  <small>
                    回転軸: {settings.pivots[pivotEditPart]
                      ? `${settings.pivots[pivotEditPart].x}, ${settings.pivots[pivotEditPart].y}`
                      : "自動推定"}
                  </small>
                  {settings.pivots[pivotEditPart] && (
                    <button className="btn btn-secondary" data-action-tone="edit" onClick={() => {
                      const next = { ...settings.pivots };
                      delete next[pivotEditPart];
                      dispatch({ type: "set", patch: { pivots: next } });
                    }}>自動に戻す</button>
                  )}
                </div>
              </>
            )}
            <div className="motion-lab-note">
              パーツを選ぶとプレビューに回転軸マーカー（＋印）が出ます。プレビューをクリックすると回転軸を移動できます。
              可動域=回転角の上限（±度、0=制限なし）。揺れ幅=このパーツだけの振れ倍率。
              前髪・後ろ髪は回転軸のY位置、獣耳は指定した付け根が「揺れの根元」として効きます。
            </div>
          </div>

          {earSwayParts.length > 0 && (
            <div className="motion-lab-section">
              <div className="motion-lab-section-title">
                <strong>獣耳の動き方</strong>
                <small className="motion-lab-ear-material-note">
                  ON/OFFと強さは上の「エフェクト」で調整します。
                </small>
              </div>
              <div className="motion-lab-segmented three">
                <button
                  className={settings.earTwitchMode === "bounce" ? "active" : ""}
                  onClick={() => dispatch({ type: "set", patch: { earTwitchMode: "bounce" } })}
                  title="回転を加えず、耳全体を上へ軽く跳ねさせます"
                >上にピコッ</button>
                <button
                  className={settings.earTwitchMode === "tilt" ? "active" : ""}
                  onClick={() => dispatch({ type: "set", patch: { earTwitchMode: "tilt" } })}
                  title="指定した付け根を中心に、耳を左右へ傾けます"
                >左右にピコッ</button>
                <button
                  className={settings.earTwitchMode === "double" ? "active" : ""}
                  onClick={() => dispatch({ type: "set", patch: { earTwitchMode: "double" } })}
                  title="跳ねと傾きを組み合わせ、短く二度ピコッと動かします"
                >2回ピコッ</button>
              </div>
              <small className="motion-lab-ear-material-note">
                耳の付け根は、上の「回転軸・可動域」で獣耳を選び、プレビュー上の根元をクリックして指定します。
              </small>
              {earSwayParts.length === 1 && (
                <small className="motion-lab-ear-material-note">
                  一体の耳素材は両耳が一緒に動きます。左右別々に動かすには sway_ear_l / sway_ear_r が必要です。
                </small>
              )}
            </div>
          )}

          {(parts?.armL || parts?.armR) ? (
            <div className="motion-lab-section">
              <div className="motion-lab-section-title">
                <strong>腕揺れ</strong>
              </div>
              {renderRange("腕の振れ幅", Number((settings.armMaxAngle * 100).toFixed(0)), 0, 60, 1, v => dispatch({ type: "set", patch: { armMaxAngle: v / 100 } }), "×0.01rad")}
              {renderRange("回転軸の高さ", Math.round(settings.armPivotRatio * 100), 0, 60, 2, v => dispatch({ type: "set", patch: { armPivotRatio: v / 100 } }), "%")}
              <div className="motion-lab-note">振れ幅は腕が左右へ動ける上限、回転軸の高さは肩を基準にした揺れ始めの位置です。</div>
            </div>
          ) : null}
        </details>

        </section>
      </div>

      {exportChoiceOpen && (
        <div
          className="motion-lab-export-choice-overlay"
          role="dialog"
          aria-label="書き出す形式を選択"
          onClick={() => { if (!busy) setExportChoiceOpen(false); }}
        >
          <div className="motion-lab-export-choice-modal" onClick={event => event.stopPropagation()}>
            <div className="motion-lab-export-choice-header">
              <strong>書き出す形式を選択</strong>
              <button
                className="motion-lab-export-choice-close"
                disabled={busy}
                onClick={() => setExportChoiceOpen(false)}
                aria-label="閉じる"
              >×</button>
            </div>

            <div className="motion-lab-export-choice-option">
              <div className="motion-lab-export-choice-option-heading">
                <strong>SpriTalk向けに保存</strong>
                <span className="motion-lab-export-choice-tag">通常はこちら</span>
              </div>
              <p>基本画像とPachiPakuGen用モーション設定をSpriTalkの読み込み用フォルダへ保存します。</p>
              <button
                className="btn btn-primary"
                disabled={busy || !parts}
                onClick={() => void chooseSpritalkExport()}
              >
                {spritalkSaving ? "保存中..." : "SpriTalk向けに保存"}
              </button>
            </div>

            <div className="motion-lab-export-choice-option">
              <div
                className="motion-lab-export-choice-option-heading motion-lab-export-choice-option-toggle"
                role="button"
                tabIndex={0}
                onClick={() => setLoopExportPanelOpen(open => !open)}
                onKeyDown={event => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setLoopExportPanelOpen(open => !open);
                  }
                }}
              >
                <strong>ループ素材を書き出す</strong>
                <span className="motion-lab-export-choice-chevron">{loopExportPanelOpen ? "▲" : "▼"}</span>
              </div>
              <p>つなぎ目なく繰り返せるアニメ素材（PNG連番/APNG）を出力します。動画編集やOBSでのループ再生向けです。</p>
              {loopExportPanelOpen && (
                <div className="motion-lab-loop-export-panel">
                  <div className="motion-lab-loop-export-grid">
                    <label>
                      <span>ループ長</span>
                      <select
                        value={loopExportConfig.periodMs}
                        disabled={busy}
                        onChange={event => setLoopExportConfig({ ...loopExportConfig, periodMs: Number(event.target.value) })}
                      >
                        <option value={7200}>7.2秒（軽量）</option>
                        <option value={14400}>14.4秒（推奨）</option>
                        <option value={21600}>21.6秒（変化が多い）</option>
                      </select>
                    </label>
                    <label>
                      <span>口</span>
                      <select
                        value={loopExportConfig.mouth}
                        disabled={busy}
                        onChange={event => setLoopExportConfig({ ...loopExportConfig, mouth: event.target.value as "silent" | MotionLabMouthKey })}
                      >
                        <option value="silent">口パクなし（待機）</option>
                        {MOTION_LAB_VOWEL_KEYS.map(vowel => (
                          <option key={vowel} value={vowel}>「{MOTION_LAB_MOUTH_LABELS[vowel]}」開閉ループ</option>
                        ))}
                      </select>
                    </label>
                    <div className="motion-lab-loop-export-formats">
                      <label>
                        <input
                          type="checkbox"
                          checked={loopExportConfig.frames}
                          disabled={busy}
                          onChange={event => setLoopExportConfig({ ...loopExportConfig, frames: event.target.checked })}
                        />
                        <span>PNG連番</span>
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={loopExportConfig.apng}
                          disabled={busy}
                          onChange={event => setLoopExportConfig({ ...loopExportConfig, apng: event.target.checked })}
                        />
                        <span>APNG</span>
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={loopExportConfig.gif}
                          disabled={busy}
                          onChange={event => setLoopExportConfig({ ...loopExportConfig, gif: event.target.checked })}
                        />
                        <span>AGIF</span>
                      </label>
                    </div>
                    <div className="motion-lab-loop-export-hints">
                      {loopExportConfig.frames && (
                        <details className="motion-lab-loop-export-hint">
                          <summary>PNG連番→WebMへの変換方法</summary>
                          <p>
                            PNG連番は最も画質が良く、そのままではファイル数が多いだけですが、
                            お手元にffmpegがあれば1コマンドで透過付きWebMへ変換できます
                            (OBS等の透過ループ素材として一番軽くて扱いやすい形式です)。
                          </p>
                          <code>ffmpeg -framerate 30 -i frames/%04d.png -c:v libvpx-vp9 -pix_fmt yuva420p loop.webm</code>
                        </details>
                      )}
                      {loopExportConfig.apng && (
                        <p className="motion-lab-loop-export-hint-note">
                          APNGは画質・透過とも最良ですが、揺れが画面全体に広がる素材では差分圧縮の効きが弱く、ファイルサイズが大きくなりがちです。
                        </p>
                      )}
                      {loopExportConfig.gif && (
                        <p className="motion-lab-loop-export-hint-note">
                          AGIFはAPNGよりかなり軽くなりますが、256色までの制限で陰影にバンディングが出たり、透過が完全な0/100%のみのため輪郭がやや硬くなります。
                        </p>
                      )}
                    </div>
                  </div>
                  {loopExportProgress ? (
                    <div className="motion-lab-loop-export-progress">
                      <div className="motion-lab-loop-export-progress-main">
                        <span>
                          {loopExportProgress.phase}
                          {!loopExportProgress.indeterminate && loopExportProgress.total > 0
                            ? `: ${loopExportProgress.current}/${loopExportProgress.total}`
                            : ""}
                        </span>
                        <div
                          className={
                            loopExportProgress.indeterminate
                              ? "motion-lab-loop-export-bar is-indeterminate"
                              : "motion-lab-loop-export-bar"
                          }
                          role="progressbar"
                          aria-valuemin={0}
                          aria-valuemax={loopExportProgress.total || 100}
                          aria-valuenow={loopExportProgress.indeterminate ? undefined : loopExportProgress.current}
                          aria-label={loopExportProgress.phase}
                        >
                          <div
                            className="motion-lab-loop-export-bar-fill"
                            style={{
                              width: loopExportProgress.indeterminate
                                ? undefined
                                : `${Math.min(100, (loopExportProgress.current / Math.max(1, loopExportProgress.total)) * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                      <button className="btn btn-secondary" onClick={() => { loopExportCancelRef.current = true; }}>キャンセル</button>
                    </div>
                  ) : (
                    <div className="motion-lab-loop-export-actions">
                      <button
                        className="btn btn-secondary"
                        disabled={busy || !parts || !images}
                        onClick={() => void exportMotionLoop()}
                      >
                        ループ素材を書き出す（30fps）
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
