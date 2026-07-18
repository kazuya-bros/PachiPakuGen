import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { MOTION_LAB_MOUTH_KEYS } from "./constants";
import {
  createMotionLabPhysics,
  drawMotionLabScene,
  loadMotionLabImage,
  prepareMotionLabCanvas,
  resetMotionLabRuntime,
} from "./render";
import type {
  MotionLabImageSet,
  MotionLabManifestResult,
  MotionLabMouthKey,
  MotionLabMouthRuntime,
  MotionLabPartsResult,
} from "./types";
import { toRenderSettings, useMotionLabSettings } from "./useMotionLabSettings";
import { useMicrophoneLevel } from "./useMicrophoneLevel";

export interface MotionLiveViewProps {
  partsDir: string | null;
  onBack: () => void;
  onError?: (message: string) => void;
  onNotify?: (message: string) => void;
}

type ChromaBackground = "green" | "blue" | "magenta" | "dark";
type LiveInput = () => { mouth: MotionLabMouthKey; energy: number; openness: number };

const CHROMA_BACKGROUNDS: Record<ChromaBackground, { label: string; color: string }> = {
  green: { label: "グリーン", color: "#00ff00" },
  blue: { label: "ブルー", color: "#0000ff" },
  magenta: { label: "マゼンタ", color: "#ff00ff" },
  dark: { label: "ダーク", color: "#111827" },
};

const MOUTH_LABELS: Record<MotionLabMouthKey, string> = {
  closed: "閉じ口",
  a: "あ",
  i: "い",
  u: "う",
  e: "え",
  o: "お",
};

const MOTION_LIVE_STYLES = `
.motion-live-view {
  position: relative;
  display: grid;
  grid-template-columns: minmax(250px, 320px) minmax(0, 1fr);
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  color: var(--text, #152034);
  background: var(--bg, #eef3fa);
}
.motion-live-controls {
  position: relative;
  z-index: 2;
  display: flex;
  min-height: 0;
  flex-direction: column;
  gap: 14px;
  padding: 18px;
  overflow: auto;
  border-right: 1px solid var(--border, #bdcbe0);
  background: var(--surface, #fff);
}
.motion-live-header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.motion-live-header h2 { margin: 0; font-size: 1.2rem; }
.motion-live-section { display: grid; gap: 8px; }
.motion-live-section > label, .motion-live-control-label { font-size: .82rem; font-weight: 700; }
.motion-live-select, .motion-live-button {
  min-height: 38px;
  padding: 7px 10px;
  border: 1px solid var(--border-strong, #8da5c7);
  border-radius: 7px;
  color: inherit;
  background: var(--surface-bg, #fff);
}
.motion-live-button { cursor: pointer; font-weight: 700; }
.motion-live-button.primary { color: #fff; border-color: #1764d7; background: #1764d7; }
.motion-live-button.danger { color: #fff; border-color: #c93655; background: #c93655; }
.motion-live-button:disabled { cursor: default; opacity: .55; }
.motion-live-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.motion-live-meter { display: grid; gap: 5px; }
.motion-live-meter-row { display: flex; justify-content: space-between; gap: 10px; font-size: .78rem; }
.motion-live-meter-track { height: 11px; overflow: hidden; border-radius: 99px; background: #d7dfec; }
.motion-live-meter-fill { height: 100%; border-radius: inherit; background: #1764d7; transition: width 30ms linear; }
.motion-live-meter-fill.raw { background: #71819a; }
.motion-live-range { display: grid; grid-template-columns: minmax(0, 1fr) 62px; align-items: center; gap: 8px; }
.motion-live-range input[type="range"] { width: 100%; min-width: 0; }
.motion-live-range output { text-align: right; font-size: .78rem; font-variant-numeric: tabular-nums; }
.motion-live-colors { display: grid; grid-template-columns: repeat(2, 1fr); gap: 7px; }
.motion-live-color {
  display: flex;
  align-items: center;
  gap: 7px;
  min-height: 34px;
  padding: 5px 8px;
  border: 2px solid transparent;
  border-radius: 6px;
  color: inherit;
  background: var(--surface-bg, #fff);
  cursor: pointer;
}
.motion-live-color.active { border-color: #1764d7; }
.motion-live-color-swatch { width: 17px; height: 17px; flex: 0 0 auto; border: 1px solid rgba(0,0,0,.35); border-radius: 3px; }
.motion-live-message { margin: 0; padding: 9px 10px; border-left: 3px solid #c93655; background: rgba(201,54,85,.1); font-size: .8rem; }
.motion-live-note { margin: 0; color: var(--muted, #52647e); font-size: .75rem; line-height: 1.55; }
.motion-live-stage-shell {
  position: relative;
  display: grid;
  min-width: 0;
  min-height: 0;
  place-items: center;
  overflow: hidden;
  padding: 12px;
  background: var(--surface-sunken, #e6edf7);
}
.motion-live-stage {
  position: relative;
  display: grid;
  max-width: 100%;
  max-height: 100%;
  min-width: 0;
  min-height: 0;
  place-items: center;
  overflow: hidden;
  border: 2px solid #1764d7;
  box-shadow: 0 7px 24px rgba(20, 38, 66, .16);
}
.motion-live-capture-frame-label {
  position: absolute;
  z-index: 3;
  top: 18px;
  left: 18px;
  padding: 4px 7px;
  border-radius: 4px;
  color: #fff;
  background: rgba(23, 100, 215, .88);
  font-size: .68rem;
  font-weight: 800;
  pointer-events: none;
}
.motion-live-stage.is-draggable { cursor: grab; touch-action: none; }
.motion-live-stage.is-dragging { cursor: grabbing; }
.motion-live-stage canvas {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  pointer-events: none;
  will-change: transform;
  transition: transform .12s ease-out;
}
.motion-live-stage.is-dragging canvas,
.motion-live-view.is-capture-only .motion-live-stage canvas { transition: none; }
.motion-live-loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #fff;
  background: rgba(0,0,0,.42);
  font-weight: 700;
}
.motion-live-view.is-capture-only {
  position: fixed;
  inset: 0;
  z-index: 2147483000;
  display: block;
  width: 100vw;
  height: 100vh;
  background: transparent;
}
.motion-live-view.is-capture-only .motion-live-controls { display: none; }
.motion-live-view.is-capture-only .motion-live-stage-shell {
  width: 100vw;
  height: 100vh;
  padding: 0;
  background: transparent;
}
.motion-live-view.is-capture-only .motion-live-stage {
  width: 100%;
  height: 100%;
  border: 0;
  box-shadow: none;
}
.motion-live-view.is-capture-only .motion-live-capture-frame-label { display: none; }
.motion-live-capture-exit-zone {
  position: fixed;
  z-index: 2147483640;
  top: 0;
  right: 0;
  display: flex;
  width: 230px;
  height: 76px;
  align-items: flex-start;
  justify-content: flex-end;
  padding: 14px;
}
.motion-live-capture-exit {
  min-height: 40px;
  padding: 8px 13px;
  border: 1px solid rgba(255,255,255,.72);
  border-radius: 7px;
  color: #fff;
  background: rgba(12,20,34,.78);
  box-shadow: 0 4px 16px rgba(0,0,0,.28);
  font: inherit;
  font-weight: 800;
  cursor: pointer;
  opacity: 0;
  transform: translateY(-5px);
  transition: opacity .15s ease, transform .15s ease;
}
.motion-live-capture-exit.visible,
.motion-live-capture-exit-zone:hover .motion-live-capture-exit,
.motion-live-capture-exit:focus-visible {
  opacity: 1;
  transform: translateY(0);
}
@media (max-width: 760px) {
  .motion-live-view:not(.is-capture-only) { grid-template-columns: 1fr; grid-template-rows: auto minmax(320px, 1fr); }
  .motion-live-controls { max-height: 46vh; border-right: 0; border-bottom: 1px solid var(--border, #bdcbe0); }
}
`;

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

async function decodeMotionLabImages(parts: MotionLabPartsResult): Promise<MotionLabImageSet> {
  const mouthEntries = await Promise.all(
    MOTION_LAB_MOUTH_KEYS.map(async key => [
      key,
      await Promise.all((parts.mouths[key] ?? []).map(loadMotionLabImage)),
    ] as const),
  );
  const swayEntries = await Promise.all(
    Object.entries(parts.sways).map(async ([name, source]) => [
      name,
      await loadMotionLabImage(source),
    ] as const),
  );
  const linkedPartEntries = await Promise.all(
    Object.entries(parts.linkedParts ?? {}).map(async ([name, linked]) => [
      name,
      { parent: linked.parent, image: await loadMotionLabImage(linked.image) },
    ] as const),
  );

  return {
    body: await loadMotionLabImage(parts.body),
    hair: parts.hair ? await loadMotionLabImage(parts.hair) : null,
    hairBack: parts.hairBack ? await loadMotionLabImage(parts.hairBack) : null,
    armL: parts.armL ? await loadMotionLabImage(parts.armL) : null,
    armR: parts.armR ? await loadMotionLabImage(parts.armR) : null,
    chest: parts.chest ? await loadMotionLabImage(parts.chest) : null,
    sways: Object.fromEntries(swayEntries),
    linkedParts: Object.fromEntries(linkedPartEntries),
    eyebrow: parts.eyebrow ? await loadMotionLabImage(parts.eyebrow) : null,
    eyewhite: parts.eyewhite ? await loadMotionLabImage(parts.eyewhite) : null,
    irides: parts.irides ? await loadMotionLabImage(parts.irides) : null,
    highlight: parts.highlight ? await loadMotionLabImage(parts.highlight) : null,
    eyeFrames: await Promise.all(parts.eyeFrames.map(loadMotionLabImage)),
    mouths: Object.fromEntries(mouthEntries) as Partial<Record<MotionLabMouthKey, HTMLImageElement[]>>,
  };
}

export function MotionLiveView({
  partsDir,
  onBack,
  onError,
  onNotify,
}: MotionLiveViewProps) {
  const [parts, setParts] = useState<MotionLabPartsResult | null>(null);
  const [images, setImages] = useState<MotionLabImageSet | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [captureOnly, setCaptureOnly] = useState(false);
  const [captureControlsVisible, setCaptureControlsVisible] = useState(false);
  const [background, setBackground] = useState<ChromaBackground>("green");
  const [selectedOpenMouth, setSelectedOpenMouth] = useState<MotionLabMouthKey>("a");
  const [displayScale, setDisplayScale] = useState(100);
  const [displayOffsetX, setDisplayOffsetX] = useState(0);
  const [displayOffsetY, setDisplayOffsetY] = useState(0);
  const [stageDragging, setStageDragging] = useState(false);
  const [previewFrame, setPreviewFrame] = useState(() => ({
    width: 1,
    height: 1,
    captureWidth: Math.max(1, window.innerWidth),
    captureHeight: Math.max(1, window.innerHeight),
  }));
  const [micBusy, setMicBusy] = useState(false);
  const [settings, settingsDispatch] = useMotionLabSettings();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageShellRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<MotionLabMouthRuntime>(createRuntime());
  const openAmountRef = useRef(0);
  const selectedOpenMouthRef = useRef<MotionLabMouthKey>("a");
  const notifyRef = useRef(onNotify);
  const errorRef = useRef(onError);
  const captureControlsTimerRef = useRef<number | null>(null);
  const stageDragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startOffsetX: number;
    startOffsetY: number;
    width: number;
    height: number;
  } | null>(null);

  const microphone = useMicrophoneLevel();

  notifyRef.current = onNotify;
  errorRef.current = onError;
  openAmountRef.current = microphone.openAmount;
  selectedOpenMouthRef.current = selectedOpenMouth;

  const availableOpenMouths = useMemo(
    () => MOTION_LAB_MOUTH_KEYS.filter(key => key !== "closed" && (parts?.mouths[key]?.length ?? 0) > 0),
    [parts],
  );

  useEffect(() => {
    if (availableOpenMouths.includes(selectedOpenMouth)) return;
    setSelectedOpenMouth(availableOpenMouths.includes("a") ? "a" : availableOpenMouths[0] ?? "closed");
  }, [availableOpenMouths, selectedOpenMouth]);

  useEffect(() => {
    if (!partsDir) {
      setParts(null);
      setImages(null);
      setLoadError("");
      return;
    }
    let cancelled = false;
    setLoading(true);
    setLoadError("");
    setParts(null);
    setImages(null);

    void (async () => {
      try {
        const loadedParts = await invoke<MotionLabPartsResult>("load_motion_lab_parts", { dir: partsDir });
        try {
          const saved = await invoke<MotionLabManifestResult>("load_motion_lab_manifest", {
            sourceDir: loadedParts.sourceDir,
          });
          if (!cancelled) settingsDispatch({ type: "applyManifest", manifest: saved.manifest });
        } catch {
          // A saved manifest is optional; defaults remain usable for a first live launch.
        }
        const loadedImages = await decodeMotionLabImages(loadedParts);
        if (cancelled) return;
        resetMotionLabRuntime(runtimeRef.current);
        setParts(loadedParts);
        setImages(loadedImages);
        notifyRef.current?.(`ライブ表示用の素材を読み込みました: ${loadedParts.sourceDir}`);
      } catch (cause) {
        if (cancelled) return;
        const message = cause instanceof Error ? cause.message : String(cause);
        setLoadError(message);
        errorRef.current?.(message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [partsDir, settingsDispatch]);

  useEffect(() => {
    if (microphone.error) errorRef.current?.(microphone.error);
  }, [microphone.error]);

  useLayoutEffect(() => {
    if (captureOnly) return;
    const shell = stageShellRef.current;
    if (!shell) return;

    const updateFrame = () => {
      const computed = window.getComputedStyle(shell);
      const availableWidth = Math.max(
        1,
        shell.clientWidth - Number.parseFloat(computed.paddingLeft) - Number.parseFloat(computed.paddingRight),
      );
      const availableHeight = Math.max(
        1,
        shell.clientHeight - Number.parseFloat(computed.paddingTop) - Number.parseFloat(computed.paddingBottom),
      );
      const captureWidth = Math.max(1, window.innerWidth);
      const captureHeight = Math.max(1, window.innerHeight);
      const fit = Math.min(availableWidth / captureWidth, availableHeight / captureHeight);
      const width = Math.max(1, Math.floor(captureWidth * fit));
      const height = Math.max(1, Math.floor(captureHeight * fit));
      setPreviewFrame(previous => (
        previous.width === width
        && previous.height === height
        && previous.captureWidth === captureWidth
        && previous.captureHeight === captureHeight
          ? previous
          : { width, height, captureWidth, captureHeight }
      ));
    };

    const observer = new ResizeObserver(updateFrame);
    observer.observe(shell);
    window.addEventListener("resize", updateFrame);
    updateFrame();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateFrame);
    };
  }, [captureOnly]);

  const liveInput = useCallback<LiveInput>(() => {
    const openness = Math.max(0, Math.min(1, openAmountRef.current));
    return {
      mouth: openness > 0.01 ? selectedOpenMouthRef.current : "closed",
      energy: openness,
      openness,
    };
  }, []);

  const renderSettings = useMemo(() => {
    type RenderExtras = Parameters<typeof toRenderSettings>[1] & { liveInput: LiveInput };
    const extras: RenderExtras = { pivotEditPart: null, liveInput };
    const result = toRenderSettings(settings, extras);
    return Object.assign(result, { liveInput });
  }, [liveInput, settings]);

  useEffect(() => {
    if (!parts || !images) return;
    const context = prepareMotionLabCanvas(canvasRef.current, parts.width, parts.height);
    if (!context) return;
    const startedAt = performance.now();
    let animationFrame = 0;
    const draw = (now: number) => {
      drawMotionLabScene(context, parts, images, runtimeRef.current, now - startedAt, renderSettings);
      animationFrame = window.requestAnimationFrame(draw);
    };
    animationFrame = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [images, parts, renderSettings]);

  useEffect(() => {
    if (!captureOnly) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const leaveCapture = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setCaptureOnly(false);
    };
    window.addEventListener("keydown", leaveCapture, true);
    window.addEventListener("keyup", leaveCapture, true);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", leaveCapture, true);
      window.removeEventListener("keyup", leaveCapture, true);
    };
  }, [captureOnly]);

  useEffect(() => {
    if (captureControlsTimerRef.current !== null) {
      window.clearTimeout(captureControlsTimerRef.current);
      captureControlsTimerRef.current = null;
    }
    if (!captureOnly) {
      setCaptureControlsVisible(false);
      return;
    }
    setCaptureControlsVisible(true);
    captureControlsTimerRef.current = window.setTimeout(() => {
      setCaptureControlsVisible(false);
      captureControlsTimerRef.current = null;
    }, 2400);
    return () => {
      if (captureControlsTimerRef.current !== null) {
        window.clearTimeout(captureControlsTimerRef.current);
        captureControlsTimerRef.current = null;
      }
    };
  }, [captureOnly]);

  const resetDisplayComposition = () => {
    setDisplayScale(100);
    setDisplayOffsetX(0);
    setDisplayOffsetY(0);
  };

  const handleStagePointerDown = (event: ReactPointerEvent<HTMLElement>) => {
    if (captureOnly || !images) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    stageDragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startOffsetX: displayOffsetX,
      startOffsetY: displayOffsetY,
      width: Math.max(1, bounds.width),
      height: Math.max(1, bounds.height),
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setStageDragging(true);
  };

  const handleStagePointerMove = (event: ReactPointerEvent<HTMLElement>) => {
    const drag = stageDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const nextX = drag.startOffsetX + ((event.clientX - drag.startX) / drag.width) * 100;
    const nextY = drag.startOffsetY + ((event.clientY - drag.startY) / drag.height) * 100;
    setDisplayOffsetX(Math.max(-100, Math.min(100, Math.round(nextX * 10) / 10)));
    setDisplayOffsetY(Math.max(-100, Math.min(100, Math.round(nextY * 10) / 10)));
  };

  const finishStageDrag = (event: ReactPointerEvent<HTMLElement>) => {
    if (stageDragRef.current?.pointerId !== event.pointerId) return;
    stageDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setStageDragging(false);
  };

  const handleStageWheel = (event: ReactWheelEvent<HTMLElement>) => {
    if (captureOnly || !images || event.deltaY === 0) return;
    event.preventDefault();
    const direction = event.deltaY < 0 ? 1 : -1;
    setDisplayScale(previous => Math.max(25, Math.min(250, previous + direction * 5)));
  };

  const handleMicrophoneStart = async () => {
    setMicBusy(true);
    const started = await microphone.start();
    setMicBusy(false);
    if (started) notifyRef.current?.("マイク連動を開始しました");
  };

  const handleMicrophoneStop = () => {
    microphone.stop();
    notifyRef.current?.("マイク連動を停止しました");
  };

  const handleBack = () => {
    microphone.stop();
    setCaptureOnly(false);
    onBack();
  };

  const backgroundColor = CHROMA_BACKGROUNDS[background].color;
  const visibleError = loadError || microphone.error;
  const stageStyle = captureOnly
    ? { backgroundColor }
    : { backgroundColor, width: previewFrame.width, height: previewFrame.height };

  return (
    <div
      className={`motion-live-view${captureOnly ? " is-capture-only" : ""}`}
    >
      <style>{MOTION_LIVE_STYLES}</style>
      <aside className="motion-live-controls" aria-label="ライブ表示設定">
        <header className="motion-live-header">
          <h2>配信表示</h2>
          <button type="button" className="motion-live-button" onClick={handleBack}>戻る</button>
        </header>

        <section className="motion-live-section">
          <label htmlFor="motion-live-device">マイク</label>
          <select
            id="motion-live-device"
            className="motion-live-select"
            value={microphone.selectedDeviceId}
            disabled={microphone.active || micBusy}
            onChange={event => microphone.setSelectedDeviceId(event.target.value)}
          >
            {microphone.devices.length === 0 && <option value="">既定のマイク</option>}
            {microphone.devices.map(device => (
              <option key={device.deviceId || device.label} value={device.deviceId}>{device.label}</option>
            ))}
          </select>
          <div className="motion-live-actions">
            <button
              type="button"
              className="motion-live-button primary"
              disabled={microphone.active || micBusy}
              onClick={() => void handleMicrophoneStart()}
            >
              {micBusy ? "開始中…" : "マイク開始"}
            </button>
            <button
              type="button"
              className="motion-live-button danger"
              disabled={!microphone.active || micBusy}
              onClick={handleMicrophoneStop}
            >
              停止
            </button>
          </div>
        </section>

        <section className="motion-live-section" aria-label="入力レベル">
          <div className="motion-live-meter">
            <div className="motion-live-meter-row"><span>口の開き</span><strong>{Math.round(microphone.openAmount * 100)}%</strong></div>
            <div className="motion-live-meter-track"><div className="motion-live-meter-fill" style={{ width: `${microphone.openAmount * 100}%` }} /></div>
          </div>
          <div className="motion-live-meter">
            <div className="motion-live-meter-row"><span>マイク入力</span><span>{(microphone.rawLevel * 100).toFixed(1)}%</span></div>
            <div className="motion-live-meter-track"><div className="motion-live-meter-fill raw" style={{ width: `${Math.min(100, microphone.rawLevel * 500)}%` }} /></div>
          </div>
        </section>

        <section className="motion-live-section">
          <label htmlFor="motion-live-mouth">開いたときの口</label>
          <select
            id="motion-live-mouth"
            className="motion-live-select"
            value={selectedOpenMouth}
            disabled={availableOpenMouths.length === 0}
            onChange={event => setSelectedOpenMouth(event.target.value as MotionLabMouthKey)}
          >
            {availableOpenMouths.length === 0 && <option value="closed">開き口素材がありません</option>}
            {availableOpenMouths.map(key => <option key={key} value={key}>{MOUTH_LABELS[key]}</option>)}
          </select>
        </section>

        <section className="motion-live-section" aria-label="口パク感度">
          <span className="motion-live-control-label">感度</span>
          <div className="motion-live-range">
            <input type="range" min="1" max="40" step="0.5" value={microphone.sensitivity} onChange={event => microphone.setSensitivity(Number(event.target.value))} />
            <output>{microphone.sensitivity.toFixed(1)}×</output>
          </div>
          <span className="motion-live-control-label">ノイズゲート</span>
          <div className="motion-live-range">
            <input type="range" min="0" max="0.08" step="0.001" value={microphone.noiseGate} onChange={event => microphone.setNoiseGate(Number(event.target.value))} />
            <output>{(microphone.noiseGate * 100).toFixed(1)}%</output>
          </div>
          <span className="motion-live-control-label">開く速さ</span>
          <div className="motion-live-range">
            <input type="range" min="0" max="300" step="5" value={microphone.attackMs} onChange={event => microphone.setAttackMs(Number(event.target.value))} />
            <output>{microphone.attackMs}ms</output>
          </div>
          <span className="motion-live-control-label">閉じる速さ</span>
          <div className="motion-live-range">
            <input type="range" min="20" max="500" step="5" value={microphone.releaseMs} onChange={event => microphone.setReleaseMs(Number(event.target.value))} />
            <output>{microphone.releaseMs}ms</output>
          </div>
        </section>

        <section className="motion-live-section" aria-label="表示構図">
          <span className="motion-live-control-label">表示倍率</span>
          <div className="motion-live-range">
            <input type="range" min="25" max="250" step="5" value={displayScale} onChange={event => setDisplayScale(Number(event.target.value))} />
            <output>{displayScale}%</output>
          </div>
          <span className="motion-live-control-label">横位置</span>
          <div className="motion-live-range">
            <input type="range" min="-100" max="100" step="0.1" value={displayOffsetX} onChange={event => setDisplayOffsetX(Number(event.target.value))} />
            <output>{displayOffsetX > 0 ? "+" : ""}{displayOffsetX.toFixed(1)}%</output>
          </div>
          <span className="motion-live-control-label">縦位置</span>
          <div className="motion-live-range">
            <input type="range" min="-100" max="100" step="0.1" value={displayOffsetY} onChange={event => setDisplayOffsetY(Number(event.target.value))} />
            <output>{displayOffsetY > 0 ? "+" : ""}{displayOffsetY.toFixed(1)}%</output>
          </div>
          <button type="button" className="motion-live-button" onClick={resetDisplayComposition}>構図をリセット</button>
          <p className="motion-live-note">右のプレビューをドラッグして位置を、マウスホイールで倍率を調整できます。</p>
        </section>

        <section className="motion-live-section">
          <span className="motion-live-control-label">OBS用背景</span>
          <div className="motion-live-colors">
            {(Object.entries(CHROMA_BACKGROUNDS) as Array<[ChromaBackground, { label: string; color: string }]>).map(([key, item]) => (
              <button
                type="button"
                key={key}
                className={`motion-live-color${background === key ? " active" : ""}`}
                aria-pressed={background === key}
                onClick={() => setBackground(key)}
              >
                <span className="motion-live-color-swatch" style={{ background: item.color }} />
                {item.label}
              </button>
            ))}
          </div>
        </section>

        {visibleError && <p className="motion-live-message" role="alert">{visibleError}</p>}
        <p className="motion-live-note">OBSでは「ウィンドウキャプチャ」でこの画面を選び、選択した背景色をカラーキーで除去します。音声はOBSにも別途マイク入力として追加してください。</p>
        <button
          type="button"
          className="motion-live-button primary"
          disabled={!parts || !images || loading}
          onClick={() => setCaptureOnly(true)}
        >
          キャプチャ表示に切り替える
        </button>
      </aside>

      <div className="motion-live-stage-shell" ref={stageShellRef}>
        <main
          className={`motion-live-stage${!captureOnly && images ? " is-draggable" : ""}${stageDragging ? " is-dragging" : ""}`}
          style={stageStyle}
          aria-label="ライブキャラクター表示"
          onPointerDown={handleStagePointerDown}
          onPointerMove={handleStagePointerMove}
          onPointerUp={finishStageDrag}
          onPointerCancel={finishStageDrag}
          onWheel={handleStageWheel}
          onLostPointerCapture={() => {
            stageDragRef.current = null;
            setStageDragging(false);
          }}
        >
          <span className="motion-live-capture-frame-label">
            キャプチャ範囲 {previewFrame.captureWidth}×{previewFrame.captureHeight}
          </span>
          <canvas
            ref={canvasRef}
            style={{ transform: `translate(${displayOffsetX}%, ${displayOffsetY}%) scale(${displayScale / 100})` }}
          />
          {(loading || !partsDir) && (
            <div className="motion-live-loading">
              {loading ? "素材を読み込んでいます…" : "表示する素材が選択されていません"}
            </div>
          )}
        </main>
      </div>
      {captureOnly && (
        <div className="motion-live-capture-exit-zone">
          <button
            type="button"
            className={`motion-live-capture-exit${captureControlsVisible ? " visible" : ""}`}
            onClick={() => setCaptureOnly(false)}
          >設定へ戻る（Esc）</button>
        </div>
      )}
    </div>
  );
}
