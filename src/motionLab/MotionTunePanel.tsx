import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
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
  MOTION_LAB_TEMPLATES,
  motionLabTimelineFromText,
} from "./constants";
import {
  createMotionLabPhysics,
  drawMotionLabScene,
  loadMotionLabImage,
  prepareMotionLabCanvas,
  resetMotionLabRuntime,
} from "./render";
import { buildMotionLabManifest, buildSpritalkMotionProfile } from "./manifest";
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
}

function createRuntime(): MotionLabMouthRuntime {
  return {
    openY: 0,
    activeTarget: "closed",
    previousTarget: "closed",
    transitionStartMs: 0,
    lastMs: 0,
    physics: createMotionLabPhysics(),
  };
}

/**
 * STEP7「モーション調整」パネル。
 * 旧Motion Preview Lab（2レーン比較実験画面）を製品向けの1レーン調整画面へ再構成したもの。
 * 素材の読込・物理プレビュー・設定の保存/読込・SpriTalk用設定JSONの出力まで自己完結する。
 */
export function MotionTunePanel({ partsDir, active = true, onNotify, onError }: MotionTunePanelProps) {
  const [parts, setParts] = useState<MotionLabPartsResult | null>(null);
  const [images, setImages] = useState<MotionLabImageSet | null>(null);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [text, setText] = useState("こんにちは、あいうえお");
  const [customTimeline, setCustomTimeline] = useState<{
    timeline: MotionLabTimelineEvent[];
    durationMs: number;
  } | null>(null);
  const [pivotEditPart, setPivotEditPart] = useState<string | null>(null);
  const [manifestPath, setManifestPath] = useState("");
  const [profilePath, setProfilePath] = useState("");
  const [settings, dispatch] = useMotionLabSettings();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runtimeRef = useRef<MotionLabMouthRuntime>(createRuntime());

  const notify = (message: string) => onNotify?.(message);
  const fail = (cause: unknown) => onError?.(cause instanceof Error ? cause.message : String(cause));

  async function loadPartsFromDir(dir: string) {
    setBusy(true);
    try {
      const result = await invoke<MotionLabPartsResult>("load_motion_lab_parts", { dir });
      setParts(result);
      setPlaying(true);
      setProfilePath("");
      setManifestPath("");
      // 保存済み設定（motion-preview-manifest.json）があれば自動復元（つづきから対応）
      try {
        const manifest = await invoke<MotionLabManifestResult>("load_motion_lab_manifest", {
          sourceDir: result.sourceDir,
        });
        dispatch({ type: "applyManifest", manifest: manifest.manifest });
        setManifestPath(manifest.path);
        notify(`モーション素材と保存済み設定を読み込みました: ${result.sourceDir}`);
      } catch {
        notify(`モーション素材を読み込みました: ${result.sourceDir}`);
      }
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }

  // partsDir変化時の自動ロード（同じフォルダを読込済みならスキップ = 調整中に設定が飛ばない）
  useEffect(() => {
    if (!partsDir) return;
    if (parts?.sourceDir === partsDir) return;
    void loadPartsFromDir(partsDir);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [partsDir]);

  // 素材 → HTMLImageElement デコード
  useEffect(() => {
    let cancelled = false;
    if (!parts) {
      setImages(null);
      setImagesLoading(false);
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
          eyewhite: current.eyewhite ? await loadMotionLabImage(current.eyewhite) : null,
          irides: current.irides ? await loadMotionLabImage(current.irides) : null,
          highlight: current.highlight ? await loadMotionLabImage(current.highlight) : null,
          eyeFrames: await Promise.all(current.eyeFrames.map(loadMotionLabImage)),
          mouths: Object.fromEntries(mouthEntries) as Partial<Record<MotionLabMouthKey, HTMLImageElement[]>>,
        };
        if (!cancelled) {
          setImages(nextImages);
          setImagesLoading(false);
        }
      } catch (cause) {
        if (!cancelled) {
          fail(cause);
          setImagesLoading(false);
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

  async function pickAnotherDir() {
    const selected = await open({
      multiple: false,
      directory: true,
      title: "モーション素材フォルダを選択（04_spritalk_parts）",
    });
    const dir = typeof selected === "string" ? selected : null;
    if (!dir) return;
    await loadPartsFromDir(dir);
  }

  async function saveManifest() {
    if (!parts) return;
    setBusy(true);
    try {
      const manifest = buildMotionLabManifest(settings, parts.sourceDir);
      const result = await invoke<MotionLabManifestResult>("save_motion_lab_manifest", {
        request: { sourceDir: parts.sourceDir, manifest },
      });
      setManifestPath(result.path);
      notify(`モーション設定を保存しました: ${result.path}`);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
    }
  }

  async function exportSpritalkProfile() {
    if (!parts) return;
    setBusy(true);
    try {
      // 設定JSONと一緒にmanifestも保存し、次回のつづきから復元と内容を一致させる
      const manifest = buildMotionLabManifest(settings, parts.sourceDir);
      const manifestResult = await invoke<MotionLabManifestResult>("save_motion_lab_manifest", {
        request: { sourceDir: parts.sourceDir, manifest },
      });
      setManifestPath(manifestResult.path);
      const profile = buildSpritalkMotionProfile(settings, parts.sourceDir);
      const result = await invoke<SpritalkMotionProfileResult>("save_spritalk_motion_profile", {
        request: { sourceDir: parts.sourceDir, profile },
      });
      setProfilePath(result.path);
      notify(`SpriTalk用アニメーション設定を出力しました: ${result.path}`);
    } catch (cause) {
      fail(cause);
    } finally {
      setBusy(false);
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
    gaze: { value: settings.gazeStrength, min: 0, max: 2, step: 0.05, set: v => dispatch({ type: "set", patch: { gazeStrength: v } }), format: percentFormat },
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
  ) => (
    <label className="motion-lab-range">
      <span>{label}<b>{value}{suffix}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );

  const missingBody = !parts;

  return (
    <div className="motion-tune">
      <section className="motion-lab-preview-panel motion-tune-preview">
        <div className="motion-lab-preview-toolbar">
          <button className="btn btn-secondary" disabled={missingBody} onClick={() => setPlaying(prev => !prev)}>
            {playing ? "停止" : "再生"}
          </button>
          <button className="btn btn-secondary" disabled={missingBody} onClick={restartPlayback}>
            最初から
          </button>
          <div className="motion-lab-text-row motion-tune-text">
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="ひらがな・カタカナで入力（例: こんにちは)"
            />
            <button className="btn btn-secondary" disabled={missingBody} onClick={() => {
              setCustomTimeline(motionLabTimelineFromText(text));
              restartPlayback();
            }}>テキスト再生</button>
            {customTimeline && (
              <button className="btn btn-secondary" onClick={() => setCustomTimeline(null)}>内蔵あいうえお</button>
            )}
          </div>
        </div>

        <div className="motion-lab-stage motion-tune-stage">
          {parts ? (
            <>
              <canvas
                ref={canvasRef}
                style={pivotEditPart ? { cursor: "crosshair" } : undefined}
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
            </>
          ) : (
            <span className="motion-lab-placeholder">
              {busy ? "素材を読み込んでいます..." : "RIFE補完の出力（04_spritalk_parts）を読み込むとプレビューが表示されます"}
            </span>
          )}
        </div>

        <div className="motion-lab-mouth-strip">
          {MOTION_LAB_MOUTH_KEYS.map(key => {
            const count = parts?.mouths[key]?.length ?? 0;
            return (
              <span key={key} className={count > 0 ? "ready" : ""}>
                <b>{MOTION_LAB_MOUTH_LABELS[key]}</b>
                <small>{count}</small>
              </span>
            );
          })}
        </div>
      </section>

      <section className="motion-lab-control-panel motion-tune-controls">
        {parts?.missing.length ? (
          <div className="motion-lab-note warning">不足素材: {parts.missing.join(", ")}</div>
        ) : null}
        {parts?.warnings.length ? <div className="motion-lab-note">{parts.warnings.join(" / ")}</div> : null}

        <div className="motion-lab-section motion-lab-simple">
          <div className="motion-lab-section-title">
            <strong>方式</strong>
            <div className="motion-lab-segmented">
              <button
                className={settings.engineFamily === "rotejin" ? "active" : ""}
                title="PuruPuruPNGTuber系: 進行波の髪揺れ＋ぷるぷるした弾み"
                onClick={() => dispatch({ type: "applyEngineFamily", family: "rotejin" })}
              >ろてじん式（波・ぷるぷる）</button>
              <button
                className={settings.engineFamily === "hachigoni" ? "active" : ""}
                title="Anime2.5DRig系: バネ・チェーンの髪物理＋パララックス首振り"
                onClick={() => dispatch({ type: "applyEngineFamily", family: "hachigoni" })}
              >852話式（バネ・リグ）</button>
            </div>
          </div>
          <div className="motion-lab-section-title">
            <strong>テンプレート</strong>
            <div className="motion-lab-segmented">
              {Object.entries(MOTION_LAB_TEMPLATES)
                .filter(([, template]) => template.engine === settings.engineFamily)
                .map(([key, template]) => (
                  <button
                    key={key}
                    className={settings.templateName === key ? "active" : ""}
                    title={template.description}
                    onClick={() => dispatch({ type: "applyTemplate", key })}
                  >{template.label}</button>
                ))}
            </div>
          </div>
          {settings.templateName && (
            <div className="motion-lab-note">
              {MOTION_LAB_TEMPLATES[settings.templateName].description}。適用後も各項目で微調整できます。
            </div>
          )}
          {renderRange("動きの強さ", Math.round(settings.intensity * 100), 50, 150, 5, (value) => {
            dispatch({ type: "applyIntensity", value: value / 100 });
          }, "%")}
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
              if (def.key === "chest") return !!parts?.chest;
              if (def.key === "gaze") return !!(parts?.eyewhite && parts?.irides);
              if (def.key === "earTwitch") return Object.keys(parts?.sways ?? {}).some(name => /(^|_)ears?(_|$)/i.test(name));
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
          <summary>詳細パラメータ</summary>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>口パク</strong>
              <div className="motion-lab-segmented three">
                <button className={settings.method === "baseline" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { method: "baseline" } })}>直接</button>
                <button className={settings.method === "smooth" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { method: "smooth" } })}>スムーズ</button>
                <button className={settings.method === "bridge" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { method: "bridge" } })}>ブリッジ</button>
              </div>
            </div>
            {renderRange("attack", settings.attackMs, 40, 180, 5, v => dispatch({ type: "set", patch: { attackMs: v } }), "ms")}
            {renderRange("release", settings.releaseMs, 80, 260, 5, v => dispatch({ type: "set", patch: { releaseMs: v } }), "ms")}
            {renderRange("crossfade", settings.crossfadeMs, 0, 120, 5, v => dispatch({ type: "set", patch: { crossfadeMs: v } }), "ms")}
            {renderRange("rest", Math.round(settings.restBias * 100), 0, 100, 1, v => dispatch({ type: "set", patch: { restBias: v / 100 } }), "%")}
            {renderRange("smooth", Math.round(settings.shapeSmoothing * 100), 0, 100, 1, v => dispatch({ type: "set", patch: { shapeSmoothing: v / 100 } }), "%")}
            {renderRange("bridge", Math.round(settings.bridgeBias * 100), 0, 85, 1, v => dispatch({ type: "set", patch: { bridgeBias: v / 100 } }), "%")}
          </div>

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
              <button className={settings.layerMode === "spring" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { layerMode: "spring" } })}>spring</button>
              <button className={settings.layerMode === "mesh" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { layerMode: "mesh" } })}>mesh</button>
            </div>
            <div className="motion-lab-segmented">
              <button className={settings.hairEngine === "spring" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { hairEngine: "spring" } })}>バネ物理</button>
              <button className={settings.hairEngine === "wave" ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { hairEngine: "wave" } })}>波揺れ（ろてじん式）</button>
            </div>
            {settings.hairEngine === "wave" &&
              renderRange("波の強さ", Math.round(settings.hairWaveStrength * 100), 0, 200, 5, v => dispatch({ type: "set", patch: { hairWaveStrength: v / 100 } }), "%")}
            {settings.layerMode === "mesh" && (
              <div className="motion-lab-segmented">
                <button className={!settings.strandsEnabled ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { strandsEnabled: false } })}>一枚チェーン</button>
                <button className={settings.strandsEnabled ? "active" : ""} onClick={() => dispatch({ type: "set", patch: { strandsEnabled: true } })}>房分割</button>
              </div>
            )}
            {renderRange("柔らかさ k", settings.hairK, 10, 200, 5, v => dispatch({ type: "set", patch: { hairK: v } }))}
            {renderRange("収まり c", settings.hairC, 1, 30, 1, v => dispatch({ type: "set", patch: { hairC: v } }))}
            {renderRange("風 wind", Number((settings.hairWind * 1000).toFixed(0)), 0, 60, 2, v => dispatch({ type: "set", patch: { hairWind: v / 1000 } }), "‰")}
            {renderRange("体追従 drive", Number((settings.hairDrive * 100).toFixed(0)), 0, 20, 1, v => dispatch({ type: "set", patch: { hairDrive: v / 100 } }), "%")}
          </div>

          <div className="motion-lab-section">
            <div className="motion-lab-section-title">
              <strong>回転軸・可動域</strong>
            </div>
            <div className="motion-lab-segmented">
              {([
                ["hair", "前髪", !!parts?.hair],
                ["hair_back", "後ろ髪", !!parts?.hairBack],
                ["arm_l", "左腕", !!parts?.armL],
                ["arm_r", "右腕", !!parts?.armR],
              ] as Array<[string, string, boolean]>).filter(([, , exists]) => exists).map(([part, label]) => (
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
                <div className="motion-lab-text-row">
                  <small>
                    回転軸: {settings.pivots[pivotEditPart]
                      ? `${settings.pivots[pivotEditPart].x}, ${settings.pivots[pivotEditPart].y}`
                      : "自動推定"}（プレビューをクリックで指定）
                  </small>
                  {settings.pivots[pivotEditPart] && (
                    <button className="btn btn-secondary" onClick={() => {
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
              前髪・後ろ髪は回転軸のY位置が「揺れの根元」として効きます。
            </div>
          </div>

          {(parts?.armL || parts?.armR) ? (
            <div className="motion-lab-section">
              <div className="motion-lab-section-title">
                <strong>腕揺れ</strong>
              </div>
              {renderRange("最大角", Number((settings.armMaxAngle * 100).toFixed(0)), 0, 60, 1, v => dispatch({ type: "set", patch: { armMaxAngle: v / 100 } }), "×0.01rad")}
              {renderRange("回転軸位置", Math.round(settings.armPivotRatio * 100), 0, 60, 2, v => dispatch({ type: "set", patch: { armPivotRatio: v / 100 } }), "%")}
            </div>
          ) : null}
        </details>

        <div className="motion-lab-section motion-tune-export">
          <div className="motion-lab-section-title">
            <strong>SpriTalkへ出力</strong>
          </div>
          <button
            className="btn btn-primary motion-tune-export-btn"
            disabled={busy || !parts}
            onClick={() => void exportSpritalkProfile()}
          >
            SpriTalk用アニメーション設定を出力
          </button>
          {profilePath ? (
            <div className="motion-lab-note">
              出力済み: <code>{profilePath}</code>
            </div>
          ) : (
            <div className="motion-lab-note">
              調整した揺れ・口パク設定を <code>spritalk-motion-profile.json</code> として素材フォルダへ書き出します。
              SpriTalkのキャラクター読込時にこのフォルダごと取り込む想定です。
            </div>
          )}
          <div className="motion-lab-manifest-actions">
            <button className="btn btn-secondary" disabled={busy || !parts} onClick={() => void saveManifest()}>
              調整内容を保存
            </button>
            <button
              className="btn btn-secondary"
              disabled={busy || !parts}
              onClick={() => { if (parts) void openPath(parts.sourceDir); }}
            >
              出力フォルダを開く
            </button>
            {manifestPath ? <small title={manifestPath}>{manifestPath}</small> : null}
          </div>
          <div className="motion-lab-manifest-actions">
            <button className="btn btn-secondary" disabled={busy} onClick={() => void pickAnotherDir()}>
              別の素材フォルダを開く
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
