import { useCallback, useEffect, useRef, useState } from "react";

export interface MicrophoneInputDevice {
  deviceId: string;
  groupId: string;
  label: string;
}

export interface UseMicrophoneLevelOptions {
  initialDeviceId?: string;
  initialSensitivity?: number;
  initialNoiseGate?: number;
  initialAttackMs?: number;
  initialReleaseMs?: number;
  fftSize?: number;
}

export interface UseMicrophoneLevelResult {
  devices: MicrophoneInputDevice[];
  selectedDeviceId: string;
  setSelectedDeviceId: (deviceId: string) => void;
  active: boolean;
  rawLevel: number;
  openAmount: number;
  error: string | null;
  sensitivity: number;
  setSensitivity: (value: number) => void;
  noiseGate: number;
  setNoiseGate: (value: number) => void;
  attackMs: number;
  setAttackMs: (value: number) => void;
  releaseMs: number;
  setReleaseMs: (value: number) => void;
  refreshDevices: () => Promise<MicrophoneInputDevice[]>;
  start: (deviceId?: string) => Promise<boolean>;
  stop: () => void;
}

const DEFAULT_SENSITIVITY = 16;
const DEFAULT_NOISE_GATE = 0.012;
const DEFAULT_ATTACK_MS = 45;
const DEFAULT_RELEASE_MS = 160;
const DEFAULT_FFT_SIZE = 2048;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
}

function validFftSize(value: number | undefined): number {
  if (!value || value < 32 || value > 32768) return DEFAULT_FFT_SIZE;
  const power = 2 ** Math.round(Math.log2(value));
  return clamp(power, 32, 32768);
}

/** Calculate the root-mean-square amplitude of time-domain audio samples. */
export function microphoneRms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sumSquares = 0;
  for (const sample of samples) sumSquares += sample * sample;
  return clamp(Math.sqrt(sumSquares / samples.length), 0, 1);
}

/** Map an RMS level to a 0..1 mouth target after applying the noise gate. */
export function microphoneOpenTarget(
  rms: number,
  noiseGate: number,
  sensitivity: number,
): number {
  const gate = clamp(noiseGate, 0, 0.25);
  const gain = clamp(sensitivity, 0.1, 100);
  return clamp((clamp(rms, 0, 1) - gate) * gain, 0, 1);
}

/** Attack/release exponential smoothing that remains stable across frame rates. */
export function smoothMicrophoneLevel(
  current: number,
  target: number,
  deltaMs: number,
  attackMs: number,
  releaseMs: number,
): number {
  const from = clamp(current, 0, 1);
  const to = clamp(target, 0, 1);
  const duration = to > from
    ? clamp(attackMs, 0, 2000)
    : clamp(releaseMs, 0, 4000);
  if (duration <= 0 || deltaMs <= 0) return deltaMs <= 0 ? from : to;
  const alpha = 1 - Math.exp(-clamp(deltaMs, 0, 1000) / duration);
  return clamp(from + (to - from) * alpha, 0, 1);
}

function microphoneErrorMessage(cause: unknown): string {
  if (cause instanceof DOMException) {
    if (cause.name === "NotAllowedError" || cause.name === "SecurityError") {
      return "マイクの使用が許可されていません。Windowsとアプリのマイク権限を確認してください。";
    }
    if (cause.name === "NotFoundError" || cause.name === "DevicesNotFoundError") {
      return "利用できるマイクが見つかりません。";
    }
    if (cause.name === "NotReadableError" || cause.name === "TrackStartError") {
      return "マイクを開始できません。ほかのアプリによる排他使用を確認してください。";
    }
    if (cause.name === "OverconstrainedError") {
      return "選択したマイクを利用できません。別のマイクを選択してください。";
    }
  }
  return cause instanceof Error ? cause.message : String(cause);
}

function mediaDevicesApi(): MediaDevices | null {
  return typeof navigator !== "undefined" ? navigator.mediaDevices ?? null : null;
}

export function useMicrophoneLevel(
  options: UseMicrophoneLevelOptions = {},
): UseMicrophoneLevelResult {
  const [devices, setDevices] = useState<MicrophoneInputDevice[]>([]);
  const [selectedDeviceIdState, setSelectedDeviceIdState] = useState(options.initialDeviceId ?? "");
  const [active, setActive] = useState(false);
  const [rawLevel, setRawLevel] = useState(0);
  const [openAmount, setOpenAmount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [sensitivityState, setSensitivityState] = useState(
    clamp(options.initialSensitivity ?? DEFAULT_SENSITIVITY, 0.1, 100),
  );
  const [noiseGateState, setNoiseGateState] = useState(
    clamp(options.initialNoiseGate ?? DEFAULT_NOISE_GATE, 0, 0.25),
  );
  const [attackMsState, setAttackMsState] = useState(
    clamp(options.initialAttackMs ?? DEFAULT_ATTACK_MS, 0, 2000),
  );
  const [releaseMsState, setReleaseMsState] = useState(
    clamp(options.initialReleaseMs ?? DEFAULT_RELEASE_MS, 0, 4000),
  );

  const mountedRef = useRef(true);
  const sessionIdRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const smoothedLevelRef = useRef(0);
  const lastFrameMsRef = useRef(0);
  const sensitivityRef = useRef(sensitivityState);
  const noiseGateRef = useRef(noiseGateState);
  const attackMsRef = useRef(attackMsState);
  const releaseMsRef = useRef(releaseMsState);
  const selectedDeviceIdRef = useRef(selectedDeviceIdState);
  const fftSizeRef = useRef(validFftSize(options.fftSize));

  useEffect(() => {
    sensitivityRef.current = sensitivityState;
  }, [sensitivityState]);

  useEffect(() => {
    noiseGateRef.current = noiseGateState;
  }, [noiseGateState]);

  useEffect(() => {
    attackMsRef.current = attackMsState;
  }, [attackMsState]);

  useEffect(() => {
    releaseMsRef.current = releaseMsState;
  }, [releaseMsState]);

  useEffect(() => {
    selectedDeviceIdRef.current = selectedDeviceIdState;
  }, [selectedDeviceIdState]);

  const releaseResources = useCallback(() => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    sourceRef.current?.disconnect();
    analyserRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current = null;
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    streamRef.current = null;
    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context && context.state !== "closed") void context.close().catch(() => undefined);
    smoothedLevelRef.current = 0;
    lastFrameMsRef.current = 0;
  }, []);

  const stop = useCallback(() => {
    sessionIdRef.current += 1;
    releaseResources();
    if (!mountedRef.current) return;
    setActive(false);
    setRawLevel(0);
    setOpenAmount(0);
  }, [releaseResources]);

  const refreshDevices = useCallback(async (): Promise<MicrophoneInputDevice[]> => {
    const mediaDevices = mediaDevicesApi();
    if (!mediaDevices?.enumerateDevices) {
      const message = "この環境ではマイク入力を利用できません。";
      if (mountedRef.current) setError(message);
      return [];
    }
    try {
      const next = (await mediaDevices.enumerateDevices())
        .filter(device => device.kind === "audioinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          groupId: device.groupId,
          label: device.label || `マイク ${index + 1}`,
        }));
      if (mountedRef.current) {
        setDevices(next);
        setSelectedDeviceIdState(previous => {
          if (previous && next.some(device => device.deviceId === previous)) return previous;
          return next[0]?.deviceId ?? "";
        });
      }
      return next;
    } catch (cause) {
      if (mountedRef.current) setError(microphoneErrorMessage(cause));
      return [];
    }
  }, []);

  const start = useCallback(async (requestedDeviceId?: string): Promise<boolean> => {
    const mediaDevices = mediaDevicesApi();
    if (!mediaDevices?.getUserMedia) {
      if (mountedRef.current) setError("この環境ではマイク入力を利用できません。");
      return false;
    }

    stop();
    const sessionId = ++sessionIdRef.current;
    if (mountedRef.current) setError(null);

    let pendingStream: MediaStream | null = null;
    let pendingContext: AudioContext | null = null;
    let pendingSource: MediaStreamAudioSourceNode | null = null;
    let pendingAnalyser: AnalyserNode | null = null;
    try {
      const deviceId = requestedDeviceId ?? selectedDeviceIdRef.current;
      pendingStream = await mediaDevices.getUserMedia({
        audio: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      if (!mountedRef.current || sessionId !== sessionIdRef.current) {
        for (const track of pendingStream.getTracks()) track.stop();
        return false;
      }

      const AudioContextConstructor = window.AudioContext
        ?? (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextConstructor) throw new Error("この環境では音声解析を利用できません。");
      pendingContext = new AudioContextConstructor();
      if (pendingContext.state === "suspended") await pendingContext.resume();
      pendingSource = pendingContext.createMediaStreamSource(pendingStream);
      pendingAnalyser = pendingContext.createAnalyser();
      pendingAnalyser.fftSize = fftSizeRef.current;
      pendingAnalyser.smoothingTimeConstant = 0;
      pendingSource.connect(pendingAnalyser);

      if (!mountedRef.current || sessionId !== sessionIdRef.current) {
        pendingSource.disconnect();
        pendingAnalyser.disconnect();
        for (const track of pendingStream.getTracks()) track.stop();
        await pendingContext.close().catch(() => undefined);
        return false;
      }

      streamRef.current = pendingStream;
      audioContextRef.current = pendingContext;
      sourceRef.current = pendingSource;
      analyserRef.current = pendingAnalyser;
      pendingStream = null;
      pendingContext = null;
      pendingSource = null;
      pendingAnalyser = null;

      const actualDeviceId = streamRef.current.getAudioTracks()[0]?.getSettings().deviceId;
      if (actualDeviceId) setSelectedDeviceIdState(actualDeviceId);
      setActive(true);

      const samples = new Float32Array(analyserRef.current.fftSize);
      lastFrameMsRef.current = performance.now();
      const analyse = (now: number) => {
        if (!mountedRef.current || sessionId !== sessionIdRef.current || !analyserRef.current) return;
        analyserRef.current.getFloatTimeDomainData(samples);
        const rms = microphoneRms(samples);
        const target = microphoneOpenTarget(rms, noiseGateRef.current, sensitivityRef.current);
        const deltaMs = Math.min(100, Math.max(0, now - lastFrameMsRef.current));
        lastFrameMsRef.current = now;
        const smoothed = smoothMicrophoneLevel(
          smoothedLevelRef.current,
          target,
          deltaMs,
          attackMsRef.current,
          releaseMsRef.current,
        );
        smoothedLevelRef.current = smoothed;
        setRawLevel(previous => Math.abs(previous - rms) >= 0.0005 ? rms : previous);
        setOpenAmount(previous => Math.abs(previous - smoothed) >= 0.001 ? smoothed : previous);
        animationFrameRef.current = window.requestAnimationFrame(analyse);
      };
      animationFrameRef.current = window.requestAnimationFrame(analyse);
      void refreshDevices();
      return true;
    } catch (cause) {
      pendingSource?.disconnect();
      pendingAnalyser?.disconnect();
      for (const track of pendingStream?.getTracks() ?? []) track.stop();
      if (pendingContext && pendingContext.state !== "closed") {
        await pendingContext.close().catch(() => undefined);
      }
      if (sessionId === sessionIdRef.current) {
        releaseResources();
        if (mountedRef.current) {
          setActive(false);
          setRawLevel(0);
          setOpenAmount(0);
          setError(microphoneErrorMessage(cause));
        }
      }
      return false;
    }
  }, [refreshDevices, releaseResources, stop]);

  const setSelectedDeviceId = useCallback((deviceId: string) => {
    selectedDeviceIdRef.current = deviceId;
    setSelectedDeviceIdState(deviceId);
  }, []);

  const setSensitivity = useCallback((value: number) => {
    const next = clamp(value, 0.1, 100);
    sensitivityRef.current = next;
    setSensitivityState(next);
  }, []);

  const setNoiseGate = useCallback((value: number) => {
    const next = clamp(value, 0, 0.25);
    noiseGateRef.current = next;
    setNoiseGateState(next);
  }, []);

  const setAttackMs = useCallback((value: number) => {
    const next = clamp(value, 0, 2000);
    attackMsRef.current = next;
    setAttackMsState(next);
  }, []);

  const setReleaseMs = useCallback((value: number) => {
    const next = clamp(value, 0, 4000);
    releaseMsRef.current = next;
    setReleaseMsState(next);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refreshDevices();
    const mediaDevices = mediaDevicesApi();
    const handleDeviceChange = () => void refreshDevices();
    mediaDevices?.addEventListener?.("devicechange", handleDeviceChange);
    return () => {
      mountedRef.current = false;
      sessionIdRef.current += 1;
      mediaDevices?.removeEventListener?.("devicechange", handleDeviceChange);
      releaseResources();
    };
  }, [refreshDevices, releaseResources]);

  return {
    devices,
    selectedDeviceId: selectedDeviceIdState,
    setSelectedDeviceId,
    active,
    rawLevel,
    openAmount,
    error,
    sensitivity: sensitivityState,
    setSensitivity,
    noiseGate: noiseGateState,
    setNoiseGate,
    attackMs: attackMsState,
    setAttackMs,
    releaseMs: releaseMsState,
    setReleaseMs,
    refreshDevices,
    start,
    stop,
  };
}
