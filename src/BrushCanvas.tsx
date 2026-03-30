import { useRef, useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

interface BlendResult {
  blend_image: string;
  width: number;
  height: number;
}

interface TargetMaskResult {
  mask_image: string;
  width: number;
  height: number;
}

interface BrushApplyResult {
  overlay_preview: string;
}

export type BrushTarget = "eye" | "mouth" | "eyebrow";

const TARGET_COLORS: Record<BrushTarget, [number, number, number]> = {
  eye: [0, 80, 255],     // strong blue — visible on white
  mouth: [220, 20, 20],  // strong red — visible on white
  eyebrow: [0, 160, 60], // strong green — visible on white
};

interface BrushCanvasProps {
  maskWidth: number;
  maskHeight: number;
  target: BrushTarget;
  onBrushApplied: (result: BrushApplyResult) => void;
  refreshKey: number;
}

export default function BrushCanvas({
  maskWidth,
  maskHeight,
  target,
  onBrushApplied,
  refreshKey,
}: BrushCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const brushCanvasRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);

  const [brushSize, setBrushSize] = useState(20);
  const [painting, setPainting] = useState(false);
  const [erasing, setErasing] = useState(false);
  const lastPos = useRef<[number, number] | null>(null);
  const [loading, setLoading] = useState(true);
  const [displayScale, setDisplayScale] = useState(1);

  // Load background and target mask
  const loadBackgrounds = useCallback(async () => {
    if (maskWidth === 0 || maskHeight === 0) return;
    setLoading(true);

    try {
      const [blend, maskResult] = await Promise.all([
        invoke<BlendResult>("get_brush_background_cmd", { target }),
        invoke<TargetMaskResult>("get_target_mask_cmd", { target }),
      ]);

      const bgCanvas = bgCanvasRef.current;
      if (bgCanvas) {
        bgCanvas.width = maskWidth;
        bgCanvas.height = maskHeight;
        const bgCtx = bgCanvas.getContext("2d")!;
        const bgImg = new Image();
        bgImg.onload = () => {
          bgCtx.drawImage(bgImg, 0, 0, maskWidth, maskHeight);
        };
        bgImg.src = blend.blend_image;
      }

      const maskCanvas = maskCanvasRef.current;
      if (maskCanvas) {
        maskCanvas.width = maskWidth;
        maskCanvas.height = maskHeight;
        const maskCtx = maskCanvas.getContext("2d")!;
        const maskImg = new Image();
        maskImg.onload = () => {
          maskCtx.clearRect(0, 0, maskWidth, maskHeight);
          maskCtx.drawImage(maskImg, 0, 0, maskWidth, maskHeight);
          const imageData = maskCtx.getImageData(0, 0, maskWidth, maskHeight);
          const data = imageData.data;
          const [cr, cg, cb] = TARGET_COLORS[target];
          for (let i = 0; i < data.length; i += 4) {
            const v = data[i];
            data[i] = cr;
            data[i + 1] = cg;
            data[i + 2] = cb;
            data[i + 3] = Math.floor(v * 0.6);
          }
          maskCtx.putImageData(imageData, 0, 0);
        };
        maskImg.src = maskResult.mask_image;
      }

      const brushCanvas = brushCanvasRef.current;
      if (brushCanvas) {
        brushCanvas.width = maskWidth;
        brushCanvas.height = maskHeight;
        const brushCtx = brushCanvas.getContext("2d")!;
        const maskImg2 = new Image();
        maskImg2.onload = () => {
          brushCtx.clearRect(0, 0, maskWidth, maskHeight);
          brushCtx.drawImage(maskImg2, 0, 0, maskWidth, maskHeight);
        };
        maskImg2.src = maskResult.mask_image;
      }
    } catch (e) {
      console.error("BrushCanvas loadBackgrounds:", e);
    } finally {
      setLoading(false);
    }
  }, [maskWidth, maskHeight, target]);

  useEffect(() => {
    loadBackgrounds();
  }, [loadBackgrounds, refreshKey]);

  function canvasCoords(e: React.MouseEvent<HTMLCanvasElement>): [number, number] {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [
      (e.clientX - rect.left) * scaleX,
      (e.clientY - rect.top) * scaleY,
    ];
  }

  // Draw on the brush canvas (white grayscale, used as mask data for backend)
  function drawBrushData(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    radius: number, isErase: boolean
  ) {
    ctx.globalCompositeOperation = isErase ? "destination-out" : "source-over";
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, "rgba(255,255,255,1.0)");
    gradient.addColorStop(0.7, "rgba(255,255,255,0.8)");
    gradient.addColorStop(1.0, "rgba(255,255,255,0.0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw on the mask overlay canvas (colored, visible to user in real-time)
  function drawMaskPreview(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    radius: number, isErase: boolean
  ) {
    const [cr, cg, cb] = TARGET_COLORS[target];
    ctx.globalCompositeOperation = isErase ? "destination-out" : "source-over";
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    if (isErase) {
      // For erase, same as brush data - just remove pixels
      gradient.addColorStop(0, "rgba(255,255,255,1.0)");
      gradient.addColorStop(0.7, "rgba(255,255,255,0.8)");
      gradient.addColorStop(1.0, "rgba(255,255,255,0.0)");
    } else {
      gradient.addColorStop(0, `rgba(${cr},${cg},${cb},0.6)`);
      gradient.addColorStop(0.7, `rgba(${cr},${cg},${cb},0.48)`);
      gradient.addColorStop(1.0, `rgba(${cr},${cg},${cb},0.0)`);
    }
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw on both canvases simultaneously
  function drawDual(x: number, y: number, radius: number, isErase: boolean) {
    const brushCanvas = brushCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (brushCanvas) {
      drawBrushData(brushCanvas.getContext("2d")!, x, y, radius, isErase);
    }
    if (maskCanvas) {
      drawMaskPreview(maskCanvas.getContext("2d")!, x, y, radius, isErase);
    }
  }

  function drawStrokeLineDual(
    x0: number, y0: number, x1: number, y1: number,
    radius: number, isErase: boolean
  ) {
    const dist = Math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2);
    const steps = Math.max(1, Math.ceil(dist / (radius * 0.3)));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      drawDual(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius, isErase);
    }
  }

  // Update cursor position
  function updateCursor(e: React.MouseEvent) {
    const cursor = cursorRef.current;
    const stack = e.currentTarget.closest(".brush-canvas-stack");
    if (!cursor || !stack) return;
    const rect = stack.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const displayDiam = brushSize / displayScale;
    cursor.style.left = `${x - displayDiam / 2}px`;
    cursor.style.top = `${y - displayDiam / 2}px`;
    cursor.style.width = `${displayDiam}px`;
    cursor.style.height = `${displayDiam}px`;
    cursor.style.display = "block";
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    e.preventDefault();
    const [x, y] = canvasCoords(e);
    const isErase = e.button === 2;
    if (isErase) setErasing(true); else setPainting(true);
    drawDual(x, y, brushSize, isErase);
    lastPos.current = [x, y];
    updateCursor(e);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    updateCursor(e);
    if (!painting && !erasing) return;
    const [x, y] = canvasCoords(e);
    if (!lastPos.current) return;
    const [lx, ly] = lastPos.current;
    drawStrokeLineDual(lx, ly, x, y, brushSize, erasing);
    lastPos.current = [x, y];
  }

  function handleMouseLeaveCanvas() {
    const cursor = cursorRef.current;
    if (cursor) cursor.style.display = "none";
    if (painting || erasing) handleMouseUp();
  }

  async function handleMouseUp() {
    if (!painting && !erasing) return;
    setPainting(false);
    setErasing(false);
    lastPos.current = null;

    const brushCanvas = brushCanvasRef.current;
    if (!brushCanvas) return;
    const dataUrl = brushCanvas.toDataURL("image/png");

    try {
      const result = await invoke<BrushApplyResult>("apply_brush_mask_cmd", {
        target,
        brushData: dataUrl,
      });
      onBrushApplied(result);

      // Sync mask overlay from brush data (ensure they stay consistent)
      const maskCanvas = maskCanvasRef.current;
      if (maskCanvas) {
        const maskCtx = maskCanvas.getContext("2d")!;
        maskCtx.clearRect(0, 0, maskWidth, maskHeight);
        const brushCtx = brushCanvas.getContext("2d")!;
        const srcData = brushCtx.getImageData(0, 0, maskWidth, maskHeight);
        const imageData = maskCtx.createImageData(maskWidth, maskHeight);
        const [cr, cg, cb] = TARGET_COLORS[target];
        for (let i = 0; i < srcData.data.length; i += 4) {
          const v = srcData.data[i];
          imageData.data[i] = cr;
          imageData.data[i + 1] = cg;
          imageData.data[i + 2] = cb;
          imageData.data[i + 3] = Math.floor(v * 0.6);
        }
        maskCtx.putImageData(imageData, 0, 0);
      }
    } catch (e) {
      console.error("apply_brush_mask_cmd:", e);
    }
  }

  // Compute CSS dimensions + display scale
  const [canvasStyle, setCanvasStyle] = useState<React.CSSProperties>({});

  useEffect(() => {
    function updateSize() {
      const container = containerRef.current;
      if (!container || maskWidth === 0 || maskHeight === 0) return;

      const cw = container.clientWidth - 24;
      const ch = container.clientHeight - 60; // room for slider + hint
      const aspect = maskWidth / maskHeight;

      let w: number, h: number;
      if (cw / ch > aspect) {
        h = ch;
        w = h * aspect;
      } else {
        w = cw;
        h = w / aspect;
      }

      setCanvasStyle({ width: `${w}px`, height: `${h}px` });
      setDisplayScale(maskWidth / w);
    }

    updateSize();
    const ro = new ResizeObserver(updateSize);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [maskWidth, maskHeight]);

  const targetLabel = target === "eye" ? "目" : target === "mouth" ? "口" : "眉毛";
  const [cr, cg, cb] = TARGET_COLORS[target];

  return (
    <div className="brush-canvas-container" ref={containerRef}>
      {loading && (
        <span className="placeholder">ブラシキャンバス読み込み中...</span>
      )}
      <div className="brush-canvas-stack" style={canvasStyle}>
        <canvas ref={bgCanvasRef} className="layer-bg" width={maskWidth} height={maskHeight} />
        <canvas ref={maskCanvasRef} className="layer-mask" width={maskWidth} height={maskHeight} />
        <canvas
          ref={brushCanvasRef}
          className="layer-brush"
          width={maskWidth}
          height={maskHeight}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeaveCanvas}
          onContextMenu={(e) => e.preventDefault()}
        />
        <div ref={cursorRef} className="brush-cursor" style={{
          display: "none",
          borderColor: `rgba(${cr},${cg},${cb},0.8)`,
        }} />
      </div>
      <div className="brush-toolbar">
        <span className="brush-toolbar-label" style={{ color: `rgb(${cr},${cg},${cb})` }}>
          {targetLabel}
        </span>
        <span className="brush-toolbar-label">サイズ</span>
        <input
          type="range" min={3} max={100}
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span className="brush-toolbar-value">{brushSize}</span>
        <span className="brush-toolbar-hint">
          左: 追加 / 右: 消去
        </span>
      </div>
    </div>
  );
}
