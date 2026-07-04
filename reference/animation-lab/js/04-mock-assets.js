/**
 * モック素材ジェネレータ
 * 手持ち素材がなくても全機能を試せるよう、簡易キャラクターを
 * Canvasでプログラム描画して CharacterAssets を組み立てる。
 * 顔向き軸（axis_face_lr / axis_face_ud）のモックもここで生成する。
 */
(function (NS) {
  'use strict';

  const W = 512, H = 512;
  const FACE = { cx: 256, cy: 205, r: 118 };   // 頭の中心・半径
  const MOUTH = { cx: 256, cy: 272 };
  const EYE_L = { cx: 213, cy: 216 };
  const EYE_R = { cx: 299, cy: 216 };

  const SKIN = '#ffe3c8';
  const SKIN_EDGE = '#d9a97f';
  const HAIR_COL = '#5a6fbf';
  const HAIR_DARK = '#42539a';

  function mkCanvas() {
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    return c;
  }

  async function toFrame(canvas) {
    return {
      bitmap: await createImageBitmap(canvas),
      url: canvas.toDataURL('image/png'),
    };
  }

  // ===== パーツ描画（headDx/headDy で頭部グループを平行移動 = 顔向きモック） =====

  function headTransform(ctx, opt, fn) {
    ctx.save();
    ctx.translate(opt.headDx || 0, opt.headDy || 0);
    ctx.rotate((opt.headDx || 0) * 0.0012);
    fn();
    ctx.restore();
  }

  function drawHairBack(ctx, opt = {}) {
    headTransform(ctx, opt, () => {
      ctx.fillStyle = HAIR_DARK;
      ctx.beginPath();
      ctx.ellipse(FACE.cx, FACE.cy + 40, FACE.r + 34, FACE.r + 80, 0, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawBody(ctx, opt = {}) {
    // 胴体（頭部オフセットの影響を受けない）
    ctx.fillStyle = '#7c8aa5';
    ctx.beginPath();
    ctx.moveTo(176, 512);
    ctx.quadraticCurveTo(180, 350, 256, 340);
    ctx.quadraticCurveTo(332, 350, 336, 512);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = SKIN;
    ctx.fillRect(238, 310, 36, 44); // 首

    headTransform(ctx, opt, () => {
      // 頭
      ctx.fillStyle = SKIN;
      ctx.strokeStyle = SKIN_EDGE;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.ellipse(FACE.cx, FACE.cy, FACE.r, FACE.r + 8, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      // 鼻
      ctx.strokeStyle = SKIN_EDGE;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(FACE.cx - 2, 238);
      ctx.lineTo(FACE.cx + 3, 246);
      ctx.stroke();
      // 頬
      ctx.fillStyle = 'rgba(255,140,140,0.35)';
      ctx.beginPath(); ctx.ellipse(FACE.cx - 74, 240, 16, 9, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(FACE.cx + 74, 240, 16, 9, 0, 0, Math.PI * 2); ctx.fill();
      // 閉じ口（body.png に口閉じ状態が含まれる前提 = SpriTalk仕様）
      drawClosedMouthLine(ctx);
    });
  }

  function drawClosedMouthLine(ctx) {
    ctx.strokeStyle = '#a3543f';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(MOUTH.cx - 20, MOUTH.cy);
    ctx.quadraticCurveTo(MOUTH.cx, MOUTH.cy + 7, MOUTH.cx + 20, MOUTH.cy);
    ctx.stroke();
  }

  function drawHair(ctx, opt = {}) {
    headTransform(ctx, opt, () => {
      ctx.fillStyle = HAIR_COL;
      ctx.beginPath();
      ctx.ellipse(FACE.cx, FACE.cy - 34, FACE.r + 14, FACE.r - 4, 0, Math.PI, Math.PI * 2);
      // 前髪ギザギザ
      const y0 = FACE.cy - 34;
      for (let i = 0; i <= 6; i++) {
        const x = FACE.cx + FACE.r + 14 - (i * (2 * (FACE.r + 14)) / 6);
        const y = y0 + (i % 2 === 0 ? 4 : 30);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      // サイドの房
      ctx.beginPath();
      ctx.moveTo(FACE.cx - FACE.r - 6, y0);
      ctx.quadraticCurveTo(FACE.cx - FACE.r - 26, FACE.cy + 90, FACE.cx - FACE.r + 10, FACE.cy + 130);
      ctx.quadraticCurveTo(FACE.cx - FACE.r - 2, FACE.cy + 40, FACE.cx - FACE.r + 18, y0 + 10);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(FACE.cx + FACE.r + 6, y0);
      ctx.quadraticCurveTo(FACE.cx + FACE.r + 26, FACE.cy + 90, FACE.cx + FACE.r - 10, FACE.cy + 130);
      ctx.quadraticCurveTo(FACE.cx + FACE.r + 2, FACE.cy + 40, FACE.cx + FACE.r - 18, y0 + 10);
      ctx.closePath();
      ctx.fill();
    });
  }

  /** 腕: side = -1(左) / +1(右)。肩から手先へのカプセル形状 */
  function drawArm(ctx, side, opt = {}) {
    const sx = 256 + side * 62;   // 肩
    const sy = 356;
    const hx = 256 + side * 88;   // 手先
    const hy = 474;
    ctx.strokeStyle = '#68758f';
    ctx.lineWidth = 32;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(256 + side * 86, 410, hx, hy);
    ctx.stroke();
    // 手
    ctx.fillStyle = SKIN;
    ctx.beginPath();
    ctx.arc(hx, hy + 4, 15, 0, Math.PI * 2);
    ctx.fill();
  }

  /** 目: t=0 全開, t=1 閉じ */
  function drawEyes(ctx, t, opt = {}) {
    headTransform(ctx, opt, () => {
      for (const e of [EYE_L, EYE_R]) {
        const openH = 22 * (1 - t);
        if (openH > 2) {
          ctx.fillStyle = '#ffffff';
          ctx.strokeStyle = '#4a3b32';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.ellipse(e.cx, e.cy, 20, openH, 0, 0, Math.PI * 2);
          ctx.fill(); ctx.stroke();
          // 虹彩
          ctx.fillStyle = '#4a6fd8';
          ctx.beginPath();
          ctx.ellipse(e.cx, e.cy, 9, Math.min(12, openH), 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#20242c';
          ctx.beginPath();
          ctx.ellipse(e.cx, e.cy, 4.5, Math.min(6, openH * 0.6), 0, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // 閉じ: まつ毛ライン
          ctx.strokeStyle = '#4a3b32';
          ctx.lineWidth = 4;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(e.cx - 18, e.cy + 2);
          ctx.quadraticCurveTo(e.cx, e.cy + 9, e.cx + 18, e.cy + 2);
          ctx.stroke();
        }
      }
    });
  }

  // 母音別の口形状（全開時の rx/ry と歯の見え方）
  const MOUTH_SHAPES = {
    a: { rx: 33, ry: 30, teeth: true },
    i: { rx: 38, ry: 9, teeth: true },
    u: { rx: 14, ry: 17, teeth: false },
    e: { rx: 31, ry: 17, teeth: true },
    o: { rx: 18, ry: 27, teeth: false },
  };

  /** 口: t=0 閉じ, t=1 その母音の全開 */
  function drawMouth(ctx, vowel, t, opt = {}) {
    headTransform(ctx, opt, () => {
      // 肌パッチ: 口周辺を不透明に被覆する（A3メッシュワープ時に下の閉じ口が
      // 透けない = 「キーフレームは口周辺の肌ごと含める」素材要件のモック実証）
      ctx.fillStyle = SKIN;
      ctx.beginPath();
      ctx.ellipse(MOUTH.cx, MOUTH.cy + 6, 52, 42, 0, 0, Math.PI * 2);
      ctx.fill();
      if (t <= 0.02) {
        drawClosedMouthLine(ctx);
        return;
      }
      const s = MOUTH_SHAPES[vowel];
      const rx = 20 + (s.rx - 20) * t;
      const ry = Math.max(2.5, s.ry * t);
      // 口内
      ctx.fillStyle = '#7e2f27';
      ctx.strokeStyle = '#a3543f';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.ellipse(MOUTH.cx, MOUTH.cy + ry * 0.35, rx, ry, 0, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      // 舌
      if (ry > 10) {
        ctx.fillStyle = '#d96a5e';
        ctx.beginPath();
        ctx.ellipse(MOUTH.cx, MOUTH.cy + ry * 0.35 + ry * 0.55, rx * 0.62, ry * 0.4, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      // 上歯
      if (s.teeth && ry > 5) {
        ctx.fillStyle = '#fdf6ee';
        ctx.beginPath();
        ctx.ellipse(MOUTH.cx, MOUTH.cy + ry * 0.35 - ry * 0.72, rx * 0.8, Math.min(5, ry * 0.3), 0, 0, Math.PI);
        ctx.fill();
      }
    });
  }

  // ===== アセット組み立て =====

  async function layerFromDraws(name, drawFns) {
    const frames = [];
    for (const fn of drawFns) {
      const c = mkCanvas();
      fn(c.getContext('2d'));
      frames.push(await toFrame(c));
    }
    return { name, zIndex: NS.LAYER_Z_INDEX[name] != null ? NS.LAYER_Z_INDEX[name] : 0, frames };
  }

  async function generateAssets(mouthFrameCount = 5, eyeFrameCount = 5) {
    const layers = [];

    layers.push(await layerFromDraws('hair_back', [(ctx) => drawHairBack(ctx)]));
    layers.push(await layerFromDraws('body', [(ctx) => drawBody(ctx)]));
    layers.push(await layerFromDraws('arm_l', [(ctx) => drawArm(ctx, -1)]));
    layers.push(await layerFromDraws('arm_r', [(ctx) => drawArm(ctx, 1)]));
    layers.push(await layerFromDraws('hair', [(ctx) => drawHair(ctx)]));

    // eye: frame0=全開 → 最終=閉じ（PachiPakuGen出力と同じ並び）
    const eyeDraws = [];
    for (let i = 0; i < eyeFrameCount; i++) {
      const t = i / (eyeFrameCount - 1);
      eyeDraws.push((ctx) => drawEyes(ctx, t));
    }
    layers.push(await layerFromDraws('eye', eyeDraws));

    // mouth_a〜o: frame0=閉じ → 最終=全開
    for (const v of NS.VOWELS) {
      const draws = [];
      for (let i = 0; i < mouthFrameCount; i++) {
        const t = i / (mouthFrameCount - 1);
        draws.push((ctx) => drawMouth(ctx, v, t));
      }
      layers.push(await layerFromDraws('mouth_' + v, draws));
    }

    // 顔向き軸モック（全身合成フレームの連番 = 「焼き込みパラメータ軸」の概念実証）
    const axes = new Map();
    axes.set('axis_face_lr', await axisFrames(9, (k, n) => ({ headDx: -28 + (56 * k) / (n - 1), headDy: 0 })));
    axes.set('axis_face_ud', await axisFrames(7, (k, n) => ({ headDx: 0, headDy: -16 + (34 * k) / (n - 1) })));

    const assets = finalizeAssets(layers, axes, W, H);
    assets.isMock = true;
    return assets;
  }

  async function axisFrames(count, optFn) {
    const frames = [];
    for (let k = 0; k < count; k++) {
      const opt = optFn(k, count);
      const c = mkCanvas();
      const ctx = c.getContext('2d');
      drawHairBack(ctx, opt);
      drawBody(ctx, opt);
      drawArm(ctx, -1, opt);
      drawArm(ctx, 1, opt);
      drawEyes(ctx, 0, opt);
      drawHair(ctx, opt);
      frames.push(await toFrame(c));
    }
    return { name: 'axis', zIndex: 0, frames };
  }

  /** レイヤー配列＋軸Mapから CharacterAssets を組み立てる（ローダーと共用） */
  function finalizeAssets(layers, axes, width, height) {
    layers.sort((a, b) => a.zIndex - b.zIndex);
    const byName = new Map();
    let totalBytes = 0;
    for (const l of layers) {
      byName.set(l.name, l);
      for (const f of l.frames) totalBytes += f.bitmap.width * f.bitmap.height * 4;
    }
    for (const [, ax] of axes) {
      for (const f of ax.frames) totalBytes += f.bitmap.width * f.bitmap.height * 4;
    }
    return { width, height, layers, byName, axes, totalBytes };
  }

  NS.Mock = { generateAssets };
  NS.finalizeAssets = finalizeAssets;
})(window.AnimLab);
