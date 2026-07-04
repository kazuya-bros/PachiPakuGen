/**
 * 体・髪アニメーション方式 B0 / B1 / B2 / B3（純ロジック、BodyPose出力）
 *
 * B0 移植元: src/shared/animation/procedural-animator.ts
 *   呼吸: 周期3秒 sin → Y / 揺れ: X周期5秒 sin・Y周期7秒 cos / 前髪: 周期2.5秒（rotは0.7倍位相）
 *   後ろ髪: 周期3秒（rotは0.6倍位相） / 発話中 damping 0.2 へ 3.0/s で遷移
 */
(function (NS) {
  'use strict';

  const U = NS.U;
  const TAU = Math.PI * 2;
  const SPEECH_DAMPING_SPEED = 3.0;

  function zeroPose() {
    return {
      root: { x: 0, y: 0, rot: 0, scaleX: 1, scaleY: 1 },
      hair: { x: 0, rot: 0 },
      hairBack: { x: 0, rot: 0 },
      hairMeshAngles: null,
    };
  }

  /** B0の呼吸＋アイドル揺れ（共通利用）。ampScale で発話連動変調も可 */
  function makeIdleState() {
    return { breath: 0, swayX: 0, swayY: 0, hair: 0, hairBack: 0, damping: 1, prevRootX: 0, rootVX: 0 };
  }

  function updateIdleRoot(st, dt, speaking, ampScaleOverride) {
    const P = NS.P.b0;
    // 発話減衰のスムーズ遷移（本体仕様）
    const target = speaking && P.reduceOnSpeech ? 0.2 : 1.0;
    if (st.damping < target) st.damping = Math.min(st.damping + SPEECH_DAMPING_SPEED * dt, target);
    else if (st.damping > target) st.damping = Math.max(st.damping - SPEECH_DAMPING_SPEED * dt, target);

    const scale = ampScaleOverride != null ? ampScaleOverride : st.damping;

    st.breath += dt * P.breathSpeed * (TAU / 3);
    st.swayX += dt * P.swaySpeed * (TAU / 5);
    st.swayY += dt * P.swaySpeed * (TAU / 7);
    if (st.breath > Math.PI * 200) st.breath -= Math.PI * 200;
    if (st.swayX > Math.PI * 200) st.swayX -= Math.PI * 200;
    if (st.swayY > Math.PI * 200) st.swayY -= Math.PI * 200;

    const x = Math.sin(st.swayX) * P.swayAmpX * scale;
    const y = Math.sin(st.breath) * P.breathAmp + Math.cos(st.swayY) * P.swayAmpY * scale;

    // 頭（root）のX速度を計測（B1/B3の駆動源）
    st.rootVX = dt > 0 ? (x - st.prevRootX) / dt : 0;
    st.prevRootX = x;
    return { x, y };
  }

  function updateHairSin(st, dt, pose) {
    const P = NS.P.b0;
    st.hair += dt * P.hairSpeed * (TAU / 2.5);
    st.hairBack += dt * P.hairSpeed * (TAU / 3);
    if (st.hair > Math.PI * 200) st.hair -= Math.PI * 200;
    if (st.hairBack > Math.PI * 200) st.hairBack -= Math.PI * 200;
    pose.hair.x = Math.sin(st.hair) * P.hairAmp;
    pose.hair.rot = Math.sin(st.hair * 0.7) * P.hairRot;
    pose.hairBack.x = Math.sin(st.hairBack) * P.hairAmp * 0.8;
    pose.hairBack.rot = Math.sin(st.hairBack * 0.6) * P.hairRot;
  }

  /** バネ-ダンパー（半陰的オイラー＋サブステップ） */
  function springStep(state, target, k, c, dt) {
    const steps = dt > 1 / 60 ? 2 : 1;
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const a = -k * (state.x - target) - c * state.v;
      state.v += a * h;
      state.x += state.v * h;
    }
    return state.x;
  }

  // ============================================================
  const B0 = {
    id: 'B0', label: 'B0: 現行方式（sin波）', requiresPixi: false,
    create() {
      const st = makeIdleState();
      return {
        update(dt, ctx) {
          const pose = zeroPose();
          const r = updateIdleRoot(st, dt, ctx.speaking);
          pose.root.x = r.x; pose.root.y = r.y;
          updateHairSin(st, dt, pose);
          return pose;
        },
      };
    },
  };

  // ============================================================
  const B1 = {
    id: 'B1', label: 'B1: スプリング遅延追従', requiresPixi: false,
    create() {
      const st = makeIdleState();
      const hairSpring = { x: 0, v: 0 };
      const backSpring = { x: 0, v: 0 };
      return {
        update(dt, ctx) {
          const P = NS.P.b1;
          const pose = zeroPose();
          const r = updateIdleRoot(st, dt, ctx.speaking);
          pose.root.x = r.x; pose.root.y = r.y;

          // 頭が右へ動くと髪は左へ流れる（速度カップリング）
          const dtc = Math.min(dt, 0.1);
          const target = U.clamp(-P.coupling * st.rootVX, -P.maxAngle, P.maxAngle);
          const theta = U.clamp(springStep(hairSpring, target, P.k, P.c, dtc), -P.maxAngle, P.maxAngle);
          // 後ろ髪: 柔らかく（低剛性=より遅延）・少し大きく
          const thetaB = U.clamp(springStep(backSpring, target * 1.25, P.k * 0.45, P.c * 0.8, dtc), -P.maxAngle * 1.4, P.maxAngle * 1.4);

          pose.hair.rot = theta;
          pose.hair.x = theta * 26;      // 回転に伴う平行移動（房のしなり感）
          pose.hairBack.rot = thetaB;
          pose.hairBack.x = thetaB * 34;
          return pose;
        },
      };
    },
  };

  // ============================================================
  const B2 = {
    id: 'B2', label: 'B2: 発話反応（バウンス＋開度連動）', requiresPixi: false,
    create() {
      const st = makeIdleState();
      let bounceT = -1; // 発話開始からの経過秒。-1=非アクティブ
      return {
        update(dt, ctx) {
          const P = NS.P.b2;
          const pose = zeroPose();

          // 現行の「一律20%減衰」の逆: 喋るほど動く（開度連動の振幅変調）
          const ampScale = ctx.speaking ? 0.3 + 0.7 * ctx.openness : 1.0;
          const r = updateIdleRoot(st, dt, false, ampScale);
          pose.root.x = r.x; pose.root.y = r.y;
          updateHairSin(st, dt, pose);

          // 発話開始で減衰振動バウンス（squash & stretch、体積保存）
          if (ctx.speechStarted) bounceT = 0;
          if (bounceT >= 0) {
            bounceT += dt;
            const env = Math.exp(-NS.P.b2.bounceLambda * bounceT);
            if (env < 0.01) {
              bounceT = -1;
            } else {
              const s = P.bounceAmp * env * Math.sin(P.bounceFreq * bounceT);
              pose.root.scaleY = 1 + s;
              pose.root.scaleX = 1 - s * 0.7; // 体積保存近似
            }
          }
          return pose;
        },
      };
    },
  };

  // ============================================================
  // B3: メッシュ髪揺れ（角度チェーン物理。描画は31-hair-mesh-b3.jsのpixiランタイム）
  const B3_SEGMENTS = 6;
  const B3 = {
    id: 'B3', label: 'B3: メッシュ髪揺れ（pixi）', requiresPixi: true,
    create() {
      const st = makeIdleState();
      const n = B3_SEGMENTS;
      const angles = new Float32Array(n);
      const omegas = new Float32Array(n);
      let t = Math.random() * 100;
      return {
        update(dt, ctx) {
          const P = NS.P.b3;
          const pose = zeroPose();
          const r = updateIdleRoot(st, dt, ctx.speaking);
          pose.root.x = r.x; pose.root.y = r.y;

          t += dt;
          const dtc = Math.min(dt, 0.1);
          const steps = dtc > 1 / 60 ? 2 : 1;
          const h = dtc / steps;
          const wind = Math.sin(t * 1.7) * P.wind + U.noise1d(t * 0.6) * P.wind * 0.6;
          const drive = U.clamp(-P.drive * st.rootVX * 0.05, -0.2, 0.2);

          for (let s = 0; s < steps; s++) {
            for (let i = 0; i < n; i++) {
              // 親行に追従（1行分の伝播遅延が波を生む）。根元の目標=頭の速度駆動＋風
              const target = i === 0 ? drive + wind : angles[i - 1];
              const k = P.k * (1 - 0.6 * (i / n)); // 毛先ほど柔らかい
              const a = -k * (angles[i] - target) - P.c * omegas[i];
              omegas[i] += a * h;
              angles[i] += omegas[i] * h;
              angles[i] = U.clamp(angles[i], -0.5, 0.5);
            }
          }
          pose.hairMeshAngles = Array.from(angles);
          // 後ろ髪はB1相当の回転で追従
          pose.hairBack.rot = angles[Math.floor(n / 2)] * 0.8;
          pose.hairBack.x = pose.hairBack.rot * 30;
          return pose;
        },
      };
    },
  };

  B3.SEGMENTS = B3_SEGMENTS;
  NS.BodyMethods = [B0, B1, B2, B3];
})(window.AnimLab);
