/**
 * 口パク方式 A0 / A1 / A2（コンポジター非依存の純ロジック）
 *
 * 共通インターフェース:
 *   { id, label, requiresPixi, create(assets) => { reset(), update(dtMs, now, mora) => MouthCommand } }
 *
 * MouthCommand.entries = [{layer, frame, opacity}] （空配列 = 口レイヤー非表示）
 */
(function (NS) {
  'use strict';

  const U = NS.U;

  /** 利用可能な mouth_* レイヤー集合とデフォルトレイヤーを取得 */
  function mouthInfo(assets) {
    const available = new Set();
    for (const name of NS.MOUTH_LAYER_NAMES) {
      if (assets.byName.has(name)) available.add(name);
    }
    const defaultMouth = available.has('mouth_a') ? 'mouth_a' : (available.values().next().value || null);
    const frameCount = defaultMouth ? assets.byName.get(defaultMouth).frames.length : 1;
    return { available, defaultMouth, frameCount };
  }

  function frameCountOf(assets, layer, fallback) {
    const l = assets.byName.get(layer);
    return l ? l.frames.length : fallback;
  }

  // ============================================================
  // A0: 現行方式再現（ベースライン）
  // 移植元: src/windows/character/components/CharacterApp.tsx moraLipSyncLoop (L437-541)
  // 50ms基準の等速ステップ送り。母音→別母音は lastFrame-1 まで閉じてから
  // レイヤー切替→再度開く（高速スイッチオーバー方式）。
  // ============================================================
  const A0 = {
    id: 'A0', label: 'A0: 現行方式（ステップ送り）', requiresPixi: false,
    create(assets) {
      const info = mouthInfo(assets);
      let displayedLayer = null;
      let currentFrame = 0;
      let lastStepTime = 0;

      function command() {
        return {
          kind: 'entries',
          entries: displayedLayer === null ? [] : [{ layer: displayedLayer, frame: currentFrame, opacity: 1 }],
        };
      }

      return {
        reset() { displayedLayer = null; currentFrame = 0; lastStepTime = 0; },
        update(dtMs, now, mora) {
          const rawTarget = NS.vowelToMouthLayer(mora.playing ? mora.vowel : 'pause');
          if (rawTarget === 'keep') return command(); // 促音: 維持

          let targetLayer = rawTarget;
          const lastFrame = info.frameCount - 1;
          const stepDuration = Math.max(16, Math.floor(50 / Math.max(lastFrame, 1)));

          if (now - lastStepTime < stepDuration) return command();

          // 無音方向: 徐々に閉じる
          if (targetLayer === null) {
            if (currentFrame > 0) {
              currentFrame -= 1;
              lastStepTime = now;
            } else if (displayedLayer !== null && info.defaultMouth && displayedLayer !== info.defaultMouth) {
              displayedLayer = info.defaultMouth; // 閉じ口はデフォルトレイヤーのframe0で維持
            }
            return command();
          }

          // フォールバック
          if (!info.available.has(targetLayer)) {
            if (info.defaultMouth === null) return command();
            targetLayer = info.defaultMouth;
          }

          if (displayedLayer === targetLayer) {
            // 同一レイヤー: 全開へ向けてステップ
            if (currentFrame < lastFrame) currentFrame += 1;
            else if (currentFrame > lastFrame) currentFrame -= 1;
            lastStepTime = now;
          } else if (displayedLayer === null) {
            // 無音→母音
            displayedLayer = targetLayer;
            currentFrame = currentFrame < lastFrame ? currentFrame + 1 : lastFrame;
            lastStepTime = now;
          } else {
            // 母音→別母音: switchoverFrame まで閉じてから切替（本体仕様の忠実再現）
            const switchoverFrame = Math.max(0, lastFrame - 1);
            if (currentFrame > switchoverFrame) {
              currentFrame -= 1;
            } else {
              displayedLayer = targetLayer;
              currentFrame = currentFrame < lastFrame ? currentFrame + 1 : lastFrame;
            }
            lastStepTime = now;
          }
          return command();
        },
      };
    },
  };

  // ============================================================
  // A1: 調音結合対応
  // openness（連続開度）を保持し、母音が変わっても閉じずにレイヤーだけ即差し替え。
  // フレーム番号 = round(openness * lastFrame)。開度はエンベロープ駆動＋SmoothDamp。
  // ============================================================
  const A1 = {
    id: 'A1', label: 'A1: 調音結合（閉じない遷移）', requiresPixi: false,
    create(assets) {
      const info = mouthInfo(assets);
      let currentLayer = null;
      let openness = 0;
      const vel = { v: 0 };

      return {
        reset() { currentLayer = null; openness = 0; vel.v = 0; },
        update(dtMs, now, mora) {
          const dt = dtMs / 1000;
          const rawTarget = NS.vowelToMouthLayer(mora.playing ? mora.vowel : 'pause');

          if (rawTarget !== 'keep' && rawTarget !== null) {
            // 母音レイヤーへ即切替（openness維持 = 調音結合）
            currentLayer = info.available.has(rawTarget) ? rawTarget : info.defaultMouth;
          }
          // rawTarget null（pause/N）はレイヤー維持のまま開度が0へ向かう

          openness = U.clamp(U.smoothDamp(openness, mora.openness, vel, NS.P.a1.smoothTime, dt), 0, 1);

          if (currentLayer === null || openness < 0.005) {
            return { kind: 'entries', entries: currentLayer === null ? [] : [{ layer: currentLayer, frame: 0, opacity: 1 }] };
          }
          const lastFrame = frameCountOf(assets, currentLayer, info.frameCount) - 1;
          const frame = Math.round(openness * lastFrame);
          return { kind: 'entries', entries: [{ layer: currentLayer, frame, opacity: 1 }] };
        },
      };
    },
  };

  // ============================================================
  // A2: クロスフェード合成
  // 端点キーフレーム（各母音の全開フレーム）のみ使用。
  // 母音変化時に front/back の2スロットで opacity ブレンド、
  // 全体の開度は SmoothDamp で連続化（＝RIFE中間フレーム不要の検証）。
  // ============================================================
  const A2 = {
    id: 'A2', label: 'A2: クロスフェード（端点のみ）', requiresPixi: false,
    create(assets) {
      const info = mouthInfo(assets);
      let front = null;   // 現在の母音レイヤー
      let back = null;    // 直前の母音レイヤー
      let blendT = 1;     // 0→1 でback→frontへ遷移
      let openness = 0;
      const vel = { v: 0 };

      return {
        reset() { front = null; back = null; blendT = 1; openness = 0; vel.v = 0; },
        update(dtMs, now, mora) {
          const dt = dtMs / 1000;
          const rawTarget = NS.vowelToMouthLayer(mora.playing ? mora.vowel : 'pause');

          if (rawTarget !== 'keep' && rawTarget !== null) {
            const layer = info.available.has(rawTarget) ? rawTarget : info.defaultMouth;
            if (layer !== front) {
              back = front;
              front = layer;
              blendT = 0;
            }
          }

          blendT = U.clamp(blendT + dtMs / Math.max(1, NS.P.a2.blendTimeMs), 0, 1);
          openness = U.clamp(U.smoothDamp(openness, mora.openness, vel, NS.P.a2.smoothTime, dt), 0, 1);

          const entries = [];
          if (front && openness > 0.01) {
            const fLast = frameCountOf(assets, front, info.frameCount) - 1;
            entries.push({ layer: front, frame: fLast, opacity: openness * blendT });
            if (back && blendT < 1) {
              const bLast = frameCountOf(assets, back, info.frameCount) - 1;
              entries.push({ layer: back, frame: bLast, opacity: openness * (1 - blendT) });
            }
          }
          return { kind: 'entries', entries };
        },
      };
    },
  };

  NS.MouthMethods = [A0, A1, A2]; // A3 は 21-mouth-mesh-a3.js で追加
})(window.AnimLab);
