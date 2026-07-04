/**
 * Animation Lab — グローバル名前空間と定数
 * file:// で動作させるため ES modules を使わず、window.AnimLab に集約する。
 */
(function () {
  'use strict';

  window.AnimLab = window.AnimLab || {};
  const NS = window.AnimLab;

  // 母音（SpriTalk src/shared/types/tts.ts VowelType 互換）
  NS.VOWELS = ['a', 'i', 'u', 'e', 'o'];

  // レイヤー名 → z-index（SpriTalk準拠＋腕レイヤー拡張: body と eye の間に arm_l/arm_r）
  NS.LAYER_Z_INDEX = {
    hair_back: -1,
    body: 0,
    arm_l: 1,
    arm_r: 1,
    eye: 2,
    mouth: 3,
    mouth_a: 3,
    mouth_i: 3,
    mouth_u: 3,
    mouth_e: 3,
    mouth_o: 3,
    hair: 4,
  };

  NS.ARM_LAYER_NAMES = ['arm_l', 'arm_r'];

  NS.MOUTH_LAYER_NAMES = ['mouth', 'mouth_a', 'mouth_i', 'mouth_u', 'mouth_e', 'mouth_o'];

  // 母音 → mouth_* レイヤー名（vowelToMouthLayer 互換）
  // 'keep' = 促音っ: 前の口形状を維持 / null = 口閉じ
  NS.vowelToMouthLayer = function (vowel) {
    switch (vowel) {
      case 'a': return 'mouth_a';
      case 'i': return 'mouth_i';
      case 'u': return 'mouth_u';
      case 'e': return 'mouth_e';
      case 'o': return 'mouth_o';
      case 'cl': return 'keep';
      case 'N':
      case 'pause':
      default:
        return null;
    }
  };

  // 母音ごとの目標開度（A4エンベロープ用）
  NS.VOWEL_OPENNESS = { a: 1.0, o: 0.85, e: 0.65, i: 0.5, u: 0.45, N: 0.1, pause: 0 };

  // ===== 共有パラメータ（UIスライダーが直接書き換える） =====
  NS.P = {
    envelope: { mode: 'envelope', attackMs: 40, releaseMs: 90 },
    a1: { smoothTime: 0.05 },
    a2: { blendTimeMs: 80, smoothTime: 0.05 },
    a3: { smoothTime: 0.06, innerStart: 0.15, innerFull: 0.6, jawFactor: 0.35 },
    b0: {
      breathAmp: 5, breathSpeed: 1.0,
      swayAmpX: 2, swayAmpY: 1, swaySpeed: 1.0, reduceOnSpeech: 1,
      hairAmp: 5, hairRot: 0.02, hairSpeed: 1.0,
    },
    b1: { k: 90, c: 10, coupling: 0.025, maxAngle: 0.18 },
    b2: { bounceAmp: 0.05, bounceLambda: 5, bounceFreq: 16, opennessCoupling: 0.7 },
    b3: { k: 70, c: 7, wind: 0.012, drive: 0.03 },
    arm: {
      enabled: 1, segments: 3, k: 90, c: 10, coupling: 0.02, noise: 0.008, maxAngle: 0.12,
      // 肩の弾み: 体の上下動に肩が遅延追従＋発話開始バウンス
      liftEnabled: 1, liftCoupling: 0.08, liftBounce: 26, liftMax: 6,
    },
    blink: { closeMs: 90, openMs: 130, intervalMin: 2, intervalMax: 10, halfLevel: 0.5 },
    face: { driftRange: 0.28, driftSpeed: 0.12, nodAmp: 0.6, nodK: 140, nodC: 16 },
  };

  // 実行時の共有状態
  NS.state = {
    assets: null,        // CharacterAssets
    cells: [],           // Cell[]
    axes: [],            // AxisViewer[]
    running: false,
  };
})();
