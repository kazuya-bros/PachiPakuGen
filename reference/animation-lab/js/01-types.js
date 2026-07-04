/**
 * 型定義（JSDoc typedef）— SpriTalk本体の型と互換
 * 移植元: src/shared/types/tts.ts
 */
(function () {
  'use strict';

  /**
   * @typedef {'a'|'i'|'u'|'e'|'o'|'N'|'pause'|'cl'} VowelType
   *   a〜o: 5母音 / N: 撥音ん（口閉じ） / cl: 促音っ（前の形状維持） / pause: 無音
   */

  /**
   * @typedef {Object} MoraTiming — SpriTalk MoraTiming 完全互換
   * @property {VowelType} vowel        母音の種類
   * @property {string} text            モーラのテキスト（例: "こ"）
   * @property {number} startTime       母音の開始時刻（ミリ秒、音声先頭からの累積）
   * @property {number} duration        母音の持続時間（ミリ秒）
   * @property {string} [consonant]     子音（例: "k"）
   * @property {number} [consonantDuration] 子音の持続時間（ミリ秒）。子音は startTime の前に位置する
   */

  /**
   * @typedef {Object} MoraState — 毎フレーム全方式へ配られる駆動信号
   * @property {number} elapsed         再生経過時間（ミリ秒）
   * @property {VowelType|undefined} vowel 現在の母音（モーラ外は undefined）
   * @property {VowelType|undefined} prevVowel 直前の異なる母音
   * @property {number} vowelChangedAt  母音が変わった時刻（ミリ秒）
   * @property {boolean} speaking       発話中か（vowel が pause/undefined 以外）
   * @property {boolean} speechStarted  このフレームで発話が始まったか（イベント）
   * @property {number} openness        A4エンベロープ出力（0..1）
   * @property {number} rawTarget       エンベロープ前の目標開度（矩形値）
   * @property {boolean} playing        再生中か
   */

  /**
   * @typedef {Object} FrameImage
   * @property {ImageBitmap} bitmap     pixi/canvas 用
   * @property {string} url             DOM <img> 用（ObjectURL または dataURL）
   */

  /**
   * @typedef {Object} LayerAsset
   * @property {string} name            'body'|'hair'|'hair_back'|'eye'|'mouth_a'...
   * @property {number} zIndex
   * @property {FrameImage[]} frames    連番フレーム（PachiPakuGen出力順）
   */

  /**
   * @typedef {Object} CharacterAssets
   * @property {number} width           基準キャンバス幅（body.png）
   * @property {number} height
   * @property {LayerAsset[]} layers    z-index順ソート済み
   * @property {Map<string, LayerAsset>} byName
   * @property {Map<string, LayerAsset>} axes  axis_* フォルダ（汎用パラメータ軸）
   * @property {number} totalBytes      概算メモリ（生ピクセル）
   */

  /**
   * @typedef {Object} MouthCommand — コンポジターへの描画命令
   * @property {'entries'|'mesh'} kind
   * @property {{layer:string, frame:number, opacity:number}[]} [entries] 表示する口画像の集合（空=口閉じ）
   * @property {Object} [mesh]          A3用: {vowel, prevVowel, blend, openness}
   */

  /**
   * @typedef {Object} BodyPose — 体・髪の変形命令
   * @property {{x:number,y:number,rot:number,scaleX:number,scaleY:number}} root
   * @property {{x:number,rot:number}} hair
   * @property {{x:number,rot:number}} hairBack
   * @property {number[]|null} hairMeshAngles  B3用チェーン角（rad, 根元→毛先）
   */
})();
