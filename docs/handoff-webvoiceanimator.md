# WebVoiceAnimator改修ハンドオフ — レイヤーモード追加（PachiPakuGen素材対応）

WebVoiceAnimatorリポジトリでの実装作業者（AIモデル含む）向けの自己完結設計書。
全体の経緯は [animation-lab-tech.md](animation-lab-tech.md) §10、
プロファイル仕様は [motion-lab-integration.md](motion-lab-integration.md) を参照。

## 目的

PachiPakuGenの出力パッケージ（素材フォルダ＋spritalk-motion-profile.json）を
WebVoiceAnimatorでも読み込み、**配信者・コラボ相手の生音声**で
同じキャラクターを動かせるようにする。SpriTalk（TTS/モーラ駆動）との違いは
駆動源だけで、アニメーションランタイムは共通。

## 設計原則: 駆動源とランタイムの分離

アニメーションランタイム（A1調音結合・髪/腕/胸のバネ物理・登場演出）は
すべて以下の駆動信号 `MoraState` だけで動く:

```ts
interface MoraState {
  vowel: 'a'|'i'|'u'|'e'|'o'|'N'|'pause'|undefined;
  openness: number;        // 0..1 連続開度
  speaking: boolean;
  speechStarted: boolean;  // このフレームで発話開始（撃力トリガ）
}
```

WVA側の作業の中心は **HQ Audio出力→MoraStateへのアダプタ**:

| MoraStateフィールド | WVAの既存資産からの生成 |
|--------------------|------------------------|
| openness | HQ Audioエンベロープフォロワー出力（ノイズフロア・ピーク追跡済み） |
| vowel | 既存の母音判定（AudioWorklet 約60fps、音量＋周波数解析。MPNGモードで実証済み） |
| speaking | ヒステリシス付き発話判定（既存） |
| speechStarted | speaking の立ち上がりエッジ |

※SpriTalkの息継ぎ（pause_mora予知駆動）はモーラ情報が無いWVAでは再現しない。
　代わりに「無音が一定時間続いた後の発話再開」で登場撃力相当の小バウンスを発火してよい

## 実装項目（優先順）

### 1. レイヤーモード（第4のモード）
- PachiPakuGenパッケージ（body/hair/hair_back/arm_l/arm_r/chest/eye/mouth_a〜o
  ＋spritalk-motion-profile.json）の読込
- 共通ランタイム移植元: PachiPakuGen `reference/animation-lab/js/`
  （20-mouth-methods.js のA1 / 30-body-methods.js のB3 / 32-arm-sway.js /
  02-utils.js。依存なしIIFE、既存のPixiJS基盤にそのまま載る）
- プロファイルの物理パラメータを適用。WVA側に詳細設定UIは追加しない
  （既存のシンプル設定思想を維持。強さ倍率1本は可）

### 2. キャラクターごとの音声ソース割当（コラボ対応）
- 複数キャラクター登録は実装済み。拡張点は各キャラに音声入力デバイスを
  割り当てるUI（マイク / タブ音声 / 仮想オーディオデバイス）
- コラボ相手はDiscord等→仮想オーディオデバイス経由で割当（相手の声で相手のキャラが動く）
- 話者分離（1本の音声の話者判定）はスコープ外。「1ソース=1キャラ」で確実に動かす
- 技術注意: 既存のOffscreen Document API構成でAudioWorkletを複数ソース分
  並列に走らせる際のCPU負荷を確認（キャラ数×60fps解析）

### 3. カメラ感情推定（任意・最後）
- MediaPipe（Apache-2.0）Face Landmarker のブレンドシェイプから簡易感情分類
  →感情フォルダ（emotionId相当）の自動切替
- 切替時は物理を引き継がず小さな切替撃力（SpriTalkと同じ演出規約）
- 対象は配信者本人のキャラのみ（コラボ相手のカメラは扱わない）

## 検証方法
- PachiPakuGenのMotion Labプレビュー／reference/animation-lab と同一素材で
  見た目一致を確認（駆動源の違いによる差だけが残ること）
- マイクで「あ・い・う・え・お」をゆっくり発声し、母音レイヤーの追従と
  調音結合（母音間で口が閉じない）を確認
