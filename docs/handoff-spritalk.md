# SpriTalk改修ハンドオフ — アニメーション刷新（A1調音結合＋物理揺れ）

SpriTalkリポジトリでの実装作業者（AIモデル含む）向けの自己完結設計書。
経緯・数式の詳細は [animation-lab-tech.md](animation-lab-tech.md)、
プロファイル仕様は [motion-lab-integration.md](motion-lab-integration.md) を参照。
参照実装は本リポジトリ `reference/animation-lab/`（SpriTalkリポジトリの
`samples/animation-lab/` と同一。ブランチ `claude/busy-haibt-144d8f` 参照）。

## 採用決定事項（2026-07-04〜05 比較検証済み）

| 項目 | 決定 |
|------|------|
| 口パク | **A1調音結合**: 開度(0..1)を独立保持し、母音間で閉じずにmouth_レイヤー即差し替え。フレーム=round(開度×最終フレーム) |
| 開度駆動 | **A4エンベロープ**: 母音別目標開度(a=1.0/o=0.85/e=0.65/i=0.5/u=0.45/N=0.1/pause=0)＋attack40ms/release90msの指数スムージング |
| 髪 | **B3角度チェーン**: 縦6分割、親行追従バネ、毛先ほど低剛性(k×(1-0.6i/n))、WebGL/Canvasメッシュ描画 |
| 腕 | **角度チェーン（3分割）＋肩の弾み(lift)**: 肩ピボットは不透明bbox上端中央を自動推定。liftは体Y速度への遅延追従＋発話撃力 |
| 胸 | **縦バネ**（肩liftの半分の剛性・強減衰・最大±4px）。chest.png読込時のみ |
| 設定UI | **3項目のみ**: アニメーションON/OFF・動きの強さ(倍率)・口パク方式(自然/従来)。詳細はプロファイルから読む |

## SpriTalk特性連動（差別化要素・必須実装）

1. **登場・退場演出**: 表示開始時に全物理へ撃力→揺れて収まる。物理状態は
   表示ごとにリセットし、sin/ノイズ位相をランダム化
2. **息継ぎ（予知駆動）**: モーラタイミングのpause_moraの位置・長さは再生前に
   既知。ポーズ開始で肩＋胸が持ち上がり（吸気）、再開で戻る動きを事前スケジュール
3. **感情切替**: 呼び出し側がemotionIdを変えたら物理は引き継がず、
   小さな切替撃力で不連続を隠す。感情ごとのプロファイル
   （motionScale/bounceScale）を適用
4. **発話中の揺れ**: 従来の一律20%減衰ではなく、開度連動の振幅変調
   （amp×(0.3+0.7×openness)）も選択肢として検証済み

## 実装フェーズ

### Phase A: A1調音結合（依存追加なし・最優先）
- `src/windows/character/components/CharacterApp.tsx` の `moraLipSyncLoop`
  LayeredEmotion分岐（L437-541相当）を差し替え。既存の
  `getVowelAtTime`/`vowelToMouthLayer`（src/shared/utils/vowel-timing.ts）は流用
- 後方互換: プロファイルの lipSyncRenderer が `smoothedFrameStepper` なら従来動作
- 参照: reference/animation-lab/js/20-mouth-methods.js（A1）、11-mora-player.js（A4）

### Phase B: 物理揺れ（描画基盤の判断が必要）
- 描画: pixi.js v8導入（推奨。`await app.init()`必須、MeshPlane、
  `geometry.getBuffer('aPosition')`→`buffer.update()`）または
  MPNGモードのCanvas 2D三角形ワープ流用
- 物理ロジックは reference/animation-lab/js/30-body-methods.js（B3）・
  32-arm-sway.js（腕＋lift）がコンポジター非依存なのでほぼコピー可
- 数値安定化必須: dt≤100msクランプ / dt>1/60でサブステップ / 角度・liftクランプ
- 既存 `ProceduralAnimator`（src/shared/animation/procedural-animator.ts）は
  温存し、プロファイルの layerRenderer が `existingProceduralAnimator` なら従来動作

### Phase C: プロファイル読込＋設定UI縮小
- `spritalk-motion-profile.json`（v1/v2、[スキーマ](motion-lab-integration.md)）を
  素材インポート時に読み込み、キャラクター/感情ごとに保存
- プロファイル無し素材は既定値で動作（ゼロコンフィグ）
- 設定UIは3項目（既存の詳細スライダーは撤去またはプロファイル上書きの上級枠へ）

## 検証方法
- reference/animation-lab（または SpriTalkリポジトリ samples/animation-lab）を
  隣に開き、同一素材・同一テキストで見た目一致を確認
- ハマりどころ: 発話減衰と髪の速度駆動の干渉 / 肩liftは両肩共通1本バネ /
  腕チェーンは左右でノイズ位相をずらす
