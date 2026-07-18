# モーション調整とライブ表示（v0.4.0）

この文書は、PachiPakuGen v0.4.0のSTEP 7「モーション調整」、ライブ表示、保存されるモーション設定の境界を説明します。利用者向けの制作手順は[README](../README.md)、実装の由来とライセンスは[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)を優先してください。

## 1. 位置づけ

STEP 6までに作成した`04_spritalk_parts`を、STEP 7で読み込みます。STEP 7では1つのプレビューを見ながら設定を調整し、次の2つを行えます。

- SpriTalkへ渡す基本画像素材と、PachiPakuGen用モーション設定を同じフォルダへ保存する
- 現在または保存済みの設定を使い、PachiPakuGenのライブ表示へ移動する

ライブ表示はスタートメニューから直接開くこともできます。作業フォルダを選ぶ場合、STEP 6完了後の有効な`04_spritalk_parts`だけを読み込みます。STEP 4またはSTEP 5を再編集して上流の成果物が変わった場合は、古いSTEP 6出力をそのままライブ表示へ使わず、RIFE補完を再実行します。

## 2. 素材構成

標準の入力先は`<作業フォルダ>/04_spritalk_parts`です。

```text
04_spritalk_parts/
├─ body.png                         # 必須
├─ hair.png / hair_back.png         # 任意
├─ eye/                             # まばたき用RIFEフレーム
├─ mouth_a/ ... mouth_o/            # 閉じ口から各母音へのRIFEフレーム
├─ arm_l.png / arm_r.png            # 任意の独立腕
├─ chest.png                        # 任意の胸部変形ガイド
├─ sway_*.png                       # 任意の揺れパーツ。sway_ear*は獣耳として扱う
├─ eyebrow.png                      # 任意の独立眉
├─ eyewhite.png / irides.png        # 任意の視線・虹彩効果用パーツ
├─ highlight.png                    # 任意の目ハイライト
├─ layer-order.json                 # 任意の描画順
├─ motion-preview-manifest.json     # PachiPakuGenの編集状態
└─ spritalk-motion-profile.json     # STEP 7で出力するprofile v2
```

`hair`、腕、胸、揺れパーツ、眉、目の分離パーツは任意です。存在しないパーツを必要とする効果は利用できないか、基本描画へフォールバックします。

PachiPakuGenのRIFE出力では、各`mouth_*`フォルダの先頭フレームを閉じ口として利用できます。そのため専用の`mouth_closed`がなくても正常です。別形式の素材を直接読み込む場合は、互換名の閉じ口素材または母音フレームが必要です。

独立した`eyebrow.png`がある場合、読み込み時に目フレーム内の固定眉が重ならないよう補正してから独立眉を描画します。`eyewhite.png`、`irides.png`、`highlight.png`も現行実装で読み込み・描画されます。

## 3. STEP 7で調整できる内容

### プレビューと口パク

- 再生、停止、先頭から再生
- 内蔵の「あいうえお」確認シーケンス
- ひらがな・カタカナを母音へ簡易変換するテキスト再生
- プレビューの拡大縮小とドラッグ移動
- 口形の切替方式、開閉の反応時間、母音間の滑らかさ

口パク方式は、直接切替、平滑なフレーム進行、閉じ口を経由するブレンドの3系統です。テキスト再生は調整確認用の簡易機能で、漢字や英語の読み上げ・音素解析は行いません。

### 動きとエフェクト

主な調整対象は次のとおりです。必要な分離パーツがない効果は自動的に制限されます。

- 呼吸、体の揺れ、発話バウンス
- 前髪・後ろ髪のバネ／波揺れ、毛先変形、髪房ごとの揺れ
- 首振りパララックス、ランダム首振り
- 自動まばたき、視線の横揺れ、虹彩の伸縮、目のうるみ、眉の微動
- 腕揺れ、肩の弾み、胸部追従
- 汎用`sway_*`パーツと、`sway_ear*`の獣耳ピコピコ

各効果はON/OFF、強さ、単体確認（ソロ）を持ちます。詳細調整では、髪の物理値、パーツごとの回転軸・可動域・揺れ幅、獣耳の動き方、腕の回転軸などを設定できます。

自動まばたきは、視線を中央へ戻してからRIFEの目フレームへ切り替え、開眼後に動的な目の効果を復帰させます。これにより、視線や虹彩効果とRIFEフレームを同時に重ねて生じる不自然さを抑えます。

## 4. 保存ファイル

### `motion-preview-manifest.json`

PachiPakuGenがSTEP 7の編集を再開するための内部ファイルです。現在の設定と、内蔵／テキスト確認シーケンスを保存します。保存済みファイルがある場合、STEP 7とライブ表示はこの設定を復元します。

スキーマは`pachipakugen.motionPreview.v1`です。このファイルをSpriTalk向けの公開互換形式として扱わないでください。

### `spritalk-motion-profile.json`

「SpriTalk向けに保存」で、画像素材と同じ`04_spritalk_parts`直下へ出力します。スキーマは`spritalk.motionProfile.v2`です。

主なフィールド群は次のとおりです。

| フィールド | 内容 |
|---|---|
| `sourcePartsDir` | 素材フォルダ自身を表す相対値`.` |
| `contentFingerprint` | 編集状態との対応を確認する識別値 |
| `blink` | 中央復帰、RIFE閉眼・開眼、動的な目効果の抑制と復帰 |
| `lipSync` | 口パク方式と開閉・ブレンドの時間設定 |
| `layerMotion` | 揺れ方式、テンプレート、呼吸・体揺れ、髪の遅延 |
| `physics` | 髪、腕、胸、揺れパーツ、獣耳、視線、虹彩、眉などの設定 |
| `presence` / `depth` | 表示時の位相・深度に関する既定値 |
| `runtimeRequirements` | 想定する口パク・レイヤー描画方式 |

profile v2は絶対パスを埋め込まず、素材フォルダを移動しても同じフォルダを基準に解決できる形で保存します。

> [!IMPORTANT]
> PachiPakuGen v0.4.0はprofile v2を出力できますが、SpriTalk v1.1.2側にはprofile v2の読み込み機能がありません。SpriTalkへ現時点で渡せるのは`body.png`、髪、目・口のフレーム列などの基本画像素材です。PachiPakuGenのSTEP 7とライブ表示は、同等の設定を`motion-preview-manifest.json`から復元して描画します。

SpriTalk側でprofile v2へ対応するときの実装境界は[SpriTalk側の実装ハンドオフ](handoff-spritalk.md)を参照してください。

## 5. ライブ表示

ライブ表示はSTEP 7と同じ素材ローダー・描画処理・保存済み設定を使用し、マイク入力を口の開閉へ割り当てます。

- 使用するマイクの選択と開始・停止
- 開いたときに使う母音の選択
- 感度、ノイズゲート、開く速さ、閉じる速さ
- 表示倍率、横位置、縦位置、ドラッグ・マウスホイールによる構図調整
- グリーン、ブルー、マゼンタ、ダークのOBS向け背景
- UIを隠すキャプチャ表示。Escまたは右上の操作で設定画面へ戻る

OBSではPachiPakuGenのウィンドウをウィンドウキャプチャし、選択した背景色をカラーキーで除去します。PachiPakuGenは映像を描画しますが、OBSへの音声入力は別途マイクソースとして設定します。

## 6. 互換性の境界

| 対象 | v0.4.0での扱い |
|---|---|
| `04_spritalk_parts`の基本画像 | SpriTalk v1.1.2の`LayeredSpriteImporter`で基本構成を確認済み |
| `motion-preview-manifest.json` | PachiPakuGen専用。SpriTalkへ渡す契約ではない |
| `spritalk-motion-profile.json` v2 | PachiPakuGenが連携用に出力。SpriTalk側の読み込みは未実装 |
| ライブ表示 | PachiPakuGen内で利用。OBSはウィンドウキャプチャ＋カラーキー |

## 7. 関連文書

- [README](../README.md) — 利用者向けの7STEP制作フロー
- [第三者ソフトウェア・モデルの通知](../THIRD_PARTY_NOTICES.md) — モーション実装の由来・変更通知・ライセンス
- [SpriTalk側の実装ハンドオフ](handoff-spritalk.md) — profile v2対応時の実装境界
- [RIFEモデルの由来](rife-model-provenance.md) — 同梱ONNXモデルの変換・検証情報
