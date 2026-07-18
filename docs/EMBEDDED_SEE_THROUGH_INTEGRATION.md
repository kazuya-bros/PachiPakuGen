# Embedded See-Through Integration

> [!NOTE]
> この文書はSee-Through統合を検討した際の履歴資料です。STEP番号、作業フォルダ構成、モデル取得手順の一部は現行v0.4.0と異なります。利用者向けの現行仕様は`README.md`、実装上の条件は`AGENTS.md`を参照してください。

## 1. Product Requirement

新規制作の入口は、開眼・口が確認できる立ち絵1枚だけとする。
ユーザーへSee-ThroughのPSDや、目・口・眉を除去済みの素体を事前準備させない。

PachiPakuGenがSee-Throughを内蔵ワーカーとして実行し、次を半自動で作成する。

- `base.png` 候補
- `eyes-open`
- `eyebrows`
- `front-hair`
- 後続処理用のSee-Through PSD・PNGレイヤー群

自動構成結果は必ず人間が確認する。問題がある箇所だけ既存の素体出力・補正機能で修正し、
承認した `base.png` を後続工程で固定する。

## 2. Why a Managed Worker

See-Throughの公式実装はPython、PyTorch、Diffusers系依存とモデルデータを必要とする。
標準構成は大きなGPUメモリを要求し、公式リポジトリには低VRAM向け量子化構成もある。

そのため、Tauriプロセスへ直接組み込まず、PachiPakuGenがライフサイクルを管理する
ローカルPythonワーカーとして扱う。

```text
React UI
  -> Tauri command: prepare / run / cancel / inspect
    -> Managed See-Through worker
      -> source.png
      -> layers/*.png + result.psd + progress.json
    -> Existing PSD parser and correction UI
      -> automatic mapping
      -> user review and correction
      -> approved base.png
```

## 3. Runtime Profiles

| Profile | Selection | UI |
|---|---|---|
| Auto | GPUメモリを検出し、利用可能な構成を選ぶ | 既定 |
| Standard | 十分なGPUメモリがある場合の通常構成 | 詳細設定 |
| Low VRAM | 量子化版または省VRAM構成 | STEP2で選択可能 |
| External PSD recovery | 内蔵ワーカーが利用できない環境向け | 高度な復旧経路 |

初回実行前に、必要容量、保存先、ライセンス、ダウンロード内容を表示する。
モデルと実行環境はアプリ更新とは分離し、チェックサムとバージョンをmanifestへ保存する。

## 4. Workspace Contract

```text
project/
  source.png
  work/
    see-through/
      result.psd
      layers/
      worker.json
      progress.json
    mapping.json
    base-draft.png
  base.png
  manifest.json
```

`worker.json`にはSee-Throughのバージョン、モデル、プロファイル、実行時間、終了状態を保存する。
`mapping.json`には自動分類と手動変更を分けて保存する。

## 5. Implemented STEP4 Commands

```text
get_see_through_runtime_status
prepare_see_through_runtime
run_see_through
cancel_see_through
```

`run_see_through`は進捗イベントを送信し、成功時に既存のPSD解析処理へ成果物を渡す。
生成PSDは自動的に `load_slot_inner` と `get_mapping_preview_inner` へ渡し、分類プレビューを返す。
UIは分類結果を確認・承認するまで次工程をロックし、必要時は既存の補正画面へ遷移する。

実装は公式リポジトリの検証済みコミット
`e4cb250dc69defe6f982168dab684aa461552b5b` へ固定する。Windowsで `assets` が
シンボリックリンクではなくファイルとして展開された場合は、`common/assets` を管理ランタイムへコピーする。
標準推論は重みを最初から `bfloat16` で読み込み、Windows上での一時的な
`float32` 展開によるメモリ不足・ネイティブアクセス違反を避ける。

管理ランタイムの既定保存先はTauriの `app_local_data_dir()/see-through`。開発・復旧用途では
`PACHIPAKUGEN_SEE_THROUGH_ROOT`、`PACHIPAKUGEN_GIT`、`PACHIPAKUGEN_UV` で上書きできる。

初回セットアップはUIの明示ボタンからのみ開始する。モデル本体は公式実装の初回推論時に
Hugging Faceからダウンロードされ、`HF_HOME` は管理ランタイム配下へ固定される。

## 6. Automatic Mapping

See-Throughのレイヤー名と位置情報から、次を初期分類する。

- `base/body`
- `eyes-open`
- `eyebrows`
- `front-hair`
- `hair-back`
- `mouth-source`
- `unassigned`

自動分類の信頼度が低いレイヤーだけを確認対象として強調する。
全レイヤーを毎回ユーザーへ分類させない。

## 7. References

- See-Through official repository: https://github.com/shitagaki-lab/see-through
- See-Through low-VRAM guidance and quantized scripts:
  https://github.com/shitagaki-lab/see-through#low-vram-users
