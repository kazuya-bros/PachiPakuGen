# Third-Party Notices

PachiPakuGen本体はMIT Licenseで提供されます。PachiPakuGenに含まれる、参照された、または実行時に取得される第三者ソフトウェア・モデルには、それぞれの権利者が定めた条件が適用されます。

本書はv0.4.0の主要な第三者要素と、PachiPakuGen側で行った変更を記録するものです。名称の記載は出所の明示を目的とし、各権利者がPachiPakuGenを推奨、保証、提携していることを意味しません。

## 1. See-Through

- Project: [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through)
- License: Apache License 2.0
- Upstream revision used by PachiPakuGen: [`e4cb250dc69defe6f982168dab684aa461552b5b`](https://github.com/shitagaki-lab/see-through/commit/e4cb250dc69defe6f982168dab684aa461552b5b)
- Distribution: See-Throughのソースコードと大型モデルはPachiPakuGenのインストーラーに含まれません。ユーザーがSTEP 3の準備を開始したとき、公式リポジトリと各モデル配布元からアプリ管理領域へ取得します。
- Full license text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)

### PachiPakuGenによる変更

PachiPakuGenは取得した固定リビジョンに対し、実行時に次の互換変更を適用します。

- Windows環境でのBF16重み読み込み互換
- standardプロファイルの推論ステージ間におけるVRAM解放
- low-vramプロファイルにおけるLayerDiffおよびMarigoldのCPU offload
- Accelerateの実行デバイスへテンソルを配置するための修正
- bitsandbytes 4-bit重みとgroup offloadの競合回避

変更対象となる上流ファイル:

```text
inference/scripts/inference_psd.py
inference/scripts/inference_psd_quantized.py
common/utils/inference_utils.py
common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py
common/modules/marigold/marigold_depth_pipeline.py
```

これらの変更はPachiPakuGen側の `src-tauri/src/commands/see_through.rs` から適用され、上流プロジェクトへ加えられた公式変更ではありません。

### See-Throughが取得するモデル

PachiPakuGen v0.4.0は、再現可能な取得のためリポジトリID、リビジョン、必要ファイルとサイズを `src-tauri/scripts/see_through_model_requirements.json` に固定しています。

| プロファイル | 配布元 | 用途・条件の確認状況 |
|---|---|---|
| standard | [`layerdifforg/seethroughv0.0.2_layerdiff3d`](https://huggingface.co/layerdifforg/seethroughv0.0.2_layerdiff3d) | モデルページにApache-2.0表示あり |
| standard | [`24yearsold/seethroughv0.0.1_marigold`](https://huggingface.co/24yearsold/seethroughv0.0.1_marigold) | See-Through開発者が同プロジェクトの全モデルをApache-2.0へ揃える旨を明示 |
| low-vram | [`24yearsold/seethroughv0.0.2_layerdiff3d_nf4`](https://huggingface.co/24yearsold/seethroughv0.0.2_layerdiff3d_nf4) | 同上 |
| low-vram | [`24yearsold/seethroughv0.0.1_marigold_nf4`](https://huggingface.co/24yearsold/seethroughv0.0.1_marigold_nf4) | 同上 |
| low-vram scheduler | [`layerdifforg/seethroughv0.0.2_layerdiff3d`](https://huggingface.co/layerdifforg/seethroughv0.0.2_layerdiff3d) | Apache-2.0表示のあるstandardモデルからscheduler設定だけを取得。NF4版と同一内容 |

See-Through開発者によるモデルライセンスの回答:

- [Open-source license for seethroughv0.0.2_layerdiff3d?](https://huggingface.co/layerdifforg/seethroughv0.0.2_layerdiff3d/discussions/1)

モデル配布元が将来条件を変更する場合があります。取得時点の配布ページも併せて確認してください。

## 2. PuruPuruPNGTuber

- Project: [rotejin/PuruPuruPNGTuber](https://github.com/rotejin/PuruPuruPNGTuber)
- Revision reviewed and adapted: [`9dc1e735155faae8f54f9ee3076b52db7da36624`](https://github.com/rotejin/PuruPuruPNGTuber/tree/9dc1e735155faae8f54f9ee3076b52db7da36624)
- Copyright notice in upstream license: Copyright 2026 masa
- Software license: Apache License 2.0
- Full license text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)

PachiPakuGenの `src/motionLab/render.ts` にあるウェーブ式髪ワープは、上流 `app.js` の `pyokopyokoHairShift` が持つ周期・位相設計と根元固定マスクを、Canvasの横ストリップ描画へ変更・適応したものです。振幅、描画方式、発話連動、前後髪の扱いをPachiPakuGen向けに変更しています。

前髪・後ろ髪の遅延追従、目元演出、自動瞬きについても設計を参考にしましたが、PachiPakuGen側では状態モデル、数値安定化、UI、SpriTalk用パーツ構成に合わせて独自に実装しています。変更されたPachiPakuGen側のソースには、上流・ライセンス・変更済みである旨をコメントで記録しています。

PuruPuruPNGTuberのデモアバター、画像、音声、その他のサンプル素材はPachiPakuGenへ収録していません。上流のビジュアル素材にはソフトウェアとは別の[Asset License](https://github.com/rotejin/PuruPuruPNGTuber/blob/9dc1e735155faae8f54f9ee3076b52db7da36624/ASSET_LICENSE.md)が適用されます。

## 3. Anime2.5DRig

- Project: [852wa/Anime2.5DRig](https://github.com/852wa/Anime2.5DRig)
- Revision reviewed and adapted: [`d48825867acd081de22b0e7b5585bb562288796d`](https://github.com/852wa/Anime2.5DRig/tree/d48825867acd081de22b0e7b5585bb562288796d)
- Copyright notice: Copyright (c) 2026 hakoniwa
- License: MIT License
- Full license text: [licenses/Anime2.5DRig-MIT.txt](licenses/Anime2.5DRig-MIT.txt)

PachiPakuGenの `src/motionLabPhysics.ts` にある髪房検出と房単位の二重バネは、上流 `lib/rigger.js` および `index.html` の実装をTypeScriptへ変更・適応したものです。透過PNG入力、キャッシュ、変位制限、ソフトブレンド、Canvas描画へ合わせて変更しています。`src/motionLab/render.ts` のランダムな顔向きの更新間隔も上流設計を参考にしています。

レイヤー深度によるパララックスは設計上の着想を得ていますが、上流のWebGLメッシュ変形とは異なり、PachiPakuGenではPNGレイヤー単位の移動・回転・シアーとして独自に実装しています。変更されたPachiPakuGen側のソースには出所をコメントで記録しています。Anime2.5DRigのWebGLリグやサンプル素材は同梱していません。

852話氏およびhakoniwaの名称は上流プロジェクトとライセンスに基づく帰属表示です。PachiPakuGenの共同著作権者または推奨者であることを示すものではありません。

## 4. RIFE

- Project and model: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- Upstream revision used for export: [`17d8c7a1005b37f4c97bfee04e316aaec7fdc536`](https://github.com/hzwer/Practical-RIFE/tree/17d8c7a1005b37f4c97bfee04e316aaec7fdc536)
- Original research/source implementation: [hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)
- License: MIT License, Copyright (c) 2021 hzwer
- Full license text: [licenses/Practical-RIFE-MIT.txt](licenses/Practical-RIFE-MIT.txt)
- Bundled file: `src-tauri/models/rife.onnx`
- Bundled SHA-256: `0F9F5D969D5221DB40A30CC1C4CA9E66D34A408D8BDF146256121ED0304A25A6`
- Reproducible exporter: [scripts/export_rife_v492_onnx.py](scripts/export_rife_v492_onnx.py)
- Provenance and audited input hashes: [docs/rife-model-provenance.md](docs/rife-model-provenance.md)

同梱ONNXは、Practical-RIFEが公開する公式v4.9.2モデルアーカイブの重みと公式実装から、PachiPakuGenの3入力インターフェース（`img0`、`img1`、`timestep`）へ再出力したものです。入力アーカイブ、重み、上流ソースのリビジョンを固定し、再出力スクリプトがハッシュ不一致を拒否します。

上流のサンプリング格子キャッシュをそのままONNX化すると64×64の定数が埋め込まれるため、ONNX出力時の`warp`だけを、実行時サイズから同等のalign-corners格子を作るDirectML互換表現へ変更しています。公式IFNet、v4.9.2重み、ensemble処理、scale listは変更していません。変更内容と検証結果は出所文書に記録しています。

以前同梱していた出所未確定のONNX（SHA-256 `76E4CEF9AB42FA7DD4E8F6E4ABA47462051E3FAA969E4BCA6479784FBAB0AC6F`）はv0.4.0の配布対象から削除しました。

## 5. JavaScript・Rust・Python依存関係

PachiPakuGenはReact、Tauri、ONNX Runtime、image、psd、Diffusers、PyTorchなど多数のオープンソース依存関係を利用します。各パッケージにはそれぞれのライセンスが適用されます。

v0.4.0の配布物には、次の依存関係ライセンス一覧を同梱します。

- [JavaScript本番依存](licenses/NPM_DEPENDENCIES.html) — `package-lock.json` と `npm ls --omit=dev` から生成
- [Rust Windows本番依存](licenses/CARGO_DEPENDENCIES.html) — `src-tauri/Cargo.lock` の `x86_64-pc-windows-msvc` 依存グラフから `cargo-about` で生成
- [See-Through Python依存](licenses/SEE_THROUGH_PYTHON_DEPENDENCIES.html) — 固定See-Throughリビジョンから構築して動作確認したアプリ管理環境のスナップショット

JavaScript一覧はインストーラーへ組み込まれる本番依存だけを対象とし、開発専用ツールを除外しています。Rust一覧はWindows向けリリース依存を対象とします。See-ThroughのPythonパッケージはインストーラーに含まれず、ユーザーがSTEP 3の準備を明示的に開始した後、アプリ管理領域へ取得されます。推移的依存の解決結果は配布元の更新で変わり得るため、実際の環境に同梱された各ライセンスファイルが最終的な条件です。

一覧の再生成手順は `npm run licenses:npm`、`npm run licenses:cargo`、`scripts/generate_python_licenses.py` に保存しています。

## 6. ユーザーが用意する素材

PachiPakuGenはユーザーが選択した立ち絵、参照画像、AI生成・編集画像を処理します。ユーザーは、入力素材、生成物、配信、録画、SpriTalkへの取り込みについて必要な権利・許諾を確認してください。本プロジェクトのMIT Licenseは、ユーザーが入力した画像や第三者サービスの生成物に権利を付与するものではありません。
