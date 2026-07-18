# Changelog

このプロジェクトの主要な変更を記録します。形式は[Keep a Changelog](https://keepachangelog.com/ja/1.1.0/)を参考にし、バージョン番号は[Semantic Versioning](https://semver.org/lang/ja/)に従います。

## [0.4.0] - Unreleased

### Added

- 「はじめから」「つづきから」「ライブ表示」のスタートメニュー
- `project.json`を使った作業フォルダと工程の再開
- Codex、ほかの画像編集AI、手作業から選べる表情素材作成ガイド
- See-Through公式実装の自動セットアップ、固定リビジョン管理、モデル事前取得
- `standard`と`low-vram`のSee-Through実行プロファイル
- 素体レイヤー、腕、獣耳、目・口差分を補正する編集画面
- RIFEによる表情差分の一括フレーム補完
- SpriTalk向け`04_spritalk_parts`フォルダ出力
- `spritalk-motion-profile.json` v2とモーション設定の再開
- ウェーブ式・スプリング式のモーション調整と4種類のテンプレート
- 瞬き、口パク、視線、瞳孔、目の潤み、眉、髪、腕、胸、獣耳のモーション調整
- マイク連動ライブ表示、OBS向けクロマキー背景、拡大縮小・移動・キャプチャ表示
- Hugging FaceトークンのWindows資格情報への保存

### Changed

- v0.3系の個別モード中心UIを、7STEPの制作フローへ全面刷新
- See-Throughを外部手順ではなく、アプリ管理の固定版ランタイムとして統合
- 表情素材の抽出をSee-Throughレイヤーと画像差分中心の処理へ変更
- RIFE出力をモーション調整とSpriTalk受け渡しまで含む制作工程の一部へ変更
- 同梱RIFEモデルを、Practical-RIFE公式v4.9.2重みから再現可能な手順で出力した、動的解像度・DirectML対応の監査済みONNXへ置換
- STEP 4〜7のプレビュー、操作位置、固定下部操作を統一
- モーション調整を専用の別アプリではなく、中央編集領域とライブ表示へ統合
- 実行時に適用するSee-Through互換パッチへ、変更元リビジョンと変更目的の通知を埋め込むよう変更

### Fixed

- STEP 1〜6の入力・生成物を再編集したとき、保存済みの古いRIFE出力やライブ素材が再び有効になる問題を修正
- 立ち絵または参照画像を差し替えたとき、表情素材の作成ガイドが更新済みと誤判定される問題を修正
- 各工程のファイル更新前に進捗を安全側へ戻し、途中終了後に部分出力を完成品として扱わないよう修正
- 7枚の表情素材を画像内容の指紋で検査し、手動差し替え後に古い抽出・RIFE結果を再利用しないよう修正
- 画像更新処理が途中で失敗した場合、ディスク上の進捗を再読込して画面の古い完了状態を即座に無効化

### Removed

- SAM3の実行経路、`sam3.pt`、専用Python環境、抽出スクリプト
- 現行7STEPから到達できないv0.3系の旧画面、補間IPC、専用スタイル

### Compatibility

- v0.3系のPSD・PNGはv0.4.0の作業フォルダへ自動移行しません。
- `04_spritalk_parts`の基本画像はSpriTalkのLayeredEmotion用素材として渡せる構成です。実際に生成した素材をSpriTalk v1.1.2の`LayeredSpriteImporter`へ読み込み、基本構成の検証に成功しています。
- 高度なモーションプロファイルv2はSpriTalk側の将来連携用仕様です。v0.4.0時点で確実に再生できるのはPachiPakuGenのライブ表示です。

### Security and privacy

- Hugging Faceトークンを平文ファイルではなくWindows資格情報へ保存するようにしました。
- モデル取得と推論を分離し、推論中の暗黙ダウンロードを無効化しました。

## [0.3.0] - 2026-05-21

- See-Through PSDからの素体・差分作成とRIFEフレーム補完を中心とした制作フロー。

## [0.2.0] - 2026-04-07

- 初期公開版。

[0.4.0]: https://github.com/kazuya-bros/PachiPakuGen/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kazuya-bros/PachiPakuGen/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kazuya-bros/PachiPakuGen/releases/tag/v0.2.0
