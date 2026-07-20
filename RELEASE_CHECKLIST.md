# v0.4.0 Release Checklist

v0.4.0を「正式リリース済み」と扱う前に、次の項目をすべて完了します。

## Release blockers

- [x] 現在の`rife.onnx`を、再配布ライセンスと変換元が明示されたモデルへ置換する
- [x] ライセンス表記のないJuggernautXL由来schedulerを、Apache-2.0明示のSee-Throughモデル内の同一設定へ置換
- [x] 現行7STEPで到達不能なSAM3旧経路、`pyproject.toml`、`uv.lock`、`scripts/extract_neck_mask.py`、語彙ファイルを削除し、ビルドから除外する
- [x] npm、Cargo、See-Through Python環境の第三者ライセンス一覧を生成し、配布リソースへ追加する

## Documentation and metadata

- [x] READMEを現行7STEP、ライブ表示、SpriTalk受け渡しに合わせて全面改稿
- [x] `THIRD_PARTY_NOTICES.md`を追加し、See-Throughの互換変更と着想元を分離して記載
- [x] `CHANGELOG.md`へv0.4.0の変更を追加
- [x] Apache-2.0とMITの必要なライセンス本文を配布物へ同梱
- [x] `package.json`、`package-lock.json`、`Cargo.toml`、`Cargo.lock`、`tauri.conf.json`を0.4.0へ更新
- [x] `Cargo.toml`へlicense・repository・README、`package.json`へlicense・repository・READMEへの導線を追加
- [x] 古いSAM3・個別モード説明が残る`CLAUDE.md`、`AGENTS.md`、設計文書を更新または履歴資料として明示

## Repository hygiene

- [x] ビルドに必要な未追跡ファイルを確認して追加
- [x] ログ、破損バックアップ、検証用一時ファイル、別リポジトリをリリースコミットから除外
- [x] `.gitignore`へログ、バックアップ、`tmp/`などの生成物を追加
- [x] `build_dev.ps1`と`check_dev.ps1`の古い固定パスを修正
- [x] クリーンなチェックアウトで同じ成果物を作れることを確認

## Automated verification

- [x] `npm ci`
- [x] `npm audit --omit=dev`（脆弱性0件）
- [x] `npm audit`（開発依存を含め脆弱性0件）
- [x] `npm run build`
- [x] `npm run test:blink`
- [x] `npm run test:hair`
- [x] `npm run test:chest`
- [x] `npm run test:eyes`
- [x] `npm run test:ears`
- [x] `npm run test:templates`
- [x] `npm run test:workspace`
- [x] `cargo fmt --check`
- [x] `cargo test --locked`（128件すべて成功）
- [x] `cargo audit`（脆弱性0件。情報警告は対象OS・到達条件を確認）
- [x] 監査済みRIFEモデルを64／256／512／1280pxのDirectML実画像で推論
- [x] `npm run tauri -- build`
- [x] NSISから展開した正式バイナリを起動し、WebView2初期化後8秒間の生存を確認

## Manual smoke tests

- [x] 新規Windows環境でNSISインストール、起動、アンインストール
- [x] `low-vram`のSee-Through初回セットアップ、モデル取得、分解
- [x] `standard`のSee-Through初回セットアップ、モデル取得、分解
- [x] STEP 1〜7の新規制作と「つづきから」の再開
- [x] STEP 4・5の再編集後にRIFEを再実行
- [x] STEP 7の設定保存と再読込
- [x] ライブ表示のマイク入力、拡大縮小、ドラッグ、ESC復帰
- [x] OBSウィンドウキャプチャと各クロマキー背景
- [x] 既存の`04_spritalk_parts`実データをSpriTalk v1.1.2（`12c519f`）の`LayeredSpriteImporter`へ通し、`LayeredEmotion`検証成功
- [x] `spritalk-motion-profile.json` v2がSpriTalk側では未対応であることをUI・READMEと照合
- [x] インストーラーへ`sam3.pt`、ユーザー画像、トークン、ログ、一時ファイルが混入していないことを確認

## Release publication

- [x] READMEから案内するSpriTalk BOOTH商品ページを匿名アクセスで確認
- [x] `CHANGELOG.md`の`Unreleased`をリリース日に置換し、比較先を`HEAD`から`v0.4.0`へ固定
- [x] NSISインストーラーのSHA-256を計算
  - 最終値はインストーラーと同じ配布ディレクトリの`SHA256SUMS.txt`へ記録する
- [ ] GitHub Releaseへ変更点、要件、既知の制約、SHA-256を掲載
- [x] クリーンなリリースコミットから`v0.4.0`タグを作成
- [ ] 公開後にインストーラーを再ダウンロードし、ハッシュと起動を確認
