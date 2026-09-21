# PachiPakuGen v0.4.1 Release Checklist

## Scope

- [x] STEP 4へ補正済みPSDの読み込みを追加
- [x] PSD検証後に作業フォルダへ反映し、STEP 4から再編集できるようにする
- [x] 素体・差分位置・RIFE結果を安全に無効化する

## Version and documentation

- [x] `package.json` / `package-lock.json` は v0.4.1
- [x] `src-tauri/Cargo.toml` / `src-tauri/Cargo.lock` は v0.4.1
- [x] `src-tauri/tauri.conf.json` は v0.4.1
- [x] `CHANGELOG.md` を更新

## Validation

- [x] `npm run build`
- [x] `npm run test:workspace`
- [x] `cargo fmt --check`
- [x] `npm run tauri -- build`
- [x] インストーラー内容を確認（ユーザー画像・トークン・ログ・作業フォルダ・大型モデルを含まない）

## GitHub publication

- [ ] v0.4.1の変更コミットを作成
- [ ] `main`と`v0.4.1`タグをpush
- [ ] GitHubのv0.4.1リリースを更新
- [ ] 公開後にこのチェックリストをmainから削除
