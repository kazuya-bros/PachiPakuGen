# PachiPakuGen v0.4.1 Release Checklist

## Scope

- [x] STEP 4のレイヤー順をSTEP 5/7へ保持
- [x] 固定・連動オーバーレイの分離描画
- [x] 胸部ワープ範囲の最適化
- [x] アルファ再合成境界の修正

## Version and documentation

- [x] `package.json` / `package-lock.json`
- [x] `src-tauri/Cargo.toml` / `src-tauri/Cargo.lock`
- [x] `src-tauri/tauri.conf.json`
- [x] `CHANGELOG.md`
- [x] `README.md`の現行仕様との不整合確認

## Validation

- [x] `npm run build`
- [x] `npm run test:blink`
- [x] `npm run test:hair`
- [x] `npm run test:chest`
- [x] `npm run test:eyes`
- [x] `npm run test:ears`
- [x] `npm run test:templates`
- [x] `npm run test:workspace`
- [x] `npm run test:loop`
- [x] `cargo fmt --check`
- [x] `cargo test --locked`
- [x] `npm run tauri -- build`
- [x] インストーラー内容にユーザー画像・トークン・ログ・作業フォルダ・大型モデルが混入していない

## GitHub publication

- [ ] v0.4.1のコミットを作成
- [ ] `v0.4.1`タグを作成
- [ ] `main`とタグをpush
- [ ] GitHubのv0.4.1リリースを公開
- [ ] 公開後にこのチェックリストをmainから削除
