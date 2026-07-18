# Codex Generated Parts Workflow

PachiPakuGen の UI を使わず、Codex がコマンド実行と ImageGen を担当するための
フォルダ規約です。

この方式では、アプリ内の OpenAI / Gemini API を直接使いません。Codex の会話内
ImageGen で作った `Codex生成素材` をローカルスクリプトが扱います。

ただし、この文書の `finish` は差分マスク推定による旧式の補助経路です。現在の本線は、
Google AI Studio / Nano Banana / Codex ImageGen などで作った完成差分画像を
PachiPakuGen の内蔵 See-Through に通し、目/口レイヤーだけを抽出して元画像PSDへ戻す
方式です。髪、首、チョーカー、鼻、肌色パッチの巻き込みを避けるため、差分マスク推定は
See-Through分類が使えない場合の診断・暫定出力に限定します。

## Folder Layout

最小構成:

```text
job-folder/
  source.png
```

任意で参照画像を置けます。

```text
job-folder/
  source.png
  reference.png
```

`reference.png` は口内、歯、舌、瞳、まぶたの描き方だけを参照する想定です。

## Prepare

```powershell
python scripts/codex_expression_job.py prepare "C:\path\to\job-folder"
```

作成されるもの:

```text
job-folder/
  _codex/
    source.png
    reference.png             # optional
    codex_request.md
    codex_job.json
    generated_parts/
    masks/
    candidates/
```

Codex は `_codex/codex_request.md` を読み、ImageGen で次の PNG を作って
`_codex/generated_parts/` に保存します。

```text
mouth-closed.png
mouth-a.png
mouth-i.png
mouth-u.png
mouth-e.png
mouth-o.png
eyes-closed.png
```

注意: Codex ImageGen はローカル CLI から直接呼び出せる API ではありません。
このスクリプトは ImageGen 呼び出し自体は行わず、Codex が生成した表情差分素材を
受け取るためのファイル規約を提供します。

## Status

```powershell
python scripts/codex_expression_job.py status "C:\path\to\job-folder"
```

不足している Codex生成素材 PNG を表示します。

## Finish

```powershell
python scripts/codex_expression_job.py finish "C:\path\to\job-folder"
```

出力:

```text
job-folder/
  eyes-open.png
  eyes-closed.png
  mouth-closed.png
  mouth-a.png
  mouth-i.png
  mouth-u.png
  mouth-e.png
  mouth-o.png
  manifest.json
  job-folder.zip
```

特定の差分だけ処理する場合:

```powershell
python scripts/codex_expression_job.py finish "C:\path\to\job-folder" --target mouth-i
```

ZIP を作らない場合:

```powershell
python scripts/codex_expression_job.py finish "C:\path\to\job-folder" --no-zip
```

## Current Limitations

- RIFE 補間はまだこのヘッドレススクリプトには接続していません。
- See-Through レイヤー分類もこのスクリプト内では実行しません。標準パイプラインでは
  PachiPakuGen側の内蔵 See-Through 差分インポートを使います。
- マスクは元画像とCodex生成素材の差分から推定します。これは旧式の補助経路です。必要なら
  `_codex/masks/<target>.png` を手動で置くと、そのマスクを優先します。
- ImageGen の出力を Codex からローカルフォルダへ自動保存できるかは、実行環境の
  画像生成ツールの返却形式に依存します。

## Intended Codex Operation

1. ユーザーが `job-folder/source.png` を置く。
2. Codex が `prepare` を実行する。
3. Codex が `_codex/codex_request.md` に従って ImageGen でCodex生成素材を作る。
4. PNG を `_codex/generated_parts/` に置く。
5. Codex が `finish` を実行する。
6. 必要ならマスクを調整して `finish` を再実行する。

この流れは、アプリ内API生成を避け、画像生成の試行錯誤を Codex 側へ逃がす暫定手段です。
量産品質を狙う場合は、`_codex/generated_parts/` の各PNGをCodex生成素材として扱い、
PachiPakuGen の See-Through 差分インポートへ渡します。

## Naming

ユーザー向けには `ドナー` ではなく `Codex生成素材` または `表情差分素材` と呼びます。
フォルダ名は日本語パス事故を避けるため `generated_parts/` を標準にします。
既存の古いヘルパーや実験出力に `_codex/donors/` が残っている場合は、移行時に
`_codex/generated_parts/` へ読み替えます。
