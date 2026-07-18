# PachiPakuGen同梱 Practical-RIFE v4.9.2 ONNXの出所

PachiPakuGen v0.4.0の`src-tauri/models/rife.onnx`は、Practical-RIFEが公開する公式v4.9.2モデルアーカイブの重みと、固定した公式ソースから再出力しています。この文書は、入力、変更点、再現手順、検証結果を記録するものです。

## 配布ファイル

| File | Purpose | Size | SHA-256 |
|---|---|---:|---|
| `src-tauri/models/rife.onnx` | v0.4.0同梱モデル | 21,457,925 bytes | `0f9f5d969d5221db40a30cc1c4ca9e66d34a408d8bdf146256121ed0304a25a6` |
| `scripts/export_rife_v492_onnx.py` | 再出力用スクリプト | － | `97caf968601bfdbe81614b746b8db26963852596daf730713c532a3fb3e6e8ad` |

ONNXインターフェースは次のとおりです。

- `img0`: float32 `[1, 3, height, width]`
- `img1`: float32 `[1, 3, height, width]`
- `timestep`: float32 `[1]`
- `output`: float32 `[1, 3, height, width]`
- height／widthは動的、opset 17
- ensemble有効、scale listは`[8, 4, 2, 1]`

PachiPakuGenは推論前に画像を64の倍数へパディングし、推論後に元の大きさへ戻します。

## 上流ソースとライセンス

- Project: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- 使用リビジョン: `17d8c7a1005b37f4c97bfee04e316aaec7fdc536`
- 公式v4.9.2モデル: Practical-RIFE README掲載の配布アーカイブ
- License: MIT License, Copyright (c) 2021 hzwer
- ライセンス本文: `licenses/Practical-RIFE-MIT.txt`

元研究と実装は[megvii-research/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)です。配布物にはPractical-RIFEのMIT表示を`THIRD_PARTY_NOTICES.md`とともに収録します。

## 監査済み入力

| Input | Size | SHA-256 |
|---|---:|---|
| 公式v4.9.2 ZIP | 19,816,778 bytes | `f57de4828ae902eec5c1c518bec05edd510f37919b29d5c138cc0d9072b5b63c` |
| `train_log/flownet.pkl` | 21,349,595 bytes | `ef91580a020abb7ddfbd3a51573dc395cf2c2a9530ff653ef3f8a1fc6845857f` |
| `train_log/IFNet_HDv3.py` | 5,697 bytes | `fadb25d8fc3fb6bac52c834356b7b9e27422c9d5ebb060afe4790e2b52cb0f7b` |
| `train_log/RIFE_HDv3.py` | 3,079 bytes | `5041316615eeb28c1101a764896522ba24316b8c8f6cb0d57358254551fd936d` |

再出力スクリプトは、これらのハッシュまたはPractical-RIFEのリビジョンが一致しない場合に処理を中止します。

## ONNX出力時の互換変更

公式`model.warplayer.warp`は、初回に使ったサンプリング格子をPython辞書へキャッシュします。通常の64×64トレースでは、その格子が定数としてONNXへ埋め込まれ、入出力の軸だけを動的にしても256×256以上のDirectML推論が失敗しました。

そこで、ONNX出力時に限って`warp`の格子生成を変更しています。

- 2×2の端点格子を実行時テンソルのheight／widthへbilinear resize
- `align_corners=True`、border padding、flow正規化は上流と同じ
- IFNet、v4.9.2重み、ensemble処理、scale listは変更しない
- `Range`や固定64×64格子をDirectMLの主経路へ置かない

これはPachiPakuGenの3入力・動的解像度ONNXへ変換するための変更であり、上流のPyTorchファイルや重みを配布物内で書き換えるものではありません。

## 再出力

Python 3.10.9、PyTorch 2.10.0、ONNX 1.17.0で監査した出力です。公式v4.9.2 ZIPを用意し、必要なら固定リビジョンのPractical-RIFEチェックアウトを渡します。

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install onnx==1.17.0 onnxruntime==1.20.1 numpy
python .\scripts\export_rife_v492_onnx.py `
  --archive C:\path\to\Practical-RIFE-v4.9.2.zip `
  --repo-dir C:\path\to\Practical-RIFE `
  --output .\src-tauri\models\rife.onnx `
  --verify-ort
```

`--repo-dir`を省略すると、一時ディレクトリへ公式リポジトリをcloneし、監査済みリビジョンをcheckoutします。

## 検証結果

- `onnx.checker.check_model`合格
- ONNX Runtime CPUで64／256／512pxの動的入力を確認
- 同じDirectMLセッションで64／256／512／1280pxを切り替え、有限値の正しい出力形状を確認
- 256pxのプロファイルで動的格子のResize、Add、全GridSampleがDirectML上で実行
- Rustアプリと同じセッション生成・素体合成・差分再抽出経路で64／256／512／1280px、各3フレームを生成
- 64pxでは、最初の公式重み再出力版と全3フレームがピクセル単位で一致（最大差0）

正式配布前には、アプリの実画像を使った64／256／512／1280px推論とSTEP 6の回帰確認を、`RELEASE_CHECKLIST.md`に従って再実施します。

## 旧モデル

v0.3系で同梱していたONNX（SHA-256 `76e4cef9ab42fa7dd4e8f6e4aba47462051e3faa969e4bca6479784fbab0ac6f`）は、変換元と再配布条件をプロジェクト内の記録だけで確定できなかったため、v0.4.0の配布対象から除外しました。
