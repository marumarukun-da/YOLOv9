# YOLO Tools

YOLOv9モデルのエクスポートと推論のためのユーティリティツール集です。

## 目次

- [ONNX Export](#onnx-export)
- [ONNX Inference](#onnx-inference)
- [クイックスタート例](#クイックスタート例)

---

## ONNX Export

YOLOv9モデルをONNX形式にエクスポートします。NMS（Non-Maximum Suppression）の有無を選択できます。

### 使い方

```bash
python -m yolo.tools.onnx_export [OPTIONS]
```

### 必須引数

- `--checkpoint`, `-p`: チェックポイントファイルのパス (.ckpt)
- `--model-name`, `-m`: モデル名 (例: v9-s, v9-c, v9-m, v9-e)
- `--num-classes`, `-n`: データセットのクラス数

### オプション引数

- `--output`, `-o`: 出力ONNXファイルのパス (デフォルト: モードに基づいて自動生成)
- `--mode`: エクスポートモード - `full` (NMS込み) または `raw` (NMS無し) [デフォルト: full]
- `--opset-version`: ONNX opsetバージョン [デフォルト: 17]

### エクスポートモード

#### Full Mode (NMS込み)
NMSを含めてモデルをエクスポートします。出力は最終的な検出結果を含む単一のテンソル `[N, 6]` で、各検出は `[class_id, x1, y1, x2, y2, confidence]` の形式です。

**メリット:** すぐに使用可能、後処理不要
**デメリット:** NMSパラメータが固定、柔軟性が低い

#### Raw Mode (NMS無し)
NMS無しでモデルをエクスポートします。出力は複数のテンソルで構成されます:
- `pred_class`: `[B, num_anchors, num_classes]` - クラスロジット (sigmoid前)
- `pred_bbox`: `[B, num_anchors, 4]` - バウンディングボックス (xyxy形式)
- `pred_conf`: `[B, num_anchors, 1]` - オブジェクトネス信頼度 (オプション)

**メリット:** 柔軟なNMSパラメータ、推論時に調整可能
**デメリット:** 外部NMS実装が必要

### 実行例

#### NMS込みでエクスポート (Full Mode)

```bash
# 基本的なNMS込みエクスポート
python -m yolo.tools.onnx_export \
  --checkpoint runs/train/mitococa_v9_10epoch/best-epoch=09-map=0.3239.ckpt \
  --model-name v9-s \
  --num-classes 2 \
  --mode full

# カスタム出力パスを指定
python -m yolo.tools.onnx_export \
  -p runs/train/exp/best.ckpt \
  -m v9-c \
  -n 80 \
  -o models/yolov9c_coco.onnx \
  --mode full
```

#### NMS無しでエクスポート (Raw Mode)

```bash
# 基本的なNMS無しエクスポート
python -m yolo.tools.onnx_export \
  --checkpoint runs/train/exp/best.ckpt \
  --model-name v9-s \
  --num-classes 2 \
  --mode raw

# カスタムopsetバージョンを指定
python -m yolo.tools.onnx_export \
  -p best.ckpt \
  -m v9-e \
  -n 80 \
  --mode raw \
  --opset-version 17 \
  -o yolov9e_raw.onnx
```

---

## ONNX Inference

エクスポートされたONNXモデル（NMS込み/無し）を使用して画像推論を実行します。

### 使い方

```bash
python -m yolo.tools.onnx_inference [OPTIONS]
```

### 必須引数

- `--onnx`, `-x`: ONNXモデルファイルのパス
- `--image`, `-i`: 画像ファイルまたはディレクトリのパス
- `--classes`: カンマ区切りのクラス名 (例: "person,car,dog")

### オプション引数

- `--img-size`: 入力画像サイズ [デフォルト: 640]
- `--output-dir`, `-d`: 結果の出力ディレクトリ [デフォルト: outputs]
- `--mode`, `-m`: モデルタイプ - `full`, `raw`, または `auto` (自動検出) [デフォルト: auto]
- `--confidence`, `-c`: NMSの信頼度閾値 (rawモードのみ) [デフォルト: 0.25]
- `--iou`: NMSのIoU閾値 (rawモードのみ) [デフォルト: 0.45]
- `--max-det`: 最大検出数 (rawモードのみ) [デフォルト: 300]
- `--test-only`: モデルテストのみ実行 (推論は行わない)
- `--display`: 結果を画面に表示

### モード自動検出

ツールは出力数を調べることで、モデルがfullモードかrawモードかを自動検出できます:
- 1つの出力 → Fullモード (NMS込み)
- 2つ以上の出力 → Rawモード (NMS無し)

自動検出には `--mode auto` (デフォルト) を使用するか、`--mode full` または `--mode raw` を明示的に指定できます。

### 実行例

#### 基本的な推論

```bash
# 単一画像で自動検出
python -m yolo.tools.onnx_inference \
  --onnx model.onnx \
  --image data/images/test.jpg \
  --classes "person,head"

# 画像ディレクトリで推論
python -m yolo.tools.onnx_inference \
  -x model_full.onnx \
  -i data/images/ \
  --classes "person,car,dog,cat" \
  -d results/
```

#### カスタムNMSパラメータでRawモード推論

```bash
# 信頼度とIoU閾値を調整
python -m yolo.tools.onnx_inference \
  --onnx model_raw.onnx \
  --image data/images/ \
  --classes "person,head" \
  --confidence 0.3 \
  --iou 0.5 \
  --max-det 200

# カスタム画像サイズと出力ディレクトリを指定
python -m yolo.tools.onnx_inference \
  -x yolov9_raw.onnx \
  -i test.jpg \
  --classes "cat,dog,bird" \
  --img-size 1280 \
  -c 0.4 \
  -d inference_results/
```

#### モデルテストのみ実行

```bash
# 推論せずにモデルテストのみ
python -m yolo.tools.onnx_inference \
  --onnx model.onnx \
  --classes "person,car" \
  --test-only \
  --image dummy.jpg
```

#### 結果を画面表示

```bash
# 推論結果を画面に表示
python -m yolo.tools.onnx_inference \
  -x model.onnx \
  -i data/images/ \
  --classes "person,head" \
  --display
```

---

## クイックスタート例

### 完全なワークフロー: 学習 → エクスポート → 推論

#### 1. モデルを学習

```bash
python yolo/lazy.py task=train dataset=coco model=v9-s
```

#### 2. ONNX形式にエクスポート (柔軟性のためRawモード)

```bash
python -m yolo.tools.onnx_export \
  --checkpoint runs/train/exp/best.ckpt \
  --model-name v9-s \
  --num-classes 80 \
  --mode raw \
  --output yolov9s_coco_raw.onnx
```

#### 3. カスタムパラメータで推論実行

```bash
python -m yolo.tools.onnx_inference \
  --onnx yolov9s_coco_raw.onnx \
  --image data/coco/images/val2017/ \
  --classes "person,bicycle,car,motorcycle,airplane,bus,train,truck" \
  --confidence 0.35 \
  --iou 0.5 \
  --output-dir inference_results/
```

### 2クラス検出の例 (Person & Head)

#### エクスポート

```bash
python -m yolo.tools.onnx_export \
  -p runs/train/mitococa_v9_10epoch/best-epoch=09-map=0.3239.ckpt \
  -m v9-s \
  -n 2 \
  --mode raw \
  -o mitococa_v9s_raw.onnx
```

#### 推論

```bash
python -m yolo.tools.onnx_inference \
  -x mitococa_v9s_raw.onnx \
  -i dataset/mitococa_v9/images/val \
  --classes "person,head" \
  -c 0.25 \
  --iou 0.45 \
  -d outputs/mitococa_results/
```

### COCO 80クラス検出の例

#### エクスポート (Fullモード)

```bash
python -m yolo.tools.onnx_export \
  -p yolov9c.ckpt \
  -m v9-c \
  -n 80 \
  --mode full \
  -o yolov9c_coco_full.onnx \
  --opset-version 17
```

#### 推論

```bash
python -m yolo.tools.onnx_inference \
  -x yolov9c_coco_full.onnx \
  -i coco_test_images/ \
  --classes "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush" \
  -d coco_inference_results/
```

---

## Tips

### FullモードとRawモードの選び方

**Fullモードを使う場合:**
- シンプルなデプロイメントパイプラインが必要
- NMSパラメータを調整する必要がない
- 後処理済み出力を期待する既存システムとの統合

**Rawモードを使う場合:**
- 異なるシナリオに応じてNMSパラメータを微調整したい
- 後処理の最大限の柔軟性が必要
- 異なる信頼度/IoU閾値で実験している

### パフォーマンス最適化

1. **画像サイズ**: 大きい画像は精度が高いが推論が遅い
   ```bash
   --img-size 640   # 高速、リアルタイム向け
   --img-size 1280  # 低速、高精度
   ```

2. **NMSパラメータ** (Rawモード):
   - 低い信頼度 → より多くの検出、偽陽性が増加
   - 高いIoU閾値 → 重複するボックスが増加
   - ユースケースに応じて調整

3. **ONNX Opsetバージョン**: ランタイムに互換性のある最新バージョンを使用
   ```bash
   --opset-version 17  # ほとんどの場合に推奨
   ```

### トラブルシューティング

**問題**: 次元不一致でモデルテストが失敗する
- **解決策**: `--num-classes` が学習済みモデルと一致していることを確認

**問題**: Rawモードで検出結果が出ない
- **解決策**: `--confidence` 閾値を下げてみる (テスト時は0.1など)

**問題**: 重複する検出が多すぎる
- **解決策**: `--iou` 閾値を上げるか、`--confidence` を下げる

**問題**: ONNXエクスポートが失敗する
- **解決策**: CUDA/PyTorchの互換性を確認、別の `--opset-version` を試す

---

## 追加リソース

- [YOLOv9 論文](https://arxiv.org/abs/2402.13616)
- [ONNX ドキュメント](https://onnx.ai/)
- [ONNX Runtime ドキュメント](https://onnxruntime.ai/)

YOLOv9プロジェクトの詳細については、プロジェクトルートの [README.md](../../README.md) を参照してください。