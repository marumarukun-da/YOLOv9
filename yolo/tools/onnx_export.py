#!/usr/bin/env python3
"""
YOLOv9 ONNX Export Script

Export YOLOv9 models to ONNX format with or without NMS.
Supports flexible configuration via command-line arguments.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize

from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess


class FullYoloExport(nn.Module):
    """YOLO Export with NMS included"""

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model.eval()
        self.converter = create_converter(
            cfg.model.name,
            model,
            cfg.model.anchor,
            cfg.task.data.image_size,
            device="cpu",
        )
        self.nms_cfg = cfg.task.nms

    def forward(self, x):
        preds = self.model(x)
        heads = preds["Main"] if isinstance(preds, dict) else preds
        dets = PostProcess(self.converter, self.nms_cfg)({"Main": heads}, None)
        return torch.cat(dets, dim=0)  # Output shape: [N, 6]


class RawYoloExport(nn.Module):
    """
    YOLO Export without NMS (for flexible post-processing)
    Returns raw predictions: class logits, bboxes, and confidence scores
    """

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model.eval()
        self.converter = create_converter(
            cfg.model.name,
            model,
            cfg.model.anchor,
            cfg.task.data.image_size,
            device="cpu",
        )

    def forward(self, x):
        preds = self.model(x)
        heads = preds["Main"] if isinstance(preds, dict) else preds

        # Convert model output to bbox format (without NMS)
        prediction = self.converter(heads)
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None

        # Output raw predictions for external NMS processing
        # pred_class: [B, num_anchors, num_classes] - class logits (before sigmoid)
        # pred_bbox: [B, num_anchors, 4] - bounding boxes in xyxy format
        # pred_conf: [B, num_anchors, 1] - objectness confidence (sigmoid applied) or None

        if pred_conf is not None:
            return pred_class, pred_bbox, pred_conf
        else:
            return pred_class, pred_bbox


def export_full_onnx(ckpt_path, model_name, num_classes, output_file, opset_version=17):
    """Export ONNX model with NMS included"""
    print("🚀 ONNX変換を開始 (NMS込み)...")

    try:
        # 1. 設定読み込み
        print("📋 設定を読み込み中...")
        with initialize(config_path="yolo/config", version_base=None):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "task=inference",
                    f"model={model_name}",
                    f"dataset.class_num={num_classes}",
                ],
            )
        print(f"✅ 設定読み込み完了 (モデル: {model_name}, クラス数: {num_classes})")
        print(f"   - 画像サイズ: {cfg.task.data.image_size}")
        print(
            f"   - NMS設定: min_confidence={cfg.task.nms.min_confidence}, min_iou={cfg.task.nms.min_iou}"
        )

        # 2. モデル作成
        print("🏗️ モデルを作成中...")
        model = create_model(cfg.model, class_num=num_classes)
        print("✅ モデル作成完了")

        # 3. チェックポイント読み込み
        print(f"📦 チェックポイントを読み込み中: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Lightning形式のキー名変換 (model.model.* → model.*)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # "model."を削除
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # モデルに重みを読み込み
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=False
        )
        if missing_keys:
            print(f"⚠️ 不足キー: {len(missing_keys)}個")
        if unexpected_keys:
            print(f"⚠️ 余分キー: {len(unexpected_keys)}個")
        print("✅ チェックポイント読み込み完了")

        # 4. エクスポート用ラッパー作成
        print("🔧 エクスポート用ラッパーを作成中...")
        wrapper = FullYoloExport(model, cfg).eval()

        # 5. ダミー入力作成
        image_size = cfg.task.data.image_size
        dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
        print(f"✅ ダミー入力作成完了 (サイズ: {dummy_input.shape})")

        # 6. ONNX Export
        print(f"🔄 ONNX変換中: {output_file}")
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_file,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
            },
            verbose=False,
        )

        print(f"🎉 ONNX変換完了: {output_file}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def export_raw_onnx(ckpt_path, model_name, num_classes, output_file, opset_version=17):
    """Export ONNX model without NMS for flexible post-processing"""
    print("🚀 ONNX変換を開始 (NMS無し)...")

    try:
        # 1. 設定読み込み
        print("📋 設定を読み込み中...")
        with initialize(config_path="yolo/config", version_base=None):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "task=inference",
                    f"model={model_name}",
                    f"dataset.class_num={num_classes}",
                ],
            )
        print(f"✅ 設定読み込み完了 (モデル: {model_name}, クラス数: {num_classes})")
        print(f"   - 画像サイズ: {cfg.task.data.image_size}")

        # 2. モデル作成
        print("🏗️ モデルを作成中...")
        model = create_model(cfg.model, class_num=num_classes)
        print("✅ モデル作成完了")

        # 3. チェックポイント読み込み
        print(f"📦 チェックポイントを読み込み中: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Lightning形式のキー名変換 (model.model.* → model.*)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # "model."を削除
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # モデルに重みを読み込み
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=False
        )
        if missing_keys:
            print(f"⚠️ 不足キー: {len(missing_keys)}個")
        if unexpected_keys:
            print(f"⚠️ 余分キー: {len(unexpected_keys)}個")
        print("✅ チェックポイント読み込み完了")

        # 4. エクスポート用ラッパー作成 (NMS無し)
        print("🔧 エクスポート用ラッパーを作成中 (NMS無し)...")
        wrapper = RawYoloExport(model, cfg).eval()

        # 5. ダミー入力作成
        image_size = cfg.task.data.image_size
        dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
        print(f"✅ ダミー入力作成完了 (サイズ: {dummy_input.shape})")

        # 6. 出力テスト
        print("🧪 出力形状テスト中...")
        with torch.no_grad():
            outputs = wrapper(dummy_input)
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    pred_class, pred_bbox, pred_conf = outputs
                    print(f"   - pred_class: {pred_class.shape} (クラススコア)")
                    print(f"   - pred_bbox: {pred_bbox.shape} (バウンディングボックス)")
                    print(f"   - pred_conf: {pred_conf.shape} (信頼度)")
                    output_names = ["pred_class", "pred_bbox", "pred_conf"]
                else:
                    pred_class, pred_bbox = outputs
                    print(f"   - pred_class: {pred_class.shape} (クラススコア)")
                    print(f"   - pred_bbox: {pred_bbox.shape} (バウンディングボックス)")
                    output_names = ["pred_class", "pred_bbox"]

        # 7. ONNX Export
        print(f"🔄 ONNX変換中: {output_file}")
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_file,
            opset_version=opset_version,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes={
                "input": {0: "batch"},
                output_names[0]: {0: "batch"},
                output_names[1]: {0: "batch"},
            },
            verbose=False,
        )

        print(f"🎉 ONNX変換完了: {output_file}")
        print("   - NMS処理は推論時に外部で実行してください")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv9 ONNX Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with NMS (full mode)
  python -m yolo.tools.onnx_export --checkpoint runs/train/exp/best.ckpt --model-name v9-s --num-classes 2

  # Export without NMS (raw mode)
  python -m yolo.tools.onnx_export -p runs/train/exp/best.ckpt -m v9-s -n 2 --mode raw

  # Custom output and opset version
  python -m yolo.tools.onnx_export -p best.ckpt -m v9-c -n 80 -o model.onnx --opset-version 17
""",
    )

    parser.add_argument(
        "--checkpoint",
        "-p",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        required=True,
        help="Model name (e.g., v9-s, v9-c, v9-m)",
    )
    parser.add_argument(
        "--num-classes",
        "-n",
        type=int,
        required=True,
        help="Number of classes",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output ONNX file path (default: auto-generated based on mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "raw"],
        default="full",
        help="Export mode: 'full' (with NMS) or 'raw' (without NMS) [default: full]",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version [default: 17]",
    )

    args = parser.parse_args()

    # Generate output filename if not specified
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        suffix = "_raw" if args.mode == "raw" else "_full"
        args.output = f"{checkpoint_name}{suffix}.onnx"

    print(f"{'='*60}")
    print(f"YOLOv9 ONNX Export")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Classes: {args.num_classes}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"ONNX Opset: {args.opset_version}")
    print(f"{'='*60}\n")

    # Export based on mode
    if args.mode == "full":
        export_full_onnx(
            args.checkpoint,
            args.model_name,
            args.num_classes,
            args.output,
            args.opset_version,
        )
    else:
        export_raw_onnx(
            args.checkpoint,
            args.model_name,
            args.num_classes,
            args.output,
            args.opset_version,
        )


if __name__ == "__main__":
    main()