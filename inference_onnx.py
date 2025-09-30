#!/usr/bin/env python3
"""
YOLOv9 ONNX Inference Script for current repository
Supports inference using exported ONNX model with or without NMS.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Configuration - このリポジトリに合わせて設定
ONNX_PATH = "yolov9_full.onnx"
ONNX_RAW_PATH = "yolov9_full_raw.onnx"  # NMS無しモデル
IMG_SIZE = 640  # 変換時と同じサイズ
CLASS_LIST = ["person", "head"]  # mitococa_v9の2クラス
DEFAULT_IMAGE_PATH = "dataset/mitococa_v9/images/val"  # デフォルト画像パス

# NMS Parameters (for raw mode)
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_MIN_IOU = 0.45
DEFAULT_MAX_BBOX = 300


def apply_nms(
    pred_class,
    pred_bbox,
    pred_conf=None,
    min_confidence=DEFAULT_MIN_CONFIDENCE,
    min_iou=DEFAULT_MIN_IOU,
    max_bbox=DEFAULT_MAX_BBOX,
):
    """
    Apply NMS to raw ONNX model outputs

    Args:
        pred_class: [B, num_anchors, num_classes] - class logits (before sigmoid)
        pred_bbox: [B, num_anchors, 4] - bounding boxes in xyxy format
        pred_conf: [B, num_anchors, 1] - objectness confidence (sigmoid applied) or None
        min_confidence: minimum confidence threshold
        min_iou: minimum IoU threshold for NMS
        max_bbox: maximum number of bboxes to return

    Returns:
        List of [N, 6] arrays (cls, x1, y1, x2, y2, conf) for each batch
    """
    # Apply sigmoid to class logits
    cls_dist = 1 / (1 + np.exp(-pred_class))  # sigmoid

    # Multiply by objectness confidence if available
    if pred_conf is not None:
        cls_dist = cls_dist * pred_conf

    # Find valid detections above confidence threshold
    batch_size = cls_dist.shape[0]
    all_detections = []

    for batch_idx in range(batch_size):
        # Get predictions for this batch
        batch_cls = cls_dist[batch_idx]  # [num_anchors, num_classes]
        batch_bbox = pred_bbox[batch_idx]  # [num_anchors, 4]

        # Find all detections above threshold
        valid_mask = batch_cls > min_confidence
        valid_indices = np.where(valid_mask)

        if len(valid_indices[0]) == 0:
            all_detections.append(np.zeros((0, 6), dtype=np.float32))
            continue

        anchor_idx = valid_indices[0]
        class_idx = valid_indices[1]
        confidences = batch_cls[valid_mask]
        boxes = batch_bbox[anchor_idx]

        # Apply NMS per class
        keep_indices = []
        for cls_id in np.unique(class_idx):
            cls_mask = class_idx == cls_id
            cls_boxes = boxes[cls_mask]
            cls_conf = confidences[cls_mask]
            cls_anchor_idx = np.where(cls_mask)[0]

            # NMS using cv2 (compatible with torchvision.ops.batched_nms logic)
            indices = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(), cls_conf.tolist(), min_confidence, min_iou
            )

            if len(indices) > 0:
                indices = indices.flatten()
                keep_indices.extend(cls_anchor_idx[indices].tolist())

        if len(keep_indices) == 0:
            all_detections.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # Collect results
        keep_indices = np.array(keep_indices)
        final_boxes = boxes[keep_indices]
        final_conf = confidences[keep_indices]
        final_cls = class_idx[keep_indices]

        # Sort by confidence and limit to max_bbox
        sort_idx = np.argsort(-final_conf)[:max_bbox]

        # Format: [cls, x1, y1, x2, y2, conf]
        detections = np.column_stack(
            [final_cls[sort_idx], final_boxes[sort_idx], final_conf[sort_idx]]
        )

        all_detections.append(detections.astype(np.float32))

    return all_detections


def letterbox(pil_img, size=IMG_SIZE, color=(114, 114, 114)):
    """
    Resize the image with aspect ratio preserved and pad to square (same as training).
    Returns:
    - padded PIL.Image (size×size)
    - r float: scaling ratio
    - pad_w int : width padding (left-right)
    - pad_h int : height padding (top-bottom)
    """
    w0, h0 = pil_img.size
    r = min(size / w0, size / h0)
    nw, nh = int(w0 * r), int(h0 * r)
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    pad_w, pad_h = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(resized, (pad_w, pad_h))
    return canvas, r, pad_w, pad_h


def get_image_list(path_str):
    """
    If path_str is a file, return [path_str];
    If it's a folder, return a list of all common image file paths.
    """
    p = Path(path_str)
    if p.is_file():
        return [p]
    elif p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([x for x in p.iterdir() if x.suffix.lower() in exts])
    else:
        raise FileNotFoundError(f"Path not found: {path_str}")


def test_onnx_model(onnx_path, is_raw=False):
    """
    段階的テスト: ONNXモデルの基本動作確認

    Args:
        onnx_path: Path to ONNX model
        is_raw: True if model outputs raw predictions (without NMS)
    """
    print("🧪 ONNX推論テスト開始...")

    try:
        # 1. ONNXファイル存在確認
        print(f"📦 ONNXファイル確認: {onnx_path}")
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNXファイルが見つかりません: {onnx_path}")

        print(
            f"✅ ONNXファイル存在確認 (サイズ: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f}MB)"
        )

        # 2. ONNXRuntimeセッション作成
        print("🔧 ONNXRuntimeセッション作成...")
        providers = (
            ["CUDAExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        print(f"   - 使用プロバイダー: {providers[0]}")

        sess = ort.InferenceSession(onnx_path, providers=providers)

        # 3. モデルの入出力情報確認
        print("📊 モデル情報:")
        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape

        print(f"   - 入力名: {inp_name}")
        print(f"   - 入力形状: {inp_shape}")

        output_names = []
        for i, out in enumerate(sess.get_outputs()):
            print(f"   - 出力{i}: {out.name}, 形状: {out.shape}")
            output_names.append(out.name)

        # 4. ダミー入力でテスト推論
        print("🔄 ダミー入力でテスト推論...")
        dummy_input = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

        start_time = datetime.now()
        outputs = sess.run(None, {inp_name: dummy_input})
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds() * 1000

        print("✅ 推論成功:")
        print(f"   - 推論時間: {inference_time:.1f}ms")

        if is_raw:
            # Raw model: multiple outputs (pred_class, pred_bbox, pred_conf)
            print(f"   - 出力数: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"   - 出力{i}形状: {out.shape}")

            # Test NMS on raw outputs
            if len(outputs) == 3:
                dets = apply_nms(outputs[0], outputs[1], outputs[2])
            else:
                dets = apply_nms(outputs[0], outputs[1])
            print(f"   - NMS後検出数: {dets[0].shape[0]}個")
        else:
            # Full model: single output with NMS already applied
            dets = outputs[0]
            print(f"   - 出力形状: {dets.shape}")
            print(f"   - 検出数: {dets.shape[0]}個")

        return sess, inp_name, is_raw

    except Exception as e:
        print(f"❌ ONNXテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return None, None, False


def inference_single_image(
    sess,
    inp_name,
    img_path,
    is_raw=False,
    min_confidence=DEFAULT_MIN_CONFIDENCE,
    min_iou=DEFAULT_MIN_IOU,
    max_bbox=DEFAULT_MAX_BBOX,
    save_result=True,
):
    """
    単一画像でのONNX推論

    Args:
        sess: ONNX Runtime session
        inp_name: Input tensor name
        img_path: Path to image
        is_raw: True if model outputs raw predictions (without NMS)
        min_confidence: Confidence threshold for NMS (raw mode only)
        min_iou: IoU threshold for NMS (raw mode only)
        max_bbox: Maximum number of detections (raw mode only)
        save_result: Whether to save result image
    """
    try:
        print(f"🖼️ 推論実行: {img_path}")

        # 1. 画像読み込み・前処理
        pil_img = Image.open(img_path).convert("RGB")
        print(f"   - 元画像サイズ: {pil_img.size}")

        img_pad, r, pad_w, pad_h = letterbox(pil_img, IMG_SIZE)

        # 正規化 & 次元変換
        inp = np.array(img_pad, dtype=np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # [1, 3, H, W]

        # 2. ONNX推論
        start_time = datetime.now()
        outputs = sess.run(None, {inp_name: inp})
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds() * 1000

        print(f"   - 推論時間: {inference_time:.1f}ms")

        # 3. Process outputs based on model type
        if is_raw:
            # Apply NMS to raw outputs
            if len(outputs) == 3:
                dets_list = apply_nms(
                    outputs[0],
                    outputs[1],
                    outputs[2],
                    min_confidence,
                    min_iou,
                    max_bbox,
                )
            else:
                dets_list = apply_nms(
                    outputs[0], outputs[1], None, min_confidence, min_iou, max_bbox
                )
            dets = dets_list[0]  # Get first batch
            print(f"   - NMS後検出数: {dets.shape[0]}個")
        else:
            # Full model already has NMS applied
            dets = outputs[0]
            print(f"   - 検出数: {dets.shape[0]}個")

        if dets.shape[0] == 0:
            print("   - 検出結果なし")
            return None

        # 4. 結果の座標変換・描画
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        valid_detections = 0

        for det in dets:
            cls, x1, y1, x2, y2, conf = det

            # 座標を元画像に戻す
            xx1 = (x1 - pad_w) / r
            yy1 = (y1 - pad_h) / r
            xx2 = (x2 - pad_w) / r
            yy2 = (y2 - pad_h) / r

            # 画像範囲内にクリップ
            W0, H0 = pil_img.size
            xx1 = max(min(xx1, W0 - 1), 0)
            yy1 = max(min(yy1, H0 - 1), 0)
            xx2 = max(min(xx2, W0 - 1), 0)
            yy2 = max(min(yy2, H0 - 1), 0)

            # 有効な検出のみ描画
            if xx2 > xx1 and yy2 > yy1:
                xi1, yi1, xi2, yi2 = map(int, (xx1, yy1, xx2, yy2))

                # バウンディングボックス描画
                cv2.rectangle(img_bgr, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)

                # ラベル描画
                cls_name = (
                    CLASS_LIST[int(cls)]
                    if int(cls) < len(CLASS_LIST)
                    else f"Class{int(cls)}"
                )
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    img_bgr,
                    label,
                    (xi1, yi1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                valid_detections += 1

        print(f"   - 有効検出数: {valid_detections}個")

        # 5. 結果保存
        if save_result:
            out_dir = Path("outputs")
            out_dir.mkdir(exist_ok=True)
            suffix = "_raw" if is_raw else ""
            out_path = out_dir / f"onnx_result{suffix}_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(out_path), img_bgr)
            print(f"   - 結果保存: {out_path}")

        return img_bgr

    except Exception as e:
        print(f"❌ 推論エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLOv9 ONNX推論")
    parser.add_argument(
        "--image",
        "-i",
        default=DEFAULT_IMAGE_PATH,
        help="推論対象の画像またはフォルダパス",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["full", "raw"],
        default="full",
        help="モデルタイプ: 'full' (NMS込み) or 'raw' (NMS無し)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="信頼度閾値 (raw mode only)",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_MIN_IOU, help="IoU閾値 (raw mode only)"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_BBOX,
        help="最大検出数 (raw mode only)",
    )
    parser.add_argument(
        "--test-only", action="store_true", help="基本動作テストのみ実行"
    )
    parser.add_argument("--display", action="store_true", help="結果を画面表示")

    args = parser.parse_args()

    # Determine ONNX path and mode
    is_raw = args.model == "raw"
    onnx_path = ONNX_RAW_PATH if is_raw else ONNX_PATH

    print(f"モード: {'RAW (NMS無し)' if is_raw else 'FULL (NMS込み)'}")
    if is_raw:
        print(
            f"NMSパラメータ: confidence={args.confidence}, iou={args.iou}, max_det={args.max_det}"
        )

    # 1. 基本テスト
    sess, inp_name, is_raw_model = test_onnx_model(onnx_path, is_raw)
    if sess is None or inp_name is None:
        print("💥 ONNXモデルテスト失敗")
        return

    if args.test_only:
        print("🎉 基本テスト完了")
        return

    # 2. 実際の画像で推論
    try:
        img_paths = get_image_list(args.image)
        print(f"📸 推論対象: {len(img_paths)}枚の画像")

        for img_path in img_paths[:5]:  # 最初の5枚をテスト
            result_img = inference_single_image(
                sess,
                inp_name,
                img_path,
                is_raw=is_raw_model,
                min_confidence=args.confidence,
                min_iou=args.iou,
                max_bbox=args.max_det,
            )

            if result_img is not None and args.display:
                # 結果表示（サイズを半分に縮小）
                h, w = result_img.shape[:2]
                disp = cv2.resize(
                    result_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR
                )
                cv2.imshow(f"ONNX推論結果: {img_path.name}", disp)
                print("   - 画面表示中（キー入力で次へ）")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("🎉 ONNX推論完了")

    except Exception as e:
        print(f"❌ メイン処理エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
