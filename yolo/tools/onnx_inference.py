#!/usr/bin/env python3
"""
YOLOv9 ONNX Inference Script

Run inference on images using exported ONNX models (with or without NMS).
Supports flexible configuration via command-line arguments.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Default NMS Parameters
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_MIN_IOU = 0.45
DEFAULT_MAX_BBOX = 300
DEFAULT_IMG_SIZE = 640


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


def letterbox(pil_img, size, color=(114, 114, 114)):
    """
    Resize the image with aspect ratio preserved and pad to square.

    Returns:
        - padded PIL.Image (size×size)
        - r float: scaling ratio
        - pad_w int: width padding (left-right)
        - pad_h int: height padding (top-bottom)
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


def test_onnx_model(onnx_path, img_size, mode=None):
    """
    Test ONNX model and auto-detect mode if not specified

    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
        mode: 'full' or 'raw', auto-detect if None

    Returns:
        (session, input_name, is_raw)
    """
    print("🧪 ONNX推論テスト開始...")

    try:
        # 1. Check ONNX file exists
        print(f"📦 ONNXファイル確認: {onnx_path}")
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNXファイルが見つかりません: {onnx_path}")

        print(
            f"✅ ONNXファイル存在確認 (サイズ: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f}MB)"
        )

        # 2. Create ONNX Runtime session
        print("🔧 ONNXRuntimeセッション作成...")
        providers = (
            ["CUDAExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        print(f"   - 使用プロバイダー: {providers[0]}")

        sess = ort.InferenceSession(onnx_path, providers=providers)

        # 3. Get model input/output info
        print("📊 モデル情報:")
        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape

        print(f"   - 入力名: {inp_name}")
        print(f"   - 入力形状: {inp_shape}")

        num_outputs = len(sess.get_outputs())
        for i, out in enumerate(sess.get_outputs()):
            print(f"   - 出力{i}: {out.name}, 形状: {out.shape}")

        # 4. Auto-detect mode if not specified
        if mode is None:
            is_raw = num_outputs > 1
            mode_detected = "raw" if is_raw else "full"
            print(f"   - モード自動検出: {mode_detected} ({num_outputs}個の出力)")
        else:
            is_raw = mode == "raw"

        # 5. Test inference with dummy input
        print("🔄 ダミー入力でテスト推論...")
        dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)

        start_time = datetime.now()
        outputs = sess.run(None, {inp_name: dummy_input})
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds() * 1000

        print("✅ 推論成功:")
        print(f"   - 推論時間: {inference_time:.1f}ms")

        if is_raw:
            # Raw model: multiple outputs
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
            # Full model: single output
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
    img_size,
    class_names,
    output_dir,
    is_raw=False,
    min_confidence=DEFAULT_MIN_CONFIDENCE,
    min_iou=DEFAULT_MIN_IOU,
    max_bbox=DEFAULT_MAX_BBOX,
    save_result=True,
):
    """
    Run inference on a single image

    Args:
        sess: ONNX Runtime session
        inp_name: Input tensor name
        img_path: Path to image
        img_size: Input image size
        class_names: List of class names
        output_dir: Output directory for results
        is_raw: True if model outputs raw predictions
        min_confidence: Confidence threshold for NMS (raw mode only)
        min_iou: IoU threshold for NMS (raw mode only)
        max_bbox: Maximum number of detections (raw mode only)
        save_result: Whether to save result image
    """
    try:
        print(f"🖼️ 推論実行: {img_path}")

        # 1. Load and preprocess image
        pil_img = Image.open(img_path).convert("RGB")
        print(f"   - 元画像サイズ: {pil_img.size}")

        img_pad, r, pad_w, pad_h = letterbox(pil_img, img_size)

        # Normalize & transpose
        inp = np.array(img_pad, dtype=np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # [1, 3, H, W]

        # 2. ONNX inference
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

        # 4. Draw bounding boxes and labels
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        valid_detections = 0

        for det in dets:
            cls, x1, y1, x2, y2, conf = det

            # Transform coordinates back to original image
            xx1 = (x1 - pad_w) / r
            yy1 = (y1 - pad_h) / r
            xx2 = (x2 - pad_w) / r
            yy2 = (y2 - pad_h) / r

            # Clip to image bounds
            W0, H0 = pil_img.size
            xx1 = max(min(xx1, W0 - 1), 0)
            yy1 = max(min(yy1, H0 - 1), 0)
            xx2 = max(min(xx2, W0 - 1), 0)
            yy2 = max(min(yy2, H0 - 1), 0)

            # Draw only valid detections
            if xx2 > xx1 and yy2 > yy1:
                xi1, yi1, xi2, yi2 = map(int, (xx1, yy1, xx2, yy2))

                # Draw bounding box
                cv2.rectangle(img_bgr, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)

                # Draw label
                cls_idx = int(cls)
                cls_name = (
                    class_names[cls_idx]
                    if cls_idx < len(class_names)
                    else f"Class{cls_idx}"
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

        # 5. Save result
        if save_result:
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_raw" if is_raw else ""
            out_path = output_dir / f"result{suffix}_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(out_path), img_bgr)
            print(f"   - 結果保存: {out_path}")

        return img_bgr

    except Exception as e:
        print(f"❌ 推論エラー: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv9 ONNX Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python -m yolo.tools.onnx_inference --onnx model.onnx --image data/images/test.jpg --classes "person,car,dog"

  # Inference on directory with custom parameters
  python -m yolo.tools.onnx_inference -x model_raw.onnx -i data/images/ --classes "person,head" -c 0.3 --iou 0.5

  # Test only (no actual inference)
  python -m yolo.tools.onnx_inference -x model.onnx --test-only --classes "cat,dog"
""",
    )

    parser.add_argument(
        "--onnx",
        "-x",
        type=str,
        required=True,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to image file or directory",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help='Comma-separated class names (e.g., "person,car,dog")',
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input image size [default: {DEFAULT_IMG_SIZE}]",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="outputs",
        help="Output directory for results [default: outputs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["full", "raw", "auto"],
        default="auto",
        help="Model type: 'full' (with NMS), 'raw' (without NMS), or 'auto' (auto-detect) [default: auto]",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Confidence threshold for NMS (raw mode only) [default: {DEFAULT_MIN_CONFIDENCE}]",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_MIN_IOU,
        help=f"IoU threshold for NMS (raw mode only) [default: {DEFAULT_MIN_IOU}]",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_BBOX,
        help=f"Maximum number of detections (raw mode only) [default: {DEFAULT_MAX_BBOX}]",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run model test only (no actual inference)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display results on screen",
    )

    args = parser.parse_args()

    # Parse class names
    class_names = [name.strip() for name in args.classes.split(",")]

    print(f"{'='*60}")
    print(f"YOLOv9 ONNX Inference")
    print(f"{'='*60}")
    print(f"ONNX Model: {args.onnx}")
    print(f"Image: {args.image}")
    print(f"Classes: {class_names}")
    print(f"Image Size: {args.img_size}")
    print(f"Mode: {args.mode}")
    if args.mode == "raw":
        print(
            f"NMS Params: confidence={args.confidence}, iou={args.iou}, max_det={args.max_det}"
        )
    print(f"Output Dir: {args.output_dir}")
    print(f"{'='*60}\n")

    # 1. Test model
    mode = None if args.mode == "auto" else args.mode
    sess, inp_name, is_raw = test_onnx_model(args.onnx, args.img_size, mode)
    if sess is None or inp_name is None:
        print("💥 ONNXモデルテスト失敗")
        return

    if args.test_only:
        print("🎉 基本テスト完了")
        return

    # 2. Run inference on images
    try:
        img_paths = get_image_list(args.image)
        print(f"\n📸 推論対象: {len(img_paths)}枚の画像")

        output_dir = Path(args.output_dir)

        for img_path in img_paths[:5]:  # Process first 5 images
            result_img = inference_single_image(
                sess,
                inp_name,
                img_path,
                args.img_size,
                class_names,
                output_dir,
                is_raw=is_raw,
                min_confidence=args.confidence,
                min_iou=args.iou,
                max_bbox=args.max_det,
            )

            if result_img is not None and args.display:
                # Display result (resized to half)
                h, w = result_img.shape[:2]
                disp = cv2.resize(
                    result_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR
                )
                cv2.imshow(f"ONNX推論結果: {img_path.name}", disp)
                print("   - 画面表示中（キー入力で次へ）")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("\n🎉 ONNX推論完了")

    except Exception as e:
        print(f"❌ メイン処理エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()