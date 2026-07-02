#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化模型推理脚本：使用 ONNX Runtime 加载量化 ONNX 模型，对图像进行目标检测
使用方式（由 Qt GUI 调用）：
    python infer_quant.py <quant_model_onnx> <image_path> <output_json> [conf_thresh] [iou_thresh]
参数：
    quant_model_onnx : 量化后的 .onnx 模型路径
    image_path       : 输入图片路径
    output_json      : 检测结果 JSON 文件路径（Qt 读取）
    conf_thresh      : 置信度阈值（默认 0.25）
    iou_thresh       : NMS IoU 阈值（默认 0.45）
输出 JSON 格式：
    [{"label":"puddle","conf":0.88,"box":[x,y,w,h]}, ...]  (像素坐标)
"""

import sys
import os
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

def nms(boxes, scores, iou_thresh):
    """简单的 NMS 实现"""
    import numpy as np
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep

def main():
    if len(sys.argv) < 4:
        log("[ERROR] 用法: python infer_quant.py <onnx_model> <image_path> <output_json> [conf] [iou]")
        sys.exit(1)

    onnx_model  = sys.argv[1]
    image_path  = sys.argv[2]
    output_json = sys.argv[3]
    conf_thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    iou_thresh  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45

    # ── 检查文件 ──────────────────────────────────────────
    if not os.path.isfile(onnx_model):
        log(f"[ERROR] 量化模型不存在: {onnx_model}")
        sys.exit(1)
    if not os.path.isfile(image_path):
        log(f"[ERROR] 图片不存在: {image_path}")
        sys.exit(1)

    log("=" * 50)
    log("[推理] 使用量化 ONNX 模型进行目标检测")
    log(f"  模型: {os.path.basename(onnx_model)}")
    log(f"  图片: {os.path.basename(image_path)}")
    log(f"  置信度阈值: {conf_thresh}  IoU 阈值: {iou_thresh}")

    try:
        import onnxruntime as ort
        import numpy as np
        import cv2

        # ── 加载模型 ──────────────────────────────────────
        log("[推理] 加载量化模型...")
        
        # 注册 onnxruntime-extensions 以支持 ConvInteger 等算子（INT8 量化模型必需）
        try:
            import onnxruntime_extensions as ort_ext
            log("  [INFO] 已加载 onnxruntime-extensions")
        except ImportError as _ext_err:
            log(f"  [WARN] 未安装 onnxruntime-extensions ({_ext_err})，可能无法运行 INT8 量化模型")
            log("  [TIP] 运行: pip install onnxruntime-extensions")
        
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            onnx_model,
            sess_options=sess_opts,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        )
        input_name  = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape  # [1, 3, H, W]
        imgsz = input_shape[2] if isinstance(input_shape[2], int) else 640
        log(f"  模型输入尺寸: {imgsz}x{imgsz}")

        # ── 读取并预处理图片 ───────────────────────────────
        log("[推理] 读取图片...")
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            log(f"[ERROR] 无法读取图片: {image_path}")
            sys.exit(1)
        orig_h, orig_w = img_orig.shape[:2]

        img = cv2.resize(img_orig, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_f = img.astype(np.float32) / 255.0
        img_f = np.transpose(img_f, (2, 0, 1))   # HWC → CHW
        img_f = np.expand_dims(img_f, axis=0)     # → 1,C,H,W

        # ── 推理 ──────────────────────────────────────────
        log("[推理] 执行推理...")
        outputs = sess.run(None, {input_name: img_f})
        log(f"  输出张量数: {len(outputs)}，第一个形状: {outputs[0].shape}")

        # ── 解析 YOLOv8 ONNX 输出 ─────────────────────────
        # YOLOv8 ONNX 输出: [1, num_classes+4, num_boxes] 或 [1, num_boxes, num_classes+4]
        pred = outputs[0]
        if pred.ndim == 3:
            # shape: [1, 4+nc, na] → transpose → [1, na, 4+nc]
            if pred.shape[1] < pred.shape[2]:
                pred = pred.transpose(0, 2, 1)
            pred = pred[0]  # [na, 4+nc]
        else:
            pred = pred[0]

        # 列: cx, cy, w, h, cls0_conf, cls1_conf, ...
        # YOLOv8 ONNX 输出坐标为 imgsz(640) 像素坐标系（非归一化！）
        num_cls = pred.shape[1] - 4
        boxes_xywh   = pred[:, :4]   # cx,cy,w,h (像素, 基于 imgsz=640)
        class_scores = pred[:, 4:]   # [na, nc]

        # 获取每个框的最大类别
        cls_ids     = np.argmax(class_scores, axis=1)
        cls_scores  = np.max(class_scores, axis=1)

        # 过滤低置信度
        mask = cls_scores >= conf_thresh
        if mask.sum() == 0:
            log("[推理] ⚠️ 未检测到目标（置信度不足）")
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump([], f)
            log("INFER_DONE:0")
            return

        boxes_f  = boxes_xywh[mask]
        scores_f = cls_scores[mask]
        ids_f    = cls_ids[mask]

        # cx,cy,w,h (基于 imgsz=640 的像素坐标) → x,y,w,h (原图像素坐标)
        scale_x = orig_w / float(imgsz)
        scale_y = orig_h / float(imgsz)
        boxes_px = np.zeros_like(boxes_f)
        boxes_px[:, 0] = (boxes_f[:, 0] - boxes_f[:, 2] / 2) * scale_x   # x = (cx - w/2) * sx
        boxes_px[:, 1] = (boxes_f[:, 1] - boxes_f[:, 3] / 2) * scale_y   # y = (cy - h/2) * sy
        boxes_px[:, 2] = boxes_f[:, 2] * scale_x                         # w *= sx
        boxes_px[:, 3] = boxes_f[:, 3] * scale_y                         # h *= sy

        # NMS
        keep = nms(boxes_px, scores_f, iou_thresh)
        boxes_px = boxes_px[keep]
        scores_f = scores_f[keep]
        ids_f    = ids_f[keep]

        # ── 类别名 ─────────────────────────────────────────
        # 尝试从模型元数据读取类别名
        try:
            meta = sess.get_modelmeta()
            custom = meta.custom_metadata_map
            if 'names' in custom:
                import ast
                names_map = ast.literal_eval(custom['names'])
                if isinstance(names_map, dict):
                    class_names = [names_map.get(i, f"cls{i}") for i in range(num_cls)]
                else:
                    class_names = list(names_map)
            else:
                class_names = [f"cls{i}" for i in range(num_cls)]
        except Exception:
            class_names = ["puddle"] if num_cls == 1 else [f"cls{i}" for i in range(num_cls)]

        # ── 组装结果 ──────────────────────────────────────
        results = []
        for i in range(len(boxes_px)):
            x, y, w, h = boxes_px[i].tolist()
            conf = float(scores_f[i])
            cid  = int(ids_f[i])
            label = class_names[cid] if cid < len(class_names) else f"cls{cid}"
            results.append({
                "label": label,
                "conf":  round(conf, 4),
                "box":   [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]
            })

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        log(f"[推理] ✅ 检测完成，共找到 {len(results)} 个目标")
        for r in results:
            log(f"       {r['label']}  conf={r['conf']:.3f}  box={r['box']}")
        log("=" * 50)
        log(f"INFER_DONE:{len(results)}")

    except ImportError as e:
        log(f"[ERROR] 缺少依赖: {e}")
        log("  请安装: pip install onnxruntime opencv-python numpy")
        sys.exit(1)
    except Exception as e:
        log(f"[ERROR] 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
