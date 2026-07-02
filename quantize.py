#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型量化脚本：将 YOLOv8 .pt 模型 → ONNX → INT8 静态量化 ONNX
使用方式（由 Qt GUI 调用）：
    python quantize.py <model_pt> <output_dir> <calib_images_dir> <quant_type>
参数：
    model_pt        : YOLOv8 .pt 模型文件路径（如 best.pt）
    output_dir      : 量化后模型输出目录
    calib_images_dir: 校准图片目录（用于静态量化），如 dataset/images/val
    quant_type      : dynamic（动态量化）或 static（静态量化）
"""

import sys
import os
import io

# 确保 stdout 以 UTF-8 输出，Qt 可以正确接收
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

def main():
    if len(sys.argv) < 5:
        log("[ERROR] 用法: python quantize.py <model_pt> <output_dir> <calib_dir> <quant_type>")
        sys.exit(1)

    model_pt   = sys.argv[1]
    output_dir = sys.argv[2]
    calib_dir  = sys.argv[3]
    quant_type = sys.argv[4].lower()  # "dynamic" or "static"

    # ── 检查输入 ──────────────────────────────────────────
    if not os.path.isfile(model_pt):
        log(f"[ERROR] 模型文件不存在: {model_pt}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    base_name   = os.path.splitext(os.path.basename(model_pt))[0]
    onnx_path   = os.path.join(output_dir, base_name + ".onnx")
    quant_path  = os.path.join(output_dir, base_name + f"_{quant_type}_int8.onnx")

    # ── 步骤1：导出 ONNX ──────────────────────────────────
    log("=" * 50)
    log("[步骤 1/3] 导出 ONNX 模型...")
    log(f"  源模型: {model_pt}")
    log(f"  ONNX 输出: {onnx_path}")

    try:
        from ultralytics import YOLO
        model = YOLO(model_pt)
        # 导出为 ONNX，opset=12 兼容性好
        export_result = model.export(format='onnx', imgsz=640, opset=12, simplify=True)
        # ultralytics 默认导出到模型同目录，需要移动
        default_onnx = os.path.splitext(model_pt)[0] + ".onnx"
        if os.path.isfile(default_onnx) and default_onnx != onnx_path:
            import shutil
            shutil.move(default_onnx, onnx_path)
        elif str(export_result) != onnx_path and os.path.isfile(str(export_result)):
            import shutil
            shutil.copy(str(export_result), onnx_path)
        log(f"[步骤 1/3] ✅ ONNX 导出完成: {onnx_path}")
    except Exception as e:
        log(f"[ERROR] ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 步骤2：量化 ───────────────────────────────────────
    log("")
    log(f"[步骤 2/3] 开始 ONNX Runtime {quant_type.upper()} INT8 量化...")
    log(f"  输入 ONNX: {onnx_path}")
    log(f"  输出路径 : {quant_path}")

    try:
        from onnxruntime.quantization import (
            quantize_dynamic, quantize_static,
            QuantType, QuantFormat,
            CalibrationDataReader
        )
        import onnxruntime as ort

        if quant_type == "dynamic":
            # 动态量化：只量化权重，激活在推理时动态量化
            log("[步骤 2/3] 使用动态量化（仅量化权重，速度快）")
            # onnxruntime >= 1.16 起 quantize_dynamic 不再支持 optimize_model 参数
            import inspect
            _dyn_sig = inspect.signature(quantize_dynamic).parameters
            # 使用 QDQ (Quantize-DeQuantize) 格式，避免 ConvInteger 算子（CPU 不支持）
            _dyn_kwargs = dict(
                model_input=onnx_path,
                model_output=quant_path,
                weight_type=QuantType.QUInt8,  # QUInt8 配合 QDQ 格式兼容性最好
            )
            # 可选参数检测
            for _extra in ['op_types_to_quantize', 'per_channel', 'reduce_range', 'optimize_model']:
                if _extra in _dyn_sig:
                    if _extra == 'op_types_to_quantize':
                        _dyn_kwargs[_extra] = ['MatMul', 'Conv']
                    elif _extra == 'optimize_model':
                        _dyn_kwargs[_extra] = True
                    elif _extra == 'per_channel':
                        _dyn_kwargs[_extra] = False
                    elif _extra == 'reduce_range':
                        _dyn_kwargs[_extra] = False
            quantize_dynamic(**_dyn_kwargs)
            log("  [INFO] 量化格式: QDQ (ConvInteger 兼容模式)")
        else:
            # 静态量化：需要校准数据
            log(f"[步骤 2/3] 使用静态量化（需要校准图片目录: {calib_dir}）")

            if not os.path.isdir(calib_dir):
                log(f"[WARN] 校准目录不存在: {calib_dir}，将改用动态量化。")
                import inspect as _inspect2
                _sig2 = _inspect2.signature(quantize_dynamic).parameters
                _kw2 = dict(model_input=onnx_path, model_output=quant_path, weight_type=QuantType.QUInt8)
                for _ex in ['op_types_to_quantize', 'optimize_model']:
                    if _ex in _sig2:
                        if _ex == 'op_types_to_quantize':
                            _kw2[_ex] = ['MatMul', 'Conv']
                        else:
                            _kw2[_ex] = True
                quantize_dynamic(**_kw2)
            else:
                # 构建校准数据读取器
                import numpy as np

                def get_calib_images(calib_dir, max_num=100):
                    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                    files = [
                        os.path.join(calib_dir, f)
                        for f in os.listdir(calib_dir)
                        if f.lower().endswith(exts)
                    ]
                    return files[:max_num]

                class YoloCalibDataReader(CalibrationDataReader):
                    def __init__(self, images, input_name, imgsz=640):
                        import cv2
                        self.images = images
                        self.input_name = input_name
                        self.imgsz = imgsz
                        self.index = 0
                        self.cv2 = cv2

                    def get_next(self):
                        if self.index >= len(self.images):
                            return None
                        path = self.images[self.index]
                        self.index += 1
                        log(f"  校准图片 [{self.index}/{len(self.images)}]: {os.path.basename(path)}")
                        img = self.cv2.imread(path)
                        if img is None:
                            return self.get_next()
                        img = self.cv2.resize(img, (self.imgsz, self.imgsz))
                        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
                        img = np.expand_dims(img, axis=0)   # → 1,C,H,W
                        return {self.input_name: img}

                # 获取模型输入名
                sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                del sess

                calib_images = get_calib_images(calib_dir)
                if not calib_images:
                    log("[WARN] 校准目录中没有找到图片，改用动态量化。")
                    import inspect as _inspect3
                    _sig3 = _inspect3.signature(quantize_dynamic).parameters
                    _kw3 = dict(model_input=onnx_path, model_output=quant_path, weight_type=QuantType.QUInt8)
                    for _ex in ['op_types_to_quantize', 'optimize_model']:
                        if _ex in _sig3:
                            if _ex == 'op_types_to_quantize':
                                _kw3[_ex] = ['MatMul', 'Conv']
                            else:
                                _kw3[_ex] = True
                    quantize_dynamic(**_kw3)
                else:
                    log(f"  找到 {len(calib_images)} 张校准图片")
                    dr = YoloCalibDataReader(calib_images, input_name)
                    import inspect as _inspect4
                    _sig4 = _inspect4.signature(quantize_static).parameters
                    _kw4 = dict(
                        model_input=onnx_path,
                        model_output=quant_path,
                        calibration_data_reader=dr,
                        quant_format=QuantFormat.QOperator,
                        weight_type=QuantType.QInt8,
                    )
                    if 'optimize_model' in _sig4:
                        _kw4['optimize_model'] = True
                    quantize_static(**_kw4)

        log(f"[步骤 2/3] ✅ 量化完成: {quant_path}")

    except ImportError as e:
        log(f"[ERROR] 缺少依赖库: {e}")
        log("  请运行: pip install onnxruntime onnx onnxruntime-extensions")
        sys.exit(1)
    except Exception as e:
        log(f"[ERROR] 量化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 步骤3：验证量化模型 ────────────────────────────────
    log("")
    log("[步骤 3/3] 验证量化模型...")
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(quant_path, providers=['CPUExecutionProvider'])
        input_meta = sess.get_inputs()[0]
        log(f"  模型输入: {input_meta.name}  形状: {input_meta.shape}  类型: {input_meta.type}")
        # 生成随机输入测试一次
        dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
        outputs = sess.run(None, {input_meta.name: dummy})
        log(f"  推理测试: 输出 {len(outputs)} 个张量，第一个形状: {outputs[0].shape}")
        log("[步骤 3/3] ✅ 量化模型验证通过")
    except Exception as e:
        log(f"[WARN] 验证阶段出错（模型可能仍可用）: {e}")

    # ── 完成 ──────────────────────────────────────────────
    log("")
    log("=" * 50)
    log("✅ 量化全部完成！")
    log(f"  ONNX 原始模型  : {onnx_path}")
    log(f"  量化后模型     : {quant_path}")
    log(f"  量化类型       : {quant_type.upper()} INT8")

    # 计算大小对比
    if os.path.isfile(onnx_path) and os.path.isfile(quant_path):
        sz_orig  = os.path.getsize(onnx_path)  / 1024 / 1024
        sz_quant = os.path.getsize(quant_path) / 1024 / 1024
        log(f"  原始 ONNX 大小 : {sz_orig:.2f} MB")
        log(f"  量化后大小     : {sz_quant:.2f} MB")
        ratio = (1 - sz_quant / sz_orig) * 100 if sz_orig > 0 else 0
        log(f"  压缩率         : {ratio:.1f}%")
    log("=" * 50)
    log(f"QUANT_OUTPUT_PATH:{quant_path}")  # Qt 用于捕捉输出路径的特殊标记行

if __name__ == "__main__":
    main()
