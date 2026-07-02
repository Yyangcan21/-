"""
train.py  —  YOLO 模型训练脚本
由 Qt GUI 调用，参数通过命令行传入：
  sys.argv[1]  data_path  数据集 yaml 路径
  sys.argv[2]  epochs     训练轮数
  sys.argv[3]  batch      批大小
  sys.argv[4]  lr         初始学习率
  sys.argv[5]  imgsz      图像尺寸（可选，默认 640）
  sys.argv[6]  device     训练设备（可选，默认 cpu）
  sys.argv[7]  model      预训练模型文件名（可选，默认 yolov8n.pt）
  sys.argv[8]  workers    数据加载线程数（可选，默认 0）
"""

import sys
import os

# ---- 强制 UTF-8 输出（解决 Windows 下 Qt 读取乱码问题）----
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

def main():
    # ──────────────────────────────────────────
    # 1. 参数解析
    # ──────────────────────────────────────────
    if len(sys.argv) < 5:
        print("❌ Error: 参数不足！Qt 未正确传递参数。", flush=True)
        print(f"   当前接收到的参数列表: {sys.argv}", flush=True)
        sys.exit(1)

    try:
        data_path = sys.argv[1]
        epochs    = int(sys.argv[2])
        batch     = int(sys.argv[3])
        lr        = float(sys.argv[4])
        imgsz     = int(sys.argv[5])   if len(sys.argv) > 5 else 640
        device    = sys.argv[6]        if len(sys.argv) > 6 else "cpu"
        model_name= sys.argv[7]        if len(sys.argv) > 7 else "yolov8n.pt"
        workers   = int(sys.argv[8])   if len(sys.argv) > 8 else 0
    except (ValueError, IndexError) as e:
        print(f"❌ 参数解析失败: {e}", flush=True)
        sys.exit(1)

    # ──────────────────────────────────────────
    # 2. 打印接收到的参数（Qt 日志可见）
    # ──────────────────────────────────────────
    print(f"✅ 参数接收成功", flush=True)
    print(f"   数据集  : {data_path}", flush=True)
    print(f"   模型    : {model_name}", flush=True)
    print(f"   轮数    : {epochs}", flush=True)
    print(f"   批大小  : {batch}", flush=True)
    print(f"   学习率  : {lr}", flush=True)
    print(f"   图像尺寸: {imgsz}", flush=True)
    print(f"   设备    : {device}", flush=True)
    print(f"   Workers : {workers}", flush=True)

    # ──────────────────────────────────────────
    # 3. 检查数据集文件是否存在
    # ──────────────────────────────────────────
    if not os.path.isfile(data_path):
        print(f"❌ 数据集文件不存在: {data_path}", flush=True)
        sys.exit(1)

    # ──────────────────────────────────────────
    # 4. 导入 ultralytics（放在参数检查之后，避免启动时卡住）
    # ──────────────────────────────────────────
    try:
        print("正在导入 ultralytics，请稍候...", flush=True)
        from ultralytics import YOLO
        print("✅ ultralytics 导入成功", flush=True)
    except ImportError as e:
        print(f"❌ 无法导入 ultralytics: {e}", flush=True)
        print("   请在当前 Python 环境中执行: pip install ultralytics", flush=True)
        sys.exit(1)

    # ──────────────────────────────────────────
    # 5. 加载模型
    # ──────────────────────────────────────────
    print(f"正在加载预训练模型 ({model_name})...", flush=True)
    try:
        model = YOLO(model_name)
        print(f"✅ 模型加载成功: {model_name}", flush=True)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}", flush=True)
        sys.exit(1)

    # ──────────────────────────────────────────
    # 6. 开始训练
    # ──────────────────────────────────────────
    print("", flush=True)
    print("训练即将开始，请稍候...", flush=True)
    print(f"{'='*50}", flush=True)

    try:
        results = model.train(
            data     = data_path,
            epochs   = epochs,
            batch    = batch,
            lr0      = lr,
            imgsz    = imgsz,
            device   = device,
            workers  = workers,
            project  = 'runs/train',
            name     = 'exp',
            exist_ok = True,
            verbose  = True,   # 输出详细进度（包含 epoch/total 格式）
        )

        print(f"{'='*50}", flush=True)
        print("✅ 训练任务已完成！", flush=True)

        # 打印指标摘要
        if results and hasattr(results, 'results_dict'):
            print("", flush=True)
            print("── 训练结果摘要 ──", flush=True)
            for k, v in results.results_dict.items():
                print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}", flush=True)

    except KeyboardInterrupt:
        print("", flush=True)
        print("⏹ 训练被中断（KeyboardInterrupt）", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"", flush=True)
        print(f"❌ 训练过程中发生错误:", flush=True)
        print(f"   {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
