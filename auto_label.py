"""
YOLO 自动标注脚本
用法: python auto_label.py <图片路径> <模型路径> <输出JSON路径>
输出: JSON数组，每项 { "label": "类别名", "box": [cx, cy, w, h] (像素坐标) }
"""
import sys
import json
import numpy as np

# 强制 UTF-8
sys.stdout = __import__('io').TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = __import__('io').TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from ultralytics import YOLO

def main():
    if len(sys.argv) < 4:
        print("用法: python auto_label.py <图片> <模型.pt> <输出.json>", file=sys.stderr)
        sys.exit(1)

    img_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    if not img_path.exists():
        print(f"错误: 图片不存在: {img_path}", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"错误: 模型不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载模型: {model_path}")
    model = YOLO(str(model_path))

    print(f"正在推理: {img_path}")
    results = model(str(img_path), verbose=False, device='cpu')

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            w = float(x2 - x1)
            h = float(y2 - y1)
            detections.append({
                "label": label,
                "box": [cx, cy, w, h]
            })

    print(f"检测到 {len(detections)} 个目标")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_path}")
    sys.exit(0)

if __name__ == "__main__":
    main()
