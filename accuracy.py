#!/usr/bin/env python3
"""
Edge AI benchmark: accuracy (mAP) across model formats on COCO128.

Mirrors benchmark.py's CANDIDATES list so the same set of exported
formats gets both a speed number and an accuracy number.

Usage:
    python accuracy.py

Outputs:
    accuracy.csv
    accuracy.md
"""

import csv
from pathlib import Path

from ultralytics import YOLO

CANDIDATES = [
    ("yolo11n.pt", "PyTorch FP32"),
    ("yolo11n.onnx", "ONNX FP32"),
    ("yolo11n_ncnn_model", "NCNN"),
    ("yolo11n_int8.tflite", "TFLite INT8"),
    ("yolo11n_saved_model/yolo11n_full_integer_quant.tflite", "TFLite INT8 (full)"),
]


def val_model(model_path, label):
    path = Path(model_path)
    if not path.exists():
        print(f"[skip] {label}: {model_path} not found")
        return None
    try:
        model = YOLO(str(path), task="detect")
    except Exception as e:
        print(f"[skip] {label}: failed to load ({e})")
        return None

    print(f"\n=== {label}: validating on COCO128 ===")
    metrics = model.val(data="coco128.yaml", imgsz=640, verbose=False)
    return {
        "format": label,
        "map50_95": round(float(metrics.box.map), 4),
        "map50": round(float(metrics.box.map50), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def main():
    results = []
    for path, label in CANDIDATES:
        r = val_model(path, label)
        if r:
            results.append(r)

    if not results:
        raise SystemExit("No models validated. Is yolo11n.pt in this folder?")

    keys = list(results[0].keys())
    with open("accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    with open("accuracy.md", "w") as f:
        f.write("| Format | mAP50-95 | mAP50 | Precision | Recall |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['format']} | {r['map50_95']} | {r['map50']} | "
                    f"{r['precision']} | {r['recall']} |\n")

    print("\nSaved accuracy.csv and accuracy.md")
    print(open("accuracy.md").read())


if __name__ == "__main__":
    main()
