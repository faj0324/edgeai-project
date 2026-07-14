"""mAP accuracy validation across model formats."""

from pathlib import Path


def bench_accuracy(path: Path, label: str, data: str, imgsz: int):
    from ultralytics import YOLO

    if not path.exists():
        print(f"[skip] {label}: {path} not found")
        return None
    try:
        model = YOLO(str(path), task="detect")
    except Exception as e:
        print(f"[skip] {label}: failed to load ({e})")
        return None

    print(f"\n=== {label}: validating on {data} ===")
    metrics = model.val(data=data, imgsz=imgsz, verbose=False)
    return {
        "format": label,
        "map50_95": round(float(metrics.box.map), 4),
        "map50": round(float(metrics.box.map50), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
