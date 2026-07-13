#!/usr/bin/env python3
"""
Edge AI benchmark: YOLOv11n inference speed across model formats on Raspberry Pi.

Benchmarks any subset of: PyTorch (.pt), ONNX (.onnx), NCNN, TFLite INT8.
Skips formats whose files or runtimes are missing, so you can run it
after each export phase and watch the table grow.

Usage:
    python benchmark.py --source test.mp4
    python benchmark.py --source test.mp4 --frames 300 --imgsz 640
    python benchmark.py --source 0            # webcam

Outputs:
    results.csv       raw numbers, one row per format
    results.md        markdown table for the README
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

# ---------------------------------------------------------------- helpers

def read_cpu_temp():
    """CPU temp in Celsius on Linux, or None if unavailable."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def load_frames(source, n_frames, warmup):
    """Read frames once into memory so disk I/O doesn't pollute timing."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    frames = []
    needed = n_frames + warmup
    while len(frames) < needed:
        ok, frame = cap.read()
        if not ok:
            if not frames:
                raise SystemExit("No frames read from source.")
            # Loop the video if it's shorter than needed
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(frame)
    cap.release()
    return frames


def bench_model(model_path, frames, warmup, imgsz, label):
    """Run Ultralytics inference over frames, return stats dict or None."""
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        print(f"[skip] {label}: {model_path} not found")
        return None
    try:
        model = YOLO(str(path), task="detect")
    except Exception as e:
        print(f"[skip] {label}: failed to load ({e})")
        return None

    size_mb = round(
        sum(f.stat().st_size for f in path.rglob("*")) / 1e6
        if path.is_dir()
        else path.stat().st_size / 1e6,
        1,
    )

    temp_start = read_cpu_temp()
    print(f"\n=== {label} ({size_mb} MB) | temp start: {temp_start}°C ===")

    # Warmup (not timed)
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, verbose=False)

    latencies = []
    for i, f in enumerate(frames[warmup:], 1):
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, verbose=False)
        latencies.append(time.perf_counter() - t0)
        if i % 50 == 0:
            print(f"  {i}/{len(frames) - warmup} frames | "
                  f"running avg {i / sum(latencies):.2f} FPS")

    temp_end = read_cpu_temp()
    total = sum(latencies)
    latencies.sort()
    stats = {
        "format": label,
        "size_mb": size_mb,
        "fps": round(len(latencies) / total, 2),
        "latency_ms_mean": round(total / len(latencies) * 1000, 1),
        "latency_ms_p95": round(latencies[int(len(latencies) * 0.95)] * 1000, 1),
        "temp_start_c": temp_start,
        "temp_end_c": temp_end,
    }
    print(f"  -> {stats['fps']} FPS | mean {stats['latency_ms_mean']} ms | "
          f"p95 {stats['latency_ms_p95']} ms | temp end: {temp_end}°C")
    return stats


# ---------------------------------------------------------------- main

CANDIDATES = [
    ("yolo11n.pt", "PyTorch FP32"),
    ("yolo11n.onnx", "ONNX FP32"),
    ("yolo11n_ncnn_model", "NCNN"),
    ("yolo11n_int8.tflite", "TFLite INT8"),
    ("yolo11n_saved_model/yolo11n_full_integer_quant.tflite", "TFLite INT8 (full)"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test.mp4", help="video file or webcam index")
    ap.add_argument("--frames", type=int, default=200, help="timed frames per format")
    ap.add_argument("--warmup", type=int, default=20, help="untimed warmup frames")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    args = ap.parse_args()

    print(f"Loading {args.frames + args.warmup} frames from {args.source} ...")
    frames = load_frames(args.source, args.frames, args.warmup)

    results = []
    for path, label in CANDIDATES:
        r = bench_model(path, frames, args.warmup, args.imgsz, label)
        if r:
            results.append(r)
        # Cooldown between formats so thermals don't favor the first run
        time.sleep(30)

    if not results:
        raise SystemExit("No models benchmarked. Is yolo11n.pt in this folder?")

    keys = list(results[0].keys())
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    base_fps = results[0]["fps"]
    with open("results.md", "w") as f:
        f.write("| Format | Size (MB) | FPS | Mean latency (ms) | p95 (ms) | Speedup |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['format']} | {r['size_mb']} | {r['fps']} | "
                    f"{r['latency_ms_mean']} | {r['latency_ms_p95']} | "
                    f"{round(r['fps'] / base_fps, 2)}x |\n")

    print("\nSaved results.csv and results.md")
    print(open("results.md").read())


if __name__ == "__main__":
    main()
