#!/usr/bin/env python3
"""
Edge AI benchmark: YOLOv11n inference speed across model formats on Raspberry Pi 5.

Benchmarks any subset of: PyTorch (.pt), ONNX (.onnx), NCNN, TFLite INT8.
Skips formats whose files or runtimes are missing, so you can run it
after each export phase and watch the table grow.

Usage:
    python benchmark.py --source test.mp4
    python benchmark.py --source test.mp4 --frames 300 --imgsz 640
    python benchmark.py --source 0            # webcam
    python benchmark.py --sweep               # NCNN + TFLite INT8 @ 320/480/640

Outputs:
    results.csv       raw numbers, one row per format (default mode)
    results.md        markdown table for the README
    sweep.csv         raw numbers, one row per format x imgsz (--sweep mode)
    sweep.md          markdown table for the README
"""

import argparse
import contextlib
import csv
import functools
import time
from pathlib import Path

import cv2

SWEEP_IMGSZ = [320, 480, 640]


@contextlib.contextmanager
def litert_num_threads(n=4):
    """Force ai-edge-litert's Interpreter to use n CPU threads.

    Ultralytics' LiteRTBackend hardcodes `Interpreter(str(tflite_file))` with
    no num_threads argument, which ai-edge-litert then defaults to 1 (see
    ai_edge_litert/interpreter.py: `int(num_threads or 1)`). That leaves 3 of
    the Pi 5's 4 cores idle during TFLite inference. LiteRTBackend does a
    local `from ai_edge_litert.interpreter import Interpreter` at load time,
    so patching the module attribute here is picked up on the next load.
    """
    from ai_edge_litert import interpreter as litert_interpreter

    original = litert_interpreter.Interpreter
    litert_interpreter.Interpreter = functools.partial(original, num_threads=n)
    try:
        yield
    finally:
        litert_interpreter.Interpreter = original

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

    is_tflite = path.suffix == ".tflite"
    # Real context manager for TFLite so the Interpreter is built with
    # num_threads=4; a no-op stack for everything else.
    with (litert_num_threads(4) if is_tflite else contextlib.nullcontext()):
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

# NCNN (and TFLite) bake the input shape into the exported graph: feeding a
# mismatched imgsz to a fixed-shape NCNN model corrupts memory instead of
# raising a clean error (reproduced: "corrupted size vs. prev_size" /
# "munmap_chunk(): invalid pointer", exit 134). So the sweep needs one NCNN
# export per size (see yolo11n_ncnn_model_{320,480}/, made with a one-off
# export script; yolo11n_ncnn_model/ is already the 640 export).
NCNN_PATH_BY_IMGSZ = {
    320: "yolo11n_ncnn_model_320",
    480: "yolo11n_ncnn_model_480",
    640: "yolo11n_ncnn_model",
}

def run_sweep(args):
    """Benchmark NCNN across SWEEP_IMGSZ; TFLite INT8 only at 640 (its export size).

    TFLite INT8 is exported once, in Colab, at imgsz=640 -- it's also
    shape-locked, but re-exporting at other sizes needs full TensorFlow,
    which this script must not install locally (see CLAUDE.md). Feeding it
    a mismatched imgsz raises a clean ValueError rather than corrupting
    memory, so it's simply skipped for 320/480 instead of crashing.
    """
    print(f"Loading {args.frames + args.warmup} frames from {args.source} ...")
    frames = load_frames(args.source, args.frames, args.warmup)

    results = []
    for imgsz in SWEEP_IMGSZ:
        ncnn_path = NCNN_PATH_BY_IMGSZ.get(imgsz)
        if ncnn_path and Path(ncnn_path).exists():
            r = bench_model(ncnn_path, frames, args.warmup, imgsz, "NCNN")
            if r:
                r["imgsz"] = imgsz
                results.append(r)
            time.sleep(30)

        if imgsz == 640:
            r = bench_model("yolo11n_int8.tflite", frames, args.warmup, imgsz, "TFLite INT8")
            if r:
                r["imgsz"] = imgsz
                results.append(r)
            time.sleep(30)

    if not results:
        raise SystemExit("No models benchmarked. Are yolo11n_ncnn_model_*/ and "
                          "yolo11n_int8.tflite in this folder?")

    keys = ["format", "imgsz", "size_mb", "fps", "latency_ms_mean",
            "latency_ms_p95", "temp_start_c", "temp_end_c"]
    with open("sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    with open("sweep.md", "w") as f:
        f.write("| Format | imgsz | FPS | Mean latency (ms) | p95 (ms) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['format']} | {r['imgsz']} | {r['fps']} | "
                    f"{r['latency_ms_mean']} | {r['latency_ms_p95']} |\n")

    print("\nSaved sweep.csv and sweep.md")
    print(open("sweep.md").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test.mp4", help="video file or webcam index")
    ap.add_argument("--frames", type=int, default=200, help="timed frames per format")
    ap.add_argument("--warmup", type=int, default=20, help="untimed warmup frames")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    ap.add_argument("--sweep", action="store_true",
                     help="run NCNN + TFLite INT8 across imgsz 320/480/640 "
                          "instead of the normal single-imgsz, all-formats run")
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args)
        return

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
