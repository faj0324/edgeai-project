"""Inference-speed benchmarking: load frames once, time each format."""

import contextlib
import functools
import time
from pathlib import Path


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


def read_cpu_temp():
    """CPU temp in Celsius on Linux, or None if unavailable."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def load_frames(source, n_frames, warmup):
    """Read frames once into memory so disk I/O doesn't pollute timing."""
    import cv2

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
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop a video shorter than needed
            continue
        frames.append(frame)
    cap.release()
    return frames


def bench_speed(path: Path, frames, warmup, imgsz, label, threads=4):
    """Run Ultralytics inference over frames, return a stats dict or None."""
    from ultralytics import YOLO

    if not path.exists():
        print(f"[skip] {label}: {path} not found")
        return None

    is_tflite = path.suffix == ".tflite"
    with (litert_num_threads(threads) if is_tflite else contextlib.nullcontext()):
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
        "imgsz": imgsz,
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
