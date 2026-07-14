"""yolo-pi-bench: CLI benchmark tool for YOLOv11n across PyTorch/ONNX/NCNN/TFLite."""

import argparse
import time
from pathlib import Path

from .accuracy import bench_accuracy
from .chart import plot_fps_chart
from .formats import FORMAT_ORDER, resolve_candidates
from .report import write_csv, write_report
from .speed import bench_speed, load_frames


def cmd_benchmark(args):
    model_path = Path(args.model)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - set(FORMAT_ORDER)
    if unknown:
        raise SystemExit(f"Unknown format(s): {', '.join(sorted(unknown))}. "
                          f"Choose from: {', '.join(FORMAT_ORDER)}")

    overrides = {
        "onnx": args.onnx_path,
        "ncnn": args.ncnn_path,
        "tflite": args.tflite_path,
    }
    candidates = resolve_candidates(model_path, args.imgsz, formats, overrides)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    speed_results = []
    if not args.skip_speed:
        print(f"Loading {args.frames + args.warmup} frames from {args.source} ...")
        frames = load_frames(args.source, args.frames, args.warmup)
        for i, c in enumerate(candidates):
            r = bench_speed(c.path, frames, args.warmup, args.imgsz, c.label, args.threads)
            if r:
                speed_results.append(r)
            if i < len(candidates) - 1:
                time.sleep(args.cooldown)  # avoid later formats inheriting earlier formats' heat

    accuracy_results = []
    if not args.skip_accuracy:
        for c in candidates:
            r = bench_accuracy(c.path, c.label, args.data, args.imgsz)
            if r:
                accuracy_results.append(r)

    if not speed_results and not accuracy_results:
        raise SystemExit("No formats benchmarked -- check the model path and --formats.")

    if speed_results:
        write_csv(speed_results, outdir / "speed_results.csv")
    if accuracy_results:
        write_csv(accuracy_results, outdir / "accuracy_results.csv")

    chart_filename = "fps_chart.png" if speed_results else None
    if speed_results:
        plot_fps_chart(speed_results, args.imgsz, outdir / chart_filename)

    report_path = outdir / "report.md"
    write_report(report_path, args.model, args.imgsz, args.source,
                 speed_results, accuracy_results, chart_filename)

    print(f"\nSaved report to {report_path}")


def cmd_export(args):
    from ultralytics import YOLO

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unsupported = set(formats) - {"onnx", "ncnn"}
    if unsupported:
        raise SystemExit(
            f"export does not support: {', '.join(sorted(unsupported))}. "
            "TFLite INT8 needs full TensorFlow -- export it off-Pi (e.g. Google Colab) "
            "and point --tflite-path at the copied file when benchmarking."
        )

    model_path = Path(args.model)
    stem = model_path.stem
    model = YOLO(str(model_path))

    for fmt in formats:
        print(f"Exporting {fmt.upper()} at imgsz={args.imgsz} ...")
        model.export(format=fmt, imgsz=args.imgsz)
        if fmt == "ncnn" and args.imgsz != 640:
            # Ultralytics always names the NCNN export dir "{stem}_ncnn_model";
            # suffix it so a later benchmark at this imgsz can find it (see
            # resolve_candidates()'s shape-lock note in formats.py).
            default_dir = model_path.parent / f"{stem}_ncnn_model"
            sized_dir = model_path.parent / f"{stem}_ncnn_model_{args.imgsz}"
            default_dir.rename(sized_dir)
            print(f"  -> {sized_dir}")

    print("Done.")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="yolo-pi-bench",
        description="Benchmark YOLOv11n speed and accuracy across PyTorch/ONNX/NCNN/TFLite "
                     "on Raspberry Pi 5, without installing full TensorFlow.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="benchmark FPS and mAP across formats")
    b.add_argument("model", help="path to the base .pt model")
    b.add_argument("--imgsz", type=int, default=640, help="inference size (default: 640)")
    b.add_argument("--formats", default=",".join(FORMAT_ORDER),
                    help=f"comma-separated subset of: {', '.join(FORMAT_ORDER)}")
    b.add_argument("--source", default="test.mp4", help="video file or webcam index")
    b.add_argument("--frames", type=int, default=200, help="timed frames per format")
    b.add_argument("--warmup", type=int, default=20, help="untimed warmup frames")
    b.add_argument("--cooldown", type=int, default=30,
                    help="seconds between formats, to avoid thermal bias (default: 30)")
    b.add_argument("--threads", type=int, default=4, help="TFLite interpreter thread count")
    b.add_argument("--data", default="coco128.yaml", help="dataset yaml for mAP validation")
    b.add_argument("--skip-speed", action="store_true", help="skip the FPS benchmark")
    b.add_argument("--skip-accuracy", action="store_true", help="skip the mAP validation")
    b.add_argument("--onnx-path", help="override the guessed ONNX file path")
    b.add_argument("--ncnn-path", help="override the guessed NCNN model dir")
    b.add_argument("--tflite-path", help="override the guessed TFLite file path")
    b.add_argument("--outdir", default="bench_out", help="output directory (default: bench_out)")
    b.set_defaults(func=cmd_benchmark)

    e = sub.add_parser("export", help="export a .pt model to ONNX/NCNN (local, no TFLite)")
    e.add_argument("model", help="path to the base .pt model")
    e.add_argument("--imgsz", type=int, default=640, help="export input size (default: 640)")
    e.add_argument("--formats", default="onnx,ncnn", help="comma-separated subset of: onnx, ncnn")
    e.set_defaults(func=cmd_export)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
