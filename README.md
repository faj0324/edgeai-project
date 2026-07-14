# Edge AI Optimization Benchmark: YOLOv11n on Raspberry Pi 5

Benchmarking YOLOv11n object detection across model formats on a Raspberry Pi
5, comparing inference speed and accuracy trade-offs for edge deployment.
Pretrained COCO weights only — no training was done.

**Status: all four formats — PyTorch, ONNX, NCNN, and TFLite INT8 — are
benchmarked below. Phase 3 (TFLite INT8) is complete.**

## Hardware

- Raspberry Pi 5 Model B, 4 GB RAM, Ubuntu 24.04 LTS (aarch64)
- 4-core CPU, no GPU/NPU acceleration used
- CPU temperature logged before/after each format to catch thermal throttling

## Method

1. `export_models.py` converts `yolo11n.pt` to ONNX and NCNN using Ultralytics'
   built-in exporters, run directly on the Pi.
2. `benchmark.py` loads a fixed set of video frames into memory once (so disk
   I/O doesn't pollute timing), runs a 20-frame warmup per format, then times
   200 inference calls per format at 640px. A 30-second cooldown separates
   formats so later formats aren't penalized by heat carried over from earlier
   ones. TFLite inference is forced to `num_threads=4` (Ultralytics' LiteRT
   backend otherwise defaults to 1 thread, leaving 3 of the Pi 5's 4 cores
   idle — see `litert_num_threads()` in `benchmark.py`).
3. `accuracy.py` runs Ultralytics' standard `val()` on COCO128 for each format
   and records mAP50-95, mAP50, precision, and recall.
4. TFLite INT8: exported in Google Colab (`YOLO("yolo11n.pt").export(format="tflite",
   int8=True)`, calibrated on COCO128) since full TensorFlow is too heavy to
   install on the Pi. The resulting `yolo11n_int8.tflite` was copied back and
   run here through `ai-edge-litert` (the actively maintained successor to
   `tflite-runtime`, published for Python 3.12/aarch64) — no full TensorFlow
   was installed on the Pi, per the project's hardware constraints.
5. Test video: a 54-second, 768x432, 12fps clip containing people, bicycles,
   and cars (public sample footage, not recorded on this Pi — swap in your own
   `test.mp4` and re-run `benchmark.py` to reproduce with different footage).

Everything here is reproducible: `python export_models.py`, then
`python benchmark.py --source test.mp4`, then `python accuracy.py`. The
imgsz sweep is `python benchmark.py --sweep --source test.mp4`, then
`python make_sweep_chart.py` (needs `yolo11n_ncnn_model_320/` and `_480/`,
see Method above).

## Results: inference speed

| Format | Size (MB) | FPS | Mean latency (ms) | p95 (ms) | Speedup |
|---|---|---|---|---|---|
| PyTorch FP32 | 5.6 | 3.80 | 263.5 | 267.3 | 1.0x |
| ONNX FP32 | 10.7 | 4.74 | 210.8 | 258.9 | 1.25x |
| NCNN | 10.7 | 10.74 | 93.2 | 95.7 | 2.83x |
| TFLite INT8 | 3.1 | 18.53 | 54.0 | 62.2 | 4.88x |

![FPS by format](fps_chart.png)

TFLite INT8 is the fastest format tested: 4.88x the PyTorch baseline and
comfortably ahead of NCNN, while also being the smallest file by a wide
margin (3.1 MB vs 10.7 MB for the FP32 exports) — INT8 weights are a quarter
the size of FP32, and XNNPACK's integer NEON kernels convert that into real
latency savings, not just a smaller file. That gap only shows up correctly
with `num_threads=4` on the LiteRT interpreter (see Method); left at
Ultralytics' default of 1 thread, TFLite INT8 measured 10.29 FPS — still
fastest, but understating it by nearly half since 3 of the Pi 5's 4 cores
sat idle. NCNN remains a strong second at 2.83x with zero accuracy cost (see
below). ONNX Runtime's CPU execution provider on ARM64 barely beats
PyTorch — it's not using any ARM-specific kernel optimizations NCNN or
LiteRT have, so the export alone doesn't buy much without a compatible
runtime.

Note: this table was regenerated in one continuous session across all four
formats, so absolute FPS differs from earlier partial runs — the CPU started
each format a few degrees warmer than the original cool-start runs despite
the 30-second cooldowns. Relative ordering is consistent either way; absolute
FPS numbers are sensitive to thermal state at the start of a run.

CPU temperature rose to ~80-85°C over the PyTorch/ONNX/NCNN runs and stayed
lower for TFLite INT8, since it's simply less work per frame. None of the
runs showed throttling-induced slowdown (FPS was stable across each run),
but sustained real-world use (minutes, not tens of seconds) would likely
need active cooling or a duty cycle.

## Results: inference speed vs. input size (imgsz)

| Format | imgsz | FPS | Mean latency (ms) | p95 (ms) |
|---|---|---|---|---|
| NCNN | 320 | 32.90 | 30.4 | 51.9 |
| NCNN | 480 | 21.09 | 47.4 | 54.7 |
| NCNN | 640 | 11.21 | 89.2 | 93.6 |
| TFLite INT8 | 640 | 19.51 | 51.3 | 52.5 |

![FPS vs input size](sweep_chart.png)

`benchmark.py --sweep` benchmarks NCNN at imgsz 320/480/640 — nearly 3x
faster at 320 than 640, the expected roughly-quadratic cost of processing a
larger input. TFLite INT8 only appears at 640: both NCNN and TFLite bake the
input shape into the exported graph rather than tolerating a resize at
inference time like PyTorch/ONNX/TFLite-with-dynamic-shapes do, so feeding a
mismatched imgsz to a fixed-shape NCNN model doesn't raise a clean error —
it corrupts memory and crashes the process (`corrupted size vs. prev_size`,
reproduced twice). NCNN sidesteps this by exporting one model per size
(`yolo11n_ncnn_model_320/`, `_480/`, plus the existing 640 export) — cheap,
since it needs no TensorFlow and runs locally. TFLite INT8 export needs full
TensorFlow and happens in Colab (see Method), so re-exporting it at 320/480
just for this chart was out of scope; feeding it a mismatched imgsz raises a
clean `ValueError` instead of crashing, so it's simply skipped rather than
faked.

## Results: accuracy (COCO128, mAP50-95)

| Format | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|
| PyTorch FP32 | 0.5044 | 0.6724 | 0.6597 | 0.5932 |
| ONNX FP32 | 0.5051 | 0.6744 | 0.7415 | 0.5598 |
| NCNN | 0.4997 | 0.6756 | 0.7386 | 0.5613 |
| TFLite INT8 | 0.4212 | 0.6091 | 0.6329 | 0.5405 |

ONNX and NCNN are lossless graph conversions of the same FP32 weights, so
mAP50-95 is unchanged within noise (0.504 -> 0.505 -> 0.500, a 0.9% relative
difference). TFLite INT8 is not lossless: mAP50-95 drops from 0.5044 to
0.4212, a 16.5% relative loss (mAP50 drops a smaller but still real 9.4%,
from 0.6724 to 0.6091). This is exactly the tradeoff called out below —
quantizing YOLOv11n's small detection head to INT8 costs real accuracy in
exchange for the 4.88x speedup above, and on this model that cost is not
negligible.

## Quantization and why edge inference differs from Colab

Colab benchmarks (and most published model benchmarks) run on a GPU or a
server-grade CPU with wide SIMD, more cache, and no thermal ceiling. A
Raspberry Pi 5's CPU has none of that: no GPU here, ARM NEON instead of AVX,
much smaller caches, and it throttles under sustained load. So the same model
that hits real-time on a T4 in Colab can be single-digit FPS on a Pi purely
from missing acceleration, independent of INT8 quantization.

INT8 quantization (TFLite) attacks a different problem: even with a good CPU
runtime, FP32 matmuls are still doing 4x the memory bandwidth and no
integer SIMD path. Quantizing weights and activations to INT8 shrinks the
model, cuts memory bandwidth, and lets ARM's integer NEON instructions run
the convolutions, typically 2-4x faster than FP32 on the same hardware. The
cost is precision: rounding weights and activations to 256 discrete levels
loses information, and how much accuracy that costs is architecture- and
calibration-dependent — small detection heads with wide dynamic range can
lose several mAP points.

That prediction held up: TFLite INT8 is the fastest format measured (4.88x
PyTorch) but also the only one with a real accuracy cost, losing 16.5%
relative mAP50-95. Whether that trade is worth it depends on the deployment
— a coarse presence/absence detector can probably absorb it, a system relying
on tight box localization probably can't.

## Files

- `benchmark.py` — speed benchmark across formats, writes `results.csv`/`results.md`;
  `--sweep` mode writes `sweep.csv`/`sweep.md`
- `export_models.py` — ONNX + NCNN export (640px)
- `accuracy.py` — COCO128 mAP validation across formats, writes `accuracy.csv`/`accuracy.md`
- `make_chart.py` — builds `fps_chart.png` from `results.csv`
- `make_sweep_chart.py` — builds `sweep_chart.png` from `sweep.csv`
- `test.mp4` — test video used for speed benchmarking
- `yolo11n.pt` — pretrained COCO weights (not trained/fine-tuned here)
- `yolo11n.onnx` / `yolo11n_ncnn_model/` — exported by `export_models.py` on the Pi (640px)
- `yolo11n_ncnn_model_320/` / `_480/` — NCNN exports at those imgsz, for the sweep only
- `yolo11n_int8.tflite` — INT8 export, done in Google Colab (not on the Pi) and copied in
