#!/usr/bin/env python3
"""
Line chart of inference FPS vs. input size (imgsz), one line per format,
built from sweep.csv (see benchmark.py --sweep).

Usage:
    python make_sweep_chart.py
Outputs:
    sweep_chart.png
"""

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same fixed categorical order as make_chart.py — NCNN and TFLite INT8 keep
# the colors they had there (amber, dark green), not reassigned to slots
# 1/2, so a format's color stays stable across every chart in this repo.
FORMAT_COLORS = {
    "PyTorch FP32": "#2a78d6",
    "ONNX FP32": "#1baf7a",
    "NCNN": "#eda100",
    "TFLite INT8": "#008300",
    "TFLite INT8 (full)": "#4a3aa7",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def main():
    rows = []
    with open("sweep.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    by_format = defaultdict(list)
    for r in rows:
        by_format[r["format"]].append((int(r["imgsz"]), float(r["fps"])))
    for pts in by_format.values():
        pts.sort()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # TFLite INT8 is a single point (shape-locked to its one Colab export,
    # see benchmark.py NCNN_PATH_BY_IMGSZ comment) -- offset its label above
    # the marker and NCNN's below, so the two end-labels never collide.
    label_offset = {"NCNN": (8, -14), "TFLite INT8": (8, 8)}

    for fmt, pts in by_format.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = FORMAT_COLORS.get(fmt, INK_MUTED)
        ax.plot(
            xs, ys,
            color=color, linewidth=2, marker="o", markersize=8,
            zorder=3, label=fmt,
        )
        # Direct label at the last point only.
        ax.annotate(
            f"{ys[-1]:.1f} FPS",
            xy=(xs[-1], ys[-1]),
            xytext=label_offset.get(fmt, (8, 0)), textcoords="offset points",
            ha="left", va="center", fontsize=9, color=INK_PRIMARY,
        )

    ax.set_xlabel("Input size (imgsz, px)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Frames per second", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "YOLOv11n inference speed vs. input size (Raspberry Pi 5)",
        color=INK_PRIMARY, fontsize=12, pad=14, loc="left",
    )
    sizes = sorted({p[0] for pts in by_format.values() for p in pts})
    ax.set_xticks(sizes)
    ax.set_xlim(sizes[0] - 40, sizes[-1] + 90)  # room for trailing labels

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=10)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)

    legend = ax.legend(
        loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    fig.savefig("sweep_chart.png", facecolor=SURFACE)
    print("Saved sweep_chart.png")


if __name__ == "__main__":
    main()
