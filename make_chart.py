#!/usr/bin/env python3
"""
Bar chart of inference FPS by model format, built from results.csv.

Usage:
    python make_chart.py
Outputs:
    fps_chart.png
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (categorical, fixed order) — see dataviz skill.
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def main():
    rows = []
    with open("results.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    formats = [r["format"] for r in rows]
    fps = [float(r["fps"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(
        formats, fps,
        color=COLORS[: len(formats)],
        width=0.55,
        zorder=3,
    )

    ax.set_ylabel("Frames per second", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "YOLOv11n inference speed by format (Raspberry Pi, 640px)",
        color=INK_PRIMARY, fontsize=12, pad=14, loc="left",
    )

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=10)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)

    # Direct labels on each bar (value + speedup vs first row).
    base = fps[0]
    for bar, val in zip(bars, fps):
        speedup = val / base
        label = f"{val:.2f} FPS" if speedup == 1 else f"{val:.2f} FPS ({speedup:.1f}x)"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=INK_PRIMARY,
        )

    fig.tight_layout()
    fig.savefig("fps_chart.png", facecolor=SURFACE)
    print("Saved fps_chart.png")


if __name__ == "__main__":
    main()
