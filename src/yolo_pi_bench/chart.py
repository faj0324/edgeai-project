"""Bar chart of inference FPS by format (see dataviz skill for palette)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .formats import FORMAT_COLORS

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_fps_chart(speed_results: list[dict], imgsz: int, outpath: Path):
    formats = [r["format"] for r in speed_results]
    fps = [r["fps"] for r in speed_results]
    colors = [FORMAT_COLORS.get(f, INK_MUTED) for f in formats]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(formats, fps, color=colors, width=0.55, zorder=3)

    ax.set_ylabel("Frames per second", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"YOLOv11n inference speed by format (Raspberry Pi 5, {imgsz}px)",
        color=INK_PRIMARY, fontsize=12, pad=14, loc="left",
    )

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=10)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)

    base = fps[0]
    for bar, val in zip(bars, fps):
        speedup = val / base
        text = f"{val:.2f} FPS" if speedup == 1 else f"{val:.2f} FPS ({speedup:.1f}x)"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=INK_PRIMARY,
        )

    fig.tight_layout()
    fig.savefig(outpath, facecolor=SURFACE)
    plt.close(fig)
