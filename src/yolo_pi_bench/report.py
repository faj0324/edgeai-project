"""CSV + markdown report generation."""

import csv
from datetime import datetime, timezone
from pathlib import Path


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _speed_table(speed_results):
    base_fps = speed_results[0]["fps"]
    lines = [
        "| Format | Size (MB) | FPS | Mean latency (ms) | p95 (ms) | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in speed_results:
        lines.append(
            f"| {r['format']} | {r['size_mb']} | {r['fps']} | "
            f"{r['latency_ms_mean']} | {r['latency_ms_p95']} | "
            f"{round(r['fps'] / base_fps, 2)}x |"
        )
    return "\n".join(lines)


def _accuracy_table(accuracy_results):
    lines = [
        "| Format | mAP50-95 | mAP50 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for r in accuracy_results:
        lines.append(
            f"| {r['format']} | {r['map50_95']} | {r['map50']} | "
            f"{r['precision']} | {r['recall']} |"
        )
    return "\n".join(lines)


def write_report(path: Path, model: str, imgsz: int, source: str,
                  speed_results: list[dict], accuracy_results: list[dict],
                  chart_filename: str | None):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# yolo-pi-bench report",
        "",
        f"- Model: `{model}`",
        f"- Image size: {imgsz}px",
        f"- Source: `{source}`",
        f"- Generated: {ts}",
        "",
    ]

    parts.append("## Inference speed")
    parts.append("")
    if speed_results:
        parts.append(_speed_table(speed_results))
        parts.append("")
        if chart_filename:
            parts.append(f"![FPS by format]({chart_filename})")
            parts.append("")
    else:
        parts.append("No formats benchmarked.")
        parts.append("")

    if accuracy_results:
        parts.append("## Accuracy")
        parts.append("")
        parts.append(_accuracy_table(accuracy_results))
        parts.append("")

    path.write_text("\n".join(parts))
