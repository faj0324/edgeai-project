"""Model format definitions and file-path resolution."""

from dataclasses import dataclass
from pathlib import Path

FORMAT_ORDER = ["pytorch", "onnx", "ncnn", "tflite"]

FORMAT_LABELS = {
    "pytorch": "PyTorch FP32",
    "onnx": "ONNX FP32",
    "ncnn": "NCNN",
    "tflite": "TFLite INT8",
}

# Fixed categorical colors, kept stable across every chart (see dataviz skill).
FORMAT_COLORS = {
    "PyTorch FP32": "#2a78d6",
    "ONNX FP32": "#1baf7a",
    "NCNN": "#eda100",
    "TFLite INT8": "#008300",
}


@dataclass
class Candidate:
    key: str
    label: str
    path: Path


def resolve_candidates(model_path: Path, imgsz: int, formats: list[str],
                        overrides: dict[str, str | None]) -> list[Candidate]:
    """Guess each format's exported file/dir path from the base model's stem.

    NCNN bakes its input shape into the exported graph -- feeding it a
    mismatched imgsz corrupts memory instead of raising a clean error, so a
    non-640 imgsz is only attempted against a size-suffixed export
    (`{stem}_ncnn_model_{imgsz}`, see the `export` command) unless the user
    points --ncnn-path at one explicitly. TFLite INT8 is exported off-Pi at
    640 (see `export`, which excludes tflite -- it needs full TensorFlow),
    so it's only attempted at 640 unless --tflite-path overrides it.
    """
    stem = model_path.stem
    parent = model_path.parent
    out = []

    if "pytorch" in formats:
        out.append(Candidate("pytorch", FORMAT_LABELS["pytorch"],
                              Path(overrides["pytorch"]) if overrides.get("pytorch") else model_path))

    if "onnx" in formats:
        default = parent / f"{stem}.onnx"
        out.append(Candidate("onnx", FORMAT_LABELS["onnx"],
                              Path(overrides["onnx"]) if overrides.get("onnx") else default))

    if "ncnn" in formats:
        if overrides.get("ncnn"):
            out.append(Candidate("ncnn", FORMAT_LABELS["ncnn"], Path(overrides["ncnn"])))
        else:
            default = parent / (f"{stem}_ncnn_model" if imgsz == 640
                                 else f"{stem}_ncnn_model_{imgsz}")
            out.append(Candidate("ncnn", FORMAT_LABELS["ncnn"], default))

    if "tflite" in formats:
        if overrides.get("tflite"):
            out.append(Candidate("tflite", FORMAT_LABELS["tflite"], Path(overrides["tflite"])))
        elif imgsz == 640:
            out.append(Candidate("tflite", FORMAT_LABELS["tflite"], parent / f"{stem}_int8.tflite"))
        # else: shape-locked to its export size with no override given -- skip;
        # the caller reports it as not found rather than crashing on a bad shape.

    return out
