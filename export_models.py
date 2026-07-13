#!/usr/bin/env python3
"""
Export YOLOv11n to ONNX and NCNN on the Pi.

TFLite INT8 is NOT exported here on purpose: it needs full TensorFlow,
which is too heavy for the Pi. Do that export in Google Colab (see
CLAUDE.md, Phase 3b) and copy the .tflite file back.
"""
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

print("Exporting ONNX ...")
model.export(format="onnx", imgsz=640)

print("Exporting NCNN ...")
model.export(format="ncnn", imgsz=640)

print("Done. Files: yolo11n.onnx, yolo11n_ncnn_model/")
