#!/usr/bin/env python3
"""Run a student-trained YOLO model on a static image or USB webcam on Raspberry Pi 5."""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

SUPPORTED_FILES = {".pt", ".onnx"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def deployment_paths() -> tuple[Path, Path, Path]:
    root = repository_root()
    model_dir = root / "student_work/models"
    result_dir = root / "student_work/results/pi5"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Student model folder not found: {model_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return root, model_dir, result_dir


def discover_models(model_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILES:
            candidates.append(path)
        elif path.is_dir() and path.name.endswith("_ncnn_model"):
            candidates.append(path)
    return candidates


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def choose_model(root: Path, model_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = resolve_path(root, explicit)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(
            f"No .pt, .onnx, or NCNN model found in {model_dir}. "
            "Complete Session 1 and push the model to the fork first."
        )

    print("\nAvailable student models")
    for index, path in enumerate(models, start=1):
        print(f"[{index}] {path.name}")

    while True:
        raw = input(f"Select model [1-{len(models)}]: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            selected = -1
        if 1 <= selected <= len(models):
            return models[selected - 1]
        print("Invalid selection.")


def annotate(result, model_name: str, confidence: float, latency_ms: float, fps: float):
    frame = result.plot()
    count = len(result.boxes) if result.boxes is not None else 0
    lines = [
        f"Model: {model_name}",
        f"Megalopa count: {count}",
        f"Confidence threshold: {confidence:.2f}",
        f"Inference latency: {latency_ms:.1f} ms",
        f"Average FPS: {fps:.2f}",
        "Q: quit    S: save frame",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 26
    return frame, count


def save_summary(result_dir: Path, name: str, summary: dict) -> Path:
    path = result_dir / name
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved summary:", path)
    return path


def run_image(root: Path, model: YOLO, model_path: Path, source_raw: str, result_dir: Path, conf: float, imgsz: int) -> int:
    source = resolve_path(root, source_raw)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    started = time.perf_counter()
    result = model.predict(source=str(source), conf=conf, imgsz=imgsz, verbose=False)[0]
    latency_ms = (time.perf_counter() - started) * 1000.0
    fps = 1000.0 / max(latency_ms, 1e-9)
    frame, count = annotate(result, model_path.name, conf, latency_ms, fps)

    output = result_dir / "static_detection.jpg"
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Unable to save result image: {output}")

    print("Predicted count:", count)
    print("Latency (ms):", round(latency_ms, 2))
    print("Saved:", output)

    cv2.imshow("Lab 4 - Static Megalopa Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_summary(result_dir, "static_performance_summary.json", {
        "mode": "image",
        "model": str(model_path.relative_to(root)),
        "source": str(source.relative_to(root) if source.is_relative_to(root) else source),
        "confidence": conf,
        "imgsz": imgsz,
        "predicted_count": count,
        "latency_ms": latency_ms,
        "fps": fps,
    })
    return 0


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open USB webcam index {index}")
    return cap


def run_camera(root: Path, model: YOLO, model_path: Path, result_dir: Path, camera_index: int, conf: float, imgsz: int) -> int:
    cap = open_camera(camera_index)
    latencies: deque[float] = deque(maxlen=60)
    frame_count = 0
    saved_count = 0
    last_count = 0
    started = time.perf_counter()

    print("USB webcam opened. Press Q to quit or S to save the current annotated frame.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("USB webcam did not return a frame")

            inference_started = time.perf_counter()
            result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            latencies.append(latency_ms)
            average_latency = sum(latencies) / len(latencies)
            average_fps = 1000.0 / max(average_latency, 1e-9)
            annotated, last_count = annotate(result, model_path.name, conf, average_latency, average_fps)

            cv2.imshow("Lab 4 - USB Webcam Megalopa Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            frame_count += 1

            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                saved_count += 1
                output = result_dir / f"live_detection_{saved_count:02d}.jpg"
                cv2.imwrite(str(output), annotated)
                print("Saved:", output)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    average_fps = 1000.0 / average_latency if average_latency else 0.0
    summary = {
        "mode": "camera",
        "model": str(model_path.relative_to(root)),
        "camera_index": camera_index,
        "confidence": conf,
        "imgsz": imgsz,
        "frames_processed": frame_count,
        "elapsed_seconds": elapsed,
        "average_latency_ms": average_latency,
        "average_fps": average_fps,
        "last_predicted_count": last_count,
        "saved_frames": saved_count,
    }
    save_summary(result_dir, "live_performance_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("image", "camera"), default="camera")
    parser.add_argument("--model", help="Model path relative to the repository root, or an absolute path")
    parser.add_argument("--source", help="Static-image path when --mode image")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")

    root, model_dir, result_dir = deployment_paths()
    model_path = choose_model(root, model_dir, args.model)
    print("Repository:", root)
    print("Selected model:", model_path)
    model = YOLO(str(model_path))

    if args.mode == "image":
        if not args.source:
            parser.error("--source is required in image mode")
        return run_image(root, model, model_path, args.source, result_dir, args.confidence, args.imgsz)

    return run_camera(root, model, model_path, result_dir, args.camera_index, args.confidence, args.imgsz)


if __name__ == "__main__":
    raise SystemExit(main())
