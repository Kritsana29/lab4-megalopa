#!/usr/bin/env python3
"""Run pretrained or student-trained YOLO detection on images or a USB webcam."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

SUPPORTED_MODELS = {".pt", ".onnx"}
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_stem(text: str, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return (cleaned or "result")[:limit]


def deployment_paths() -> tuple[Path, Path, Path, Path]:
    root = repository_root()
    baseline_model = root / "shared/models/yolo26n.pt"
    custom_model_dir = root / "student_work/models"
    result_root = root / "student_work/results/pi5"
    result_root.mkdir(parents=True, exist_ok=True)
    return root, baseline_model, custom_model_dir, result_root


def discover_models(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []

    candidates: list[Path] = []
    for path in sorted(model_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODELS:
            candidates.append(path)
        elif path.is_dir() and path.name.endswith("_ncnn_model"):
            candidates.append(path)
    return candidates


def choose_model(
    root: Path,
    baseline_model: Path,
    custom_model_dir: Path,
    explicit: str | None,
    baseline: bool,
) -> tuple[Path, str]:
    if baseline and explicit:
        raise ValueError("Use either --baseline or --model, not both.")

    if baseline:
        if not baseline_model.is_file():
            raise FileNotFoundError(
                f"Baseline model not found: {baseline_model}\n"
                "Confirm that shared/models/yolo26n.pt exists in the cloned fork."
            )
        return baseline_model, "baseline"

    if explicit:
        path = resolve_path(root, explicit)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path, "custom"

    models = discover_models(custom_model_dir)
    if not models:
        raise FileNotFoundError(
            f"No .pt, .onnx, or NCNN model found in {custom_model_dir}. "
            "Complete Session 1 and push student_work/models/ to the group fork."
        )

    print("\nAvailable custom models")
    for index, path in enumerate(models, start=1):
        print(f"[{index}] {path.name}")

    while True:
        raw = input(f"Select model [1-{len(models)}]: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            selected = -1

        if 1 <= selected <= len(models):
            return models[selected - 1], "custom"
        print("Invalid selection.")


def choose_image(
    root: Path,
    source: str | None,
    image_index: int | None,
    image_dir: str,
) -> Path:
    if source and image_index is not None:
        raise ValueError("Use either --source or --image-index, not both.")

    if source:
        path = resolve_path(root, source)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    folder = resolve_path(root, image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Validation image folder not found: {folder}")

    images = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
        ),
        key=lambda path: path.name.lower(),
    )

    if not images:
        raise FileNotFoundError(f"No supported images found in {folder}")

    print("\nAvailable validation images")
    for index, path in enumerate(images, start=1):
        print(f"[{index}] {path.name}")

    if image_index is None:
        while True:
            raw = input(f"Select image [1-{len(images)}]: ").strip()
            try:
                image_index = int(raw)
            except ValueError:
                image_index = -1

            if 1 <= image_index <= len(images):
                break
            print("Invalid selection.")

    if not 1 <= image_index <= len(images):
        raise ValueError(f"--image-index must be between 1 and {len(images)}")

    selected = images[image_index - 1]
    print("Selected image:", selected.name)
    return selected


def class_counts(result) -> Counter[str]:
    counts: Counter[str] = Counter()
    if result.boxes is None or result.boxes.cls is None:
        return counts

    names = result.names
    for raw_class_id in result.boxes.cls.detach().cpu().tolist():
        class_id = int(raw_class_id)
        if isinstance(names, dict):
            name = str(names.get(class_id, f"class_{class_id}"))
        else:
            name = str(names[class_id]) if class_id < len(names) else f"class_{class_id}"
        counts[name] += 1
    return counts


def annotate(
    result,
    model_name: str,
    model_role: str,
    confidence: float,
    latency_ms: float,
    fps: float,
):
    frame = result.plot()
    counts = class_counts(result)
    total = sum(counts.values())

    count_text = ", ".join(f"{name}={count}" for name, count in counts.items())
    count_lines = []
    if count_text:
        while len(count_text) > 58:
            split_at = count_text.rfind(", ", 0, 58)
            split_at = split_at if split_at > 0 else 58
            count_lines.append(count_text[:split_at])
            count_text = count_text[split_at:].lstrip(", ")
        count_lines.append(count_text)
    else:
        count_lines.append("none")

    lines = [
        f"Mode: {model_role}",
        f"Model: {model_name}",
        f"Total detections: {total}",
        *[f"Classes: {line}" if index == 0 else f"         {line}"
          for index, line in enumerate(count_lines)],
        f"Confidence: {confidence:.2f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Average FPS: {fps:.2f}",
        "Q: quit    S: save frame",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (0, 0, 0), 1, cv2.LINE_AA,
        )
        y += 25

    return frame, total, dict(counts)


def save_summary(result_dir: Path, name: str, summary: dict) -> Path:
    path = result_dir / name
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved summary:", path)
    return path


def run_image(
    root: Path,
    model: YOLO,
    model_path: Path,
    model_role: str,
    source: Path,
    result_dir: Path,
    conf: float,
    imgsz: int,
) -> int:
    started = time.perf_counter()
    result = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )[0]

    latency_ms = (time.perf_counter() - started) * 1000.0
    fps = 1000.0 / max(latency_ms, 1e-9)
    frame, total, counts = annotate(
        result, model_path.name, model_role, conf, latency_ms, fps
    )

    stem = safe_stem(source.stem)
    output = result_dir / f"static_{stem}_detection.jpg"
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Unable to save result image: {output}")

    print("Total detections:", total)
    print("Class counts:", counts)
    print("Latency (ms):", round(latency_ms, 2))
    print("Saved:", output)

    cv2.imshow(f"Lab 4 - {model_role.title()} Static Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_summary(
        result_dir,
        f"static_{stem}_summary.json",
        {
            "mode": "image",
            "model_role": model_role,
            "model": relative_or_absolute(model_path, root),
            "source": relative_or_absolute(source, root),
            "confidence": conf,
            "imgsz": imgsz,
            "total_detections": total,
            "class_counts": counts,
            "latency_ms": latency_ms,
            "fps": fps,
            "output_image": relative_or_absolute(output, root),
        },
    )
    return 0


def open_camera(index: int):
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open USB webcam index {index}")
    return capture


def run_camera(
    root: Path,
    model: YOLO,
    model_path: Path,
    model_role: str,
    result_dir: Path,
    camera_index: int,
    conf: float,
    imgsz: int,
) -> int:
    capture = open_camera(camera_index)
    latencies: deque[float] = deque(maxlen=60)
    frame_count = 0
    saved_count = 0
    last_total = 0
    last_counts: dict[str, int] = {}
    started = time.perf_counter()

    print("USB webcam opened. Press Q to quit or S to save the annotated frame.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("USB webcam did not return a frame")

            inference_started = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )[0]

            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            latencies.append(latency_ms)
            average_latency = sum(latencies) / len(latencies)
            average_fps = 1000.0 / max(average_latency, 1e-9)

            annotated, last_total, last_counts = annotate(
                result,
                model_path.name,
                model_role,
                conf,
                average_latency,
                average_fps,
            )

            cv2.imshow(
                f"Lab 4 - {model_role.title()} USB Webcam Detection",
                annotated,
            )

            key = cv2.waitKey(1) & 0xFF
            frame_count += 1

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                saved_count += 1
                output = result_dir / f"live_detection_{saved_count:02d}.jpg"
                if cv2.imwrite(str(output), annotated):
                    print("Saved:", output)
                else:
                    print("Unable to save:", output)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    average_fps = 1000.0 / average_latency if average_latency else 0.0

    summary = {
        "mode": "camera",
        "model_role": model_role,
        "model": relative_or_absolute(model_path, root),
        "camera_index": camera_index,
        "confidence": conf,
        "imgsz": imgsz,
        "frames_processed": frame_count,
        "elapsed_seconds": elapsed,
        "average_latency_ms": average_latency,
        "average_fps": average_fps,
        "last_total_detections": last_total,
        "last_class_counts": last_counts,
        "saved_frames": saved_count,
        "result_directory": relative_or_absolute(result_dir, root),
    }

    save_summary(result_dir, "live_performance_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("image", "camera"), default="camera")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use shared/models/yolo26n.pt instead of a student custom model",
    )
    parser.add_argument(
        "--model",
        help="Custom model path relative to the repository root or an absolute path",
    )
    parser.add_argument(
        "--source",
        help="Explicit static-image path when --mode image",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        help="1-based index in the alphabetically sorted validation-image list",
    )
    parser.add_argument(
        "--image-dir",
        default="shared/validation/images",
        help="Validation-image folder relative to the repository root",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")

    if args.baseline and args.model:
        parser.error("Use either --baseline or --model, not both.")

    root, baseline_model, custom_model_dir, result_root = deployment_paths()

    try:
        model_path, model_role = choose_model(
            root=root,
            baseline_model=baseline_model,
            custom_model_dir=custom_model_dir,
            explicit=args.model,
            baseline=args.baseline,
        )

        result_dir = result_root / model_role
        result_dir.mkdir(parents=True, exist_ok=True)

        print("Repository:", root)
        print("Model role:", model_role)
        print("Selected model:", model_path)
        print("Result folder:", result_dir)

        model = YOLO(str(model_path))

        if args.mode == "image":
            source = choose_image(
                root=root,
                source=args.source,
                image_index=args.image_index,
                image_dir=args.image_dir,
            )
            return run_image(
                root=root,
                model=model,
                model_path=model_path,
                model_role=model_role,
                source=source,
                result_dir=result_dir,
                conf=args.confidence,
                imgsz=args.imgsz,
            )

        return run_camera(
            root=root,
            model=model,
            model_path=model_path,
            model_role=model_role,
            result_dir=result_dir,
            camera_index=args.camera_index,
            conf=args.confidence,
            imgsz=args.imgsz,
        )

    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
