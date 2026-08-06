#!/usr/bin/env python3
"""Check the prepared Raspberry Pi 5 software environment and USB webcam."""
from __future__ import annotations

import argparse
import glob
import os
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Architecture:", platform.machine())
    print("DISPLAY:", os.environ.get("DISPLAY", "<not set>"))
    print("Video devices:", ", ".join(glob.glob("/dev/video*")) or "<none>")

    failures: list[str] = []

    try:
        import cv2
        print("OpenCV:", cv2.__version__)
    except Exception as exc:
        failures.append(f"OpenCV import failed: {exc}")
        cv2 = None

    try:
        import ultralytics
        print("Ultralytics:", ultralytics.__version__)
    except Exception as exc:
        failures.append(f"Ultralytics import failed: {exc}")

    if cv2 is not None:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            failures.append(f"Unable to open USB webcam index {args.camera_index}")
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures.append("Webcam opened but did not return a frame")
            else:
                output = Path("webcam_environment_test.jpg")
                cv2.imwrite(str(output), frame)
                print("Webcam frame shape:", frame.shape)
                print("Saved test image:", output.resolve())
        cap.release()

    if failures:
        print("\nENVIRONMENT CHECK FAILED")
        for item in failures:
            print(" -", item)
        return 1

    print("\nENVIRONMENT CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
