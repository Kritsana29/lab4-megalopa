# Raspberry Pi 5 Deployment

The laboratory uses a prepared Raspberry Pi 5 development kit with a screen and a USB webcam.

From the cloned student fork:

```bash
python3 shared/pi5/check_pi5_environment.py --camera-index 0
python3 shared/pi5/megalopa_detection_usb.py --mode image --source shared/validation/images/IMAGE_NAME.jpg
python3 shared/pi5/megalopa_detection_usb.py --mode camera --camera-index 0 --confidence 0.25 --imgsz 640
```

The detector scans `student_work/models/` and asks the student to select a `.pt`, `.onnx`, or optional NCNN model.

Controls:

- `S` saves an annotated frame.
- `Q` stops live inference and writes a JSON performance summary.
