# Lab 4 - Megalopa Detection: Fork, Train, and Deploy on Raspberry Pi 5

This repository is the instructor-owned upstream repository for a two-session Computer Engineering laboratory.

## Current workflow

1. The instructor maintains `WUStuLab/lab4-megalopa`.
2. Each student group forks the repository to a student-owned GitHub account.
3. Session 1 clones the group fork into Google Colab, trains YOLO26n, evaluates the model, exports ONNX, and pushes only `student_work/` back to the fork.
4. Session 2 clones the same fork on a prepared Raspberry Pi 5 development kit and runs the group model using a USB webcam.
5. The official graded submissions are two PDFs exported from the completed Colab notebooks.

## Repository layout

```text
shared/
  dataset/                 Instructor-provided images and YOLO labels
  validation/              Instructor-provided validation images and video
  models/                  Optional baseline weight storage
  notebooks/               Session 1 and Session 2 Colab notebooks
  pi5/                     Raspberry Pi 5 USB-webcam scripts
student_work/
  models/                  Student-trained .pt, ONNX, and optional NCNN models
  results/training/        Selected training/evaluation figures
  results/validation/      Image and video validation outputs
  results/pi5/             Pi screenshots and performance files, if retained
  session_information/     Student/fork metadata and result summaries
instructor/                Repository reset and release instructions
```

## Graded deliverables

- `SecXX_GroupXX_Lab4_Session1.pdf` - 60 points
- `SecXX_GroupXX_Lab4_Session2.pdf` - 40 points

Total: 100 points.

## Before publishing

Insert the real labelled dataset and validation materials, then test both notebooks and the Pi script once using a temporary student fork.
