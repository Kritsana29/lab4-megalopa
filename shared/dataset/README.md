# Dataset Placement

Place the final YOLO box-label dataset here:

```text
shared/dataset/images/
shared/dataset/labels/
```

Requirements:

- Each image must have a matching `.txt` label file with the same stem.
- Each non-empty YOLO row must contain: `class_id x_center y_center width height`.
- Coordinates must be normalized to `[0, 1]`.
- The class IDs must match `CLASS_NAMES` in the Session 1 notebook.
- Remove `.gitkeep` after adding real files if desired.
