# YOLOv11 Predictor Module Design

**Date:** 2026-01-29
**Status:** Approved
**Author:** Claude Code

## Overview

Design and implementation of a YOLOv11 inference module with Ultralytics-style interface, supporting letterbox preprocessing, coordinate mapping, and image/video prediction with result visualization.

## Design Goals

1. **Ultralytics-compatible API**: Provide familiar `results[0].boxes.xyxy` style interface
2. **Accurate coordinate mapping**: Handle letterbox transformations correctly for any aspect ratio
3. **Multi-format support**: Images (single/batch) and video files
4. **Flexible preprocessing**: Hybrid letterbox mode (dynamic default, fixed optional)

## Architecture

### File Organization

```
engine/
├── train.py          # Existing training module
├── validate.py       # Existing validation module
└── predict.py        # New prediction module

utils/
└── __init__.py       # Export new classes
```

### Core Components

#### 1. LetterBox Class

Preprocessing class that maintains aspect ratio during resize.

**Responsibilities:**
- Apply letterbox transformation (resize + pad)
- Store transformation parameters for coordinate reversal
- Support both dynamic and fixed size modes

**Key Methods:**
- `__call__(img, target_size=640, auto=False)`: Transform image and return params
- Returns: `(transformed_img, (ratio, (pad_x, pad_y)))`

#### 2. Results Class

Ultralytics-style result container.

**API:**
```python
results[0].boxes.xyxy    # (N, 4) Bounding boxes in xyxy format
results[0].boxes.conf    # (N,) Confidence scores
results[0].boxes.cls     # (N,) Class indices
results[0].orig_shape    # (H, W) Original image shape
```

#### 3. YOLO Class

Main inference interface.

**Usage:**
```python
model = YOLO("runs/train/exp/weights/best.pt")
results = model.predict(source="image.jpg", conf=0.25, save=True)
```

**Parameters:**
- `source`: Image path, directory, or video file
- `conf`: Confidence threshold (default: 0.25)
- `iou`: NMS IoU threshold (default: 0.45)
- `img_size`: Target size for letterbox (default: 640, None for dynamic)
- `save`: Whether to save annotated results (default: False)
- `save_dir`: Output directory (default: "runs/predict")

#### 4. _post_process Function

Enhanced version of `validate._post_process_predictions` with coordinate mapping.

**Responsibilities:**
- Extract boxes, scores, labels from model output
- Apply confidence threshold
- Convert cxcywh to xyxy
- Apply NMS
- Scale coordinates back to original image space

## Data Flow

```
Original Image (1280x720)
    ↓
[LetterBox] ratio=0.5, pad=(0, 140)
    ↓
Resized + Padded (640x640)
    ↓
[YOLOv11 Forward] Predictions in 640x640 space
    ↓
[NMS Filter] (N, 4) [cx, cy, w, h]
    ↓
[ScaleCoords] Inverse transform to 1280x720
    ↓
[Results] Encapsulated results
```

## Coordinate Mapping

### LetterBox Transformation

**Preprocessing:**
```python
ratio = min(target_size / img_h, target_size / img_w)
scaled_h, scaled_w = img_h * ratio, img_w * ratio
pad_h = (target_size - scaled_h) / 2
pad_w = (target_size - scaled_w) / 2
```

**Post-processing (Inverse):**
```python
def scale_coords(coords, orig_shape, ratio, pad):
    """coords: (N, 4) [cx, cy, w, h] in target space"""
    coords[:, 0] = (coords[:, 0] - pad[0]) / ratio      # cx
    coords[:, 1] = (coords[:, 1] - pad[1]) / ratio      # cy
    coords[:, 2:] = coords[:, 2:] / ratio               # w, h
    return coords
```

### Key Formulas

| Step | Operation |
|------|-----------|
| 1. Subtract padding | `coord - pad` |
| 2. Divide by ratio | `coord / ratio` |

This correctly handles images of any aspect ratio (16:9, 4:3, 1:1, etc.).

## Error Handling

### Input Validation

- Check file/path existence
- Validate weight file format (.pt only)
- Warn on class count mismatch
- Detect corrupted images

### Edge Cases

| Scenario | Handling |
|----------|----------|
| No detections | Return empty Results object |
| Tiny images (< target_size) | Upscale via letterbox |
| Large images (> 4K) | Warning, continue (may OOM) |
| Video stream interruption | Save processed frames |
| Unsupported image format | Clear error message |

### Save Logic

- Auto-create output directories
- Use same codec/fps for video output
- Preserve original image quality

## Implementation Plan

1. Create `engine/predict.py` with core components
2. Implement `LetterBox` class
3. Implement `Results` class
4. Implement `YOLO` class with `predict()` method
5. Add coordinate mapping in `_scale_coords()`
6. Implement visualization with `cv2`
7. Add tests in `tests/test_predict.py`

## Dependencies

- Existing: `torch`, `cv2`, `numpy`
- Reuse: `validate._post_process_predictions` (enhanced)
- Model: `models.YOLOv11`, `modules.head.DetectAnchorFree`

## Notes

- Model output format: `(predictions, dict)` where predictions is `(bs, anchors, 4+nc)` in `[cx, cy, w, h, cls1, cls2, ...]` format
- Checkpoint format: `{'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'}`
- Inference mode requires `model.detect.eval()` for proper output format
