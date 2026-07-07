# Liveness Detection System

Real-time face liveness detection and anti-spoofing using [YOLOv8](https://github.com/ultralytics/ultralytics). The system distinguishes a **real** human face captured live from a **fake** one presented via a photo, screen replay, or printed image — a core building block for secure face-recognition and identity-verification pipelines.

## How It Works

The project is a full pipeline, from data collection to real-time inference:

1. **Collect data** (`collectingData.py`) — Captures webcam frames, detects faces with `cvzone`'s `FaceDetector`, filters out blurry frames using a Laplacian variance threshold, and saves each accepted frame together with a YOLO-format label (`class x_center y_center width height`).
2. **Split the dataset** (`SpitData.py`) — Shuffles the collected samples and splits them into `train` / `val` / `test` sets (70% / 20% / 10%), organizing them into the YOLO directory structure and auto-generating the `data.yaml` config.
3. **Train the model** (`Train.py`) — Fine-tunes a YOLOv8n (`yolov8n.pt`) model on the two classes (`fake`, `real`) for 300 epochs.
4. **Run detection** (`main.py`) — Loads the trained weights and runs real-time inference on a webcam feed, drawing a bounding box around each detected face, color-coded green for `real` and red for `fake`, with the confidence score and live FPS overlaid.

## Project Structure

```
Liveness-detection-System/
├── images/               # Sample results / demo screenshots
├── collectingData.py     # Step 1: capture and label webcam data
├── SpitData.py            # Step 2: split data into train/val/test
├── Train.py               # Step 3: train the YOLOv8 model
├── main.py                # Step 4: real-time liveness detection
└── LICENSE
```

## Requirements

- Python 3.8+
- A webcam
- Key dependencies:
  - `ultralytics` (YOLOv8)
  - `opencv-python`
  - `cvzone`
  - `numpy`

Install them with:

```bash
pip install ultralytics opencv-python cvzone numpy
```

> The repo doesn't currently ship a `requirements.txt` — consider adding one (`pip freeze > requirements.txt`) so setup is fully reproducible.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MukhriddinAI/Liveness-detection-System.git
cd Liveness-detection-System
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install ultralytics opencv-python cvzone numpy
```

### 3. Build your own dataset (optional — skip if using a pretrained model)

Set `classid` in `collectingData.py` to `0` for fake samples or `1` for real samples, then run it once per class in front of the webcam:

```bash
python collectingData.py
```

Frames and YOLO-format labels are saved to `Datasets/Real` (or wherever `direct` points).

### 4. Split the dataset

```bash
python SpitData.py
```

This produces `Datasets/SplitNewData/` with `train`, `val`, and `test` subfolders plus a `data.yaml`.

### 5. Train the model

```bash
python Train.py
```

Trains a YOLOv8n model for 300 epochs on the split dataset. The best weights are saved under `runs/detect/train/weights/best.pt`.

### 6. Run real-time detection

Update the model path in `main.py` to point at your trained weights (or a provided pretrained model), then run:

```bash
python main.py
```

A webcam window opens, showing a bounding box, label (`REAL`/`FAKE`), and confidence score for each detected face. Press `q` or close the window to exit.

## Notes & Known Limitations

- `main.py` currently loads the model from a hardcoded relative path (`../models/nano_vers_60.pt`) — update this to match your own trained weights location.
- Detection confidence threshold (`given_confidence`) defaults to `0.6` and can be tuned in `main.py`.
- Some code comments (in `collectingData.py`) are in Uzbek — contributions to translate/document further are welcome.

If you find this project useful, consider giving it a ⭐! 
