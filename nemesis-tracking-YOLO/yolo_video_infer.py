from ultralytics import YOLO
import cv2
from pathlib import Path


# This file is for creating output (already bounded) video using trained model (I used initial trained model).
# Just modify config below for your environment, and add best.pt (or whatever you have trained model, file_name.pt) in WEIGHTS_PATH. 

# --------- config ---------
WEIGHTS_PATH = "/workspace/ARC/dataset/yolo_dataset/runs/robot_det/weights/best.pt"
INPUT_VIDEO  = "/workspace/ARC/input/fight2.mp4"
OUTPUT_DIR   = "/workspace/ARC/output"
OUTPUT_NAME  = "fight3_from_13400.mp4"
START_FRAME  = 13400            # <-- start detection from here
IMG_SIZE     = 960
CONF_THRESH  = 0.25
DEVICE       = 0           # "0", "cpu", etc.
# --------------------------

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / OUTPUT_NAME)

    print(f"[info] loading model from: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    print(f"[info] opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("[error] failed to open input video")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < START_FRAME:
            frame_idx += 1
            continue

        # Run YOLO on this frame
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            device=DEVICE,
            verbose=False,
        )

        # results[0].plot() returns the frame with boxes drawn (BGR)
        annotated = results[0].plot()
        writer.write(annotated)

        if processed % 100 == 0:
            print(f"[info] processed frames: {processed} (video frame idx: {frame_idx})")

        frame_idx += 1
        processed += 1

    cap.release()
    writer.release()
    print(f"[ok] done. saved: {out_path}")
    print(f"[info] total frames written: {processed}")

if __name__ == "__main__":
    main()
