import cv2
import os
import argparse
from pathlib import Path

SOURCE = "/Users/tsukasamiyaji/Desktop/Python3/Real Fights/fight1.avi"
OUT_FILE = "fight1_start.png"
OUT_IDX = "fight1_start_idx.txt"

def draw_hud(frame, idx, total, fps):
    h, w = frame.shape[:2]
    ts = (idx / fps) if fps and fps > 0 else 0.0
    text = f"Frame: {idx}/{total-1} | Time: {ts:.2f}s | FPS: {fps:.2f} | SPACE=Play/Pause  ←/→=Step  b=Save baseline  q=Quit"
    cv2.rectangle(frame, (0, h-28), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    ap = argparse.ArgumentParser(description="Pick and save a baseline frame from a video.")
    # ap.add_argument("video", help="Path to input video (e.g., video.avi)")
    ap.add_argument("--outdir", default="out/baseline", help="Directory to save baseline outputs")
    ap.add_argument("--start", type=int, default=18500, help="Start at this frame index")
    args = ap.parse_args()

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open video: {SOURCE}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        raise SystemExit("Error: unable to read frame count from video.")
    if fps <= 0:
        # Fallback: try to estimate later by stepping, but we’ll just mark as 30 for HUD
        fps = 30.0

    # Prepare output directory and filenames
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_stem = Path(SOURCE).stem
    baseline_img_path = outdir / OUT_FILE
    baseline_idx_path = outdir / OUT_IDX

    # Seek to start frame
    current_idx = clamp(args.start, 0, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)

    playing = True
    cv2.namedWindow("Baseline Picker", cv2.WINDOW_NORMAL)

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                # Reached the end; pause at last valid frame
                playing = False
                current_idx = total - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
                ret, frame = cap.read()
                if not ret:
                    break
        else:
            # When paused, we need to fetch the current frame to show it
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            if not ret:
                break

        if frame is None:
            break

        # Compute current index from capture (some backends lag; we track ourselves)
        if playing:
            # OpenCV returns next position *after* read; adjust for HUD
            current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            current_idx = clamp(current_idx, 0, total - 1)

        frame_disp = frame.copy()
        draw_hud(frame_disp, current_idx, total, fps)
        cv2.imshow("Baseline Picker", frame_disp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == 32:  # SPACE toggles play/pause
            playing = not playing
        elif key in (81, 2424832):  # Left arrow
            playing = False
            current_idx = clamp(current_idx - 1, 0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        elif key in (83, 2555904):  # Right arrow
            playing = False
            current_idx = clamp(current_idx + 1, 0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        elif key == ord('b'):  # Save current frame as baseline
            cv2.imwrite(str(baseline_img_path), frame)
            with open(baseline_idx_path, "w") as f:
                f.write(str(current_idx))
            print(f"Saved baseline image to: {baseline_img_path}")
            print(f"Saved baseline frame index to: {baseline_idx_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
