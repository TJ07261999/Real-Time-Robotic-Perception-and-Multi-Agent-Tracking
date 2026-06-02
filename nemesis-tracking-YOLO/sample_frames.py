# sample_frames.py
import os, random, sys
from pathlib import Path
import cv2

# ====== CONFIG ======
INPUT_VIDEO   = "/Users/tsukasamiyaji/Desktop/Python3/Real Fights/fight2.avi"
START_FRAME   = 13400
N_SAMPLES     = 400
SEED          = 0  # change for a different random draw
OUT_ROOT      = Path("/Users/tsukasamiyaji/Desktop/Python3/Real Fights/hard")
IMG_DIR       = OUT_ROOT / "images"
LBL_DIR       = OUT_ROOT / "labels"
EXT           = "png"   # jpg or png

# ====== UTILS ======
def deterministic_name(frame_idx: int, ext: str) -> str:
    return f"f_{frame_idx:06d}.{ext}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ====== MAIN ======
def main():
    random.seed(SEED)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {INPUT_VIDEO}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        print("[ERR] Could not read total frame count.")
        sys.exit(1)

    # valid frame indices to sample from
    start = max(0, START_FRAME)
    end = total - 1
    if start > end:
        print(f"[ERR] START_FRAME {START_FRAME} is past end ({end}).")
        sys.exit(1)

    valid = list(range(start, end + 1))
    k = min(N_SAMPLES, len(valid))
    chosen = sorted(random.sample(valid, k))

    ensure_dir(IMG_DIR)
    ensure_dir(LBL_DIR)
    manifest_path = OUT_ROOT / "selected_frames.txt"

    saved = 0
    with open(manifest_path, "w") as mf:
        for idx in chosen:
            # seek to frame idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[warn] failed to read frame {idx}, skipping")
                continue

            fname = deterministic_name(idx, EXT)
            img_path = IMG_DIR / fname
            lbl_path = LBL_DIR / (Path(fname).stem + ".txt")

            # save image (overwrite if exists)
            if EXT.lower() == "jpg":
                cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(img_path), frame)

            # create empty YOLO label file (you will fill it later)
            lbl_path.write_text("")

            mf.write(f"{idx}\t{fname}\n")
            saved += 1

    cap.release()
    print(f"[done] total frames in video: {total}")
    print(f"[done] sampled: {saved} frames from [{start}..{end}]")
    print(f"[out] images -> {IMG_DIR}")
    print(f"[out] labels -> {LBL_DIR}")
    print(f"[out] manifest -> {manifest_path}")

if __name__ == "__main__":
    main()
