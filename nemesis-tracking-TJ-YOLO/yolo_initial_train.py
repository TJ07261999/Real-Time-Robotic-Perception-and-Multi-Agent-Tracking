import os, random, shutil
from pathlib import Path
from typing import List
import argparse
import sys

# This model is trained by initial dataset that I make bounding box by my hand. 


# -------- utility --------
def list_images(root: Path, exts: List[str]) -> List[Path]:
    files = []
    for e in exts:
        files += list(root.glob(f"*.{e}"))
    return sorted(files)

def stem(p: Path) -> str:
    return p.stem

def symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)

def write_text_utf8(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def verify_two_class_labels(txt: Path) -> bool:
    """
    Returns True if label file exists and contains only class ids 0 or 1.
    Empty file is OK (will be considered unlabeled and skipped for training).
    """
    if not txt.exists():
        return False
    ok_any = False
    for line in txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            # malformed -> reject
            return False
        try:
            cid = int(float(parts[0]))
        except Exception:
            return False
        if cid not in (0, 1):
            return False
        ok_any = True
    return ok_any  # must have at least one valid box

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str,
                    help="Your labeled images folder (PNG/JPG).")
    ap.add_argument("--labels_dir", required=True, type=str,
                    help="Your YOLO .txt labels folder (same stems).")
    ap.add_argument("--out_root", type=str, default="yolo_dataset",
                    help="Where to build YOLO dataset structure.")
    ap.add_argument("--exts", type=str, default="png",
                    help="Comma-separated image extensions. Default=png")
    ap.add_argument("--val_ratio", type=float, default=0.25,
                    help="Validation split ratio. Default=0.10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", type=str, default="auto",
                    help="e.g. '0' for GPU, 'cpu' for CPU, 'auto' to let YOLO pick")
    ap.add_argument("--export_onnx", action="store_true",
                    help="Export best.pt to ONNX after training")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser()
    labels_dir = Path(args.labels_dir).expanduser()
    out_root   = Path(args.out_root).expanduser().resolve()

    # dataset target structure
    im_train = out_root / "images" / "train"
    im_val   = out_root / "images" / "val"
    lb_train = out_root / "labels" / "train"
    lb_val   = out_root / "labels" / "val"

    for p in [im_train, im_val, lb_train, lb_val]:
        p.mkdir(parents=True, exist_ok=True)

    # collect images
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    imgs = list_images(images_dir, exts)
    if len(imgs) == 0:
        print(f"[error] No images with extensions {exts} in {images_dir}")
        sys.exit(1)

    # pair with labels and keep only samples that have valid 0/1 labels
    pairs = []
    skipped = 0
    for img in imgs:
        lbl = labels_dir / f"{img.stem}.txt"
        if verify_two_class_labels(lbl):
            pairs.append((img, lbl))
        else:
            skipped += 1
    if len(pairs) == 0:
        print("[error] Found no valid labeled images (class ids must be 0 or 1, at least one box).")
        sys.exit(1)

    print(f"[info] total images: {len(imgs)} | usable labeled: {len(pairs)} | skipped: {skipped}")

    # split
    random.seed(args.seed)
    random.shuffle(pairs)
    n_total = len(pairs)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # link/copy
    for img, lbl in train_pairs:
        symlink_or_copy(img, im_train / img.name)
        symlink_or_copy(lbl, lb_train / f"{img.stem}.txt")
    for img, lbl in val_pairs:
        symlink_or_copy(img, im_val / img.name)
        symlink_or_copy(lbl, lb_val / f"{img.stem}.txt")

    # write data.yaml
    data_yaml = out_root / "data.yaml"
    yaml_text = f"""# auto-generated
path: {out_root}
train: images/train
val: images/val
nc: 2
names: ["our_robot", "their_robot"]
"""
    write_text_utf8(data_yaml, yaml_text)
    print(f"[ok] dataset ready at: {out_root}")
    print(f"[ok] data.yaml: {data_yaml}")

    # ---- train YOLO11s via Python API (no subprocess) ----
    try:
        from ultralytics import YOLO # type: ignore
    except Exception:
        print("[error] Ultralytics not installed in this Python. Run: pip install ultralytics")
        sys.exit(1)

    print("[info] starting YOLO11s training via Python API...")
    model = YOLO("yolo11s.pt")  # will download if not present

    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=8,
        project=str(out_root / "runs"),
        name="robot_det",
        exist_ok=True,
        # augmentations
        mosaic=0.8,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # determinism
        seed=0,
        deterministic=True,
    )

    # ---- optional export to ONNX via Python API ----
    if args.export_onnx:
        # find best.pt from last run
        runs_dir = out_root / "runs" / "robot_det"
        weights = runs_dir / "weights" / "best.pt"
        if not weights.exists():
            # try to find any best.pt under runs
            cands = list((out_root / "runs").glob("**/best.pt"))
            if cands:
                weights = cands[-1]

        if weights.exists():
            print(f"[export] exporting ONNX from: {weights}")
            exp_model = YOLO(str(weights))
            exp_model.export(
                format="onnx",
                opset=12,
                imgsz=args.imgsz,
                dynamic=False,
            )
        else:
            print("[warn] best.pt not found; skip ONNX export.")

if __name__ == "__main__":
    main()
