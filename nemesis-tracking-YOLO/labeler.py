import argparse, os, glob, math
from pathlib import Path
import cv2
import numpy as np


# This code is for making bounding box by hand each frame from the avi or mp4 video.
# ------------------------- Helpers -------------------------

def yolo_to_xyxy(cx, cy, w, h, W, H):
    x = (cx - w/2.0) * W
    y = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return [int(round(x)), int(round(y)), int(round(x2)), int(round(y2))]

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1, x2 = max(0, min(W-1, x1)), max(0, min(W-1, x2))
    y1, y2 = max(0, min(H-1, y1)), max(0, min(H-1, y2))
    w = max(1.0, abs(x2 - x1))
    h = max(1.0, abs(y2 - y1))
    cx = (min(x1, x2) + w/2.0) / W
    cy = (min(y1, y2) + h/2.0) / H
    nw = w / W
    nh = h / H
    return cx, cy, nw, nh

def load_labels(txt_path, W, H):
    boxes = []
    if not txt_path.exists():
        return boxes
    try:
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 5: 
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
            boxes.append([cls, x1, y1, x2, y2])
    except Exception:
        pass
    return boxes

def save_labels(txt_path, boxes, W, H):
    lines = []
    for (cls, x1, y1, x2, y2) in boxes:
        cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

def draw_boxes(img, boxes, classes, cur_class, mouse_xy=None):
    out = img.copy()
    H, W = out.shape[:2]
    # colors per class
    palette = [
        (50, 220, 50),     # class 0: our_robot
        (60, 140, 255),    # class 1: their_robot
        (220, 60, 60),
        (200, 200, 60),
    ]
    for (cls, x1, y1, x2, y2) in boxes:
        c = palette[cls % len(palette)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"{cls}:{classes[cls] if cls < len(classes) else 'cls'}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        xx1, yy1 = x1, max(0, y1 - th - 6)
        cv2.rectangle(out, (xx1, yy1), (xx1 + tw + 6, yy1 + th + 4), c, -1)
        cv2.putText(out, label, (xx1 + 3, yy1 + th + 0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)

    header = f"[{cur_class}] {classes[cur_class] if cur_class < len(classes) else 'class'} | " \
             f"0..9 switch, TAB cycle | U undo | D del | S save | SPACE next | P/N prev/next | Q quit"
    cv2.rectangle(out, (0,0), (out.shape[1], 26), (32,32,32), -1)
    cv2.putText(out, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    if mouse_xy is not None:
        mx, my = mouse_xy
        cv2.drawMarker(out, (mx, my), (255,255,255), cv2.MARKER_CROSS, 14, 1)
    return out

def box_under_cursor(boxes, x, y):
    for i in range(len(boxes)-1, -1, -1):
        _, x1, y1, x2, y2 = boxes[i]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return -1

# ------------------------- Main UI -------------------------

class Labeler:
    def __init__(self, images_dir, labels_dir, exts, classes, start_idx=0):
        # Gather images for all extensions, case-insensitive
        p = Path(images_dir)
        imgs = []
        for ext in exts:
            imgs += list(p.glob(f"*.{ext}"))
            imgs += list(p.glob(f"*.{ext.upper()}"))
        self.images = sorted(imgs, key=lambda x: x.name)

        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
        self.cur = max(0, min(start_idx, len(self.images)-1))
        self.cur_class = 0
        self.dragging = False
        self.x0 = self.y0 = 0
        self.mouse_xy = (0, 0)
        self.boxes = []  # list of [cls, x1, y1, x2, y2]
        self.win = "Labeler"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.load_current()

    def current_paths(self):
        if not self.images: return None, None
        img_path = Path(self.images[self.cur])
        txt_path = self.labels_dir / (img_path.stem + ".txt")
        return img_path, txt_path

    def load_current(self):
        img_path, txt_path = self.current_paths()
        if img_path is None:
            raise RuntimeError("No images found.")
        self.img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        H, W = self.img.shape[:2]
        self.boxes = load_labels(txt_path, W, H)

    def save_current(self):
        img_path, txt_path = self.current_paths()
        H, W = self.img.shape[:2]
        save_labels(txt_path, self.boxes, W, H)
        print(f"[saved] {txt_path}")

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_xy = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.x0, self.y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            pass
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.x0, self.y0
            x2, y2 = x, y
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                H, W = self.img.shape[:2]
                x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
                self.boxes.append([self.cur_class, x1, y1, x2, y2])

    def run(self):
        while True:
            disp = draw_boxes(self.img, self.boxes, self.classes, self.cur_class, self.mouse_xy)
            if self.dragging:
                x1, y1 = self.x0, self.y0
                x2, y2 = self.mouse_xy
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255,255,255), 1)
            cv2.imshow(self.win, disp)
            k = cv2.waitKey(15) & 0xFFFF

            if k == 255:
                continue

            if ord('0') <= k <= ord('9'):
                cid = k - ord('0')
                if cid < len(self.classes):
                    self.cur_class = cid
                continue

            if k == 9:  # TAB
                self.cur_class = (self.cur_class + 1) % len(self.classes)
                continue

            if k in (ord('u'), ord('U')):
                if self.boxes:
                    self.boxes.pop()
                continue

            if k in (ord('d'), ord('D')):
                i = box_under_cursor(self.boxes, *self.mouse_xy)
                if i >= 0:
                    self.boxes.pop(i)
                continue

            if k in (ord('s'), ord('S')):
                self.save_current()
                continue

            if k in (13, 32):  # ENTER/SPACE
                self.save_current()
                if self.cur < len(self.images) - 1:
                    self.cur += 1
                    self.load_current()
                continue

            if k in (ord('n'), ord('N')):
                self.save_current()
                if self.cur < len(self.images) - 1:
                    self.cur += 1
                    self.load_current()
                continue

            if k in (ord('p'), ord('P')):
                self.save_current()
                if self.cur > 0:
                    self.cur -= 1
                    self.load_current()
                continue

            if k in (ord('r'), ord('R')):
                self.load_current()
                continue

            if k in (27, ord('q'), ord('Q')):
                self.save_current()
                break

        cv2.destroyAllWindows()

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images_dir",
        required=False,
        default="/Users/tsukasamiyaji/Desktop/Python3/Real Fights/hard/images",
        type=str)
    ap.add_argument(
        "--labels_dir",
        required=False,
        default="/Users/tsukasamiyaji/Desktop/Python3/Real Fights/hard/labels",
        type=str)
    # Backwards-compatible single-ext flag
    ap.add_argument("--ext", default=None, type=str,
                    help="(Deprecated) single extension like 'png' or 'jpg'.")
    # Preferred multiple-exts flag (comma-separated)
    ap.add_argument("--exts", default="png", type=str,
                    help="Comma-separated list of extensions (e.g., 'png' or 'jpg,jpeg,png').")
    ap.add_argument("--classes", default="our_robot,their_robot",
                    help="Comma-separated class names; order defines IDs.")
    ap.add_argument("--start", type=int, default=0, help="Start index into image list.")
    args = ap.parse_args()

    # Resolve extensions
    if args.ext is not None and args.exts:
        # If both given, prioritize --exts but allow --ext alone
        pass
    exts = []
    if args.ext:
        exts = [args.ext.strip().lower()]
    else:
        exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    if not exts:
        exts = ["png"]  # safe default

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(classes) == 0:
        classes = ["class0", "class1"]

    lbl = Labeler(args.images_dir, args.labels_dir, exts, classes, start_idx=args.start)
    if len(lbl.images) == 0:
        raise RuntimeError(f"No images with extensions {exts} found in {args.images_dir}")
    print(f"[info] {len(lbl.images)} images | exts={exts} | classes={classes} (IDs: 0..{len(classes)-1})")
    lbl.run()

if __name__ == "__main__":
    main()
