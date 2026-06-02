#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, random, shutil, glob, time, sys
from pathlib import Path
import numpy as np
import cv2

# ================= YOUR ORIGINAL CONFIG (unchanged) =================
INPUT_VIDEO = "/Users/tsukasamiyaji/Desktop/Python3/Real Fights/fight2.avi"
OUTPUT_VIDEO = "/Users/tsukasamiyaji/Desktop/Python3/Real Fights/fight2a.mp4"
OUTPUT_CSV = "/Users/tsukasamiyaji/Desktop/Python3/Real Fights/fight2a.csv"
START_FRAME = 13400

KEEP_POLY = np.array([[0,640],[710,753],[1436,663],[1145,46],[740,0],[354,38]], np.int32)
IGNORE_MASK_PATH = None

BG_HISTORY = 700
BG_THRESH   = 28
BG_SHADOWS  = False
BG_LR       = 0.001

USE_BASELINE = True
BASELINE_IMG = "out/baseline/fight2_baseline.png"
AUTO_BASELINE_FRAMES = 300
AUTO_BASELINE_STRIDE = 4
STARTUP_FREEZE_SEC = 5
ADAPT_K = 1.5

PERSIST_ALPHA  = 0.25
PERSIST_THRESH = 0.60

MIN_AREA = 1200
MAX_FRACTION_W = 0.55
MAX_FRACTION_H = 0.55
ASPECT_MIN = 0.35
ASPECT_MAX = 3.0
SOLIDITY_MIN = 0.70

MIN_HITS   = 3
MAX_AGE    = 12
IOU_ASSOC  = 0.30
SMOOTH_POS = 0.6
SMOOTH_SIZE= 0.4
KEEP_TOP_2 = True

# ================= NEW: HARVEST + TRAIN LOOP CONFIG =================
# Dataset layout
DATA_ROOT = Path("/Users/tsukasamiyaji/Desktop/Python3/Real Fights/dataset")
IM_TRAIN  = DATA_ROOT/"images/train"
LB_TRAIN  = DATA_ROOT/"labels/train"
IM_VAL    = DATA_ROOT/"images/val"
LB_VAL    = DATA_ROOT/"labels/val"
for p in [IM_TRAIN, LB_TRAIN, IM_VAL, LB_VAL]:
    p.mkdir(parents=True, exist_ok=True)

CLASS_ID = 0
SAVE_EVERY_N       = 1    # harvest every N frames with clean tracks
MIN_AREA_SAVE      = 3000
MAX_EDGE_TOUCH_PX  = 4

# YOLO training knobs
YOLO_BASE_WEIGHTS = "yolov10n.pt"   # light + fast
YOLO_IMGSZ_TRAIN  = 800             # train square size
YOLO_EPOCHS       = 25
YOLO_BATCH        = 10
YOLO_DEVICE       = "mps"           # Apple GPU
YOLO_WORKERS      = 6
YOLO_CONF_THR     = 0.25            # for prediction checks (not ONNX)
YOLO_ONNX_IMGSZ   = 800

# Iterative self-training
AUTO_ITERS            = 30           # how many train->export->fuse cycles
MIN_IMAGES_TO_TRAIN   = 100           # start training once you have >= this many
MIN_NEW_PER_ITER      = 50           # require at least this many new imgs per iter

# NEW: YOLO ONNX fusion in runtime
USE_YOLO_IN_FUSION    = True
YOLO_ONNX_PATH        = None         # will be set after first training/export
YOLO_IOU_NMS          = 0.5
YOLO_MAX_DETS         = 50

# ================== NEW: FLOW/EMA ACCURACY ADD-ONS ==================
FALLBACK_EMA_ALPHA   = 0.20
FALLBACK_EMA_THRESH  = 0.45
FLOW_THR_FRACTION    = 0.22
FLOW_KERNEL          = (5,5)
MERGE_IOU            = 0.15
MERGE_CENTER_DIST_FR = 0.08
SPLIT_ASPECT_THR     = 2.6
SPLIT_GAP_FR         = 0.06

# ========================= BASICS =========================
def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union

def bbox_center(x, y, w, h):
    return x + w / 2.0, y + h / 2.0

class StableTracker:
    class Trk:
        def __init__(self, tid, bbox):
            x, y, w, h = bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            self.id = tid
            self.cx, self.cy = cx, cy
            self.w, self.h = w, h
            self.vx, self.vy = 0.0, 0.0
            self.age = 0
            self.hits = 1
            self.hit_streak = 1
            self.confirmed = False
        def bbox(self):
            return [int(self.cx - self.w/2.0), int(self.cy - self.h/2.0), int(self.w), int(self.h)]
        def predict(self):
            if self.age > 0:
                self.cx += self.vx; self.cy += self.vy
        def update(self, det_bbox, a_pos, a_size):
            x, y, w, h = det_bbox
            mx, my = x + w/2.0, y + h/2.0
            vx_new, vy_new = mx - self.cx, my - self.cy
            self.cx = (1-a_pos)*self.cx + a_pos*mx
            self.cy = (1-a_pos)*self.cy + a_pos*my
            self.w  = (1-a_size)*self.w  + a_size*w
            self.h  = (1-a_size)*self.h  + a_size*h
            self.vx = 0.5*self.vx + 0.5*vx_new
            self.vy = 0.5*self.vy + 0.5*vy_new
            self.age = 0; self.hits += 1; self.hit_streak += 1
            if not self.confirmed and self.hit_streak >= MIN_HITS:
                self.confirmed = True

    def __init__(self, iou_thresh=0.3, max_age=12, min_hits=3, next_id=1, a_pos=0.6, a_size=0.4):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = next_id
        self.a_pos = a_pos
        self.a_size = a_size
        self.tracks = []

    def _pred_bbox(self, t):
        cx = t.cx + (t.vx if t.age > 0 else 0.0)
        cy = t.cy + (t.vy if t.age > 0 else 0.0)
        return [int(cx - t.w/2.0), int(cy - t.h/2.0), int(t.w), int(t.h)]

    def update(self, detections):
        for t in self.tracks:
            t.age += 1; t.predict()
        unmatched = list(range(len(detections)))
        for t in self.tracks:
            best, bi = 0.0, -1
            tb = self._pred_bbox(t)
            for di in unmatched:
                v = iou(tb, detections[di])
                if v > best: best, bi = v, di
            if best >= self.iou_thresh and bi >= 0:
                t.update(detections[bi], self.a_pos, self.a_size)
                unmatched.remove(bi)
            else:
                t.hit_streak = 0
        for di in unmatched:
            self.tracks.append(self.Trk(self.next_id, detections[di])); self.next_id += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        out=[]
        for t in self.tracks:
            if t.confirmed: out.append((t.id, t.bbox()))
        return out

def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hist_match_linear(src, ref, mask=None):
    s = src.astype(np.float32); r = ref.astype(np.float32)
    if mask is None: mask = np.ones_like(s, np.uint8)
    m = mask > 0
    if not np.any(m): return src
    s_m, r_m = s[m], r[m]
    a = np.sqrt((r_m.var()+1e-6)/(s_m.var()+1e-6))
    b = r_m.mean() - a*s_m.mean()
    out = a*s + b
    return np.clip(out, 0, 255).astype(np.uint8)

def build_region_masks(shape, keep_poly, ignore_mask_path):
    h, w = shape[:2]
    keep = np.zeros((h, w), np.uint8)
    cv2.fillPoly(keep, [keep_poly], 255)
    if ignore_mask_path and os.path.exists(ignore_mask_path):
        ign = cv2.imread(ignore_mask_path, cv2.IMREAD_GRAYSCALE)
        ign = cv2.resize(ign, (w, h), interpolation=cv2.INTER_NEAREST)
        keep = cv2.bitwise_and(keep, cv2.bitwise_not((ign>0).astype(np.uint8)*255))
    return keep

def build_temporal_median(cap, frames=300, stride=3, start_pos=None, keep_mask=None):
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if start_pos is not None: cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
    xs=[]; t=0
    while t<frames:
        ok, f = cap.read()
        
        if not ok: break
        g = to_gray(f)

        if keep_mask is not None: g = cv2.bitwise_and(g,g,mask=keep_mask)
        xs.append(g)
        cur = cap.get(cv2.CAP_PROP_POS_FRAMES); cap.set(cv2.CAP_PROP_POS_FRAMES, cur+stride)
        t+=1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return None if not xs else np.median(np.stack(xs,0),0).astype(np.uint8)

def make_baseline(cap, keep_mask):
    if BASELINE_IMG and os.path.exists(BASELINE_IMG):
        bgr = cv2.imread(BASELINE_IMG)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return to_gray(cv2.resize(bgr,(w,h)))
    return build_temporal_median(cap, AUTO_BASELINE_FRAMES, AUTO_BASELINE_STRIDE,
                                 max(0,int(cap.get(cv2.CAP_PROP_POS_FRAMES))-600), keep_mask)

def combine_masks_with_persistence(diff_gray, base_gray, mog2_mask, keep_mask, ema_state):
    adj = hist_match_linear(diff_gray, base_gray, mask=keep_mask)
    d = cv2.absdiff(adj, base_gray)
    km = keep_mask>0; mu=float(d[km].mean()) if np.any(km) else float(d.mean())
    sd = float(d[km].std()) if np.any(km) else float(d.std())
    thr = mu + ADAPT_K*sd
    _, diff_bin = cv2.threshold(d, max(15,thr), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN,k,1)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE,k,1)
    comb = cv2.bitwise_and(diff_bin, mog2_mask)
    comb = cv2.bitwise_and(comb, keep_mask)
    comb_f = (comb>0).astype(np.float32)
    ema_state[:] = (1.0-PERSIST_ALPHA)*ema_state + PERSIST_ALPHA*comb_f
    final = (ema_state >= PERSIST_THRESH).astype(np.uint8) * 255
    return final

def filter_by_shape(contours, frame_w, frame_h):
    boxes=[]
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        if w>frame_w*MAX_FRACTION_W or h>frame_h*MAX_FRACTION_H: continue
        ar = w/max(1.0,h)
        if not(ASPECT_MIN<=ar<=ASPECT_MAX): continue
        hull=cv2.convexHull(c); ha=max(1.0, cv2.contourArea(hull))
        if area/ha < SOLIDITY_MIN: continue
        boxes.append([x,y,w,h])
    return boxes

# --------- ACCURACY ADD-ONS (same as your last script) ----------
def flow_mag(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.1,0)
    mag  = cv2.magnitude(flow[...,0], flow[...,1])
    mag  = cv2.GaussianBlur(mag, FLOW_KERNEL, 0)
    mmax = float(mag.max()) if mag.size else 0.0
    thr  = FLOW_THR_FRACTION * (mmax + 1e-6)
    mask = (mag >= thr).astype(np.uint8) * 255
    return mask

def fallback_enhance_mask(mog2_mask, flow_mask, keep_mask, ema_fb_state):
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    binm = cv2.morphologyEx(mog2_mask, cv2.MORPH_OPEN,k5,1)
    binm = cv2.morphologyEx(binm,    cv2.MORPH_CLOSE,k5,1)
    comb_f = (binm>0).astype(np.float32)
    ema_fb_state[:] = (1.0-FALLBACK_EMA_ALPHA)*ema_fb_state + FALLBACK_EMA_ALPHA*comb_f
    stable = (ema_fb_state >= FALLBACK_EMA_THRESH).astype(np.uint8) * 255
    flow_and = cv2.bitwise_and(binm, flow_mask)
    fused = cv2.bitwise_or(stable, flow_and)
    fused = cv2.bitwise_and(fused, keep_mask)
    return fused

def merge_close_boxes(boxes, W, H):
    if not boxes: return []
    diag = (W**2 + H**2) ** 0.5
    used=[False]*len(boxes); out=[]
    for i,a in enumerate(boxes):
        if used[i]: continue
        ax,ay,aw,ah = a
        acx,acy = ax+aw/2, ay+ah/2
        mx,my,mx2,my2 = ax,ay,ax+aw,ay+ah
        used[i]=True
        for j,b in enumerate(boxes):
            if used[j] or i==j: continue
            bx,by,bw,bh = b
            bcx,bcy = bx+bw/2, by+bh/2
            cdist = ((acx-bcx)**2 + (acy-bcy)**2)**0.5
            if iou(a,b) >= MERGE_IOU or (cdist <= MERGE_CENTER_DIST_FR*diag):
                used[j]=True
                mx=min(mx,bx); my=min(my,by); mx2=max(mx2,bx+bw); my2=max(my2,by+bh)
        out.append([int(mx),int(my),int(mx2-mx),int(my2-my)])
    return out

def split_wide_boxes(boxes):
    out=[]
    for (x,y,w,h) in boxes:
        ar = w/max(1.0,h)
        if ar >= SPLIT_ASPECT_THR:
            gap = max(2, int(w*SPLIT_GAP_FR))
            w2  = (w-gap)//2
            out += [[x,y,w2,h],[x+w2+gap,y,w-(w2+gap),h]]
        else:
            out.append([x,y,w,h])
    return out

# ==================== NEW: YOLO INFERENCE (ONNX) ====================
class YOLOv10_ONNX:
    def __init__(self, model_path, img_size=640, conf_thr=0.25, iou_thr=0.5, max_det=50):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.img_size = img_size
        self.conf_thr = conf_thr
        self.iou_thr  = iou_thr
        self.max_det  = max_det

    @staticmethod
    def letterbox(im, new_shape=640, color=(114,114,114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
        dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
        dw, dh = dw/2, dh/2
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))
        im = cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    @staticmethod
    def nms(boxes, scores, iou_thr, max_det):
        idxs = cv2.dnn.NMSBoxes(
            [list(map(int,[x,y,w,h])) for x,y,w,h in boxes],
            list(map(float, scores)), 0.0, iou_thr)
        idxs = idxs.flatten().tolist() if len(idxs)>0 else []
        return idxs[:max_det]

    def infer(self, bgr):
        ih, iw = bgr.shape[:2]
        img, r, (dw, dh) = self.letterbox(bgr, self.img_size)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        blob = np.transpose(blob,(2,0,1))[None,...]
        out = self.sess.run(None, {self.inp: blob})
        pred = out[0][0] if out[0].ndim==3 else out[0]
        boxes, scores = [], []
        for row in pred:
            x,y,w,h,conf = row[:5]
            if conf < self.conf_thr: continue
            if row.shape[0]>5:
                cls_conf=row[5:].max(); conf=float(conf*cls_conf)
                if conf < self.conf_thr: continue
            cx,cy=x,y
            x1=(cx-w/2 - dw)/r; y1=(cy-h/2 - dh)/r
            x2=(cx+w/2 - dw)/r; y2=(cy+h/2 - dh)/r
            x1=max(0,x1); y1=max(0,y1); x2=min(iw-1,x2); y2=min(ih-1,y2)
            bw=max(1,x2-x1); bh=max(1,y2-y1)
            boxes.append([int(x1),int(y1),int(bw),int(bh)]); scores.append(float(conf))
        if not boxes: return []
        keep=self.nms(boxes,scores,self.iou_thr,self.max_det)
        return [boxes[i] for i in keep]

# ===================== NEW: DATASET HELPERS =====================
def touches_edge(x,y,w,h,W,H,margin):
    return (x<=margin) or (y<=margin) or (x+w>=W-margin) or (y+h>=H-margin)

def save_yolo_pair(frame, boxes, frame_idx, W, H, class_id=0, out_images=IM_TRAIN, out_labels=LB_TRAIN):
    img_path = out_images/f"f_{frame_idx:06d}.jpg"
    txt_path = out_labels/f"f_{frame_idx:06d}.txt"
    cv2.imwrite(str(img_path), frame)
    lines=[]
    for (x,y,w,h) in boxes:
        cx=(x+w/2)/W; cy=(y+h/2)/H; nw=w/W; nh=h/H
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    if lines:
        with open(txt_path,"w") as f: f.write("\n".join(lines))

def count_images(p): return len(list(Path(p).glob("*.jpg")))

def write_data_yaml():
    yml = DATA_ROOT/"data.yaml"
    yml.write_text(f"""path: {DATA_ROOT}
train: images/train
val: images/val
nc: 1
names: ["robot"]
""")
    return yml

def make_val_split(val_ratio=0.15):
    imgs = sorted(glob.glob(str(IM_TRAIN/"*.jpg")))
    if not imgs: return 0
    random.seed(0)
    want = max(1, int(len(imgs)*val_ratio))
    picked = set(random.sample(imgs, want))
    moved=0
    for img in imgs:
        if img not in picked: continue
        stem = Path(img).stem
        lbl  = LB_TRAIN/f"{stem}.txt"
        if Path(img).exists() and lbl.exists():
            shutil.move(img, IM_VAL/f"{stem}.jpg")
            shutil.move(lbl, LB_VAL/f"{stem}.txt")
            moved += 1
    return moved

# ===================== NEW: TRAIN & EXPORT =====================
def train_yolo(iter_idx: int):
    """
    Train YOLO for this iteration, export ONNX, archive artifacts, and
    return (onnx_path, run_dir).
    """
    import pandas as pd # type: ignore
    from datetime import datetime
    from ultralytics import YOLO # type: ignore


    data_yaml = write_data_yaml()
    # create val split if needed
    if count_images(IM_VAL) == 0 and count_images(IM_TRAIN) > 10:
        make_val_split(0.15)

    # Name the run by iteration and keep everything inside DATA_ROOT/runs
    project_dir = DATA_ROOT / "runs"
    run_name = f"iter_{iter_idx:02d}"
    model = YOLO(YOLO_BASE_WEIGHTS)

    results = model.train(
        data=str(data_yaml),
        imgsz=YOLO_IMGSZ_TRAIN,
        epochs=YOLO_EPOCHS,
        batch=YOLO_BATCH,
        device=YOLO_DEVICE,
        workers=YOLO_WORKERS,
        mosaic=0.8, flipud=0.1, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)              # e.g. DATA_ROOT/runs/iter_01
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"

    # Export the freshly trained best weights to ONNX
    # (Ultralytics will load the best automatically when calling export on 'model')
    model = YOLO(str(best_pt))
    onnx_path = model.export(format="onnx", opset=12, imgsz=YOLO_ONNX_IMGSZ, dynamic=False)
    onnx_path = Path(onnx_path) if isinstance(onnx_path, (str, Path)) else (run_dir / "weights" / "best.onnx")

    # Copy/rename best artifacts for convenience
    archive_dir = DATA_ROOT / "archive" / run_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    # Keep a copy of best.pt and best.onnx with iteration in the name
    shutil.copy2(best_pt, archive_dir / f"best_{run_name}.pt")
    if onnx_path.exists():
        shutil.copy2(onnx_path, archive_dir / f"best_{run_name}.onnx")

    # Append final metrics to a global CSV
    res_csv = run_dir / "results.csv"  # Ultralytics writes this every epoch
    metrics_log = DATA_ROOT / "metrics_log.csv"
    if res_csv.exists():
        df = pd.read_csv(res_csv)
        # Take the last epoch row
        last = df.iloc[-1].to_dict()
        # standard cols include: train/val losses, precision (P), recall (R), mAP50, mAP50-95, etc.
        row = {
            "iter": iter_idx,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "best_pt": str(archive_dir / f"best_{run_name}.pt"),
            "best_onnx": str(archive_dir / f"best_{run_name}.onnx"),
            # pull common metrics if present
            "R": last.get("metrics/recall", last.get("recall", None)),
            "mAP50": last.get("metrics/mAP50(B)", last.get("metrics/mAP50", last.get("map50", None))),
            "mAP50-95": last.get("metrics/mAP50-95(B)", last.get("metrics/mAP50-95", last.get("map", None))),
            "box_loss": last.get("train/box_loss", None),
            "cls_loss": last.get("train/cls_loss", None),
            "dfl_loss": last.get("train/dfl_loss", None),
            "val_box_loss": last.get("val/box_loss", None),
            "val_cls_loss": last.get("val/cls_loss", None),
            "val_dfl_loss": last.get("val/dfl_loss", None),
        }
        import csv as _csv
        write_header = not metrics_log.exists()
        with open(metrics_log, "a", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    return str(onnx_path), str(run_dir)


# ===================== ONE PASS (RUN + HARVEST) =====================
def run_one_pass(use_yolo_onnx=None, save_overlay=True):
    ensure_parent(OUTPUT_VIDEO); ensure_parent(OUTPUT_CSV)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open {INPUT_VIDEO}")
    if START_FRAME>0: cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    keep_mask = build_region_masks((H,W,3), KEEP_POLY, IGNORE_MASK_PATH)
    bgs = cv2.createBackgroundSubtractorMOG2(BG_HISTORY, BG_THRESH, BG_SHADOWS)
    baseline=None
    if USE_BASELINE:
        baseline = make_baseline(cap, keep_mask)
        if baseline is None: print("[warn] no baseline; MOG2-only path.")
    ema_state = np.zeros((H,W), np.float32)
    ema_fb_state = np.zeros((H,W), np.float32)
    prev_gray=None
    tracker = StableTracker(IOU_ASSOC, MAX_AGE, MIN_HITS, 1, SMOOTH_POS, SMOOTH_SIZE)
    csv_f=open(OUTPUT_CSV,"w",newline=""); csv_w=csv.writer(csv_f)
    csv_w.writerow(["frame","time_sec","id","cx_img","cy_img","x","y","w","h"])
    yolo=None
    if use_yolo_onnx and USE_YOLO_IN_FUSION:
        try:
            yolo = YOLOv10_ONNX(use_yolo_onnx, img_size=YOLO_ONNX_IMGSZ,
                                conf_thr=YOLO_CONF_THR, iou_thr=YOLO_IOU_NMS, max_det=YOLO_MAX_DETS)
            print(f"[yolo] ONNX loaded: {use_yolo_onnx}")
        except Exception as e:
            print(f"[yolo] failed to load: {e}")

    harvested=0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = to_gray(frame)
        if prev_gray is None: prev_gray = gray.copy()
        t_sec = frame_idx / fps
        lr = 0.0 if t_sec < STARTUP_FREEZE_SEC else BG_LR
        masked_gray = cv2.bitwise_and(gray, gray, mask=keep_mask)
        mog2 = bgs.apply(masked_gray, learningRate=lr)
        mog2 = cv2.threshold(mog2, 200, 255, cv2.THRESH_BINARY)[1]

        if USE_BASELINE and baseline is not None:
            binm = combine_masks_with_persistence(gray, baseline, mog2, keep_mask, ema_state)
        else:
            fmask = flow_mag(prev_gray, gray)
            binm = fallback_enhance_mask(mog2, fmask, keep_mask, ema_fb_state)

        contours,_ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = filter_by_shape(contours, W, H)
        if dets:
            dets = merge_close_boxes(dets, W, H)
            dets = split_wide_boxes(dets)

        # YOLO fusion (union)
        if yolo is not None:
            try:
                ydets = yolo.infer(frame)
                dets  = (dets or []) + ydets
            except Exception as e:
                print(f"[yolo] infer error: {e}")

        tracks = tracker.update(dets or [])
        if KEEP_TOP_2 and len(tracks)>2:
            tracks = sorted(tracks, key=lambda t: t[1][2]*t[1][3], reverse=True)[:2]

        # HARVEST (train split only in this pass)
        boxes=[]
        for tid,(x,y,w,h) in tracks:
            if w*h < MIN_AREA_SAVE: continue
            if touches_edge(x,y,w,h,W,H,MAX_EDGE_TOUCH_PX): continue
            ar = w/max(1.0,h)
            if not(0.25<=ar<=4.0): continue
            boxes.append([x,y,w,h])
        if boxes and frame_idx % SAVE_EVERY_N == 0:
            save_yolo_pair(frame, boxes, frame_idx, W, H, class_id=CLASS_ID, out_images=IM_TRAIN, out_labels=LB_TRAIN)
            harvested += 1

        # draw & log
        overlay = frame.copy()
        qa = cv2.cvtColor(cv2.resize(binm,(320,180)), cv2.COLOR_GRAY2BGR)
        overlay[0:180,0:320]=qa
        for tid,(x,y,w,h) in tracks:
            cx,cy = bbox_center(x,y,w,h)
            csv_w.writerow([frame_idx,f"{t_sec:.3f}",tid,f"{cx:.1f}",f"{cy:.1f}",x,y,w,h])
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(overlay,f"ID {tid}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.circle(overlay,(int(cx),int(cy)),3,(0,255,255),-1)
        cv2.putText(overlay,f"frame {frame_idx}  t={t_sec:0.2f}s  tracks={len(tracks)}",
                    (10,H-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        if save_overlay:
            writer.write(overlay)
            cv2.imshow("tracking+harvest", overlay)
            if cv2.waitKey(1)&0xFF==27: break

        prev_gray = gray.copy()
        frame_idx += 1

    cap.release(); writer.release(); csv_f.close(); cv2.destroyAllWindows()
    return harvested

# =========================== MAIN LOOP ===========================
def main():
    print("=== pass 0: classical only, harvest ===")
    harvested = run_one_pass(use_yolo_onnx=None)
    print(f"[harvest] pass0 new frames: {harvested}")

    # train if enough data
    trained_any = False
    for it in range(1, AUTO_ITERS+1):
        n_train = count_images(IM_TRAIN)
        if n_train < MIN_IMAGES_TO_TRAIN:
            print(f"[skip] not enough images to train yet: {n_train}/{MIN_IMAGES_TO_TRAIN}")
            break
        print(f"=== training iter {it} (images/train={n_train}) ===")

        onnx_path, run_dir = train_yolo(it)
        global YOLO_ONNX_PATH
        YOLO_ONNX_PATH = onnx_path
        print(f"[export] iter {it}: ONNX at {onnx_path}")
        print(f"[logs]   iter {it}: run dir {run_dir}")

        trained_any = True

        print(f"=== pass {it}: yolo+classical fusion, harvest more ===")
        harvested = run_one_pass(use_yolo_onnx=onnx_path)
        print(f"[harvest] pass{it} new frames: {harvested}")
        if harvested < MIN_NEW_PER_ITER:
            print(f"[stop] not enough new data this iter ({harvested}<{MIN_NEW_PER_ITER})")
            break

    print("[done] self-training pipeline finished.")
    if trained_any and YOLO_ONNX_PATH:
        print(f"[final] YOLO ONNX ready at: {YOLO_ONNX_PATH}")
    else:
        print("[final] No YOLO model was trained (not enough images).")

if __name__ == "__main__":
    main()
