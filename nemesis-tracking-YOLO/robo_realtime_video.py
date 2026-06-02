# robo_realtime_video_fixed.py
# FULL: ROBO/ROBO-BN realtime AVI inference + STRICT (<=1 per class) decoding
# + baseline-foreground gating to reject background/traps/blades
# + writes annotated output video + latency stats

import cv2
import time
import torch
import numpy as np

# ------------------------------------------------
# IMPORT YOUR MODEL DEFINITIONS (ROBO.py must exist)
# ------------------------------------------------
from ROBO import ROBO, ROBO_BN


# ------------------------------------------------
# CONFIG (EDIT THESE)
# ------------------------------------------------
VIDEO_PATH    = "in/fight3.mp4"                 # your AVI
CKPT_PATH     = "runs/robo_bn_real_ft/best.pt"    # stage-2 ckpt
BASELINE_PATH = "out/baseline/fight2_baseline.png"             # empty arena baseline

OUTPUT_VIDEO  = "out/robo_output_bbox_fixed.avi"
OUTPUT_FPS    = 30

IMG_W, IMG_H  = 512, 384

# Decoder / filtering
CONF_THRESH_LOW  = 0.25   # min conf to accept a class prediction
MIN_BOX_AREA     = 700    # reject tiny boxes
FG_DIFF_THRESH   = 50     # pixel diff threshold vs baseline (12~25)
FG_RATIO_THRESH  = 0.2  # foreground pixel ratio inside bbox (0.02~0.10)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------
# Baseline foreground gating helpers
# ------------------------------------------------
def build_foreground_mask(frame_bgr, baseline_bgr, diff_thresh=18, morph=True):
    fg = cv2.absdiff(frame_bgr, baseline_bgr)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(fg, diff_thresh, 255, cv2.THRESH_BINARY)

    if morph:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return fg


def load_model_auto(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Detect which architecture the checkpoint belongs to
    if any(k.startswith("stem.") for k in state.keys()):
        arch = "robo_bn"
        model = ROBO_BN(in_ch=3, out_ch=10).to(device)
    elif any(k.startswith("backbone.") for k in state.keys()):
        arch = "robo"
        model = ROBO(in_ch=3, out_ch=10).to(device)
    else:
        raise RuntimeError("Cannot infer model type from checkpoint keys.")

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[OK] Loaded checkpoint as {arch} on {device}")
    return model


def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W,     int(x2)))
    y2 = max(0, min(H,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def foreground_ratio_in_box(fg_mask, box):
    x1, y1, x2, y2 = box
    roi = fg_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())


# ------------------------------------------------
# STRICT ROBO decoder (one box per class per head)
# ------------------------------------------------
def decode_robo_strict_one_per_class(pred, stride, conf_thresh=0.30, min_area=400):
    """
    pred: (1,10,H,W)
    returns list of (cls, conf, x1,y1,x2,y2) with <=1 per class
    """
    B, C, H, W = pred.shape
    assert B == 1 and C == 10, f"Expected (1,10,H,W), got {pred.shape}"

    pred = pred.view(1, 2, 5, H, W)
    tx = torch.sigmoid(pred[:, :, 0])
    ty = torch.sigmoid(pred[:, :, 1])
    tw = pred[:, :, 2]
    th = pred[:, :, 3]
    to = torch.sigmoid(pred[:, :, 4])

    dets = []
    for cls in (0, 1):
        obj = to[0, cls]  # (H,W)
        conf, flat_idx = torch.max(obj.view(-1), dim=0)
        confv = conf.item()
        if confv < conf_thresh:
            continue

        y = int(flat_idx // W)
        x = int(flat_idx % W)

        cx = (x + tx[0, cls, y, x]) * stride
        cy = (y + ty[0, cls, y, x]) * stride
        w  = torch.exp(tw[0, cls, y, x]) * stride
        h  = torch.exp(th[0, cls, y, x]) * stride

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        if (x2 - x1) * (y2 - y1) < min_area:
            continue

        dets.append((cls, confv, x1, y1, x2, y2))
    return dets


# ------------------------------------------------
# Merge low+high heads: keep BEST per class total
# ------------------------------------------------
def merge_best_per_class(dets_list):
    """
    dets_list: list of detections (cls, conf, x1,y1,x2,y2)
    returns: list with <= 1 per class
    """
    best = {0: None, 1: None}
    for cls, conf, x1, y1, x2, y2 in dets_list:
        if best[cls] is None or conf > best[cls][0]:
            best[cls] = (conf, x1, y1, x2, y2)
    out = []
    for cls in (0, 1):
        if best[cls] is None:
            continue
        conf, x1, y1, x2, y2 = best[cls]
        out.append((cls, conf, x1, y1, x2, y2))
    return out


# ------------------------------------------------
# Load model
# ------------------------------------------------
model = model = load_model_auto(CKPT_PATH, DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print("Model loaded on:", DEVICE)


# ------------------------------------------------
# Load baseline
# ------------------------------------------------
baseline = cv2.imread(BASELINE_PATH, cv2.IMREAD_COLOR)
if baseline is None:
    raise FileNotFoundError(f"Cannot read baseline: {BASELINE_PATH}")
baseline = cv2.resize(baseline, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


# ------------------------------------------------
# Video IO
# ------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps > 120:
    fps = OUTPUT_FPS

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (IMG_W, IMG_H))
print(f"Output video: {OUTPUT_VIDEO} @ {fps:.1f} FPS")


# ------------------------------------------------
# Loop
# ------------------------------------------------
latencies = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.perf_counter()

    # resize to model input
    frame_resized = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    # foreground mask vs baseline (reject static arena)
    fg_mask = build_foreground_mask(frame_resized, baseline, diff_thresh=FG_DIFF_THRESH)

    # preprocess tensor
    x = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(DEVICE)

    # inference
    with torch.no_grad():
        out_low, out_high = model(x)

    # decode STRICT from each head
    dets = []
    dets += decode_robo_strict_one_per_class(out_low,  stride=64, conf_thresh=CONF_THRESH_LOW, min_area=MIN_BOX_AREA)
    dets += decode_robo_strict_one_per_class(out_high, stride=32, conf_thresh=CONF_THRESH_LOW, min_area=MIN_BOX_AREA)

    # merge heads -> <= 1 per class total
    dets = merge_best_per_class(dets)

    # baseline gating: reject boxes that are mostly static background
    filtered = []
    for cls, conf, x1, y1, x2, y2 in dets:
        box = clamp_box(x1, y1, x2, y2, IMG_W, IMG_H)
        if box is None:
            continue

        fg_ratio = foreground_ratio_in_box(fg_mask, box)
        if fg_ratio < FG_RATIO_THRESH:
            continue  # reject background/trap/blade-like detections

        filtered.append((cls, conf, *box))

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    latencies.append(latency_ms)

    # draw detections (<= 2 total)
    for cls, conf, x1, y1, x2, y2 in filtered:
        if cls == 0:
            color = (0, 255, 0)  # ours
            label = f"Ours {conf:.2f}"
        else:
            color = (0, 0, 255)  # theirs
            label = f"Theirs {conf:.2f}"

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_resized, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # overlay latency
    cv2.putText(frame_resized, f"Latency: {latency_ms:.1f} ms", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # (optional) debug show fg mask
    # cv2.imshow("FG Mask", fg_mask)

    # save + show
    writer.write(frame_resized)
    cv2.imshow("ROBO Fixed (2 boxes max)", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ------------------------------------------------
# Cleanup + latency report
# ------------------------------------------------
cap.release()
writer.release()
cv2.destroyAllWindows()

lat = np.array(latencies, dtype=np.float32)
print("\n=== Latency Report ===")
print(f"Mean : {lat.mean():.2f} ms")
print(f"P50  : {np.percentile(lat, 50):.2f} ms")
print(f"P90  : {np.percentile(lat, 90):.2f} ms")
print(f"P99  : {np.percentile(lat, 99):.2f} ms")
print(f"FPS  : {1000.0 / lat.mean():.2f}")
print(f"Saved: {OUTPUT_VIDEO}")
