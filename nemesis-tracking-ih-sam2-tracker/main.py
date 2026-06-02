import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# -------------------------
# Config
# -------------------------
VIDEO_PATH = "data/fight5-sample.mp4"  # can be .mp4 OR a frames directory
CHECKPOINT = "ckpts/sam2.1_hiera_tiny.pt"  # change to your downloaded ckpt
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # config that matches the checkpoint
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

OUT_VIDEO = "sample.mp4"
ALPHA = 0.45

OBJ_COLORS = {
    1: (0, 255, 255),  # HOME = yellow
    2: (255, 0, 255),  # OPPONENT = magenta
}
OBJ_NAMES = {
    1: "NEMESIS",
    2: "OPPONENT",
}

# -------------------------
# Prompt UI
# -------------------------
# Left click = positive, Right click = negative
# Press 'n' to finish current object, start next
# Press 'q' to begin tracking
clicks = []  # list of dicts: {"points": [(x,y),...], "labels":[1/0,...]}

cur_points, cur_labels = [], []


def on_mouse(event, x, y, flags, param):
    global cur_points, cur_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        cur_points.append((x, y))
        cur_labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cur_points.append((x, y))
        cur_labels.append(0)


def draw_points(img, points, labels):
    out = img.copy()
    for (x, y), lab in zip(points, labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(out, (x, y), 5, color, -1)
    return out


def overlay_mask(frame_bgr, mask_bool, color_bgr, alpha=0.45):
    out = frame_bgr.copy()
    color = np.array(color_bgr, dtype=np.uint8)
    out[mask_bool] = (out[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def mask_centroid(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def draw_label(frame, text, xy, color_bgr):
    x, y = xy
    # Offset so text doesn't sit exactly on centroid
    x, y = x + 10, y - 10

    # Background box for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x - 3, y - th - 3),
        (x + tw + 3, y + baseline + 3),
        color_bgr,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


# -------------------------
# Load first frame
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ok, first = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Could not read first frame from: {VIDEO_PATH}")

cv2.namedWindow("Prompt robots (L=pos, R=neg, n=next obj, q=track)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Prompt robots (L=pos, R=neg, n=next obj, q=track)", on_mouse)

while True:
    vis = first.copy()
    vis = draw_points(vis, cur_points, cur_labels)

    # show already-finished objects' points too
    for obj in clicks:
        vis = draw_points(vis, obj["points"], obj["labels"])

    cv2.imshow("Prompt robots (L=pos, R=neg, n=next obj, q=track)", vis)
    k = cv2.waitKey(20) & 0xFF

    if k == ord("n"):
        if len(cur_points) == 0:
            print("No points for this object; add at least one positive point.")
            continue
        clicks.append({"points": cur_points, "labels": cur_labels})
        cur_points, cur_labels = [], []
        print(f"Saved object {len(clicks)} prompt.")
    elif k == ord("q"):
        if len(cur_points) > 0:
            clicks.append({"points": cur_points, "labels": cur_labels})
        break

cv2.destroyAllWindows()

if len(clicks) == 0:
    raise RuntimeError("No objects prompted.")

# -------------------------
# Build SAM2 video predictor + init
# -------------------------
predictor = build_sam2_video_predictor(
    config_file=MODEL_CFG,
    ckpt_path=CHECKPOINT,
    device=DEVICE,
)

state = predictor.init_state(
    video_path=VIDEO_PATH,
    offload_video_to_cpu=True,
    async_loading_frames=True,
)

# Add objects (one object id per robot)
# SAM2 expects points as Nx2 and labels as N, per frame for a given obj_id
for i, obj in enumerate(clicks, start=1):
    pts = np.array(obj["points"], dtype=np.float32)
    labs = np.array(obj["labels"], dtype=np.int32)

    # Add prompts on frame 0
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=i,
        points=pts,
        labels=labs,
    )

# -------------------------
# Prepare writer
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"avc1")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (W, H))
if not writer.isOpened():
    print("Warning: Could not open VideoWriter with 'avc1', trying 'mp4v'...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {OUT_VIDEO} with fps={fps}, size={(W, H)}"
        )

print(f"VideoWriter initialized: {OUT_VIDEO}, {fps} fps, {(W, H)}")

# -------------------------
# Propagate + render
# -------------------------
# predictor.propagate_in_video yields (frame_idx, obj_ids, masks)
# masks are typically float/logits; we threshold to boolean
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

for out in predictor.propagate_in_video(state):
    # The exact tuple structure can differ slightly by repo version;
    # these names match the SAM2 predictor API in the official repo.
    out_frame_idx, out_obj_ids, out_masks = out

    # read frames until we match out_frame_idx
    while frame_idx <= out_frame_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx == out_frame_idx:
            # out_masks: [num_objs, H, W] (or similar)
            masks = out_masks
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()

            # Ensure masks are (N,H,W)
            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks[:, 0]  # (N,1,H,W)->(N,H,W)

            # Overlay each object
            # out_obj_ids should align with masks[mi]
            for mi, obj_id in enumerate(out_obj_ids):
                obj_id = int(obj_id)
                mask_bool = masks[mi] > 0.0

                color = OBJ_COLORS.get(obj_id, (0, 255, 0))
                name = OBJ_NAMES.get(obj_id, f"OBJ_{obj_id}")

                frame = overlay_mask(frame, mask_bool, color_bgr=color, alpha=ALPHA)

                c = mask_centroid(mask_bool)
                if c is not None:
                    draw_label(frame, name, c, color_bgr=color)

            writer.write(frame)
        frame_idx += 1

cap.release()
writer.release()
print(f"Saved: {OUT_VIDEO}")
