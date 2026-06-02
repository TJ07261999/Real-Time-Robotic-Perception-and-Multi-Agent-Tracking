import os
import glob
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def imread_color(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def imread_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def list_images(images_dir: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for p in glob.glob(os.path.join(images_dir, "*")):
        if p.lower().endswith(exts):
            paths.append(p)
    return sorted(paths)

def read_yolo_txt(label_path: str) -> np.ndarray:
    """
    returns Nx5 float32: [cls, cx, cy, w, h] normalized
    """
    if not os.path.exists(label_path):
        return np.zeros((0, 5), np.float32)
    rows = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, w, h = map(float, parts[:5])
            rows.append([c, cx, cy, w, h])
    if not rows:
        return np.zeros((0, 5), np.float32)
    return np.array(rows, np.float32)

def yolo_to_xyxy(row, W, H):
    c, cx, cy, w, h = row
    cx *= W; cy *= H; w *= W; h *= H
    x1 = int(round(cx - w/2))
    y1 = int(round(cy - h/2))
    x2 = int(round(cx + w/2))
    y2 = int(round(cy + h/2))
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return int(c), x1, y1, x2, y2

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# -------------------------
# Scene parameters (100 sets)
# -------------------------
@dataclass
class SceneParams:
    # global
    brightness: float      # multiplicative
    contrast: float        # multiplicative
    gamma: float
    noise_sigma: float
    blur_ksize: int
    jpeg_quality: int
    vignette: float        # 0..1
    # robot-specific
    robot_scale_min: float
    robot_scale_max: float
    robot_rot_deg: float
    robot_alpha_jitter: float   # +/- alpha jitter
    shadow_strength: float      # 0..1
    shadow_blur: int

def sample_scene_params(rng: random.Random) -> SceneParams:
    # keep ranges realistic for your fixed camera / grayscale look
    blur_choices = [0, 0, 0, 3, 5]  # mostly no blur
    sb_choices   = [0, 3, 5, 7]

    return SceneParams(
        brightness=rng.uniform(0.80, 1.25),
        contrast=rng.uniform(0.80, 1.30),
        gamma=rng.uniform(0.85, 1.20),
        noise_sigma=rng.uniform(0.0, 12.0),
        blur_ksize=rng.choice(blur_choices),
        jpeg_quality=rng.randint(45, 95),
        vignette=rng.uniform(0.0, 0.35),

        robot_scale_min=rng.uniform(0.75, 0.95),
        robot_scale_max=rng.uniform(1.05, 1.45),
        robot_rot_deg=rng.uniform(0, 360),
        robot_alpha_jitter=rng.uniform(0.00, 0.10),
        shadow_strength=rng.uniform(0.00, 0.25),
        shadow_blur=rng.choice(sb_choices),
    )


# -------------------------
# Patch extraction from initial_images + baseline
# -------------------------
def extract_patches_rgba(
    initial_images_dir: str,
    initial_labels_dir: str,
    baseline_path: str,
    out_patches_dir: str,
    out_img_wh: Tuple[int, int],
    min_box_px: int = 7,
):
    """
    Extract RGBA patches by subtracting baseline inside each bbox.
    Saves: out_patches_dir/class_<c>/*.png (RGBA)
    """
    ensure_dir(out_patches_dir)
    baseline = imread_color(baseline_path)
    outW, outH = out_img_wh
    baseline = cv2.resize(baseline, (outW, outH), interpolation=cv2.INTER_AREA)

    img_paths = list_images(initial_images_dir)
    if not img_paths:
        raise RuntimeError(f"No images in {initial_images_dir}")

    saved = 0
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(initial_labels_dir, stem + ".txt")
        labels = read_yolo_txt(lbl_path)

        img = imread_color(img_path)
        img = cv2.resize(img, (outW, outH), interpolation=cv2.INTER_AREA)

        H, W = img.shape[:2]

        for row in labels:
            xyxy = yolo_to_xyxy(row, W, H)
            if xyxy is None:
                continue
            cls, x1, y1, x2, y2 = xyxy
            bw = x2 - x1
            bh = y2 - y1
            if bw < min_box_px or bh < min_box_px:
                continue

            crop = img[y1:y2, x1:x2].copy()
            base = baseline[y1:y2, x1:x2].copy()

            # baseline subtraction -> mask
            diff = cv2.absdiff(crop, base)
            diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # robust threshold: use percentile-based
            thr = max(10, int(np.percentile(diff_g, 75)))
            mask = (diff_g > thr).astype(np.uint8) * 255

            # cleanup
            k = max(3, (min(bw, bh) // 20) * 2 + 1)
            k = min(k, 15)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # keep largest component
            num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best = 1 + int(np.argmax(areas))
                mask = (cc == best).astype(np.uint8) * 255

            # soften edges
            mask = cv2.GaussianBlur(mask, (k, k), 0)

            # RGBA patch
            alpha = mask.astype(np.float32) / 255.0
            rgba = np.dstack([crop, (alpha * 255.0).astype(np.uint8)])

            # save
            cls_dir = os.path.join(out_patches_dir, f"class_{int(cls)}")
            ensure_dir(cls_dir)
            out_path = os.path.join(cls_dir, f"{stem}_{saved:06d}.png")
            cv2.imwrite(out_path, rgba)
            saved += 1

    if saved == 0:
        raise RuntimeError(
            "No patches extracted. Likely causes:\n"
            "- baseline isn't aligned with initial images (different camera/scene)\n"
            "- labels missing or not YOLO normalized\n"
            "- threshold too strict\n"
        )
    print(f"[PatchExtraction] saved patches: {saved}")
    return saved


def load_patch_index(patches_dir: str) -> Dict[int, List[str]]:
    idx = {0: [], 1: []}
    for c in [0, 1]:
        cls_dir = os.path.join(patches_dir, f"class_{c}")
        if not os.path.isdir(cls_dir):
            continue
        for p in glob.glob(os.path.join(cls_dir, "*.png")):
            idx[c].append(p)
    for c in [0, 1]:
        idx[c] = sorted(idx[c])
    return idx


# -------------------------
# Compositing
# -------------------------
def rotate_and_scale_rgba(rgba: np.ndarray, scale: float, angle_deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    rgba = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    # compute bounds
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    bw = int((h * sin) + (w * cos))
    bh = int((h * cos) + (w * sin))
    M[0, 2] += (bw/2) - w/2
    M[1, 2] += (bh/2) - h/2
    out = cv2.warpAffine(rgba, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    return out

def add_shadow(img_bgr: np.ndarray, alpha_mask: np.ndarray, x: int, y: int, strength: float, blur: int):
    """
    alpha_mask is (h,w) float in [0,1] for robot at position (x,y).
    Create a soft darkening offset shadow.
    """
    if strength <= 0:
        return img_bgr

    h, w = alpha_mask.shape[:2]
    H, W = img_bgr.shape[:2]

    # simple offset shadow direction (fixed-ish)
    dx = int(round(0.02 * W))
    dy = int(round(0.02 * H))

    sx1 = max(0, x + dx)
    sy1 = max(0, y + dy)
    sx2 = min(W, sx1 + w)
    sy2 = min(H, sy1 + h)

    if sx2 <= sx1 or sy2 <= sy1:
        return img_bgr

    mask = alpha_mask[:(sy2 - sy1), :(sx2 - sx1)].copy()
    if blur and blur >= 3:
        k = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    dark = (1.0 - strength * mask)[..., None]  # (h,w,1)
    region = img_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    region = region * dark
    img_bgr[sy1:sy2, sx1:sx2] = np.clip(region, 0, 255).astype(np.uint8)
    return img_bgr

def overlay_rgba(img_bgr: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha_jitter: float, shadow_strength: float, shadow_blur: int):
    H, W = img_bgr.shape[:2]
    h, w = rgba.shape[:2]

    if x < 0 or y < 0 or x + w > W or y + h > H:
        return img_bgr, None  # out of bounds

    bot_rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    if alpha_jitter > 0:
        jitter = np.random.uniform(-alpha_jitter, alpha_jitter)
        alpha = np.clip(alpha + jitter, 0.0, 1.0)

    # optional shadow first
    img_bgr = add_shadow(img_bgr, alpha, x, y, shadow_strength, shadow_blur)

    region = img_bgr[y:y+h, x:x+w].astype(np.float32)
    out = alpha[..., None] * bot_rgb + (1.0 - alpha[..., None]) * region
    img_bgr[y:y+h, x:x+w] = np.clip(out, 0, 255).astype(np.uint8)

    # bbox from alpha support
    ys, xs = np.where(alpha > 0.1)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, None
    bx1 = x + int(xs.min()); bx2 = x + int(xs.max()) + 1
    by1 = y + int(ys.min()); by2 = y + int(ys.max()) + 1
    return img_bgr, (bx1, by1, bx2, by2)


# -------------------------
# Global image augmentations per scene params
# -------------------------
def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv = 1.0 / gamma
    table = np.array([(i/255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    H, W = img.shape[:2]
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx*xx + yy*yy)
    mask = 1.0 - strength * np.clip(rr, 0, 1)
    out = img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_scene_augment(img_bgr: np.ndarray, sp: SceneParams, rng: random.Random) -> np.ndarray:
    img = img_bgr.astype(np.float32)

    # brightness/contrast
    img = img * sp.brightness
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * sp.contrast + mean

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = apply_gamma(img, sp.gamma)

    # noise
    if sp.noise_sigma > 0:
        noise = rng.normalvariate(0, sp.noise_sigma)
        n = np.random.normal(0, sp.noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    # blur
    if sp.blur_ksize and sp.blur_ksize >= 3:
        k = sp.blur_ksize if sp.blur_ksize % 2 == 1 else sp.blur_ksize + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # vignette
    img = apply_vignette(img, sp.vignette)

    # jpeg artifacts (encode/decode)
    q = int(sp.jpeg_quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode(".jpg", img, encode_param)
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img


# -------------------------
# Placement sampling (arena valid mask)
# -------------------------
def build_valid_mask_from_baseline(baseline_bgr: np.ndarray) -> np.ndarray:
    """
    Rough arena floor mask: threshold + morphology.
    You can replace with a hand-made mask if you want.
    """
    gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    return m

def sample_position(valid_mask: np.ndarray, w: int, h: int, rng: random.Random, tries: int = 200):
    H, W = valid_mask.shape[:2]
    for _ in range(tries):
        x = rng.randint(0, W - w - 1)
        y = rng.randint(0, H - h - 1)
        cx = x + w // 2
        cy = y + h // 2
        if valid_mask[cy, cx] > 0:
            return x, y
    return None


# -------------------------
# Main generator
# -------------------------
def generate_split(
    split_name: str,
    out_root: str,
    baseline_path: str,
    patches_index: Dict[int, List[str]],
    out_img_wh: Tuple[int, int],
    scene_params_list: List[SceneParams],
    arrangements_per_scene: int,
    min_box_px: int,
    seed: int,
):
    outW, outH = out_img_wh
    split_img_dir = os.path.join(out_root, split_name, "images")
    split_lbl_dir = os.path.join(out_root, split_name, "labels")
    ensure_dir(split_img_dir)
    ensure_dir(split_lbl_dir)

    rng = random.Random(seed)

    baseline = imread_color(baseline_path)
    baseline = cv2.resize(baseline, (outW, outH), interpolation=cv2.INTER_AREA)
    valid_mask = build_valid_mask_from_baseline(baseline)

    # verify we have patches
    if not patches_index.get(0) and not patches_index.get(1):
        raise RuntimeError("No patches found. Did patch extraction run?")

    img_id = 0
    total_target = len(scene_params_list) * arrangements_per_scene

    for si, sp in enumerate(scene_params_list):
        for ai in range(arrangements_per_scene):
            img = baseline.copy()
            labels = []

            # 0..2 robots, but keep "at most 2" strictly
            num = rng.choice([0, 1, 2])

            placed_boxes = []

            for cls in range(num):
                candidates = patches_index.get(cls, [])
                if not candidates:
                    # if you don't have class-specific crops, fallback to any
                    candidates = (patches_index.get(0, []) + patches_index.get(1, []))
                    if not candidates:
                        break
                patch_path = rng.choice(candidates)
                rgba = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
                if rgba is None or rgba.shape[2] != 4:
                    continue

                scale = rng.uniform(sp.robot_scale_min, sp.robot_scale_max)
                angle = rng.uniform(0, sp.robot_rot_deg)
                rgba_t = rotate_and_scale_rgba(rgba, scale=scale, angle_deg=angle)

                h, w = rgba_t.shape[:2]
                pos = sample_position(valid_mask, w, h, rng)
                if pos is None:
                    continue
                x, y = pos

                # mild anti-clutter: avoid large overlap
                ok = True
                for (bx1, by1, bx2, by2) in placed_boxes:
                    ix1 = max(x, bx1); iy1 = max(y, by1)
                    ix2 = min(x+w, bx2); iy2 = min(y+h, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area = (w * h)
                        if inter / max(1.0, area) > 0.25:
                            ok = False
                            break
                if not ok:
                    continue

                img, bb = overlay_rgba(
                    img, rgba_t, x, y,
                    alpha_jitter=sp.robot_alpha_jitter,
                    shadow_strength=sp.shadow_strength,
                    shadow_blur=sp.shadow_blur
                )
                if bb is None:
                    continue

                bx1, by1, bx2, by2 = bb
                bw = bx2 - bx1
                bh = by2 - by1

                if bw < min_box_px or bh < min_box_px:
                    continue

                # YOLO normalized
                cx = (bx1 + bx2) / 2.0 / outW
                cy = (by1 + by2) / 2.0 / outH
                ww = bw / outW
                hh = bh / outH

                labels.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                placed_boxes.append((bx1, by1, bx2, by2))

            # global scene augment
            img = apply_scene_augment(img, sp, rng)

            # save
            name = f"{split_name}_{si:03d}_{ai:03d}_{img_id:06d}"
            out_img = os.path.join(split_img_dir, name + ".png")
            out_lbl = os.path.join(split_lbl_dir, name + ".txt")
            cv2.imwrite(out_img, img)
            with open(out_lbl, "w") as f:
                f.write("\n".join(labels))

            img_id += 1

            if img_id % 200 == 0:
                print(f"[{split_name}] {img_id}/{total_target}")

    print(f"[{split_name}] done: {img_id} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="empty arena baseline image (png/jpg)")
    ap.add_argument("--initial_images", required=True, help="real images folder")
    ap.add_argument("--initial_labels", required=True, help="YOLO labels folder for real images")
    ap.add_argument("--out_root", default="synthetic_out", help="output dataset root")

    ap.add_argument("--img_w", type=int, default=512)
    ap.add_argument("--img_h", type=int, default=384)

    ap.add_argument("--train_scenes", type=int, default=100)
    ap.add_argument("--train_arrangements", type=int, default=50)  # 100 * 50 = 5000
    ap.add_argument("--test_scenes", type=int, default=100)
    ap.add_argument("--test_arrangements", type=int, default=20)   # 100 * 20 = 2000

    ap.add_argument("--min_box_px", type=int, default=7)

    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_img_wh = (args.img_w, args.img_h)

    ensure_dir(args.out_root)
    patches_dir = os.path.join(args.out_root, "assets", "patches_rgba")
    ensure_dir(patches_dir)

    # 1) Extract patches from initial_images/labels using baseline subtraction
    extract_patches_rgba(
        initial_images_dir=args.initial_images,
        initial_labels_dir=args.initial_labels,
        baseline_path=args.baseline,
        out_patches_dir=patches_dir,
        out_img_wh=out_img_wh,
        min_box_px=args.min_box_px,
    )

    # 2) Load patch index
    patches_index = load_patch_index(patches_dir)
    print("[PatchIndex] class0:", len(patches_index.get(0, [])), " class1:", len(patches_index.get(1, [])))

    # 3) Sample scene parameter sets
    rng_train = random.Random(args.seed + 1000)
    rng_test  = random.Random(args.seed + 2000)

    train_params = [sample_scene_params(rng_train) for _ in range(args.train_scenes)]
    test_params  = [sample_scene_params(rng_test)  for _ in range(args.test_scenes)]

    # 4) Generate splits
    generate_split(
        split_name="train",
        out_root=args.out_root,
        baseline_path=args.baseline,
        patches_index=patches_index,
        out_img_wh=out_img_wh,
        scene_params_list=train_params,
        arrangements_per_scene=args.train_arrangements,
        min_box_px=args.min_box_px,
        seed=args.seed + 1,
    )

    generate_split(
        split_name="test",
        out_root=args.out_root,
        baseline_path=args.baseline,
        patches_index=patches_index,
        out_img_wh=out_img_wh,
        scene_params_list=test_params,
        arrangements_per_scene=args.test_arrangements,
        min_box_px=args.min_box_px,
        seed=args.seed + 2,
    )

    print("All done.")
    print(f"Output written to: {args.out_root}")
    print("Train:", os.path.join(args.out_root, "train"))
    print("Test :", os.path.join(args.out_root, "test"))


if __name__ == "__main__":
    main()
