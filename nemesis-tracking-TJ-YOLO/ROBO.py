# robo_train_prune.py
# Full ROBO / ROBO-BN training + L1 sparsity + pruning + masked finetune
# Dataset: images + YOLO txt labels (class cx cy w h normalized)

import os
import math
import glob
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
except ImportError:
    raise ImportError("Please install pillow: pip install pillow")


# -------------------------
# 1) Model blocks
# -------------------------
class ConvBNLeaky(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ROBO(nn.Module):
    """
    ROBO-like backbone (2-head, class-specific outputs):
    output per head: (B, 10, H, W) where 10 = 2 classes * 5 params
    """
    def __init__(self, in_ch=3, out_ch=10):
        super().__init__()
        spec = [
            (4,   2),  # /2
            (8,   2),  # /4
            (16,  2),  # /8
            (16,  1),
            (32,  2),  # /16
            (32,  1),
            (64,  2),  # /32
            (64,  1),
            (128, 2),  # /64
            (256, 1),  # (paper variants differ; keep <=256)
            (128, 1),
            (256, 1),
            (128, 1),
            (64,  1),
        ]
        layers = []
        prev = in_ch
        for oc, s in spec:
            layers.append(ConvBNLeaky(prev, oc, k=3, s=s))
            prev = oc
        self.backbone = nn.ModuleList(layers)

        # Choose an earlier feature for the higher-res head:
        # After /32 stage: typically around index where stride becomes 32.
        self.hr_index = 7   # feature map around /32 (tune if needed)

        # Head outputs: 10 channels = 2 classes * (tx,ty,tw,th,to)
        # low-res head from last feature (usually /64)
        self.head_low = nn.Conv2d(prev, out_ch, kernel_size=1)

        # high-res head from earlier feature (must match channel count there)
        hr_ch = self.backbone[self.hr_index].bn.num_features
        self.head_high = nn.Conv2d(hr_ch, out_ch, kernel_size=1)

    def forward(self, x):
        feats = []
        for layer in self.backbone:
            x = layer(x)
            feats.append(x)
        out_low = self.head_low(feats[-1])
        out_high = self.head_high(feats[self.hr_index])
        return out_low, out_high


class ROBO_BN(nn.Module):
    """
    ROBO-BN (bottleneck-ish): doubled channels + 1x1 bottlenecks.
    Still outputs (B,10,H,W) per head.
    """
    def __init__(self, in_ch=3, out_ch=10):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNLeaky(in_ch,  8, 3, 2),   # /2
            ConvBNLeaky(8,     16, 3, 2),   # /4
            ConvBNLeaky(16,    32, 3, 2),   # /8
            ConvBNLeaky(32,    32, 3, 1),
            ConvBNLeaky(32,    64, 3, 2),   # /16
            ConvBNLeaky(64,    64, 3, 1),
            ConvBNLeaky(64,   128, 3, 2),   # /32
        )
        # keep a feature for high-res head
        self.hr_ch = 128

        # bottleneck chain
        self.mid = nn.Sequential(
            ConvBNLeaky(128,  64, 1, 1),
            ConvBNLeaky(64,  256, 3, 2),    # /64
            ConvBNLeaky(256, 128, 1, 1),
            ConvBNLeaky(128, 256, 3, 1),
            ConvBNLeaky(256, 128, 1, 1),
            ConvBNLeaky(128, 256, 3, 1),
            ConvBNLeaky(256, 128, 3, 1),
            ConvBNLeaky(128,  64, 1, 1),
        )

        self.head_low  = nn.Conv2d(64, out_ch, kernel_size=1)      # from /64
        self.head_high = nn.Conv2d(self.hr_ch, out_ch, kernel_size=1)  # from /32

    def forward(self, x):
        hr = self.stem(x)          # /32 feature
        low = self.mid(hr)         # /64 feature -> ends at 64ch
        out_low  = self.head_low(low)
        out_high = self.head_high(hr)
        return out_low, out_high


# -------------------------
# 2) Dataset (YOLO txt)
# -------------------------
class YoloTxtDataset(Dataset):
    """
    Expects:
      images_dir/*.jpg|png
      labels_dir/<same_stem>.txt with lines: class cx cy w h (normalized 0..1)
    """
    def __init__(self, images_dir: str, labels_dir: str, img_size: Tuple[int, int]=(512, 384)):
        self.images = sorted([p for p in glob.glob(os.path.join(images_dir, "*"))
                              if p.lower().endswith((".jpg", ".jpeg", ".png"))])
        self.labels_dir = labels_dir
        self.W, self.H = img_size  # desired (W,H)
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.images)

    def _read_labels(self, path_txt: str) -> torch.Tensor:
        # returns (N,5): [cls, cx, cy, w, h] normalized
        if not os.path.exists(path_txt):
            return torch.zeros((0, 5), dtype=torch.float32)
        rows = []
        with open(path_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, w, h = map(float, parts[:5])
                rows.append([c, cx, cy, w, h])
        if len(rows) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.labels_dir, stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        # resize to fixed size (no padding here; you can change to letterbox if needed)
        img = img.resize((self.W, self.H), Image.BILINEAR)

        x = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy())
        x = x.view(self.H, self.W, 3).permute(2, 0, 1).float() / 255.0  # (3,H,W)

        labels = self._read_labels(lbl_path)  # (N,5) normalized
        return x, labels


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)


# -------------------------
# 3) Targets + Loss (ROBO format)
# -------------------------
@dataclass
class HeadSpec:
    stride: int
    grid_h: int
    grid_w: int
    anchors_wh_grid: torch.Tensor  # (2,2) in GRID units for this head


def build_targets_robo(
    labels_list: List[torch.Tensor],
    head: HeadSpec,
    img_wh: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    labels_list: list length B, each (N,5): [cls, cx, cy, w, h] normalized
    head: stride/grid/anchors
    returns:
      obj: (B,2,H,W) in {0,1}
      t:   (B,2,4,H,W) where t = [tx_target, ty_target, tw_target, th_target]
           tx_target,ty_target in [0,1), tw/th are log-space targets.
    """
    B = len(labels_list)
    H, W = head.grid_h, head.grid_w
    obj = torch.zeros((B, 2, H, W), dtype=torch.float32, device=device)
    t   = torch.zeros((B, 2, 4, H, W), dtype=torch.float32, device=device)

    imgW, imgH = img_wh
    S = head.stride
    aw = head.anchors_wh_grid[:, 0].view(2, 1, 1)  # (2,1,1)
    ah = head.anchors_wh_grid[:, 1].view(2, 1, 1)

    for b in range(B):
        labels = labels_list[b].to(device)
        if labels.numel() == 0:
            continue

        # if you truly have "at most one per class", you can optionally keep the highest-area box per class:
        # but we support multiple by letting later overwrite (rare in your domain).
        for row in labels:
            cls = int(row[0].item())
            if cls not in (0, 1):
                continue

            cx = row[1].item() * imgW
            cy = row[2].item() * imgH
            bw = row[3].item() * imgW
            bh = row[4].item() * imgH

            gx = cx / S
            gy = cy / S
            j = int(math.floor(gx))
            i = int(math.floor(gy))
            if i < 0 or i >= H or j < 0 or j >= W:
                continue

            x_off = max(0.0, min(0.999, gx - j))
            y_off = max(0.0, min(0.999, gy - i))

            # wh targets in grid units
            bw_g = max(1e-6, bw / S)
            bh_g = max(1e-6, bh / S)

            # tw/th as log ratio vs class-specific anchor
            tw_t = math.log(bw_g / float(head.anchors_wh_grid[cls, 0].item()))
            th_t = math.log(bh_g / float(head.anchors_wh_grid[cls, 1].item()))

            obj[b, cls, i, j] = 1.0
            t[b, cls, 0, i, j] = x_off
            t[b, cls, 1, i, j] = y_off
            t[b, cls, 2, i, j] = tw_t
            t[b, cls, 3, i, j] = th_t

    return obj, t


def robo_head_loss(
    pred: torch.Tensor,
    obj_t: torch.Tensor,
    t_t: torch.Tensor,
    lambda_box: float = 5.0,
    lambda_obj: float = 1.0,
) -> torch.Tensor:
    """
    pred: (B,10,H,W) logits
    obj_t: (B,2,H,W)
    t_t:   (B,2,4,H,W)
    """
    B, C, H, W = pred.shape
    p = pred.view(B, 2, 5, H, W)
    tx, ty, tw, th, to = p[:, :, 0], p[:, :, 1], p[:, :, 2], p[:, :, 3], p[:, :, 4]

    # objectness BCE on logits
    loss_obj = F.binary_cross_entropy_with_logits(to, obj_t, reduction="mean")

    pos = obj_t.bool()
    if pos.sum().item() == 0:
        return lambda_obj * loss_obj

    # x,y offsets in [0,1)
    loss_xy = F.binary_cross_entropy_with_logits(tx[pos], t_t[:, :, 0][pos]) + \
              F.binary_cross_entropy_with_logits(ty[pos], t_t[:, :, 1][pos])

    # w,h log-ratios
    loss_wh = F.smooth_l1_loss(tw[pos], t_t[:, :, 2][pos]) + \
              F.smooth_l1_loss(th[pos], t_t[:, :, 3][pos])

    loss_box = (loss_xy + loss_wh) / float(pos.sum().item())
    return lambda_box * loss_box + lambda_obj * loss_obj


def l1_penalty(model: nn.Module) -> torch.Tensor:
    l1 = 0.0
    for p in model.parameters():
        if p.requires_grad:
            l1 = l1 + p.abs().sum()
    return l1


# -------------------------
# 4) Pruning + Masked finetune
# -------------------------
def make_prune_masks(model: nn.Module, threshold_ratio: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Layer-wise magnitude pruning mask:
      mask = |W| >= threshold_ratio * max(|W|)  (per tensor)
    We mask only conv/linear weights (skip BN params).
    """
    masks = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:  # skip biases and BN gamma/beta
            continue
        maxv = p.abs().max()
        if maxv.item() == 0.0:
            mask = torch.zeros_like(p, dtype=torch.bool)
        else:
            thr = threshold_ratio * maxv
            mask = (p.abs() >= thr)
        masks[name] = mask
    return masks


@torch.no_grad()
def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]):
    for name, p in model.named_parameters():
        if name in masks:
            p.mul_(masks[name])


def register_mask_hooks(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """
    Gradient hook: grad *= mask
    """
    for name, p in model.named_parameters():
        if name in masks:
            mask = masks[name]
            p.register_hook(lambda g, m=mask: g * m)


def count_sparsity(model: nn.Module, masks: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    total = 0
    kept = 0
    for name, p in model.named_parameters():
        if name in masks:
            m = masks[name]
            total += m.numel()
            kept += int(m.sum().item())
    return kept, total


# -------------------------
# 5) Train / Eval loop
# -------------------------
@dataclass
class TrainConfig:
    lr: float
    eta_min: float
    weight_decay: float
    beta1: float
    epochs: int
    batch_size: int
    lambda_l1: float
    lambda_box: float
    lambda_obj: float
    amp: bool


def build_heads(img_wh: Tuple[int, int], device: torch.device,
                anchor_px_per_class: Dict[int, Tuple[float, float]]) -> Tuple[HeadSpec, HeadSpec]:
    """
    Two heads: stride 64 (low-res) and stride 32 (high-res).
    anchors given in PIXELS per class, convert to GRID units per head.
    """
    imgW, imgH = img_wh

    # head strides
    strides = [64, 32]
    heads = []
    for S in strides:
        grid_w = imgW // S
        grid_h = imgH // S
        anchors_wh = []
        for c in [0, 1]:
            aw_px, ah_px = anchor_px_per_class[c]
            anchors_wh.append([aw_px / S, ah_px / S])
        anchors_wh_grid = torch.tensor(anchors_wh, dtype=torch.float32, device=device)  # (2,2)
        heads.append(HeadSpec(stride=S, grid_h=grid_h, grid_w=grid_w, anchors_wh_grid=anchors_wh_grid))
    return heads[0], heads[1]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: TrainConfig,
    head_low: HeadSpec,
    head_high: HeadSpec,
    img_wh: Tuple[int, int],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, labels_list in loader:
        imgs = imgs.to(device, non_blocking=True)

        # targets for each head
        obj_l, t_l = build_targets_robo(labels_list, head_low,  img_wh, device)
        obj_h, t_h = build_targets_robo(labels_list, head_high, img_wh, device)

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                out_low, out_high = model(imgs)
                loss_low  = robo_head_loss(out_low,  obj_l, t_l, cfg.lambda_box, cfg.lambda_obj)
                loss_high = robo_head_loss(out_high, obj_h, t_h, cfg.lambda_box, cfg.lambda_obj)
                loss = loss_low + loss_high
                if cfg.lambda_l1 > 0:
                    loss = loss + cfg.lambda_l1 * l1_penalty(model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_low, out_high = model(imgs)
            loss_low  = robo_head_loss(out_low,  obj_l, t_l, cfg.lambda_box, cfg.lambda_obj)
            loss_high = robo_head_loss(out_high, obj_h, t_h, cfg.lambda_box, cfg.lambda_obj)
            loss = loss_low + loss_high
            if cfg.lambda_l1 > 0:
                loss = loss + cfg.lambda_l1 * l1_penalty(model)
            loss.backward()
            optimizer.step()

        # if pruning masks exist, enforce zeros after optimizer step
        if masks is not None:
            apply_masks_(model, masks)

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    head_low: HeadSpec,
    head_high: HeadSpec,
    img_wh: Tuple[int, int],
    device: torch.device,
):
    """
    Returns:
      val_loss_avg, stats dict
    Stats include:
      - pos_count_gt: number of GT positive cells
      - pred_pos_50: predicted positives where sigmoid(obj) > 0.5
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    pos_count_gt = 0
    pred_pos_50  = 0

    for imgs, labels_list in loader:
        imgs = imgs.to(device, non_blocking=True)

        # targets
        obj_l, t_l = build_targets_robo(labels_list, head_low,  img_wh, device)
        obj_h, t_h = build_targets_robo(labels_list, head_high, img_wh, device)

        out_low, out_high = model(imgs)

        loss_low  = robo_head_loss(out_low,  obj_l, t_l, cfg.lambda_box, cfg.lambda_obj)
        loss_high = robo_head_loss(out_high, obj_h, t_h, cfg.lambda_box, cfg.lambda_obj)
        loss = loss_low + loss_high

        total_loss += float(loss.item())
        n_batches += 1

        # simple monitoring stats
        pos_count_gt += int(obj_l.sum().item() + obj_h.sum().item())

        p_low  = out_low.view(out_low.size(0), 2, 5, out_low.size(2), out_low.size(3))[:, :, 4]
        p_high = out_high.view(out_high.size(0), 2, 5, out_high.size(2), out_high.size(3))[:, :, 4]
        pred_pos_50 += int((torch.sigmoid(p_low) > 0.5).sum().item() + (torch.sigmoid(p_high) > 0.5).sum().item())

    val_loss = total_loss / max(n_batches, 1)
    stats = {"pos_count_gt": pos_count_gt, "pred_pos_50": pred_pos_50}
    return val_loss, stats



@torch.no_grad()
def estimate_class_anchors_px(dataset: Dataset, img_wh: Tuple[int, int]) -> Dict[int, Tuple[float, float]]:
    """
    Compute class-specific mean (w,h) in pixels from YOLO labels.
    """
    imgW, imgH = img_wh
    sums = {0: [0.0, 0.0, 0], 1: [0.0, 0.0, 0]}
    for i in range(len(dataset)):
        _, labels = dataset[i]
        if labels.numel() == 0:
            continue
        for row in labels:
            c = int(row[0].item())
            if c not in (0, 1):
                continue
            w_px = float(row[3].item()) * imgW
            h_px = float(row[4].item()) * imgH
            sums[c][0] += w_px
            sums[c][1] += h_px
            sums[c][2] += 1

    anchors = {}
    for c in (0, 1):
        if sums[c][2] == 0:
            # fallback if no samples (shouldn't happen)
            anchors[c] = (40.0, 40.0)
        else:
            anchors[c] = (sums[c][0] / sums[c][2], sums[c][1] / sums[c][2])
    return anchors


def save_ckpt(path: str, model: nn.Module, optimizer=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


# -------------------------
# 5.5) Synthetic transfer helpers (ADDED, does not change defaults)
# -------------------------
def load_pretrained_if_requested(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def apply_synthetic_transfer_policy(
    model: nn.Module,
    model_name: str,
    unfreeze_stem: int,
    unfreeze_mid: int,
    freeze_heads: bool,
):
    """
    Implements the idea:
      - freeze everything
      - unfreeze first few layers (and optionally some mid layers)
      - optionally keep heads frozen
    """
    freeze_all(model)

    if model_name == "robo_bn":
        # Unfreeze first unfreeze_stem layers in stem
        stem_layers = list(model.stem.children())
        for layer in stem_layers[:max(0, unfreeze_stem)]:
            unfreeze_module(layer)

        # Unfreeze first unfreeze_mid layers in mid
        mid_layers = list(model.mid.children())
        for layer in mid_layers[:max(0, unfreeze_mid)]:
            unfreeze_module(layer)

        # Heads: freeze by default (paper idea), but allow turning off
        if not freeze_heads:
            unfreeze_module(model.head_low)
            unfreeze_module(model.head_high)

    else:
        # ROBO (ModuleList backbone)
        backbone_layers = list(model.backbone)
        for layer in backbone_layers[:max(0, unfreeze_stem)]:
            unfreeze_module(layer)

        # allow unfreezing extra mid layers using unfreeze_mid as "additional after stem"
        if unfreeze_mid > 0:
            for layer in backbone_layers[max(0, unfreeze_stem):max(0, unfreeze_stem + unfreeze_mid)]:
                unfreeze_module(layer)

        if not freeze_heads:
            unfreeze_module(model.head_low)
            unfreeze_module(model.head_high)


def trainable_param_count(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


# -------------------------
# 6) Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="initial_images")
    ap.add_argument("--labels", required=True, help="initial_labels")
    ap.add_argument("--val_images", type=str, default="", help="test/val images dir")
    ap.add_argument("--val_labels", type=str, default="", help="test/val labels dir")

    ap.add_argument("--model", choices=["robo", "robo_bn"], default="robo_bn")
    ap.add_argument("--img_w", type=int, default=512)
    ap.add_argument("--img_h", type=int, default=384)

    ap.add_argument("--epochs", type=int, default=125)     # detection training N
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eta_min", type=float, default=5e-5)
    ap.add_argument("--decay", type=float, default=1e-5)
    ap.add_argument("--beta1", type=float, default=0.9)

    ap.add_argument("--lambda_l1", type=float, default=0.0, help="L1 reg weight (sparsity)")
    ap.add_argument("--lambda_box", type=float, default=5.0)
    ap.add_argument("--lambda_obj", type=float, default=1.0)

    ap.add_argument("--amp", action="store_true")

    # pruning controls
    ap.add_argument("--prune_ratio", type=float, default=0.01, help="layer-wise threshold ratio")
    ap.add_argument("--prune_and_finetune", action="store_true")
    ap.add_argument("--finetune_epochs", type=int, default=25)

    ap.add_argument("--outdir", type=str, default="runs/robo")

    # -------- Synthetic transfer args (ADDED, defaults do nothing) --------
    ap.add_argument("--pretrained_ckpt", type=str, default="", help="path to pretrained checkpoint (synthetic pretrain)")
    ap.add_argument("--synthetic_transfer", action="store_true", help="enable synthetic->real transfer by retraining early layers")
    ap.add_argument("--unfreeze_stem", type=int, default=0, help="how many early layers to unfreeze (stem or backbone)")
    ap.add_argument("--unfreeze_mid", type=int, default=0, help="how many mid layers to unfreeze (first K layers of mid or after stem)")
    ap.add_argument("--freeze_heads", action="store_true", help="keep heads frozen (recommended for synthetic transfer)")
    # --------------------------------------------------------------------

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_wh = (args.img_w, args.img_h)

    ds = YoloTxtDataset(args.images, args.labels, img_size=img_wh)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = None
    if args.val_images and args.val_labels:
        val_ds = YoloTxtDataset(args.val_images, args.val_labels, img_size=img_wh)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )
        print(f"[Eval] Using val set: {len(val_ds)} images")
    else:
        print("[Eval] No val set provided (skip evaluation).")


    # Compute class-specific anchors (mean w,h per class) in pixels
    anchor_px = estimate_class_anchors_px(ds, img_wh)
    print("Class-specific anchors (pixels):", anchor_px)

    head_low, head_high = build_heads(img_wh, device, anchor_px)

    # model
    if args.model == "robo":
        model = ROBO(in_ch=3, out_ch=10).to(device)
    else:
        model = ROBO_BN(in_ch=3, out_ch=10).to(device)

    # -------- Synthetic transfer load + freeze/unfreeze (ADDED) --------
    if args.pretrained_ckpt:
        print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        load_pretrained_if_requested(model, args.pretrained_ckpt, device)

    if args.synthetic_transfer:
        # If user forgot to set unfreeze counts, pick a sensible safe default:
        # (does not affect non-synthetic-transfer runs)
        if args.unfreeze_stem <= 0 and args.unfreeze_mid <= 0:
            # default early layers only
            args.unfreeze_stem = 3
            args.unfreeze_mid = 0
        apply_synthetic_transfer_policy(
            model=model,
            model_name=args.model,
            unfreeze_stem=args.unfreeze_stem,
            unfreeze_mid=args.unfreeze_mid,
            freeze_heads=args.freeze_heads
        )
        tr, tot = trainable_param_count(model)
        print(f"[SyntheticTransfer] trainable params: {tr}/{tot} ({tr/tot*100:.2f}%)")
    # ------------------------------------------------------------------

    cfg = TrainConfig(
        lr=args.lr, eta_min=args.eta_min, weight_decay=args.decay, beta1=args.beta1,
        epochs=args.epochs, batch_size=args.batch,
        lambda_l1=args.lambda_l1, lambda_box=args.lambda_box, lambda_obj=args.lambda_obj,
        amp=args.amp
    )

    # IMPORTANT: if synthetic_transfer is on, only optimize trainable params
    params_for_optim = model.parameters()
    if args.synthetic_transfer:
        params_for_optim = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(
        params_for_optim,
        lr=cfg.lr,
        betas=(cfg.beta1, 0.999),
        weight_decay=cfg.weight_decay
    )

    # CosineAnnealingLR in PyTorch is typically stepped per-epoch.
    # Here we step per-batch for smoother schedule, so T_max = total_steps.
    total_steps = cfg.epochs * max(len(loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.eta_min
    )

    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None

    # ---- Train (sparse training if lambda_l1 > 0) ----
    best = 1e9
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(
            model, loader, optimizer, scheduler, cfg,
            head_low, head_high, img_wh, device, scaler=scaler, masks=None
        )
        msg = f"[epoch {epoch+1:03d}/{cfg.epochs}] train_loss={loss:.6f} lr={optimizer.param_groups[0]['lr']:.6e}"

        if val_loader is not None:
            val_loss, stats = evaluate(model, val_loader, cfg, head_low, head_high, img_wh, device)
            msg += f" | val_loss={val_loss:.6f} | gt_pos={stats['pos_count_gt']} pred_pos@0.5={stats['pred_pos_50']}"

        print(msg)


        score = val_loss if val_loader is not None else loss

        if score < best:
            best = score
            save_ckpt(os.path.join(args.outdir, "best.pt"), model, optimizer,
                    extra={"epoch": epoch, "train_loss": loss, "val_loss": val_loss if val_loader is not None else None})

        if (epoch + 1) % 10 == 0:
            save_ckpt(os.path.join(args.outdir, f"epoch_{epoch+1:03d}.pt"), model, optimizer, extra={"epoch": epoch, "loss": loss})

    save_ckpt(os.path.join(args.outdir, "last.pt"), model, optimizer, extra={"loss": best})

    # ---- Prune + Masked finetune ----
    if args.prune_and_finetune:
        print("\n==> Creating pruning masks...")
        masks = make_prune_masks(model, threshold_ratio=args.prune_ratio)
        apply_masks_(model, masks)
        kept, total = count_sparsity(model, masks)
        pruned = total - kept
        print(f"Masked params: kept={kept}/{total} ({kept/total*100:.2f}%)  pruned={pruned/total*100:.2f}%")

        # Make pruned weights stay zero:
        register_mask_hooks(model, masks)

        # Fine-tune optimizer: lr = eta_min/2, no cosine schedule
        ft_lr = cfg.eta_min / 2.0
        ft_cfg = TrainConfig(
            lr=ft_lr, eta_min=ft_lr, weight_decay=cfg.weight_decay, beta1=cfg.beta1,
            epochs=args.finetune_epochs, batch_size=cfg.batch_size,
            lambda_l1=0.0, lambda_box=cfg.lambda_box, lambda_obj=cfg.lambda_obj,
            amp=cfg.amp
        )

        # if synthetic_transfer was used, still respect requires_grad flags
        ft_params = model.parameters()
        if args.synthetic_transfer:
            ft_params = filter(lambda p: p.requires_grad, model.parameters())

        ft_opt = torch.optim.Adam(
            ft_params,
            lr=ft_cfg.lr,
            betas=(ft_cfg.beta1, 0.999),
            weight_decay=ft_cfg.weight_decay
        )
        ft_scaler = torch.cuda.amp.GradScaler() if (ft_cfg.amp and device.type == "cuda") else None

        print(f"==> Finetuning {args.finetune_epochs} epochs with lr={ft_lr:.6e} (eta_min/2), masked grads...")
        for e in range(args.finetune_epochs):
            loss = train_one_epoch(
                model, loader, ft_opt, scheduler=None, cfg=ft_cfg,
                head_low=head_low, head_high=head_high, img_wh=img_wh,
                device=device, scaler=ft_scaler, masks=masks
            )
            print(f"[finetune {e+1:03d}/{args.finetune_epochs}] loss={loss:.6f} lr={ft_opt.param_groups[0]['lr']:.6e}")
            # ensure zeros stay zero
            apply_masks_(model, masks)

        save_ckpt(os.path.join(args.outdir, "pruned_finetuned.pt"), model, ft_opt,
                  extra={"prune_ratio": args.prune_ratio, "finetune_epochs": args.finetune_epochs})

    print("Done.")


if __name__ == "__main__":
    main()
