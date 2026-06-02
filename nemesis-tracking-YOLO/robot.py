import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F

class ConvBNLeaky(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ROBO(nn.Module):
    """
    ROBO detector, as in:
    M. Szemenyei & V. Estivill-Castro,
    'ROBO: Robust, Fully Neural Object Detection for Robot Soccer' (RoboCup 2019)
    """
    def __init__(self, in_ch=3, n_outputs=10):
        super().__init__()

        # (out_channels, stride, kernel_size)
        spec: List[Tuple[int, int, int]] = [
            (4,   2, 3),
            (8,   2, 3),
            (16,  2, 3),
            (16,  1, 3),
            (32,  2, 3),
            (32,  1, 3),
            (64,  2, 3),
            (64,  1, 3),
            (128, 2, 3),
            (64,  1, 3),
            (128, 1, 3),
            (64,  1, 3),
            (64,  1, 3),
        ]

        layers = []
        prev_ch = in_ch
        self._feat_for_head2 = 9  # index of feature map used for high-res head (after Conv128/2)

        for i, (out_ch, stride, k) in enumerate(spec):
            block = ConvBNLeaky(prev_ch, out_ch, k=k, stride=stride)
            layers.append(block)
            prev_ch = out_ch

        self.backbone = nn.ModuleList(layers)

        # detection heads (1×1 convs, no BN/activation)
        self.head1 = nn.Conv2d(64, n_outputs, kernel_size=1)  # final low-res 8×6 grid
        self.head2 = nn.Conv2d(64, n_outputs, kernel_size=1)  # from earlier 64-ch map

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            feats.append(x)

        # low-res output from last 64-ch feature (layer 13)
        out1 = self.head1(feats[-1])

        # high-res output from earlier feature before some downscaling
        feat_hr = feats[self._feat_for_head2]  # choose according to stride 32 layer
        out2 = self.head2(feat_hr)

        return out1, out2  # shapes roughly [B,10,8,6] and [B,10,16,12] for 512×384 input


class ROBO_BN(nn.Module):
    """
    ROBO-BN (ROBO-Bottleneck) – doubled channels + 1×1 bottlenecks.
    Based on section 3.2 and Fig. 1 in the same paper.
    """
    def __init__(self, in_ch=3, n_outputs=10):
        super().__init__()

        # (out_channels, stride, kernel, is_1x1)
        spec = [
            (8,   2, 3),
            (16,  2, 3),
            (32,  2, 3),
            (32,  1, 3),
            (64,  2, 3),
            (64,  1, 3),
            (128, 2, 3),

            (64,  1, 1),  # bottleneck
            (256, 2, 3),

            (128, 1, 1),  # bottleneck
            (256, 1, 3),
            (128, 1, 1),  # bottleneck
            (256, 1, 3),
            (128, 1, 3),
            (64,  1, 1),  # bottleneck
        ]

        layers = []
        prev_ch = in_ch
        self._feat_for_head2 = 10  # earlier 128-ch map used for high-res head (tune if needed)
        feat_channels = []

        for out_ch, stride, k in spec:
            layers.append(ConvBNLeaky(prev_ch, out_ch, k=k, stride=stride))
            prev_ch = out_ch
            feat_channels.append(out_ch)

        self.backbone = nn.ModuleList(layers)

        self.head1 = nn.Conv2d(64,   n_outputs, kernel_size=1)   # last 64-ch bottleneck
        self.head2 = nn.Conv2d(128,  n_outputs, kernel_size=1)   # earlier 128-ch feature

    def forward(self, x):
        feats = []
        for layer in self.backbone:
            x = layer(x)
            feats.append(x)

        out1 = self.head1(feats[-1])                # low-res
        feat_hr = feats[self._feat_for_head2]        # high-res feature (128-ch)
        out2 = self.head2(feat_hr)

        return out1, out2



class ConvBNLeaky1(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)


class DeconvBNLeaky(nn.Module):
    def __init__(self, in_ch, out_ch, k=2, s=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)



class ROBOUNetV2(nn.Module):
    """
    ROBO-UNet-v2:
      - strided conv downsampling
      - concatenation skip connections (U-Net style)
      - low channel count (8/16/32/64)
      - many 64-channel bottleneck convs
    """
    def __init__(self, in_ch=3, out_ch=5, bottleneck_blocks=7):
        super().__init__()

        # Encoder
        self.enc0 = ConvBNLeaky1(in_ch, 8,  k=3, s=1)       # keep res
        self.enc1 = ConvBNLeaky1(8,    16, k=3, s=2)       # /2
        self.enc2 = ConvBNLeaky1(16,   32, k=3, s=2)       # /4
        self.enc3 = ConvBNLeaky1(32,   64, k=3, s=2)       # /8

        # Bottleneck: many 64->64 convs
        blocks = [ConvBNLeaky1(64, 64, k=3, s=1) for _ in range(bottleneck_blocks)]
        self.bottleneck = nn.Sequential(*blocks)

        # Decoder (upsample + concat skips)
        self.up2 = DeconvBNLeaky(64, 32)                  # /4
        self.dec2 = nn.Sequential(
            ConvBNLeaky1(32 + 32, 32, k=3, s=1),
            ConvBNLeaky1(32,      32, k=3, s=1),
        )

        self.up1 = DeconvBNLeaky(32, 16)                  # /2
        self.dec1 = nn.Sequential(
            ConvBNLeaky1(16 + 16, 16, k=3, s=1),
            ConvBNLeaky1(16,      16, k=3, s=1),
        )

        self.up0 = DeconvBNLeaky(16, 8)                   # /1 (back to original)
        self.dec0 = nn.Sequential(
            ConvBNLeaky1(8 + 8, 8, k=3, s=1),
            ConvBNLeaky1(8,     8, k=3, s=1),
        )

        # Output head: 1x1 conv (no BN/activation)
        self.out = nn.Conv2d(8, out_ch, kernel_size=1)

    def forward(self, x):
        s0 = self.enc0(x)         # H,W   (8ch)
        s1 = self.enc1(s0)        # H/2   (16ch)
        s2 = self.enc2(s1)        # H/4   (32ch)
        s3 = self.enc3(s2)        # H/8   (64ch)

        b  = self.bottleneck(s3)  # H/8   (64ch)

        u2 = self.up2(b)          # H/4   (32ch)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = self.up1(d2)         # H/2   (16ch)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        u0 = self.up0(d1)         # H,W   (8ch)
        d0 = self.dec0(torch.cat([u0, s0], dim=1))

        return self.out(d0)       # (B, out_ch, H, W)




def decode_robo_head(pred, anchors_wh, stride):
    """
    pred: (B, 2*5, H, W)
    anchors_wh: tensor shape (2,2) -> [[aw0,ah0],[aw1,ah1]] in GRID units for this head
    stride: int
    returns: boxes (B,2,H,W,4), obj (B,2,H,W)
    """
    B, C, H, W = pred.shape
    pred = pred.view(B, 2, 5, H, W)  # class, (tx,ty,tw,th,to)

    tx = pred[:, :, 0]
    ty = pred[:, :, 1]
    tw = pred[:, :, 2]
    th = pred[:, :, 3]
    to = pred[:, :, 4]

    # grid
    gy, gx = torch.meshgrid(torch.arange(H, device=pred.device),
                            torch.arange(W, device=pred.device), indexing="ij")
    gx = gx[None, None].float()
    gy = gy[None, None].float()

    sx = torch.sigmoid(tx)
    sy = torch.sigmoid(ty)
    obj = torch.sigmoid(to)

    aw = anchors_wh[:, 0].view(1,2,1,1)
    ah = anchors_wh[:, 1].view(1,2,1,1)

    x = (gx + sx) * stride
    y = (gy + sy) * stride
    w = aw * torch.exp(tw) * stride
    h = ah * torch.exp(th) * stride

    boxes = torch.stack([x, y, w, h], dim=-1)  # (B,2,H,W,4)
    return boxes, obj

def robo_loss(pred, target_obj, target_xywh, anchors_wh, stride,
              lambda_box=5.0, lambda_obj=1.0):
    """
    pred: (B,10,H,W)
    target_obj: (B,2,H,W) {0,1}
    target_xywh: (B,2,H,W,4) in pixels (x,y,w,h), valid where obj=1
    """
    B, C, H, W = pred.shape
    p = pred.view(B, 2, 5, H, W)
    tx,ty,tw,th,to = p[:, :, 0], p[:, :, 1], p[:, :, 2], p[:, :, 3], p[:, :, 4]

    # objectness BCE on logits
    loss_obj = F.binary_cross_entropy_with_logits(to, target_obj.float(), reduction="mean")

    # box loss only where positive
    pos = target_obj.bool()

    # targets for tx,ty in [0,1) cell offsets
    # convert target xy to cell coords
    # target_xywh is pixels
    x_t = target_xywh[..., 0] / stride
    y_t = target_xywh[..., 1] / stride
    j = torch.floor(x_t).clamp(0, W-1)
    i = torch.floor(y_t).clamp(0, H-1)

    # offsets inside cell
    x_off = (x_t - j).clamp(0, 1)
    y_off = (y_t - i).clamp(0, 1)

    # wh targets in log space vs anchor (grid units)
    aw = anchors_wh[:, 0].view(1,2,1,1)
    ah = anchors_wh[:, 1].view(1,2,1,1)
    w_t = (target_xywh[..., 2] / stride).clamp_min(1e-6)
    h_t = (target_xywh[..., 3] / stride).clamp_min(1e-6)
    tw_t = torch.log(w_t / aw)
    th_t = torch.log(h_t / ah)

    # apply only on positives
    loss_xy = F.binary_cross_entropy_with_logits(tx[pos], x_off[pos]) + \
              F.binary_cross_entropy_with_logits(ty[pos], y_off[pos])
    loss_wh = F.smooth_l1_loss(tw[pos], tw_t[pos]) + \
              F.smooth_l1_loss(th[pos], th_t[pos])

    loss_box = (loss_xy + loss_wh) / max(pos.sum().item(), 1)

    return lambda_box * loss_box + lambda_obj * loss_obj
