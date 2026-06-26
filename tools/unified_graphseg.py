#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, time, math, argparse, random, warnings
from contextlib import nullcontext
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*param_schemas.*", category=FutureWarning)

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- optional deps ----------------
TIMM_OK = False
try:
    import timm
    TIMM_OK = True
except Exception:
    TIMM_OK = False

TV_OK = False
if os.environ.get("SKIP_TORCHVISION", "0") != "1":
    try:
        import torchvision
        from torchvision.models import segmentation as tvseg
        TV_OK = True
    except Exception:
        TV_OK = False

MMSEG_OK = False
if os.environ.get("USE_MMSEG_BACKBONE", "0") == "1":
    try:
        from mmseg.models.backbones import MixVisionTransformer
        MMSEG_OK = True
    except Exception:
        MMSEG_OK = False

PYG_OK = False
if os.environ.get("SKIP_PYG", "0") != "1":
    try:
        from torch_geometric.nn import SAGEConv
        PYG_OK = True
    except Exception:
        PYG_OK = False

DCRF_OK = False
try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as dcrf_utils
    DCRF_OK = True
except Exception:
    DCRF_OK = False

WANDB_OK = False
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False


ARGS = None

# ---------------- utils ----------------
GRAPH_OUTPUT_MODES = ("residual", "blend_legacy")
GRAPH_SAFETY_GATES = ("none", "entropy", "margin")

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def normalize_graph_output_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in GRAPH_OUTPUT_MODES:
        raise ValueError(f"graph_output_mode must be one of {GRAPH_OUTPUT_MODES}, got {mode!r}")
    return mode

def normalize_graph_safety_gate(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in GRAPH_SAFETY_GATES:
        raise ValueError(f"graph_safety_gate must be one of {GRAPH_SAFETY_GATES}, got {mode!r}")
    return mode

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def freeze_host_params(model: nn.Module):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("graph.")
    model._freeze_host = True

def set_frozen_host_eval(model: nn.Module):
    if not getattr(model, "_freeze_host", False):
        return
    for name, module in model.named_children():
        if name != "graph":
            module.eval()

def load_host_weights(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    filtered = {k: v for k, v in sd.items() if not k.startswith("graph.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing_non_graph = [k for k in missing if not k.startswith("graph.")]
    return ckpt, missing, unexpected, missing_non_graph

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _strip_ext(x: str):
    x = str(x).strip()
    x = re.sub(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", "", x, flags=re.IGNORECASE)
    return x

def read_idx_file(path: str):
    lines = Path(path).read_text().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return [_strip_ext(ln) for ln in lines]

def infer_npz_keys(d):
    keys = list(d.keys())
    k_img = None
    k_msk = None
    k_id = None

    img_cands = ["images_u8", "images", "imgs", "x", "image", "img", "data", "X"]
    msk_cands = ["masks_u8", "masks", "mask", "y", "label", "labels", "Y", "gt", "gts"]
    id_cands  = ["names", "ids", "stems", "files", "filenames", "fnames"]

    for k in img_cands:
        if k in keys:
            k_img = k
            break
    for k in msk_cands:
        if k in keys:
            k_msk = k
            break
    for k in id_cands:
        if k in keys:
            k_id = k
            break

    if k_img is None:
        for k in keys:
            v = d[k]
            if isinstance(v, np.ndarray) and v.ndim in (4, 3) and v.shape[-1] in (1, 3):
                k_img = k
                break

    if k_msk is None:
        for k in keys:
            v = d[k]
            if isinstance(v, np.ndarray) and v.ndim in (3, 4):
                if k != k_img:
                    k_msk = k
                    break

    return k_img, k_msk, k_id

def sanitize_mask_to_012(mask):
    m = mask
    if m.ndim == 3 and m.shape[-1] == 3:
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    m = m.astype(np.int32)

    uniq = np.unique(m)
    if np.array_equal(np.sort(uniq), np.array([0, 1, 2], dtype=np.int32)):
        return m.astype(np.uint8)

    vals, cnts = np.unique(m, return_counts=True)
    bg_val = vals[np.argmax(cnts)]
    others = [v for v in vals if v != bg_val]
    if len(others) == 0:
        return np.zeros_like(m, np.uint8)

    areas = {v: int(cnts[list(vals).index(v)]) for v in others}
    disc_val = max(areas, key=areas.get)
    cup_val = None
    if len(others) >= 2:
        cup_val = min(areas, key=areas.get)

    disc = (m == disc_val)
    cup = (m == cup_val) if cup_val is not None else np.zeros_like(m, bool)
    ring = disc & (~cup)

    out = np.zeros_like(m, np.uint8)
    out[ring] = 1
    out[cup] = 2
    return out

def resize_to(img, mask, size):
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask

def aug_train(img, mask, size):
    if random.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        mask = np.ascontiguousarray(np.fliplr(mask))
    if random.random() < 0.15:
        img = np.ascontiguousarray(np.flipud(img))
        mask = np.ascontiguousarray(np.flipud(mask))

    if random.random() < 0.7:
        ang = random.uniform(-12, 12)
        scl = random.uniform(0.90, 1.10)
        tx = random.uniform(-0.02, 0.02) * img.shape[1]
        ty = random.uniform(-0.02, 0.02) * img.shape[0]
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, scl)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < 0.6:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= (1.0 + random.uniform(-0.12, 0.12))
        hsv[..., 2] *= (1.0 + random.uniform(-0.18, 0.18))
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if random.random() < 0.4:
        s = random.uniform(0.85, 1.15)
        new = int(round(size * s))
        img2 = cv2.resize(img, (new, new), interpolation=cv2.INTER_LINEAR)
        m2 = cv2.resize(mask, (new, new), interpolation=cv2.INTER_NEAREST)
        if new >= size:
            x0 = random.randint(0, new - size)
            y0 = random.randint(0, new - size)
            img = img2[y0:y0 + size, x0:x0 + size]
            mask = m2[y0:y0 + size, x0:x0 + size]
        else:
            pad = size - new
            px0 = random.randint(0, pad)
            py0 = random.randint(0, pad)
            img = cv2.copyMakeBorder(img2, py0, pad - py0, px0, pad - px0, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(m2, py0, pad - py0, px0, pad - px0, cv2.BORDER_CONSTANT, value=0)

    return img, mask

def parse_ce_weights(s: str):
    parts = [p.strip() for p in str(s).split(",")]
    w = [float(x) for x in parts]
    if len(w) != 3:
        raise RuntimeError("ce_weights/dice_tv_weights muszą mieć 3 liczby: bg, ring, cup")
    return w

# ---------------- dataset ----------------
class Refuge2NPZ(Dataset):
    def __init__(self, npz_path, idx_list, img_size=512, train=False, verbose=False):
        self.npz_path = str(npz_path)
        self.img_size = int(img_size)
        self.train = bool(train)
        self.verbose = bool(verbose)

        z = np.load(self.npz_path, allow_pickle=True)
        self._npz_keys = list(z.keys())
        k_img, k_msk, k_id = infer_npz_keys(z)
        if k_img is None or k_msk is None:
            raise RuntimeError(f"Nie umiem wykryć kluczy w npz. keys={self._npz_keys}")

        self.k_img = k_img
        self.k_msk = k_msk
        self.k_id = k_id

        self.images = z[k_img]
        self.masks = z[k_msk]

        if len(self.images) != len(self.masks):
            raise RuntimeError(f"N różne: images={len(self.images)} masks={len(self.masks)}")

        if k_id is not None:
            ids = z[k_id]
            if isinstance(ids, np.ndarray):
                self.ids = [_strip_ext(str(x)) for x in ids.tolist()]
            else:
                self.ids = [_strip_ext(str(x)) for x in list(ids)]
        else:
            self.ids = [str(i) for i in range(len(self.images))]

        want = [str(x) for x in idx_list]
        is_all_digits = all(re.fullmatch(r"\d+", x) for x in want)

        if is_all_digits:
            nums = [int(x) for x in want]
            n = len(self.ids)
            has0 = any(v == 0 for v in nums)
            maxv = max(nums) if nums else -1
            minv = min(nums) if nums else 0

            if (not has0) and (minv >= 1) and (maxv <= n):
                idx = [v - 1 for v in nums]
            else:
                idx = nums

            bad = [v for v in idx if v < 0 or v >= n]
            if bad:
                raise RuntimeError(f"Idx poza zakresem: przykłady {bad[:10]} (n={n}).")
        else:
            pos = {s: i for i, s in enumerate(self.ids)}
            missing = [s for s in want if s not in pos]
            if missing:
                raise RuntimeError(f"Brak {len(missing)} id w NPZ. Przykłady: {missing[:10]}")
            idx = [pos[s] for s in want]

        self.idx = idx

        if self.verbose:
            print(f"[{ts()}][data] npz={self.npz_path}", flush=True)
            print(f"[{ts()}][data] keys={self._npz_keys}", flush=True)
            print(f"[{ts()}][data] k_img={self.k_img} k_msk={self.k_msk} k_id={self.k_id}", flush=True)
            print(f"[{ts()}][data] images shape={self.images.shape} dtype={self.images.dtype}", flush=True)
            print(f"[{ts()}][data] masks  shape={self.masks.shape} dtype={self.masks.dtype}", flush=True)
            print(f"[{ts()}][data] subset n={len(self.idx)} train={self.train} img_size={self.img_size}", flush=True)
            ex = [self.ids[i] for i in self.idx[:8]]
            print(f"[{ts()}][data] examples={ex}", flush=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = self.images[j]
        msk = self.masks[j]

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if msk.dtype != np.uint8:
            msk = msk.astype(np.uint8)

        if msk.ndim == 3 and msk.shape[-1] == 3:
            msk = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
        if msk.ndim == 3 and msk.shape[-1] == 1:
            msk = msk[..., 0]

        msk = sanitize_mask_to_012(msk)

        img, msk = resize_to(img, msk, self.img_size)
        if self.train:
            img, msk = aug_train(img, msk, self.img_size)

        x = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
        y = torch.from_numpy(msk.astype(np.int64))

        meta = {"id": self.ids[j], "idx": int(j), "img_uint8": img, "msk_uint8": msk}
        return x, y, meta

def collate(batch):
    xs, ys, ms = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(ms)

# ---------------- graph builders ----------------
def build_grid_edges(h, w, grid4=True, device="cpu"):
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if not grid4:
        dirs = dirs + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    edges = []
    def nid(r, c): return r * w + c
    for r in range(h):
        for c in range(w):
            i = nid(r, c)
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    edges.append([i, nid(rr, cc)])
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    e = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    rev = e[[1, 0], :]
    return torch.cat([e, rev], dim=1)

def build_window_candidates(h, w, r, device):
    N = h * w
    offs = []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0:
                continue
            offs.append((dr, dc))

    cand = []
    for rr in range(h):
        for cc in range(w):
            neigh = []
            for dr, dc in offs:
                r2, c2 = rr + dr, cc + dc
                if 0 <= r2 < h and 0 <= c2 < w:
                    neigh.append(r2 * w + c2)
            if len(neigh) == 0:
                neigh = [rr * w + cc]
            cand.append(neigh)

    M = max(len(x) for x in cand)
    out = torch.empty((N, M), dtype=torch.long, device=device)
    for i in range(N):
        row = cand[i]
        if len(row) < M:
            row = row + [row[-1]] * (M - len(row))
        out[i] = torch.tensor(row, dtype=torch.long, device=device)
    return out

def build_dyn_edges_local_knn(feat, cand_idx, k):
    N, D = feat.shape
    M = cand_idx.shape[1]
    f = F.normalize(feat, dim=1)
    cand = f[cand_idx]
    sim = (f[:, None, :] * cand).sum(dim=-1)
    k_eff = min(int(k), int(M))
    top = torch.topk(sim, k_eff, dim=1).indices
    nbr = cand_idx.gather(1, top)
    src = torch.arange(N, device=feat.device, dtype=torch.long)[:, None].expand(N, k_eff).reshape(-1)
    dst = nbr.reshape(-1)
    e = torch.stack([src, dst], dim=0)
    rev = e[[1, 0], :]
    return torch.cat([e, rev], dim=1)

# ---------------- metrics ----------------
def dice_bin(pred_bin, gt_bin):
    p = pred_bin.astype(np.uint8)
    g = gt_bin.astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    if den == 0:
        return 1.0
    return float((2 * inter) / (den + 1e-6))

def dice_multiclass(pred, gt, C=3):
    out = []
    for c in range(C):
        p = (pred == c)
        g = (gt == c)
        inter = (p & g).sum()
        den = p.sum() + g.sum()
        if den == 0:
            out.append(1.0)
        else:
            out.append(float((2 * inter) / (den + 1e-6)))
    return np.array(out, dtype=np.float32)

@torch.no_grad()
def eval_metrics_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    gt = y.detach().cpu().numpy()

    cup_d = []
    disc_d = []
    for b in range(pred.shape[0]):
        cup_d.append(dice_bin((pred[b] == 2), (gt[b] == 2)))
        disc_d.append(dice_bin((pred[b] > 0), (gt[b] > 0)))
    return {
        "dice_C": float(np.mean(cup_d)),
        "dice_D": float(np.mean(disc_d)),
    }

# ---------------- losses (Twoje) ----------------
def tversky_loss(probs, y_onehot, alpha=0.62, beta=0.38, w=None, eps=1e-6):
    C = probs.shape[1]
    if w is None:
        w = [1.0] * C
    w = torch.tensor(w, device=probs.device, dtype=probs.dtype).view(1, C, 1, 1)

    tp = (probs * y_onehot).sum(dim=(2, 3), keepdim=True)
    fp = (probs * (1 - y_onehot)).sum(dim=(2, 3), keepdim=True)
    fn = ((1 - probs) * y_onehot).sum(dim=(2, 3), keepdim=True)

    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = (1 - tv) * w
    return loss.mean()

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    if gts == 0:
        return gt_sorted * 0.0
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - inter / (union + 1e-6)
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax(probs, labels, classes="present"):
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        pc = probs[:, c, :, :]
        errors = (fg - pc).abs().view(-1)
        fg = fg.view(-1)
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()

def boundary_target(mask_bin, device):
    x = torch.from_numpy(mask_bin.astype(np.float32))[None, None].to(device)
    ker = 3
    dil = F.max_pool2d(x, kernel_size=ker, stride=1, padding=ker // 2)
    ero = -F.max_pool2d(-x, kernel_size=ker, stride=1, padding=ker // 2)
    b = (dil - ero).clamp(0, 1)
    return b[0, 0]

def boundary_loss_from_probs(prob, tgt_b, w=0.35):
    ker = 3
    dil = F.max_pool2d(prob.unsqueeze(1), kernel_size=ker, stride=1, padding=ker // 2)
    ero = -F.max_pool2d(-prob.unsqueeze(1), kernel_size=ker, stride=1, padding=ker // 2)
    bpred = (dil - ero).squeeze(1).clamp(0, 1)
    with torch.autocast(device_type=prob.device.type, enabled=False):
        bpred_f = bpred.float().clamp(1e-6, 1.0 - 1e-6)
        tgt_f = tgt_b.float()
        return w * F.binary_cross_entropy(bpred_f, tgt_f)

def compute_losses(
    logits,
    y,
    ce_weights,
    tversky_on=True,
    tversky_alpha=0.62,
    tversky_beta=0.38,
    dice_tv_weights=(0.1, 1.3, 1.0),
    lovasz_on=True,
    boundary_on=True,
    boundary_w=0.35,
):
    ce_w = torch.tensor(ce_weights, device=logits.device, dtype=torch.float32)
    loss = F.cross_entropy(logits, y, weight=ce_w)

    probs = F.softmax(logits, dim=1).clamp(1e-6, 1.0)
    y_one = F.one_hot(y, num_classes=3).permute(0, 3, 1, 2).float()

    if tversky_on:
        loss = loss + tversky_loss(probs, y_one, alpha=tversky_alpha, beta=tversky_beta, w=list(dice_tv_weights))

    if lovasz_on:
        loss = loss + lovasz_softmax(probs, y, classes="present")

    if boundary_on:
        y_np = y.detach().cpu().numpy()
        dev = logits.device
        b_disc = []
        b_cup = []
        for b in range(y_np.shape[0]):
            disc = (y_np[b] > 0).astype(np.uint8)
            cup = (y_np[b] == 2).astype(np.uint8)
            b_disc.append(boundary_target(disc, device=dev))
            b_cup.append(boundary_target(cup, device=dev))
        b_disc = torch.stack(b_disc, 0)
        b_cup = torch.stack(b_cup, 0)
        loss = loss + boundary_loss_from_probs(probs[:, 1] + probs[:, 2], b_disc, w=boundary_w)
        loss = loss + boundary_loss_from_probs(probs[:, 2], b_cup, w=boundary_w)

    return loss

# ---------------- model bits ----------------
class GraphRefiner(nn.Module):
    def __init__(self, in_dim, hidden=512, depth=3, out_dim=3, dropout=0.1):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError("Brak torch_geometric.")
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)
        self.lin_out = nn.Linear(hidden, out_dim)

    def zero_init_output(self):
        nn.init.zeros_(self.lin_out.weight)
        if self.lin_out.bias is not None:
            nn.init.zeros_(self.lin_out.bias)

    def forward(self, x, edge_index):
        h = self.lin_in(x)
        for conv, norm in zip(self.convs, self.norms):
            h2 = conv(h, edge_index)
            h2 = F.gelu(h2)
            h2 = norm(h2)
            h = h + self.drop(h2)
        return self.lin_out(h), h

class _FeatureInfo:
    def __init__(self, channels, reductions):
        self._channels = list(channels)
        self._reductions = list(reductions)
    def channels(self):
        return self._channels
    def reduction(self):
        return self._reductions

class DeepLabV3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        if not TV_OK:
            raise RuntimeError("Brak torchvision segmentation.")
        self.m = tvseg.deeplabv3_resnet50(weights=None, weights_backbone=None)
        # best-effort meta
        self.feature_info = _FeatureInfo(channels=[2048, 1024], reductions=[16, 8])

    def forward(self, x):
        feats = self.m.backbone(x)  # dict: {'out':..., 'aux':...} usually
        if isinstance(feats, dict):
            out = feats.get("out", None)
            aux = feats.get("aux", None)
            if aux is not None and out is not None:
                return [aux, out]
            if out is not None:
                return [out]
            # fallback: first tensor
            for v in feats.values():
                if torch.is_tensor(v):
                    return [v]
            raise RuntimeError("DeepLab backbone zwrócił pusty dict.")
        if torch.is_tensor(feats):
            return [feats]
        raise RuntimeError(f"DeepLab backbone: nieznany typ wyjścia: {type(feats)}")


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class LocalOverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, H, W


class LocalDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class LocalMixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = LocalDWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocalEfficientAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = int(sr_ratio)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            xs = x.transpose(1, 2).reshape(B, C, H, W)
            xs = self.sr(xs).reshape(B, C, -1).transpose(1, 2)
            xs = self.norm(xs)
        else:
            xs = x
        kv = self.kv(xs).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class LocalMitBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalEfficientAttention(dim, num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LocalMixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class LocalMixVisionTransformer(nn.Module):
    """Small local MiT/SegFormer encoder used when mmseg/mmcv ops are absent."""

    def __init__(
        self,
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        patch_sizes=(7, 3, 3, 3),
        sr_ratios=(8, 4, 2, 1),
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.out_indices = tuple(out_indices)
        dims = [embed_dims, embed_dims * 2, embed_dims * 5, embed_dims * 8]
        in_ch = [in_channels] + dims[:-1]
        strides = [4, 2, 2, 2]
        dprs = torch.linspace(0, drop_path_rate, sum(num_layers)).tolist()
        cur = 0
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_stages):
            self.patch_embeds.append(LocalOverlapPatchEmbed(in_ch[i], dims[i], patch_sizes[i], strides[i]))
            stage = nn.ModuleList([
                LocalMitBlock(
                    dims[i], num_heads[i], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[i], qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dprs[cur + j]
                )
                for j in range(num_layers[i])
            ])
            cur += num_layers[i]
            self.blocks.append(stage)
            self.norms.append(nn.LayerNorm(dims[i]))
        self.feature_info = _FeatureInfo(channels=dims, reductions=[4, 8, 16, 32])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        outs = []
        for i, (patch, blocks, norm) in enumerate(zip(self.patch_embeds, self.blocks, self.norms)):
            x, H, W = patch(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            B, N, C = x.shape
            feat = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
            if i in self.out_indices:
                outs.append(feat)
            x = feat
        return outs


class SegFormerB0Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone_cls = MixVisionTransformer if MMSEG_OK else LocalMixVisionTransformer
        if not MMSEG_OK:
            print(f"[{ts()}][backbone] mmseg unavailable/incompatible; using local MiT-B0 fallback", flush=True)
        self.m = backbone_cls(
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )
        self.feature_info = _FeatureInfo(channels=[32, 64, 160, 256], reductions=[4, 8, 16, 32])

    def forward(self, x):
        feats = self.m(x)
        if isinstance(feats, (list, tuple)):
            return list(feats)
        return [feats]

def make_backbone(name: str):
    name0 = str(name).strip().lower()
    if name0 in ("hrnet_w48", "hrnet-w48"):
        if not TIMM_OK:
            raise RuntimeError("Brak timm, a prosisz o hrnet_w48.")
        # features_only daje listę feature maps + feature_info
        m = timm.create_model("hrnet_w48", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        return m

    if name0 in ("segformer_b0", "mit_b0", "segformer"):
        return SegFormerB0Backbone()

    if name0 in ("deeplabv3_resnet50", "deeplabv3"):
        return DeepLabV3Backbone()

    raise RuntimeError(f"Nieznany backbone: {name}. Użyj: hrnet_w48 | segformer_b0 | deeplabv3_resnet50")

class GraphSegUnified(nn.Module):
    def __init__(
        self,
        backbone="segformer_b0",
        num_classes=3,
        feat_dim=256,
        hidden=512,
        depth=3,
        graph_down=2,
        grid4=True,
        dyn_on="feat",
        dyn_k=4,
        dyn_window=2,
        alpha_graph=0.55,
        graph_output_mode="residual",
        graph_safety_gate="none",
        graph_gate_floor=0.0,
        graph_gate_power=1.0,
        graph_residual_clip=0.0,
    ):
        super().__init__()
        self.backbone_name = str(backbone)
        self.backbone = make_backbone(backbone)

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)

        # pick highest-res feature index if possible
        self.high_idx = None
        if hasattr(self.backbone, "feature_info"):
            try:
                reds = self.backbone.feature_info.reduction()
                if len(reds) > 0:
                    self.high_idx = int(np.argmin(np.array(reds, dtype=np.int64)))
            except Exception:
                self.high_idx = None

        # lazy heads
        self._proj_in_ch = None
        self.feat_proj = None
        self.seg_head = None

        # graph config
        self.graph_down = int(graph_down)
        self.grid4 = bool(grid4)
        self.dyn_on = str(dyn_on)
        self.dyn_k = int(dyn_k)
        self.dyn_window = int(dyn_window)
        self.alpha_graph = float(alpha_graph)
        self.graph_output_mode = normalize_graph_output_mode(graph_output_mode)
        self.graph_safety_gate = normalize_graph_safety_gate(graph_safety_gate)
        self.graph_gate_floor = float(graph_gate_floor)
        self.graph_gate_power = float(graph_gate_power)
        self.graph_residual_clip = float(graph_residual_clip)

        self.graph = None
        if self.alpha_graph > 0:
            self.graph = GraphRefiner(in_dim=self.feat_dim, hidden=hidden, depth=depth, out_dim=num_classes, dropout=0.1)
            if self.graph_output_mode == "residual":
                self.graph.zero_init_output()
        self._cache = {}

    def _cached(self, key, fn):
        v = self._cache.get(key, None)
        if v is None:
            v = fn()
            self._cache[key] = v
        return v

    def _get_feat(self, feats):
        if isinstance(feats, dict):
            if "aux" in feats and "out" in feats:
                feats = [feats["aux"], feats["out"]]
            else:
                feats = list(feats.values())
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        hi = self.high_idx
        if hi is None or hi < 0 or hi >= len(feats):
            return feats[-1]
        return feats[hi]

    def forward(self, x, dyn_on_eval=None, eval_dyn_k=None):
        B, _, H, W = x.shape
        feats = self.backbone(x)
        f = self._get_feat(feats)

        # lazy init heads
        in_ch = int(f.shape[1])
        if (self.feat_proj is None) or (self._proj_in_ch != in_ch):
            self._proj_in_ch = in_ch
            self.feat_proj = nn.Sequential(
                nn.Conv2d(in_ch, self.feat_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.feat_dim),
                nn.GELU(),
            ).to(f.device)
            self.seg_head = nn.Conv2d(self.feat_dim, self.num_classes, 1).to(f.device)

        f = self.feat_proj(f)
        logits = self.seg_head(f)
        logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        if self.alpha_graph <= 0:
            return logits_up, logits_up, None
        if self.graph is None:
            raise RuntimeError("Graph refiner is not initialized although alpha_graph > 0.")

        # graph resolution
        if self.graph_down > 1:
            f_g = F.avg_pool2d(f, kernel_size=self.graph_down, stride=self.graph_down)
        else:
            f_g = f

        _, Cg, Hg, Wg = f_g.shape
        N = Hg * Wg

        f_nodes = f_g.flatten(2).permute(0, 2, 1).contiguous()   # B,N,D
        f_nodes_all = f_nodes.view(B * N, Cg)

        edge_base = self._cached(
            ("grid", Hg, Wg, int(self.grid4), x.device.type),
            lambda: build_grid_edges(Hg, Wg, grid4=self.grid4, device=x.device),
        )

        if dyn_on_eval is None:
            dyn_on_eval = self.dyn_on
        if eval_dyn_k is None:
            eval_dyn_k = self.dyn_k

        use_dyn = (str(dyn_on_eval).lower() in ("feat", "feature", "features")) and (int(eval_dyn_k) > 0)

        if use_dyn:
            cand = self._cached(
                ("cand", Hg, Wg, int(self.dyn_window), x.device.type),
                lambda: build_window_candidates(Hg, Wg, self.dyn_window, device=x.device),
            )

        edges = []
        for b in range(B):
            e = edge_base + b * N
            if use_dyn:
                dyn = build_dyn_edges_local_knn(f_nodes[b], cand, k=eval_dyn_k) + b * N
                e = torch.cat([e, dyn], dim=1)
            edges.append(e)

        edge_index = torch.cat(edges, dim=1) if len(edges) else torch.zeros((2, 0), dtype=torch.long, device=x.device)

        node_logits_all, _ = self.graph(f_nodes_all, edge_index)
        node_logits = node_logits_all.view(B, N, -1).permute(0, 2, 1).contiguous().view(B, -1, Hg, Wg)
        node_logits_up = F.interpolate(node_logits, size=(H, W), mode="bilinear", align_corners=False)
        if self.graph_residual_clip > 0:
            node_logits_up = torch.clamp(node_logits_up, -self.graph_residual_clip, self.graph_residual_clip)

        if self.graph_output_mode == "residual":
            gate = self._safety_gate(logits_up)
            out = logits_up + self.alpha_graph * gate * node_logits_up
        elif self.graph_output_mode == "blend_legacy":
            out = (1.0 - self.alpha_graph) * logits_up + self.alpha_graph * node_logits_up
        else:
            raise RuntimeError(f"Unexpected graph_output_mode={self.graph_output_mode!r}")
        return out, logits_up, node_logits_up

    def _safety_gate(self, logits):
        mode = self.graph_safety_gate
        if mode == "none":
            return 1.0

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            if mode == "entropy":
                ent = -(probs * torch.clamp(probs, min=1e-6).log()).sum(dim=1, keepdim=True)
                gate = ent / math.log(float(self.num_classes))
            elif mode == "margin":
                top2 = torch.topk(probs, k=2, dim=1).values
                gate = 1.0 - (top2[:, 0:1] - top2[:, 1:2])
            else:
                raise RuntimeError(f"Unexpected graph_safety_gate={mode!r}")

            gate = torch.clamp(gate, 0.0, 1.0)
            if self.graph_gate_power != 1.0:
                gate = torch.pow(gate, self.graph_gate_power)
            if self.graph_gate_floor > 0:
                floor = min(max(self.graph_gate_floor, 0.0), 1.0)
                gate = floor + (1.0 - floor) * gate
        return gate

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=(1 - d))

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in getattr(self, "backup", {}):
                p.copy_(self.backup[n])
        self.backup = {}

# ---------------- refinement: MORPH + CRF ----------------
def _largest_cc_cv2(mask01):
    m = mask01.astype(np.uint8)
    num, lab = cv2.connectedComponents(m)
    if num <= 1:
        return m
    best = 1
    best_sz = 0
    for k in range(1, num):
        sz = int((lab == k).sum())
        if sz > best_sz:
            best_sz = sz
            best = k
    return (lab == best).astype(np.uint8)

def _fill_holes_cv2(mask01):
    m = mask01.astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, flood_inv)
    return (filled > 0).astype(np.uint8)

def morph_refine(pred012, k=7):
    pred = pred012.astype(np.uint8)
    disc = (pred > 0).astype(np.uint8)
    cup = (pred == 2).astype(np.uint8)

    disc = _largest_cc_cv2(disc)
    disc = _fill_holes_cv2(disc)

    cup = (cup & disc).astype(np.uint8)
    cup = _largest_cc_cv2(cup)
    cup = _fill_holes_cv2(cup)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    disc2 = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, ker, iterations=1)
    disc2 = cv2.morphologyEx(disc2, cv2.MORPH_OPEN, ker, iterations=1)
    disc2 = (disc2 > 0).astype(np.uint8)

    cup2 = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, ker, iterations=1)
    cup2 = cv2.morphologyEx(cup2, cv2.MORPH_OPEN, ker, iterations=1)
    cup2 = (cup2 > 0).astype(np.uint8)
    cup2 = (cup2 & disc2).astype(np.uint8)

    out = np.zeros_like(pred, np.uint8)
    out[(disc2 == 1) & (cup2 == 0)] = 1
    out[cup2 == 1] = 2
    return out

def crf_refine(img_rgb_u8, probs, sxy=3, compat=4, iters=5):
    if not DCRF_OK:
        return None
    H, W = img_rgb_u8.shape[:2]
    C = probs.shape[0]
    p = probs.astype(np.float32)
    p = np.clip(p, 1e-6, 1.0)
    p = p / np.sum(p, axis=0, keepdims=True)

    unary = dcrf_utils.unary_from_softmax(p)
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary)

    # gaussian pairwise
    d.addPairwiseGaussian(sxy=(sxy, sxy), compat=compat)

    # bilateral pairwise
    d.addPairwiseBilateral(sxy=(sxy*4, sxy*4), srgb=(13,13,13),
                           rgbim=img_rgb_u8, compat=compat)

    Q = d.inference(iters)
    Q = np.array(Q, dtype=np.float32).reshape((C, H, W))
    return Q

# ---------------- loops ----------------
def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    ema=None,
    amp=True,
    ce_weights=(0.2, 1.2, 1.0),
    tversky_on=True,
    tversky_alpha=0.62,
    tversky_beta=0.38,
    lovasz_on=True,
    boundary_on=True,
    boundary_w=0.35,
    dice_tv_weights=(0.1, 1.3, 1.0),
    dyn_on_eval=None,
    eval_dyn_k=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if is_train:
        set_frozen_host_eval(model)

    total_loss = 0.0
    n = 0
    mets_sum = defaultdict(float)

    for xb, yb, meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # label guard (łapie “device-side assert” wcześniej)
        with torch.no_grad():
            y_min = int(yb.min().item())
            y_max = int(yb.max().item())
            if y_min < 0 or y_max >= 3:
                ids = [mm.get("id", "?") for mm in meta]
                uniq = torch.unique(yb).detach().cpu().tolist()
                print(f"{ts()} [FATAL] bad labels! min={y_min} max={y_max} uniq={uniq} ids={ids}", flush=True)
                raise RuntimeError("Bad labels (out of range). Fix mask pipeline.")

        if is_train:
            use_amp = (amp and device.type == "cuda")
            amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                out, _, _ = model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
                loss = compute_losses(
                    out, yb,
                    ce_weights=ce_weights,
                    tversky_on=tversky_on,
                    tversky_alpha=tversky_alpha,
                    tversky_beta=tversky_beta,
                    dice_tv_weights=dice_tv_weights,
                    lovasz_on=lovasz_on,
                    boundary_on=boundary_on,
                    boundary_w=boundary_w,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
        else:
            with torch.no_grad():
                out, _, _ = model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
                loss = compute_losses(
                    out, yb,
                    ce_weights=ce_weights,
                    tversky_on=tversky_on,
                    tversky_alpha=tversky_alpha,
                    tversky_beta=tversky_beta,
                    dice_tv_weights=dice_tv_weights,
                    lovasz_on=lovasz_on,
                    boundary_on=boundary_on,
                    boundary_w=boundary_w,
                )

        bs = xb.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        n += bs

        m = eval_metrics_from_logits(out, yb)
        for k, v in m.items():
            mets_sum[k] += v * bs

    out_mets = {k: (v / max(1, n)) for k, v in mets_sum.items()}
    out_mets["loss"] = total_loss / max(1, n)
    return out_mets

@torch.no_grad()
def eval_refinements(model, loader, device, dyn_on_eval, eval_dyn_k, morph_k=7, crf_sxy=3, crf_compat=4, crf_iters=5, max_batches=None):
    model.eval()
    n = 0

    sum_raw_C = 0.0
    sum_raw_D = 0.0

    sum_morph_C = 0.0
    sum_morph_D = 0.0
    sum_crf_C = 0.0
    sum_crf_D = 0.0

    t_raw = 0.0
    t_morph = 0.0
    t_crf = 0.0

    for bi, (xb, yb, meta) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        t0 = time.time()
        logits, _, _ = model(xb, dyn_on_eval=dyn_on_eval, eval_dyn_k=eval_dyn_k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_raw += (time.time() - t0)

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        gt = yb.detach().cpu().numpy()

        bs = pred.shape[0]
        for i in range(bs):
            raw_C = dice_bin(pred[i] == 2, gt[i] == 2)
            raw_D = dice_bin(pred[i] > 0, gt[i] > 0)
            sum_raw_C += raw_C
            sum_raw_D += raw_D

        # MORPH
        t1 = time.time()
        pred_m = np.empty_like(pred)
        for i in range(bs):
            pred_m[i] = morph_refine(pred[i], k=morph_k)
        t_morph += (time.time() - t1)

        for i in range(bs):
            sum_morph_C += dice_bin(pred_m[i] == 2, gt[i] == 2)
            sum_morph_D += dice_bin(pred_m[i] > 0, gt[i] > 0)

        # CRF
        if DCRF_OK:
            t2 = time.time()
            pred_c = np.empty_like(pred)
            for i in range(bs):
                img_u8 = meta[i]["img_uint8"]  # RGB uint8
                Q = crf_refine(img_u8, probs[i], sxy=crf_sxy, compat=crf_compat, iters=crf_iters)
                if Q is None:
                    pred_c[i] = pred[i]
                else:
                    pred_c[i] = np.argmax(Q, axis=0).astype(np.uint8)
            t_crf += (time.time() - t2)

            for i in range(bs):
                sum_crf_C += dice_bin(pred_c[i] == 2, gt[i] == 2)
                sum_crf_D += dice_bin(pred_c[i] > 0, gt[i] > 0)
        n += bs

    def safe_div(a, b): return float(a / max(1, b))

    out = {
        "raw_CpD": safe_div(sum_raw_C + sum_raw_D, n),
        "raw_C": safe_div(sum_raw_C, n),
        "raw_D": safe_div(sum_raw_D, n),
        "raw_sec_img": safe_div(t_raw, n),

        "morph_CpD": safe_div(sum_morph_C + sum_morph_D, n),
        "morph_C": safe_div(sum_morph_C, n),
        "morph_D": safe_div(sum_morph_D, n),
        "morph_sec_img": safe_div(t_morph, n),
    }

    if DCRF_OK:
        out.update({
            "crf_CpD": safe_div(sum_crf_C + sum_crf_D, n),
            "crf_C": safe_div(sum_crf_C, n),
            "crf_D": safe_div(sum_crf_D, n),
            "crf_sec_img": safe_div(t_crf, n),
        })
    else:
        out.update({
            "crf_CpD": None, "crf_C": None, "crf_D": None, "crf_sec_img": None
        })
    return out

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="train", choices=["train", "eval", "smoke"])
    ap.add_argument("--npz_all", type=str, required=False)
    ap.add_argument("--idx_train", type=str, required=False)
    ap.add_argument("--idx_val", type=str, required=False)

    ap.add_argument("--backbone", type=str, default="segformer_b0")
    ap.add_argument("--img_size", type=int, default=512)

    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)

    ap.add_argument("--graph_down", type=int, default=2)
    ap.add_argument("--grid4", type=int, default=1)

    ap.add_argument("--dyn_on", type=str, default="feat")
    ap.add_argument("--dyn_k", type=int, default=4)
    ap.add_argument("--dyn_window", type=int, default=2)
    ap.add_argument("--dyn_on_eval", type=str, default="feat")
    ap.add_argument("--eval_dyn_k", type=int, default=4)

    ap.add_argument("--alpha_graph", type=float, default=0.55)
    ap.add_argument(
        "--graph_output_mode",
        type=str,
        default="residual",
        choices=GRAPH_OUTPUT_MODES,
        help="residual: Z_tilde = Z + alpha * DeltaZ; blend_legacy: old (1-alpha)*Z + alpha*G.",
    )
    ap.add_argument(
        "--graph_safety_gate",
        type=str,
        default="none",
        choices=GRAPH_SAFETY_GATES,
        help="none: apply residual everywhere; entropy/margin: gate residual by host uncertainty.",
    )
    ap.add_argument("--graph_gate_floor", type=float, default=0.0, help="Minimum residual gate value for safety-gated GRM.")
    ap.add_argument("--graph_gate_power", type=float, default=1.0, help="Exponent applied to safety gate; >1 makes refinement more conservative.")
    ap.add_argument("--graph_residual_clip", type=float, default=0.0, help="If >0, clamp DeltaZ before applying alpha/gate.")

    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)

    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--restart_T0", type=int, default=120)
    ap.add_argument("--restart_Tmult", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=3e-5)

    ap.add_argument("--tversky", type=int, default=1)
    ap.add_argument("--tversky_alpha", type=float, default=0.62)
    ap.add_argument("--tversky_beta", type=float, default=0.38)
    ap.add_argument("--lovasz", type=int, default=1)
    ap.add_argument("--boundary_on", type=int, default=1)
    ap.add_argument("--boundary_w", type=float, default=0.35)

    ap.add_argument("--ce_weights", type=str, default="0.2,1.2,1.0")
    ap.add_argument("--dice_tv_weights", type=str, default="0.1,1.3,1.0")

    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--epochs", type=int, default=80)

    ap.add_argument("--out_dir", type=str, default="runs/unified_graphseg")
    ap.add_argument("--run_name", type=str, default="run")

    ap.add_argument("--resume", type=str, default=None, help="ścieżka do ckpt/best.pt; loads the full model state.")
    ap.add_argument("--resume_host", type=str, default=None, help="Load host backbone/head weights from a no-graph checkpoint, skipping graph.*.")
    ap.add_argument("--freeze_host", type=int, default=0, help="If 1, train only graph.* parameters after loading/materializing the host.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--data_verbose", type=int, default=0)

    # eval refinements
    ap.add_argument("--morph_k", type=int, default=7)
    ap.add_argument("--crf_sxy", type=int, default=3)
    ap.add_argument("--crf_compat", type=int, default=4)
    ap.add_argument("--crf_iters", type=int, default=5)
    ap.add_argument("--eval_max_batches", type=int, default=0, help="0=all")

    # wandb
    ap.add_argument("--wandb", type=int, default=0)
    ap.add_argument("--wb_project", type=str, default="seg-refuge2")
    ap.add_argument("--wb_entity", type=str, default=None)
    ap.add_argument("--wb_mode", type=str, default="online")

    args = ap.parse_args()
    global ARGS
    ARGS = args

    if args.mode == "smoke":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{ts()}][smoke] device={device}", flush=True)
        print(f"[{ts()}][smoke] TIMM_OK={TIMM_OK} MMSEG_OK={MMSEG_OK} TV_OK={TV_OK} PYG_OK={PYG_OK} DCRF_OK={DCRF_OK}", flush=True)
        m = GraphSegUnified(
            backbone=args.backbone,
            feat_dim=args.feat_dim,
            hidden=args.hidden,
            depth=args.depth,
            graph_output_mode=args.graph_output_mode,
            graph_safety_gate=args.graph_safety_gate,
            graph_gate_floor=args.graph_gate_floor,
            graph_gate_power=args.graph_gate_power,
            graph_residual_clip=args.graph_residual_clip,
        ).to(device)
        x = torch.randn(2,3,args.img_size,args.img_size, device=device)
        with torch.no_grad():
            y, raw, g = m(x)
        print(f"[{ts()}][smoke] ok: out={tuple(y.shape)} raw={tuple(raw.shape)} graph={'None' if g is None else tuple(g.shape)}", flush=True)
        return

    # hard requirements for training/eval
    if args.npz_all is None or args.idx_val is None:
        raise RuntimeError("Dla train/eval podaj: --npz_all i --idx_val (i dla train także --idx_train).")

    if not PYG_OK:
        raise RuntimeError("Brak torch_geometric (PyG).")

    bname = str(args.backbone).strip().lower()
    if bname.startswith("hrnet"):
        if not TIMM_OK:
            raise RuntimeError("Chcesz hrnet, ale brak timm.")
    if bname.startswith("segformer") or bname.startswith("mit"):
        if not MMSEG_OK:
            print(f"[{ts()}][cfg] mmseg unavailable/incompatible; local MiT-B0 fallback is enabled", flush=True)
    if bname.startswith("deeplab"):
        if not TV_OK:
            raise RuntimeError("Chcesz deeplab, ale brak torchvision segmentation.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp) and device.type == "cuda"

    out_root = Path(args.out_dir) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ckpt").mkdir(parents=True, exist_ok=True)

    print(f"[{ts()}][cfg] mode={args.mode} device={device} amp={int(amp)}", flush=True)
    print(f"[{ts()}][cfg] backbone={args.backbone} img_size={args.img_size}", flush=True)
    print(f"[{ts()}][cfg] feat_dim={args.feat_dim} hidden={args.hidden} depth={args.depth}", flush=True)
    print(f"[{ts()}][cfg] alpha_graph={args.alpha_graph} graph_output_mode={args.graph_output_mode} graph_down={args.graph_down} grid4={args.grid4}", flush=True)
    print(f"[{ts()}][cfg] graph_safety_gate={args.graph_safety_gate} gate_floor={args.graph_gate_floor} gate_power={args.graph_gate_power} residual_clip={args.graph_residual_clip}", flush=True)
    print(f"[{ts()}][cfg] dyn_on={args.dyn_on} dyn_k={args.dyn_k} dyn_window={args.dyn_window} | eval: {args.dyn_on_eval}/{args.eval_dyn_k}", flush=True)
    print(f"[{ts()}][cfg] DCRF_OK={DCRF_OK} (pydensecrf)", flush=True)

    idx_va = read_idx_file(args.idx_val)
    val_ds = Refuge2NPZ(args.npz_all, idx_va, img_size=args.img_size, train=False, verbose=bool(args.data_verbose))
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(10, args.batch)),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,
        collate_fn=collate,
    )

    model = GraphSegUnified(
        backbone=args.backbone,
        num_classes=3,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        depth=args.depth,
        graph_down=args.graph_down,
        grid4=bool(args.grid4),
        dyn_on=args.dyn_on,
        dyn_k=args.dyn_k,
        dyn_window=args.dyn_window,
        alpha_graph=args.alpha_graph,
        graph_output_mode=args.graph_output_mode,
        graph_safety_gate=args.graph_safety_gate,
        graph_gate_floor=args.graph_gate_floor,
        graph_gate_power=args.graph_gate_power,
        graph_residual_clip=args.graph_residual_clip,
    ).to(device)

    # Materialize lazy heads (feat_proj/seg_head) BEFORE loading ckpt / creating optimizer.
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(_dummy, dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k)
    del _dummy

    # resume weights if provided (eval mode typically)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[{ts()}][ckpt] loaded={args.resume}", flush=True)
        if missing:
            print(f"[{ts()}][ckpt] missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"[{ts()}][ckpt] unexpected keys: {len(unexpected)}", flush=True)

    if args.resume_host is not None:
        _host_ckpt, missing, unexpected, missing_non_graph = load_host_weights(model, args.resume_host)
        print(f"[{ts()}][host] loaded host={args.resume_host}", flush=True)
        print(f"[{ts()}][host] missing={len(missing)} unexpected={len(unexpected)} missing_non_graph={len(missing_non_graph)}", flush=True)
        if missing_non_graph:
            print(f"[{ts()}][host] missing_non_graph sample={missing_non_graph[:8]}", flush=True)
            raise RuntimeError("Host checkpoint did not load cleanly outside graph.* keys.")
        if unexpected:
            print(f"[{ts()}][host] unexpected sample={unexpected[:8]}", flush=True)

    if int(args.freeze_host):
        freeze_host_params(model)
        print(f"[{ts()}][freeze] host frozen; only graph.* parameters are trainable", flush=True)

    total_params, trainable_params = count_params(model)
    print(f"[{ts()}][params] total={total_params} trainable={trainable_params} ({100.0 * trainable_params / max(1, total_params):.2f}%)", flush=True)

    if args.mode == "eval":
        max_batches = None if int(args.eval_max_batches) <= 0 else int(args.eval_max_batches)
        r = eval_refinements(
            model, val_loader, device,
            dyn_on_eval=args.dyn_on_eval,
            eval_dyn_k=args.eval_dyn_k,
            morph_k=args.morph_k,
            crf_sxy=args.crf_sxy,
            crf_compat=args.crf_compat,
            crf_iters=args.crf_iters,
            max_batches=max_batches,
        )
        print("\nrun\traw_CpD\traw_sec_img\tmorph_CpD\tmorph_sec_img\tcrf_CpD\tcrf_sec_img", flush=True)
        print(f"{args.run_name}\t{r['raw_CpD']:.6f}\t{r['raw_sec_img']:.4f}\t{r['morph_CpD']:.6f}\t{r['morph_sec_img']:.4f}\t"
              f"{('-' if r['crf_CpD'] is None else ('{:.6f}'.format(r['crf_CpD'])))}\t{('-' if r['crf_sec_img'] is None else ('{:.4f}'.format(r['crf_sec_img'])))}",
              flush=True)
        return

    # ---------------- TRAIN ----------------
    if args.idx_train is None:
        raise RuntimeError("Dla train podaj też --idx_train")

    idx_tr = read_idx_file(args.idx_train)
    train_ds = Refuge2NPZ(args.npz_all, idx_tr, img_size=args.img_size, train=True, verbose=bool(args.data_verbose))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=True,
        collate_fn=collate,
    )

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params_list:
        raise RuntimeError("No trainable parameters. Check --freeze_host / graph config.")

    opt = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.restart_T0, T_mult=args.restart_Tmult, eta_min=args.min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    ce_weights = parse_ce_weights(args.ce_weights)
    dice_tv_w  = parse_ce_weights(args.dice_tv_weights)

    wb = None
    if args.wandb and WANDB_OK:
        wb = wandb.init(project=args.wb_project, entity=args.wb_entity, name=args.run_name, mode=args.wb_mode)
        if wb:
            wandb.config.update(vars(args))

    best = -1.0
    best_ep = -1

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        if ep <= args.warmup_epochs:
            warm = ep / max(1, args.warmup_epochs)
            lr_now = args.lr * warm
            for pg in opt.param_groups:
                pg["lr"] = lr_now

        tr = run_epoch(
            model, train_loader, device,
            optimizer=opt, scaler=scaler, ema=ema, amp=amp,
            ce_weights=ce_weights,
            tversky_on=bool(args.tversky), tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
            lovasz_on=bool(args.lovasz),
            boundary_on=bool(args.boundary_on), boundary_w=args.boundary_w,
            dice_tv_weights=dice_tv_w,
            dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k,
        )

        if ep > args.warmup_epochs:
            sched.step(ep - args.warmup_epochs)

        use_ema = (ema is not None)
        if use_ema:
            ema.apply_to(model)

        va = run_epoch(
            model, val_loader, device,
            optimizer=None, scaler=None, ema=None, amp=amp,
            ce_weights=ce_weights,
            tversky_on=bool(args.tversky), tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
            lovasz_on=bool(args.lovasz),
            boundary_on=bool(args.boundary_on), boundary_w=args.boundary_w,
            dice_tv_weights=dice_tv_w,
            dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k,
        )

        lr_cur = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        score = va["dice_C"] + va["dice_D"]

        print(
            f"{ts()} Epoch {ep:03d} | "
            f"TL {tr['loss']:.4f} VL {va['loss']:.4f} | "
            f"Val C={va['dice_C']:.4f} D={va['dice_D']:.4f} C+D={score:.4f} | "
            f"lr={lr_cur:.2e} | {dt/60:.2f} min",
            flush=True
        )

        if wb:
            wandb.log(
                {"epoch": ep, "lr": lr_cur,
                 "train/loss": tr["loss"], "val/loss": va["loss"],
                 "val/dice_C": va["dice_C"], "val/dice_D": va["dice_D"], "val/CpD": score},
                step=ep
            )

        if score > best:
            best = score
            best_ep = ep
            ck = {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "best_score": best,
                "args": vars(args),
                "weights_source": "ema" if use_ema else "model",
            }
            torch.save(ck, out_root / "ckpt" / "best.pt")
            torch.save(ck, out_root / "ckpt" / f"ep{ep:03d}.pt")
            print(f"{ts()} [CKPT] new best(C+D)={best:.6f} @ ep={best_ep}", flush=True)

        if use_ema:
            ema.restore(model)

    print(f"{ts()} DONE best(C+D)={best:.6f} at ep={best_ep}", flush=True)
    if wb:
        wandb.run.summary["best_score_CpD"] = best
        wandb.run.summary["best_epoch"] = best_ep
        wandb.finish()

if __name__ == "__main__":
    main()
