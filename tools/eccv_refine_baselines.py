#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Optional CRF (pydensecrf)
# ----------------------------
CRF_OK = False
try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    CRF_OK = True
except Exception:
    CRF_OK = False


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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
    id_cands = ["names", "ids", "stems", "files", "filenames", "fnames"]

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


class NPZSeg(Dataset):
    def __init__(self, npz_path, idx_list, img_size=512):
        z = np.load(npz_path, allow_pickle=True)
        self.keys = list(z.keys())
        k_img, k_msk, k_id = infer_npz_keys(z)
        if k_img is None or k_msk is None:
            raise RuntimeError(f"Cannot infer keys in npz. keys={self.keys}")
        self.k_img, self.k_msk, self.k_id = k_img, k_msk, k_id
        self.images = z[k_img]
        self.masks = z[k_msk]
        if k_id is not None:
            ids = z[k_id]
            if isinstance(ids, np.ndarray):
                self.ids = [_strip_ext(str(x)) for x in ids.tolist()]
            else:
                self.ids = [_strip_ext(str(x)) for x in list(ids)]
        else:
            self.ids = [str(i) for i in range(len(self.images))]

        want = [str(x) for x in idx_list]
        pos = {s: i for i, s in enumerate(self.ids)}
        missing = [s for s in want if s not in pos]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} ids in NPZ, e.g. {missing[:10]}")
        self.idx = [pos[s] for s in want]
        self.img_size = int(img_size)

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

        x = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
        y = torch.from_numpy(msk.astype(np.int64))
        meta = {"id": self.ids[j], "img_u8": img, "msk_u8": msk}
        return x, y, meta


def collate(batch):
    xs, ys, ms = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(ms)


def dice_bin(pred_bin, gt_bin):
    p = pred_bin.astype(np.uint8)
    g = gt_bin.astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    if den == 0:
        return 1.0
    return float((2 * inter) / (den + 1e-6))


def eval_dice_CD_from_pred(pred_u8: np.ndarray, gt_u8: np.ndarray) -> Tuple[float, float]:
    # pred_u8, gt_u8: HxW in {0,1,2}
    cup = dice_bin((pred_u8 == 2), (gt_u8 == 2))
    disc = dice_bin((pred_u8 > 0), (gt_u8 > 0))
    return float(cup), float(disc)


# ----------------------------
# Morphology + CC baseline
# ----------------------------
def _largest_cc(bin_mask: np.ndarray) -> np.ndarray:
    # expects uint8 0/1
    num, lab = cv2.connectedComponents(bin_mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return bin_mask.astype(np.uint8)
    areas = []
    for k in range(1, num):
        areas.append((k, int((lab == k).sum())))
    areas.sort(key=lambda x: x[1], reverse=True)
    keep = areas[0][0]
    out = (lab == keep).astype(np.uint8)
    return out


def _fill_holes(bin_mask: np.ndarray) -> np.ndarray:
    # flood fill background from border
    m = bin_mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)  # mark outside as 1
    outside = ff
    holes = (outside == 0).astype(np.uint8)
    out = (m | holes).astype(np.uint8)
    return out


def morph_refine(pred_u8: np.ndarray,
                 k_close_disc: int = 9,
                 k_open_disc: int = 5,
                 k_close_cup: int = 7,
                 k_open_cup: int = 3) -> np.ndarray:
    # enforce cup inside disc; keep largest CC
    ring = (pred_u8 == 1).astype(np.uint8)
    cup = (pred_u8 == 2).astype(np.uint8)
    disc = ((pred_u8 > 0).astype(np.uint8))

    disc = _largest_cc(disc)
    disc = _fill_holes(disc)
    if k_close_disc > 1:
        disc = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_disc, k_close_disc)))
    if k_open_disc > 1:
        disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_disc, k_open_disc)))
    disc = (disc > 0).astype(np.uint8)

    cup = (cup & disc).astype(np.uint8)
    cup = _largest_cc(cup) if cup.sum() > 0 else cup
    cup = _fill_holes(cup) if cup.sum() > 0 else cup
    if cup.sum() > 0 and k_close_cup > 1:
        cup = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_cup, k_close_cup)))
    if cup.sum() > 0 and k_open_cup > 1:
        cup = cv2.morphologyEx(cup, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_cup, k_open_cup)))
    cup = (cup > 0).astype(np.uint8)

    ring = (disc & (1 - cup)).astype(np.uint8)

    out = np.zeros_like(pred_u8, dtype=np.uint8)
    out[ring == 1] = 1
    out[cup == 1] = 2
    return out


# ----------------------------
# DenseCRF baseline
# ----------------------------
def crf_refine(image_u8_rgb: np.ndarray, probs: np.ndarray,
               sxy_gaussian: int = 3,
               compat_gaussian: int = 3,
               sxy_bilateral: int = 60,
               srgb_bilateral: int = 10,
               compat_bilateral: int = 5,
               n_iters: int = 5) -> np.ndarray:
    """
    image_u8_rgb: HxWx3 uint8
    probs: CxHxW float32, sum over C = 1
    returns pred_u8 in {0,1,2}
    """
    if not CRF_OK:
        raise RuntimeError("pydensecrf not available")

    C, H, W = probs.shape
    d = dcrf.DenseCRF2D(W, H, C)
    U = crf_utils.unary_from_softmax(probs)  # (C, H*W)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)

    d.addPairwiseBilateral(
        sxy=sxy_bilateral, srgb=srgb_bilateral,
        rgbim=image_u8_rgb, compat=compat_bilateral
    )

    Q = d.inference(n_iters)
    Q = np.array(Q, dtype=np.float32).reshape((C, H, W))
    pred = np.argmax(Q, axis=0).astype(np.uint8)
    return pred


# ----------------------------
# Learned refiner
# ----------------------------
class TinyRefiner(nn.Module):
    def __init__(self, in_ch=3, hidden=32, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _infer_logits_from_base(model, xb, device) -> torch.Tensor:
    """
    model is expected to return either:
      - logits (B,C,H,W), or
      - tuple/list where first is logits
    """
    out = model(xb)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise RuntimeError("Model forward returned non-tensor")
    return out


def build_base_model_from_ckpt(ckpt_path: str, device: torch.device):
    """
    We support two cases:
    1) checkpoint saved by YOUR unified script and contains 'args' + state_dict, and model class is available in repo
       -> we try to import your GraphSeg definition from a local file if exists:
          - tools/unified_graphseg_train.py (optional)
          - or fallback: error with clear message.
    2) If you don't have a single import path, pass --base_script which has GraphSeg class (rare).
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise RuntimeError(f"Bad ckpt format: {ckpt_path}")

    # Try to locate a python file that defines GraphSeg + make_backbone etc
    # Common: you keep training script in repo root, so we'll import by path if user passes it.
    # If not, we can still do a workaround: require user to pass --model_py.
    return ck


def import_model_from_py(model_py: str):
    """
    Dynamically import python file that defines a function build_model_from_args(args_dict) -> nn.Module
    OR defines class GraphSeg compatible with args.
    We strongly recommend providing build_model_from_args for stability.
    """
    import importlib.util
    p = Path(model_py)
    if not p.exists():
        raise RuntimeError(f"--model_py not found: {model_py}")
    spec = importlib.util.spec_from_file_location("user_model_mod", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def safe_mean(xs):
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz_all", required=True, type=str)
    ap.add_argument("--idx_train", required=True, type=str)
    ap.add_argument("--idx_val", required=True, type=str)

    ap.add_argument("--img_size", default=512, type=int)
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--workers", default=4, type=int)

    ap.add_argument("--ckpt", required=True, type=str, help="Path to best.pt")
    ap.add_argument("--model_py", required=True, type=str,
                    help="Python file that defines build_model_from_args(args_dict)->nn.Module (recommended)")

    ap.add_argument("--device", default="cuda", type=str)

    # CRF params
    ap.add_argument("--crf_iters", default=5, type=int)
    ap.add_argument("--crf_sxy_g", default=3, type=int)
    ap.add_argument("--crf_compat_g", default=3, type=int)
    ap.add_argument("--crf_sxy_b", default=60, type=int)
    ap.add_argument("--crf_srgb_b", default=10, type=int)
    ap.add_argument("--crf_compat_b", default=5, type=int)

    # morph params
    ap.add_argument("--m_kc_disc", default=9, type=int)
    ap.add_argument("--m_ko_disc", default=5, type=int)
    ap.add_argument("--m_kc_cup", default=7, type=int)
    ap.add_argument("--m_ko_cup", default=3, type=int)

    # learned refiner
    ap.add_argument("--ref_epochs", default=15, type=int)
    ap.add_argument("--ref_lr", default=3e-3, type=float)
    ap.add_argument("--ref_hidden", default=32, type=int)

    ap.add_argument("--limit_val", default=0, type=int, help="0=all, else first N val samples (for quick sanity)")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[{ts()}][cfg] device={device} CRF_OK={int(CRF_OK)}", flush=True)

    # data
    tr_ids = read_idx_file(args.idx_train)
    va_ids = read_idx_file(args.idx_val)

    tr_ds = NPZSeg(args.npz_all, tr_ids, img_size=args.img_size)
    va_ds = NPZSeg(args.npz_all, va_ids, img_size=args.img_size)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
                           drop_last=True, collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=max(1, min(args.batch, 8)), shuffle=False,
                           num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
                           drop_last=False, collate_fn=collate)

    # model load
    ck = torch.load(args.ckpt, map_location="cpu")
    args_dict = ck.get("args", None)
    if args_dict is None:
        raise RuntimeError("ckpt does not contain 'args'. Add it when saving, or provide a custom builder in model_py.")
    if not isinstance(args_dict, dict):
        raise RuntimeError("ckpt['args'] must be a dict")

    mod = import_model_from_py(args.model_py)
    if hasattr(mod, "build_model_from_args"):
        base_model = mod.build_model_from_args(args_dict)
    else:
        raise RuntimeError(
            "model_py must define build_model_from_args(args_dict)->nn.Module. "
            "Create a tiny wrapper that constructs your model from saved args."
        )

    sd = ck["state_dict"]
    base_model.load_state_dict(sd, strict=True)
    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # ----------------------------
    # (1) RAW baseline evaluation
    # ----------------------------
    raw_c, raw_d = [], []
    crf_c, crf_d = [], []
    morph_c, morph_d = [], []

    t_raw = 0.0
    t_crf = 0.0
    t_morph = 0.0

    print(f"[{ts()}] eval: RAW / CRF / MORPH on val", flush=True)
    n_seen = 0

    with torch.no_grad():
        for xb, yb, meta in va_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            t0 = time.time()
            logits = _infer_logits_from_base(base_model, xb, device)  # (B,C,H,W)
            t_raw += (time.time() - t0)

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            pred = np.argmax(probs, axis=1).astype(np.uint8)
            gt = yb.detach().cpu().numpy().astype(np.uint8)

            B = pred.shape[0]
            for i in range(B):
                cup, disc = eval_dice_CD_from_pred(pred[i], gt[i])
                raw_c.append(cup); raw_d.append(disc)

            # MORPH
            t1 = time.time()
            for i in range(B):
                pm = morph_refine(
                    pred[i],
                    k_close_disc=args.m_kc_disc, k_open_disc=args.m_ko_disc,
                    k_close_cup=args.m_kc_cup, k_open_cup=args.m_ko_cup
                )
                cup, disc = eval_dice_CD_from_pred(pm, gt[i])
                morph_c.append(cup); morph_d.append(disc)
            t_morph += (time.time() - t1)

            # CRF (optional)
            if CRF_OK:
                t2 = time.time()
                for i in range(B):
                    img_u8 = meta[i]["img_u8"]
                    pr = probs[i].astype(np.float32)
                    pr = np.clip(pr, 1e-6, 1.0)
                    pr = pr / pr.sum(axis=0, keepdims=True)
                    pc = crf_refine(
                        img_u8, pr,
                        sxy_gaussian=args.crf_sxy_g,
                        compat_gaussian=args.crf_compat_g,
                        sxy_bilateral=args.crf_sxy_b,
                        srgb_bilateral=args.crf_srgb_b,
                        compat_bilateral=args.crf_compat_b,
                        n_iters=args.crf_iters
                    )
                    cup, disc = eval_dice_CD_from_pred(pc, gt[i])
                    crf_c.append(cup); crf_d.append(disc)
                t_crf += (time.time() - t2)

            n_seen += B
            if args.limit_val > 0 and n_seen >= args.limit_val:
                break

    # ----------------------------
    # (2) Learned refiner: train on (base logits -> tiny CNN -> refined logits)
    # ----------------------------
    print(f"[{ts()}] train: learned refiner on train split (base frozen)", flush=True)
    ref = TinyRefiner(in_ch=3, hidden=args.ref_hidden, out_ch=3).to(device)
    opt = torch.optim.Adam(ref.parameters(), lr=args.ref_lr)

    def run_ref_epoch(loader, train: bool):
        ref.train() if train else ref.eval()
        loss_sum = 0.0
        n_sum = 0
        c_list, d_list = [], []
        t_infer = 0.0
        t_ref = 0.0

        for xb, yb, meta in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.no_grad():
                t0 = time.time()
                logits = _infer_logits_from_base(base_model, xb, device)  # (B,3,H,W)
                t_infer += (time.time() - t0)

            t1 = time.time()
            out = ref(logits.detach())
            t_ref += (time.time() - t1)

            if train:
                loss = F.cross_entropy(out, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            else:
                loss = F.cross_entropy(out, yb)

            bs = xb.size(0)
            loss_sum += float(loss.detach().cpu()) * bs
            n_sum += bs

            pred = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.uint8)
            gt = yb.detach().cpu().numpy().astype(np.uint8)
            for i in range(pred.shape[0]):
                cup, disc = eval_dice_CD_from_pred(pred[i], gt[i])
                c_list.append(cup); d_list.append(disc)

        return {
            "loss": loss_sum / max(1, n_sum),
            "C": safe_mean(c_list),
            "D": safe_mean(d_list),
            "t_infer": t_infer,
            "t_ref": t_ref
        }

    best_val = -1.0
    best_state = None

    for ep in range(1, args.ref_epochs + 1):
        tr = run_ref_epoch(tr_loader, train=True)
        va = run_ref_epoch(va_loader, train=False)
        score = va["C"] + va["D"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
        print(f"{ts()} [ref] ep={ep:02d} trLoss={tr['loss']:.4f} vaLoss={va['loss']:.4f} "
              f"vaC={va['C']:.4f} vaD={va['D']:.4f} sum={score:.4f}", flush=True)

    if best_state is not None:
        ref.load_state_dict(best_state, strict=True)

    # final val eval for learned refiner
    print(f"[{ts()}] eval: learned refiner (best) on val", flush=True)
    ref_c, ref_d = [], []
    t_lr_infer = 0.0
    t_lr_ref = 0.0
    ref.eval()
    with torch.no_grad():
        n_seen = 0
        for xb, yb, meta in va_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            t0 = time.time()
            logits = _infer_logits_from_base(base_model, xb, device)
            t_lr_infer += (time.time() - t0)

            t1 = time.time()
            out = ref(logits)
            t_lr_ref += (time.time() - t1)

            pred = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.uint8)
            gt = yb.detach().cpu().numpy().astype(np.uint8)
            for i in range(pred.shape[0]):
                cup, disc = eval_dice_CD_from_pred(pred[i], gt[i])
                ref_c.append(cup); ref_d.append(disc)

            n_seen += pred.shape[0]
            if args.limit_val > 0 and n_seen >= args.limit_val:
                break

    # ----------------------------
    # Print summary table
    # ----------------------------
    rawC, rawD = safe_mean(raw_c), safe_mean(raw_d)
    morC, morD = safe_mean(morph_c), safe_mean(morph_d)

    if CRF_OK and len(crf_c) > 0:
        crfC, crfD = safe_mean(crf_c), safe_mean(crf_d)
        crf_note = "OK"
    else:
        crfC, crfD = float("nan"), float("nan")
        crf_note = "SKIP (no pydensecrf)"

    refC, refD = safe_mean(ref_c), safe_mean(ref_d)

    print("\n=== refinement baselines (VAL) ===")
    print("method            C(dice)    D(dice)    C+D       note", flush=True)
    print("---------------  --------  --------  --------  -----------------------------", flush=True)
    print(f"raw              {rawC:8.4f}  {rawD:8.4f}  {(rawC+rawD):8.4f}  base logits", flush=True)
    print(f"morph+cc         {morC:8.4f}  {morD:8.4f}  {(morC+morD):8.4f}  k_disc={args.m_kc_disc}/{args.m_ko_disc} k_cup={args.m_kc_cup}/{args.m_ko_cup}", flush=True)
    if CRF_OK and len(crf_c) > 0:
        print(f"densecrf         {crfC:8.4f}  {crfD:8.4f}  {(crfC+crfD):8.4f}  iters={args.crf_iters}", flush=True)
    else:
        print(f"densecrf         {crfC!s:>8}  {crfD!s:>8}  {'nan':>8}  {crf_note}", flush=True)
    print(f"learned_refiner  {refC:8.4f}  {refD:8.4f}  {(refC+refD):8.4f}  epochs={args.ref_epochs} hidden={args.ref_hidden}", flush=True)

    print("\n=== rough timings (seconds total, val pass) ===")
    print(f"raw_infer   : {t_raw:.2f}", flush=True)
    print(f"morph_post  : {t_morph:.2f}", flush=True)
    if CRF_OK:
        print(f"crf_post    : {t_crf:.2f}", flush=True)
    print(f"lr_infer    : {t_lr_infer:.2f}", flush=True)
    print(f"lr_ref_post : {t_lr_ref:.2f}", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
