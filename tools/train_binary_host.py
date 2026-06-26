#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from unified_graphseg import GraphSegUnified, set_seed, ts  # noqa: E402


def read_idx_file(path):
    return [int(x.strip()) for x in Path(path).read_text().splitlines() if x.strip()]


def infer_npz_keys(z):
    img_keys = ("images", "imgs", "x", "X")
    mask_keys = ("masks", "mask", "labels", "y", "Y", "gt")
    id_keys = ("ids", "id", "names", "stems")
    k_img = next((k for k in img_keys if k in z), None)
    k_mask = next((k for k in mask_keys if k in z), None)
    k_id = next((k for k in id_keys if k in z), None)
    if k_img is None or k_mask is None:
        raise RuntimeError(f"Cannot infer npz keys. keys={list(z.keys())}")
    return k_img, k_mask, k_id


class BinaryNPZ(Dataset):
    def __init__(self, npz_path, indices, img_size=352, train=False, verbose=False):
        self.npz_path = str(npz_path)
        self.indices = list(indices)
        self.img_size = int(img_size)
        self.train = bool(train)
        z = np.load(self.npz_path, allow_pickle=True)
        k_img, k_mask, k_id = infer_npz_keys(z)
        self.images = z[k_img]
        self.masks = z[k_mask]
        self.ids = z[k_id] if k_id is not None else np.asarray([str(i) for i in range(len(self.images))], dtype=object)
        if verbose:
            print(f"[{ts()}][data] npz={self.npz_path} images={self.images.shape} masks={self.masks.shape}", flush=True)

    def __len__(self):
        return len(self.indices)

    def _prep(self, img, mask):
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                mask = np.ascontiguousarray(np.fliplr(mask))
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(np.flipud(img))
                mask = np.ascontiguousarray(np.flipud(mask))
        img = img.astype(np.float32)
        if img.max() > 2.0:
            img = img / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img.transpose(2, 0, 1), mask.astype(np.int64)

    def __getitem__(self, i):
        j = int(self.indices[i])
        x, y = self._prep(self.images[j], self.masks[j])
        return torch.from_numpy(x), torch.from_numpy(y), {"id": str(self.ids[j]), "idx": j}


def collate(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)


def dice_iou_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    fg_p = pred == 1
    fg_y = y == 1
    inter = (fg_p & fg_y).sum(dim=(1, 2)).float()
    den = fg_p.sum(dim=(1, 2)).float() + fg_y.sum(dim=(1, 2)).float()
    union = (fg_p | fg_y).sum(dim=(1, 2)).float()
    dice = (2 * inter + 1e-6) / (den + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice.mean().item()), float(iou.mean().item())


def dice_loss(logits, y):
    probs = F.softmax(logits, dim=1)[:, 1]
    tgt = (y == 1).float()
    dims = (0, 1, 2)
    inter = (probs * tgt).sum(dims)
    den = probs.sum(dims) + tgt.sum(dims)
    return 1.0 - (2.0 * inter + 1e-5) / (den + 1e-5)


def boundary_loss(logits, y):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probs = F.softmax(logits.float(), dim=1)[:, 1:2]
        edge = torch.abs(F.avg_pool2d(probs, 3, stride=1, padding=1) - probs)
        edge = torch.clamp(edge * 8.0, 0.0, 1.0)
    yy = y.detach().cpu().numpy()
    targets = []
    for m in yy:
        k = np.ones((3, 3), np.uint8)
        er = cv2.erode((m > 0).astype(np.uint8), k, iterations=1)
        bd = ((m > 0).astype(np.uint8) - er).astype(np.float32)
        targets.append(torch.from_numpy(bd))
    tgt = torch.stack(targets, 0).to(logits.device, dtype=torch.float32).unsqueeze(1)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.binary_cross_entropy(edge, tgt)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=2, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = torch.nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.u4 = torch.nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = ConvBlock(base * 16, base * 8)
        self.u3 = torch.nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = torch.nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = torch.nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = torch.nn.Conv2d(base, num_classes, 1)

    def forward(self, x, **_kwargs):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1), None, None


def make_model(args, device):
    if str(getattr(args, "arch", "graphseg")).lower() in ("unet", "unet_small", "simple_unet"):
        return SimpleUNet(in_ch=3, num_classes=2, base=int(getattr(args, "unet_base", 32))).to(device)
    model = GraphSegUnified(
        backbone=args.backbone,
        num_classes=2,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        depth=args.depth,
        graph_down=args.graph_down,
        grid4=bool(args.grid4),
        dyn_on="none",
        dyn_k=0,
        alpha_graph=0.0,
        graph_output_mode="residual",
    ).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(dummy, dyn_on_eval="none", eval_dyn_k=0)
    return model


def run_epoch(model, loader, device, optimizer=None, amp=False, boundary_w=0.1, max_batches=0):
    train = optimizer is not None
    model.train(train)
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and train and device.type == "cuda"))
    for bi, (xb, yb, _meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        use_amp = bool(amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _, _ = model(xb, dyn_on_eval="none", eval_dyn_k=0)
            loss = F.cross_entropy(logits, yb) + dice_loss(logits, yb)
            if boundary_w > 0:
                loss = loss + float(boundary_w) * boundary_loss(logits, yb)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            d, j = dice_iou_from_logits(logits, yb)
        bs = xb.size(0)
        total_loss += float(loss.detach().item()) * bs
        total_dice += d * bs
        total_iou += j * bs
        n += bs
        if max_batches > 0 and (bi + 1) >= max_batches:
            break
    return {
        "loss": total_loss / max(1, n),
        "dice": total_dice / max(1, n),
        "iou": total_iou / max(1, n),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--idx_train", required=True)
    ap.add_argument("--idx_val", required=True)
    ap.add_argument("--out_dir", default="runs/binary_host")
    ap.add_argument("--run_name", default="binary_segformer_b0")
    ap.add_argument("--arch", default="graphseg")
    ap.add_argument("--backbone", default="segformer_b0")
    ap.add_argument("--unet_base", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--graph_down", type=int, default=2)
    ap.add_argument("--grid4", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--weight_decay", type=float, default=3e-5)
    ap.add_argument("--boundary_w", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)
    ap.add_argument("--data_verbose", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = BinaryNPZ(args.npz_all, read_idx_file(args.idx_train), args.img_size, train=True, verbose=bool(args.data_verbose))
    va = BinaryNPZ(args.npz_all, read_idx_file(args.idx_val), args.img_size, train=False, verbose=bool(args.data_verbose))
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate)
    model = make_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    run_dir = Path(args.out_dir) / args.run_name
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    best = -1.0
    hist = []
    print(f"[{ts()}][cfg] run_dir={run_dir} device={device} train={len(tr)} val={len(va)}", flush=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, tr_loader, device, optimizer=opt, amp=bool(args.amp), boundary_w=args.boundary_w, max_batches=args.max_train_batches)
        val_m = run_epoch(model, va_loader, device, optimizer=None, amp=False, boundary_w=0.0, max_batches=args.max_val_batches)
        sched.step()
        hist.append({"epoch": ep, "train": train_m, "val": val_m})
        if val_m["dice"] > best:
            best = val_m["dice"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": ep, "val": val_m}, run_dir / "ckpt" / "best.pt")
        print(
            f"{ts()} Ep {ep:03d} train_loss={train_m['loss']:.4f} val_dice={val_m['dice']:.6f} "
            f"val_iou={val_m['iou']:.6f} best={best:.6f} sec={time.time()-t0:.1f}",
            flush=True,
        )
        (run_dir / "history.json").write_text(json.dumps(hist, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
