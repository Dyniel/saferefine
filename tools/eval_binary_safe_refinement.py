#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_binary_host import BinaryNPZ, collate, make_model, read_idx_file  # noqa: E402
from unified_graphseg import set_seed, ts  # noqa: E402


def dice_iou(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = float(np.logical_and(p, g).sum())
    den = float(p.sum() + g.sum())
    union = float(np.logical_or(p, g).sum())
    dice = (2.0 * inter + 1e-6) / (den + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice, iou


def largest_cc(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.astype(np.uint8)
    areas = [(lab == i).sum() for i in range(1, n)]
    keep = 1 + int(np.argmax(areas))
    return (lab == keep).astype(np.uint8)


def fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = (1 - m).copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 2)
    holes = flood == 0
    return np.logical_or(m > 0, holes).astype(np.uint8)


def morph_refine(mask, k):
    if k <= 0:
        return mask.astype(np.uint8)
    kk = int(k) if int(k) % 2 == 1 else int(k) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker, iterations=1)
    out = fill_holes(largest_cc(out))
    return out.astype(np.uint8)


def binary_stats(mask):
    mask = mask.astype(bool)
    h, w = mask.shape
    area = float(mask.mean())
    if mask.any():
        yy, xx = np.nonzero(mask)
        cy = float(yy.mean() / max(1, h - 1))
        cx = float(xx.mean() / max(1, w - 1))
        comps = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    else:
        cy, cx, comps = 0.0, 0.0, 0
    return area, cy, cx, comps


def geom_features(pred, host):
    pa, py, px, pc = binary_stats(pred)
    ha, hy, hx, hc = binary_stats(host)
    shift = float(((py - hy) ** 2 + (px - hx) ** 2) ** 0.5)
    area_delta = abs(pa - ha)
    component_delta = abs(pc - hc)
    changed = float((pred != host).mean())
    geom_risk = changed + area_delta + 0.25 * shift + 0.01 * component_delta
    return {
        "changed": changed,
        "disc_area_delta": area_delta,
        "cup_area_delta": 0.0,
        "cup_disc_ratio_delta": 0.0,
        "disc_centroid_shift": shift,
        "cup_centroid_shift": 0.0,
        "component_delta": int(component_delta),
        "disc_changed": changed,
        "cup_changed": 0.0,
        "geom_risk": float(geom_risk),
    }


def uncertainty_features(prob):
    p = np.clip(prob.astype(np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    confidence = np.maximum(p, 1.0 - p)
    margin = np.abs(2.0 * p - 1.0)
    return {
        "host_entropy": float(entropy.mean()),
        "host_confidence": float(confidence.mean()),
        "host_margin": float(margin.mean()),
    }


def parse_ks(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def load_model(args, device):
    model = make_model(args, device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{ts()}][ckpt] loaded={args.ckpt}", flush=True)
    if missing:
        print(f"[{ts()}][ckpt] missing={len(missing)}", flush=True)
    if unexpected:
        print(f"[{ts()}][ckpt] unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BinaryNPZ(args.npz_all, read_idx_file(args.idx_eval), args.img_size, train=False, verbose=bool(args.data_verbose))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate)
    model = load_model(args, device)
    ks = parse_ks(args.morph_ks)
    rows = []
    t0 = time.time()
    for bi, (xb, yb, meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        logits, _, _ = model(xb, dyn_on_eval="none", eval_dyn_k=0)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        host = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        gt = yb.numpy().astype(np.uint8)
        for i in range(host.shape[0]):
            uncertainty = uncertainty_features(prob[i])
            d, j = dice_iou(host[i], gt[i])
            rows.append({
                "id": str(meta[i].get("id", "?")),
                "idx": int(meta[i].get("idx", -1)),
                "method": "host",
                "alpha": 0.0,
                "cup": d,
                "disc": 0.0,
                "cpd": d,
                "iou": j,
                **uncertainty,
                **geom_features(host[i], host[i]),
            })
            for k in ks:
                pred = morph_refine(host[i], k)
                d, j = dice_iou(pred, gt[i])
                rows.append({
                    "id": str(meta[i].get("id", "?")),
                    "idx": int(meta[i].get("idx", -1)),
                    "method": "binary_morph",
                    "alpha": float(k),
                    "cup": d,
                    "disc": 0.0,
                    "cpd": d,
                    "iou": j,
                    **uncertainty,
                    **geom_features(pred, host[i]),
                })
        if args.max_batches > 0 and (bi + 1) >= args.max_batches:
            break
    print(f"[{ts()}][eval] rows={len(rows)} images={len(set(r['id'] for r in rows))} sec={time.time()-t0:.1f}", flush=True)
    return rows


def summarize(rows):
    host = {r["id"]: r for r in rows if r["method"] == "host"}
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], float(r["alpha"])), []).append(r)
    out = []
    for (method, alpha), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gains = np.asarray([r["cpd"] - host[r["id"]]["cpd"] for r in rs], dtype=np.float64)
        cpds = np.asarray([r["cpd"] for r in rs], dtype=np.float64)
        ious = np.asarray([r["iou"] for r in rs], dtype=np.float64)
        out.append({
            "method": method,
            "alpha": alpha,
            "n": len(rs),
            "mean_dice": float(cpds.mean()),
            "mean_iou": float(ious.mean()),
            "mean_gain": float(gains.mean()),
            "mean_harm": float(np.maximum(0.0, -gains).mean()),
            "worst_drop": float(gains.min()),
            "improved_rate": float((gains > 0).mean()),
            "harmed_rate": float((gains < 0).mean()),
        })
    return out


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--idx_eval", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--morph_ks", default="3,5,7,9")
    ap.add_argument("--arch", default="graphseg")
    ap.add_argument("--backbone", default="segformer_b0")
    ap.add_argument("--unet_base", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--graph_down", type=int, default=2)
    ap.add_argument("--grid4", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--data_verbose", type=int, default=0)
    args = ap.parse_args()

    rows = evaluate(args)
    summary = summarize(rows)
    print("\nmethod\talpha\tn\tmean_dice\tmean_iou\tmean_gain\tmean_harm\tworst_drop\timproved_rate\tharmed_rate", flush=True)
    for s in summary:
        print(
            f"{s['method']}\t{s['alpha']:.2f}\t{s['n']}\t{s['mean_dice']:.6f}\t{s['mean_iou']:.6f}\t"
            f"{s['mean_gain']:+.6f}\t{s['mean_harm']:.6f}\t{s['worst_drop']:+.6f}\t"
            f"{s['improved_rate']:.4f}\t{s['harmed_rate']:.4f}",
            flush=True,
        )
    write_csv(args.out_csv, rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
