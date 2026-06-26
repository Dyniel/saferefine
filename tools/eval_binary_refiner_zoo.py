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

from eval_binary_safe_refinement import dice_iou, fill_holes, geom_features, largest_cc, morph_refine  # noqa: E402
from eval_binary_safe_refinement import uncertainty_features  # noqa: E402
from train_binary_host import BinaryNPZ, collate, make_model, read_idx_file  # noqa: E402
from unified_graphseg import set_seed, ts  # noqa: E402


def parse_ints(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def open_close(mask, k):
    kk = int(k) if int(k) % 2 == 1 else int(k) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ker, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=1)
    return out.astype(np.uint8)


def close_only(mask, k):
    kk = int(k) if int(k) % 2 == 1 else int(k) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker, iterations=1).astype(np.uint8)


def open_only(mask, k):
    kk = int(k) if int(k) % 2 == 1 else int(k) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ker, iterations=1).astype(np.uint8)


def remove_small(mask, min_area):
    m = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return m
    out = np.zeros_like(m)
    for i in range(1, n):
        comp = lab == i
        if int(comp.sum()) >= int(min_area):
            out[comp] = 1
    return out.astype(np.uint8)


def prob_smooth(prob, sigma):
    sigma = float(sigma)
    k = max(3, int(round(sigma * 6)) | 1)
    p = cv2.GaussianBlur(prob.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)
    return (p >= 0.5).astype(np.uint8)


def bilateral_smooth(prob, sigma_color):
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    out = cv2.bilateralFilter(p, d=7, sigmaColor=float(sigma_color), sigmaSpace=5.0)
    return (out >= 0.5).astype(np.uint8)


def candidate_masks(host, prob, args):
    out = []
    for k in parse_ints(args.morph_ks):
        out.append(("binary_morph", float(k), morph_refine(host, k)))
        out.append(("close_only", float(k), close_only(host, k)))
        out.append(("open_close", float(k), open_close(host, k)))
    out.append(("fill_holes", 0.0, fill_holes(host)))
    out.append(("largest_cc", 0.0, largest_cc(host)))
    out.append(("lcc_fill", 0.0, fill_holes(largest_cc(host))))
    for min_area in parse_ints(args.min_areas):
        out.append(("remove_small", float(min_area), remove_small(host, min_area)))
    for sigma in parse_floats(args.gaussian_sigmas):
        out.append(("prob_gaussian", float(sigma), prob_smooth(prob, sigma)))
        out.append(("prob_gaussian_lcc", float(sigma), fill_holes(largest_cc(prob_smooth(prob, sigma)))))
    for sc in parse_floats(args.bilateral_sigmas):
        out.append(("prob_bilateral", float(sc), bilateral_smooth(prob, sc)))
    return out


def load_model(args, device):
    model = make_model(args, device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
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
    rows = []
    t0 = time.time()
    for bi, (xb, yb, meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        logits, _, _ = model(xb, dyn_on_eval="none", eval_dyn_k=0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        host = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        gt = yb.numpy().astype(np.uint8)
        for i in range(host.shape[0]):
            uncertainty = uncertainty_features(probs[i])
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
            seen = set()
            for method, alpha, pred in candidate_masks(host[i], probs[i], args):
                key = (method, float(alpha))
                if key in seen:
                    continue
                seen.add(key)
                d, j = dice_iou(pred, gt[i])
                rows.append({
                    "id": str(meta[i].get("id", "?")),
                    "idx": int(meta[i].get("idx", -1)),
                    "method": method,
                    "alpha": float(alpha),
                    "cup": d,
                    "disc": 0.0,
                    "cpd": d,
                    "iou": j,
                    **uncertainty,
                    **geom_features(pred, host[i]),
                })
        if args.max_batches > 0 and (bi + 1) >= args.max_batches:
            break
    print(f"[{ts()}][zoo] rows={len(rows)} images={len(set(r['id'] for r in rows))} sec={time.time()-t0:.1f}", flush=True)
    return rows


def summarize(rows):
    host = {r["id"]: r for r in rows if r["method"] == "host"}
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], float(row["alpha"])), []).append(row)
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--idx_eval", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--morph_ks", default="3,5,7,9,11")
    ap.add_argument("--min_areas", default="25,50,100,250")
    ap.add_argument("--gaussian_sigmas", default="0.75,1.25,2.0")
    ap.add_argument("--bilateral_sigmas", default="0.05,0.10,0.20")
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
    p = Path(args.out_summary)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
