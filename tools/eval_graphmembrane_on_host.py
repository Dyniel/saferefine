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
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_graphmembrane_refiner import GraphMembraneRefiner, labels_to_logits  # noqa: E402
from unified_graphseg import (  # noqa: E402
    GraphSegUnified,
    Refuge2NPZ,
    collate,
    dice_bin,
    morph_refine,
    read_idx_file,
    set_seed,
    ts,
)


def parse_alphas(s):
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return sorted(set(vals))


def dice_cpd(pred, gt):
    cup = dice_bin(pred == 2, gt == 2)
    disc = dice_bin(pred > 0, gt > 0)
    return cup, disc, cup + disc


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
        cy = 0.0
        cx = 0.0
        comps = 0
    return area, cy, cx, comps


def geometric_features(pred, host_pred):
    host_disc = host_pred > 0
    host_cup = host_pred == 2
    pred_disc = pred > 0
    pred_cup = pred == 2

    hd_area, hd_y, hd_x, hd_comp = binary_stats(host_disc)
    hc_area, hc_y, hc_x, hc_comp = binary_stats(host_cup)
    pd_area, pd_y, pd_x, pd_comp = binary_stats(pred_disc)
    pc_area, pc_y, pc_x, pc_comp = binary_stats(pred_cup)

    host_ratio = hc_area / max(hd_area, 1e-8)
    pred_ratio = pc_area / max(pd_area, 1e-8)
    disc_shift = float(((pd_y - hd_y) ** 2 + (pd_x - hd_x) ** 2) ** 0.5)
    cup_shift = float(((pc_y - hc_y) ** 2 + (pc_x - hc_x) ** 2) ** 0.5)
    comp_delta = abs(pd_comp - hd_comp) + abs(pc_comp - hc_comp)
    disc_changed = float((pred_disc != host_disc).mean())
    cup_changed = float((pred_cup != host_cup).mean())
    disc_area_delta = abs(pd_area - hd_area)
    cup_area_delta = abs(pc_area - hc_area)
    ratio_delta = abs(pred_ratio - host_ratio)
    geom_risk = (
        disc_changed
        + cup_changed
        + disc_area_delta
        + cup_area_delta
        + 0.25 * (disc_shift + cup_shift)
        + 0.01 * comp_delta
        + 0.10 * ratio_delta
    )
    return {
        "host_disc_area": hd_area,
        "host_cup_area": hc_area,
        "pred_disc_area": pd_area,
        "pred_cup_area": pc_area,
        "disc_area_delta": disc_area_delta,
        "cup_area_delta": cup_area_delta,
        "cup_disc_ratio_delta": ratio_delta,
        "disc_centroid_shift": disc_shift,
        "cup_centroid_shift": cup_shift,
        "disc_components": pd_comp,
        "cup_components": pc_comp,
        "component_delta": int(comp_delta),
        "disc_changed": disc_changed,
        "cup_changed": cup_changed,
        "geom_risk": float(geom_risk),
    }


def load_host(args, device):
    model = GraphSegUnified(
        backbone=args.backbone,
        num_classes=3,
        feat_dim=args.feat_dim,
        hidden=args.host_hidden,
        depth=args.host_depth,
        graph_down=args.graph_down,
        grid4=bool(args.grid4),
        dyn_on=args.dyn_on,
        dyn_k=args.dyn_k,
        dyn_window=args.dyn_window,
        alpha_graph=0.0,
        graph_output_mode="residual",
    ).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(dummy, dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k)
    ckpt = torch.load(args.host_ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{ts()}][host] loaded={args.host_ckpt}", flush=True)
    if missing:
        print(f"[{ts()}][host] missing={len(missing)}", flush=True)
    if unexpected:
        print(f"[{ts()}][host] unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def load_membrane(args, device):
    ckpt = torch.load(args.membrane_ckpt, map_location="cpu")
    ck_args = ckpt.get("args", {})
    model = GraphMembraneRefiner(
        hidden=int(ck_args.get("hidden", args.hidden)),
        steps=int(ck_args.get("steps", args.steps)),
        dt=float(ck_args.get("dt", args.dt)),
        alpha=1.0,
        residual_clip=float(ck_args.get("residual_clip", args.residual_clip)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"[{ts()}][membrane] loaded={args.membrane_ckpt}", flush=True)
    print(
        f"[{ts()}][membrane] hidden={ck_args.get('hidden', args.hidden)} steps={ck_args.get('steps', args.steps)} "
        f"dt={ck_args.get('dt', args.dt)} clip={ck_args.get('residual_clip', args.residual_clip)}",
        flush=True,
    )
    return model


@torch.no_grad()
def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphas = parse_alphas(args.alphas)
    idx = read_idx_file(args.idx_val)
    ds = Refuge2NPZ(args.npz_all, idx, img_size=args.img_size, train=False, verbose=bool(args.data_verbose))
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,
        collate_fn=collate,
    )

    host = load_host(args, device)
    membrane = load_membrane(args, device)
    rows = []
    t0 = time.time()

    for bi, (xb, yb, meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        y_np = yb.numpy()
        host_logits, _, _ = host(xb, dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k)
        host_pred_t = torch.argmax(host_logits, dim=1)
        host_pred = host_pred_t.detach().cpu().numpy()

        if args.input_mode == "hard":
            membrane_input = labels_to_logits(host_pred_t, base_logit=args.base_logit)
        elif args.input_mode == "soft":
            membrane_input = torch.clamp(host_logits.detach(), -args.soft_clip, args.soft_clip)
        else:
            raise RuntimeError(f"Unknown input_mode={args.input_mode!r}")

        bs = xb.size(0)
        for i in range(bs):
            cup, disc, cpd = dice_cpd(host_pred[i], y_np[i])
            rows.append({
                "id": str(meta[i].get("id", "?")),
                "idx": int(meta[i].get("idx", -1)),
                "method": "host",
                "alpha": 0.0,
                "cup": cup,
                "disc": disc,
                "cpd": cpd,
                "changed": 0.0,
                **geometric_features(host_pred[i], host_pred[i]),
            })
            pred_m = morph_refine(host_pred[i], k=args.morph_k)
            cup, disc, cpd = dice_cpd(pred_m, y_np[i])
            rows.append({
                "id": str(meta[i].get("id", "?")),
                "idx": int(meta[i].get("idx", -1)),
                "method": "host_morph",
                "alpha": 0.0,
                "cup": cup,
                "disc": disc,
                "cpd": cpd,
                "changed": float((pred_m != host_pred[i]).mean()),
                **geometric_features(pred_m, host_pred[i]),
            })

        for alpha in alphas:
            membrane.alpha = float(alpha)
            refined, _q, aux = membrane(xb, membrane_input)
            pred = torch.argmax(refined, dim=1).detach().cpu().numpy()
            for i in range(bs):
                cup, disc, cpd = dice_cpd(pred[i], y_np[i])
                rows.append({
                    "id": str(meta[i].get("id", "?")),
                    "idx": int(meta[i].get("idx", -1)),
                    "method": "graphmembrane",
                    "alpha": float(alpha),
                    "cup": cup,
                    "disc": disc,
                    "cpd": cpd,
                    "changed": float((pred[i] != host_pred[i]).mean()),
                    "force_abs": float(aux["force_abs"].detach().cpu()),
                    "stiffness": float(aux["stiffness"].detach().cpu()),
                    "damping": float(aux["damping"].detach().cpu()),
                    **geometric_features(pred[i], host_pred[i]),
                })

        if args.max_batches > 0 and (bi + 1) >= args.max_batches:
            break

    print(f"[{ts()}][eval] rows={len(rows)} images={len(set(r['id'] for r in rows))} sec={time.time()-t0:.1f}", flush=True)
    return rows


def summarize(rows):
    host = {r["id"]: r for r in rows if r["method"] == "host"}
    groups = {}
    for r in rows:
        key = (r["method"], float(r["alpha"]))
        groups.setdefault(key, []).append(r)

    out = []
    for (method, alpha), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gains = np.asarray([r["cpd"] - host[r["id"]]["cpd"] for r in rs], dtype=np.float64)
        cpds = np.asarray([r["cpd"] for r in rs], dtype=np.float64)
        cups = np.asarray([r["cup"] for r in rs], dtype=np.float64)
        discs = np.asarray([r["disc"] for r in rs], dtype=np.float64)
        changed = np.asarray([r.get("changed", 0.0) for r in rs], dtype=np.float64)
        out.append({
            "method": method,
            "alpha": alpha,
            "n": int(len(rs)),
            "mean_cpd": float(cpds.mean()),
            "mean_cup": float(cups.mean()),
            "mean_disc": float(discs.mean()),
            "mean_gain": float(gains.mean()),
            "mean_harm": float(np.maximum(0.0, -gains).mean()),
            "worst_drop": float(gains.min()),
            "improved_rate": float((gains > 0).mean()),
            "harmed_rate": float((gains < 0).mean()),
            "changed": float(changed.mean()),
        })
    return out


def write_csv(path, rows):
    if not path:
        return
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
    ap.add_argument("--idx_val", required=True)
    ap.add_argument("--host_ckpt", required=True)
    ap.add_argument("--membrane_ckpt", required=True)
    ap.add_argument("--input_mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--alphas", default="0,0.25,0.50,0.75,1.00")
    ap.add_argument("--base_logit", type=float, default=3.0)
    ap.add_argument("--soft_clip", type=float, default=5.0)
    ap.add_argument("--morph_k", type=int, default=7)

    ap.add_argument("--backbone", type=str, default="segformer_b0")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--host_hidden", type=int, default=512)
    ap.add_argument("--host_depth", type=int, default=3)
    ap.add_argument("--graph_down", type=int, default=2)
    ap.add_argument("--grid4", type=int, default=1)
    ap.add_argument("--dyn_on", type=str, default="feat")
    ap.add_argument("--dyn_k", type=int, default=16)
    ap.add_argument("--dyn_window", type=int, default=2)
    ap.add_argument("--dyn_on_eval", type=str, default="feat")
    ap.add_argument("--eval_dyn_k", type=int, default=16)

    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.35)
    ap.add_argument("--residual_clip", type=float, default=5.0)

    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--data_verbose", type=int, default=0)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--out_summary", default="")
    args = ap.parse_args()

    print(f"[{ts()}][cfg] host={args.host_ckpt}", flush=True)
    print(f"[{ts()}][cfg] membrane={args.membrane_ckpt}", flush=True)
    print(f"[{ts()}][cfg] input_mode={args.input_mode} alphas={args.alphas}", flush=True)
    rows = evaluate(args)
    summary = summarize(rows)

    print("\nmethod\talpha\tn\tmean_cpd\tmean_gain\tmean_harm\tworst_drop\timproved_rate\tharmed_rate\tchanged", flush=True)
    for s in summary:
        print(
            f"{s['method']}\t{s['alpha']:.2f}\t{s['n']}\t{s['mean_cpd']:.6f}\t{s['mean_gain']:+.6f}\t"
            f"{s['mean_harm']:.6f}\t{s['worst_drop']:+.6f}\t{s['improved_rate']:.4f}\t"
            f"{s['harmed_rate']:.4f}\t{s['changed']:.6f}",
            flush=True,
        )

    write_csv(args.out_csv, rows)
    if args.out_summary:
        p = Path(args.out_summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
