#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_binary_safe_refinement import dice_iou, geom_features, uncertainty_features  # noqa: E402
from train_binary_host import BinaryNPZ, collate, make_model, read_idx_file  # noqa: E402
from unified_graphseg import set_seed, ts  # noqa: E402


def model_args(args, prefix):
    return SimpleNamespace(
        arch=getattr(args, f"{prefix}_arch"),
        backbone=getattr(args, f"{prefix}_backbone"),
        unet_base=getattr(args, f"{prefix}_unet_base"),
        img_size=args.img_size,
        feat_dim=getattr(args, f"{prefix}_feat_dim"),
        hidden=getattr(args, f"{prefix}_hidden"),
        depth=getattr(args, f"{prefix}_depth"),
        graph_down=getattr(args, f"{prefix}_graph_down"),
        grid4=getattr(args, f"{prefix}_grid4"),
    )


def load_model(args, prefix, device):
    margs = model_args(args, prefix)
    model = make_model(margs, device)
    ckpt_path = getattr(args, f"{prefix}_ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[{ts()}][ckpt] {prefix} loaded={ckpt_path}", flush=True)
    if missing:
        print(f"[{ts()}][ckpt] {prefix} missing={len(missing)}", flush=True)
    if unexpected:
        print(f"[{ts()}][ckpt] {prefix} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def area_frac(mask):
    return float(np.asarray(mask, dtype=np.uint8).mean())


def comp_count(mask):
    n, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    return float(max(0, n - 1))


def feature_row(host_mask, ref_mask, host_prob, ref_prob):
    host_area = area_frac(host_mask)
    ref_area = area_frac(ref_mask)
    changed = float((host_mask != ref_mask).mean())
    ref_unc = uncertainty_features(ref_prob)
    feats = {
        "changed": changed,
        "host_area": host_area,
        "ref_area": ref_area,
        "area_ratio": float(ref_area / max(host_area, 1e-6)),
        "area_delta_abs": float(abs(ref_area - host_area)),
        "ref_components": comp_count(ref_mask),
        "host_components": comp_count(host_mask),
        "ref_entropy": float(ref_unc["host_entropy"]),
        "ref_low_confidence": float(max(0.0, 1.0 - ref_unc["host_confidence"])),
        "ref_margin_inverse": float(max(0.0, 1.0 - ref_unc["host_margin"])),
    }
    feats["guard_risk"] = (
        feats["changed"]
        + feats["area_delta_abs"]
        + 0.05 * max(0.0, feats["ref_components"] - 1.0)
        + 0.50 * feats["ref_low_confidence"]
        + 0.25 * feats["ref_margin_inverse"]
    )
    return feats


@torch.no_grad()
def collect_predictions(args, idx_file, split_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BinaryNPZ(args.npz_all, read_idx_file(idx_file), args.img_size, train=False, verbose=bool(args.data_verbose))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate)
    host_model = load_model(args, "host", device)
    ref_model = load_model(args, "refiner", device)

    cases = []
    t0 = time.time()
    for bi, (xb, yb, meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        host_logits, _, _ = host_model(xb, dyn_on_eval="none", eval_dyn_k=0)
        ref_logits, _, _ = ref_model(xb, dyn_on_eval="none", eval_dyn_k=0)
        host_prob = torch.softmax(host_logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        ref_prob = torch.softmax(ref_logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        host_mask = torch.argmax(host_logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        ref_mask = torch.argmax(ref_logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        gt = yb.numpy().astype(np.uint8)
        for i in range(host_mask.shape[0]):
            hd, hiou = dice_iou(host_mask[i], gt[i])
            rd, riou = dice_iou(ref_mask[i], gt[i])
            feats = feature_row(host_mask[i], ref_mask[i], host_prob[i], ref_prob[i])
            cases.append({
                "split_name": split_name,
                "id": str(meta[i].get("id", "?")),
                "idx": int(meta[i].get("idx", -1)),
                "host_dice": float(hd),
                "host_iou": float(hiou),
                "refiner_dice": float(rd),
                "refiner_iou": float(riou),
                "gain": float(rd - hd),
                **feats,
                "host_uncertainty": uncertainty_features(host_prob[i]),
                "geom": geom_features(ref_mask[i], host_mask[i]),
            })
        if args.max_batches > 0 and (bi + 1) >= args.max_batches:
            break
    print(f"[{ts()}][collect] split={split_name} cases={len(cases)} sec={time.time()-t0:.1f}", flush=True)
    return cases


def summarize_gain(cases, pred_key):
    gains = np.asarray([c[pred_key] - c["host_dice"] for c in cases], dtype=np.float64)
    return {
        "n": int(len(cases)),
        "mean_gain": float(gains.mean()) if len(gains) else 0.0,
        "mean_harm": float(np.maximum(0.0, -gains).mean()) if len(gains) else 0.0,
        "worst_drop": float(gains.min()) if len(gains) else 0.0,
        "harmed_rate": float((gains < 0.0).mean()) if len(gains) else 0.0,
        "drop05_rate": float((gains < -0.05).mean()) if len(gains) else 0.0,
        "improved_rate": float((gains > 0.0).mean()) if len(gains) else 0.0,
    }


def tune_guard(cases, args):
    if not cases:
        return {"guard_threshold": 0.0, "summary": {}}
    risks = sorted({0.0, *[float(c["guard_risk"]) for c in cases]})
    best = None
    candidates = []
    for thr in risks:
        selected = []
        for c in cases:
            use_ref = float(c["guard_risk"]) <= thr + 1e-12
            selected.append(c["refiner_dice"] if use_ref else c["host_dice"])
        tmp = [dict(c, guarded_dice=float(d)) for c, d in zip(cases, selected)]
        s = summarize_gain(tmp, "guarded_dice")
        s["guard_threshold"] = float(thr)
        s["utility"] = float(s["mean_gain"] - args.beta_harm * s["mean_harm"])
        candidates.append(s)
        feasible = (
            s["harmed_rate"] <= args.guard_max_harm_rate
            and s["drop05_rate"] <= args.guard_max_drop05_rate
            and s["mean_harm"] <= args.guard_max_mean_harm
            and s["mean_gain"] > args.guard_min_gain
        )
        if feasible and (best is None or (s["utility"], s["mean_gain"], -s["mean_harm"]) > (best["utility"], best["mean_gain"], -best["mean_harm"])):
            best = s
    if best is None:
        best = max(candidates, key=lambda s: (s["utility"], s["mean_gain"], -s["mean_harm"]))
        best["guard_forced_unconstrained"] = True
    else:
        best["guard_forced_unconstrained"] = False
    best["guard_candidates"] = len(candidates)
    return {"guard_threshold": float(best["guard_threshold"]), "summary": best, "candidates": candidates}


def make_action_rows(cases, guard_threshold):
    rows = []
    for c in cases:
        host_unc = c["host_uncertainty"]
        rows.append({
            "id": c["id"],
            "idx": c["idx"],
            "method": "host",
            "alpha": 0.0,
            "cup": c["host_dice"],
            "disc": 0.0,
            "cpd": c["host_dice"],
            "iou": c["host_iou"],
            **host_unc,
            **geom_features(np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)),
            "quality_risk": 0.0,
        })

        use_ref = float(c["guard_risk"]) <= float(guard_threshold) + 1e-12
        pred_dice = c["refiner_dice"] if use_ref else c["host_dice"]
        pred_iou = c["refiner_iou"] if use_ref else c["host_iou"]
        geom = c["geom"] if use_ref else geom_features(np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8))
        rows.append({
            "id": c["id"],
            "idx": c["idx"],
            "method": "learned_unet_guarded_refiner",
            "alpha": 1.0,
            "cup": pred_dice,
            "disc": 0.0,
            "cpd": pred_dice,
            "iou": pred_iou,
            **host_unc,
            **geom,
            "quality_risk": float(c["guard_risk"]),
        })
    return rows


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "alpha", "changed", "component_delta", "cpd", "cup", "cup_area_delta",
        "cup_centroid_shift", "cup_changed", "cup_disc_ratio_delta", "disc",
        "disc_area_delta", "disc_centroid_shift", "disc_changed", "geom_risk",
        "host_confidence", "host_entropy", "host_margin", "id", "idx", "iou",
        "method", "quality_risk",
    ]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0.0) for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--idx_guard", required=True, help="Validation split used only to freeze the learned-refiner guard.")
    ap.add_argument("--idx_eval", required=True, help="Held-out split used for SafeRefine calibration/test.")
    ap.add_argument("--host_ckpt", required=True)
    ap.add_argument("--refiner_ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--host_arch", default="graphseg")
    ap.add_argument("--host_backbone", default="segformer_b0")
    ap.add_argument("--host_unet_base", type=int, default=32)
    ap.add_argument("--host_feat_dim", type=int, default=256)
    ap.add_argument("--host_hidden", type=int, default=512)
    ap.add_argument("--host_depth", type=int, default=3)
    ap.add_argument("--host_graph_down", type=int, default=2)
    ap.add_argument("--host_grid4", type=int, default=1)
    ap.add_argument("--refiner_arch", default="unet")
    ap.add_argument("--refiner_backbone", default="segformer_b0")
    ap.add_argument("--refiner_unet_base", type=int, default=32)
    ap.add_argument("--refiner_feat_dim", type=int, default=256)
    ap.add_argument("--refiner_hidden", type=int, default=512)
    ap.add_argument("--refiner_depth", type=int, default=3)
    ap.add_argument("--refiner_graph_down", type=int, default=2)
    ap.add_argument("--refiner_grid4", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--torch_threads", type=int, default=4)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--guard_max_harm_rate", type=float, default=0.12)
    ap.add_argument("--guard_max_drop05_rate", type=float, default=0.05)
    ap.add_argument("--guard_max_mean_harm", type=float, default=0.01)
    ap.add_argument("--guard_min_gain", type=float, default=0.005)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--data_verbose", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    set_seed(args.seed)
    guard_cases = collect_predictions(args, args.idx_guard, "guard")
    guard = tune_guard(guard_cases, args)
    eval_cases = collect_predictions(args, args.idx_eval, "eval")
    rows = make_action_rows(eval_cases, guard["guard_threshold"])
    write_csv(args.out_csv, rows)

    guarded_eval = []
    for c in eval_cases:
        use_ref = float(c["guard_risk"]) <= float(guard["guard_threshold"]) + 1e-12
        guarded_eval.append(dict(c, guarded_dice=(c["refiner_dice"] if use_ref else c["host_dice"])))
    payload = {
        "args": vars(args),
        "guard_threshold": guard["guard_threshold"],
        "guard_summary": guard["summary"],
        "eval_fixed_refiner": summarize_gain(eval_cases, "refiner_dice"),
        "eval_guarded_refiner": summarize_gain(guarded_eval, "guarded_dice"),
        "n_eval_rows": len(rows),
    }
    p = Path(args.out_summary)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
