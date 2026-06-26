#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_safe_action_portfolio as P  # noqa: E402
from eval_binary_safe_refinement import dice_iou, morph_refine  # noqa: E402
from eval_binary_refiner_zoo import bilateral_smooth, close_only, open_close, open_only, prob_smooth, remove_small  # noqa: E402
from eval_binary_safe_refinement import fill_holes, largest_cc  # noqa: E402
from train_binary_host import BinaryNPZ, infer_npz_keys, make_model  # noqa: E402
from unified_graphseg import set_seed  # noqa: E402


def parse_action_method(action, fallback="binary_morph"):
    parts = str(action).split(":")
    if len(parts) >= 2 and parts[-1].startswith("a"):
        return parts[-2]
    return str(fallback)


def parse_action_alpha(action, fallback=0):
    m = re.search(r":a([-+]?\d+(?:\.\d+)?)$", str(action))
    if m:
        return int(round(float(m.group(1))))
    return int(round(float(fallback)))


def apply_action(method, alpha, host, prob):
    method = str(method)
    if method in ("binary_morph", "host_morph", "morph"):
        return morph_refine(host, int(round(alpha)))
    if method == "close_only":
        return close_only(host, int(round(alpha)))
    if method == "open_only":
        return open_only(host, int(round(alpha)))
    if method == "open_close":
        return open_close(host, int(round(alpha)))
    if method == "fill_holes":
        return fill_holes(host)
    if method == "largest_cc":
        return largest_cc(host)
    if method == "lcc_fill":
        return fill_holes(largest_cc(host))
    if method == "remove_small":
        return remove_small(host, int(round(alpha)))
    if method == "prob_gaussian":
        return prob_smooth(prob, float(alpha))
    if method == "prob_gaussian_lcc":
        return fill_holes(largest_cc(prob_smooth(prob, float(alpha))))
    if method == "prob_bilateral":
        return bilateral_smooth(prob, float(alpha))
    raise ValueError(f"Unsupported action method for plotting: {method}")


def load_raw(npz_path, idx, img_size):
    z = np.load(npz_path, allow_pickle=True)
    k_img, k_mask, _ = infer_npz_keys(z)
    img = np.asarray(z[k_img][idx])
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    mask = np.asarray(z[k_mask][idx])
    if mask.ndim == 3:
        mask = mask[..., 0]
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    if img.max() <= 2.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, (mask > 0).astype(np.uint8)


def load_model(args, device):
    model = make_model(args, device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[plot] missing checkpoint keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[plot] unexpected checkpoint keys: {len(unexpected)}", flush=True)
    model.eval()
    return model


def choose_cases(args, policy):
    rows = P.load_rows([(args.dataset, Path(args.input_csv))])
    actions = P.all_actions(rows)
    _cal_ids, test_ids = P.split_ids(rows, args.cal_fraction)
    grouped = P.group_rows(rows, test_ids)
    hosts = P.host_by_id(grouped)

    crc = policy["crc_best"]
    action = crc.get("selected_action", "host")
    threshold = crc.get("threshold")
    risk_mode = crc.get("risk_score", policy.get("risk_score", args.risk_score))

    candidates = []
    if action != "host" and threshold is not None:
        for image_id, by_action in grouped.items():
            row = by_action.get(action)
            if row is None:
                continue
            gain = row["cpd"] - hosts[image_id]["cpd"]
            risk = P.risk_score(row, risk_mode)
            if gain < -args.min_drop and risk > float(threshold) + 1e-12:
                candidates.append((gain, risk, image_id, row, hosts[image_id], action, float(threshold), risk_mode))

    if not candidates:
        for image_id, by_action in grouped.items():
            host = hosts[image_id]
            for cand_action in actions:
                if cand_action == "host" or cand_action not in by_action:
                    continue
                row = by_action[cand_action]
                gain = row["cpd"] - host["cpd"]
                if gain < -args.min_drop:
                    risk = P.risk_score(row, risk_mode)
                    candidates.append((gain, risk, image_id, row, host, cand_action, None, risk_mode))

    candidates.sort(key=lambda x: (x[0], -x[1]))
    return candidates[: args.max_cases]


@torch.no_grad()
def predict_host(model, args, idx, device):
    ds = BinaryNPZ(args.npz_all, [idx], args.img_size, train=False)
    x, y, _meta = ds[0]
    logits, _, _ = model(x.unsqueeze(0).to(device), dyn_on_eval="none", eval_dyn_k=0)
    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
    host = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return host, y.numpy().astype(np.uint8), prob


def panel(ax, title, arr, cmap=None):
    ax.imshow(arr, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_case(args, model, device, case, out_dir):
    gain, risk, image_id, row, host_row, action, threshold, risk_mode = case
    idx = int(row["idx"])
    method = parse_action_method(action, row.get("method", "binary_morph"))
    alpha = parse_action_alpha(action, row.get("alpha", 0.0))
    raw_img, raw_gt = load_raw(args.npz_all, idx, args.img_size)
    host, gt, prob = predict_host(model, args, idx, device)
    if raw_gt.shape == gt.shape:
        gt = raw_gt
    fixed = apply_action(method, alpha, host, prob)
    crc_out = host

    host_dice, _ = dice_iou(host, gt)
    fixed_dice, _ = dice_iou(fixed, gt)
    crc_dice, _ = dice_iou(crc_out, gt)
    changed = (fixed != host).astype(np.float32)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), constrained_layout=True)
    panel(axes[0], f"image\n{id_short(image_id)}", raw_img)
    panel(axes[1], "ground truth", gt, cmap="gray")
    panel(axes[2], f"host\nDice {host_dice:.3f}", host, cmap="gray")
    action_title = f"{method} a={alpha:g}" if abs(float(alpha)) > 1e-12 else method
    panel(axes[3], f"fixed {action_title}\nDice {fixed_dice:.3f}, gain {fixed_dice-host_dice:+.3f}", fixed, cmap="gray")
    t = "none" if threshold is None else f"{threshold:.4g}"
    panel(axes[4], f"CRC revert\nDice {crc_dice:.3f}\nrisk {risk:.4g} > {t}", crc_out, cmap="gray")
    axes[0].contour(gt, colors="lime", linewidths=0.7)
    axes[0].contour(host, colors="deepskyblue", linewidths=0.7)
    axes[3].contour(changed, colors="red", linewidths=0.6)
    fig.suptitle(f"{args.dataset}: fixed refiner harms, risk policy reverts ({risk_mode})", fontsize=11)

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id))[:80]
    out_png = out_dir / f"{args.dataset}_{safe_id}_{method}_a{alpha:g}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return {
        "dataset": args.dataset,
        "id": image_id,
        "idx": idx,
        "action": action,
        "method": method,
        "alpha": alpha,
        "risk_score": risk_mode,
        "risk": risk,
        "threshold": "" if threshold is None else threshold,
        "host_dice": host_dice,
        "fixed_dice": fixed_dice,
        "crc_dice": crc_dice,
        "fixed_gain": fixed_dice - host_dice,
        "figure": str(out_png),
    }


def id_short(image_id):
    s = str(image_id)
    return s if len(s) <= 18 else s[:15] + "..."


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "id", "idx", "action", "method", "alpha", "risk_score", "risk", "threshold",
        "host_dice", "fixed_dice", "crc_dice", "fixed_gain", "figure",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--policy_json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="results/figures/crc_reversion_cases")
    ap.add_argument("--arch", default="graphseg")
    ap.add_argument("--backbone", default="segformer_b0")
    ap.add_argument("--unet_base", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=352)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--graph_down", type=int, default=2)
    ap.add_argument("--grid4", type=int, default=1)
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--risk_score", default="change_plus_geom")
    ap.add_argument("--max_cases", type=int, default=3)
    ap.add_argument("--min_drop", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    policy = json.loads(Path(args.policy_json).read_text())
    cases = choose_cases(args, policy)
    if not cases:
        raise RuntimeError("No harmful fixed-refiner cases found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in cases:
        manifest.append(draw_case(args, model, device, case, out_dir))
    write_manifest(out_dir / f"{args.dataset}_manifest.csv", manifest)
    print(json.dumps({"out_dir": str(out_dir), "cases": len(manifest)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
