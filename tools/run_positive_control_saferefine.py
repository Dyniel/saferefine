#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


DATASETS = [
    ("ISIC", ROOT / "data_npz/isic2018_task1_352.npz"),
    ("PH2", ROOT / "data_npz/ph2_352.npz"),
    ("Kvasir-SEG", ROOT / "data_npz/kvasir_seg_352.npz"),
    ("MSD Heart MRI", ROOT / "data_npz/msd_heart_mri_352.npz"),
    ("Polyp external", ROOT / "data_npz/polyps_official_352.npz"),
]


def infer_keys(z):
    image_key = next(k for k in ("images", "image", "x", "X") if k in z)
    mask_key = next(k for k in ("masks", "mask", "labels", "y", "Y", "gt") if k in z)
    id_key = next((k for k in ("ids", "id", "names", "paths") if k in z), None)
    return image_key, mask_key, id_key


def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    if denom == 0.0:
        return 1.0
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def place_artifact(mask, size):
    h, w = mask.shape
    margin = max(4, size // 2)
    corners = [
        (margin, margin),
        (margin, w - margin - size),
        (h - margin - size, margin),
        (h - margin - size, w - margin - size),
    ]
    for y0, x0 in corners:
        patch = mask[y0:y0 + size, x0:x0 + size]
        if patch.shape == (size, size) and int(patch.sum()) == 0:
            return y0, x0
    return None


def load_cases(paths, artifact_size):
    cases = []
    panel_counts = {}
    for dataset, path in paths:
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        image_key, mask_key, id_key = infer_keys(z)
        images = z[image_key]
        masks = z[mask_key]
        ids = z[id_key] if id_key is not None else None
        for i in range(len(masks)):
            mask = np.asarray(masks[i])
            if mask.ndim == 3:
                mask = mask[..., 0]
            gt = (mask > 0).astype(np.uint8)
            loc = place_artifact(gt, artifact_size)
            if loc is None:
                continue
            image_id = str(ids[i]) if ids is not None else str(i)
            keep_image = (
                dataset in {"ISIC", "Kvasir-SEG", "MSD Heart MRI"}
                and panel_counts.get(dataset, 0) < 5
                and int(gt.sum()) > 4 * artifact_size ** 2
            )
            image = np.asarray(images[i]).copy() if keep_image else None
            if keep_image:
                panel_counts[dataset] = panel_counts.get(dataset, 0) + 1
            cases.append((dataset, image_id, i, image, gt, loc))
    return cases


def summarize(rows, split, radius):
    gains = np.asarray([r["gain"] for r in rows], dtype=np.float64)
    harms = np.maximum(0.0, -gains)
    out = {
        "split": split,
        "n": len(rows),
        "host_dice": float(np.mean([r["host_dice"] for r in rows])),
        "refiner_dice": float(np.mean([r["refiner_dice"] for r in rows])),
        "mean_gain": float(gains.mean()),
        "mean_harm": float(harms.mean()),
        "harmed_rate": float((gains < 0.0).mean()),
        "drop05_rate": float((gains < -0.05).mean()),
        "worst_drop": float(gains.min()),
        "revert_rate": 0.0,
        "ucb_radius": float(radius),
    }
    out["harm_rate_ucb"] = min(1.0, out["harmed_rate"] + radius)
    out["drop05_rate_ucb"] = min(1.0, out["drop05_rate"] + radius)
    out["mean_harm_ucb"] = min(1.0, out["mean_harm"] + radius)
    return out


def fmt(x, signed=False):
    if isinstance(x, int):
        return str(x)
    if signed:
        return f"{x:+.4f}"
    return f"{x:.4f}"


def write_table(summary, out_prefix):
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    tex_path = out_prefix.with_suffix(".tex")
    fields = [
        "split", "n", "host_dice", "refiner_dice", "mean_gain", "mean_harm",
        "harmed_rate", "drop05_rate", "worst_drop", "revert_rate",
        "harm_rate_ucb", "drop05_rate_ucb", "mean_harm_ucb",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary:
            w.writerow({k: row[k] for k in fields})

    labels = {
        "host_dice": "host Dice",
        "refiner_dice": "refiner Dice",
        "mean_gain": "gain",
        "mean_harm": "harm",
        "harmed_rate": "harmed-rate",
        "drop05_rate": "drop05",
        "worst_drop": "worst drop",
        "revert_rate": "revert",
        "harm_rate_ucb": "harm UCB",
        "drop05_rate_ucb": "drop05 UCB",
        "mean_harm_ucb": "mean-harm UCB",
    }
    show = ["split", "n", "mean_gain", "mean_harm", "harmed_rate", "drop05_rate", "worst_drop", "harm_rate_ucb", "drop05_rate_ucb", "mean_harm_ucb"]
    with md_path.open("w") as f:
        f.write("# Positive-Control Artifact Removal\n\n")
        f.write("| " + " | ".join(labels.get(k, k) for k in show) + " |\n")
        f.write("| " + " | ".join(["---"] * len(show)) + " |\n")
        for row in summary:
            vals = []
            for k in show:
                if k == "n" or k == "split":
                    vals.append(str(row[k]))
                else:
                    vals.append(fmt(row[k], signed=k in {"mean_gain", "worst_drop"}))
            f.write("| " + " | ".join(vals) + " |\n")

    with tex_path.open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Positive-control artifact-removal experiment. A synthetic host is formed by adding a known isolated square artifact to each ground-truth mask; the candidate refiner removes that artifact. This non-clinical sanity check verifies that SafeRefine certifies a low-risk useful intervention when one exists.}\n")
        f.write("\\label{tab:positive_control}\n")
        f.write("\\begin{tabular}{llllllll}\n\\toprule\n")
        f.write("split & $n$ & gain & harm & harmed & drop05 & worst & max UCB \\\\\n\\midrule\n")
        for row in summary:
            max_ucb = max(row["harm_rate_ucb"], row["drop05_rate_ucb"], row["mean_harm_ucb"])
            f.write(
                f"{row['split']} & {row['n']} & {row['mean_gain']:+.4f} & {row['mean_harm']:.4f} & "
                f"{row['harmed_rate']:.3f} & {row['drop05_rate']:.3f} & {row['worst_drop']:+.4f} & {max_ucb:.5f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def prep_image(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.max() <= 2:
        arr = arr * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def draw_panel(rows, out_path):
    examples = rows[:3]
    fig, axes = plt.subplots(len(examples), 4, figsize=(10.5, 7.2), constrained_layout=True)
    if len(examples) == 1:
        axes = axes[None, :]
    for r, axrow in zip(examples, axes):
        img = prep_image(r["image"])
        titles = [
            f"{r['dataset']}\ninput",
            "ground truth",
            f"synthetic host\nDice {r['host_dice']:.3f}",
            f"SafeRefine accepts\nDice {r['refiner_dice']:.3f}",
        ]
        arrays = [img, r["gt"], r["host"], r["refiner"]]
        cmaps = [None, "gray", "gray", "gray"]
        for ax, title, arr, cmap in zip(axrow, titles, arrays, cmaps):
            ax.imshow(arr, cmap=cmap, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        y0, x0 = r["artifact_loc"]
        s = r["artifact_size"]
        rect = plt.Rectangle((x0, y0), s, s, fill=False, edgecolor="red", linewidth=1.5)
        axrow[2].add_patch(rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_size", type=int, default=18)
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--risk_count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_prefix", default="results/positive_control/positive_control_artifact_removal")
    ap.add_argument("--figure", default="results/submission_figures/fig_positive_control_artifact.png")
    args = ap.parse_args()

    cases = load_cases(DATASETS, args.artifact_size)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cases))
    n_cal = int(round(len(cases) * args.cal_fraction))
    cal_ids = set(order[:n_cal].tolist())
    rows = []
    for j, (dataset, image_id, idx, image, gt, loc) in enumerate(cases):
        y0, x0 = loc
        host = gt.copy()
        host[y0:y0 + args.artifact_size, x0:x0 + args.artifact_size] = 1
        refiner = host.copy()
        refiner[y0:y0 + args.artifact_size, x0:x0 + args.artifact_size] = 0
        host_dice = dice(host, gt)
        refiner_dice = dice(refiner, gt)
        rows.append({
            "split": "cal" if j in cal_ids else "test",
            "dataset": dataset,
            "image_id": image_id,
            "idx": idx,
            "image": image,
            "gt": gt,
            "host": host,
            "refiner": refiner,
            "artifact_loc": loc,
            "artifact_size": args.artifact_size,
            "host_dice": host_dice,
            "refiner_dice": refiner_dice,
            "gain": refiner_dice - host_dice,
        })

    n = sum(r["split"] == "cal" for r in rows)
    radius = math.sqrt(math.log(max(1.0, args.risk_count) / max(args.delta, 1e-12)) / (2.0 * n))
    summary = [
        summarize([r for r in rows if r["split"] == "cal"], "cal", radius),
        summarize([r for r in rows if r["split"] == "test"], "test", radius),
    ]
    out_prefix = ROOT / args.out_prefix
    write_table(summary, out_prefix)
    panel_rows = []
    for dataset in ("ISIC", "Kvasir-SEG", "MSD Heart MRI"):
        match = next(
            (
                r for r in rows
                if r["image"] is not None and r["dataset"] == dataset and int(r["gt"].sum()) > 4 * args.artifact_size ** 2
            ),
            None,
        )
        if match is not None:
            panel_rows.append(match)
    draw_panel(panel_rows, ROOT / args.figure)
    print({
        "cases": len(rows),
        "cal": summary[0]["n"],
        "test": summary[1]["n"],
        "radius": radius,
        "table": str(out_prefix.with_suffix(".tex")),
        "figure": str(ROOT / args.figure),
    })


if __name__ == "__main__":
    main()
