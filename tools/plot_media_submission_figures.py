#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 140,
})


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "submission_figures"


LABELS = {
    "isic2018_task1_mediafinal_unet_e120": "ISIC / UNet",
    "isic2018_task1_mediafinal_unet_e120_zoo": "ISIC / UNet / zoo",
    "kvasir_seg_mediafinal_graphseg_e120": "Kvasir / GraphSeg",
    "kvasir_seg_mediafinal_graphseg_e120_zoo": "Kvasir / GraphSeg / zoo",
    "kvasir_seg_mediafinal_unet_e120": "Kvasir / UNet",
    "kvasir_seg_mediafinal_unet_e120_zoo": "Kvasir / UNet / zoo",
    "msd_heart_mri_mediafinal_graphseg_mri_e120": "MSD Heart MRI / GraphSeg",
    "msd_heart_mri_mediafinal_graphseg_mri_e120_zoo": "MSD Heart MRI / GraphSeg / zoo",
    "msd_heart_mri_mediafinal_unet_mri_e120": "MSD Heart MRI / UNet",
    "msd_heart_mri_mediafinal_unet_mri_e120_zoo": "MSD Heart MRI / UNet / zoo",
    "ph2_mediafinal_unet_e120": "PH2 / UNet",
    "ph2_mediafinal_unet_e120_zoo": "PH2 / UNet / zoo",
    "polyps_official_mediafinal_graphseg_e120": "Polyp ext. / GraphSeg",
    "polyps_official_mediafinal_graphseg_e120_zoo": "Polyp ext. / GraphSeg / zoo",
    "polyps_official_mediafinal_unet_e120": "Polyp ext. / UNet",
    "polyps_official_mediafinal_unet_e120_zoo": "Polyp ext. / UNet / zoo",
}


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    if x is None or x == "":
        return 0.0
    return float(str(x).replace("+", ""))


def parse_ci(text):
    vals = [float(x) for x in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)]
    if len(vals) < 3:
        return 0.0, 0.0, 0.0
    return vals[0], vals[1], vals[2]


def best_practical_bootstrap_rows():
    rows = read_csv(ROOT / "results/paper_tables/mediafinal_bootstrap_crc_ablation.csv")
    wanted = {
        ("isic2018_task1_mediafinal_unet_e120_zoo", "changed"),
        ("kvasir_seg_mediafinal_graphseg_e120", "geom"),
        ("polyps_official_mediafinal_graphseg_e120_zoo", "changed"),
        ("polyps_official_mediafinal_unet_e120_zoo", "geom"),
    }
    out = []
    for row in rows:
        key = (row["dataset"], row["risk"])
        if row["mode"] == "practical" and key in wanted:
            out.append(row)
    out.sort(key=lambda r: LABELS.get(r["dataset"], r["dataset"]))
    return out


def plot_gain_harm_scatter():
    rows = read_csv(ROOT / "results/paper_tables/mediafinal_consistent_results.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = {
        "fixed": "#8c564b",
        "crc": "#1f77b4",
        "zoo": "#2ca02c",
    }
    markers = {"fixed": "x", "crc": "o", "zoo": "s"}
    for row in rows:
        label = LABELS.get(row["run"], row["run"])
        points = [
            ("fixed", f(row["best_fixed_gain"]), f(row["best_fixed_harm"])),
            ("crc", f(row["crc_gain"]), f(row["crc_harm"])),
            ("zoo", f(row["zoo_crc_gain"]), f(row["zoo_crc_harm"])),
        ]
        for kind, gain, harm in points:
            ax.scatter(gain, harm, c=colors[kind], marker=markers[kind], s=54, alpha=0.9)
            if kind == "zoo" and (abs(gain) > 1e-6 or abs(harm) > 1e-6):
                ax.annotate(label, (gain, harm), xytext=(4, 4), textcoords="offset points", fontsize=7)
    handles = [
        plt.Line2D([0], [0], marker=markers[k], color="w", markerfacecolor=colors[k],
                   markeredgecolor=colors[k], label=k, markersize=7)
        for k in ("fixed", "crc", "zoo")
    ]
    ax.axvline(0, color="#777777", lw=0.8)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Mean Dice gain over host")
    ax.set_ylabel("Mean harm")
    ax.set_title("Utility-safety trade-off for fixed and calibrated refinement")
    ax.legend(handles=handles, frameon=False, loc="upper right")
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gain_harm_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_bootstrap_key_results():
    rows = best_practical_bootstrap_rows()
    labels = [LABELS.get(r["dataset"], r["dataset"]) for r in rows]
    mids, los, his = [], [], []
    harms = []
    for row in rows:
        mid, lo, hi = parse_ci(row["gain_CI"])
        mids.append(mid)
        los.append(lo)
        his.append(hi)
        harms.append(parse_ci(row["harm_CI"])[0])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    xerr = np.vstack([np.asarray(mids) - np.asarray(los), np.asarray(his) - np.asarray(mids)])
    ax.errorbar(mids, y, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=4)
    for i, harm in enumerate(harms):
        ax.text(his[i] + 0.0003, y[i], f"harm={harm:.4f}", va="center", fontsize=8)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("CRC mean Dice gain with 95% bootstrap CI")
    ax.set_title("Key final practical-policy effects")
    ax.grid(True, axis="x", color="#dddddd", lw=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_key_bootstrap_effects.png", dpi=220)
    plt.close(fig)


def plot_strict_fallback():
    rows = read_csv(ROOT / "results/paper_tables/mediafinal_consistent_results.csv")
    base = sum(1 for r in rows if r["strict_exact_fallback"] == "yes")
    zoo = sum(1 for r in rows if r["zoo_strict_exact_fallback"] == "yes")
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.bar(["baseline actions", "refiner zoo"], [base, zoo], color=["#1f77b4", "#2ca02c"], width=0.55)
    ax.set_ylim(0, len(rows) + 0.5)
    ax.set_ylabel("Runs with exact strict host fallback")
    ax.set_title("Strict policy preserves the host in all final runs")
    for x, val in enumerate([base, zoo]):
        ax.text(x, val + 0.08, f"{val}/{len(rows)}", ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_strict_exact_fallback.png", dpi=220)
    plt.close(fig)


def plot_safety_value():
    rows = read_csv(ROOT / "results/paper_tables/safety_value_over_fixed.csv")
    labels = [r["setting"] for r in rows]
    harm_prevented = [float(r["harm_prevented"].replace("+", "")) for r in rows]
    worst_improvement = [float(r["worst_drop_improvement"].replace("+", "")) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.barh(y - 0.18, harm_prevented, height=0.34, color="#1f77b4", label="mean harm prevented")
    ax.barh(y + 0.18, worst_improvement, height=0.34, color="#2ca02c", label="worst-drop improvement")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Dice units relative to fixed refinement")
    ax.set_title("Safety value: calibrated policies prevent harm from fixed refinement")
    ax.axvline(0, color="#777777", lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", color="#dddddd", lw=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_safety_value_over_fixed.png", dpi=220)
    plt.close(fig)


def plot_harm_budget():
    rows = [
        r for r in read_csv(ROOT / "results/paper_tables/harm_budget_sweep.csv")
        if r["dataset"] in ("isic2018_task1", "kvasir_seg_endo_graphseg_e120", "polyps_official_endo_graphseg_e120")
        and r["risk"] == "geom"
    ]
    labels = {
        "isic2018_task1": "ISIC",
        "kvasir_seg_endo_graphseg_e120": "Kvasir GraphSeg",
        "polyps_official_endo_graphseg_e120": "Polyp ext. GraphSeg",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for dataset in labels:
        vals = sorted([r for r in rows if r["dataset"] == dataset], key=lambda r: float(r["cal_harm_budget"]))
        x = [float(r["cal_harm_budget"]) for r in vals]
        gain = [float(r["gain"].replace("+", "")) for r in vals]
        harm = [float(r["harm"]) for r in vals]
        rev = [float(r["revert_rate"]) for r in vals]
        axes[0].plot(x, gain, marker="o", label=labels[dataset])
        axes[0].plot(x, harm, marker="x", linestyle="--", alpha=0.8)
        axes[1].plot(x, rev, marker="o", label=labels[dataset])
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set_title("Utility appears only at practical budgets")
    axes[0].set_ylabel("Mean gain (solid) / harm (dashed)")
    axes[0].set_xlabel("Calibration harmed-rate budget")
    axes[1].set_title("Lower budgets revert to host")
    axes[1].set_ylabel("Revert-rate")
    axes[1].set_xlabel("Calibration harmed-rate budget")
    for ax in axes:
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_harm_budget_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_method_schematic():
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.16, 0.22, "Medical\nimage"),
        (0.25, 0.58, 0.18, 0.22, "Host\nsegmenter"),
        (0.51, 0.62, 0.20, 0.26, "Candidate actions\nhost | morph | CC\nsmooth | graph"),
        (0.76, 0.58, 0.20, 0.22, "Calibrated\nrisk-action gate"),
        (0.76, 0.18, 0.20, 0.18, "One output mask\nrefine or exact host"),
    ]
    colors = ["#f2f2f2", "#dbe9f6", "#e6f4ea", "#fff0d8", "#e8e1f4"]
    for (x, y, w, h, text), color in zip(boxes, colors):
        rect = plt.Rectangle((x, y), w, h, fc=color, ec="#333333", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.19, 0.69), (0.25, 0.69)),
        ((0.43, 0.69), (0.51, 0.73)),
        ((0.71, 0.73), (0.76, 0.69)),
        ((0.86, 0.58), (0.86, 0.36)),
        ((0.34, 0.58), (0.34, 0.27)),
        ((0.34, 0.27), (0.76, 0.27)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    ax.text(0.48, 0.22, "fallback path returns the host prediction exactly", ha="center", va="center", fontsize=9)
    ax.text(0.61, 0.96, "SafeRefine: host-preserving calibrated action selection", ha="center", va="center",
            fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_method_schematic.png", dpi=220)
    plt.close(fig)


def plot_case_contactsheet():
    paths = [
        ROOT / "results/figures/mediafinal_crc_reversion_cases/isic2018_task1_mediafinal_unet_e120_zoo_test_ISIC_0022219_largest_cc_a0.png",
        ROOT / "results/figures/mediafinal_crc_reversion_cases/kvasir_seg_mediafinal_graphseg_e120_cju5x7iskmad90818frchyfwd_binary_morph_a5.png",
        ROOT / "results/figures/mediafinal_crc_reversion_cases/polyps_official_mediafinal_unet_e120_zoo_test_ETIS-LaribPolypDB_136_binary_morph_a9.png",
    ]
    titles = ["ISIC: largest-component refiner rejected", "Kvasir-SEG: morphology rejected", "External polyp: severe fixed-refiner failure rejected"]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 11.0), constrained_layout=True)
    for ax, path, title in zip(axes, paths, titles):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(OUT / "fig_case_contactsheet.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_method_schematic()
    plot_gain_harm_scatter()
    plot_safety_value()
    plot_harm_budget()
    plot_bootstrap_key_results()
    plot_strict_fallback()
    plot_case_contactsheet()
    print({"out_dir": str(OUT), "figures": 7})


if __name__ == "__main__":
    main()
