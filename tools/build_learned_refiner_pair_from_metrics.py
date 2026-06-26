#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def action_csv(host_csv, refiner_csv, out_csv):
    host_all = pd.read_csv(host_csv)
    ref_all = pd.read_csv(refiner_csv)
    host = host_all[host_all["method"] == "host"].copy()
    ref = ref_all[ref_all["method"] == "host"][["id", "cup", "cpd", "iou"]].copy()
    learned = host.drop(columns=["cup", "cpd", "iou"]).merge(ref, on="id", how="inner")
    learned["method"] = "learned_unet_refiner"
    learned["alpha"] = 1.0
    learned["quality_risk"] = (
        learned["host_entropy"].astype(float)
        + np.maximum(0.0, 1.0 - learned["host_confidence"].astype(float))
        + np.maximum(0.0, 1.0 - learned["host_margin"].astype(float))
    )

    if len(learned) != len(host):
        raise RuntimeError(f"Host/refiner id mismatch: host={len(host)} learned={len(learned)}")

    for col in ["quality_risk"]:
        if col not in host.columns:
            host[col] = 0.0
    out = pd.concat([host, learned[host.columns]], ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def summarize_fixed(action_rows):
    host = action_rows[action_rows["method"] == "host"][["id", "cpd"]].rename(columns={"cpd": "host_cpd"})
    learned = action_rows[action_rows["method"] == "learned_unet_refiner"][["id", "cpd"]].rename(columns={"cpd": "learned_cpd"})
    merged = host.merge(learned, on="id")
    gain = merged["learned_cpd"].to_numpy(dtype=float) - merged["host_cpd"].to_numpy(dtype=float)
    return {
        "n": int(len(gain)),
        "mean_gain": float(gain.mean()),
        "mean_harm": float(np.maximum(0.0, -gain).mean()),
        "worst_drop": float(gain.min()),
        "harmed_rate": float((gain < 0.0).mean()),
        "drop05_rate": float((gain < -0.05).mean()),
        "drop20_rate": float((gain < -0.20).mean()),
        "improved_rate": float((gain > 0.0).mean()),
    }


def run_controller(args):
    cmd = [
        args.python,
        "tools/eval_safe_action_portfolio.py",
        "--inputs",
        f"{args.label}={args.out_action_csv}",
        "--risk_score",
        args.risk_score,
        "--split_group",
        "image",
        "--harm_eps",
        "0",
        "--max_cal_harm_rate",
        str(args.max_cal_harm_rate),
        "--max_cal_drop05_rate",
        str(args.max_cal_drop05_rate),
        "--max_cal_mean_harm",
        str(args.max_cal_mean_harm),
        "--tail_constraint_mode",
        "full",
        "--bound_mode",
        "hoeffding",
        "--out_csv",
        args.out_policy_csv,
        "--out_summary",
        args.out_policy_json,
    ]
    subprocess.run(cmd, check=True)


def read_policy_rows(path):
    df = pd.read_csv(path)
    keep = df[df["split"].eq("test") & df["policy"].isin([
        "fixed:learned_pair:learned_unet_refiner:a1.00",
        "fixed:polyps_graphseg_unet:learned_unet_refiner:a1.00",
        "calibrated_portfolio",
        "crc_portfolio",
    ])].copy()
    return keep


def fmt(x, digits=4, signed=False):
    if x == "" or pd.isna(x):
        return ""
    return f"{float(x):+.{digits}f}" if signed else f"{float(x):.{digits}f}"


def write_tex(path, fixed_all, policy_df):
    rows = []
    fixed_test = policy_df[policy_df["policy"].str.startswith("fixed:")]
    if len(fixed_test):
        r = fixed_test.iloc[0]
        rows.append(("Fixed learned refiner", "UNet action", r))
    cal = policy_df[policy_df["policy"].eq("calibrated_portfolio")]
    if len(cal):
        rows.append(("Utility threshold", "Calibrated, no UCB", cal.iloc[0]))
    crc = policy_df[policy_df["policy"].eq("crc_portfolio")]
    if len(crc):
        rows.append(("SafeRefine primary", "Hoeffding UCB", crc.iloc[0]))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Learned-refiner stress diagnostic on the external polyp GraphSeg setting. A trained UNet checkpoint is treated as a stronger learned refiner candidate over the GraphSeg host. The fixed learned refiner improves average Dice but has a severe harm tail; the primary SafeRefine controller therefore falls back to the host.}\n")
        f.write("\\label{tab:learned_refiner_pair}\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Policy & Selection & Test gain & Mean harm & Worst drop & Harmed & Drop$>0.05$ \\\\\n")
        f.write("\\midrule\n")
        for name, sel, r in rows:
            f.write(
                f"{name} & {sel} & {fmt(r['mean_gain'], signed=True)} & {fmt(r['mean_harm'])} & "
                f"{fmt(r['worst_drop'], signed=True)} & {fmt(r['harmed_rate'])} & {fmt(r['drop_gt_0.05'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host_csv", default="results/refiner_zoo_uncert/polyps_official_mediafinal_graphseg_e120_zoo.csv")
    ap.add_argument("--refiner_csv", default="results/refiner_zoo_uncert/polyps_official_mediafinal_unet_e120_zoo.csv")
    ap.add_argument("--label", default="learned_pair")
    ap.add_argument("--risk_score", default="host_uncertainty")
    ap.add_argument("--python", default="/users/scratch1/dancies/conda_envs/py312/bin/python")
    ap.add_argument("--out_action_csv", default="results/learned_refiner_pair/polyps_graphseg_unet_learned_pair.csv")
    ap.add_argument("--out_policy_csv", default="results/learned_refiner_pair/polyps_graphseg_unet_learned_pair_policy.csv")
    ap.add_argument("--out_policy_json", default="results/learned_refiner_pair/polyps_graphseg_unet_learned_pair_policy.json")
    ap.add_argument("--out_summary_json", default="results/learned_refiner_pair/polyps_graphseg_unet_learned_pair_summary.json")
    ap.add_argument("--out_tex", default="docs/submission/full_submission/tables/learned_refiner_pair.tex")
    ap.add_argument("--max_cal_harm_rate", type=float, default=0.25)
    ap.add_argument("--max_cal_drop05_rate", type=float, default=0.10)
    ap.add_argument("--max_cal_mean_harm", type=float, default=0.02)
    args = ap.parse_args()

    rows = action_csv(args.host_csv, args.refiner_csv, args.out_action_csv)
    fixed_all = summarize_fixed(rows)
    run_controller(args)
    policy_df = read_policy_rows(args.out_policy_csv)
    write_tex(args.out_tex, fixed_all, policy_df)
    payload = {
        "host_csv": args.host_csv,
        "refiner_csv": args.refiner_csv,
        "action_csv": args.out_action_csv,
        "policy_csv": args.out_policy_csv,
        "fixed_all": fixed_all,
        "test_policies": policy_df.to_dict(orient="records"),
    }
    Path(args.out_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
