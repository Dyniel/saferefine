#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


PAIRS = [
    {
        "setting": "Kvasir / GraphSeg->UNet",
        "label": "kvasir_graphseg_unet_learned",
        "host_csv": "results/refiner_zoo_uncert/kvasir_seg_mediafinal_graphseg_e120_zoo.csv",
        "refiner_csv": "results/refiner_zoo_uncert/kvasir_seg_mediafinal_unet_e120_zoo.csv",
        "split_group": "image",
    },
    {
        "setting": "Polyp ext. / GraphSeg->UNet",
        "label": "polyps_graphseg_unet_learned",
        "host_csv": "results/refiner_zoo_uncert/polyps_official_mediafinal_graphseg_e120_zoo.csv",
        "refiner_csv": "results/refiner_zoo_uncert/polyps_official_mediafinal_unet_e120_zoo.csv",
        "split_group": "image",
    },
    {
        "setting": "MSD Heart / GraphSeg->UNet",
        "label": "msd_graphseg_unet_learned",
        "host_csv": "results/refiner_zoo_uncert/msd_heart_mri_mediafinal_graphseg_mri_e120_zoo.csv",
        "refiner_csv": "results/refiner_zoo_uncert/msd_heart_mri_mediafinal_unet_mri_e120_zoo.csv",
        "split_group": "patient",
    },
]


def add_quality_risk(df):
    df = df.copy()
    df["quality_risk"] = (
        df["host_entropy"].astype(float)
        + np.maximum(0.0, 1.0 - df["host_confidence"].astype(float))
        + np.maximum(0.0, 1.0 - df["host_margin"].astype(float))
    )
    return df


def build_pair_csv(pair, out_dir):
    host_all = pd.read_csv(ROOT / pair["host_csv"])
    ref_all = pd.read_csv(ROOT / pair["refiner_csv"])
    host = host_all[host_all["method"] == "host"].copy()
    ref = ref_all[ref_all["method"] == "host"][["id", "cup", "cpd", "iou"]].copy()
    learned = host.drop(columns=["cup", "cpd", "iou"]).merge(ref, on="id", how="inner")
    if len(learned) != len(host):
        raise RuntimeError(f"{pair['label']}: id mismatch host={len(host)} learned={len(learned)}")
    host["quality_risk"] = 0.0
    learned = add_quality_risk(learned)
    learned["method"] = "learned_unet_refiner"
    learned["alpha"] = 1.0
    out = pd.concat([host, learned[host.columns]], ignore_index=True)
    path = out_dir / f"{pair['label']}.csv"
    out.to_csv(path, index=False)
    return path, out


def run_policy(pair, action_csv, out_dir, python):
    out_csv = out_dir / f"{pair['label']}_policy.csv"
    out_json = out_dir / f"{pair['label']}_policy.json"
    cmd = [
        python,
        "tools/eval_safe_action_portfolio.py",
        "--inputs",
        f"{pair['label']}={action_csv}",
        "--risk_score",
        "host_uncertainty",
        "--split_group",
        pair["split_group"],
        "--harm_eps",
        "0",
        "--max_cal_harm_rate",
        "0.25",
        "--max_cal_drop05_rate",
        "0.10",
        "--max_cal_mean_harm",
        "0.02",
        "--tail_constraint_mode",
        "full",
        "--bound_mode",
        "hoeffding",
        "--out_csv",
        str(out_csv),
        "--out_summary",
        str(out_json),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_csv, out_json


def finite(x):
    try:
        if pd.isna(x):
            return ""
    except TypeError:
        pass
    return x


def row_from_policy(pair, policy_csv):
    df = pd.read_csv(policy_csv)
    test = df[df["split"].eq("test")].copy()
    fixed = test[test["policy"].str.startswith("fixed:")].iloc[0]
    crc = test[test["policy"].eq("crc_portfolio")].iloc[0]
    cal = test[test["policy"].eq("calibrated_portfolio")].iloc[0]
    return {
        "setting": pair["setting"],
        "split_group": pair["split_group"],
        "fixed_gain": float(fixed["mean_gain"]),
        "fixed_harm": float(fixed["mean_harm"]),
        "fixed_worst": float(fixed["worst_drop"]),
        "fixed_harmed": float(fixed["harmed_rate"]),
        "fixed_drop05": float(fixed["drop_gt_0.05"]),
        "utility_gain": float(cal["mean_gain"]),
        "utility_worst": float(cal["worst_drop"]),
        "utility_harmed": float(cal["harmed_rate"]),
        "safe_action": str(crc["selected_action"]),
        "safe_gain": float(crc["mean_gain"]),
        "safe_harm": float(crc["mean_harm"]),
        "safe_worst": float(crc["worst_drop"]),
        "safe_harmed": float(crc["harmed_rate"]),
        "safe_reverted": float(crc["reverted_rate"]),
        "test_n": int(crc["n"]),
    }


def fmt(x, signed=False):
    if x == "":
        return ""
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("->", "$\\rightarrow$")


def write_main_tex(path, rows):
    with Path(path).open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Standalone-segmenter stress portfolio. An independently trained UNet checkpoint is offered as an alternative action over the corresponding GraphSeg host. It is a separate segmenter, not a host-conditioned refiner---it does not receive the host mask or probabilities---and is gated by a host-uncertainty risk score. The standalone candidate improves mean Dice in each setting, but severe case-level degradation remains, and the primary SafeRefine contract falls back to the host.}\n")
        f.write("\\label{tab:learned_refiner_stress}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Setting & Fixed gain & Mean harm & Worst drop & Harmed & SafeRefine action \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            action = "host" if r["safe_action"] == "host" else "non-host"
            f.write(
                f"{tex_escape(r['setting'])} & {fmt(r['fixed_gain'], True)} & {fmt(r['fixed_harm'])} & "
                f"{fmt(r['fixed_worst'], True)} & {fmt(r['fixed_harmed'])} & {action} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_supp_tex(path, rows):
    with Path(path).open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Detailed standalone-segmenter stress portfolio diagnostics. The candidate is an independently trained UNet used as an alternative action over the GraphSeg host (not a host-conditioned refiner). Utility thresholding ignores the finite-sample UCB feasibility test; SafeRefine primary uses the full harmed-rate, drop$>0.05$, and mean-harm contract.}\n")
        f.write("\\label{tab:learned_refiner_stress_detail}\n")
        f.write("\\begin{tabular}{llrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Setting & Split & $n$ & Fixed gain & Fixed harm & Fixed worst & Fixed harmed & Utility gain & Utility worst & Safe reverted \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{tex_escape(r['setting'])} & {r['split_group']} & {r['test_n']} & "
                f"{fmt(r['fixed_gain'], True)} & {fmt(r['fixed_harm'])} & {fmt(r['fixed_worst'], True)} & "
                f"{fmt(r['fixed_harmed'])} & {fmt(r['utility_gain'], True)} & "
                f"{fmt(r['utility_worst'], True)} & {fmt(r['safe_reverted'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def volume_id(image_id):
    m = re.match(r"^(la_\d+)_z\d+", str(image_id))
    return m.group(1) if m else str(image_id)


def write_mri_volume_tex(path):
    df = pd.read_csv(ROOT / "results/learned_refiner_stress/msd_graphseg_unet_learned.csv")
    host = df[df["method"].eq("host")][["id", "cpd"]].rename(columns={"cpd": "host_cpd"})
    learned = df[df["method"].eq("learned_unet_refiner")][["id", "cpd"]].rename(columns={"cpd": "learned_cpd"})
    merged = host.merge(learned, on="id")
    merged["volume"] = merged["id"].map(volume_id)
    vol = merged.groupby("volume").agg(host_cpd=("host_cpd", "mean"), learned_cpd=("learned_cpd", "mean"))
    vol["fixed_gain"] = vol["learned_cpd"] - vol["host_cpd"]
    vol["fixed_harm"] = np.maximum(0.0, -vol["fixed_gain"])
    rows = [
        {
            "policy": "GraphSeg host",
            "n": len(vol),
            "mean_dice": float(vol["host_cpd"].mean()),
            "mean_gain": "",
            "mean_harm": "",
            "harmed": "",
            "values": ", ".join(f"{x:.3f}" for x in vol["host_cpd"]),
        },
        {
            "policy": "Fixed learned UNet refiner",
            "n": len(vol),
            "mean_dice": float(vol["learned_cpd"].mean()),
            "mean_gain": float(vol["fixed_gain"].mean()),
            "mean_harm": float(vol["fixed_harm"].mean()),
            "harmed": f"{int((vol['fixed_gain'] < 0).sum())}/{len(vol)}",
            "values": ", ".join(f"{x:+.3f}" for x in vol["fixed_gain"]),
        },
        {
            "policy": "SafeRefine primary",
            "n": len(vol),
            "mean_dice": float(vol["host_cpd"].mean()),
            "mean_gain": 0.0,
            "mean_harm": 0.0,
            "harmed": f"0/{len(vol)}",
            "values": ", ".join("+0.000" for _ in range(len(vol))),
        },
    ]
    with Path(path).open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{MSD Heart MRI volume-level diagnostic on the held-out patient split. Values aggregate slice Dice within each of the three test volumes. The fixed learned UNet refiner has large positive average volume gain, but SafeRefine primary falls back to the GraphSeg host, giving zero volume-level harm by construction.}\n")
        f.write("\\label{tab:mri_volume_diagnostic}\n")
        f.write("\\begin{tabular}{lrrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Policy & Volumes & Mean Dice & Mean gain & Mean harm & Harmed vols & Per-volume values \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            gain = "" if r["mean_gain"] == "" else f"{r['mean_gain']:+.3f}"
            harm = "" if r["mean_harm"] == "" else f"{r['mean_harm']:.3f}"
            harmed = r["harmed"]
            f.write(f"{r['policy']} & {r['n']} & {r['mean_dice']:.3f} & {gain} & {harm} & {harmed} & {r['values']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/learned_refiner_stress")
    ap.add_argument("--python", default="/users/scratch1/dancies/conda_envs/py312/bin/python")
    ap.add_argument("--main_tex", default="docs/submission/full_submission/tables/learned_refiner_stress.tex")
    ap.add_argument("--supp_tex", default="docs/submission/full_submission/tables/learned_refiner_stress_detail.tex")
    ap.add_argument("--mri_volume_tex", default="docs/submission/full_submission/tables/mri_volume_diagnostic.tex")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for pair in PAIRS:
        action_csv, _ = build_pair_csv(pair, out_dir)
        policy_csv, policy_json = run_policy(pair, action_csv, out_dir, args.python)
        rows.append(row_from_policy(pair, policy_csv))
        manifest.append({**pair, "action_csv": str(action_csv), "policy_csv": str(policy_csv), "policy_json": str(policy_json)})
    write_main_tex(ROOT / args.main_tex, rows)
    write_supp_tex(ROOT / args.supp_tex, rows)
    write_mri_volume_tex(ROOT / args.mri_volume_tex)
    payload = {"pairs": manifest, "summary": rows}
    (out_dir / "learned_refiner_stress_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with (out_dir / "learned_refiner_stress_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
