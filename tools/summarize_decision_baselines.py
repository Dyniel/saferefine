#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path


LABELS = {
    "isic2018_task1_mediafinal_unet_e120_zoo": "ISIC / UNet",
    "kvasir_seg_mediafinal_graphseg_e120_zoo": "Kvasir / GraphSeg",
    "kvasir_seg_mediafinal_unet_e120_zoo": "Kvasir / UNet",
    "ph2_mediafinal_unet_e120_zoo": "PH2 / UNet",
    "polyps_official_mediafinal_graphseg_e120_zoo": "Polyp ext. / GraphSeg",
    "polyps_official_mediafinal_unet_e120_zoo": "Polyp ext. / UNet",
    "msd_heart_mri_mediafinal_graphseg_mri_e120_zoo": "MSD Heart MRI / GraphSeg",
    "msd_heart_mri_mediafinal_unet_mri_e120_zoo": "MSD Heart MRI / UNet",
}

POLICIES = {
    "host",
    "best_fixed_cal_utility",
    "oracle",
    "crc_changed",
    "crc_geom",
    "crc_change_plus_geom",
    "crc_host_uncertainty",
    "crc_quality_risk",
    "random_rate_matched",
}


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def f(x, default=0.0):
    if x is None or x == "":
        return default
    return float(str(x).replace("+", ""))


def fmt(x, digits=4, signed=False):
    x = f(x)
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, rows, title):
    fields = list(rows[0].keys()) if rows else []
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def latex_escape(x):
    return str(x).replace("_", "\\_").replace("%", "\\%")


def write_tex(path, rows, caption, label):
    fields = list(rows[0].keys()) if rows else []
    cols = "l" * len(fields)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(latex_escape(k) for k in fields) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(k, "")) for k in fields) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    Path(path).write_text("\n".join(lines))


def run_from_path(path, suffix):
    name = Path(path).name
    if not name.endswith(suffix):
        return name
    return name[: -len(suffix)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    policy_rows = []
    diag_rows = []

    for path in sorted(input_dir.glob("*_policies.csv")):
        run = run_from_path(path, "_policies.csv")
        if run not in LABELS:
            continue
        for row in read_csv(path):
            if row.get("policy") not in POLICIES:
                continue
            policy_rows.append({
                "setting": LABELS.get(run, run),
                "policy": row["policy"],
                "gain": fmt(row.get("mean_gain"), 4, True),
                "harm": fmt(row.get("mean_harm"), 4),
                "harmed_rate": fmt(row.get("harmed_rate"), 3),
                "drop_gt_0.05": fmt(row.get("drop_gt_0.05"), 3),
                "cvar_harm_10": fmt(row.get("cvar_harm_10"), 4),
                "worst_drop": fmt(row.get("worst_drop"), 4, True),
                "revert_rate": fmt(row.get("reverted_rate"), 3),
                "action": row.get("selected_action", ""),
            })

    for path in sorted(input_dir.glob("*_risk_diagnostics.csv")):
        run = run_from_path(path, "_risk_diagnostics.csv")
        if run not in LABELS:
            continue
        for row in read_csv(path):
            if row.get("risk_score") not in {"changed", "geom", "change_plus_geom", "host_uncertainty", "quality_risk"}:
                continue
            diag_rows.append({
                "setting": LABELS.get(run, run),
                "risk_score": row["risk_score"],
                "harm_rate": fmt(row.get("harm_rate"), 3),
                "AUROC": fmt(row.get("auroc"), 3),
                "AUPRC": fmt(row.get("auprc"), 3),
                "n_actions": row.get("n_actions", ""),
            })

    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + "_policies.csv", policy_rows)
    write_md(str(out_prefix) + "_policies.md", policy_rows, "Decision Baseline Policy Comparison")
    write_tex(
        str(out_prefix) + "_policies.tex",
        policy_rows,
        "Decision baselines for safe refinement. Quality risk is a learned calibration-split harm predictor; host uncertainty uses entropy/confidence features when available.",
        "tab:decision_baselines",
    )
    write_csv(str(out_prefix) + "_risk_diagnostics.csv", diag_rows)
    write_md(str(out_prefix) + "_risk_diagnostics.md", diag_rows, "Risk Score Diagnostics")
    write_tex(
        str(out_prefix) + "_risk_diagnostics.tex",
        diag_rows,
        "Action-level diagnostics for predicting whether a candidate refinement harms the host prediction.",
        "tab:risk_diagnostics",
    )
    print({"policy_rows": len(policy_rows), "diagnostic_rows": len(diag_rows), "out_prefix": str(out_prefix)})


if __name__ == "__main__":
    main()
