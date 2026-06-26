#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from summarize_decision_baselines import LABELS, fmt, write_csv, write_md, write_tex


def load_json(path):
    return json.loads(Path(path).read_text())


def test_crc(payload):
    return next(r for r in payload["summaries"] if r.get("split") == "test" and r.get("policy") == "crc_portfolio")


def run_label(path):
    name = Path(path).name.replace("_tail_primary.json", "")
    for risk in ("change_plus_geom", "host_uncertainty", "changed", "geom"):
        suffix = f"_{risk}"
        if name.endswith(suffix):
            return name[: -len(suffix)], risk
    return name, ""


def compact_failed(text):
    mapping = {
        "harm_rate": "H",
        "drop05_rate": "D05",
        "drop20_rate": "D20",
        "mean_harm": "MH",
    }
    return ",".join(mapping.get(x, x) for x in str(text).split(",") if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    candidates = {}
    rows_all = []
    diagnostics = []
    for path in sorted(Path(args.input_dir).glob("*_tail_primary.json")):
        payload = load_json(path)
        label, risk = run_label(path)
        if label not in LABELS:
            continue
        row = test_crc(payload)
        out = {
            "setting": LABELS[label],
            "risk_score": risk,
            "mode": payload.get("tail_constraint_mode", "full"),
            "selection_mode": payload.get("selection_mode", "joint"),
            "harm_eps": fmt(payload.get("harm_eps", 0.0), 3),
            "gain": fmt(row.get("mean_gain"), 4, True),
            "harm": fmt(row.get("mean_harm"), 4),
            "harmed_rate": fmt(row.get("harmed_rate"), 3),
            "drop_gt_0.05": fmt(row.get("drop_gt_0.05"), 3),
            "drop_gt_0.20": fmt(row.get("drop_gt_0.20"), 3),
            "cvar_harm_10": fmt(row.get("cvar_harm_10"), 4),
            "worst_drop": fmt(row.get("worst_drop"), 4, True),
            "revert_rate": fmt(row.get("reverted_rate"), 3),
            "cal_utility": fmt(row.get("cal_utility"), 4, True),
            "action": row.get("selected_action", ""),
            "split_group": payload.get("split_group", "image"),
            "tested_candidates": row.get("tested_candidates", ""),
            "tested_thresholds_for_ucb": row.get("tested_thresholds_for_ucb", ""),
        }
        rows_all.append(out)
        key = label
        score = (
            float(row.get("cal_utility", 0.0)),
            -float(row.get("mean_harm", 0.0)),
            -float(row.get("drop_gt_0.05", 0.0)),
        )
        if key not in candidates or score > candidates[key][0]:
            candidates[key] = (score, out)

        nonhost = payload.get("best_nonhost_crc", {})
        if nonhost:
            diagnostics.append({
                "setting": LABELS[label],
                "risk_score": risk,
                "bound": payload.get("bound_mode", "hoeffding"),
                "mode": payload.get("tail_constraint_mode", "full"),
                "selection_mode": payload.get("selection_mode", "joint"),
                "harm_eps": fmt(payload.get("harm_eps", 0.0), 3),
                "action": nonhost.get("selected_action", ""),
                "threshold": fmt(nonhost.get("threshold"), 4),
                "cal_gain": fmt(nonhost.get("mean_gain"), 4, True),
                "cal_harm": fmt(nonhost.get("mean_harm"), 4),
                "cal_harmed_rate": fmt(nonhost.get("harmed_rate"), 3),
                "ucb_harmed_rate": fmt(nonhost.get("harm_rate_ucb"), 3),
                "cal_drop_gt_0.05": fmt(nonhost.get("drop_gt_0.05"), 3),
                "ucb_drop_gt_0.05": fmt(nonhost.get("drop05_rate_ucb"), 3),
                "cal_drop_gt_0.20": fmt(nonhost.get("drop_gt_0.20"), 3),
                "ucb_drop_gt_0.20": fmt(nonhost.get("drop20_rate_ucb"), 3),
                "cal_mean_harm": fmt(nonhost.get("mean_harm"), 4),
                "ucb_mean_harm": fmt(nonhost.get("mean_harm_ucb"), 4),
                "tested_thresholds_for_ucb": nonhost.get(
                    "tested_thresholds_for_ucb",
                    payload.get("crc_best", {}).get("tested_thresholds_for_ucb", ""),
                ),
                "failed": nonhost.get("failed_constraints", ""),
            })

    rows_best = [candidates[k][1] for k in sorted(candidates, key=lambda x: LABELS[x])]
    rows_best_tex = [
        {
            "setting": row["setting"],
            "split": row["split_group"],
                "action": row["action"],
                "eps": row["harm_eps"],
                "gain": row["gain"],
            "harm": row["harm"],
            "harmed_rate": row["harmed_rate"],
            "drop_gt_0.05": row["drop_gt_0.05"],
            "drop_gt_0.20": row["drop_gt_0.20"],
            "worst_drop": row["worst_drop"],
            "revert_rate": row["revert_rate"],
        }
        for row in rows_best
    ]
    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + "_all.csv", rows_all)
    write_md(str(out_prefix) + "_all.md", rows_all, "Tail-Risk Primary CRC All Risk Scores")
    write_tex(
        str(out_prefix) + "_all.tex",
        rows_all,
        "Tail-risk controlled primary SafeRefine policies across candidate risk scores.",
        "tab:tail_risk_primary_all",
    )
    write_csv(str(out_prefix) + "_best.csv", rows_best)
    write_md(str(out_prefix) + "_best.md", rows_best, "Tail-Risk Primary CRC Best Risk Score")
    write_tex(
        str(out_prefix) + "_best.tex",
        rows_best_tex,
        "Tail-risk controlled primary SafeRefine policy per setting, selected by calibration utility for compact reporting.",
        "tab:tail_risk_primary_best",
    )
    compact_diag = {}
    for row in diagnostics:
        key = row["setting"]
        score = (float(row["cal_gain"]), -float(row["cal_harm"]))
        if key not in compact_diag or score > compact_diag[key][0]:
            compact_diag[key] = (score, {
                "setting": row["setting"],
                "score": row["risk_score"],
                "cal_gain": row["cal_gain"],
                "hr": row["cal_harmed_rate"],
                "ucb_hr": row["ucb_harmed_rate"],
                "drop05": row["cal_drop_gt_0.05"],
                "ucb_drop05": row["ucb_drop_gt_0.05"],
                "drop20": fmt(row.get("cal_drop_gt_0.20"), 3),
                "ucb_drop20": fmt(row.get("ucb_drop_gt_0.20"), 3),
                "harm": row["cal_mean_harm"],
                "ucb_harm": row["ucb_mean_harm"],
                "fail": compact_failed(row["failed"]),
                "M": row.get("tested_thresholds_for_ucb", ""),
            })
    compact_diag_rows = [compact_diag[k][1] for k in sorted(compact_diag)]
    write_csv(str(out_prefix) + "_nonhost_diagnostics.csv", diagnostics)
    write_md(
        str(out_prefix) + "_nonhost_diagnostics.md",
        diagnostics,
        "Best Non-Host Tail-Risk Feasibility Diagnostics",
    )
    write_tex(
        str(out_prefix) + "_nonhost_diagnostics.tex",
        diagnostics,
        "Calibration diagnostics for the best non-host policy under the primary tail-risk constraints.",
        "tab:tail_risk_nonhost_diagnostics",
    )
    write_csv(str(out_prefix) + "_nonhost_diagnostics_compact.csv", compact_diag_rows)
    write_md(
        str(out_prefix) + "_nonhost_diagnostics_compact.md",
        compact_diag_rows,
        "Compact Best Non-Host Tail-Risk Diagnostics",
    )
    write_tex(
        str(out_prefix) + "_nonhost_diagnostics_compact.tex",
        compact_diag_rows,
        "Compact calibration diagnostics for the highest-gain non-host policy in each setting under the primary tail-risk constraints.",
        "tab:tail_risk_nonhost_diagnostics_compact",
    )
    print({
        "all_rows": len(rows_all),
        "best_rows": len(rows_best),
        "diagnostic_rows": len(diagnostics),
        "compact_diagnostic_rows": len(compact_diag_rows),
        "out_prefix": str(out_prefix),
    })


if __name__ == "__main__":
    main()
