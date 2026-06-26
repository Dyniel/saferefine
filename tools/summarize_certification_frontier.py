#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path

from summarize_decision_baselines import LABELS, fmt, write_csv, write_md, write_tex
from summarize_tail_risk_primary import run_label


def load_json(path):
    return json.loads(Path(path).read_text())


def gamma_from_dir(path):
    name = Path(path).name
    m = re.search(r"gamma_([0-9]+)p([0-9]+)", name)
    if not m:
        return None
    return float(f"{int(m.group(1))}.{m.group(2)}")


def test_crc(payload):
    return next(r for r in payload["summaries"] if r.get("split") == "test" and r.get("policy") == "crc_portfolio")


def best_frontier_row(payload, label, risk, gamma):
    row = test_crc(payload)
    nonhost = row.get("selected_action") not in {"", "host", None}
    return {
        "setting": LABELS[label],
        "gamma": fmt(gamma, 2),
        "risk_score": risk,
        "certified": "yes" if nonhost else "no",
        "action": row.get("selected_action", ""),
        "gain": fmt(row.get("mean_gain"), 4, True),
        "harm": fmt(row.get("mean_harm"), 4),
        "harmed_rate": fmt(row.get("harmed_rate"), 3),
        "drop_gt_0.05": fmt(row.get("drop_gt_0.05"), 3),
        "drop_gt_0.20": fmt(row.get("drop_gt_0.20"), 3),
        "worst_drop": fmt(row.get("worst_drop"), 4, True),
        "revert_rate": fmt(row.get("reverted_rate"), 3),
        "cal_utility": float(row.get("cal_utility", 0.0) or 0.0),
    }


def required_n_hoeffding(rate, gamma, m, delta, risk_count=3):
    rate = float(rate)
    gamma = float(gamma)
    m = max(1, int(float(m)))
    risk_count = max(1, int(float(risk_count)))
    if rate >= gamma:
        return math.inf
    return math.ceil(math.log((m * risk_count) / max(delta, 1e-12)) / (2.0 * (gamma - rate) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Directory containing gamma_* subdirectories, or comma-separated gamma=dir entries.")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--target_gamma", type=float, default=0.10)
    ap.add_argument("--risk_count", type=int, default=3)
    args = ap.parse_args()

    specs = []
    if "=" in args.inputs:
        for spec in [x.strip() for x in args.inputs.split(",") if x.strip()]:
            gamma_s, directory = spec.split("=", 1)
            specs.append((float(gamma_s), Path(directory)))
    else:
        root = Path(args.inputs)
        for directory in sorted(root.glob("gamma_*")):
            gamma = gamma_from_dir(directory)
            if gamma is not None:
                specs.append((gamma, directory))

    frontier_candidates = {}
    sample_candidates = {}
    all_rows = []
    for gamma, directory in specs:
        for path in sorted(directory.glob("*_tail_primary.json")):
            payload = load_json(path)
            label, risk = run_label(path)
            if label not in LABELS:
                continue
            row = best_frontier_row(payload, label, risk, gamma)
            key = (label, gamma)
            score = (
                row["certified"] == "yes",
                row["cal_utility"],
                float(row["gain"].replace("+", "")),
                -float(row["harm"]),
            )
            if key not in frontier_candidates or score > frontier_candidates[key][0]:
                frontier_candidates[key] = (score, row)
            all_rows.append({k: v for k, v in row.items() if k != "cal_utility"})

            if abs(gamma - args.target_gamma) < 1e-9:
                nonhost = payload.get("best_nonhost_crc", {})
                if nonhost:
                    m = nonhost.get(
                        "tested_thresholds_for_ucb",
                        payload.get("crc_best", {}).get("tested_thresholds_for_ucb", 1),
                    )
                    n = int(nonhost.get("n", payload.get("n_cal", 0)))
                    drop = float(nonhost.get("drop_gt_0.05", 0.0))
                    req = required_n_hoeffding(drop, args.target_gamma, m, payload.get("crc_confidence", 0.10), args.risk_count)
                    sample_row = {
                        "setting": LABELS[label],
                        "risk_score": risk,
                        "cal_n": str(n),
                        "M": str(m),
                        "R": str(args.risk_count),
                        "cal_drop05": fmt(drop, 3),
                        "ucb_drop05": fmt(nonhost.get("drop05_rate_ucb"), 3),
                        "cal_drop20": fmt(nonhost.get("drop_gt_0.20"), 3),
                        "ucb_drop20": fmt(nonhost.get("drop20_rate_ucb"), 3),
                        "required_n": "inf" if math.isinf(req) else str(req),
                        "extra_n": "inf" if math.isinf(req) else str(max(0, req - n)),
                        "cal_gain": fmt(nonhost.get("mean_gain"), 4, True),
                    }
                    sample_key = label
                    sample_score = (float(sample_row["cal_gain"].replace("+", "")), -drop)
                    if sample_key not in sample_candidates or sample_score > sample_candidates[sample_key][0]:
                        sample_candidates[sample_key] = (sample_score, sample_row)

    frontier_rows = [
        {k: v for k, v in frontier_candidates[k][1].items() if k != "cal_utility"}
        for k in sorted(frontier_candidates, key=lambda x: (LABELS[x[0]], x[1]))
    ]
    frontier_compact = [
        {
            "setting": row["setting"],
            "gamma": row["gamma"],
            "certified": row["certified"],
            "gain": row["gain"],
            "harm": row["harm"],
            "drop_gt_0.05": row["drop_gt_0.05"],
            "drop_gt_0.20": row["drop_gt_0.20"],
            "worst_drop": row["worst_drop"],
            "revert_rate": row["revert_rate"],
        }
        for row in frontier_rows
    ]
    sample_rows = [sample_candidates[k][1] for k in sorted(sample_candidates, key=lambda x: LABELS[x])]

    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + "_all.csv", all_rows)
    write_md(str(out_prefix) + "_all.md", all_rows, "Nested Certification Frontier All Risk Scores")
    write_csv(str(out_prefix) + "_best.csv", frontier_rows)
    write_md(str(out_prefix) + "_best.md", frontier_rows, "Nested Certification Frontier")
    write_tex(
        str(out_prefix) + "_best.tex",
        frontier_rows,
        "Nested Bernoulli certification frontier as the large-drop budget gamma is varied.",
        "tab:nested_certification_frontier",
    )
    write_csv(str(out_prefix) + "_compact.csv", frontier_compact)
    write_md(str(out_prefix) + "_compact.md", frontier_compact, "Nested Certification Frontier Compact")
    write_tex(
        str(out_prefix) + "_compact.tex",
        frontier_compact,
        "Compact nested Bernoulli certification frontier as the large-drop budget gamma is varied.",
        "tab:nested_certification_frontier_compact",
    )
    write_csv(str(out_prefix) + "_sample_size.csv", sample_rows)
    write_md(str(out_prefix) + "_sample_size.md", sample_rows, "Nested Certification Sample-Size Diagnostic")
    write_tex(
        str(out_prefix) + "_sample_size.tex",
        sample_rows,
        "Approximate calibration sample size required for the nested best non-host candidate to satisfy the drop05 Hoeffding upper bound at gamma=0.10, holding empirical drop05, M thresholds, and R risk constraints fixed.",
        "tab:nested_sample_size_requirement",
    )
    print({
        "frontier_rows": len(frontier_rows),
        "all_rows": len(all_rows),
        "sample_rows": len(sample_rows),
        "out_prefix": str(out_prefix),
    })


if __name__ == "__main__":
    main()
