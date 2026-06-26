#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from summarize_decision_baselines import LABELS, fmt, write_csv, write_md, write_tex
from summarize_tail_risk_primary import run_label, test_crc


def load_json(path):
    return json.loads(Path(path).read_text())


def fval(text):
    try:
        return float(str(text).replace("+", ""))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    rows = []
    diagnostics = []
    for path in sorted(Path(args.input_root).glob("*/*_tail_primary.json")):
        payload = load_json(path)
        label, risk = run_label(path)
        if label not in LABELS:
            continue
        row = test_crc(payload)
        bound = payload.get("bound_mode", path.parent.name)
        rows.append({
            "setting": LABELS[label],
            "bound": bound,
            "risk_score": risk,
            "action": row.get("selected_action", ""),
            "gain": fmt(row.get("mean_gain"), 4, True),
            "harm": fmt(row.get("mean_harm"), 4),
            "harmed_rate": fmt(row.get("harmed_rate"), 3),
            "drop_gt_0.05": fmt(row.get("drop_gt_0.05"), 3),
            "worst_drop": fmt(row.get("worst_drop"), 4, True),
            "revert_rate": fmt(row.get("reverted_rate"), 3),
            "split_group": payload.get("split_group", "image"),
        })
        nonhost = payload.get("best_nonhost_crc", {})
        if nonhost:
            diagnostics.append({
                "setting": LABELS[label],
                "bound": bound,
                "risk_score": risk,
                "action": nonhost.get("selected_action", ""),
                "cal_gain": fmt(nonhost.get("mean_gain"), 4, True),
                "cal_harm": fmt(nonhost.get("mean_harm"), 4),
                "cal_harmed_rate": fmt(nonhost.get("harmed_rate"), 3),
                "ucb_harmed_rate": fmt(nonhost.get("harm_rate_ucb"), 3),
                "cal_drop_gt_0.05": fmt(nonhost.get("drop_gt_0.05"), 3),
                "ucb_drop_gt_0.05": fmt(nonhost.get("drop05_rate_ucb"), 3),
                "cal_mean_harm": fmt(nonhost.get("mean_harm"), 4),
                "ucb_mean_harm": fmt(nonhost.get("mean_harm_ucb"), 4),
                "failed": nonhost.get("failed_constraints", ""),
            })

    rows.sort(key=lambda r: (r["setting"], r["bound"], r["risk_score"]))
    diagnostics.sort(key=lambda r: (r["setting"], r["bound"], r["risk_score"]))
    aggregate = []
    for bound in sorted({r["bound"] for r in rows}):
        subset = [r for r in rows if r["bound"] == bound]
        nonhost = [r for r in subset if r["action"] != "host"]
        aggregate.append({
            "bound": bound,
            "policies": len(subset),
            "nonhost_selected": len(nonhost),
            "host_selected": len(subset) - len(nonhost),
            "settings_with_nonhost": len({r["setting"] for r in nonhost}),
            "max_gain": max((r["gain"] for r in subset), key=fval, default="+0.0000"),
            "min_revert_rate": min((r["revert_rate"] for r in subset), key=fval, default="1.000"),
        })
    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + "_aggregate.csv", aggregate)
    write_md(str(out_prefix) + "_aggregate.md", aggregate, "Tail-Risk Bound Sensitivity Aggregate")
    write_tex(
        str(out_prefix) + "_aggregate.tex",
        aggregate,
        "Aggregate sensitivity of the primary tail-risk policy to the finite-sample upper-bound choice.",
        "tab:tail_risk_bound_sensitivity_aggregate",
    )
    write_csv(str(out_prefix) + "_policies.csv", rows)
    write_md(str(out_prefix) + "_policies.md", rows, "Tail-Risk Bound Sensitivity Policies")
    write_tex(
        str(out_prefix) + "_policies.tex",
        rows,
        "Tail-risk primary policy sensitivity across finite-sample upper-bound choices.",
        "tab:tail_risk_bound_sensitivity",
    )
    write_csv(str(out_prefix) + "_nonhost_diagnostics.csv", diagnostics)
    write_md(
        str(out_prefix) + "_nonhost_diagnostics.md",
        diagnostics,
        "Tail-Risk Bound Sensitivity Non-Host Diagnostics",
    )
    write_tex(
        str(out_prefix) + "_nonhost_diagnostics.tex",
        diagnostics,
        "Best non-host calibration diagnostics across finite-sample upper-bound choices.",
        "tab:tail_risk_bound_nonhost_diagnostics",
    )
    print({
        "aggregate_rows": len(aggregate),
        "policy_rows": len(rows),
        "diagnostic_rows": len(diagnostics),
        "out_prefix": str(out_prefix),
    })


if __name__ == "__main__":
    main()
