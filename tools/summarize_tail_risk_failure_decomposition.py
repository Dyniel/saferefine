#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from summarize_decision_baselines import LABELS, write_csv, write_md, write_tex
from summarize_tail_risk_primary import run_label


def load_json(path):
    return json.loads(Path(path).read_text())


def bucket(failed):
    parts = {x for x in str(failed).split(",") if x}
    if not parts:
        return "feasible"
    if len(parts) > 1:
        return "multiple"
    return next(iter(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    counts = {}
    for path in sorted(Path(args.input_dir).glob("*_tail_primary.json")):
        payload = load_json(path)
        label, _risk = run_label(path)
        if label not in LABELS:
            continue
        setting = LABELS[label]
        row = counts.setdefault(setting, {
            "setting": setting,
            "nonhost_candidates": 0,
            "feasible": 0,
            "harm_rate": 0,
            "drop05_rate": 0,
            "mean_harm": 0,
            "multiple": 0,
        })
        for cand in payload.get("crc_candidates", []):
            if cand.get("selected_action") == "host":
                continue
            row["nonhost_candidates"] += 1
            row[bucket(cand.get("failed_constraints", ""))] += 1

    rows = [counts[k] for k in sorted(counts)]
    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + ".csv", rows)
    write_md(str(out_prefix) + ".md", rows, "Tail-Risk Failure Decomposition")
    write_tex(
        str(out_prefix) + ".tex",
        rows,
        "Failure decomposition over non-host action-threshold candidates under the full primary tail-risk constraints.",
        "tab:tail_risk_failure_decomposition",
    )
    print({"rows": len(rows), "out_prefix": str(out_prefix)})


if __name__ == "__main__":
    main()
