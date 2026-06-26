#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

from summarize_decision_baselines import LABELS, fmt, write_csv, write_md, write_tex


POLICIES = {
    "host",
    "oracle",
    "per_image_changed",
    "per_image_geom",
    "per_image_change_plus_geom",
    "per_image_host_uncertainty",
    "per_image_quality_risk",
}


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def run_from_path(path):
    name = Path(path).name
    return name[:-len("_policies.csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.input_dir).glob("*_policies.csv")):
        run = run_from_path(path)
        if run not in LABELS:
            continue
        for row in read_csv(path):
            if row.get("policy") not in POLICIES:
                continue
            rows.append({
                "setting": LABELS.get(run, run),
                "policy": row["policy"],
                "gain": fmt(row.get("mean_gain"), 4, True),
                "harm": fmt(row.get("mean_harm"), 4),
                "harmed_rate": fmt(row.get("harmed_rate"), 3),
                "drop_gt_0.05": fmt(row.get("drop_gt_0.05"), 3),
                "cvar_harm_10": fmt(row.get("cvar_harm_10"), 4),
                "worst_drop": fmt(row.get("worst_drop"), 4, True),
                "revert_rate": fmt(row.get("reverted_rate"), 3),
            })
    out = Path(args.out_prefix)
    write_csv(str(out) + ".csv", rows)
    write_md(str(out) + ".md", rows, "Per-Image Action Selection")
    write_tex(
        str(out) + ".tex",
        rows,
        "Per-image action selection with separate meta-training, calibration, and test splits and multi-constraint feasibility.",
        "tab:per_image_action_selection",
    )
    print({"rows": len(rows), "out_prefix": str(out)})


if __name__ == "__main__":
    main()
