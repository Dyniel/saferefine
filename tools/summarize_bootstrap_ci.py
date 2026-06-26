#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path


RISK_SUFFIXES = ("change_plus_geom", "changed", "geom")
MODES = ("practical", "strict")


def parse_name(path):
    stem = path.stem
    risk = next((r for r in RISK_SUFFIXES if stem.endswith("_" + r)), None)
    if risk is None:
        raise ValueError(f"Cannot parse risk score from {path}")
    prefix = stem[: -(len(risk) + 1)]
    mode = next((m for m in MODES if prefix.endswith("_" + m)), None)
    if mode is None:
        raise ValueError(f"Cannot parse mode from {path}")
    dataset = prefix[: -(len(mode) + 1)]
    return dataset, mode, risk


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "mode", "risk_score", "policy", "n",
        "mean_gain", "mean_gain_lo", "mean_gain_hi",
        "mean_harm", "mean_harm_lo", "mean_harm_hi",
        "worst_drop", "worst_drop_lo", "worst_drop_hi",
        "harmed_rate", "harmed_rate_lo", "harmed_rate_hi",
        "reverted_rate", "reverted_rate_lo", "reverted_rate_hi",
        "crc_selected_action", "crc_threshold",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/bootstrap_ci")
    ap.add_argument("--out_all", default="results/bootstrap_ci/binary_bootstrap_all_policies.csv")
    ap.add_argument("--out_crc", default="results/bootstrap_ci/binary_bootstrap_crc_ablation.csv")
    args = ap.parse_args()

    all_rows = []
    for path in sorted(Path(args.in_dir).glob("*.csv")):
        if path.name.startswith("binary_bootstrap_") or path.name.startswith("e120_bootstrap_"):
            continue
        if path.name.startswith("mediafinal_bootstrap_") or "_bootstrap_all_" in path.name or "_bootstrap_crc_" in path.name:
            continue
        dataset, mode, risk = parse_name(path)
        for row in read_rows(path):
            row = dict(row)
            row.update({"dataset": dataset, "mode": mode, "risk_score": risk})
            all_rows.append(row)

    all_rows.sort(key=lambda r: (r["dataset"], r["mode"], r["risk_score"], r["policy"]))
    crc_rows = [r for r in all_rows if r.get("policy") == "crc_portfolio"]
    write_csv(Path(args.out_all), all_rows)
    write_csv(Path(args.out_crc), crc_rows)
    print(f"wrote {args.out_all} rows={len(all_rows)}")
    print(f"wrote {args.out_crc} rows={len(crc_rows)}")


if __name__ == "__main__":
    main()
