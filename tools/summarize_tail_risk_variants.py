#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from summarize_decision_baselines import LABELS, fmt, write_csv, write_md, write_tex
from summarize_tail_risk_primary import run_label, test_crc


def load_json(path):
    return json.loads(Path(path).read_text())


def test_row(payload, policy):
    return next(r for r in payload["summaries"] if r.get("split") == "test" and r.get("policy") == policy)


def oracle_gain(payload):
    return float(test_row(payload, "oracle").get("mean_gain", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Comma-separated name=directory entries.")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    variant_rows = []
    relaxed_candidates = {}
    for spec in [x.strip() for x in args.inputs.split(",") if x.strip()]:
        name, directory = spec.split("=", 1)
        for path in sorted(Path(directory).glob("*_tail_primary.json")):
            payload = load_json(path)
            label, risk = run_label(path)
            if label not in LABELS:
                continue
            crc = test_crc(payload)
            variant_rows.append({
                "setting": LABELS[label],
                "variant": name,
                "risk_score": risk,
                "action": crc.get("selected_action", ""),
                "gain": fmt(crc.get("mean_gain"), 4, True),
                "harm": fmt(crc.get("mean_harm"), 4),
                "harmed_rate": fmt(crc.get("harmed_rate"), 3),
                "drop_gt_0.05": fmt(crc.get("drop_gt_0.05"), 3),
                "worst_drop": fmt(crc.get("worst_drop"), 4, True),
                "revert_rate": fmt(crc.get("reverted_rate"), 3),
            })

            relaxed = test_row(payload, "calibrated_portfolio")
            key = (label, name)
            score = (
                float(relaxed.get("cal_utility", 0.0)),
                float(relaxed.get("mean_gain", 0.0)),
                -float(relaxed.get("mean_harm", 0.0)),
            )
            if key not in relaxed_candidates or score > relaxed_candidates[key][0]:
                ogain = oracle_gain(payload)
                gain = float(relaxed.get("mean_gain", 0.0))
                recovery = gain / ogain if ogain > 0 else 0.0
                relaxed_candidates[key] = (score, {
                    "setting": LABELS[label],
                    "variant": name,
                    "risk_score": risk,
                    "action": relaxed.get("selected_action", ""),
                    "gain": fmt(gain, 4, True),
                    "harm": fmt(relaxed.get("mean_harm"), 4),
                    "harmed_rate": fmt(relaxed.get("harmed_rate"), 3),
                    "drop_gt_0.05": fmt(relaxed.get("drop_gt_0.05"), 3),
                    "worst_drop": fmt(relaxed.get("worst_drop"), 4, True),
                    "revert_rate": fmt(relaxed.get("reverted_rate"), 3),
                    "oracle_gain": fmt(ogain, 4, True),
                    "oracle_recovery": fmt(recovery, 3),
                })

    compact_variants = []
    for (setting, variant), group in sorted(
        {
            (r["setting"], r["variant"]): [x for x in variant_rows if x["setting"] == r["setting"] and x["variant"] == r["variant"]]
            for r in variant_rows
        }.items()
    ):
        best = max(group, key=lambda r: (float(r["gain"].replace("+", "")), -float(r["harm"])))
        compact_variants.append(best)

    relaxed_rows_all = [relaxed_candidates[k][1] for k in sorted(relaxed_candidates)]
    relaxed_rows = [r for r in relaxed_rows_all if r["variant"] == "full"]
    if not relaxed_rows:
        relaxed_rows = relaxed_rows_all
    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + "_primary_variants_all.csv", variant_rows)
    write_md(str(out_prefix) + "_primary_variants_all.md", variant_rows, "Tail-Risk Primary Variants All Risk Scores")
    write_csv(str(out_prefix) + "_primary_variants_best.csv", compact_variants)
    write_md(str(out_prefix) + "_primary_variants_best.md", compact_variants, "Tail-Risk Primary Variants Best Risk Score")
    write_tex(
        str(out_prefix) + "_primary_variants_best.tex",
        compact_variants,
        "Primary tail-risk variants. The full variant constrains harmed-rate, large-drop rate, and mean harm; the Bernoulli variant constrains only the two Bernoulli tail events.",
        "tab:tail_risk_primary_variants",
    )
    write_csv(str(out_prefix) + "_relaxed_protocol_all.csv", relaxed_rows_all)
    write_csv(str(out_prefix) + "_relaxed_protocol.csv", relaxed_rows)
    write_md(str(out_prefix) + "_relaxed_protocol.md", relaxed_rows, "Predefined Relaxed SafeRefine Protocol")
    write_tex(
        str(out_prefix) + "_relaxed_protocol.tex",
        relaxed_rows,
        "Predefined relaxed SafeRefine protocol selected by calibration utility and evaluated once on the test split.",
        "tab:relaxed_saferefine_protocol",
    )
    print({
        "variant_rows": len(variant_rows),
        "compact_variant_rows": len(compact_variants),
        "relaxed_rows": len(relaxed_rows),
        "out_prefix": str(out_prefix),
    })


if __name__ == "__main__":
    main()
