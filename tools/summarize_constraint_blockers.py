#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import eval_safe_action_portfolio as P
from summarize_decision_baselines import LABELS, fmt, write_csv, write_md, write_tex
from summarize_tail_risk_primary import compact_failed, run_label


def load_json(path):
    return json.loads(Path(path).read_text())


def grouped_test_from_payload(payload):
    specs = [(item["label"], Path(item["path"])) for item in payload["inputs"]]
    rows = P.load_rows(specs)
    split_group = payload.get("split_group", "image")
    if payload.get("selection_mode") == "nested":
        _select_ids, _cal_ids, test_ids, *_ = P.split_ids_nested(
            rows,
            payload.get("nested_select_fraction", 0.25),
            payload.get("nested_cal_fraction", 0.25),
            split_group,
        )
    else:
        _cal_ids, test_ids, *_ = P.split_ids(rows, payload.get("cal_fraction", 0.5), split_group)
    grouped_test = P.group_rows(rows, test_ids)
    hosts_test = P.host_by_id(grouped_test)
    return grouped_test, hosts_test


def apply_candidate_to_test(payload, candidate):
    grouped_test, hosts_test = grouped_test_from_payload(payload)
    action = candidate.get("selected_action", "host")
    threshold = candidate.get("threshold")
    if action == "host" or threshold is None:
        selected = P.select_action(grouped_test, "host")
    else:
        selected = P.select_threshold(grouped_test, action, float(threshold), payload.get("risk_score", "changed"))
    return P.summarize("best_nonhost_on_test", selected, hosts_test, payload.get("harm_eps", 0.0))


def yesno(flag):
    return "yes" if flag else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma05", type=float, default=0.10)
    ap.add_argument("--gamma20", type=float, default=0.05)
    args = ap.parse_args()

    best_by_label = {}
    for path in sorted(Path(args.input_dir).glob("*_tail_primary.json")):
        payload = load_json(path)
        label, risk = run_label(path)
        if label not in LABELS:
            continue
        cand = payload.get("best_nonhost_crc", {})
        if not cand or cand.get("selected_action") == "host":
            continue
        test = apply_candidate_to_test(payload, cand)
        ucb_h = float(cand.get("harm_rate_ucb", 1.0) or 1.0)
        ucb_d05 = float(cand.get("drop05_rate_ucb", 1.0) or 1.0)
        ucb_d20 = float(cand.get("drop20_rate_ucb", 1.0) or 1.0)
        primary_ok = ucb_h <= args.alpha and ucb_d05 <= args.gamma05
        severe_ok = primary_ok and ucb_d20 <= args.gamma20
        row = {
            "setting": LABELS[label],
            "risk_score": risk,
            "cal_gain": fmt(cand.get("mean_gain"), 4, True),
            "test_gain": fmt(test.get("mean_gain"), 4, True),
            "test_worst": fmt(test.get("worst_drop"), 4, True),
            "req_alpha": fmt(ucb_h, 3),
            "req_gamma05": fmt(ucb_d05, 3),
            "req_gamma20": fmt(ucb_d20, 3),
            "blocker": compact_failed(cand.get("failed_constraints", "")),
        }
        score = (
            float(cand.get("mean_gain", 0.0)),
            -float(cand.get("mean_harm", 0.0)),
            float(test.get("mean_gain", 0.0)),
        )
        if label not in best_by_label or score > best_by_label[label][0]:
            best_by_label[label] = (score, row)

    rows = [best_by_label[k][1] for k in sorted(best_by_label, key=lambda x: LABELS[x])]
    out_prefix = Path(args.out_prefix)
    write_csv(str(out_prefix) + ".csv", rows)
    write_md(str(out_prefix) + ".md", rows, "Nested Constraint Blocker Diagnostics")
    write_tex(
        str(out_prefix) + ".tex",
        rows,
        "Constraint-blocker diagnostics for the highest-gain nested non-host candidate at gamma=0.10. Required budgets are the finite-sample upper bounds that would need to be accepted to certify the candidate; the primary budgets are alpha=0.25 and gamma05=0.10, with gamma20=0.05 in the severe-tail sensitivity.",
        "tab:nested_constraint_blockers",
    )
    print({"rows": len(rows), "out_prefix": str(out_prefix)})


if __name__ == "__main__":
    main()
