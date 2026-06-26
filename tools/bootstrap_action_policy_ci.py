#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_safe_action_portfolio as P  # noqa: E402


def metric_vector(selected, hosts, harm_eps):
    gains = np.asarray([row["cpd"] - hosts[row["id"]]["cpd"] for row in selected], dtype=np.float64)
    harms = np.maximum(0.0, -gains)
    action_is_host = np.asarray([row["action"] == "host" for row in selected], dtype=bool)
    return {
        "mean_gain": gains,
        "mean_harm": harms,
        "worst_drop": gains,
        "harmed_rate": (gains < -float(harm_eps)).astype(np.float64),
        "reverted_rate": action_is_host.astype(np.float64),
    }


def summarize_boot(selected, hosts, args, rng):
    vec = metric_vector(selected, hosts, args.harm_eps)
    n = len(selected)
    idx = np.arange(n)
    rows = []
    for _ in range(args.n_boot):
        take = rng.choice(idx, size=n, replace=True)
        rows.append({
            "mean_gain": float(vec["mean_gain"][take].mean()),
            "mean_harm": float(vec["mean_harm"][take].mean()),
            "worst_drop": float(vec["worst_drop"][take].min()),
            "harmed_rate": float(vec["harmed_rate"][take].mean()),
            "reverted_rate": float(vec["reverted_rate"][take].mean()),
        })
    point = {
        "mean_gain": float(vec["mean_gain"].mean()) if n else 0.0,
        "mean_harm": float(vec["mean_harm"].mean()) if n else 0.0,
        "worst_drop": float(vec["worst_drop"].min()) if n else 0.0,
        "harmed_rate": float(vec["harmed_rate"].mean()) if n else 0.0,
        "reverted_rate": float(vec["reverted_rate"].mean()) if n else 0.0,
    }
    boot = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in point}
    out = {}
    lo_q = 100.0 * args.alpha / 2.0
    hi_q = 100.0 * (1.0 - args.alpha / 2.0)
    for key, val in point.items():
        out[key] = val
        out[f"{key}_lo"] = float(np.percentile(boot[key], lo_q)) if n else 0.0
        out[f"{key}_hi"] = float(np.percentile(boot[key], hi_q)) if n else 0.0
    return out


def fixed_selected(grouped, action):
    return P.select_action(grouped, action)


def calibrated_selected(grouped, calibration, args):
    action = calibration["selected_action"]
    threshold = calibration.get("threshold")
    if action == "host" or threshold is None:
        return P.select_action(grouped, "host")
    return P.select_threshold(grouped, action, float(threshold), args.risk_score)


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run(args):
    rows = P.load_rows(P.parse_inputs(args))
    actions = P.all_actions(rows)
    cal_ids, test_ids = P.split_ids(rows, args.cal_fraction)
    grouped_cal = P.group_rows(rows, cal_ids)
    grouped_test = P.group_rows(rows, test_ids)
    hosts_cal = P.host_by_id(grouped_cal)
    hosts_test = P.host_by_id(grouped_test)
    cal_best, _ = P.calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=False)
    crc_best, _ = P.calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=True)

    rng = np.random.default_rng(args.seed)
    selected_specs = [("host", P.select_action(grouped_test, "host")), ("oracle", P.select_oracle(grouped_test))]
    for action in actions:
        if action != "host":
            selected_specs.append((f"fixed:{action}", fixed_selected(grouped_test, action)))
    selected_specs.extend([
        ("calibrated_portfolio", calibrated_selected(grouped_test, cal_best, args)),
        ("crc_portfolio", calibrated_selected(grouped_test, crc_best, args)),
    ])

    out = []
    for policy, selected in selected_specs:
        stats = summarize_boot(selected, hosts_test, args, rng)
        out.append({
            "policy": policy,
            "n": len(selected),
            "risk_score": args.risk_score,
            "max_cal_harm_rate": args.max_cal_harm_rate,
            "crc_confidence": args.crc_confidence,
            "calibrated_selected_action": cal_best.get("selected_action"),
            "calibrated_threshold": cal_best.get("threshold"),
            "crc_selected_action": crc_best.get("selected_action"),
            "crc_threshold": crc_best.get("threshold"),
            **stats,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", action="append", default=[])
    ap.add_argument("--inputs", default="")
    ap.add_argument("--risk_score", choices=P.RISK_SCORE_MODES, default="changed")
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--harm_eps", type=float, default=0.0)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--max_cal_harm_rate", type=float, default=0.25)
    ap.add_argument("--crc_confidence", type=float, default=0.10)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    out = run(args)
    write_csv(args.out_csv, out)
    print(json.dumps({"out_csv": args.out_csv, "rows": len(out), "n_boot": args.n_boot}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
