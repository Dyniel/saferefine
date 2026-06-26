#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_safe_action_portfolio as P  # noqa: E402


def parse_specs(text):
    specs = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).stem
        specs.append((label.strip(), Path(path.strip())))
    if not specs:
        raise ValueError("Provide at least one label=csv input.")
    return specs


def canonicalize(rows):
    out = []
    for row in rows:
        row = dict(row)
        source = str(row.get("source", "src"))
        row["id"] = f"{source}:{row['id']}"
        if row["method"] == "host":
            row["action"] = "host"
        else:
            row["action"] = f"{row['method']}:a{float(row['alpha']):.2f}"
        out.append(row)
    return out


def all_ids(rows):
    return {row["id"] for row in rows}


def selected_from_calibration(grouped, calibration, args):
    action = calibration["selected_action"]
    threshold = calibration.get("threshold")
    if action == "host" or threshold is None:
        return P.select_action(grouped, "host")
    return P.select_threshold(grouped, action, float(threshold), args.risk_score)


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "policy", "n", "mean_cpd", "mean_gain", "mean_harm", "worst_drop",
        "improved_rate", "harmed_rate", "reverted_rate", "mean_changed",
        "selected_action", "threshold", "risk_score", "harm_rate_ucb", "action_counts",
    ]
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            out["action_counts"] = json.dumps(row.get("action_counts", {}), sort_keys=True)
            writer.writerow(out)


def run(args):
    cal_rows = canonicalize(P.load_rows(parse_specs(args.cal_inputs)))
    test_rows = canonicalize(P.load_rows(parse_specs(args.test_inputs)))
    actions = P.all_actions(cal_rows)

    grouped_cal = P.group_rows(cal_rows, all_ids(cal_rows))
    grouped_test = P.group_rows(test_rows, all_ids(test_rows))
    hosts_cal = P.host_by_id(grouped_cal)
    hosts_test = P.host_by_id(grouped_test)

    cal_best, cal_candidates = P.calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=False)
    crc_best, crc_candidates = P.calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=True)

    summaries = []
    for split, grouped, hosts in [("cal_source", grouped_cal, hosts_cal), ("test_target", grouped_test, hosts_test)]:
        summaries.append({"split": split, **P.summarize("host", P.select_action(grouped, "host"), hosts, args.harm_eps)})
        for action in actions:
            if action != "host":
                summaries.append({"split": split, **P.summarize(f"fixed:{action}", P.select_action(grouped, action), hosts, args.harm_eps)})
        summaries.append({"split": split, **P.summarize("oracle", P.select_oracle(grouped), hosts, args.harm_eps)})

        selected = selected_from_calibration(grouped, cal_best, args)
        row = P.summarize("calibrated_portfolio", selected, hosts, args.harm_eps)
        row.update({
            "selected_action": cal_best.get("selected_action"),
            "threshold": cal_best.get("threshold"),
            "risk_score": args.risk_score,
            "harm_rate_ucb": cal_best.get("harm_rate_ucb"),
        })
        summaries.append({"split": split, **row})

        selected = selected_from_calibration(grouped, crc_best, args)
        row = P.summarize("crc_portfolio", selected, hosts, args.harm_eps)
        row.update({
            "selected_action": crc_best.get("selected_action"),
            "threshold": crc_best.get("threshold"),
            "risk_score": args.risk_score,
            "harm_rate_ucb": crc_best.get("harm_rate_ucb"),
        })
        summaries.append({"split": split, **row})

    return {
        "cal_inputs": [{"label": label, "path": str(path)} for label, path in parse_specs(args.cal_inputs)],
        "test_inputs": [{"label": label, "path": str(path)} for label, path in parse_specs(args.test_inputs)],
        "risk_score": args.risk_score,
        "harm_eps": args.harm_eps,
        "beta_harm": args.beta_harm,
        "max_cal_harm_rate": args.max_cal_harm_rate,
        "crc_confidence": args.crc_confidence,
        "actions": actions,
        "calibrated_best": cal_best,
        "crc_best": crc_best,
        "calibrated_candidates": cal_candidates,
        "crc_candidates": crc_candidates,
        "summaries": summaries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_inputs", required=True, help="Comma-separated label=CSV sources used only for calibration.")
    ap.add_argument("--test_inputs", required=True, help="Comma-separated label=CSV targets used only for testing.")
    ap.add_argument("--risk_score", choices=P.RISK_SCORE_MODES, default="changed")
    ap.add_argument("--harm_eps", type=float, default=0.0)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--max_cal_harm_rate", type=float, default=0.25)
    ap.add_argument("--crc_confidence", type=float, default=0.10)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    payload = run(args)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(args.out_csv, payload["summaries"])
    target = next(r for r in payload["summaries"] if r["split"] == "test_target" and r["policy"] == "crc_portfolio")
    print(
        json.dumps({
            "out_csv": args.out_csv,
            "target_gain": target["mean_gain"],
            "target_harm": target["mean_harm"],
            "target_worst_drop": target["worst_drop"],
            "target_reverted_rate": target["reverted_rate"],
            "selected_action": target.get("selected_action"),
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
