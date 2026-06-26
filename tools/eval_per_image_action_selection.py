#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import eval_safe_action_portfolio as P  # noqa: E402
from eval_decision_baselines import BASE_FEATURES, extended_summary, group_key, write_csv  # noqa: E402


def load_rows(label, path):
    return P.load_rows([(label, Path(path))])


def split_groups(rows, split_group, meta_fraction, cal_fraction):
    groups = []
    seen = set()
    group_to_ids = {}
    for row in rows:
        gid = group_key(row["id"], split_group)
        group_to_ids.setdefault(gid, set()).add(row["id"])
        if gid not in seen:
            seen.add(gid)
            groups.append(gid)
    n = len(groups)
    if n < 3:
        raise ValueError(f"Need at least 3 groups for meta/cal/test split, got {n}.")
    n_meta = max(1, int(round(n * meta_fraction)))
    n_cal = max(1, int(round(n * cal_fraction)))
    if n_meta + n_cal >= n:
        n_cal = max(1, n - n_meta - 1)
    meta_groups = set(groups[:n_meta])
    cal_groups = set(groups[n_meta:n_meta + n_cal])
    test_groups = set(groups[n_meta + n_cal:])
    out = {"meta": set(), "cal": set(), "test": set()}
    for gid, ids in group_to_ids.items():
        if gid in meta_groups:
            out["meta"].update(ids)
        elif gid in cal_groups:
            out["cal"].update(ids)
        else:
            out["test"].update(ids)
    return out, {"meta": meta_groups, "cal": cal_groups, "test": test_groups}


def row_gain(row, hosts):
    return float(row["cpd"] - hosts[row["id"]]["cpd"])


def features(row):
    vals = [float(row.get(k, 0.0)) for k in BASE_FEATURES]
    vals.append(float(row.get("host_entropy", 0.0)) * float(row.get("changed", 0.0)))
    vals.append(float(row.get("geom_risk", 0.0)) * float(row.get("host_entropy", 0.0)))
    return np.asarray(vals, dtype=np.float64)


def fit_ridge(rows, hosts, target, l2=1e-2):
    x, y = [], []
    for row in rows:
        if row["action"] == "host":
            continue
        gain = row_gain(row, hosts)
        if target == "gain":
            t = gain
        elif target == "harm":
            t = max(0.0, -gain)
        elif target == "drop05":
            t = 1.0 if gain < -0.05 else 0.0
        else:
            raise ValueError(target)
        x.append(features(row))
        y.append(t)
    if not x:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    reg = np.eye(xb.shape[1]) * l2
    reg[0, 0] = 0.0
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return {"mean": mean, "std": std, "w": w}


def predict(model, row):
    if model is None or row["action"] == "host":
        return 0.0
    x = (features(row) - model["mean"]) / model["std"]
    xb = np.concatenate([[1.0], x])
    return float(xb @ model["w"])


def hoeffding_upper(mean, n, confidence, n_candidates, scale=1.0):
    if n <= 0:
        return scale
    delta = max(float(confidence), 1e-12) / max(1, int(n_candidates))
    radius = scale * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return float(min(scale, mean + radius))


def candidate_thresholds(grouped, risk_score, max_thresholds):
    vals = []
    for actions in grouped.values():
        for action, row in actions.items():
            if action != "host":
                vals.append(float(P.risk_score(row, risk_score)))
    if not vals:
        return [0.0]
    vals = np.asarray(vals, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, max(2, int(max_thresholds)))
    out = {0.0}
    out.update(float(x) for x in np.quantile(vals, qs))
    return sorted(out)


def select_per_image(grouped, models, risk_score, risk_threshold, beta_harm, min_pred_utility=0.0):
    selected = []
    for _image_id, actions in grouped.items():
        best_row = actions["host"]
        best_utility = 0.0
        for action, row in actions.items():
            if action == "host":
                continue
            if P.risk_score(row, risk_score) > risk_threshold + 1e-12:
                continue
            pred_gain = predict(models["gain"], row)
            pred_harm = max(0.0, predict(models["harm"], row))
            utility = pred_gain - beta_harm * pred_harm
            if utility > best_utility and utility >= min_pred_utility:
                best_utility = utility
                best_row = row
        selected.append(best_row)
    return selected


def calibrate(grouped_cal, hosts_cal, models, args, risk_score):
    thresholds = candidate_thresholds(grouped_cal, risk_score, args.max_thresholds)
    candidates = []
    n_candidates = max(1, len(thresholds))
    host = extended_summary("per_image_host", P.select_action(grouped_cal, "host"), hosts_cal, args.harm_eps)
    host.update({
        "risk_score": risk_score,
        "threshold": None,
        "utility": 0.0,
        "harmed_rate_ucb": 0.0,
        "drop05_ucb": 0.0,
        "mean_harm_ucb": 0.0,
        "forced_host": True,
    })
    candidates.append(host)
    for threshold in thresholds:
        selected = select_per_image(grouped_cal, models, risk_score, threshold, args.beta_harm, args.min_pred_utility)
        row = extended_summary(f"per_image_{risk_score}@{threshold:.6g}", selected, hosts_cal, args.harm_eps)
        row.update({
            "risk_score": risk_score,
            "threshold": float(threshold),
            "utility": float(row["mean_gain"] - args.beta_harm * row["mean_harm"]),
            "harmed_rate_ucb": hoeffding_upper(row["harmed_rate"], row["n"], args.crc_confidence, n_candidates),
            "drop05_ucb": hoeffding_upper(row["drop_gt_0.05"], row["n"], args.crc_confidence, n_candidates),
            "mean_harm_ucb": hoeffding_upper(
                row["mean_harm"],
                row["n"],
                args.crc_confidence,
                n_candidates,
                scale=args.mean_harm_scale,
            ),
            "forced_host": False,
        })
        candidates.append(row)
    feasible = [
        c for c in candidates
        if c["harmed_rate_ucb"] <= args.max_cal_harmed_rate
        and c["drop05_ucb"] <= args.max_cal_drop05_rate
        and c["mean_harm_ucb"] <= args.max_cal_mean_harm
    ]
    best = max(feasible, key=lambda r: (r["utility"], r["mean_gain"], -r["mean_harm"], -r["reverted_rate"]))
    best = dict(best)
    best["feasible_candidates"] = len(feasible)
    best["tested_candidates"] = len(candidates)
    return best, candidates


def apply_policy(grouped_test, hosts_test, models, args, calibration):
    if calibration.get("forced_host") or calibration.get("threshold") is None:
        selected = P.select_action(grouped_test, "host")
    else:
        selected = select_per_image(
            grouped_test,
            models,
            calibration["risk_score"],
            float(calibration["threshold"]),
            args.beta_harm,
            args.min_pred_utility,
        )
    row = extended_summary(f"per_image_{calibration['risk_score']}", selected, hosts_test, args.harm_eps)
    for key in (
        "risk_score", "threshold", "harmed_rate_ucb", "drop05_ucb", "mean_harm_ucb",
        "feasible_candidates", "tested_candidates", "forced_host",
    ):
        row[key] = calibration.get(key)
    return row


def write_md(path, rows, title):
    fields = list(rows[0].keys()) if rows else []
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--split_group", choices=["image", "patient"], default="image")
    ap.add_argument("--meta_fraction", type=float, default=0.34)
    ap.add_argument("--cal_fraction", type=float, default=0.33)
    ap.add_argument("--risk_scores", default="changed,geom,change_plus_geom,host_uncertainty,quality_risk")
    ap.add_argument("--harm_eps", type=float, default=0.0)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--min_pred_utility", type=float, default=0.0)
    ap.add_argument("--max_cal_harmed_rate", type=float, default=0.25)
    ap.add_argument("--max_cal_drop05_rate", type=float, default=0.10)
    ap.add_argument("--max_cal_mean_harm", type=float, default=0.02)
    ap.add_argument("--mean_harm_scale", type=float, default=0.10)
    ap.add_argument("--crc_confidence", type=float, default=0.10)
    ap.add_argument("--max_thresholds", type=int, default=64)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    label = args.label or Path(args.input_csv).stem
    rows = load_rows(label, args.input_csv)
    ids, group_splits = split_groups(rows, args.split_group, args.meta_fraction, args.cal_fraction)
    grouped = {split: P.group_rows(rows, ids[split]) for split in ("meta", "cal", "test")}
    hosts = {split: P.host_by_id(grouped[split]) for split in ("meta", "cal", "test")}

    meta_rows = [r for r in rows if r["id"] in ids["meta"]]
    models = {
        "gain": fit_ridge(meta_rows, hosts["meta"], "gain"),
        "harm": fit_ridge(meta_rows, hosts["meta"], "harm"),
        "drop05": fit_ridge(meta_rows, hosts["meta"], "drop05"),
    }

    summaries = []
    summaries.append({"split": "test", **extended_summary("host", P.select_action(grouped["test"], "host"), hosts["test"], args.harm_eps)})
    summaries.append({"split": "test", **extended_summary("oracle", P.select_oracle(grouped["test"]), hosts["test"], args.harm_eps)})
    calibrations = []
    for risk_score in [x.strip() for x in args.risk_scores.split(",") if x.strip()]:
        cal, candidates = calibrate(grouped["cal"], hosts["cal"], models, args, risk_score)
        calibrations.append(cal)
        summaries.append({"split": "test", **apply_policy(grouped["test"], hosts["test"], models, args, cal)})

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_csv": args.input_csv,
        "label": label,
        "split_group": args.split_group,
        "n_meta_images": len(ids["meta"]),
        "n_cal_images": len(ids["cal"]),
        "n_test_images": len(ids["test"]),
        "n_meta_groups": len(group_splits["meta"]),
        "n_cal_groups": len(group_splits["cal"]),
        "n_test_groups": len(group_splits["test"]),
        "constraints": {
            "max_cal_harmed_rate": args.max_cal_harmed_rate,
            "max_cal_drop05_rate": args.max_cal_drop05_rate,
            "max_cal_mean_harm": args.max_cal_mean_harm,
            "crc_confidence": args.crc_confidence,
        },
        "calibrations": calibrations,
        "summaries": summaries,
    }
    Path(str(out_prefix) + ".json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(str(out_prefix) + "_policies.csv", summaries)
    write_md(str(out_prefix) + "_policies.md", summaries, "Per-Image Action Selection Policies")
    print(json.dumps({
        "out_prefix": str(out_prefix),
        "n_test_images": len(ids["test"]),
        "n_test_groups": len(group_splits["test"]),
        "best_test_policy": max(summaries, key=lambda r: (r["mean_gain"], -r["mean_harm"]))["policy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
