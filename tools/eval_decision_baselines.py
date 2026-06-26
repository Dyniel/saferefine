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


BASE_FEATURES = (
    "changed",
    "geom_risk",
    "disc_area_delta",
    "disc_centroid_shift",
    "component_delta",
    "alpha",
    "host_entropy",
    "host_confidence",
    "host_margin",
)


def load_action_rows(label, csv_path):
    return P.load_rows([(label, Path(csv_path))])


def group_key(image_id, mode):
    if mode == "image":
        return image_id
    if mode == "patient":
        m = re.match(r"^(la_\d+)_z\d+", image_id)
        if m:
            return m.group(1)
        parts = image_id.split("_")
        if len(parts) >= 2 and parts[-1].startswith("z") and parts[-1][1:].isdigit():
            return "_".join(parts[:-1])
        return image_id
    raise ValueError(f"Unknown split group mode: {mode}")


def split_ids(rows, cal_fraction, group_mode):
    groups = []
    seen_group = set()
    group_to_ids = {}
    for row in rows:
        gid = group_key(row["id"], group_mode)
        group_to_ids.setdefault(gid, set()).add(row["id"])
        if gid not in seen_group:
            seen_group.add(gid)
            groups.append(gid)
    n_cal = int(round(len(groups) * cal_fraction))
    n_cal = min(max(1, n_cal), max(1, len(groups) - 1))
    cal_groups = set(groups[:n_cal])
    cal_ids, test_ids = set(), set()
    for gid, ids in group_to_ids.items():
        if gid in cal_groups:
            cal_ids.update(ids)
        else:
            test_ids.update(ids)
    return cal_ids, test_ids, cal_groups, set(groups) - cal_groups


def row_gain(row, hosts):
    return float(row["cpd"] - hosts[row["id"]]["cpd"])


def extended_summary(name, selected, hosts, harm_eps=0.0):
    row = P.summarize(name, selected, hosts, harm_eps)
    gains = np.asarray([row_gain(r, hosts) for r in selected], dtype=np.float64)
    harms = np.maximum(0.0, -gains)
    for eps in (0.01, 0.05, 0.10):
        row[f"drop_gt_{eps:.2f}"] = float((gains < -eps).mean()) if len(gains) else 0.0
    if len(harms):
        k = max(1, int(math.ceil(0.10 * len(harms))))
        row["cvar_harm_10"] = float(np.sort(harms)[-k:].mean())
    else:
        row["cvar_harm_10"] = 0.0
    return row


def feature_matrix(rows, hosts):
    x = []
    y = []
    for row in rows:
        if row["action"] == "host":
            continue
        vals = [float(row.get(k, 0.0)) for k in BASE_FEATURES]
        vals.append(float(row.get("host_entropy", 0.0)) * float(row.get("changed", 0.0)))
        vals.append(float(row.get("geom_risk", 0.0)) * float(row.get("host_entropy", 0.0)))
        x.append(vals)
        y.append(1.0 if row_gain(row, hosts) < 0.0 else 0.0)
    if not x:
        return np.zeros((0, len(BASE_FEATURES) + 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def fit_logistic_ridge(rows, hosts, l2=1e-2, lr=0.2, steps=1200):
    x, y = feature_matrix(rows, hosts)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return None
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
    w = np.zeros(xb.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = np.clip(xb @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = xb.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return {"mean": mean, "std": std, "w": w}


def predict_quality(model, row):
    if model is None:
        return 0.0
    vals = np.asarray([float(row.get(k, 0.0)) for k in BASE_FEATURES] + [
        float(row.get("host_entropy", 0.0)) * float(row.get("changed", 0.0)),
        float(row.get("geom_risk", 0.0)) * float(row.get("host_entropy", 0.0)),
    ], dtype=np.float64)
    xs = (vals - model["mean"]) / model["std"]
    xb = np.concatenate([[1.0], xs])
    z = float(np.clip(xb @ model["w"], -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def inject_quality_risk(rows, model):
    out = []
    for row in rows:
        row = dict(row)
        if row["action"] == "host":
            row["quality_risk"] = 0.0
        else:
            row["quality_risk"] = predict_quality(model, row)
        out.append(row)
    return out


def calibrate_and_apply(grouped_cal, hosts_cal, grouped_test, hosts_test, actions, args, risk_score):
    old_risk = args.risk_score
    args.risk_score = risk_score
    best, _ = P.calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=True)
    selected = P.apply_calibrated(grouped_test, hosts_test, best, args, f"crc_{risk_score}")
    args.risk_score = old_risk
    return best, selected


def select_best_fixed(grouped_cal, hosts_cal, grouped_test, actions, harm_eps):
    best = None
    for action in actions:
        if action == "host":
            continue
        summary = extended_summary(action, P.select_action(grouped_cal, action), hosts_cal, harm_eps)
        key = (summary["mean_gain"] - 2.0 * summary["mean_harm"], summary["mean_gain"], -summary["mean_harm"])
        if best is None or key > best[0]:
            best = (key, action)
    action = best[1] if best else "host"
    return action, P.select_action(grouped_test, action)


def random_rate_matched(grouped_test, hosts_test, action, accept_rate, seed, repeats, harm_eps):
    ids = list(grouped_test.keys())
    n_accept = int(round(float(accept_rate) * len(ids)))
    rng = np.random.default_rng(seed)
    summaries = []
    for rep in range(repeats):
        accept = set(rng.choice(ids, size=n_accept, replace=False)) if n_accept > 0 else set()
        selected = []
        for image_id in ids:
            actions = grouped_test[image_id]
            if image_id in accept and action in actions:
                selected.append(actions[action])
            else:
                selected.append(actions["host"])
        summaries.append(extended_summary(f"random_rate_matched_{rep}", selected, hosts_test, harm_eps))
    keys = [
        "mean_gain", "mean_harm", "worst_drop", "harmed_rate", "reverted_rate",
        "drop_gt_0.01", "drop_gt_0.05", "drop_gt_0.10", "cvar_harm_10",
    ]
    out = {"policy": "random_rate_matched", "n": len(ids)}
    for key in keys:
        vals = np.asarray([s[key] for s in summaries], dtype=np.float64)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["selected_action"] = action
    out["accept_rate"] = float(accept_rate)
    return out


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.sum() == 0:
        return float("nan")
    if len(np.unique(scores)) == 1:
        return float(labels.mean())
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(labels.sum()))
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def risk_diagnostics(rows, hosts, risk_scores):
    labels = []
    score_map = {name: [] for name in risk_scores}
    for row in rows:
        if row["action"] == "host":
            continue
        labels.append(1 if row_gain(row, hosts) < 0.0 else 0)
        for name in risk_scores:
            score_map[name].append(P.risk_score(row, name))
    out = []
    for name, scores in score_map.items():
        out.append({
            "risk_score": name,
            "n_actions": len(labels),
            "harm_rate": float(np.mean(labels)) if labels else 0.0,
            "auroc": auroc(scores, labels),
            "auprc": auprc(scores, labels),
        })
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--split_group", choices=["image", "patient"], default="image")
    ap.add_argument("--max_cal_harm_rate", type=float, default=0.25)
    ap.add_argument("--max_cal_drop05_rate", type=float, default=0.10)
    ap.add_argument("--max_cal_drop20_rate", type=float, default=1.0)
    ap.add_argument("--max_cal_mean_harm", type=float, default=0.02)
    ap.add_argument("--mean_harm_scale", type=float, default=1.0)
    ap.add_argument("--bound_mode", choices=["hoeffding", "empirical_bernstein", "clopper_pearson"], default="hoeffding")
    ap.add_argument("--tail_constraint_mode", choices=["full", "bernoulli", "full_severe", "bernoulli_severe"], default="full")
    ap.add_argument("--crc_confidence", type=float, default=0.10)
    ap.add_argument("--harm_eps", type=float, default=0.0)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random_repeats", type=int, default=500)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    label = args.label or Path(args.input_csv).stem
    rows = load_action_rows(label, args.input_csv)
    cal_ids, test_ids, cal_groups, test_groups = split_ids(rows, args.cal_fraction, args.split_group)
    grouped_cal = P.group_rows(rows, cal_ids)
    grouped_test = P.group_rows(rows, test_ids)
    hosts_cal = P.host_by_id(grouped_cal)
    hosts_test = P.host_by_id(grouped_test)
    actions = P.all_actions(rows)

    model = fit_logistic_ridge([r for r in rows if r["id"] in cal_ids], hosts_cal)
    rows_q = inject_quality_risk(rows, model)
    grouped_cal_q = P.group_rows(rows_q, cal_ids)
    grouped_test_q = P.group_rows(rows_q, test_ids)

    class Args:
        pass
    cargs = Args()
    cargs.harm_eps = args.harm_eps
    cargs.beta_harm = args.beta_harm
    cargs.max_cal_harm_rate = args.max_cal_harm_rate
    cargs.max_cal_drop05_rate = args.max_cal_drop05_rate
    cargs.max_cal_drop20_rate = args.max_cal_drop20_rate
    cargs.max_cal_mean_harm = args.max_cal_mean_harm
    cargs.mean_harm_scale = args.mean_harm_scale
    cargs.bound_mode = args.bound_mode
    cargs.tail_constraint_mode = args.tail_constraint_mode
    cargs.crc_confidence = args.crc_confidence
    cargs.risk_score = "changed"
    # Decision baselines use the joint selection path; set the remaining
    # controller knobs to their primary-contract defaults so calibrate_portfolio
    # has every attribute it reads.
    cargs.selection_mode = "joint"
    cargs.require_positive_utility = False
    cargs.min_cal_utility = 0.0
    cargs.nested_select_fraction = 0.25
    cargs.nested_cal_fraction = 0.25
    cargs.cal_fraction = args.cal_fraction

    policies = []
    policies.append({"split": "test", **extended_summary("host", P.select_action(grouped_test, "host"), hosts_test, args.harm_eps)})
    fixed_action, fixed_selected = select_best_fixed(grouped_cal, hosts_cal, grouped_test, actions, args.harm_eps)
    fixed = extended_summary("best_fixed_cal_utility", fixed_selected, hosts_test, args.harm_eps)
    fixed["selected_action"] = fixed_action
    policies.append({"split": "test", **fixed})
    policies.append({"split": "test", **extended_summary("oracle", P.select_oracle(grouped_test), hosts_test, args.harm_eps)})

    selected_for_random = fixed_action
    best_crc_for_random = None
    for risk in ("changed", "geom", "change_plus_geom", "host_entropy", "host_uncertainty", "low_confidence", "margin_inverse"):
        best, summary = calibrate_and_apply(grouped_cal, hosts_cal, grouped_test, hosts_test, actions, cargs, risk)
        ext = extended_summary(summary["policy"], P.select_threshold(grouped_test, best["selected_action"], best["threshold"], risk)
                               if best["selected_action"] != "host" and best.get("threshold") is not None
                               else P.select_action(grouped_test, "host"), hosts_test, args.harm_eps)
        ext.update({k: summary.get(k) for k in ("selected_action", "threshold", "risk_score", "harm_rate_ucb", "forced_host")})
        policies.append({"split": "test", **ext})
        if best_crc_for_random is None or (ext["mean_gain"], -ext["mean_harm"]) > (best_crc_for_random[0]["mean_gain"], -best_crc_for_random[0]["mean_harm"]):
            best_crc_for_random = (ext, best)
            if ext.get("selected_action") and ext.get("selected_action") != "host":
                selected_for_random = ext["selected_action"]

    best, summary = calibrate_and_apply(grouped_cal_q, hosts_cal, grouped_test_q, hosts_test, actions, cargs, "quality_risk")
    ext = extended_summary("crc_quality_risk",
                           P.select_threshold(grouped_test_q, best["selected_action"], best["threshold"], "quality_risk")
                           if best["selected_action"] != "host" and best.get("threshold") is not None
                           else P.select_action(grouped_test_q, "host"),
                           hosts_test, args.harm_eps)
    ext.update({k: summary.get(k) for k in ("selected_action", "threshold", "risk_score", "harm_rate_ucb", "forced_host")})
    policies.append({"split": "test", **ext})

    accept_rate = 0.0
    if best_crc_for_random is not None:
        accept_rate = 1.0 - float(best_crc_for_random[0].get("reverted_rate", 1.0))
    policies.append({"split": "test", **random_rate_matched(grouped_test, hosts_test, selected_for_random, accept_rate, args.seed, args.random_repeats, args.harm_eps)})

    diag_scores = ("changed", "geom", "change_plus_geom", "host_entropy", "host_uncertainty", "low_confidence", "margin_inverse")
    diagnostics = risk_diagnostics([r for r in rows_q if r["id"] in test_ids], hosts_test, diag_scores + ("quality_risk",))

    payload = {
        "input_csv": args.input_csv,
        "label": label,
        "split_group": args.split_group,
        "cal_fraction": args.cal_fraction,
        "n_images": len(cal_ids) + len(test_ids),
        "n_cal_images": len(cal_ids),
        "n_test_images": len(test_ids),
        "n_cal_groups": len(cal_groups),
        "n_test_groups": len(test_groups),
        "actions": actions,
        "quality_features": list(BASE_FEATURES),
        "policy_summaries": policies,
        "risk_diagnostics": diagnostics,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out_prefix) + ".json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(str(out_prefix) + "_policies.csv", policies)
    write_csv(str(out_prefix) + "_risk_diagnostics.csv", diagnostics)
    print(json.dumps({
        "out_prefix": str(out_prefix),
        "n_test_images": len(test_ids),
        "n_test_groups": len(test_groups),
        "best_policy": max(policies, key=lambda r: (float(r.get("mean_gain", 0.0)), -float(r.get("mean_harm", 0.0))))["policy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
