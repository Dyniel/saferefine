#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    from scipy.stats import beta as scipy_beta
except Exception:  # pragma: no cover - scipy is available in the project env, but keep CLI robust.
    scipy_beta = None


RISK_SCORE_MODES = (
    "changed",
    "host_entropy",
    "host_uncertainty",
    "low_confidence",
    "margin_inverse",
    "quality_risk",
    "alpha",
    "force_abs",
    "stiffness",
    "damping",
    "force_times_change",
    "changed_plus_force",
    "geom",
    "change_plus_geom",
    "topology",
)


def parse_inputs(args):
    specs = []
    for item in args.input_csv:
        specs.extend([x.strip() for x in item.split(",") if x.strip()])
    specs.extend([x.strip() for x in args.inputs.split(",") if x.strip()])
    out = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = spec
            label = Path(path).stem
        if not label or not path:
            raise ValueError(f"Bad input spec: {spec!r}")
        out.append((label, Path(path)))
    if not out:
        raise ValueError("Provide at least one --input_csv or --inputs entry.")
    return out


def fnum(row, key, default=0.0):
    val = row.get(key, "")
    if val is None or val == "":
        return float(default)
    return float(val)


def action_name(label, row):
    method = str(row.get("method", "")).strip()
    alpha = fnum(row, "alpha", 0.0)
    if method == "host":
        return "host"
    if method == "host_morph":
        return "morph"
    if method == "graphmembrane":
        if abs(alpha) < 1e-12:
            return "host"
        return f"{label}:graphmembrane:a{alpha:.2f}"
    return f"{label}:{method}:a{alpha:.2f}"


def load_rows(input_specs):
    rows = []
    seen = set()
    for label, path in input_specs:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                image_id = str(raw.get("id", "?"))
                action = action_name(label, raw)
                key = (image_id, action)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "id": image_id,
                    "idx": int(float(raw.get("idx", -1))),
                    "source": label,
                    "method": str(raw.get("method", "")),
                    "action": action,
                    "alpha": fnum(raw, "alpha", 0.0),
                    "cup": fnum(raw, "cup", 0.0),
                    "disc": fnum(raw, "disc", 0.0),
                    "cpd": fnum(raw, "cpd", 0.0),
                    "changed": fnum(raw, "changed", 0.0),
                    "force_abs": fnum(raw, "force_abs", 0.0),
                    "stiffness": fnum(raw, "stiffness", 0.0),
                    "damping": fnum(raw, "damping", 0.0),
                    "disc_area_delta": fnum(raw, "disc_area_delta", 0.0),
                    "cup_area_delta": fnum(raw, "cup_area_delta", 0.0),
                    "cup_disc_ratio_delta": fnum(raw, "cup_disc_ratio_delta", 0.0),
                    "disc_centroid_shift": fnum(raw, "disc_centroid_shift", 0.0),
                    "cup_centroid_shift": fnum(raw, "cup_centroid_shift", 0.0),
                    "component_delta": fnum(raw, "component_delta", 0.0),
                    "disc_changed": fnum(raw, "disc_changed", 0.0),
                    "cup_changed": fnum(raw, "cup_changed", 0.0),
                    "geom_risk": fnum(raw, "geom_risk", 0.0),
                    "host_entropy": fnum(raw, "host_entropy", 0.0),
                    "host_confidence": fnum(raw, "host_confidence", 1.0),
                    "host_margin": fnum(raw, "host_margin", 1.0),
                    "quality_risk": fnum(raw, "quality_risk", 0.0),
                })
    return rows


def split_group_id(image_id, mode):
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
    raise ValueError(f"Unknown split_group={mode}")


def split_ids(rows, cal_fraction, split_group="image"):
    groups = []
    seen_group = set()
    group_to_ids = {}
    for row in rows:
        gid = split_group_id(row["id"], split_group)
        group_to_ids.setdefault(gid, set()).add(row["id"])
        if gid not in seen_group:
            groups.append(gid)
            seen_group.add(gid)
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


def split_ids_nested(rows, select_fraction, cal_fraction, split_group="image"):
    groups = []
    seen_group = set()
    group_to_ids = {}
    for row in rows:
        gid = split_group_id(row["id"], split_group)
        group_to_ids.setdefault(gid, set()).add(row["id"])
        if gid not in seen_group:
            groups.append(gid)
            seen_group.add(gid)
    n_groups = len(groups)
    if n_groups < 3:
        raise ValueError("Nested selection requires at least three split groups.")
    n_select = int(round(n_groups * select_fraction))
    n_cal = int(round(n_groups * cal_fraction))
    n_select = min(max(1, n_select), n_groups - 2)
    n_cal = min(max(1, n_cal), n_groups - n_select - 1)
    select_groups = set(groups[:n_select])
    cal_groups = set(groups[n_select : n_select + n_cal])
    test_groups = set(groups[n_select + n_cal :])
    select_ids, cal_ids, test_ids = set(), set(), set()
    for gid, ids in group_to_ids.items():
        if gid in select_groups:
            select_ids.update(ids)
        elif gid in cal_groups:
            cal_ids.update(ids)
        else:
            test_ids.update(ids)
    return select_ids, cal_ids, test_ids, select_groups, cal_groups, test_groups


def group_rows(rows, ids):
    out = {}
    for row in rows:
        if row["id"] in ids:
            out.setdefault(row["id"], {})[row["action"]] = row
    return out


def host_by_id(grouped):
    hosts = {}
    for image_id, actions in grouped.items():
        if "host" not in actions:
            raise ValueError(f"Missing host row for image id={image_id}")
        hosts[image_id] = actions["host"]
    return hosts


def all_actions(rows):
    return sorted({row["action"] for row in rows}, key=lambda x: (x != "host", x))


def risk_score(row, mode):
    changed = float(row["changed"])
    force_abs = float(row["force_abs"])
    stiffness = float(row["stiffness"])
    damping = float(row["damping"])
    geom_risk = float(row["geom_risk"])
    host_entropy = float(row.get("host_entropy", 0.0))
    host_confidence = float(row.get("host_confidence", 1.0))
    host_margin = float(row.get("host_margin", 1.0))
    quality_risk = float(row.get("quality_risk", 0.0))
    topology = float(row["component_delta"]) + float(row["cup_disc_ratio_delta"])
    if geom_risk <= 0.0:
        geom_risk = (
            changed
            + float(row["disc_area_delta"])
            + float(row["cup_area_delta"])
            + 0.25 * (float(row["disc_centroid_shift"]) + float(row["cup_centroid_shift"]))
            + 0.01 * float(row["component_delta"])
            + 0.10 * float(row["cup_disc_ratio_delta"])
        )
    if mode == "changed":
        return changed
    if mode == "host_entropy":
        return host_entropy
    if mode == "host_uncertainty":
        return host_entropy + max(0.0, 1.0 - host_confidence) + max(0.0, 1.0 - host_margin)
    if mode == "low_confidence":
        return max(0.0, 1.0 - host_confidence)
    if mode == "margin_inverse":
        return max(0.0, 1.0 - host_margin)
    if mode == "quality_risk":
        return quality_risk
    if mode == "alpha":
        return float(row["alpha"])
    if mode == "force_abs":
        return force_abs if force_abs > 0.0 else changed
    if mode == "stiffness":
        return stiffness if stiffness > 0.0 else changed
    if mode == "damping":
        return damping if damping > 0.0 else changed
    if mode == "force_times_change":
        return force_abs * changed if force_abs > 0.0 else changed
    if mode == "changed_plus_force":
        return changed + 0.01 * force_abs
    if mode == "geom":
        return geom_risk if geom_risk > 0.0 else changed
    if mode == "change_plus_geom":
        return changed + geom_risk
    if mode == "topology":
        return topology if topology > 0.0 else changed
    raise ValueError(f"Unknown risk score mode: {mode}")


def select_action(grouped, action):
    selected = []
    for _image_id, actions in grouped.items():
        selected.append(actions.get(action, actions["host"]))
    return selected


def select_threshold(grouped, action, threshold, risk_mode):
    selected = []
    for _image_id, actions in grouped.items():
        row = actions.get(action)
        if row is not None and risk_score(row, risk_mode) <= threshold + 1e-12:
            selected.append(row)
        else:
            selected.append(actions["host"])
    return selected


def select_oracle(grouped):
    selected = []
    for _image_id, actions in grouped.items():
        selected.append(max(actions.values(), key=lambda row: row["cpd"]))
    return selected


def summarize(name, selected, hosts, harm_eps):
    gains = np.asarray([row["cpd"] - hosts[row["id"]]["cpd"] for row in selected], dtype=np.float64)
    harms = np.maximum(0.0, -gains)
    cpds = np.asarray([row["cpd"] for row in selected], dtype=np.float64)
    cups = np.asarray([row["cup"] for row in selected], dtype=np.float64)
    discs = np.asarray([row["disc"] for row in selected], dtype=np.float64)
    changed = np.asarray([row["changed"] for row in selected], dtype=np.float64)
    action_counts = {}
    for row in selected:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    n = int(len(selected))
    reverted = action_counts.get("host", 0)
    if n:
        k = max(1, int(math.ceil(0.10 * n)))
        cvar_harm_10 = float(np.sort(harms)[-k:].mean())
    else:
        cvar_harm_10 = 0.0
    return {
        "policy": name,
        "n": n,
        "mean_cpd": float(cpds.mean()) if n else 0.0,
        "mean_cup": float(cups.mean()) if n else 0.0,
        "mean_disc": float(discs.mean()) if n else 0.0,
        "mean_gain": float(gains.mean()) if n else 0.0,
        "median_gain": float(np.median(gains)) if n else 0.0,
        "mean_harm": float(harms.mean()) if n else 0.0,
        "var_harm": float(harms.var(ddof=1)) if n > 1 else 0.0,
        "worst_drop": float(gains.min()) if n else 0.0,
        "improved_rate": float((gains > harm_eps).mean()) if n else 0.0,
        "harmed_rate": float((gains < -harm_eps).mean()) if n else 0.0,
        "drop_gt_0.01": float((gains < -0.01).mean()) if n else 0.0,
        "drop_gt_0.05": float((gains < -0.05).mean()) if n else 0.0,
        "drop_gt_0.10": float((gains < -0.10).mean()) if n else 0.0,
        "drop_gt_0.20": float((gains < -0.20).mean()) if n else 0.0,
        "cvar_harm_10": cvar_harm_10,
        "reverted_rate": float(reverted / max(1, n)),
        "mean_changed": float(changed.mean()) if n else 0.0,
        "action_counts": action_counts,
    }


def candidate_thresholds(grouped, action, risk_mode):
    vals = {0.0}
    for actions in grouped.values():
        row = actions.get(action)
        if row is not None:
            vals.add(float(risk_score(row, risk_mode)))
    return sorted(vals)


def risk_constraint_count(args):
    count = 2
    if args.tail_constraint_mode in {"bernoulli_severe", "full_severe"}:
        count += 1
    if args.tail_constraint_mode in {"full", "full_severe"}:
        count += 1
    return count


def hoeffding_upper(mean, n, confidence, n_candidates, scale=1.0):
    if n <= 0:
        return scale
    delta = max(float(confidence), 1e-12) / max(1, int(n_candidates))
    radius = scale * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return float(min(scale, mean + radius))


def empirical_bernstein_upper(mean, variance, n, confidence, n_candidates, scale=1.0):
    if n <= 0:
        return scale
    delta = max(float(confidence), 1e-12) / max(1, int(n_candidates))
    log_term = math.log(3.0 / delta)
    radius = math.sqrt(max(0.0, 2.0 * variance * log_term / n)) + (3.0 * scale * log_term / n)
    return float(min(scale, mean + radius))


def clopper_pearson_upper(rate, n, confidence, n_candidates):
    if n <= 0:
        return 1.0
    delta = max(float(confidence), 1e-12) / max(1, int(n_candidates))
    k = int(round(float(rate) * n))
    k = min(max(k, 0), n)
    if k >= n:
        return 1.0
    if scipy_beta is None:
        return hoeffding_upper(rate, n, confidence, n_candidates)
    return float(scipy_beta.ppf(1.0 - delta, k + 1, n - k))


def risk_upper(summary, risk_name, args, n_candidates):
    n = summary["n"]
    if risk_name == "harm_rate":
        mean = summary["harmed_rate"]
        variance = mean * (1.0 - mean)
        scale = 1.0
        if args.bound_mode == "clopper_pearson":
            return clopper_pearson_upper(mean, n, args.crc_confidence, n_candidates)
    elif risk_name == "drop05_rate":
        mean = summary["drop_gt_0.05"]
        variance = mean * (1.0 - mean)
        scale = 1.0
        if args.bound_mode == "clopper_pearson":
            return clopper_pearson_upper(mean, n, args.crc_confidence, n_candidates)
    elif risk_name == "drop20_rate":
        mean = summary["drop_gt_0.20"]
        variance = mean * (1.0 - mean)
        scale = 1.0
        if args.bound_mode == "clopper_pearson":
            return clopper_pearson_upper(mean, n, args.crc_confidence, n_candidates)
    elif risk_name == "mean_harm":
        mean = summary["mean_harm"]
        variance = summary.get("var_harm", 0.0)
        scale = args.mean_harm_scale
    else:
        raise ValueError(f"Unknown risk name: {risk_name}")

    if args.bound_mode == "empirical_bernstein":
        return empirical_bernstein_upper(mean, variance, n, args.crc_confidence, n_candidates, scale=scale)
    return hoeffding_upper(mean, n, args.crc_confidence, n_candidates, scale=scale)


def annotate_constraint_failure(summary, args):
    checks = [
        ("harm_rate", summary.get("harm_rate_ucb", 1.0), args.max_cal_harm_rate),
        ("drop05_rate", summary.get("drop05_rate_ucb", 1.0), args.max_cal_drop05_rate),
    ]
    if args.tail_constraint_mode in {"bernoulli_severe", "full_severe"}:
        checks.append(("drop20_rate", summary.get("drop20_rate_ucb", 1.0), args.max_cal_drop20_rate))
    if args.tail_constraint_mode in {"full", "full_severe"}:
        checks.append(("mean_harm", summary.get("mean_harm_ucb", args.mean_harm_scale), args.max_cal_mean_harm))
    failed = [name for name, val, budget in checks if float(val) > float(budget)]
    summary["failed_constraints"] = ",".join(failed)
    summary["first_failed_constraint"] = failed[0] if failed else ""
    return summary


def positive_utility_or_host(best, candidates, min_utility):
    if best.get("selected_action") == "host":
        return best
    if float(best.get("utility", 0.0)) > float(min_utility):
        return best
    host = next(c for c in candidates if c.get("selected_action") == "host")
    host = dict(host)
    host["forced_host"] = True
    host["forced_host_reason"] = "nonpositive_calibration_utility"
    host["best_nonhost_utility"] = best.get("utility", 0.0)
    return host


def calibrate_portfolio(grouped, hosts, actions, args, crc=False, fixed_action=None):
    candidates = []
    if fixed_action is not None and fixed_action != "host":
        non_host = [fixed_action]
    elif fixed_action == "host":
        non_host = []
    else:
        non_host = [a for a in actions if a != "host"]
    n_thresholds = sum(len(candidate_thresholds(grouped, a, args.risk_score)) for a in non_host)
    n_thresholds = max(1, n_thresholds)
    n_ucb_tests = n_thresholds * risk_constraint_count(args)

    host_summary = summarize("host", select_action(grouped, "host"), hosts, args.harm_eps)
    host_summary.update({
        "selected_action": "host",
        "threshold": None,
        "risk_score": args.risk_score,
        "utility": 0.0,
        "harm_rate_ucb": 0.0,
        "drop05_rate_ucb": 0.0,
        "drop20_rate_ucb": 0.0,
        "mean_harm_ucb": 0.0,
        "failed_constraints": "",
        "first_failed_constraint": "",
        "forced_host": True,
    })
    candidates.append(host_summary)

    for action in non_host:
        for threshold in candidate_thresholds(grouped, action, args.risk_score):
            selected = select_threshold(grouped, action, threshold, args.risk_score)
            summary = summarize(f"{action}@{threshold:.6g}", selected, hosts, args.harm_eps)
            summary["selected_action"] = action
            summary["threshold"] = float(threshold)
            summary["risk_score"] = args.risk_score
            summary["utility"] = float(summary["mean_gain"] - args.beta_harm * summary["mean_harm"])
            summary["forced_host"] = False
            if crc:
                summary["harm_rate_ucb"] = risk_upper(summary, "harm_rate", args, n_ucb_tests)
                summary["drop05_rate_ucb"] = risk_upper(summary, "drop05_rate", args, n_ucb_tests)
                summary["drop20_rate_ucb"] = risk_upper(summary, "drop20_rate", args, n_ucb_tests)
                summary["mean_harm_ucb"] = risk_upper(summary, "mean_harm", args, n_ucb_tests)
                annotate_constraint_failure(summary, args)
            candidates.append(summary)

    if crc:
        feasible = [
            c for c in candidates
            if c["selected_action"] == "host"
            or (
                c.get("harm_rate_ucb", 1.0) <= args.max_cal_harm_rate
                and c.get("drop05_rate_ucb", 1.0) <= args.max_cal_drop05_rate
                and (
                    args.tail_constraint_mode not in {"bernoulli_severe", "full_severe"}
                    or c.get("drop20_rate_ucb", 1.0) <= args.max_cal_drop20_rate
                )
                and (
                    args.tail_constraint_mode in {"bernoulli", "bernoulli_severe"}
                    or c.get("mean_harm_ucb", args.mean_harm_scale) <= args.max_cal_mean_harm
                )
            )
        ]
    else:
        feasible = [
            c for c in candidates
            if c["harmed_rate"] <= args.max_cal_harm_rate
            and c["drop_gt_0.05"] <= args.max_cal_drop05_rate
            and (
                args.tail_constraint_mode not in {"bernoulli_severe", "full_severe"}
                or c["drop_gt_0.20"] <= args.max_cal_drop20_rate
            )
            and (
                args.tail_constraint_mode in {"bernoulli", "bernoulli_severe"}
                or c["mean_harm"] <= args.max_cal_mean_harm
            )
        ]

    best = max(feasible, key=lambda c: (c["utility"], c["mean_gain"], -c["mean_harm"]))
    best = dict(best)
    if args.require_positive_utility:
        best = positive_utility_or_host(best, candidates, args.min_cal_utility)
    best["feasible_candidates"] = int(len(feasible))
    best["tested_candidates"] = int(len(candidates))
    best["tested_thresholds_for_ucb"] = int(n_thresholds)
    best["tested_risk_bounds_for_ucb"] = int(n_ucb_tests)
    best["crc"] = bool(crc)
    return best, candidates


def nested_calibrate_portfolio(grouped_select, hosts_select, grouped_cal, hosts_cal, actions, args, crc=False):
    selection_args = argparse.Namespace(**vars(args))
    selection_args.require_positive_utility = True
    select_best, select_candidates = calibrate_portfolio(
        grouped_select, hosts_select, actions, selection_args, crc=False
    )
    selected_action = select_best.get("selected_action", "host")
    cal_best, cal_candidates = calibrate_portfolio(
        grouped_cal, hosts_cal, actions, args, crc=crc, fixed_action=selected_action
    )
    cal_best = dict(cal_best)
    cal_best["nested_selected_action"] = selected_action
    cal_best["nested_selection_utility"] = select_best.get("utility", 0.0)
    cal_best["nested_selection_gain"] = select_best.get("mean_gain", 0.0)
    cal_best["nested_selection_harm"] = select_best.get("mean_harm", 0.0)
    cal_best["nested_selection_candidates"] = select_best.get("tested_candidates", len(select_candidates))
    cal_best["nested_calibration_candidates"] = cal_best.get("tested_candidates", len(cal_candidates))
    cal_best["selection_mode"] = "nested"
    return cal_best, select_candidates, cal_candidates


def best_nonhost_diagnostic(candidates, args):
    nonhost = [c for c in candidates if c.get("selected_action") != "host"]
    if not nonhost:
        return {}
    best = max(nonhost, key=lambda c: (c.get("utility", 0.0), c.get("mean_gain", 0.0), -c.get("mean_harm", 0.0)))
    out = dict(best)
    if "failed_constraints" not in out:
        annotate_constraint_failure(out, args)
    return out


def apply_calibrated(grouped, hosts, calibration, args, policy_name):
    action = calibration["selected_action"]
    threshold = calibration.get("threshold")
    if action == "host" or threshold is None:
        selected = select_action(grouped, "host")
    else:
        selected = select_threshold(grouped, action, float(threshold), args.risk_score)
    summary = summarize(policy_name, selected, hosts, args.harm_eps)
    summary["selected_action"] = action
    summary["threshold"] = threshold
    summary["risk_score"] = args.risk_score
    summary["cal_utility"] = calibration.get("utility", 0.0)
    summary["feasible_candidates"] = calibration.get("feasible_candidates", 0)
    summary["tested_candidates"] = calibration.get("tested_candidates", 0)
    summary["tested_thresholds_for_ucb"] = calibration.get("tested_thresholds_for_ucb", "")
    summary["nested_selected_action"] = calibration.get("nested_selected_action", "")
    summary["nested_selection_utility"] = calibration.get("nested_selection_utility", "")
    summary["nested_selection_candidates"] = calibration.get("nested_selection_candidates", "")
    summary["nested_calibration_candidates"] = calibration.get("nested_calibration_candidates", "")
    summary["harm_rate_ucb"] = calibration.get("harm_rate_ucb")
    summary["drop05_rate_ucb"] = calibration.get("drop05_rate_ucb")
    summary["drop20_rate_ucb"] = calibration.get("drop20_rate_ucb")
    summary["mean_harm_ucb"] = calibration.get("mean_harm_ucb")
    summary["failed_constraints"] = calibration.get("failed_constraints", "")
    summary["first_failed_constraint"] = calibration.get("first_failed_constraint", "")
    summary["forced_host"] = calibration.get("forced_host", False)
    summary["tested_risk_bounds_for_ucb"] = calibration.get("tested_risk_bounds_for_ucb", "")
    return summary


def write_csv(path, summaries):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "policy", "n", "mean_cpd", "mean_gain", "mean_harm",
        "worst_drop", "improved_rate", "harmed_rate", "reverted_rate",
        "drop_gt_0.01", "drop_gt_0.05", "drop_gt_0.10", "drop_gt_0.20", "cvar_harm_10",
        "mean_changed", "selected_action", "threshold", "risk_score",
        "harm_rate_ucb", "drop05_rate_ucb", "drop20_rate_ucb", "mean_harm_ucb",
        "failed_constraints", "first_failed_constraint", "tested_candidates",
        "tested_thresholds_for_ucb", "tested_risk_bounds_for_ucb", "nested_selected_action",
        "nested_selection_utility", "nested_selection_candidates",
        "nested_calibration_candidates", "action_counts",
    ]
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            out = {k: row.get(k, "") for k in fields}
            out["action_counts"] = json.dumps(row.get("action_counts", {}), sort_keys=True)
            writer.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", action="append", default=[], help="CSV path or label=CSV path. Can be repeated.")
    ap.add_argument("--inputs", default="", help="Comma-separated CSV path or label=CSV path entries.")
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--nested_select_fraction", type=float, default=0.25)
    ap.add_argument("--nested_cal_fraction", type=float, default=0.25)
    ap.add_argument("--split_group", choices=("image", "patient"), default="image")
    ap.add_argument("--selection_mode", choices=("joint", "nested"), default="joint")
    ap.add_argument("--harm_eps", type=float, default=0.0)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--max_cal_harm_rate", type=float, default=0.25)
    ap.add_argument("--max_cal_drop05_rate", type=float, default=0.10)
    ap.add_argument("--max_cal_drop20_rate", type=float, default=1.0)
    ap.add_argument("--max_cal_mean_harm", type=float, default=0.02)
    ap.add_argument("--mean_harm_scale", type=float, default=1.0)
    ap.add_argument("--crc_confidence", type=float, default=0.10)
    ap.add_argument("--bound_mode", choices=("hoeffding", "empirical_bernstein", "clopper_pearson"), default="hoeffding")
    ap.add_argument("--tail_constraint_mode", choices=("full", "bernoulli", "full_severe", "bernoulli_severe"), default="full")
    ap.add_argument("--require_positive_utility", action="store_true")
    ap.add_argument("--min_cal_utility", type=float, default=0.0)
    ap.add_argument("--risk_score", choices=RISK_SCORE_MODES, default="changed")
    ap.add_argument("--out_summary", default="")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    input_specs = parse_inputs(args)
    rows = load_rows(input_specs)
    actions = all_actions(rows)
    if args.selection_mode == "nested":
        select_ids, cal_ids, test_ids, select_groups, cal_groups, test_groups = split_ids_nested(
            rows, args.nested_select_fraction, args.nested_cal_fraction, args.split_group
        )
        grouped_select = group_rows(rows, select_ids)
        grouped_cal = group_rows(rows, cal_ids)
        grouped_test = group_rows(rows, test_ids)
        hosts_select = host_by_id(grouped_select)
        hosts_cal = host_by_id(grouped_cal)
        hosts_test = host_by_id(grouped_test)
        cal_best, cal_select_candidates, cal_candidates = nested_calibrate_portfolio(
            grouped_select, hosts_select, grouped_cal, hosts_cal, actions, args, crc=False
        )
        crc_best, crc_select_candidates, crc_candidates = nested_calibrate_portfolio(
            grouped_select, hosts_select, grouped_cal, hosts_cal, actions, args, crc=True
        )
        split_sets = [("select", grouped_select, hosts_select), ("cal", grouped_cal, hosts_cal), ("test", grouped_test, hosts_test)]
    else:
        cal_ids, test_ids, cal_groups, test_groups = split_ids(rows, args.cal_fraction, args.split_group)
        select_ids, select_groups = set(), set()
        grouped_cal = group_rows(rows, cal_ids)
        grouped_test = group_rows(rows, test_ids)
        hosts_cal = host_by_id(grouped_cal)
        hosts_test = host_by_id(grouped_test)
        cal_best, cal_candidates = calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=False)
        crc_best, crc_candidates = calibrate_portfolio(grouped_cal, hosts_cal, actions, args, crc=True)
        cal_select_candidates = []
        crc_select_candidates = []
        split_sets = [("cal", grouped_cal, hosts_cal), ("test", grouped_test, hosts_test)]

    summaries = []
    for split, grouped, hosts in split_sets:
        summaries.append({"split": split, **summarize("host", select_action(grouped, "host"), hosts, args.harm_eps)})
        for action in actions:
            if action == "host":
                continue
            summaries.append({"split": split, **summarize(f"fixed:{action}", select_action(grouped, action), hosts, args.harm_eps)})
        summaries.append({"split": split, **summarize("oracle", select_oracle(grouped), hosts, args.harm_eps)})
        summaries.append({"split": split, **apply_calibrated(grouped, hosts, cal_best, args, "calibrated_portfolio")})
        summaries.append({"split": split, **apply_calibrated(grouped, hosts, crc_best, args, "crc_portfolio")})

    print(f"inputs={','.join(f'{label}={path}' for label, path in input_specs)}", flush=True)
    print(
        f"images={len(select_ids) + len(cal_ids) + len(test_ids)} select={len(select_ids)} cal={len(cal_ids)} test={len(test_ids)} "
        f"select_groups={len(select_groups)} cal_groups={len(cal_groups)} test_groups={len(test_groups)} "
        f"actions={len(actions)} risk_score={args.risk_score} selection_mode={args.selection_mode}",
        flush=True,
    )
    print(
        "\nsplit\tpolicy\tn\tmean_cpd\tmean_gain\tmean_harm\tworst_drop\timproved_rate\t"
        "harmed_rate\treverted_rate\tselected_action\tthreshold\tharm_rate_ucb",
        flush=True,
    )
    for row in summaries:
        print(
            f"{row['split']}\t{row['policy']}\t{row['n']}\t{row['mean_cpd']:.6f}\t{row['mean_gain']:+.6f}\t"
            f"{row['mean_harm']:.6f}\t{row['worst_drop']:+.6f}\t{row['improved_rate']:.4f}\t"
            f"{row['harmed_rate']:.4f}\t{row['reverted_rate']:.4f}\t{row.get('selected_action', '')}\t"
            f"{row.get('threshold', '')}\t{row.get('harm_rate_ucb', '')}",
            flush=True,
        )

    payload = {
        "inputs": [{"label": label, "path": str(path)} for label, path in input_specs],
        "risk_score": args.risk_score,
        "cal_fraction": args.cal_fraction,
        "nested_select_fraction": args.nested_select_fraction,
        "nested_cal_fraction": args.nested_cal_fraction,
        "selection_mode": args.selection_mode,
        "harm_eps": args.harm_eps,
        "beta_harm": args.beta_harm,
        "max_cal_harm_rate": args.max_cal_harm_rate,
        "max_cal_drop05_rate": args.max_cal_drop05_rate,
        "max_cal_drop20_rate": args.max_cal_drop20_rate,
        "max_cal_mean_harm": args.max_cal_mean_harm,
        "mean_harm_scale": args.mean_harm_scale,
        "crc_confidence": args.crc_confidence,
        "bound_mode": args.bound_mode,
        "tail_constraint_mode": args.tail_constraint_mode,
        "require_positive_utility": args.require_positive_utility,
        "min_cal_utility": args.min_cal_utility,
        "n_images": len(select_ids) + len(cal_ids) + len(test_ids),
        "n_select": len(select_ids),
        "n_cal": len(cal_ids),
        "n_test": len(test_ids),
        "split_group": args.split_group,
        "n_select_groups": len(select_groups),
        "n_cal_groups": len(cal_groups),
        "n_test_groups": len(test_groups),
        "actions": actions,
        "calibrated_best": cal_best,
        "crc_best": crc_best,
        "best_nonhost_crc": best_nonhost_diagnostic(crc_candidates, args),
        "calibrated_selection_candidates": cal_select_candidates,
        "crc_selection_candidates": crc_select_candidates,
        "calibrated_candidates": cal_candidates,
        "crc_candidates": crc_candidates,
        "summaries": summaries,
    }
    if args.out_summary:
        p = Path(args.out_summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(args.out_csv, summaries)


if __name__ == "__main__":
    main()
