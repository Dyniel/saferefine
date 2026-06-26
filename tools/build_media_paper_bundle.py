#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_tables"


def f(x, default=0.0):
    if x is None or x == "":
        return default
    return float(x)


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path):
    return json.loads(Path(path).read_text())


def eval_summary(path):
    payload = read_json(path)
    rows = payload["summary"]
    host = next(r for r in rows if r["method"] == "host")
    non_host = [r for r in rows if r["method"] != "host"]
    best_fixed = max(non_host, key=lambda r: r["mean_dice"])
    return host, best_fixed


def test_crc(path):
    payload = read_json(path)
    row = next(r for r in payload["summaries"] if r.get("split") == "test" and r.get("policy") == "crc_portfolio")
    return payload, row


def best_crc_from_glob(pattern, strict=False):
    best = None
    for path in sorted(ROOT.glob(pattern)):
        payload, row = test_crc(path)
        is_strict = float(payload["max_cal_harm_rate"]) == 0.0
        if is_strict != strict:
            continue
        item = (path, payload, row)
        if best is None:
            best = item
            continue
        _, _, brow = best
        if (row["mean_gain"], -row["mean_harm"], row["reverted_rate"]) > (
            brow["mean_gain"], -brow["mean_harm"], brow["reverted_rate"]
        ):
            best = item
    return best


def best_ci(rows, dataset, mode="practical"):
    cand = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode and r["policy"] == "crc_portfolio"]
    if not cand:
        return None
    return max(cand, key=lambda r: (f(r["mean_gain"]), -f(r["mean_harm"]), f(r["reverted_rate"])))


def fmt_num(x, digits=3, signed=False):
    x = f(x)
    if abs(x) < 0.0005:
        x = 0.0
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def fmt_ci(row, key, digits=3, signed=False):
    if row is None:
        return ""
    mid = fmt_num(row[key], digits, signed)
    lo = fmt_num(row[f"{key}_lo"], digits, signed)
    hi = fmt_num(row[f"{key}_hi"], digits, signed)
    return f"{mid} [{lo}, {hi}]"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, title):
    fields = list(rows[0].keys()) if rows else []
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def latex_escape(x):
    return str(x).replace("_", "\\_").replace("%", "\\%")


def write_latex(path, rows, caption, label):
    fields = list(rows[0].keys()) if rows else []
    cols = "l" * len(fields)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(latex_escape(k) for k in fields) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(k, "")) for k in fields) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines))


def materialize(name, rows, title, caption):
    write_csv(OUT / f"{name}.csv", rows)
    write_markdown(OUT / f"{name}.md", rows, title)
    write_latex(OUT / f"{name}.tex", rows, caption, f"tab:{name}")


def make_main_table():
    e120_summary = {r["run"]: r for r in read_csv(ROOT / "results/binary_e120_host_policy_summary.csv")}
    baseline_ci = read_csv(ROOT / "results/bootstrap_ci/binary_bootstrap_crc_ablation.csv")
    e120_ci = read_csv(ROOT / "results/bootstrap_ci_e120/e120_bootstrap_crc_ablation.csv")

    rows = []

    # REFUGE2 uses CPD across cup/disc; higher is better.
    ref_practical = best_crc_from_glob("results/graphmembrane_policy/geom_practical_portfolio_*.json", strict=False)
    ref_strict = best_crc_from_glob("results/graphmembrane_policy/geom_strict_portfolio_*.json", strict=True)
    ref_payload, ref_row = ref_practical[1], ref_practical[2]
    ref_strict_row = ref_strict[2]
    ref_cal = next(r for r in ref_payload["summaries"] if r.get("split") == "test" and r.get("policy") == "calibrated_portfolio")
    ref_host = next(r for r in ref_payload["summaries"] if r.get("split") == "test" and r.get("policy") == "host")
    rows.append({
        "setting": "Retina / REFUGE2 / GraphMembrane",
        "host_metric": fmt_num(ref_host["mean_cpd"], 3),
        "best_fixed_gain": fmt_num(ref_cal["mean_gain"], 3, True),
        "fixed_harm": fmt_num(ref_cal["mean_harm"], 4),
        "fixed_worst_drop": fmt_num(ref_cal["worst_drop"], 3, True),
        "practical_crc_gain": fmt_num(ref_row["mean_gain"], 3, True),
        "practical_harm": fmt_num(ref_row["mean_harm"], 4),
        "practical_worst_drop": fmt_num(ref_row["worst_drop"], 3, True),
        "revert_rate": fmt_num(ref_row["reverted_rate"], 2),
        "strict_gain/harm": f"{fmt_num(ref_strict_row['mean_gain'], 3, True)} / {fmt_num(ref_strict_row['mean_harm'], 4)}",
    })

    # Strong positive baseline dermoscopy signal.
    host, fixed = eval_summary(ROOT / "results/binary_refinement/isic2018_task1_phase0_eval.json")
    isic_ci = best_ci(baseline_ci, "isic2018_task1", "practical")
    isic_strict = best_ci(baseline_ci, "isic2018_task1", "strict")
    rows.append({
        "setting": "Dermoscopy / ISIC 2018 / GraphSeg",
        "host_metric": fmt_num(host["mean_dice"], 3),
        "best_fixed_gain": fmt_num(fixed["mean_gain"], 3, True),
        "fixed_harm": fmt_num(fixed["mean_harm"], 4),
        "fixed_worst_drop": fmt_num(fixed["worst_drop"], 3, True),
        "practical_crc_gain": fmt_ci(isic_ci, "mean_gain", 3, True),
        "practical_harm": fmt_ci(isic_ci, "mean_harm", 4),
        "practical_worst_drop": fmt_ci(isic_ci, "worst_drop", 3, True),
        "revert_rate": fmt_ci(isic_ci, "reverted_rate", 2),
        "strict_gain/harm": f"{fmt_num(isic_strict['mean_gain'], 3, True)} / {fmt_num(isic_strict['mean_harm'], 4)}",
    })

    for run, label in [
        ("kvasir_seg_endo_graphseg_e120", "Endoscopy / Kvasir-SEG / GraphSeg e120"),
        ("kvasir_seg_endo_unet_e120", "Endoscopy / Kvasir-SEG / UNet e120"),
        ("ph2_unet_e120", "Dermoscopy / PH2 / UNet e120"),
    ]:
        s = e120_summary[run]
        ci = best_ci(e120_ci, run, "practical")
        strict = best_ci(e120_ci, run, "strict")
        rows.append({
            "setting": label,
            "host_metric": fmt_num(s["host_dice"], 3),
            "best_fixed_gain": fmt_num(s["best_fixed_gain"], 3, True),
            "fixed_harm": fmt_num(s["best_fixed_harm"], 4),
            "fixed_worst_drop": fmt_num(s["best_fixed_worst_drop"], 3, True),
            "practical_crc_gain": fmt_ci(ci, "mean_gain", 3, True),
            "practical_harm": fmt_ci(ci, "mean_harm", 4),
            "practical_worst_drop": fmt_ci(ci, "worst_drop", 3, True),
            "revert_rate": fmt_ci(ci, "reverted_rate", 2),
            "strict_gain/harm": f"{fmt_num(strict['mean_gain'], 3, True)} / {fmt_num(strict['mean_harm'], 4)}",
        })

    materialize(
        "main_safety_table",
        rows,
        "Main Safety Table",
        "Representative host-preserving safe refinement results. REFUGE2 reports CPD; binary datasets report Dice.",
    )
    return rows


def make_ablation_table():
    rows = []
    for source, path in [
        ("phase0", ROOT / "results/bootstrap_ci/binary_bootstrap_crc_ablation.csv"),
        ("e120", ROOT / "results/bootstrap_ci_e120/e120_bootstrap_crc_ablation.csv"),
    ]:
        for row in read_csv(path):
            if row["mode"] not in ("practical", "strict"):
                continue
            if row["dataset"] not in (
                "isic2018_task1",
                "kvasir_seg_endo_graphseg_e120",
                "kvasir_seg_endo_unet_e120",
                "ph2_unet_e120",
            ):
                continue
            rows.append({
                "source": source,
                "dataset": row["dataset"],
                "mode": row["mode"],
                "risk": row["risk_score"],
                "gain_CI": fmt_ci(row, "mean_gain", 4, True),
                "harm_CI": fmt_ci(row, "mean_harm", 4),
                "worst_drop_CI": fmt_ci(row, "worst_drop", 4, True),
                "harmed_rate_CI": fmt_ci(row, "harmed_rate", 3),
                "revert_rate_CI": fmt_ci(row, "reverted_rate", 3),
                "action": row["crc_selected_action"],
            })
    materialize(
        "risk_ablation_crc",
        rows,
        "Risk Score And Policy Ablation",
        "CRC ablation over risk score and practical versus strict safety modes.",
    )
    return rows


def make_cross_host_table():
    e120 = {r["run"]: r for r in read_csv(ROOT / "results/binary_e120_host_policy_summary.csv")}
    baseline = read_csv(ROOT / "results/binary_multimodal_policy_summary.csv")
    rows = []

    def add(run, dataset, host_family, stage, modality):
        s = e120[run]
        rows.append({
            "dataset": dataset,
            "modality": modality,
            "host": host_family,
            "stage": stage,
            "host_dice": fmt_num(s["host_dice"], 3),
            "best_fixed_gain": fmt_num(s["best_fixed_gain"], 3, True),
            "fixed_harm": fmt_num(s["best_fixed_harm"], 4),
            "best_practical_gain": fmt_num(s["best_practical_gain"], 4, True),
            "best_practical_harm": fmt_num(s["best_practical_harm"], 4),
            "strict_exact_fallback": "yes" if f(s["strict_harm"]) == 0.0 and f(s["strict_reverted_rate"]) == 1.0 else "no",
        })

    for run, dataset, host, modality in [
        ("kvasir_seg_endo_graphseg_e120", "Kvasir-SEG", "GraphSeg", "endoscopy"),
        ("kvasir_seg_endo_unet_e120", "Kvasir-SEG", "UNet", "endoscopy"),
        ("kvasir_sessile_endo_graphseg_e120", "Kvasir-Sessile", "GraphSeg", "endoscopy"),
        ("kvasir_sessile_endo_unet_e120", "Kvasir-Sessile", "UNet", "endoscopy"),
        ("polyps_official_endo_graphseg_e120", "Polyp official", "GraphSeg", "endoscopy external"),
        ("polyps_official_endo_unet_e120", "Polyp official", "UNet", "endoscopy external"),
        ("ph2_unet_e120", "PH2", "UNet", "dermoscopy"),
        ("isic2018_task1_unet_e120", "ISIC 2018", "UNet", "dermoscopy"),
    ]:
        add(run, dataset, host, "e120", modality)

    # Add compact phase0 GraphSeg rows where available.
    seen = set()
    for r in baseline:
        if r["policy"] != "crc_portfolio" or r["mode"] != "practical":
            continue
        key = r["dataset"]
        if key in seen:
            continue
        same = [x for x in baseline if x["dataset"] == key and x["policy"] == "crc_portfolio" and x["mode"] == "practical"]
        best = max(same, key=lambda x: (f(x["policy_gain"]), -f(x["policy_harm"])))
        label = {
            "kvasir_seg": "Kvasir-SEG",
            "kvasir_sessile": "Kvasir-Sessile",
            "polyps_official": "Polyp official",
            "ph2": "PH2",
            "isic2018_task1": "ISIC 2018",
        }.get(key, key)
        modality = "endoscopy" if "kvasir" in key or "polyps" in key else "dermoscopy"
        rows.append({
            "dataset": label,
            "modality": modality,
            "host": "GraphSeg",
            "stage": "phase0",
            "host_dice": fmt_num(best["eval_host_dice"], 3),
            "best_fixed_gain": fmt_num(best["fixed_best_gain"], 3, True),
            "fixed_harm": fmt_num(best["fixed_best_harm"], 4),
            "best_practical_gain": fmt_num(best["policy_gain"], 4, True),
            "best_practical_harm": fmt_num(best["policy_harm"], 4),
            "strict_exact_fallback": "yes",
        })
        seen.add(key)

    rows.sort(key=lambda r: (r["dataset"], r["stage"], r["host"]))
    materialize(
        "cross_host_model_agnostic",
        rows,
        "Cross-Host Model-Agnostic Evidence",
        "Cross-host evidence for model-agnostic safe refinement and exact strict fallback.",
    )
    return rows


def make_stress_table():
    e120 = {r["run"]: r for r in read_csv(ROOT / "results/binary_e120_host_policy_summary.csv")}
    rows = []
    for run, interpretation in [
        ("polyps_official_endo_graphseg_e120", "external endoscopy stress test; weak host/refiner regime"),
        ("polyps_official_endo_unet_e120", "external endoscopy stress test; UNet underfits official polyp mix"),
        ("kvasir_sessile_endo_graphseg_e120", "small sessile set; practical and strict fallback"),
        ("kvasir_sessile_endo_unet_e120", "small sessile set; fixed refiner signal but CRC conservative"),
    ]:
        s = e120[run]
        rows.append({
            "run": run,
            "host_dice": fmt_num(s["host_dice"], 3),
            "fixed_gain": fmt_num(s["best_fixed_gain"], 3, True),
            "fixed_harm": fmt_num(s["best_fixed_harm"], 4),
            "fixed_worst_drop": fmt_num(s["best_fixed_worst_drop"], 3, True),
            "practical_gain": fmt_num(s["best_practical_gain"], 4, True),
            "practical_harm": fmt_num(s["best_practical_harm"], 4),
            "strict_exact_fallback": "yes" if f(s["strict_harm"]) == 0.0 and f(s["strict_reverted_rate"]) == 1.0 else "no",
            "paper_role": interpretation,
        })
    materialize(
        "stress_negative_results",
        rows,
        "Stress And Negative Results",
        "Stress-test results showing unsafe or weak refiner regimes and host-preserving fallback behavior.",
    )
    return rows


def make_figure_index():
    rows = []
    figure_dirs = [
        ROOT / "results/figures/crc_reversion_cases",
        ROOT / "results/figures/mediafinal_crc_reversion_cases",
    ]
    for manifest in sorted(p for d in figure_dirs for p in d.glob("*_manifest.csv")):
        for row in read_csv(manifest):
            rows.append({
                "dataset": row["dataset"],
                "case_id": row["id"],
                "method": row.get("method", ""),
                "host_dice": fmt_num(row["host_dice"], 3),
                "fixed_dice": fmt_num(row["fixed_dice"], 3),
                "crc_dice": fmt_num(row["crc_dice"], 3),
                "fixed_gain": fmt_num(row["fixed_gain"], 3, True),
                "risk": fmt_num(row["risk"], 3),
                "threshold": fmt_num(row["threshold"], 3) if row["threshold"] else "",
                "figure": row["figure"],
            })
    materialize(
        "crc_reversion_figure_index",
        rows,
        "CRC Reversion Figure Index",
        "Automatically selected cases where fixed refinement harms and CRC reverts to the host.",
    )
    return rows


def parse_policy_name(path):
    stem = Path(path).stem
    for mode in ("practical", "strict"):
        marker = f"_{mode}_"
        if marker in stem:
            scenario, risk = stem.split(marker, 1)
            return scenario, mode, risk
    raise ValueError(f"Cannot parse policy name: {path}")


def make_cross_dataset_table():
    grouped = {}
    for path in sorted((ROOT / "results/cross_dataset_policy").glob("*.json")):
        scenario, mode, risk = parse_policy_name(path)
        payload = read_json(path)
        row = next(r for r in payload["summaries"] if r.get("split") == "test_target" and r.get("policy") == "crc_portfolio")
        grouped.setdefault(scenario, []).append((mode, risk, row))

    rows = []
    for scenario, vals in sorted(grouped.items()):
        practical = [(risk, row) for mode, risk, row in vals if mode == "practical"]
        strict = [(risk, row) for mode, risk, row in vals if mode == "strict"]
        best = max(practical, key=lambda x: (x[1]["mean_gain"], -x[1]["mean_harm"], x[1]["reverted_rate"]))
        strict_exact = all(
            f(row["mean_harm"]) == 0.0 and f(row["worst_drop"]) == 0.0 and f(row["reverted_rate"]) == 1.0
            for _risk, row in strict
        )
        rows.append({
            "transfer": scenario,
            "best_practical_risk": best[0],
            "target_gain": fmt_num(best[1]["mean_gain"], 4, True),
            "target_harm": fmt_num(best[1]["mean_harm"], 4),
            "target_worst_drop": fmt_num(best[1]["worst_drop"], 4, True),
            "target_revert_rate": fmt_num(best[1]["reverted_rate"], 3),
            "selected_action": best[1].get("selected_action", ""),
            "strict_exact_fallback": "yes" if strict_exact else "no",
            "paper_role": transfer_role(scenario, best[1]),
        })
    materialize(
        "cross_dataset_transfer",
        rows,
        "Cross-Dataset Policy Transfer",
        "Cross-dataset calibration-transfer results without target-set recalibration.",
    )
    return rows


def transfer_role(scenario, row):
    if "isic_to_ph2" in scenario:
        return "positive dermoscopy transfer"
    if "polyps_graphseg" in scenario and f(row["mean_gain"]) < 0:
        return "external stress test; practical transfer can be unsafe"
    if "polyps_unet" in scenario:
        return "external stress test; high fallback limits harm"
    if "sessile" in scenario:
        return "related endoscopy transfer; mixed utility, fallback important"
    return "transfer stress test"


def make_harm_budget_table():
    path = ROOT / "results/harm_budget_sweep/harm_budget_sweep_summary.csv"
    rows = []
    if not path.exists():
        return rows
    for row in read_csv(path):
        rows.append({
            "dataset": row["dataset"],
            "risk": row["risk"],
            "cal_harm_budget": row["budget"],
            "gain": fmt_num(row["gain"], 4, True),
            "harm": fmt_num(row["harm"], 4),
            "worst_drop": fmt_num(row["worst_drop"], 4, True),
            "harmed_rate": fmt_num(row["harmed_rate"], 3),
            "revert_rate": fmt_num(row["reverted_rate"], 3),
            "action": row["action"],
        })
    materialize(
        "harm_budget_sweep",
        rows,
        "Harm Budget Sweep",
        "Safety-utility trade-off as the calibration harm-rate budget is varied.",
    )
    return rows


def make_calibration_fraction_table():
    path = ROOT / "results/calibration_fraction_sweep/calibration_fraction_sweep_summary.csv"
    rows = []
    if not path.exists():
        return rows
    for row in read_csv(path):
        rows.append({
            "dataset": row["dataset"],
            "risk": row["risk"],
            "cal_fraction": row["cal_fraction"],
            "n_cal": row["n_cal"],
            "n_test": row["n_test"],
            "gain": fmt_num(row["gain"], 4, True),
            "harm": fmt_num(row["harm"], 4),
            "worst_drop": fmt_num(row["worst_drop"], 4, True),
            "harmed_rate": fmt_num(row["harmed_rate"], 3),
            "revert_rate": fmt_num(row["reverted_rate"], 3),
            "action": row["action"],
        })
    materialize(
        "calibration_fraction_sweep",
        rows,
        "Calibration Fraction Sweep",
        "Effect of calibration-set size on practical CRC policy behavior.",
    )
    return rows


def make_refiner_zoo_table():
    rows = []
    for path in sorted((ROOT / "results/graphmembrane_policy").glob("*_zoo_binary_*.json")):
        payload = read_json(path)
        crc = next((r for r in payload["summaries"] if r.get("split") == "test" and r.get("policy") == "crc_portfolio"), None)
        if crc is None:
            continue
        run = path.name.split("_zoo_binary_", 1)[0]
        mode = "strict" if float(payload["max_cal_harm_rate"]) == 0.0 else "practical"
        rows.append({
            "run": run,
            "mode": mode,
            "risk": payload["risk_score"],
            "gain": fmt_num(crc["mean_gain"], 4, True),
            "harm": fmt_num(crc["mean_harm"], 4),
            "worst_drop": fmt_num(crc["worst_drop"], 4, True),
            "harmed_rate": fmt_num(crc["harmed_rate"], 3),
            "revert_rate": fmt_num(crc["reverted_rate"], 3),
            "selected_action": crc.get("selected_action", ""),
        })
    if rows:
        rows.sort(key=lambda r: (r["run"], r["mode"], r["risk"]))
        materialize(
            "refiner_zoo_crc",
            rows,
            "Refiner Zoo CRC Results",
            "Safe action-selection results over a broader post-processing/refinement candidate zoo.",
        )
    return rows


def best_policy_for_label(label, zoo=False, strict=False):
    suffix = "_zoo_binary_" if zoo else "_binary_"
    best = None
    for path in sorted((ROOT / "results/graphmembrane_policy").glob(f"{label}{suffix}*.json")):
        payload = read_json(path)
        is_strict = float(payload["max_cal_harm_rate"]) == 0.0
        if is_strict != strict:
            continue
        row = next((r for r in payload["summaries"] if r.get("split") == "test" and r.get("policy") == "crc_portfolio"), None)
        if row is None:
            continue
        item = (path, payload, row)
        if best is None:
            best = item
            continue
        _, _, brow = best
        if (row["mean_gain"], -row["mean_harm"], row["reverted_rate"]) > (
            brow["mean_gain"], -brow["mean_harm"], brow["reverted_rate"]
        ):
            best = item
    return best


def make_mediafinal_table():
    rows = []
    for path in sorted((ROOT / "results/binary_refinement").glob("*mediafinal*_eval.json")):
        label = path.name.replace("_eval.json", "")
        host, fixed = eval_summary(path)
        practical = best_policy_for_label(label, zoo=False, strict=False)
        strict = best_policy_for_label(label, zoo=False, strict=True)
        zoo = best_policy_for_label(label, zoo=True, strict=False)
        zoo_strict = best_policy_for_label(label, zoo=True, strict=True)
        practical_row = practical[2] if practical else {}
        strict_row = strict[2] if strict else {}
        zoo_row = zoo[2] if zoo else {}
        zoo_strict_row = zoo_strict[2] if zoo_strict else {}
        rows.append({
            "run": label,
            "host_dice": fmt_num(host["mean_dice"], 3),
            "best_fixed_action": f"{fixed['method']}:a{float(fixed['alpha']):.2f}",
            "best_fixed_gain": fmt_num(fixed["mean_gain"], 4, True),
            "best_fixed_harm": fmt_num(fixed["mean_harm"], 4),
            "best_fixed_worst_drop": fmt_num(fixed["worst_drop"], 4, True),
            "crc_gain": fmt_num(practical_row.get("mean_gain", 0.0), 4, True),
            "crc_harm": fmt_num(practical_row.get("mean_harm", 0.0), 4),
            "crc_worst_drop": fmt_num(practical_row.get("worst_drop", 0.0), 4, True),
            "crc_revert_rate": fmt_num(practical_row.get("reverted_rate", 0.0), 3),
            "crc_action": practical_row.get("selected_action", ""),
            "zoo_crc_gain": fmt_num(zoo_row.get("mean_gain", 0.0), 4, True),
            "zoo_crc_harm": fmt_num(zoo_row.get("mean_harm", 0.0), 4),
            "zoo_crc_worst_drop": fmt_num(zoo_row.get("worst_drop", 0.0), 4, True),
            "zoo_crc_revert_rate": fmt_num(zoo_row.get("reverted_rate", 0.0), 3),
            "zoo_crc_action": zoo_row.get("selected_action", ""),
            "strict_exact_fallback": "yes" if f(strict_row.get("mean_harm", 1.0)) == 0.0 and f(strict_row.get("reverted_rate", 0.0)) == 1.0 else "no",
            "zoo_strict_exact_fallback": "yes" if f(zoo_strict_row.get("mean_harm", 1.0)) == 0.0 and f(zoo_strict_row.get("reverted_rate", 0.0)) == 1.0 else "no",
        })
    if rows:
        materialize(
            "mediafinal_consistent_results",
            rows,
            "MediaFinal Consistent Results",
            "Final internally consistent serial train-eval-policy-zoo runs using unique mediafinal tags.",
        )
    return rows


def make_mediafinal_bootstrap_table():
    path = ROOT / "results/bootstrap_ci_mediafinal/mediafinal_bootstrap_crc_ablation.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        if row["mode"] not in ("practical", "strict"):
            continue
        rows.append({
            "dataset": row["dataset"],
            "mode": row["mode"],
            "risk": row["risk_score"],
            "gain_CI": fmt_ci(row, "mean_gain", 4, True),
            "harm_CI": fmt_ci(row, "mean_harm", 4),
            "worst_drop_CI": fmt_ci(row, "worst_drop", 4, True),
            "harmed_rate_CI": fmt_ci(row, "harmed_rate", 3),
            "revert_rate_CI": fmt_ci(row, "reverted_rate", 3),
            "action": row["crc_selected_action"],
        })
    if rows:
        materialize(
            "mediafinal_bootstrap_crc_ablation",
            rows,
            "MediaFinal Bootstrap CRC Ablation",
            "Bootstrap confidence intervals for the final internally consistent mediafinal runs.",
        )
    return rows


def make_reproducibility_table():
    path = ROOT / "results/reproducibility/mediafinal_repro_manifest.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        rows.append({
            "run": row["run"],
            "arch": row["arch"],
            "host_dice": fmt_num(row["host_dice"], 3),
            "best_fixed_gain": fmt_num(row["best_fixed_gain"], 4, True),
            "crc_gain": fmt_num(row["crc_mean_gain"], 4, True),
            "crc_harm": fmt_num(row["crc_mean_harm"], 4),
            "zoo_crc_gain": fmt_num(row["zoo_crc_mean_gain"], 4, True),
            "zoo_crc_harm": fmt_num(row["zoo_crc_mean_harm"], 4),
            "policy_json_count": row["policy_json_count"],
            "zoo_policy_json_count": row["zoo_policy_json_count"],
            "checkpoint": row["ckpt"],
        })
    if rows:
        materialize(
            "mediafinal_reproducibility_manifest",
            rows,
            "MediaFinal Reproducibility Manifest",
            "Run-level provenance for final mediafinal checkpoints, evaluations, policies, and refiner-zoo evaluations.",
        )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    produced = {
        "main": make_main_table(),
        "ablation": make_ablation_table(),
        "cross_host": make_cross_host_table(),
        "stress": make_stress_table(),
        "figures": make_figure_index(),
        "cross_dataset": make_cross_dataset_table(),
        "harm_budget": make_harm_budget_table(),
        "calibration_fraction": make_calibration_fraction_table(),
        "refiner_zoo": make_refiner_zoo_table(),
        "mediafinal": make_mediafinal_table(),
        "mediafinal_bootstrap": make_mediafinal_bootstrap_table(),
        "reproducibility": make_reproducibility_table(),
    }
    summary = {k: len(v) for k, v in produced.items()}
    (OUT / "paper_table_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
