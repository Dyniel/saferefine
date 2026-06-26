#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_record(path, root=ROOT):
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        return {
            "path": str(p),
            "exists": False,
            "size_bytes": "",
            "mtime_ns": "",
            "sha256": "",
        }
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha256_file(p),
    }


def fnum(x, default=0.0):
    if x is None or x == "":
        return default
    return float(x)


def fmt_float(x):
    if x is None or x == "":
        return ""
    return f"{float(x):.12g}"


def eval_summary(payload):
    rows = payload.get("summary", [])
    host = next((r for r in rows if r.get("method") == "host"), {})
    fixed = max((r for r in rows if r.get("method") != "host"), key=lambda r: fnum(r.get("mean_dice")), default={})
    return host, fixed


def best_policy(pattern, strict=False):
    best = None
    for path in sorted(ROOT.glob(pattern)):
        payload = read_json(path)
        is_strict = fnum(payload.get("max_cal_harm_rate"), 1.0) == 0.0
        if is_strict != strict:
            continue
        row = next(
            (r for r in payload.get("summaries", []) if r.get("split") == "test" and r.get("policy") == "crc_portfolio"),
            None,
        )
        if row is None:
            continue
        item = (path, payload, row)
        if best is None:
            best = item
            continue
        _, _, brow = best
        if (fnum(row.get("mean_gain")), -fnum(row.get("mean_harm")), fnum(row.get("reverted_rate"))) > (
            fnum(brow.get("mean_gain")), -fnum(brow.get("mean_harm")), fnum(brow.get("reverted_rate"))
        ):
            best = item
    return best


def parse_policy_filename(path):
    stem = Path(path).stem
    m = re.match(r"(.+)_binary_(practical|strict)_(change_plus_geom|changed|geom)_(\d+)_([0-9_]+)$", stem)
    if not m:
        return {"scenario": stem, "mode": "", "risk_score": "", "slurm_job": "", "timestamp": ""}
    return {
        "scenario": m.group(1),
        "mode": m.group(2),
        "risk_score": m.group(3),
        "slurm_job": m.group(4),
        "timestamp": m.group(5),
    }


def manifest_rows():
    rows = []
    artifact_records = []
    for eval_json in sorted((ROOT / "results/binary_refinement").glob("*mediafinal*_eval.json")):
        label = eval_json.name.replace("_eval.json", "")
        payload = read_json(eval_json)
        args = payload.get("args", {})
        host, fixed = eval_summary(payload)
        eval_csv = Path(args.get("out_csv", ROOT / "results/binary_refinement" / f"{label}_eval.csv"))
        zoo_label = f"{label}_zoo"
        zoo_csv = ROOT / "results/refiner_zoo" / f"{zoo_label}.csv"
        zoo_json = ROOT / "results/refiner_zoo" / f"{zoo_label}.json"
        practical = best_policy(f"results/graphmembrane_policy/{label}_binary_*.json", strict=False)
        strict = best_policy(f"results/graphmembrane_policy/{label}_binary_*.json", strict=True)
        zoo_practical = best_policy(f"results/graphmembrane_policy/{zoo_label}_binary_*.json", strict=False)
        zoo_strict = best_policy(f"results/graphmembrane_policy/{zoo_label}_binary_*.json", strict=True)

        policy_glob = sorted((ROOT / "results/graphmembrane_policy").glob(f"{label}_binary_*.json"))
        zoo_policy_glob = sorted((ROOT / "results/graphmembrane_policy").glob(f"{zoo_label}_binary_*.json"))

        row = {
            "run": label,
            "arch": args.get("arch", ""),
            "backbone": args.get("backbone", ""),
            "unet_base": args.get("unet_base", ""),
            "seed": args.get("seed", ""),
            "img_size": args.get("img_size", ""),
            "npz_all": args.get("npz_all", ""),
            "idx_eval": args.get("idx_eval", ""),
            "ckpt": args.get("ckpt", ""),
            "eval_json": str(eval_json),
            "eval_csv": str(eval_csv),
            "zoo_json": str(zoo_json),
            "zoo_csv": str(zoo_csv),
            "policy_json_count": len(policy_glob),
            "zoo_policy_json_count": len(zoo_policy_glob),
            "host_dice": fmt_float(host.get("mean_dice")),
            "best_fixed_method": fixed.get("method", ""),
            "best_fixed_alpha": fmt_float(fixed.get("alpha")),
            "best_fixed_gain": fmt_float(fixed.get("mean_gain")),
            "best_fixed_harm": fmt_float(fixed.get("mean_harm")),
            "best_fixed_worst_drop": fmt_float(fixed.get("worst_drop")),
        }
        for prefix, item in [
            ("crc", practical),
            ("strict_crc", strict),
            ("zoo_crc", zoo_practical),
            ("zoo_strict_crc", zoo_strict),
        ]:
            if item is None:
                row.update({
                    f"{prefix}_policy_json": "",
                    f"{prefix}_risk_score": "",
                    f"{prefix}_selected_action": "",
                    f"{prefix}_threshold": "",
                    f"{prefix}_mean_gain": "",
                    f"{prefix}_mean_harm": "",
                    f"{prefix}_worst_drop": "",
                    f"{prefix}_reverted_rate": "",
                })
                continue
            path, policy_payload, policy_row = item
            row.update({
                f"{prefix}_policy_json": str(path),
                f"{prefix}_risk_score": policy_payload.get("risk_score", ""),
                f"{prefix}_selected_action": policy_row.get("selected_action", ""),
                f"{prefix}_threshold": fmt_float(policy_row.get("threshold")),
                f"{prefix}_mean_gain": fmt_float(policy_row.get("mean_gain")),
                f"{prefix}_mean_harm": fmt_float(policy_row.get("mean_harm")),
                f"{prefix}_worst_drop": fmt_float(policy_row.get("worst_drop")),
                f"{prefix}_reverted_rate": fmt_float(policy_row.get("reverted_rate")),
            })
        rows.append(row)

        for role, path in [
            ("checkpoint", args.get("ckpt", "")),
            ("npz_all", args.get("npz_all", "")),
            ("idx_eval", args.get("idx_eval", "")),
            ("eval_json", eval_json),
            ("eval_csv", eval_csv),
            ("zoo_json", zoo_json),
            ("zoo_csv", zoo_csv),
        ]:
            if path:
                rec = file_record(path)
                artifact_records.append({"run": label, "role": role, **rec})
        for path in policy_glob + zoo_policy_glob:
            meta = parse_policy_filename(path)
            rec = file_record(path)
            artifact_records.append({"run": label, "role": "policy_json", **meta, **rec})
    return rows, artifact_records


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/reproducibility")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, artifacts = manifest_rows()
    write_csv(out_dir / "mediafinal_repro_manifest.csv", runs)
    write_csv(out_dir / "mediafinal_artifact_hashes.csv", artifacts)
    payload = {
        "root": str(ROOT),
        "runs": runs,
        "artifacts": artifacts,
        "counts": {"runs": len(runs), "artifacts": len(artifacts)},
    }
    (out_dir / "mediafinal_repro_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
