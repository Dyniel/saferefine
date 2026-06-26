#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from unified_graphseg import (  # noqa: E402
    GRAPH_OUTPUT_MODES,
    GRAPH_SAFETY_GATES,
    GraphSegUnified,
    Refuge2NPZ,
    collate,
    dice_bin,
    read_idx_file,
    set_seed,
    ts,
)


def parse_alphas(s):
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    return vals


def cp_d(pred, gt):
    cup = dice_bin(pred == 2, gt == 2)
    disc = dice_bin(pred > 0, gt > 0)
    return cup, disc, cup + disc


def host_uncertainty(logits):
    probs = F.softmax(logits, dim=1)
    ent = -(probs * torch.clamp(probs, min=1e-6).log()).sum(dim=1)
    ent = ent / math.log(float(logits.shape[1]))
    top2 = torch.topk(probs, k=2, dim=1).values
    margin_unc = 1.0 - (top2[:, 0] - top2[:, 1])
    return ent, margin_unc


RISK_SCORE_MODES = ("change", "alpha", "change_over_uncertainty", "change_times_confidence")


def risk_score(row, mode):
    mode = str(mode)
    if mode == "change":
        return float(row["risk_change"])
    if mode == "alpha":
        return float(row["alpha"])
    if mode == "change_over_uncertainty":
        unc = max(float(row["host_entropy"]), float(row["host_margin_unc"]), 1e-4)
        return float(row["risk_change"]) / unc
    if mode == "change_times_confidence":
        unc = max(float(row["host_entropy"]), float(row["host_margin_unc"]))
        return float(row["risk_change"]) * (1.0 - min(max(unc, 0.0), 1.0))
    raise ValueError(f"Unknown risk score mode: {mode}")


def summarize(name, selected, host_by_id, harm_eps):
    gains = []
    harms = []
    cups = []
    discs = []
    cpds = []
    alphas = []
    reverted = 0

    for row in selected:
        host = host_by_id[row["id"]]
        gain = row["cpd"] - host["cpd"]
        gains.append(gain)
        harms.append(max(0.0, -gain))
        cups.append(row["cup"])
        discs.append(row["disc"])
        cpds.append(row["cpd"])
        alphas.append(row["alpha"])
        if abs(row["alpha"]) < 1e-12:
            reverted += 1

    gains = np.asarray(gains, dtype=np.float64)
    harms = np.asarray(harms, dtype=np.float64)
    cpds = np.asarray(cpds, dtype=np.float64)
    cups = np.asarray(cups, dtype=np.float64)
    discs = np.asarray(discs, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    n = max(1, len(selected))

    return {
        "policy": name,
        "n": int(len(selected)),
        "mean_cpd": float(cpds.mean()) if len(cpds) else 0.0,
        "mean_cup": float(cups.mean()) if len(cups) else 0.0,
        "mean_disc": float(discs.mean()) if len(discs) else 0.0,
        "mean_gain": float(gains.mean()) if len(gains) else 0.0,
        "median_gain": float(np.median(gains)) if len(gains) else 0.0,
        "mean_harm": float(harms.mean()) if len(harms) else 0.0,
        "worst_drop": float(gains.min()) if len(gains) else 0.0,
        "improved_rate": float((gains > harm_eps).mean()) if len(gains) else 0.0,
        "harmed_rate": float((gains < -harm_eps).mean()) if len(gains) else 0.0,
        "reverted_rate": float(reverted / n),
        "mean_alpha": float(alphas.mean()) if len(alphas) else 0.0,
    }


def select_fixed(rows_by_id, alpha):
    out = []
    for _id, rows in rows_by_id.items():
        best = min(rows, key=lambda r: abs(r["alpha"] - alpha))
        out.append(best)
    return out


def select_oracle(rows_by_id):
    return [max(rows, key=lambda r: r["cpd"]) for rows in rows_by_id.values()]


def select_threshold(rows_by_id, threshold, risk_score_mode="change"):
    if threshold is None:
        return select_fixed(rows_by_id, 0.0)

    selected = []
    for _id, rows in rows_by_id.items():
        ok = [r for r in rows if risk_score(r, risk_score_mode) <= threshold + 1e-12]
        if not ok:
            ok = [min(rows, key=lambda r: abs(r["alpha"]))]
        selected.append(max(ok, key=lambda r: r["alpha"]))
    return selected


def split_rows(rows, cal_fraction):
    ids = []
    seen = set()
    for r in rows:
        if r["id"] not in seen:
            ids.append(r["id"])
            seen.add(r["id"])
    n_cal = int(round(len(ids) * cal_fraction))
    n_cal = min(max(1, n_cal), max(1, len(ids) - 1))
    cal_ids = set(ids[:n_cal])
    test_ids = set(ids[n_cal:])
    return cal_ids, test_ids


def group_rows(rows, ids):
    out = {}
    for r in rows:
        if r["id"] in ids:
            out.setdefault(r["id"], []).append(r)
    for k in out:
        out[k] = sorted(out[k], key=lambda r: r["alpha"])
    return out


def host_rows_by_id(rows):
    hosts = {}
    for r in rows:
        if abs(r["alpha"]) < 1e-12:
            hosts[r["id"]] = r
    return hosts


def candidate_thresholds(rows_by_id, risk_score_mode):
    vals = {0.0}
    for rows in rows_by_id.values():
        for r in rows:
            vals.add(float(risk_score(r, risk_score_mode)))
    return sorted(vals)


def calibrate_threshold(rows_by_id, host_by_id, harm_eps, beta_harm, max_harm_rate, risk_score_mode):
    risks = candidate_thresholds(rows_by_id, risk_score_mode)
    candidates = []
    for t in risks:
        sel = select_threshold(rows_by_id, t, risk_score_mode=risk_score_mode)
        s = summarize(f"threshold_{t:.6f}", sel, host_by_id, harm_eps)
        utility = s["mean_gain"] - beta_harm * s["mean_harm"]
        s["threshold"] = float(t)
        s["risk_score_mode"] = risk_score_mode
        s["utility"] = float(utility)
        candidates.append(s)

    feasible = [s for s in candidates if s["harmed_rate"] <= max_harm_rate]
    pool = feasible if feasible else candidates
    best = max(pool, key=lambda s: (s["utility"], s["mean_gain"], -s["mean_harm"]))
    best["feasible_thresholds"] = int(len(feasible))
    best["tested_thresholds"] = int(len(candidates))
    return best


def hoeffding_upper(mean, n, confidence, n_candidates, value_range=1.0):
    if n <= 0:
        return 1.0
    delta = max(float(confidence), 1e-12) / max(1, int(n_candidates))
    radius = float(value_range) * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return float(min(1.0, mean + radius))


def calibrate_crc_threshold(
    rows_by_id,
    host_by_id,
    harm_eps,
    beta_harm,
    max_harm_rate,
    confidence,
    risk_score_mode,
):
    risks = candidate_thresholds(rows_by_id, risk_score_mode)
    candidates = []
    for t in risks:
        sel = select_threshold(rows_by_id, t, risk_score_mode=risk_score_mode)
        s = summarize(f"crc_{t:.6f}", sel, host_by_id, harm_eps)
        n = max(1, int(s["n"]))
        s["threshold"] = float(t)
        s["risk_score_mode"] = risk_score_mode
        s["utility"] = float(s["mean_gain"] - beta_harm * s["mean_harm"])
        s["harm_rate_ucb"] = hoeffding_upper(
            s["harmed_rate"],
            n=n,
            confidence=confidence,
            n_candidates=len(risks),
            value_range=1.0,
        )
        candidates.append(s)

    feasible = [s for s in candidates if s["harm_rate_ucb"] <= max_harm_rate]
    if feasible:
        best = max(feasible, key=lambda s: (s["utility"], s["mean_gain"], -s["mean_harm"]))
        best["forced_host"] = False
    else:
        # Exact host fallback has zero harm by construction; use it when sample
        # uncertainty is too large to certify any non-trivial threshold.
        best = summarize("crc_host_fallback", select_fixed(rows_by_id, 0.0), host_by_id, harm_eps)
        best["threshold"] = None
        best["risk_score_mode"] = risk_score_mode
        best["utility"] = float(best["mean_gain"] - beta_harm * best["mean_harm"])
        best["harm_rate_ucb"] = 0.0
        best["forced_host"] = True

    best["feasible_thresholds"] = int(len(feasible))
    best["tested_thresholds"] = int(len(candidates))
    best["crc_confidence"] = float(confidence)
    best["crc_max_harm_rate"] = float(max_harm_rate)
    return best


@torch.no_grad()
def collect_rows(args, alphas):
    idx_va = read_idx_file(args.idx_val)
    ds = Refuge2NPZ(args.npz_all, idx_va, img_size=args.img_size, train=False, verbose=bool(args.data_verbose))
    loader = DataLoader(
        ds,
        batch_size=max(1, args.batch),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_alpha = max([a for a in alphas if a > 0.0] or [args.alpha_graph])
    model = GraphSegUnified(
        backbone=args.backbone,
        num_classes=3,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        depth=args.depth,
        graph_down=args.graph_down,
        grid4=bool(args.grid4),
        dyn_on=args.dyn_on,
        dyn_k=args.dyn_k,
        dyn_window=args.dyn_window,
        alpha_graph=model_alpha,
        graph_output_mode=args.graph_output_mode,
        graph_safety_gate=args.graph_safety_gate,
        graph_gate_floor=args.graph_gate_floor,
        graph_gate_power=args.graph_gate_power,
        graph_residual_clip=args.graph_residual_clip,
    ).to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(dummy, dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k)
    del dummy

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{ts()}][ckpt] loaded={args.ckpt}", flush=True)
    if missing:
        print(f"[{ts()}][ckpt] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[{ts()}][ckpt] unexpected keys: {len(unexpected)}", flush=True)

    model.eval()
    rows = []
    t0 = time.time()
    for bi, (xb, yb, meta) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        gt = yb.numpy()

        model.alpha_graph = 0.0
        host_logits, _, _ = model(xb, dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k)
        host_pred = torch.argmax(host_logits, dim=1).detach().cpu().numpy()
        ent, margin_unc = host_uncertainty(host_logits)
        ent_np = ent.detach().cpu().numpy()
        margin_np = margin_unc.detach().cpu().numpy()

        for alpha in alphas:
            model.alpha_graph = float(alpha)
            logits, _, _ = model(xb, dyn_on_eval=args.dyn_on_eval, eval_dyn_k=args.eval_dyn_k)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            bs = pred.shape[0]
            for i in range(bs):
                cup, disc, cpd = cp_d(pred[i], gt[i])
                changed = float((pred[i] != host_pred[i]).mean())
                rows.append({
                    "id": str(meta[i].get("id", "?")),
                    "idx": int(meta[i].get("idx", -1)),
                    "alpha": float(alpha),
                    "cup": float(cup),
                    "disc": float(disc),
                    "cpd": float(cpd),
                    "risk_change": changed,
                    "host_entropy": float(ent_np[i].mean()),
                    "host_margin_unc": float(margin_np[i].mean()),
                    "split_order": int(len({r["id"] for r in rows})),
                })

        if args.max_batches > 0 and (bi + 1) >= args.max_batches:
            break

    print(f"[{ts()}][eval] collected rows={len(rows)} images={len(set(r['id'] for r in rows))} sec={time.time()-t0:.1f}", flush=True)
    return rows


def write_csv(path, rows):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "idx", "alpha", "cup", "disc", "cpd", "risk_change", "host_entropy", "host_margin_unc", "split_order"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_all", required=True)
    ap.add_argument("--idx_val", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", default="0,0.05,0.10,0.15,0.25,0.40")

    ap.add_argument("--backbone", type=str, default="segformer_b0")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--graph_down", type=int, default=2)
    ap.add_argument("--grid4", type=int, default=1)
    ap.add_argument("--dyn_on", type=str, default="feat")
    ap.add_argument("--dyn_k", type=int, default=16)
    ap.add_argument("--dyn_window", type=int, default=2)
    ap.add_argument("--dyn_on_eval", type=str, default="feat")
    ap.add_argument("--eval_dyn_k", type=int, default=16)
    ap.add_argument("--alpha_graph", type=float, default=0.25)
    ap.add_argument("--graph_output_mode", type=str, default="residual", choices=GRAPH_OUTPUT_MODES)
    ap.add_argument("--graph_safety_gate", type=str, default="none", choices=GRAPH_SAFETY_GATES)
    ap.add_argument("--graph_gate_floor", type=float, default=0.0)
    ap.add_argument("--graph_gate_power", type=float, default=1.0)
    ap.add_argument("--graph_residual_clip", type=float, default=0.0)

    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--harm_eps", type=float, default=0.0)
    ap.add_argument("--beta_harm", type=float, default=2.0)
    ap.add_argument("--max_cal_harm_rate", type=float, default=0.25)
    ap.add_argument("--crc_confidence", type=float, default=0.10, help="Failure probability for CRC/union-bound harm-rate UCB.")
    ap.add_argument("--risk_score", type=str, default="change", choices=RISK_SCORE_MODES)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--data_verbose", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--out_summary", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    alphas = parse_alphas(args.alphas)
    print(f"[{ts()}][cfg] ckpt={args.ckpt}", flush=True)
    print(f"[{ts()}][cfg] alphas={alphas}", flush=True)
    print(f"[{ts()}][cfg] risk_score={args.risk_score} max_cal_harm_rate={args.max_cal_harm_rate} crc_confidence={args.crc_confidence}", flush=True)
    print(f"[{ts()}][cfg] graph_safety_gate={args.graph_safety_gate} gate_floor={args.graph_gate_floor} gate_power={args.graph_gate_power} residual_clip={args.graph_residual_clip}", flush=True)

    rows = collect_rows(args, alphas)
    write_csv(args.out_csv, rows)

    cal_ids, test_ids = split_rows(rows, args.cal_fraction)
    all_ids = cal_ids | test_ids
    host_all = host_rows_by_id(rows)
    rows_cal = group_rows(rows, cal_ids)
    rows_test = group_rows(rows, test_ids)
    host_cal = {k: host_all[k] for k in cal_ids}
    host_test = {k: host_all[k] for k in test_ids}

    threshold_cal = calibrate_threshold(
        rows_cal,
        host_cal,
        harm_eps=args.harm_eps,
        beta_harm=args.beta_harm,
        max_harm_rate=args.max_cal_harm_rate,
        risk_score_mode=args.risk_score,
    )
    crc_cal = calibrate_crc_threshold(
        rows_cal,
        host_cal,
        harm_eps=args.harm_eps,
        beta_harm=args.beta_harm,
        max_harm_rate=args.max_cal_harm_rate,
        confidence=args.crc_confidence,
        risk_score_mode=args.risk_score,
    )
    threshold = threshold_cal["threshold"]
    crc_threshold = crc_cal["threshold"]

    summaries = []
    for split_name, grouped, hosts in [("cal", rows_cal, host_cal), ("test", rows_test, host_test)]:
        summaries.append({"split": split_name, **summarize("host", select_fixed(grouped, 0.0), hosts, args.harm_eps)})
        for a in alphas:
            summaries.append({"split": split_name, **summarize(f"fixed_alpha_{a:.2f}", select_fixed(grouped, a), hosts, args.harm_eps)})
        summaries.append({"split": split_name, **summarize("oracle", select_oracle(grouped), hosts, args.harm_eps)})
        sel_thr = select_threshold(grouped, threshold, risk_score_mode=args.risk_score)
        s_thr = summarize("risk_threshold", sel_thr, hosts, args.harm_eps)
        s_thr["threshold"] = threshold
        s_thr["risk_score_mode"] = args.risk_score
        if split_name == "cal":
            s_thr["cal_utility"] = threshold_cal["utility"]
            s_thr["feasible_thresholds"] = threshold_cal["feasible_thresholds"]
            s_thr["tested_thresholds"] = threshold_cal["tested_thresholds"]
        summaries.append({"split": split_name, **s_thr})

        sel_crc = select_threshold(grouped, crc_threshold, risk_score_mode=args.risk_score)
        s_crc = summarize("crc_threshold", sel_crc, hosts, args.harm_eps)
        s_crc["threshold"] = crc_threshold
        s_crc["risk_score_mode"] = args.risk_score
        if split_name == "cal":
            s_crc["harm_rate_ucb"] = crc_cal["harm_rate_ucb"]
            s_crc["crc_confidence"] = crc_cal["crc_confidence"]
            s_crc["crc_max_harm_rate"] = crc_cal["crc_max_harm_rate"]
            s_crc["forced_host"] = crc_cal["forced_host"]
            s_crc["feasible_thresholds"] = crc_cal["feasible_thresholds"]
            s_crc["tested_thresholds"] = crc_cal["tested_thresholds"]
        summaries.append({"split": split_name, **s_crc})

    print("\nsplit\tpolicy\tn\tmean_cpd\tmean_gain\tmean_harm\tworst_drop\timproved_rate\tharmed_rate\treverted_rate\tmean_alpha\tthreshold\tharm_rate_ucb", flush=True)
    for s in summaries:
        threshold_value = s.get("threshold", "-")
        ucb_value = s.get("harm_rate_ucb", "-")
        print(
            f"{s['split']}\t{s['policy']}\t{s['n']}\t{s['mean_cpd']:.6f}\t{s['mean_gain']:.6f}\t{s['mean_harm']:.6f}\t"
            f"{s['worst_drop']:.6f}\t{s['improved_rate']:.4f}\t{s['harmed_rate']:.4f}\t{s['reverted_rate']:.4f}\t"
            f"{s['mean_alpha']:.4f}\t{threshold_value}\t{ucb_value}",
            flush=True,
        )

    payload = {
        "ckpt": args.ckpt,
        "alphas": alphas,
        "cal_fraction": args.cal_fraction,
        "harm_eps": args.harm_eps,
        "beta_harm": args.beta_harm,
        "max_cal_harm_rate": args.max_cal_harm_rate,
        "crc_confidence": args.crc_confidence,
        "risk_score": args.risk_score,
        "n_images": len(all_ids),
        "n_cal": len(cal_ids),
        "n_test": len(test_ids),
        "selected_threshold": threshold,
        "selected_crc_threshold": crc_threshold,
        "crc_calibration": crc_cal,
        "summaries": summaries,
    }
    if args.out_summary:
        p = Path(args.out_summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
