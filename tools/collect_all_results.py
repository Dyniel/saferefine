#!/usr/bin/env python3
from __future__ import annotations
import os, re, csv, time, math
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/student2/jaskra")
RUNS = ROOT / "runs"

def nowstamp():
    return time.strftime("%Y%m%d_%H%M%S")

def safe_float(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s in ("", "-", "NA", "None"): return None
        return float(s)
    except:
        return None

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals)/len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v-m)**2 for v in vals)/(len(vals)-1)
    return m, math.sqrt(var)

def find_latest(pattern: str) -> Path | None:
    cands = sorted(RUNS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def read_tsv(p: Path):
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows

def write_tsv(p: Path, rows, fieldnames):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def sniff_family_model_variant(run: str):
    r = run
    family = "other"
    model = "other"
    if "deeplab" in r:
        family = "deeplab"
        model = "deeplabv3_resnet50" if "resnet50" in r else "deeplab"
    elif "segformer" in r:
        family = "segformer"
        model = "segformer_b0" if "b0" in r else "segformer"
    elif "hrnet" in r or "hrw48" in r:
        family = "hrnet"
        model = "hrnet_w48"
    # variant
    variant = "unknown"
    if "no_graph" in r:
        variant = "no_graph"
    elif "grid_only" in r or "grid" in r:
        variant = "grid_only"
    elif "dynk" in r or "dyn" in r:
        variant = "dyn"
    # seed
    m = re.search(r"_s(\d+)\b", r)
    seed = int(m.group(1)) if m else None
    return family, model, variant, seed

def parse_console_log(console: Path):
    txt = console.read_text(encoding="utf-8", errors="ignore").splitlines()

    # DONE line: DONE best(C+D)=X at ep=Y
    bestCD = None; best_ep = None
    for ln in reversed(txt):
        if "DONE best(C+D)=" in ln:
            m = re.search(r"DONE best\(C\+D\)=([0-9.]+)\s+at ep=([0-9]+)", ln)
            if m:
                bestCD = float(m.group(1)); best_ep = int(m.group(2))
            break

    # last epoch line
    lastEp = None; lastVL = None
    for ln in reversed(txt):
        if re.search(r"\bEpoch\s+\d+\b", ln):
            m1 = re.search(r"Epoch\s+(\d+)", ln)
            m2 = re.search(r"\bVL\s+([0-9.]+)", ln)
            if m1: lastEp = int(m1.group(1))
            if m2: lastVL = float(m2.group(1))
            break

    # bestC/bestD: from the epoch line that matches best_ep if present, else last dice line
    bestC = None; bestD = None
    if best_ep is not None:
        pat = re.compile(rf"\bEpoch\s+0*{best_ep}\b")
        for ln in txt:
            if pat.search(ln) and "dice=C:" in ln and "/D:" in ln:
                mC = re.search(r"dice=C:([0-9.]+)", ln)
                mD = re.search(r"/D:([0-9.]+)", ln)
                if mC: bestC = float(mC.group(1))
                if mD: bestD = float(mD.group(1))
                break
    if bestC is None or bestD is None:
        for ln in reversed(txt):
            if "dice=C:" in ln and "/D:" in ln:
                mC = re.search(r"dice=C:([0-9.]+)", ln)
                mD = re.search(r"/D:([0-9.]+)", ln)
                if mC: bestC = float(mC.group(1))
                if mD: bestD = float(mD.group(1))
                break

    status = "DONE" if bestCD is not None else "FAIL"
    # error tag
    errTag = ""
    errLine = ""
    if status != "DONE":
        for ln in txt:
            if "OutOfMemoryError" in ln:
                errTag = "OOM"; errLine = ln.strip(); break
            if "RuntimeError" in ln or "Traceback" in ln:
                errTag = "ERR"; errLine = ln.strip(); break

    return dict(
        status=status,
        bestCD=bestCD,
        best_ep=best_ep,
        bestC=bestC,
        bestD=bestD,
        lastEp=lastEp,
        lastVL=lastVL,
        errTag=errTag,
        errLine=errLine,
    )

def collect_train_runs(roots):
    rows = []
    for root in roots:
        if not root.exists(): 
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir(): 
                continue
            console = d / "console.log"
            if not console.exists():
                continue
            run = d.name
            fam, model, variant, seed = sniff_family_model_variant(run)
            meta = parse_console_log(console)
            rows.append({
                "root": str(root),
                "run": run,
                "family": fam,
                "model": model,
                "variant": variant,
                "seed": "" if seed is None else str(seed),
                "status": meta["status"],
                "bestCD": "" if meta["bestCD"] is None else f"{meta['bestCD']:.6f}",
                "best_ep": "" if meta["best_ep"] is None else str(meta["best_ep"]),
                "bestC": "" if meta["bestC"] is None else f"{meta['bestC']:.6f}",
                "bestD": "" if meta["bestD"] is None else f"{meta['bestD']:.6f}",
                "lastEp": "" if meta["lastEp"] is None else str(meta["lastEp"]),
                "lastVL": "" if meta["lastVL"] is None else f"{meta['lastVL']:.6f}",
                "errTag": meta["errTag"],
                "errLine": meta["errLine"][:240],
                "console": str(console),
            })
    return rows

def group_train(rows):
    # group by (family, model, variant), only DONE
    g = defaultdict(list)
    for r in rows:
        if r["status"] != "DONE": 
            continue
        key = (r["family"], r["model"], r["variant"])
        g[key].append(safe_float(r["bestCD"]))
    out = []
    for (fam, model, variant), vals in sorted(g.items()):
        m, s = mean_std(vals)
        out.append({
            "family": fam,
            "model": model,
            "variant": variant,
            "n": str(len([v for v in vals if v is not None])),
            "bestCD_mean": "" if m is None else f"{m:.6f}",
            "bestCD_std": "" if s is None else f"{s:.6f}",
        })
    return out

def load_refine_results():
    latest = find_latest("_refine_all_*/results.tsv")
    if latest is None:
        return None, []
    rows = read_tsv(latest)
    # reduce columns for sanity
    keep = []
    for r in rows:
        keep.append({
            "run_dir": r.get("run_dir",""),
            "run": r.get("run",""),
            "variant": r.get("variant",""),
            "status": r.get("status",""),
            "raw_CpD": r.get("raw_CpD",""),
            "raw_C": r.get("raw_C",""),
            "raw_D": r.get("raw_D",""),
            "raw_sec_img": r.get("raw_sec_img",""),
            "morph_CpD": r.get("morph_CpD",""),
            "morph_sec_img": r.get("morph_sec_img",""),
            "crf_CpD": r.get("crf_CpD",""),
            "crf_sec_img": r.get("crf_sec_img",""),
            "learned_CpD": r.get("learned_CpD",""),
            "learned_sec_img": r.get("learned_sec_img",""),
            "log": r.get("log",""),
        })
    return latest, keep

def load_xfer_summary():
    latest = find_latest("_xfer_infer_*/_SUMMARY.tsv")
    if latest is None:
        return None, []
    rows = read_tsv(latest)
    return latest, rows

def md_table(rows, cols, max_rows=30):
    # simple markdown table
    head = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"]*len(cols)) + " |"
    lines = [head, sep]
    for r in rows[:max_rows]:
        lines.append("| " + " | ".join(str(r.get(c,"")) for c in cols) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n… ({len(rows)-max_rows} more rows)\n")
    return "\n".join(lines)

def main():
    out = RUNS / f"_all_results_{nowstamp()}"
    out.mkdir(parents=True, exist_ok=True)

    train_roots = [
        RUNS / "eccv_pub_segformer_deeplab",
        RUNS / "eccv_pub_segformer_amp0",
        RUNS / "eccv_pt6",
        RUNS / "eccv_closeout",
    ]

    train = collect_train_runs(train_roots)
    train_fields = ["root","run","family","model","variant","seed","status","bestCD","best_ep","bestC","bestD","lastEp","lastVL","errTag","errLine","console"]
    write_tsv(out/"TRAIN_runs.tsv", train, train_fields)

    train_grp = group_train(train)
    grp_fields = ["family","model","variant","n","bestCD_mean","bestCD_std"]
    write_tsv(out/"TRAIN_group_mean_std.tsv", train_grp, grp_fields)

    refine_path, refine = load_refine_results()
    if refine:
        refine_fields = list(refine[0].keys())
        write_tsv(out/"REFINE_results.tsv", refine, refine_fields)

    xfer_path, xfer = load_xfer_summary()
    if xfer:
        xfer_fields = list(xfer[0].keys())
        write_tsv(out/"XFER_summary.tsv", xfer, xfer_fields)

    # -------- supervisor markdown --------
    md = []
    md.append(f"# ECCV Graph Refiner – Results Pack\n")
    md.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
    md.append(f"- Output dir: `{out}`\n")

    md.append("\n## 1) Training runs – grouped (mean±std of best(C+D))\n")
    md.append(md_table(train_grp, ["family","model","variant","n","bestCD_mean","bestCD_std"], max_rows=50))

    md.append("\n## 2) Training runs – top by best(C+D)\n")
    done = [r for r in train if r["status"] == "DONE" and safe_float(r["bestCD"]) is not None]
    done_sorted = sorted(done, key=lambda r: safe_float(r["bestCD"]) or -1e9, reverse=True)
    md.append(md_table(done_sorted, ["run","family","model","variant","seed","bestCD","bestC","bestD","best_ep","root"], max_rows=25))

    if refine:
        md.append("\n## 3) Refinement baselines (RAW/MORPH/CRF/LEARNED) – latest run_refine_all\n")
        md.append(f"- Source: `{refine_path}`\n")
        md.append(md_table(refine, ["run","variant","status","raw_CpD","raw_sec_img","morph_CpD","morph_sec_img","crf_CpD","crf_sec_img","learned_CpD","learned_sec_img"], max_rows=40))

    if xfer:
        md.append("\n## 4) Cross-dataset inference summary (latest _xfer_infer)\n")
        md.append(f"- Source: `{xfer_path}`\n")
        # try show top 25 by mean(C+D) if columns exist
        cand_cols = ["ds","run","mean(C+D)","n","path"]
        if all(c in xfer[0] for c in cand_cols):
            xs = sorted(xfer, key=lambda r: safe_float(r.get("mean(C+D)","")) or -1e9, reverse=True)
            md.append(md_table(xs, cand_cols, max_rows=25))
        else:
            md.append(md_table(xfer, list(xfer[0].keys())[:8], max_rows=25))

    (out/"supervisor_report.md").write_text("\n".join(md), encoding="utf-8")

    print("[OK] wrote:")
    print("  ", out/"TRAIN_runs.tsv")
    print("  ", out/"TRAIN_group_mean_std.tsv")
    if refine:
        print("  ", out/"REFINE_results.tsv")
    if xfer:
        print("  ", out/"XFER_summary.tsv")
    print("  ", out/"supervisor_report.md")
    print("\nTip:")
    print(f"  column -ts $'\\t' '{out}/TRAIN_group_mean_std.tsv' | less -S")
    print(f"  less -S '{out}/supervisor_report.md'")

if __name__ == "__main__":
    main()
